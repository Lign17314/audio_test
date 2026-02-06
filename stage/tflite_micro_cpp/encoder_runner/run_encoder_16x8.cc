/* Copyright 2025
 * Run encoder using TFLite Micro library (INT8/16x8 quantized model)
 * 
 * Usage:
 *   ./run_encoder_16x8 <model.tflite> <input_fbank.npy> <output_logits.npy>
 * 
 * This program:
 * 1. Loads fbank features from .npy file (float32)
 * 2. Quantizes input to INT8 based on model quantization parameters
 * 3. Runs encoder inference using TFLite Micro
 * 4. Dequantizes output from INT8 to float32
 * 5. Saves output logits to .npy file
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

// Simple numpy .npy file reader (supports only float32, 1D/2D arrays)
struct NpyArray {
    std::vector<float> data;
    std::vector<size_t> shape;
    bool is_fortran_order;
    
    bool load(const char* filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            MicroPrintf("Failed to open file: %s\n", filename);
            return false;
        }
        
        // Read magic string
        char magic[6];
        file.read(magic, 6);
        if (memcmp(magic, "\x93NUMPY", 6) != 0) {
            MicroPrintf("Invalid numpy file format\n");
            return false;
        }
        
        // Read version
        uint8_t major, minor;
        file.read(reinterpret_cast<char*>(&major), 1);
        file.read(reinterpret_cast<char*>(&minor), 1);
        
        // Read header length
        uint16_t header_len = 0;
        if (major == 1) {
            file.read(reinterpret_cast<char*>(&header_len), 2);
        } else {
            uint32_t header_len_32 = 0;
            file.read(reinterpret_cast<char*>(&header_len_32), 4);
            header_len = header_len_32;
        }
        
        // Read header
        std::vector<char> header(header_len);
        file.read(header.data(), header_len);
        header.push_back('\0');
        
        // Parse header (simplified - assumes float32, C-order)
        char* shape_start = strstr(header.data(), "'shape'");
        char* descr_start = strstr(header.data(), "'descr'");
        
        if (!shape_start || !descr_start) {
            MicroPrintf("Failed to parse numpy header\n");
            return false;
        }
        
        // Parse shape
        shape.clear();
        char* shape_data = strstr(shape_start, "(");
        if (shape_data) {
            shape_data++;
            while (*shape_data != ')') {
                size_t dim = strtoul(shape_data, &shape_data, 10);
                shape.push_back(dim);
                while (*shape_data == ',' || *shape_data == ' ') shape_data++;
            }
        }
        
        // Verify dtype is float32
        if (strstr(descr_start, "f4") == nullptr && strstr(descr_start, "<f4") == nullptr) {
            MicroPrintf("Only float32 numpy arrays are supported\n");
            return false;
        }
        
        // Calculate data size
        size_t total_size = 1;
        for (size_t dim : shape) {
            total_size *= dim;
        }
        
        // Read data
        data.resize(total_size);
        file.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(float));
        
        MicroPrintf("Loaded numpy array: shape=[");
        for (size_t i = 0; i < shape.size(); i++) {
            MicroPrintf("%zu", shape[i]);
            if (i < shape.size() - 1) MicroPrintf(", ");
        }
        MicroPrintf("], size=%zu\n", total_size);
        
        return true;
    }
    
    bool save(const char* filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            MicroPrintf("Failed to create file: %s\n", filename);
            return false;
        }
        
        // Write magic
        file.write("\x93NUMPY", 6);
        
        // Write version (1.0)
        uint8_t major = 1, minor = 0;
        file.write(reinterpret_cast<const char*>(&major), 1);
        file.write(reinterpret_cast<const char*>(&minor), 1);
        
        // Build header
        char header[256];
        memset(header, 0, sizeof(header));
        int header_len = snprintf(header, sizeof(header),
            "{'descr': '<f4', 'fortran_order': False, 'shape': (");
        
        for (size_t i = 0; i < shape.size(); i++) {
            header_len += snprintf(header + header_len, sizeof(header) - header_len,
                "%zu", shape[i]);
            if (i < shape.size() - 1) {
                header_len += snprintf(header + header_len, sizeof(header) - header_len, ", ");
            }
        }
        header_len += snprintf(header + header_len, sizeof(header) - header_len, "), }");
        
        // Pad header to be divisible by 64
        int padded_len = ((header_len + 1 + 63) / 64) * 64;
        for (int i = header_len; i < padded_len - 1; i++) {
            header[i] = ' ';
        }
        header[padded_len - 1] = '\n';
        
        // Write header length
        uint16_t header_len_16 = padded_len;
        file.write(reinterpret_cast<const char*>(&header_len_16), 2);
        
        // Write header
        file.write(header, padded_len);
        
        // Write data
        file.write(reinterpret_cast<const char*>(data.data()),
                   data.size() * sizeof(float));
        
        MicroPrintf("Saved numpy array: shape=[");
        for (size_t i = 0; i < shape.size(); i++) {
            MicroPrintf("%zu", shape[i]);
            if (i < shape.size() - 1) MicroPrintf(", ");
        }
        MicroPrintf("], size=%zu\n", data.size());
        
        return true;
    }
};

// Quantization helper functions
inline int8_t QuantizeFloatToInt8(float value, float scale, int32_t zero_point) {
    int32_t quantized = static_cast<int32_t>(std::round(value / scale + zero_point));
    // Clip to INT8 range [-128, 127]
    quantized = std::max(-128, std::min(127, quantized));
    return static_cast<int8_t>(quantized);
}

inline int16_t QuantizeFloatToInt16(float value, float scale, int32_t zero_point) {
    int32_t quantized = static_cast<int32_t>(std::round(value / scale + zero_point));
    // Clip to INT16 range [-32768, 32767]
    quantized = std::max(-32768, std::min(32767, quantized));
    return static_cast<int16_t>(quantized);
}

inline float DequantizeInt8ToFloat(int8_t value, float scale, int32_t zero_point) {
    return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}

inline float DequantizeInt16ToFloat(int16_t value, float scale, int32_t zero_point) {
    return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}

// Get quantization parameters from tensor
bool GetQuantizationParams(const TfLiteTensor* tensor, float& scale, int32_t& zero_point) {
    // Check if quantization params exist
    if (!tensor || tensor->quantization.type != kTfLiteAffineQuantization) {
        return false;
    }
    
    // Check if params pointer is valid
    if (!tensor->quantization.params) {
        return false;
    }
    
    const TfLiteAffineQuantization* quant_params =
        static_cast<const TfLiteAffineQuantization*>(tensor->quantization.params);
    
    // Additional safety check
    if (!quant_params) {
        return false;
    }
    
    // Check if scale exists and has data
    if (!quant_params->scale || quant_params->scale->size == 0 || !quant_params->scale->data) {
        return false;
    }
    
    scale = quant_params->scale->data[0];
    zero_point = (quant_params->zero_point && quant_params->zero_point->data) 
                 ? quant_params->zero_point->data[0] : 0;
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        MicroPrintf("Usage: %s <model.tflite> <input_fbank.npy> <output_logits.npy>\n", argv[0]);
        MicroPrintf("Example: %s ../tflite_models/fsmn_encoder_stateful_16x8.tflite input.npy output.npy\n", argv[0]);
        return 1;
    }
    
    const char* model_file = argv[1];
    const char* input_file = argv[2];
    const char* output_file = argv[3];
    
    MicroPrintf("========================================\n");
    MicroPrintf("TFLite Micro Encoder Runner (INT8/16x8 Quantized)\n");
    MicroPrintf("========================================\n");
    MicroPrintf("Model: %s\n", model_file);
    MicroPrintf("Input: %s\n", input_file);
    MicroPrintf("Output: %s\n", output_file);
    MicroPrintf("\n");
    
    // 1. Load input fbank features (float32)
    NpyArray input_array;
    if (!input_array.load(input_file)) {
        MicroPrintf("ERROR: Failed to load input file\n");
        return 1;
    }
    
    // Reshape to (B, T, D) format if needed
    if (input_array.shape.size() == 2) {
        input_array.shape.insert(input_array.shape.begin(), 1);
        MicroPrintf("Reshaped input to (1, %zu, %zu)\n", 
                    input_array.shape[1], input_array.shape[2]);
    }
    
    if (input_array.shape.size() != 3 || input_array.shape[0] != 1) {
        MicroPrintf("ERROR: Expected input shape (1, T, D), got [");
        for (size_t i = 0; i < input_array.shape.size(); i++) {
            MicroPrintf("%zu", input_array.shape[i]);
            if (i < input_array.shape.size() - 1) MicroPrintf(", ");
        }
        MicroPrintf("]\n");
        return 1;
    }
    
    size_t B = input_array.shape[0];
    size_t T = input_array.shape[1];
    size_t D = input_array.shape[2];
    
    MicroPrintf("Input shape: (%zu, %zu, %zu)\n", B, T, D);
    
    // 2. Load TFLite model
    std::ifstream model_stream(model_file, std::ios::binary | std::ios::ate);
    if (!model_stream.is_open()) {
        MicroPrintf("ERROR: Failed to open model file: %s\n", model_file);
        return 1;
    }
    
    std::streamsize model_size = model_stream.tellg();
    model_stream.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> model_data(model_size);
    if (!model_stream.read(reinterpret_cast<char*>(model_data.data()), model_size)) {
        MicroPrintf("ERROR: Failed to read model file\n");
        return 1;
    }
    
    const tflite::Model* model = tflite::GetModel(model_data.data());
    if (!model) {
        MicroPrintf("ERROR: Failed to parse model\n");
        return 1;
    }
    
    // 3. Create op resolver
    tflite::MicroMutableOpResolver<50> resolver;
    resolver.AddFullyConnected();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddRelu();
    resolver.AddReshape();
    resolver.AddTranspose();
    resolver.AddStridedSlice();
    resolver.AddDequantize();
    resolver.AddQuantize();
    resolver.AddPad();
    resolver.AddConcatenation();
    resolver.AddSoftmax();
    resolver.AddMean();
    resolver.AddSum();
    resolver.AddMaximum();
    resolver.AddMinimum();
    resolver.AddSub();
    resolver.AddDiv();
    resolver.AddSqrt();
    resolver.AddRsqrt();
    resolver.AddSquare();
    resolver.AddLogistic();
    resolver.AddTanh();
    resolver.AddExp();
    resolver.AddLog();
    resolver.AddAbs();
    resolver.AddNeg();
    resolver.AddSin();
    resolver.AddCos();
    resolver.AddRound();
    resolver.AddFloor();
    resolver.AddCeil();
    resolver.AddSlice();
    resolver.AddSplit();
    resolver.AddUnpack();
    resolver.AddPack();
    resolver.AddGather();
    resolver.AddSelectV2();
    resolver.AddSqueeze();
    resolver.AddExpandDims();
    resolver.AddCast();
    resolver.AddShape();
    resolver.AddFill();
    resolver.AddZerosLike();
    
    // 4. Allocate tensor arena
    constexpr int kTensorArenaSize = 1 * 1024 * 1024;  // 1MB
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];
    
    // 5. Create interpreter
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize,
                                       nullptr,  // resource_variables
                                       nullptr,  // profiler
                                       true);    // preserve_all_tensors
    
    TfLiteStatus status = interpreter.AllocateTensors();
    if (status != kTfLiteOk) {
        MicroPrintf("ERROR: AllocateTensors() failed with status %d\n", status);
        return 1;
    }
    
    // Check if this is a stateful model (has cache inputs)
    bool is_stateful = interpreter.inputs_size() > 1;
    
    // Cache parameters for stateful model
    constexpr size_t PROJ_DIM = 128;
    constexpr size_t LEFT_CTX = 9;
    constexpr size_t RIGHT_CTX = 2;
    constexpr size_t NUM_CACHE_LAYERS = 4;
    
    // For stateful model, find input tensor indices
    int input_tensor_idx = -1;
    int right_context_idx = -1;
    int cache_input_indices[NUM_CACHE_LAYERS] = {-1, -1, -1, -1};
    int cache_output_indices[NUM_CACHE_LAYERS] = {-1, -1, -1, -1};
    int logits_output_idx = 0;
    
    if (is_stateful) {
        const auto* subgraph = model->subgraphs()->Get(0);
        
        // Find input tensor indices by checking tensor names
        // MicroPrintf("Input tensor details:\n");
        for (size_t i = 0; i < interpreter.inputs_size(); i++) {
            TfLiteTensor* tensor = interpreter.input(i);
            if (tensor) {
                const char* name = tensor->name;
                if (!name && subgraph && subgraph->inputs()) {
                    int tensor_idx = subgraph->inputs()->Get(i);
                    const tflite::Tensor* fb_tensor = subgraph->tensors()->Get(tensor_idx);
                    if (fb_tensor && fb_tensor->name()) {
                        name = fb_tensor->name()->c_str();
                    }
                }
                // MicroPrintf("  Input[%zu]: name='%s', type=%d\n", i, name ? name : "(null)", tensor->type);
            }
        }
        
        for (size_t i = 0; i < interpreter.inputs_size(); i++) {
            TfLiteTensor* tensor = interpreter.input(i);
            if (!tensor) continue;
            
            const char* name = tensor->name;
            if (!name && subgraph && subgraph->inputs()) {
                int tensor_idx = subgraph->inputs()->Get(i);
                const tflite::Tensor* fb_tensor = subgraph->tensors()->Get(tensor_idx);
                if (fb_tensor && fb_tensor->name()) {
                    name = fb_tensor->name()->c_str();
                }
            }
            
            if (name) {
                if (strstr(name, "input") && !strstr(name, "cache") && !strstr(name, "right")) {
                    input_tensor_idx = i;
                } else if (strstr(name, "right_context")) {
                    right_context_idx = i;
                } else if (strstr(name, "in_cache_0") || strstr(name, "cache_0")) {
                    cache_input_indices[0] = i;
                } else if (strstr(name, "in_cache_1") || strstr(name, "cache_1")) {
                    cache_input_indices[1] = i;
                } else if (strstr(name, "in_cache_2") || strstr(name, "cache_2")) {
                    cache_input_indices[2] = i;
                } else if (strstr(name, "in_cache_3") || strstr(name, "cache_3")) {
                    cache_input_indices[3] = i;
                }
            }
        }
        
        // Find output tensor indices
        logits_output_idx = -1;
        for (size_t i = 0; i < interpreter.outputs_size(); i++) {
            TfLiteTensor* tensor = interpreter.output(i);
            if (!tensor) continue;
            
            const char* name = tensor->name;
            if (!name && subgraph && subgraph->outputs()) {
                int tensor_idx = subgraph->outputs()->Get(i);
                const tflite::Tensor* fb_tensor = subgraph->tensors()->Get(tensor_idx);
                if (fb_tensor && fb_tensor->name()) {
                    name = fb_tensor->name()->c_str();
                }
            }
            
            if (name) {
                const char* colon = strrchr(name, ':');
                if (colon) {
                    int idx = atoi(colon + 1);
                    if (idx == 0) {
                        logits_output_idx = i;
                    } else if (idx >= 1 && idx <= 4) {
                        cache_output_indices[idx - 1] = i;
                    }
                }
            }
        }
        
        if (logits_output_idx < 0 || input_tensor_idx < 0) {
            MicroPrintf("ERROR: Failed to find required tensor indices\n");
            return 1;
        }
        
        // MicroPrintf("Stateful model tensor indices:\n");
        // MicroPrintf("  Main input: %d\n", input_tensor_idx);
        // MicroPrintf("  Right context: %d\n", right_context_idx);
        // MicroPrintf("  Logits output: %d\n", logits_output_idx);
        // for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
        //     MicroPrintf("  Cache input %zu: %d\n", i, cache_input_indices[i]);
        //     MicroPrintf("  Cache output %zu: %d\n", i, cache_output_indices[i]);
        // }
        
        // Verify all cache indices are found
        bool all_cache_indices_found = true;
        for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
            if (cache_input_indices[i] < 0) {
                MicroPrintf("ERROR: Failed to find cache input %zu index\n", i);
                all_cache_indices_found = false;
            }
            if (cache_output_indices[i] < 0) {
                MicroPrintf("ERROR: Failed to find cache output %zu index\n", i);
                all_cache_indices_found = false;
            }
        }
        if (!all_cache_indices_found) {
            MicroPrintf("ERROR: Not all cache indices were found\n");
            return 1;
        }
    }
    
    // Get input tensor and check quantization parameters
    int main_input_idx = is_stateful ? input_tensor_idx : 0;
    TfLiteTensor* input_tensor = interpreter.input(main_input_idx);
    if (!input_tensor) {
        MicroPrintf("ERROR: Failed to get input tensor\n");
        return 1;
    }
    
    // Fix type if needed: Flatbuffer FLOAT32 = 0, but C++ kTfLiteFloat32 = 1
    // If type is 0 (NoType) but flatbuffer says FLOAT32, convert it
    if (input_tensor->type == 0) {
        const auto* subgraph = model->subgraphs()->Get(0);
        if (subgraph && subgraph->inputs() && subgraph->inputs()->size() > static_cast<size_t>(main_input_idx)) {
            int fb_input_idx = subgraph->inputs()->Get(main_input_idx);
            const tflite::Tensor* flatbuffer_tensor = subgraph->tensors()->Get(fb_input_idx);
            if (flatbuffer_tensor && flatbuffer_tensor->type() == 0) {
                // Flatbuffer type 0 = FLOAT32, convert to kTfLiteFloat32 = 1
                input_tensor->type = kTfLiteFloat32;
                // MicroPrintf("Fixed input tensor type: converted flatbuffer FLOAT32 (0) to kTfLiteFloat32 (1)\n");
            }
        }
    }
    
    // MicroPrintf("Input tensor type: %d\n", input_tensor->type);
    
    // Get eval tensor first (most reliable source)
    const auto* subgraph = model->subgraphs()->Get(0);
    int fb_input_idx = -1;
    TfLiteEvalTensor* eval_tensor = nullptr;
    if (subgraph && subgraph->inputs() && subgraph->inputs()->size() > static_cast<size_t>(main_input_idx)) {
        fb_input_idx = subgraph->inputs()->Get(main_input_idx);
        eval_tensor = interpreter.GetTensor(fb_input_idx, 0);
    }
    
    // Fix type if needed: Flatbuffer FLOAT32 = 0, but C++ kTfLiteFloat32 = 1
    if (input_tensor->type == 0 && eval_tensor) {
        if (eval_tensor->type == kTfLiteFloat32) {
            input_tensor->type = kTfLiteFloat32;
            // MicroPrintf("Fixed input tensor type: converted flatbuffer FLOAT32 (0) to kTfLiteFloat32 (1)\n");
        }
    }
    
    // Fix dims if needed: get from eval tensor via GetTensor
    if (!input_tensor->dims && eval_tensor && eval_tensor->dims) {
        // Set dims from eval tensor
        input_tensor->dims = eval_tensor->dims;
        // MicroPrintf("Fixed input tensor dims from eval tensor: [...]\n");
    } else if (!input_tensor->dims && subgraph && fb_input_idx >= 0) {
        // Try to get shape from flatbuffer
        const tflite::Tensor* flatbuffer_tensor = subgraph->tensors()->Get(fb_input_idx);
        if (flatbuffer_tensor && flatbuffer_tensor->shape()) {
            // MicroPrintf("WARNING: Input tensor dims is null, expected shape from flatbuffer: [...]\n");
        }
    }
    
    // Input tensor shape (commented out to reduce output)
    // MicroPrintf("Input tensor shape: [");
    // ... (shape printing code)
    // MicroPrintf("]\n");
    
    // Get quantization parameters for input
    float input_scale = 1.0f;
    int32_t input_zero_point = 0;
    bool input_is_quantized = GetQuantizationParams(input_tensor, input_scale, input_zero_point);
    
    // Try to get quantization params from flatbuffer if not found
    if (!input_is_quantized && eval_tensor && 
        (eval_tensor->type == kTfLiteInt16 || eval_tensor->type == kTfLiteInt8)) {
        if (fb_input_idx >= 0) {
            const tflite::Tensor* fb_tensor = subgraph->tensors()->Get(fb_input_idx);
            if (fb_tensor && fb_tensor->quantization()) {
                const tflite::QuantizationParameters* quant = fb_tensor->quantization();
                if (quant && quant->scale() && quant->scale()->size() > 0) {
                    input_scale = quant->scale()->Get(0);
                    if (quant->zero_point() && quant->zero_point()->size() > 0) {
                        input_zero_point = quant->zero_point()->Get(0);
                    }
                    input_is_quantized = true;
                }
            }
        }
    }
    
    // Input tensor type and quantization (commented out to reduce output)
    // if (eval_tensor) {
    //     MicroPrintf("Input tensor type: input_tensor=%d, eval_tensor=%d\n", 
    //                input_tensor->type, eval_tensor->type);
    // }
    // if (input_is_quantized) {
    //     MicroPrintf("Input quantization: scale=%.6f, zero_point=%d\n", input_scale, input_zero_point);
    // } else {
    //     MicroPrintf("Input is NOT quantized (float32)\n");
    // }
    
    // Get output tensor and check quantization parameters
    TfLiteTensor* output_tensor = interpreter.output(logits_output_idx);
    if (!output_tensor) {
        MicroPrintf("ERROR: Failed to get output tensor\n");
        return 1;
    }
    
    // MicroPrintf("Output tensor type: %d\n", output_tensor->type);
    
    float output_scale = 1.0f;
    int32_t output_zero_point = 0;
    bool output_is_quantized = GetQuantizationParams(output_tensor, output_scale, output_zero_point);
    
    // Get output eval tensor to check actual type and quantization params
    int fb_output_idx_init = subgraph->outputs()->Get(logits_output_idx);
    TfLiteEvalTensor* output_eval_tensor_init = interpreter.GetTensor(fb_output_idx_init, 0);
    
    // Try to get quantization params from flatbuffer if not found
    if (!output_is_quantized && output_eval_tensor_init && 
        (output_eval_tensor_init->type == kTfLiteInt16 || output_eval_tensor_init->type == kTfLiteInt8)) {
        const tflite::Tensor* fb_output_tensor = subgraph->tensors()->Get(fb_output_idx_init);
        if (fb_output_tensor && fb_output_tensor->quantization()) {
            const tflite::QuantizationParameters* quant = fb_output_tensor->quantization();
            if (quant && quant->scale() && quant->scale()->size() > 0) {
                output_scale = quant->scale()->Get(0);
                if (quant->zero_point() && quant->zero_point()->size() > 0) {
                    output_zero_point = quant->zero_point()->Get(0);
                }
                output_is_quantized = true;
            }
        }
    }
    
    // Output tensor type and quantization (commented out to reduce output)
    // if (output_eval_tensor_init) {
    //     MicroPrintf("Output tensor type: output_tensor=%d, eval_tensor=%d\n", 
    //                output_tensor->type, output_eval_tensor_init->type);
    // }
    // if (output_is_quantized) {
    //     MicroPrintf("Output quantization: scale=%.6f, zero_point=%d\n", output_scale, output_zero_point);
    // } else {
    //     MicroPrintf("Output is NOT quantized (float32)\n");
    // }
    
    // 6. Initialize cache for stateful model
    std::vector<std::vector<float>> caches(NUM_CACHE_LAYERS);
    if (is_stateful) {
        for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
            caches[i].resize(1 * PROJ_DIM * LEFT_CTX, 0.0f);
        }
    }
    
    // 7. Process input in chunks
    constexpr size_t chunk_size = 10;
    
    MicroPrintf("Processing %zu frames in chunks of %zu...\n", T, chunk_size);
    NpyArray output_array;
    output_array.shape = {1, T, 0};  // Will be set after first inference
    std::vector<float> output_logits;
    
    for (size_t start = 0; start < T; start += chunk_size) {
        size_t end = std::min(start + chunk_size, T);
        size_t current_chunk_size = end - start;
        
        // For stateful model: set cache inputs and right_context FIRST
        if (is_stateful) {
            // Set cache inputs
            for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
                if (cache_input_indices[i] >= 0) {
                    if (static_cast<size_t>(cache_input_indices[i]) >= interpreter.inputs_size()) {
                        MicroPrintf("ERROR: Cache input %zu index %d >= inputs_size %zu\n", 
                                   i, cache_input_indices[i], interpreter.inputs_size());
                        return 1;
                    }
                    TfLiteTensor* cache_tensor = interpreter.input(cache_input_indices[i]);
                    if (cache_tensor) {
                        // Get cache data pointer safely - use eval_tensor directly to avoid corrupting interpreter state
                        const auto* subgraph = model->subgraphs()->Get(0);
                        int tensor_idx = subgraph->inputs()->Get(cache_input_indices[i]);
                        TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0);
                        
                        if (!eval_tensor) {
                            MicroPrintf("ERROR: Cannot get cache input %zu eval tensor (null)\n", i);
                            return 1;
                        }
                        if (!eval_tensor->data.raw) {
                            MicroPrintf("ERROR: Cannot get cache input %zu eval tensor (data.raw is null)\n", i);
                            return 1;
                        }
                        
                        // Check if tensor is quantized (by type or quantization params)
                        float cache_scale = 1.0f;
                        int32_t cache_zero_point = 0;
                        bool cache_is_quantized = GetQuantizationParams(cache_tensor, cache_scale, cache_zero_point);
                        
                        // If no quantization params but tensor is INT16/INT8, try to get from flatbuffer
                        if (!cache_is_quantized && (eval_tensor->type == kTfLiteInt16 || eval_tensor->type == kTfLiteInt8)) {
                            if (subgraph && subgraph->tensors() && tensor_idx >= 0 && 
                                tensor_idx < static_cast<int>(subgraph->tensors()->size())) {
                                const tflite::Tensor* fb_tensor = subgraph->tensors()->Get(tensor_idx);
                                if (fb_tensor && fb_tensor->quantization()) {
                                    const tflite::QuantizationParameters* quant = fb_tensor->quantization();
                                    if (quant && quant->scale() && quant->scale()->size() > 0) {
                                        cache_scale = quant->scale()->Get(0);
                                        if (quant->zero_point() && quant->zero_point()->size() > 0) {
                                            cache_zero_point = quant->zero_point()->Get(0);
                                        }
                                        cache_is_quantized = true;
                                    }
                                }
                            }
                        }
                        
                        // Use eval_tensor type as the source of truth
                        if (eval_tensor->type == kTfLiteInt8) {
                            int8_t* cache_data = eval_tensor->data.int8;
                            if (!cache_data) {
                                MicroPrintf("ERROR: Cache[%zu] eval_tensor type is INT8 but data.int8 is null\n", i);
                                return 1;
                            }
                            // Quantize cache data
                            for (size_t j = 0; j < caches[i].size(); j++) {
                                cache_data[j] = QuantizeFloatToInt8(caches[i][j], cache_scale, cache_zero_point);
                            }
                            if (start == 0) {
                                // MicroPrintf("  Cache[%zu] data set (INT8), first value: %d\n", i, cache_data[0]);
                            }
                        } else if (eval_tensor->type == kTfLiteInt16) {
                            int16_t* cache_data = eval_tensor->data.i16;
                            if (!cache_data) {
                                MicroPrintf("ERROR: Cache[%zu] eval_tensor type is INT16 but data.i16 is null\n", i);
                                return 1;
                            }
                            // Quantize cache data to INT16
                            for (size_t j = 0; j < caches[i].size(); j++) {
                                cache_data[j] = QuantizeFloatToInt16(caches[i][j], cache_scale, cache_zero_point);
                            }
                            // MicroPrintf("  Cache[%zu] data set (INT16), first value: %d, last value: %d\n", 
                            //            i, cache_data[0], cache_data[caches[i].size()-1]);
                        } else if (eval_tensor->type == kTfLiteFloat32) {
                            float* cache_data = eval_tensor->data.f;
                            if (!cache_data) {
                                MicroPrintf("ERROR: Cache[%zu] eval_tensor type is FLOAT32 but data.f is null\n", i);
                                return 1;
                            }
                            // Copy float cache data
                            memcpy(cache_data, caches[i].data(), caches[i].size() * sizeof(float));
                            if (start == 0) {
                                // MicroPrintf("  Cache[%zu] data set (FLOAT32), first value: %.6f\n", i, cache_data[0]);
                            }
                        } else {
                            MicroPrintf("ERROR: Cache[%zu] eval_tensor has unsupported type: %d\n", i, eval_tensor->type);
                            return 1;
                        }
                    } else {
                        MicroPrintf("ERROR: Failed to get cache input %zu tensor\n", i);
                        return 1;
                    }
                }
            }
            
            // Set right_context
            if (right_context_idx >= 0) {
                TfLiteTensor* rc_tensor = interpreter.input(right_context_idx);
                if (rc_tensor) {
                    float rc_scale = 1.0f;
                    int32_t rc_zero_point = 0;
                    bool rc_is_quantized = GetQuantizationParams(rc_tensor, rc_scale, rc_zero_point);
                    
                    // Get right_context data pointer safely - use eval_tensor directly
                    const auto* subgraph = model->subgraphs()->Get(0);
                    int rc_tensor_idx = subgraph->inputs()->Get(right_context_idx);
                    TfLiteEvalTensor* rc_eval_tensor = interpreter.GetTensor(rc_tensor_idx, 0);
                    
                    if (!rc_eval_tensor || !rc_eval_tensor->data.raw) {
                        MicroPrintf("ERROR: Cannot get right_context eval tensor\n");
                        return 1;
                    }
                    
                    // Try to get quantization params from flatbuffer if not found
                    if (!rc_is_quantized && (rc_eval_tensor->type == kTfLiteInt16 || rc_eval_tensor->type == kTfLiteInt8)) {
                        const tflite::Tensor* fb_tensor = subgraph->tensors()->Get(rc_tensor_idx);
                        if (fb_tensor && fb_tensor->quantization()) {
                            const tflite::QuantizationParameters* quant = fb_tensor->quantization();
                            if (quant && quant->scale() && quant->scale()->size() > 0) {
                                rc_scale = quant->scale()->Get(0);
                                if (quant->zero_point() && quant->zero_point()->size() > 0) {
                                    rc_zero_point = quant->zero_point()->Get(0);
                                }
                                rc_is_quantized = true;
                            }
                        }
                    }
                    
                    if (rc_eval_tensor->type == kTfLiteInt8) {
                        int8_t* rc_data = rc_eval_tensor->data.int8;
                        if (!rc_data) {
                            MicroPrintf("ERROR: Cannot access right_context data (INT8)\n");
                            return 1;
                        }
                        if (end < T) {
                            size_t rc_size = std::min(RIGHT_CTX, T - end);
                            for (size_t j = 0; j < rc_size * D; j++) {
                                rc_data[j] = QuantizeFloatToInt8(
                                    input_array.data[(end * D) + j], rc_scale, rc_zero_point);
                            }
                            if (rc_size < RIGHT_CTX) {
                                for (size_t j = rc_size * D; j < RIGHT_CTX * D; j++) {
                                    rc_data[j] = QuantizeFloatToInt8(0.0f, rc_scale, rc_zero_point);
                                }
                            }
                        } else {
                            for (size_t j = 0; j < RIGHT_CTX * D; j++) {
                                rc_data[j] = QuantizeFloatToInt8(0.0f, rc_scale, rc_zero_point);
                            }
                        }
                    } else if (rc_eval_tensor->type == kTfLiteInt16) {
                        int16_t* rc_data = rc_eval_tensor->data.i16;
                        if (!rc_data) {
                            MicroPrintf("ERROR: Cannot access right_context data (INT16)\n");
                            return 1;
                        }
                        if (end < T) {
                            size_t rc_size = std::min(RIGHT_CTX, T - end);
                            for (size_t j = 0; j < rc_size * D; j++) {
                                rc_data[j] = QuantizeFloatToInt16(
                                    input_array.data[(end * D) + j], rc_scale, rc_zero_point);
                            }
                            if (rc_size < RIGHT_CTX) {
                                for (size_t j = rc_size * D; j < RIGHT_CTX * D; j++) {
                                    rc_data[j] = QuantizeFloatToInt16(0.0f, rc_scale, rc_zero_point);
                                }
                            }
                        } else {
                            for (size_t j = 0; j < RIGHT_CTX * D; j++) {
                                rc_data[j] = QuantizeFloatToInt16(0.0f, rc_scale, rc_zero_point);
                            }
                        }
                    } else if (rc_eval_tensor->type == kTfLiteFloat32) {
                        float* rc_data = rc_eval_tensor->data.f;
                        if (!rc_data) {
                            MicroPrintf("ERROR: Cannot access right_context data (FLOAT32)\n");
                            return 1;
                        }
                        if (end < T) {
                            size_t rc_size = std::min(RIGHT_CTX, T - end);
                            memcpy(rc_data, &input_array.data[end * D], rc_size * D * sizeof(float));
                            if (rc_size < RIGHT_CTX) {
                                memset(rc_data + rc_size * D, 0, 
                                       (RIGHT_CTX - rc_size) * D * sizeof(float));
                            }
                        } else {
                            memset(rc_data, 0, RIGHT_CTX * D * sizeof(float));
                        }
                    } else {
                        MicroPrintf("ERROR: Right_context eval_tensor has unsupported type: %d\n", rc_eval_tensor->type);
                        return 1;
                    }
                } else {
                    MicroPrintf("ERROR: Failed to get right_context tensor\n");
                    return 1;
                }
            }
        }
        
        // Set input data - use eval_tensor type as source of truth
        // Re-acquire eval_tensor after cache/right_context setup (analyze-then-set strategy)
        const auto* subgraph = model->subgraphs()->Get(0);
        int fb_input_idx_loop = subgraph->inputs()->Get(main_input_idx);
        TfLiteEvalTensor* input_eval_tensor = interpreter.GetTensor(fb_input_idx_loop, 0);
        
        if (!input_eval_tensor || !input_eval_tensor->data.raw) {
            MicroPrintf("ERROR: Cannot get input eval tensor\n");
            return 1;
        }
        
        if (input_eval_tensor->type == kTfLiteInt8) {
            // Quantize input to INT8
            int8_t* input_data = input_eval_tensor->data.int8;
            if (!input_data) {
                MicroPrintf("ERROR: Cannot access input data (INT8)\n");
                return 1;
            }
            // Quantize chunk data
            size_t input_size = current_chunk_size * D;
            for (size_t i = 0; i < input_size; i++) {
                input_data[i] = QuantizeFloatToInt8(
                    input_array.data[start * D + i], input_scale, input_zero_point);
            }
            // Pad if needed
            if (current_chunk_size < chunk_size) {
                for (size_t i = input_size; i < chunk_size * D; i++) {
                    input_data[i] = QuantizeFloatToInt8(0.0f, input_scale, input_zero_point);
                }
            }
        } else if (input_eval_tensor->type == kTfLiteInt16) {
            // Quantize input to INT16
            int16_t* input_data = input_eval_tensor->data.i16;
            if (!input_data) {
                MicroPrintf("ERROR: Cannot access input data (INT16)\n");
                return 1;
            }
            // Quantize chunk data
            size_t input_size = current_chunk_size * D;
            for (size_t i = 0; i < input_size; i++) {
                input_data[i] = QuantizeFloatToInt16(
                    input_array.data[start * D + i], input_scale, input_zero_point);
            }
            // Pad if needed
            if (current_chunk_size < chunk_size) {
                for (size_t i = input_size; i < chunk_size * D; i++) {
                    input_data[i] = QuantizeFloatToInt16(0.0f, input_scale, input_zero_point);
                }
            }
            if (start == 0) {
                // MicroPrintf("  Input data set (INT16), first value: %d\n", input_data[0]);
            }
        } else if (input_eval_tensor->type == kTfLiteFloat32) {
            MicroPrintf("  Copying input as FLOAT32...\n");
            // Float32 input
            float* input_data = input_eval_tensor->data.f;
            if (!input_data) {
                MicroPrintf("ERROR: Cannot access input data (FLOAT32)\n");
                return 1;
            }
            
            if (input_data) {
                size_t input_size = current_chunk_size * D;
                memcpy(input_data, &input_array.data[start * D], input_size * sizeof(float));
                
                if (current_chunk_size < chunk_size) {
                    memset(input_data + input_size, 0, 
                           (chunk_size - current_chunk_size) * D * sizeof(float));
                }
            }
        }
        
        // Run inference
        status = interpreter.Invoke();
        if (status != kTfLiteOk) {
            MicroPrintf("ERROR: Invoke() failed with status %d at chunk %zu\n", 
                       status, start / chunk_size);
            return 1;
        }
        
        // Get output and dequantize if needed - use eval_tensor type as source of truth
        const auto* subgraph_output = model->subgraphs()->Get(0);
        int fb_output_idx = subgraph_output->outputs()->Get(logits_output_idx);
        TfLiteEvalTensor* output_eval_tensor = interpreter.GetTensor(fb_output_idx, 0);
        
        if (!output_eval_tensor || !output_eval_tensor->data.raw) {
            MicroPrintf("ERROR: Cannot get output eval tensor\n");
            return 1;
        }
        
        if (output_eval_tensor->type == kTfLiteInt8) {
            int8_t* output_data = output_eval_tensor->data.int8;
            if (!output_data || !output_eval_tensor->dims) {
                MicroPrintf("ERROR: Cannot access output data (INT8)\n");
                return 1;
            }
            size_t output_size = 1;
            for (int i = 0; i < output_eval_tensor->dims->size; i++) {
                output_size *= output_eval_tensor->dims->data[i];
            }
            // Dequantize output
            for (size_t i = 0; i < output_size; i++) {
                float dequantized = DequantizeInt8ToFloat(output_data[i], output_scale, output_zero_point);
                output_logits.push_back(dequantized);
            }
            if (start == 0 && output_eval_tensor->dims->size > 2) {
                output_array.shape[2] = output_eval_tensor->dims->data[2];
            }
        } else if (output_eval_tensor->type == kTfLiteInt16) {
            int16_t* output_data = output_eval_tensor->data.i16;
            if (!output_data || !output_eval_tensor->dims) {
                MicroPrintf("ERROR: Cannot access output data (INT16)\n");
                return 1;
            }
            size_t output_size = 1;
            for (int i = 0; i < output_eval_tensor->dims->size; i++) {
                output_size *= output_eval_tensor->dims->data[i];
            }
            // Dequantize output
            for (size_t i = 0; i < output_size; i++) {
                float dequantized = DequantizeInt16ToFloat(output_data[i], output_scale, output_zero_point);
                output_logits.push_back(dequantized);
            }
            if (start == 0 && output_eval_tensor->dims->size > 2) {
                output_array.shape[2] = output_eval_tensor->dims->data[2];
            }
        } else if (output_eval_tensor->type == kTfLiteFloat32) {
            float* output_data = output_eval_tensor->data.f;
            if (!output_data || !output_eval_tensor->dims) {
                MicroPrintf("ERROR: Cannot access output data (FLOAT32)\n");
                return 1;
            }
            size_t output_size = 1;
            for (int i = 0; i < output_eval_tensor->dims->size; i++) {
                output_size *= output_eval_tensor->dims->data[i];
            }
            for (size_t i = 0; i < output_size; i++) {
                output_logits.push_back(output_data[i]);
            }
            if (start == 0 && output_eval_tensor->dims->size > 2) {
                output_array.shape[2] = output_eval_tensor->dims->data[2];
            }
        }
        
        // Update cache for stateful model
        if (is_stateful) {
            for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
                if (cache_output_indices[i] >= 0) {
                    TfLiteTensor* cache_out_tensor = interpreter.output(cache_output_indices[i]);
                    if (cache_out_tensor) {
                        // Get cache output eval tensor to check actual type
                        const auto* subgraph_cache = model->subgraphs()->Get(0);
                        int fb_cache_out_idx = subgraph_cache->outputs()->Get(cache_output_indices[i]);
                        TfLiteEvalTensor* cache_out_eval_tensor = interpreter.GetTensor(fb_cache_out_idx, 0);
                        
                        if (!cache_out_eval_tensor || !cache_out_eval_tensor->data.raw) {
                            MicroPrintf("ERROR: Cannot get cache output %zu eval tensor\n", i);
                            return 1;
                        }
                        
                        float cache_out_scale = 1.0f;
                        int32_t cache_out_zero_point = 0;
                        bool cache_out_is_quantized = GetQuantizationParams(cache_out_tensor, cache_out_scale, cache_out_zero_point);
                        
                        // Try to get quantization params from flatbuffer if not found
                        if (!cache_out_is_quantized && cache_out_eval_tensor && 
                            (cache_out_eval_tensor->type == kTfLiteInt16 || cache_out_eval_tensor->type == kTfLiteInt8)) {
                            const tflite::Tensor* fb_cache_tensor = subgraph_cache->tensors()->Get(fb_cache_out_idx);
                            if (fb_cache_tensor && fb_cache_tensor->quantization()) {
                                const tflite::QuantizationParameters* quant = fb_cache_tensor->quantization();
                                if (quant && quant->scale() && quant->scale()->size() > 0) {
                                    cache_out_scale = quant->scale()->Get(0);
                                    if (quant->zero_point() && quant->zero_point()->size() > 0) {
                                        cache_out_zero_point = quant->zero_point()->Get(0);
                                    }
                                    cache_out_is_quantized = true;
                                }
                            }
                        }
                        
                        if (cache_out_eval_tensor->type == kTfLiteInt8) {
                            int8_t* cache_out_data = cache_out_eval_tensor->data.int8;
                            if (!cache_out_data) {
                                MicroPrintf("ERROR: Cannot access cache output %zu data (INT8)\n", i);
                                return 1;
                            }
                            // Dequantize cache output
                            for (size_t j = 0; j < caches[i].size(); j++) {
                                caches[i][j] = DequantizeInt8ToFloat(cache_out_data[j], cache_out_scale, cache_out_zero_point);
                            }
                        } else if (cache_out_eval_tensor->type == kTfLiteInt16) {
                            int16_t* cache_out_data = cache_out_eval_tensor->data.i16;
                            if (!cache_out_data) {
                                MicroPrintf("ERROR: Cannot access cache output %zu data (INT16)\n", i);
                                return 1;
                            }
                            // Dequantize cache output
                            for (size_t j = 0; j < caches[i].size(); j++) {
                                caches[i][j] = DequantizeInt16ToFloat(cache_out_data[j], cache_out_scale, cache_out_zero_point);
                            }
                        } else if (cache_out_eval_tensor->type == kTfLiteFloat32) {
                            float* cache_out_data = cache_out_eval_tensor->data.f;
                            if (!cache_out_data) {
                                MicroPrintf("ERROR: Cannot access cache output %zu data (FLOAT32)\n", i);
                                return 1;
                            }
                            memcpy(caches[i].data(), cache_out_data, caches[i].size() * sizeof(float));
                        } else {
                            MicroPrintf("ERROR: Cache output %zu eval_tensor has unsupported type: %d\n", i, cache_out_eval_tensor->type);
                            return 1;
                        }
                    } else {
                        MicroPrintf("ERROR: Failed to get cache output %zu tensor\n", i);
                        return 1;
                    }
                }
            }
        }
        
        // Progress indicator (only every 10 chunks or last chunk)
        if ((start / chunk_size + 1) % 10 == 0 || start + chunk_size >= T) {
            MicroPrintf("Progress: %zu/%zu frames processed\n", std::min(start + chunk_size, T), T);
        }
    }
    
    // 8. Save output
    output_array.data = std::move(output_logits);
    if (!output_array.save(output_file)) {
        MicroPrintf("ERROR: Failed to save output file\n");
        return 1;
    }
    
    MicroPrintf("\n========================================\n");
    MicroPrintf("Encoder processing completed!\n");
    MicroPrintf("========================================\n");
    MicroPrintf("Output shape: (1, %zu, %zu)\n", T, output_array.shape[2]);
    
    return 0;
}
