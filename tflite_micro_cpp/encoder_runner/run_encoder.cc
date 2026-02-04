/* Copyright 2025
 * Run encoder using TFLite Micro library
 * 
 * Usage:
 *   ./run_encoder <model.tflite> <input_fbank.npy> <output_logits.npy>
 * 
 * This program:
 * 1. Loads fbank features from .npy file
 * 2. Runs encoder inference using TFLite Micro
 * 3. Saves output logits to .npy file
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>

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
        // Look for 'shape' and 'descr'
        char* shape_start = strstr(header.data(), "'shape'");
        char* descr_start = strstr(header.data(), "'descr'");
        
        if (!shape_start || !descr_start) {
            MicroPrintf("Failed to parse numpy header\n");
            return false;
        }
        
        // Parse shape: (1, T, 400) or (T, 400)
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
        memset(header, 0, sizeof(header));  // Initialize to avoid null bytes
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
            header[i] = ' ';  // Fill with spaces
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

int main(int argc, char** argv) {
    if (argc != 4) {
        MicroPrintf("Usage: %s <model.tflite> <input_fbank.npy> <output_logits.npy>\n", argv[0]);
        MicroPrintf("Example: %s ../tflite_models/fsmn_encoder_float32.tflite input.npy output.npy\n", argv[0]);
        return 1;
    }
    
    const char* model_file = argv[1];
    const char* input_file = argv[2];
    const char* output_file = argv[3];
    
    MicroPrintf("========================================\n");
    MicroPrintf("TFLite Micro Encoder Runner\n");
    MicroPrintf("========================================\n");
    MicroPrintf("Model: %s\n", model_file);
    MicroPrintf("Input: %s\n", input_file);
    MicroPrintf("Output: %s\n", output_file);
    MicroPrintf("\n");
    
    // 1. Load input fbank features
    NpyArray input_array;
    if (!input_array.load(input_file)) {
        MicroPrintf("ERROR: Failed to load input file\n");
        return 1;
    }
    
    // Reshape to (B, T, D) format if needed
    if (input_array.shape.size() == 2) {
        // (T, D) -> (1, T, D)
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
    
    MicroPrintf("Loaded model: %zu bytes\n", model_size);
    
    const tflite::Model* model = tflite::GetModel(model_data.data());
    if (!model) {
        MicroPrintf("ERROR: Failed to parse model\n");
        return 1;
    }
    
    MicroPrintf("Model schema version: %d\n", model->version());
    
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
    
    // 5. Create interpreter with preserve_all_tensors=true to enable GetTensor()
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
    
    MicroPrintf("Allocated tensors successfully\n");
    MicroPrintf("Input tensors: %zu\n", interpreter.inputs_size());
    MicroPrintf("Output tensors: %zu\n", interpreter.outputs_size());
    
    // Check if this is a stateful model (has cache inputs)
    bool is_stateful = interpreter.inputs_size() > 1;
    MicroPrintf("Model type: %s\n", is_stateful ? "Stateful (with cache)" : "Non-stateful");
    
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
    int logits_output_idx = 0;  // Default to 0 for non-stateful models
    
    if (is_stateful) {
        // Find input tensor indices by checking tensor names
        const auto* subgraph = model->subgraphs()->Get(0);
        
        // Debug: print all input tensor names
        MicroPrintf("Input tensor details:\n");
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
                MicroPrintf("  Input[%zu]: name='%s'\n", i, name ? name : "(null)");
            }
        }
        
        for (size_t i = 0; i < interpreter.inputs_size(); i++) {
            TfLiteTensor* tensor = interpreter.input(i);
            if (!tensor) continue;
            
            // Try to get name from tensor or from flatbuffer
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
        
        // Find output tensor indices by checking tensor names
        // Outputs are NOT in order: we need to find logits (index 0) and cache outputs (indices 1-4)
        // by parsing the name like "StatefulPartitionedCall:0"
        logits_output_idx = -1;  // Reset to -1, will be set if found
        for (size_t i = 0; i < interpreter.outputs_size(); i++) {
            TfLiteTensor* tensor = interpreter.output(i);
            if (!tensor) continue;
            
            // Try to get name from tensor or from flatbuffer
            const char* name = tensor->name;
            if (!name && subgraph && subgraph->outputs()) {
                int tensor_idx = subgraph->outputs()->Get(i);
                const tflite::Tensor* fb_tensor = subgraph->tensors()->Get(tensor_idx);
                if (fb_tensor && fb_tensor->name()) {
                    name = fb_tensor->name()->c_str();
                }
            }
            
            if (name) {
                // Extract index from name like "StatefulPartitionedCall:0"
                const char* colon = strrchr(name, ':');
                if (colon) {
                    int idx = atoi(colon + 1);
                    if (idx == 0) {
                        // logits output
                        logits_output_idx = i;
                    } else if (idx >= 1 && idx <= 4) {
                        // cache output: idx 1->cache_0, 2->cache_1, 3->cache_2, 4->cache_3
                        cache_output_indices[idx - 1] = i;
                    }
                }
            }
        }
        
        // Debug: print all output tensor names and indices
        MicroPrintf("Output tensor details:\n");
        for (size_t i = 0; i < interpreter.outputs_size(); i++) {
            TfLiteTensor* tensor = interpreter.output(i);
            if (tensor) {
                const char* name = tensor->name;
                if (!name && subgraph && subgraph->outputs()) {
                    int tensor_idx = subgraph->outputs()->Get(i);
                    const tflite::Tensor* fb_tensor = subgraph->tensors()->Get(tensor_idx);
                    if (fb_tensor && fb_tensor->name()) {
                        name = fb_tensor->name()->c_str();
                    }
                }
                const char* colon = name ? strrchr(name, ':') : nullptr;
                int idx = colon ? atoi(colon + 1) : -1;
                MicroPrintf("  Output[%zu]: name='%s', index_in_name=%d\n", i, name ? name : "(null)", idx);
            }
        }
        
        if (logits_output_idx < 0) {
            MicroPrintf("ERROR: Failed to find logits output tensor\n");
            return 1;
        }
        
        MicroPrintf("Stateful model tensor indices:\n");
        MicroPrintf("  Main input: %d\n", input_tensor_idx);
        MicroPrintf("  Right context: %d\n", right_context_idx);
        MicroPrintf("  Logits output: %d\n", logits_output_idx);
        for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
            MicroPrintf("  Cache input %zu: %d\n", i, cache_input_indices[i]);
            MicroPrintf("  Cache output %zu: %d\n", i, cache_output_indices[i]);
        }
        
        // Verify logits output shape
        TfLiteTensor* logits_tensor = interpreter.output(logits_output_idx);
        if (logits_tensor && logits_tensor->dims) {
            MicroPrintf("Logits output tensor shape: [");
            for (int i = 0; i < logits_tensor->dims->size; i++) {
                MicroPrintf("%d", logits_tensor->dims->data[i]);
                if (i < logits_tensor->dims->size - 1) MicroPrintf(", ");
            }
            MicroPrintf("]\n");
        }
        
        // Verify all indices are found
        if (input_tensor_idx < 0) {
            MicroPrintf("ERROR: Failed to find main input tensor index\n");
            return 1;
        }
        
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
    
    // Get input tensor (following official examples pattern)
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
        if (subgraph && subgraph->inputs() && subgraph->inputs()->size() > 0) {
            int input_tensor_idx = subgraph->inputs()->Get(0);
            const tflite::Tensor* flatbuffer_tensor = subgraph->tensors()->Get(input_tensor_idx);
            if (flatbuffer_tensor && flatbuffer_tensor->type() == 0) {
                // Flatbuffer type 0 = FLOAT32, convert to kTfLiteFloat32 = 1
                input_tensor->type = kTfLiteFloat32;
                MicroPrintf("Fixed input tensor type: converted flatbuffer FLOAT32 (0) to kTfLiteFloat32 (1)\n");
            }
        }
    }
    
    // Verify input tensor type and shape
    if (input_tensor->type != kTfLiteFloat32) {
        MicroPrintf("ERROR: Input tensor type is %d, expected %d (float32)\n", 
                   input_tensor->type, kTfLiteFloat32);
        return 1;
    }
    
    // Fix dims if needed: get from eval tensor via GetTensor
    if (!input_tensor->dims) {
        const auto* subgraph = model->subgraphs()->Get(0);
        if (subgraph && subgraph->inputs() && subgraph->inputs()->size() > static_cast<size_t>(main_input_idx)) {
            int input_tensor_idx = subgraph->inputs()->Get(main_input_idx);
            TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(input_tensor_idx, 0);
            if (eval_tensor && eval_tensor->dims) {
                // Set dims from eval tensor
                input_tensor->dims = eval_tensor->dims;
                MicroPrintf("Fixed input tensor dims from eval tensor: [");
                for (int i = 0; i < eval_tensor->dims->size; i++) {
                    MicroPrintf("%d", eval_tensor->dims->data[i]);
                    if (i < eval_tensor->dims->size - 1) MicroPrintf(", ");
                }
                MicroPrintf("]\n");
            } else {
                const tflite::Tensor* flatbuffer_tensor = subgraph->tensors()->Get(input_tensor_idx);
                if (flatbuffer_tensor && flatbuffer_tensor->shape()) {
                    MicroPrintf("WARNING: Input tensor dims is null, expected shape from flatbuffer: [");
                    for (flatbuffers::uoffset_t i = 0; i < flatbuffer_tensor->shape()->size(); i++) {
                        MicroPrintf("%d", flatbuffer_tensor->shape()->Get(i));
                        if (i < flatbuffer_tensor->shape()->size() - 1) MicroPrintf(", ");
                    }
                    MicroPrintf("]\n");
                }
            }
        }
    }
    
    // Verify input tensor shape
    if (input_tensor->dims) {
        if (input_tensor->dims->size != 3) {
            MicroPrintf("ERROR: Input tensor dims size is %d, expected 3\n", input_tensor->dims->size);
            return 1;
        }
        MicroPrintf("Input tensor shape: [%d, %d, %d]\n",
                    input_tensor->dims->data[0],
                    input_tensor->dims->data[1],
                    input_tensor->dims->data[2]);
        
        // Verify shape is [1, 10, 400]
        if (input_tensor->dims->data[0] != 1 || 
            input_tensor->dims->data[1] != 10 || 
            input_tensor->dims->data[2] != 400) {
            MicroPrintf("WARNING: Input tensor shape is [%d, %d, %d], expected [1, 10, 400]\n",
                        input_tensor->dims->data[0],
                        input_tensor->dims->data[1],
                        input_tensor->dims->data[2]);
        }
    } else {
        MicroPrintf("ERROR: Input tensor dims is still null after fix attempt\n");
        return 1;
    }
    
    // 6. Initialize cache for stateful model
    // Cache storage: 4 layers, each [1, PROJ_DIM, LEFT_CTX]
    std::vector<std::vector<float>> caches(NUM_CACHE_LAYERS);
    // Backup cache inputs to detect memory reuse issues
    // TFLite Micro may reuse memory for cache input and output tensors
    std::vector<std::vector<float>> cache_backups(NUM_CACHE_LAYERS);
    if (is_stateful) {
        for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
            caches[i].resize(1 * PROJ_DIM * LEFT_CTX, 0.0f);
            MicroPrintf("Initialized cache layer %zu: shape [1, %zu, %zu]\n", i, PROJ_DIM, LEFT_CTX);
        }
    }
    
    // 6.5. Analyze memory layout after AllocateTensors() - BEFORE setting any data
    // This allows us to understand the memory relationships and set all data at once
    // Note: chunk_size is defined later, so we use a constant here
    constexpr size_t chunk_size = 10;  // Define chunk_size early for memory analysis
    
    struct TensorMemoryInfo {
        void* address;
        size_t size_bytes;
        const char* name;
    };
    
    TensorMemoryInfo input_memory_info = {nullptr, 0, "input"};
    std::vector<TensorMemoryInfo> cache_memory_info(NUM_CACHE_LAYERS);
    TensorMemoryInfo right_context_memory_info = {nullptr, 0, "right_context"};
    
    if (is_stateful && input_tensor_idx >= 0) {
        MicroPrintf("\n=== Memory Layout Analysis (After AllocateTensors()) ===\n");
        
        // Get input tensor initial address
        TfLiteTensor* initial_input_tensor = interpreter.input(input_tensor_idx);
        if (initial_input_tensor) {
            const auto* subgraph = model->subgraphs()->Get(0);
            int fb_input_idx = subgraph->inputs()->Get(input_tensor_idx);
            TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(fb_input_idx, 0);
            if (eval_tensor && eval_tensor->data.raw) {
                input_memory_info.address = eval_tensor->data.raw;
                input_memory_info.size_bytes = chunk_size * D * sizeof(float);
                MicroPrintf("Input tensor: address=%p, size=%zu bytes\n", 
                           input_memory_info.address, input_memory_info.size_bytes);
            }
        }
        
        // Get cache tensor initial addresses
        for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
            if (cache_input_indices[i] >= 0) {
                TfLiteTensor* cache_tensor = interpreter.input(cache_input_indices[i]);
                if (cache_tensor) {
                    const auto* subgraph = model->subgraphs()->Get(0);
                    int tensor_idx = subgraph->inputs()->Get(cache_input_indices[i]);
                    TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0);
                    if (eval_tensor && eval_tensor->data.raw) {
                        cache_memory_info[i].address = eval_tensor->data.raw;
                        cache_memory_info[i].size_bytes = 1 * PROJ_DIM * LEFT_CTX * sizeof(float);
                        MicroPrintf("Cache[%zu] tensor: address=%p, size=%zu bytes\n", 
                                   i, cache_memory_info[i].address, cache_memory_info[i].size_bytes);
                        
                        // Check if cache overlaps with input
                        if (input_memory_info.address) {
                            void* cache_start = cache_memory_info[i].address;
                            void* cache_end = (void*)((char*)cache_start + cache_memory_info[i].size_bytes);
                            void* input_start = input_memory_info.address;
                            void* input_end = (void*)((char*)input_start + input_memory_info.size_bytes);
                            
                            bool overlaps = !(cache_end <= input_start || cache_start >= input_end);
                            if (overlaps) {
                                MicroPrintf("  ⚠️  Cache[%zu] OVERLAPS with input tensor!\n", i);
                            } else {
                                MicroPrintf("  ✓ Cache[%zu] does NOT overlap with input\n", i);
                            }
                        }
                    }
                }
            }
        }
        
        // Get right_context tensor initial address
        if (right_context_idx >= 0) {
            TfLiteTensor* rc_tensor = interpreter.input(right_context_idx);
            if (rc_tensor) {
                const auto* subgraph = model->subgraphs()->Get(0);
                int tensor_idx = subgraph->inputs()->Get(right_context_idx);
                TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0);
                if (eval_tensor && eval_tensor->data.raw) {
                    right_context_memory_info.address = eval_tensor->data.raw;
                    right_context_memory_info.size_bytes = RIGHT_CTX * D * sizeof(float);
                    MicroPrintf("Right context tensor: address=%p, size=%zu bytes\n", 
                               right_context_memory_info.address, right_context_memory_info.size_bytes);
                    
                    // Check if right_context overlaps with input
                    if (input_memory_info.address) {
                        void* rc_start = right_context_memory_info.address;
                        void* rc_end = (void*)((char*)rc_start + right_context_memory_info.size_bytes);
                        void* input_start = input_memory_info.address;
                        void* input_end = (void*)((char*)input_start + input_memory_info.size_bytes);
                        
                        bool overlaps = !(rc_end <= input_start || rc_start >= input_end);
                        if (overlaps) {
                            MicroPrintf("  ⚠️  Right context OVERLAPS with input tensor!\n");
                        } else {
                            MicroPrintf("  ✓ Right context does NOT overlap with input\n");
                        }
                    }
                }
            }
        }
        
        MicroPrintf("=== Memory Layout Analysis Complete ===\n\n");
    }
    
    // 7. Process input in chunks (chunk_size = 10)
    // Note: The model expects fixed input shape [1, 10, 400]
    // chunk_size is already defined above for memory analysis
    
    NpyArray output_array;
    output_array.shape = {1, T, 0};  // Will be set after first inference
    
    std::vector<float> output_logits;
    
    for (size_t start = 0; start < T; start += chunk_size) {
        size_t end = std::min(start + chunk_size, T);
        size_t current_chunk_size = end - start;
        
        // STRATEGY: Analyze-then-set approach
        // 1. First, set cache and right_context (this may trigger Arena allocator reallocation)
        // 2. Then, re-acquire input tensor address and check if it changed
        // 3. Finally, set input data using the current (possibly new) address
        
        // For stateful model: set cache inputs and right_context FIRST
        if (is_stateful) {
            
            // Set cache inputs
            for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
                if (cache_input_indices[i] >= 0) {
                    TfLiteTensor* cache_tensor = interpreter.input(cache_input_indices[i]);
                    if (cache_tensor) {
                        // Verify cache tensor shape
                        if (cache_tensor->dims && cache_tensor->dims->size == 3) {
                            if (cache_tensor->dims->data[0] != 1 || 
                                cache_tensor->dims->data[1] != PROJ_DIM || 
                                cache_tensor->dims->data[2] != LEFT_CTX) {
                                MicroPrintf("WARNING: Cache input %zu shape mismatch: [%d, %d, %d], expected [1, %zu, %zu]\n",
                                          i, cache_tensor->dims->data[0], cache_tensor->dims->data[1], 
                                          cache_tensor->dims->data[2], PROJ_DIM, LEFT_CTX);
                            }
                        }
                        
                        float* cache_data = tflite::GetTensorData<float>(cache_tensor);
                        if (!cache_data) {
                            // Try GetTensor
                            const auto* subgraph = model->subgraphs()->Get(0);
                            int tensor_idx = subgraph->inputs()->Get(cache_input_indices[i]);
                            TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0);
                            if (eval_tensor && eval_tensor->data.raw) {
                                cache_tensor->data.data = eval_tensor->data.data;
                                cache_tensor->data.raw = eval_tensor->data.raw;
                                cache_data = tflite::GetTensorData<float>(cache_tensor);
                            }
                        }
                        if (cache_data) {
                            size_t expected_cache_size = 1 * PROJ_DIM * LEFT_CTX;
                            if (caches[i].size() != expected_cache_size) {
                                MicroPrintf("ERROR: Cache %zu size mismatch: %zu, expected %zu\n",
                                          i, caches[i].size(), expected_cache_size);
                                return 1;
                            }
                            
                            // Check if cache input and output share the same memory
                            if (cache_output_indices[i] >= 0) {
                                TfLiteTensor* cache_out_tensor = interpreter.output(cache_output_indices[i]);
                                if (cache_out_tensor) {
                                    float* cache_out_data = tflite::GetTensorData<float>(cache_out_tensor);
                                    if (!cache_out_data) {
                                        const auto* subgraph = model->subgraphs()->Get(0);
                                        if (subgraph && subgraph->outputs() && 
                                            subgraph->outputs()->size() > static_cast<size_t>(cache_output_indices[i])) {
                                            int tensor_idx = subgraph->outputs()->Get(cache_output_indices[i]);
                                            TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0);
                                            if (eval_tensor && eval_tensor->data.raw) {
                                                cache_out_tensor->data.data = eval_tensor->data.data;
                                                cache_out_tensor->data.raw = eval_tensor->data.raw;
                                                cache_out_data = tflite::GetTensorData<float>(cache_out_tensor);
                                            }
                                        }
                                    }
                                    
                                    if (cache_out_data && cache_data == cache_out_data) {
                                        // Memory reuse detected! Backup cache input before Invoke()
                                        if (start == 0 && i == 0) {
                                            MicroPrintf("WARNING: Cache %zu input and output share the same memory address!\n", i);
                                            MicroPrintf("  Input address: %p, Output address: %p\n", 
                                                       (void*)cache_data, (void*)cache_out_data);
                                            MicroPrintf("  This may cause cache input to be overwritten during Invoke()\n");
                                            MicroPrintf("  Creating backup to prevent data corruption...\n");
                                        }
                                        cache_backups[i].resize(expected_cache_size);
                                        memcpy(cache_backups[i].data(), cache_data, expected_cache_size * sizeof(float));
                                    }
                                }
                            }
                            
                            memcpy(cache_data, caches[i].data(), caches[i].size() * sizeof(float));
                        } else {
                            MicroPrintf("ERROR: Cannot access cache input %zu data\n", i);
                            return 1;
                        }
                    } else {
                        MicroPrintf("ERROR: Failed to get cache input %zu tensor\n", i);
                        return 1;
                    }
                }
            }
            
            // Set right_context (next 2 frames, or zeros if last chunk)
            if (right_context_idx >= 0) {
                TfLiteTensor* rc_tensor = interpreter.input(right_context_idx);
                if (rc_tensor) {
                    float* rc_data = tflite::GetTensorData<float>(rc_tensor);
                    if (!rc_data) {
                        // Try GetTensor
                        const auto* subgraph = model->subgraphs()->Get(0);
                        int tensor_idx = subgraph->inputs()->Get(right_context_idx);
                        TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0);
                        if (eval_tensor && eval_tensor->data.raw) {
                            rc_tensor->data.data = eval_tensor->data.data;
                            rc_tensor->data.raw = eval_tensor->data.raw;
                            rc_data = tflite::GetTensorData<float>(rc_tensor);
                        }
                    }
                    if (rc_data) {
                        if (end < T) {
                            // Not last chunk: use next 2 frames
                            size_t rc_size = std::min(RIGHT_CTX, T - end);
                            memcpy(rc_data, &input_array.data[end * D], rc_size * D * sizeof(float));
                            if (rc_size < RIGHT_CTX) {
                                // Pad with zeros
                                memset(rc_data + rc_size * D, 0, 
                                       (RIGHT_CTX - rc_size) * D * sizeof(float));
                            }
                            // Debug: Print right_context data for first few chunks
                            if (start < 30) {
                                MicroPrintf("Right context chunk %zu: frames [%zu:%zu], first 5 values: %.6f %.6f %.6f %.6f %.6f\n",
                                           start / chunk_size, end, end + RIGHT_CTX,
                                           rc_data[0], rc_data[1], rc_data[2], rc_data[3], rc_data[4]);
                            }
                        } else {
                            // Last chunk: use zeros
                            memset(rc_data, 0, RIGHT_CTX * D * sizeof(float));
                        }
                    }
                }
            }
        }
        
        // STEP 2: Re-acquire input tensor address AFTER setting cache/right_context
        // Check if address changed (Arena allocator may have reallocated)
        void* current_input_address = nullptr;
        float* input_data = nullptr;
        
        if (input_tensor_idx >= 0) {
            // Re-acquire input tensor and data pointer (always fresh, no assumptions)
            input_tensor = interpreter.input(input_tensor_idx);
            if (!input_tensor) {
                MicroPrintf("ERROR: Cannot get input tensor\n");
                return 1;
            }
            
            // Get data pointer from TfLiteEvalTensor (most reliable source)
            const auto* subgraph = model->subgraphs()->Get(0);
            int fb_input_idx = subgraph->inputs()->Get(input_tensor_idx);
            TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(fb_input_idx, 0);
            if (!eval_tensor || !eval_tensor->data.raw || eval_tensor->type != kTfLiteFloat32) {
                MicroPrintf("ERROR: Cannot get input tensor eval tensor\n");
                return 1;
            }
            
            // Use eval tensor data pointer (this is what Invoke() will use)
            input_data = eval_tensor->data.f;
            current_input_address = eval_tensor->data.raw;
            
            // Debug: Compare with initial address (for first chunk only)
            if (start == 0 && is_stateful) {
                MicroPrintf("Initial address (from analysis): %p\n", input_memory_info.address);
                MicroPrintf("Current address (after cache setup): %p\n", current_input_address);
                
                if (current_input_address == input_memory_info.address) {
                    MicroPrintf("⚠️  Address UNCHANGED - input tensor still at same location\n");
                    MicroPrintf("   This means Arena allocator did NOT reallocate input tensor\n");
                    MicroPrintf("   If cache and input shared memory initially, they may still share\n");
                } else {
                    MicroPrintf("✓ Address CHANGED - Arena allocator reallocated input tensor\n");
                    MicroPrintf("   Input tensor moved from %p to %p\n", 
                              input_memory_info.address, current_input_address);
                    MicroPrintf("   This means input and cache no longer share memory\n");
                }
            }
        }
        
        // STEP 3: Set input data using current address (one-time setup)
        if (input_data) {
            if (start == 0) {
                MicroPrintf("Input tensor data address: %p\n", (void*)input_data);
            }
            
            // Copy chunk data to input tensor (NOW, right before Invoke())
            size_t input_size = current_chunk_size * D;
            memcpy(input_data, &input_array.data[start * D], input_size * sizeof(float));
            
            // Pad if needed (for fixed input size)
            if (current_chunk_size < chunk_size) {
                memset(input_data + input_size, 0, 
                       (chunk_size - current_chunk_size) * D * sizeof(float));
            }
            
            // Debug: verify input data for first chunk
            if (start == 0) {
                MicroPrintf("  First frame, first 5 features: %.6f, %.6f, %.6f, %.6f, %.6f\n",
                           input_data[0], input_data[1], input_data[2], input_data[3], input_data[4]);
                float input_sum = 0.0f;
                for (size_t i = 0; i < chunk_size * D; i++) {
                    input_sum += input_data[i];
                }
                MicroPrintf("  Input sum: %.6f\n", input_sum);
            }
        }
        
        // Run inference
        status = interpreter.Invoke();
        if (status != kTfLiteOk) {
            MicroPrintf("ERROR: Invoke() failed with status %d at chunk %zu\n", 
                       status, start / chunk_size);
            return 1;
        }
        
        // Save intermediate tensors after Invoke() (only for first chunk)
        if (start == 0) {
            const auto* subgraph = model->subgraphs()->Get(0);
            size_t num_tensors = subgraph->tensors()->size();
            
            // Save input tensor data (for comparison, saved from original input array)
            std::map<int, std::vector<float>> input_tensor_before;
            if (input_tensor_idx >= 0) {
                int fb_input_idx = subgraph->inputs()->Get(input_tensor_idx);
                size_t num_elements = chunk_size * D;
                input_tensor_before[fb_input_idx].resize(num_elements);
                memcpy(input_tensor_before[fb_input_idx].data(), 
                       &input_array.data[start * D], 
                       current_chunk_size * D * sizeof(float));
                if (current_chunk_size < chunk_size) {
                    memset(input_tensor_before[fb_input_idx].data() + current_chunk_size * D, 0,
                           (chunk_size - current_chunk_size) * D * sizeof(float));
                }
            }
            
            // Save all tensors after Invoke() (use flatbuffer tensor index)
            std::map<int, std::vector<float>> tensors_after;
            for (size_t i = 0; i < num_tensors; i++) {
                int fb_tensor_idx = static_cast<int>(i);
                TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(fb_tensor_idx, 0);
                if (eval_tensor && eval_tensor->data.raw && eval_tensor->type == kTfLiteFloat32) {
                    float* data = eval_tensor->data.f;
                    size_t num_elements = 1;
                    if (eval_tensor->dims) {
                        for (int j = 0; j < eval_tensor->dims->size; j++) {
                            num_elements *= eval_tensor->dims->data[j];
                        }
                    }
                    tensors_after[fb_tensor_idx].resize(num_elements);
                    memcpy(tensors_after[fb_tensor_idx].data(), data, num_elements * sizeof(float));
                }
            }
            
            // Replace input tensor with the value saved before Invoke()
            // (Use the original input data, not the tensor data which may have been reused)
            if (!input_tensor_before.empty()) {
                for (const auto& pair : input_tensor_before) {
                    tensors_after[pair.first] = pair.second;
                }
            }
            
            // Save to file (using simple binary format for now)
            // Format: tensor_count, then for each tensor: index, num_elements, data
            FILE* fp = fopen("tflite_micro_intermediate.bin", "wb");
            if (fp) {
                size_t tensor_count = tensors_after.size();
                fwrite(&tensor_count, sizeof(size_t), 1, fp);
                
                for (const auto& pair : tensors_after) {
                    int tensor_idx = pair.first;
                    size_t num_elements = pair.second.size();
                    fwrite(&tensor_idx, sizeof(int), 1, fp);
                    fwrite(&num_elements, sizeof(size_t), 1, fp);
                    fwrite(pair.second.data(), sizeof(float), num_elements, fp);
                }
                fclose(fp);
                MicroPrintf("Saved %zu intermediate tensors to tflite_micro_intermediate.bin\n", tensor_count);
            }
        }
        
        // Get output tensor (logits)
        int logits_idx = is_stateful ? logits_output_idx : 0;
        TfLiteTensor* output_tensor = interpreter.output(logits_idx);
        if (!output_tensor) {
            MicroPrintf("ERROR: Failed to get output tensor\n");
            return 1;
        }
        
        // Fix output tensor dims if needed
        // For stateful model, we need to get dims from the actual output tensor
        // The output tensor from interpreter.output() should have correct dims after Invoke()
        if (!output_tensor->dims) {
            // Try to get dims from eval tensor
            const auto* subgraph = model->subgraphs()->Get(0);
            if (subgraph && subgraph->outputs() && subgraph->outputs()->size() > static_cast<size_t>(logits_idx)) {
                int output_tensor_idx = subgraph->outputs()->Get(logits_idx);
                TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(output_tensor_idx, 0);
                if (eval_tensor && eval_tensor->dims) {
                    output_tensor->dims = eval_tensor->dims;
                }
            }
        }
        
        // Debug: print output tensor info for first chunk
        if (start == 0 && output_tensor->dims) {
            MicroPrintf("Output tensor (logits) info after Invoke():\n");
            MicroPrintf("  dims size: %d\n", output_tensor->dims->size);
            MicroPrintf("  shape: [");
            for (int i = 0; i < output_tensor->dims->size; i++) {
                MicroPrintf("%d", output_tensor->dims->data[i]);
                if (i < output_tensor->dims->size - 1) MicroPrintf(", ");
            }
            MicroPrintf("]\n");
            MicroPrintf("  bytes: %zu (expected: %zu for [1,10,2599])\n", 
                       output_tensor->bytes, 1 * 10 * 2599 * sizeof(float));
            
            // Verify shape is correct
            if (output_tensor->dims->size == 3 && 
                output_tensor->dims->data[0] == 1 && 
                output_tensor->dims->data[1] == 10 && 
                output_tensor->dims->data[2] == 2599) {
                MicroPrintf("  ✅ Output tensor shape is correct!\n");
            } else {
                MicroPrintf("  ⚠️  Output tensor shape is incorrect!\n");
            }
        }
        
        // Get output data pointer using GetTensorData helper
        float* output_data = tflite::GetTensorData<float>(output_tensor);
        if (!output_data) {
            // Try to get data from TfLiteEvalTensor via GetTensor
            const auto* subgraph = model->subgraphs()->Get(0);
            if (subgraph && subgraph->outputs() && subgraph->outputs()->size() > static_cast<size_t>(logits_idx)) {
                int output_tensor_idx = subgraph->outputs()->Get(logits_idx);
                TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(output_tensor_idx, 0);
                if (eval_tensor && eval_tensor->data.raw) {
                    // Manually set the data pointer from eval tensor
                    output_tensor->data.data = eval_tensor->data.data;
                    output_tensor->data.raw = eval_tensor->data.raw;
                    if (!output_tensor->dims && eval_tensor->dims) {
                        output_tensor->dims = eval_tensor->dims;
                    }
                    output_data = tflite::GetTensorData<float>(output_tensor);
                }
            }
            
            if (!output_data) {
                MicroPrintf("ERROR: Cannot access output tensor data\n");
                MicroPrintf("  data.raw: %p\n", output_tensor->data.raw);
                MicroPrintf("  logits_idx: %d\n", logits_idx);
                return 1;
            }
        }
        
        // For stateful model: get and update cache outputs
        if (is_stateful) {
            for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
                if (cache_output_indices[i] >= 0) {
                    TfLiteTensor* cache_out_tensor = interpreter.output(cache_output_indices[i]);
                    if (cache_out_tensor) {
                        // Verify cache output shape
                        if (cache_out_tensor->dims && cache_out_tensor->dims->size == 3) {
                            if (cache_out_tensor->dims->data[0] != 1 || 
                                cache_out_tensor->dims->data[1] != PROJ_DIM || 
                                cache_out_tensor->dims->data[2] != LEFT_CTX) {
                                MicroPrintf("WARNING: Cache output %zu shape mismatch: [%d, %d, %d], expected [1, %zu, %zu]\n",
                                          i, cache_out_tensor->dims->data[0], cache_out_tensor->dims->data[1], 
                                          cache_out_tensor->dims->data[2], PROJ_DIM, LEFT_CTX);
                            }
                        }
                        
                        float* cache_out_data = tflite::GetTensorData<float>(cache_out_tensor);
                        if (!cache_out_data) {
                            // Try GetTensor
                            const auto* subgraph = model->subgraphs()->Get(0);
                            if (subgraph && subgraph->outputs() && 
                                subgraph->outputs()->size() > static_cast<size_t>(cache_output_indices[i])) {
                                int tensor_idx = subgraph->outputs()->Get(cache_output_indices[i]);
                                TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0);
                                if (eval_tensor && eval_tensor->data.raw) {
                                    cache_out_tensor->data.data = eval_tensor->data.data;
                                    cache_out_tensor->data.raw = eval_tensor->data.raw;
                                    cache_out_data = tflite::GetTensorData<float>(cache_out_tensor);
                                }
                            }
                        }
                        if (cache_out_data) {
                            // Update cache for next iteration
                            size_t cache_size = 1 * PROJ_DIM * LEFT_CTX;
                            if (caches[i].size() != cache_size) {
                                MicroPrintf("ERROR: Cache %zu size mismatch: %zu, expected %zu\n",
                                          i, caches[i].size(), cache_size);
                                return 1;
                            }
                            
                            // Check if cache input was overwritten during Invoke()
                            if (!cache_backups[i].empty()) {
                                // Get cache input tensor to check if it was modified
                                TfLiteTensor* cache_in_tensor = interpreter.input(cache_input_indices[i]);
                                if (cache_in_tensor) {
                                    float* cache_in_data = tflite::GetTensorData<float>(cache_in_tensor);
                                    if (!cache_in_data) {
                                        const auto* subgraph = model->subgraphs()->Get(0);
                                        int tensor_idx = subgraph->inputs()->Get(cache_input_indices[i]);
                                        TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0);
                                        if (eval_tensor && eval_tensor->data.raw) {
                                            cache_in_tensor->data.data = eval_tensor->data.data;
                                            cache_in_tensor->data.raw = eval_tensor->data.raw;
                                            cache_in_data = tflite::GetTensorData<float>(cache_in_tensor);
                                        }
                                    }
                                    
                                    if (cache_in_data) {
                                        // Compare cache input with backup
                                        bool was_overwritten = false;
                                        for (size_t j = 0; j < cache_size; j++) {
                                            if (std::abs(cache_in_data[j] - cache_backups[i][j]) > 1e-5f) {
                                                was_overwritten = true;
                                                break;
                                            }
                                        }
                                        
                                        if (was_overwritten && start < 30 && i == 0) {
                                            MicroPrintf("WARNING: Cache %zu input was overwritten during Invoke()!\n", i);
                                            MicroPrintf("  This confirms memory reuse issue. Using cache output instead.\n");
                                        }
                                    }
                                }
                            }
                            
                            // Debug: print cache stats for first few chunks
                            if (start < 30 && i == 0) {
                                float cache_sum_before = 0.0f;
                                float cache_max_before = 0.0f;
                                float cache_min_before = 0.0f;
                                for (size_t j = 0; j < caches[i].size(); j++) {
                                    cache_sum_before += caches[i][j];
                                    if (j == 0 || caches[i][j] > cache_max_before) cache_max_before = caches[i][j];
                                    if (j == 0 || caches[i][j] < cache_min_before) cache_min_before = caches[i][j];
                                }
                                float cache_sum_after = 0.0f;
                                float cache_max_after = 0.0f;
                                float cache_min_after = 0.0f;
                                for (size_t j = 0; j < cache_size; j++) {
                                    cache_sum_after += cache_out_data[j];
                                    if (j == 0 || cache_out_data[j] > cache_max_after) cache_max_after = cache_out_data[j];
                                    if (j == 0 || cache_out_data[j] < cache_min_after) cache_min_after = cache_out_data[j];
                                }
                                // Debug: Cache statistics (commented out to reduce output)
                                // MicroPrintf("Chunk %zu: Cache[0] sum before=%.6f (min=%.6f, max=%.6f), after=%.6f (min=%.6f, max=%.6f)\n",
                                //           start / chunk_size + 1, cache_sum_before, cache_min_before, cache_max_before,
                                //           cache_sum_after, cache_min_after, cache_max_after);
                                
                                // Check if cache output is significantly different from cache input
                                if (start > 0) {
                                    TfLiteTensor* cache_in_tensor = interpreter.input(cache_input_indices[i]);
                                    if (cache_in_tensor) {
                                        float* cache_in_data = tflite::GetTensorData<float>(cache_in_tensor);
                                        if (!cache_in_data) {
                                            const auto* subgraph = model->subgraphs()->Get(0);
                                            int tensor_idx = subgraph->inputs()->Get(cache_input_indices[i]);
                                            TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0);
                                            if (eval_tensor && eval_tensor->data.raw) {
                                                cache_in_tensor->data.data = eval_tensor->data.data;
                                                cache_in_tensor->data.raw = eval_tensor->data.raw;
                                                cache_in_data = tflite::GetTensorData<float>(cache_in_tensor);
                                            }
                                        }
                                        if (cache_in_data) {
                                            float max_diff = 0.0f;
                                            for (size_t j = 0; j < cache_size; j++) {
                                                float diff = std::abs(cache_out_data[j] - cache_in_data[j]);
                                                if (diff > max_diff) max_diff = diff;
                                            }
                                            if (max_diff > 1.0f) {
                                                MicroPrintf("  ⚠️  Cache[0] output differs from input: max_diff=%.6f\n", max_diff);
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // Always use cache output (not cache input) for next iteration
                            // This ensures we use the correct cache state even if memory was reused
                            memcpy(caches[i].data(), cache_out_data, cache_size * sizeof(float));
                        } else {
                            MicroPrintf("ERROR: Cannot access cache output %zu data\n", i);
                            return 1;
                        }
                    } else {
                        MicroPrintf("ERROR: Failed to get cache output %zu tensor\n", i);
                        return 1;
                    }
                }
            }
        }
        
        // Determine output shape (should be [1, chunk_size, output_dim])
        if (output_array.shape[2] == 0) {
            // First chunk - determine output dimension and verify shape
            if (!output_tensor->dims || output_tensor->dims->size < 2) {
                MicroPrintf("ERROR: Invalid output tensor shape\n");
                return 1;
            }
            MicroPrintf("Output tensor shape: [");
            for (int i = 0; i < output_tensor->dims->size; i++) {
                MicroPrintf("%d", output_tensor->dims->data[i]);
                if (i < output_tensor->dims->size - 1) MicroPrintf(", ");
            }
            MicroPrintf("]\n");
            
            // Output should be [1, chunk_size, output_dim]
            if (output_tensor->dims->size == 3) {
                size_t output_T = output_tensor->dims->data[1];  // Time dimension
                size_t output_D = output_tensor->dims->data[2];  // Feature dimension
                if (output_T != chunk_size) {
                    MicroPrintf("WARNING: Output time dimension is %zu, expected %zu\n", output_T, chunk_size);
                }
                output_array.shape[2] = output_D;
                MicroPrintf("Output dimension: %zu (time dim: %zu)\n", output_D, output_T);
            } else {
                output_array.shape[2] = output_tensor->dims->data[output_tensor->dims->size - 1];
                MicroPrintf("Output dimension: %zu\n", output_array.shape[2]);
            }
        }
        
        size_t output_dim = output_array.shape[2];
        // Output tensor shape should be [1, chunk_size, output_dim]
        // But we only save current_chunk_size frames (last chunk may be shorter)
        size_t output_time_dim = (output_tensor->dims && output_tensor->dims->size >= 2) 
                                  ? output_tensor->dims->data[1] 
                                  : chunk_size;
        
        // Read output: only save current_chunk_size frames (not all chunk_size frames)
        // This matches a2.py behavior where each chunk output is appended
        // In a2.py: chunk input [1, current_chunk_size, 400] -> output [1, current_chunk_size, 2599]
        // We input [1, 10, 400] (padded), output [1, 10, 2599], but only save current_chunk_size frames
        // TFLite tensor layout: [batch][time][feature] = row-major (C-style)
        // For shape [1, 10, 2599]: offset = batch*10*2599 + time*2599 + feature
        // Since batch=0 (first dimension is 1), offset = time*2599 + feature
        size_t saved_frames = 0;
        for (size_t i = 0; i < current_chunk_size; i++) {
            for (size_t j = 0; j < output_dim; j++) {
                // Output is stored as [batch][time][feature] in row-major order
                // For [1, 10, 2599]: offset = 0*10*2599 + i*2599 + j = i*2599 + j
                size_t idx = i * output_dim + j;
                output_logits.push_back(output_data[idx]);
            }
            saved_frames++;
        }
        
        // Debug: verify output data for first chunk
        if (start == 0) {
            MicroPrintf("  First frame, first 5 features: %.6f, %.6f, %.6f, %.6f, %.6f\n",
                       output_data[0], output_data[1], output_data[2], output_data[3], output_data[4]);
            float output_sum = 0.0f;
            for (size_t i = 0; i < current_chunk_size * output_dim; i++) {
                output_sum += output_data[i];
            }
        }
        
        if (start == 0 || (start / chunk_size + 1) % 10 == 0 || start + chunk_size >= T) {
            // Progress indicator (only every 10 chunks or last chunk)
            if ((start / chunk_size + 1) % 10 == 0 || start + chunk_size >= T) {
                MicroPrintf("Progress: %zu/%zu frames processed\n", std::min(start + chunk_size, T), T);
            }
            // MicroPrintf("Chunk %zu: input [1, %zu, 400] -> output [1, %zu, 2599], saved %zu frames\n",
            //           start / chunk_size + 1, current_chunk_size, output_time_dim, saved_frames);
        }
    }
    
    // 7. Save output
    output_array.data = output_logits;
    output_array.shape = {1, T, output_array.shape[2]};
    
    if (!output_array.save(output_file)) {
        MicroPrintf("ERROR: Failed to save output file\n");
        return 1;
    }
    
    MicroPrintf("\n");
    MicroPrintf("========================================\n");
    MicroPrintf("Encoder processing completed!\n");
    MicroPrintf("========================================\n");
    MicroPrintf("Output shape: (%zu, %zu, %zu)\n", 
                output_array.shape[0], output_array.shape[1], output_array.shape[2]);
    MicroPrintf("Memory used: %zu / %d bytes\n", 
                interpreter.arena_used_bytes(), kTensorArenaSize);
    
    return 0;
}
