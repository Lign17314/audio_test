/* Copyright 2025
 * Streaming FBank Extractor - 30ms Chunk Version with Threading (16x8 Quantized)
 * 
 * 功能：多线程流式FBank特征提取 + Encoder推理（16x8量化模型）
 * 
 * 线程模型：
 * - 主线程（生产者）：读取音频，提取FBank特征，将特征帧放入队列
 * - 消费者线程：从队列中取出特征帧，进行encoder推理（量化/反量化）
 * 
 * 处理流程：
 * 1. 主线程：每次接收30ms音频块（480 samples @ 16kHz）
 * 2. 主线程：将30ms块分解为3个10ms子块处理
 * 3. 主线程：提取FBank特征，应用LFR和CMVN
 * 4. 主线程：将处理好的特征帧放入线程安全队列
 * 5. 消费者线程：从队列中取出特征帧，累积到12帧后进行encoder推理
 * 6. 消费者线程：量化输入（float32 → int8/int16）
 * 7. 消费者线程：运行推理
 * 8. 消费者线程：反量化输出（int8/int16 → float32）
 * 9. 消费者线程：输出logits
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <memory>
#include <iomanip>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

#include "streaming_fbank_extractor.h"
#include "../fbank_extractor/wav_reader.h"

// TFLite Micro headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

// ========================================
// 量化辅助函数（16x8模型专用）
// ========================================

/**
 * 将float32量化为int8
 */
inline int8_t QuantizeFloatToInt8(float value, float scale, int32_t zero_point) {
    int32_t quantized = static_cast<int32_t>(std::round(value / scale + zero_point));
    // Clip to INT8 range [-128, 127]
    quantized = std::max(-128, std::min(127, quantized));
    return static_cast<int8_t>(quantized);
}

/**
 * 将float32量化为int16
 */
inline int16_t QuantizeFloatToInt16(float value, float scale, int32_t zero_point) {
    int32_t quantized = static_cast<int32_t>(std::round(value / scale + zero_point));
    // Clip to INT16 range [-32768, 32767]
    quantized = std::max(-32768, std::min(32767, quantized));
    return static_cast<int16_t>(quantized);
}

/**
 * 将int8反量化为float32
 */
inline float DequantizeInt8ToFloat(int8_t value, float scale, int32_t zero_point) {
    return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}

/**
 * 将int16反量化为float32
 */
inline float DequantizeInt16ToFloat(int16_t value, float scale, int32_t zero_point) {
    return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}

/**
 * 从tensor获取量化参数
 * @return true=成功获取量化参数，false=tensor未量化或获取失败
 */
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

/**
 * 设置输入tensor（支持float32/int8/int16）
 * 自动检测tensor类型并进行相应的量化
 */
bool SetInputTensor(TfLiteTensor* input_tensor,
                   const tflite::Model* model,
                   tflite::MicroInterpreter& interpreter,
                   int input_tensor_idx,
                   const std::vector<float>& input_data,
                   float input_scale,
                   int32_t input_zero_point) {
    if (!input_tensor) {
        return false;
    }
    
    // Get eval tensor to check actual type
    const auto* subgraph = model->subgraphs()->Get(0);
    int tensor_idx = subgraph->inputs()->Get(input_tensor_idx);
    TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0);
    
    if (!eval_tensor || !eval_tensor->data.raw) {
        return false;
    }
    
    // Set data based on tensor type
    if (eval_tensor->type == kTfLiteInt8) {
        int8_t* data = eval_tensor->data.int8;
        if (!data) return false;
        for (size_t i = 0; i < input_data.size(); i++) {
            data[i] = QuantizeFloatToInt8(input_data[i], input_scale, input_zero_point);
        }
    } else if (eval_tensor->type == kTfLiteInt16) {
        int16_t* data = eval_tensor->data.i16;
        if (!data) return false;
        for (size_t i = 0; i < input_data.size(); i++) {
            data[i] = QuantizeFloatToInt16(input_data[i], input_scale, input_zero_point);
        }
    } else if (eval_tensor->type == kTfLiteFloat32) {
        float* data = eval_tensor->data.f;
        if (!data) return false;
        memcpy(data, input_data.data(), input_data.size() * sizeof(float));
    } else {
        printf("[ERROR] Unsupported input tensor type: %d\n", eval_tensor->type);
        return false;
    }
    
    return true;
}

/**
 * 读取输出tensor（支持float32/int8/int16）
 * 自动检测tensor类型并进行相应的反量化
 */
bool ReadOutputTensor(TfLiteTensor* output_tensor,
                     const tflite::Model* model,
                     tflite::MicroInterpreter& interpreter,
                     int output_tensor_idx,
                     std::vector<float>& output_data,
                     float output_scale,
                     int32_t output_zero_point) {
    if (!output_tensor) {
        return false;
    }
    
    // Get eval tensor to check actual type
    const auto* subgraph = model->subgraphs()->Get(0);
    int tensor_idx = subgraph->outputs()->Get(output_tensor_idx);
    TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0);
    
    if (!eval_tensor || !eval_tensor->data.raw) {
        return false;
    }
    
    // Read data based on tensor type
    if (eval_tensor->type == kTfLiteInt8) {
        int8_t* data = eval_tensor->data.int8;
        if (!data) return false;
        for (size_t i = 0; i < output_data.size(); i++) {
            output_data[i] = DequantizeInt8ToFloat(data[i], output_scale, output_zero_point);
        }
    } else if (eval_tensor->type == kTfLiteInt16) {
        int16_t* data = eval_tensor->data.i16;
        if (!data) return false;
        for (size_t i = 0; i < output_data.size(); i++) {
            output_data[i] = DequantizeInt16ToFloat(data[i], output_scale, output_zero_point);
        }
    } else if (eval_tensor->type == kTfLiteFloat32) {
        float* data = eval_tensor->data.f;
        if (!data) return false;
        memcpy(output_data.data(), data, output_data.size() * sizeof(float));
    } else {
        printf("[ERROR] Unsupported output tensor type: %d\n", eval_tensor->type);
        return false;
    }
    
    return true;
}

// ========================================
// 线程安全的特征帧队列
// ========================================

/**
 * 线程安全的特征帧队列
 * 用于在生产者线程和消费者线程之间传递特征帧
 */
class FeatureQueue {
public:
    FeatureQueue() : finished_(false) {}
    
    /**
     * 将特征帧放入队列（生产者调用）
     * @param frame 特征帧（400维向量）
     */
    void push(const std::vector<float>& frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(frame);
        cv_.notify_one();  // 通知消费者有新数据
    }
    
    /**
     * 从队列中取出特征帧（消费者调用）
     * @param frame 输出特征帧
     * @return true=成功取出，false=队列已结束且为空
     */
    bool pop(std::vector<float>& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // 等待直到队列非空或生产者结束
        cv_.wait(lock, [this] { 
            return !queue_.empty() || finished_; 
        });
        
        // 如果队列为空且生产者已结束，返回false
        if (queue_.empty() && finished_) {
            return false;
        }
        
        // 取出一帧
        frame = queue_.front();
        queue_.pop();
        return true;
    }
    
    /**
     * Peek接下来的n帧（不移除）
     * 用于获取right_context
     * @param n 需要peek的帧数
     * @param feature_dim 特征维度（默认400）
     * @return n帧数据，如果不足则用零填充
     */
    std::vector<std::vector<float>> peek(size_t n, size_t feature_dim = 400) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::vector<float>> result;
        
        // 从队列中复制前n帧（不移除）
        std::queue<std::vector<float>> temp_queue = queue_;
        for (size_t i = 0; i < n && !temp_queue.empty(); i++) {
            result.push_back(temp_queue.front());
            temp_queue.pop();
        }
        
        // 如果不足n帧，用零填充
        while (result.size() < n) {
            result.push_back(std::vector<float>(feature_dim, 0.0f));
        }
        
        return result;
    }
    
    /**
     * 标记生产者已完成（不再有新数据）
     */
    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        cv_.notify_all();  // 通知所有等待的消费者
    }
    
    /**
     * 获取当前队列大小
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
private:
    std::queue<std::vector<float>> queue_;  // 特征帧队列
    mutable std::mutex mutex_;               // 互斥锁
    std::condition_variable cv_;             // 条件变量
    bool finished_;                          // 生产者是否已完成
};

// ========================================
// 消费者线程函数
// ========================================

/**
 * 消费者线程：从队列中取出特征帧并进行encoder推理
 * 
 * 流式处理策略：
 * - 第一次：累积到12帧（10帧输入 + 2帧right_context）时进行推理，输出全部10帧logits
 * - 后续：每累积10帧进行推理，输出全部10帧logits
 * - 最后：处理剩余帧（padding到12帧）
 * 
 * @param queue 特征帧队列
 * @param output_logits 输出logits集合
 * @param frame_count 处理的帧数统计
 * @param model_file TFLite模型文件路径
 */
void consumer_thread(FeatureQueue* queue, 
                    std::vector<std::vector<float>>* output_logits,
                    std::atomic<size_t>* frame_count,
                    const char* model_file) {
    printf("[Consumer] Thread started\n");
    printf("[Consumer] Model file: %s\n", model_file);
    
    // ========================================
    // 步骤1：加载TFLite模型
    // ========================================
    std::ifstream model_stream(model_file, std::ios::binary | std::ios::ate);
    if (!model_stream.is_open()) {
        printf("[Consumer] ERROR: Failed to open model file: %s\n", model_file);
        return;
    }
    
    std::streamsize model_size = model_stream.tellg();
    model_stream.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> model_data(model_size);
    if (!model_stream.read(reinterpret_cast<char*>(model_data.data()), model_size)) {
        printf("[Consumer] ERROR: Failed to read model file\n");
        return;
    }
    
    printf("[Consumer] Loaded model: %zu bytes\n", model_size);
    
    const tflite::Model* model = tflite::GetModel(model_data.data());
    if (!model) {
        printf("[Consumer] ERROR: Failed to parse model\n");
        return;
    }
    
    // ========================================
    // 步骤2：创建Op Resolver
    // ========================================
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
    
    // ========================================
    // 步骤3：分配Tensor Arena
    // ========================================
    constexpr int kTensorArenaSize = 3 * 1024 * 1024;  // 3MB (increased for 16x8 quantized model)
    static alignas(16) uint8_t tensor_arena[kTensorArenaSize];
    
    printf("[Consumer] Tensor arena size: %d bytes (%.2f MB)\n", 
           kTensorArenaSize, kTensorArenaSize / (1024.0 * 1024.0));
    
    // ========================================
    // 步骤4：创建Interpreter
    // ========================================
    printf("[Consumer] Creating interpreter...\n");
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize,
                                       nullptr,  // resource_variables
                                       nullptr,  // profiler
                                       true);    // preserve_all_tensors
    
    printf("[Consumer] Allocating tensors...\n");
    TfLiteStatus status = interpreter.AllocateTensors();
    if (status != kTfLiteOk) {
        printf("[Consumer] ERROR: AllocateTensors() failed with status %d\n", status);
        return;
    }
    
    printf("[Consumer] Allocated tensors successfully\n");
    printf("[Consumer] Arena used: %zu / %d bytes (%.2f%%)\n", 
           interpreter.arena_used_bytes(), kTensorArenaSize,
           100.0 * interpreter.arena_used_bytes() / kTensorArenaSize);
    printf("[Consumer] Input tensors: %zu\n", interpreter.inputs_size());
    printf("[Consumer] Output tensors: %zu\n", interpreter.outputs_size());
    
    // ========================================
    // 步骤5：查找Tensor索引（Stateful模型）
    // ========================================
    bool is_stateful = interpreter.inputs_size() > 1;
    printf("[Consumer] Model type: %s\n", is_stateful ? "Stateful (with cache)" : "Non-stateful");
    
    if (!is_stateful) {
        printf("[Consumer] ERROR: Expected stateful model\n");
        return;
    }
    
    // 配置参数
    constexpr size_t CHUNK_SIZE = 10;      // 每次处理10帧
    constexpr size_t RIGHT_CTX = 2;        // 右上下文2帧
    constexpr size_t FEATURE_DIM = 400;    // FBank特征维度
    constexpr size_t PROJ_DIM = 128;       // Cache投影维度
    constexpr size_t LEFT_CTX = 9;         // Cache左上下文
    constexpr size_t NUM_CACHE_LAYERS = 4; // Cache层数
    
    // 查找输入tensor索引
    int input_tensor_idx = -1;
    int right_context_idx = -1;
    int cache_input_indices[NUM_CACHE_LAYERS] = {-1, -1, -1, -1};
    int cache_output_indices[NUM_CACHE_LAYERS] = {-1, -1, -1, -1};
    int logits_output_idx = -1;
    
    const auto* subgraph = model->subgraphs()->Get(0);
    
    // 查找输入tensor
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
    
    // 查找输出tensor
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
    
    // 验证所有索引都找到了
    if (input_tensor_idx < 0 || logits_output_idx < 0) {
        printf("[Consumer] ERROR: Failed to find required tensor indices\n");
        return;
    }
    
    for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
        if (cache_input_indices[i] < 0 || cache_output_indices[i] < 0) {
            printf("[Consumer] ERROR: Failed to find cache %zu indices\n", i);
            return;
        }
    }
    
    printf("[Consumer] Tensor indices found:\n");
    printf("  Main input: %d\n", input_tensor_idx);
    printf("  Right context: %d\n", right_context_idx);
    printf("  Logits output: %d\n", logits_output_idx);
    for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
        printf("  Cache input %zu: %d\n", i, cache_input_indices[i]);
        printf("  Cache output %zu: %d\n", i, cache_output_indices[i]);
    }
    
    // ========================================
    // 步骤6：初始化Cache
    // ========================================
    std::vector<std::vector<float>> caches(NUM_CACHE_LAYERS);
    for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
        caches[i].resize(1 * PROJ_DIM * LEFT_CTX, 0.0f);
    }
    printf("[Consumer] Initialized %zu cache layers\n", NUM_CACHE_LAYERS);
    
    // ========================================
    // 步骤6.5：获取量化参数（16x8模型）
    // ========================================
    
    // 主输入tensor的量化参数
    float input_scale = 1.0f;
    int32_t input_zero_point = 0;
    bool input_is_quantized = false;
    
    TfLiteTensor* input_tensor = interpreter.input(input_tensor_idx);
    if (input_tensor) {
        input_is_quantized = GetQuantizationParams(input_tensor, input_scale, input_zero_point);
        
        // 如果从TfLiteTensor获取失败，尝试从flatbuffer获取
        if (!input_is_quantized) {
            int fb_input_idx = subgraph->inputs()->Get(input_tensor_idx);
            TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(fb_input_idx, 0);
            if (eval_tensor && (eval_tensor->type == kTfLiteInt16 || eval_tensor->type == kTfLiteInt8)) {
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
    }
    
    printf("[Consumer] Input quantization: %s\n", input_is_quantized ? "Yes" : "No");
    if (input_is_quantized) {
        printf("  Scale: %.6f, Zero point: %d\n", input_scale, input_zero_point);
    }
    
    // Right context tensor的量化参数
    float rc_scale = 1.0f;
    int32_t rc_zero_point = 0;
    bool rc_is_quantized = false;
    
    if (right_context_idx >= 0) {
        TfLiteTensor* rc_tensor = interpreter.input(right_context_idx);
        if (rc_tensor) {
            rc_is_quantized = GetQuantizationParams(rc_tensor, rc_scale, rc_zero_point);
            
            if (!rc_is_quantized) {
                int fb_rc_idx = subgraph->inputs()->Get(right_context_idx);
                TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(fb_rc_idx, 0);
                if (eval_tensor && (eval_tensor->type == kTfLiteInt16 || eval_tensor->type == kTfLiteInt8)) {
                    const tflite::Tensor* fb_tensor = subgraph->tensors()->Get(fb_rc_idx);
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
            }
        }
    }
    
    // Cache tensors的量化参数
    float cache_in_scales[NUM_CACHE_LAYERS] = {1.0f, 1.0f, 1.0f, 1.0f};
    int32_t cache_in_zero_points[NUM_CACHE_LAYERS] = {0, 0, 0, 0};
    bool cache_in_is_quantized[NUM_CACHE_LAYERS] = {false, false, false, false};
    
    float cache_out_scales[NUM_CACHE_LAYERS] = {1.0f, 1.0f, 1.0f, 1.0f};
    int32_t cache_out_zero_points[NUM_CACHE_LAYERS] = {0, 0, 0, 0};
    bool cache_out_is_quantized[NUM_CACHE_LAYERS] = {false, false, false, false};
    
    for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
        // Cache input
        TfLiteTensor* cache_in_tensor = interpreter.input(cache_input_indices[i]);
        if (cache_in_tensor) {
            cache_in_is_quantized[i] = GetQuantizationParams(cache_in_tensor, cache_in_scales[i], cache_in_zero_points[i]);
            
            if (!cache_in_is_quantized[i]) {
                int fb_cache_idx = subgraph->inputs()->Get(cache_input_indices[i]);
                TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(fb_cache_idx, 0);
                if (eval_tensor && (eval_tensor->type == kTfLiteInt16 || eval_tensor->type == kTfLiteInt8)) {
                    const tflite::Tensor* fb_tensor = subgraph->tensors()->Get(fb_cache_idx);
                    if (fb_tensor && fb_tensor->quantization()) {
                        const tflite::QuantizationParameters* quant = fb_tensor->quantization();
                        if (quant && quant->scale() && quant->scale()->size() > 0) {
                            cache_in_scales[i] = quant->scale()->Get(0);
                            if (quant->zero_point() && quant->zero_point()->size() > 0) {
                                cache_in_zero_points[i] = quant->zero_point()->Get(0);
                            }
                            cache_in_is_quantized[i] = true;
                        }
                    }
                }
            }
        }
        
        // Cache output
        TfLiteTensor* cache_out_tensor = interpreter.output(cache_output_indices[i]);
        if (cache_out_tensor) {
            cache_out_is_quantized[i] = GetQuantizationParams(cache_out_tensor, cache_out_scales[i], cache_out_zero_points[i]);
            
            if (!cache_out_is_quantized[i]) {
                int fb_cache_idx = subgraph->outputs()->Get(cache_output_indices[i]);
                TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(fb_cache_idx, 0);
                if (eval_tensor && (eval_tensor->type == kTfLiteInt16 || eval_tensor->type == kTfLiteInt8)) {
                    const tflite::Tensor* fb_tensor = subgraph->tensors()->Get(fb_cache_idx);
                    if (fb_tensor && fb_tensor->quantization()) {
                        const tflite::QuantizationParameters* quant = fb_tensor->quantization();
                        if (quant && quant->scale() && quant->scale()->size() > 0) {
                            cache_out_scales[i] = quant->scale()->Get(0);
                            if (quant->zero_point() && quant->zero_point()->size() > 0) {
                                cache_out_zero_points[i] = quant->zero_point()->Get(0);
                            }
                            cache_out_is_quantized[i] = true;
                        }
                    }
                }
            }
        }
    }
    
    // 输出tensor的量化参数
    float output_scale = 1.0f;
    int32_t output_zero_point = 0;
    bool output_is_quantized = false;
    
    TfLiteTensor* output_tensor = interpreter.output(logits_output_idx);
    if (output_tensor) {
        output_is_quantized = GetQuantizationParams(output_tensor, output_scale, output_zero_point);
        
        if (!output_is_quantized) {
            int fb_output_idx = subgraph->outputs()->Get(logits_output_idx);
            TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(fb_output_idx, 0);
            if (eval_tensor && (eval_tensor->type == kTfLiteInt16 || eval_tensor->type == kTfLiteInt8)) {
                const tflite::Tensor* fb_tensor = subgraph->tensors()->Get(fb_output_idx);
                if (fb_tensor && fb_tensor->quantization()) {
                    const tflite::QuantizationParameters* quant = fb_tensor->quantization();
                    if (quant && quant->scale() && quant->scale()->size() > 0) {
                        output_scale = quant->scale()->Get(0);
                        if (quant->zero_point() && quant->zero_point()->size() > 0) {
                            output_zero_point = quant->zero_point()->Get(0);
                        }
                        output_is_quantized = true;
                    }
                }
            }
        }
    }
    
    printf("[Consumer] Quantization parameters obtained\n");
    
    // ========================================
    // 宏定义：简化tensor操作（支持量化）
    // ========================================
    
    // 宏：设置cache输入（支持量化）
    #define SET_CACHE_INPUT(cache_idx) \
        do { \
            printf("[Consumer] SET_CACHE_INPUT(%zu) start\n", cache_idx); \
            fflush(stdout); \
            if (cache_input_indices[cache_idx] < 0 || \
                static_cast<size_t>(cache_input_indices[cache_idx]) >= interpreter.inputs_size()) { \
                printf("[Consumer] ERROR: Cache[%zu] invalid index %d (inputs_size=%zu)\n", \
                       cache_idx, cache_input_indices[cache_idx], interpreter.inputs_size()); \
                return; \
            } \
            printf("[Consumer] SET_CACHE_INPUT(%zu) getting tensor_idx\n", cache_idx); \
            fflush(stdout); \
            int tensor_idx = subgraph->inputs()->Get(cache_input_indices[cache_idx]); \
            printf("[Consumer] SET_CACHE_INPUT(%zu) tensor_idx=%d, getting eval_tensor\n", cache_idx, tensor_idx); \
            fflush(stdout); \
            TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0); \
            printf("[Consumer] SET_CACHE_INPUT(%zu) got eval_tensor=%p\n", cache_idx, (void*)eval_tensor); \
            fflush(stdout); \
            if (!eval_tensor || !eval_tensor->data.raw) { \
                printf("[Consumer] ERROR: Cache[%zu] eval_tensor is NULL or data.raw is NULL\n", cache_idx); \
                return; \
            } \
            printf("[Consumer] SET_CACHE_INPUT(%zu) type=%d\n", cache_idx, eval_tensor->type); \
            fflush(stdout); \
            if (eval_tensor->type == kTfLiteInt8) { \
                int8_t* data = eval_tensor->data.int8; \
                if (!data) { \
                    printf("[Consumer] ERROR: Cache[%zu] INT8 data pointer is NULL\n", cache_idx); \
                    return; \
                } \
                for (size_t j = 0; j < caches[cache_idx].size(); j++) { \
                    data[j] = QuantizeFloatToInt8(caches[cache_idx][j], cache_in_scales[cache_idx], cache_in_zero_points[cache_idx]); \
                } \
            } else if (eval_tensor->type == kTfLiteInt16) { \
                int16_t* data = eval_tensor->data.i16; \
                if (!data) { \
                    printf("[Consumer] ERROR: Cache[%zu] INT16 data pointer is NULL\n", cache_idx); \
                    return; \
                } \
                for (size_t j = 0; j < caches[cache_idx].size(); j++) { \
                    data[j] = QuantizeFloatToInt16(caches[cache_idx][j], cache_in_scales[cache_idx], cache_in_zero_points[cache_idx]); \
                } \
            } else if (eval_tensor->type == kTfLiteFloat32) { \
                float* data = eval_tensor->data.f; \
                if (!data) { \
                    printf("[Consumer] ERROR: Cache[%zu] FLOAT32 data pointer is NULL\n", cache_idx); \
                    return; \
                } \
                memcpy(data, caches[cache_idx].data(), caches[cache_idx].size() * sizeof(float)); \
            } \
            printf("[Consumer] SET_CACHE_INPUT(%zu) done\n", cache_idx); \
            fflush(stdout); \
        } while(0)
    
    // 宏：读取cache输出（支持反量化）
    #define READ_CACHE_OUTPUT(cache_idx) \
        do { \
            int tensor_idx = subgraph->outputs()->Get(cache_output_indices[cache_idx]); \
            TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0); \
            if (eval_tensor && eval_tensor->data.raw) { \
                if (eval_tensor->type == kTfLiteInt8) { \
                    int8_t* data = eval_tensor->data.int8; \
                    for (size_t j = 0; j < caches[cache_idx].size(); j++) { \
                        caches[cache_idx][j] = DequantizeInt8ToFloat(data[j], cache_out_scales[cache_idx], cache_out_zero_points[cache_idx]); \
                    } \
                } else if (eval_tensor->type == kTfLiteInt16) { \
                    int16_t* data = eval_tensor->data.i16; \
                    for (size_t j = 0; j < caches[cache_idx].size(); j++) { \
                        caches[cache_idx][j] = DequantizeInt16ToFloat(data[j], cache_out_scales[cache_idx], cache_out_zero_points[cache_idx]); \
                    } \
                } else if (eval_tensor->type == kTfLiteFloat32) { \
                    float* data = eval_tensor->data.f; \
                    memcpy(caches[cache_idx].data(), data, caches[cache_idx].size() * sizeof(float)); \
                } \
            } \
        } while(0)
    
    // 宏：设置主输入
    #define SET_MAIN_INPUT(input_data_vec) \
        do { \
            printf("[Consumer] SET_MAIN_INPUT start, size=%zu\n", input_data_vec.size()); \
            fflush(stdout); \
            int tensor_idx = subgraph->inputs()->Get(input_tensor_idx); \
            printf("[Consumer] SET_MAIN_INPUT tensor_idx=%d\n", tensor_idx); \
            fflush(stdout); \
            TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0); \
            printf("[Consumer] SET_MAIN_INPUT eval_tensor=%p\n", (void*)eval_tensor); \
            fflush(stdout); \
            if (eval_tensor && eval_tensor->data.raw) { \
                printf("[Consumer] SET_MAIN_INPUT type=%d\n", eval_tensor->type); \
                fflush(stdout); \
                if (eval_tensor->type == kTfLiteInt8) { \
                    int8_t* data = eval_tensor->data.int8; \
                    printf("[Consumer] SET_MAIN_INPUT INT8 data=%p\n", (void*)data); \
                    fflush(stdout); \
                    for (size_t i = 0; i < input_data_vec.size(); i++) { \
                        data[i] = QuantizeFloatToInt8(input_data_vec[i], input_scale, input_zero_point); \
                    } \
                } else if (eval_tensor->type == kTfLiteInt16) { \
                    int16_t* data = eval_tensor->data.i16; \
                    printf("[Consumer] SET_MAIN_INPUT INT16 data=%p\n", (void*)data); \
                    fflush(stdout); \
                    for (size_t i = 0; i < input_data_vec.size(); i++) { \
                        data[i] = QuantizeFloatToInt16(input_data_vec[i], input_scale, input_zero_point); \
                    } \
                } else if (eval_tensor->type == kTfLiteFloat32) { \
                    float* data = eval_tensor->data.f; \
                    printf("[Consumer] SET_MAIN_INPUT FLOAT32 data=%p\n", (void*)data); \
                    fflush(stdout); \
                    memcpy(data, input_data_vec.data(), input_data_vec.size() * sizeof(float)); \
                } \
            } \
            printf("[Consumer] SET_MAIN_INPUT done\n"); \
            fflush(stdout); \
        } while(0)
    
    // 宏：设置right_context
    #define SET_RIGHT_CONTEXT(rc_data_vec) \
        do { \
            if (right_context_idx >= 0) { \
                int tensor_idx = subgraph->inputs()->Get(right_context_idx); \
                TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0); \
                if (eval_tensor && eval_tensor->data.raw) { \
                    if (eval_tensor->type == kTfLiteInt8) { \
                        int8_t* data = eval_tensor->data.int8; \
                        for (size_t i = 0; i < rc_data_vec.size(); i++) { \
                            data[i] = QuantizeFloatToInt8(rc_data_vec[i], rc_scale, rc_zero_point); \
                        } \
                    } else if (eval_tensor->type == kTfLiteInt16) { \
                        int16_t* data = eval_tensor->data.i16; \
                        for (size_t i = 0; i < rc_data_vec.size(); i++) { \
                            data[i] = QuantizeFloatToInt16(rc_data_vec[i], rc_scale, rc_zero_point); \
                        } \
                    } else if (eval_tensor->type == kTfLiteFloat32) { \
                        float* data = eval_tensor->data.f; \
                        memcpy(data, rc_data_vec.data(), rc_data_vec.size() * sizeof(float)); \
                    } \
                } \
            } \
        } while(0)
    
    // 宏：读取输出logits（单帧）
    #define READ_OUTPUT_LOGIT_FRAME(output_vec, frame_idx) \
        do { \
            int tensor_idx = subgraph->outputs()->Get(logits_output_idx); \
            TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0); \
            if (eval_tensor && eval_tensor->data.raw) { \
                if (eval_tensor->type == kTfLiteInt8) { \
                    int8_t* data = eval_tensor->data.int8; \
                    for (size_t j = 0; j < output_dim; j++) { \
                        output_vec[j] = DequantizeInt8ToFloat(data[frame_idx * output_dim + j], output_scale, output_zero_point); \
                    } \
                } else if (eval_tensor->type == kTfLiteInt16) { \
                    int16_t* data = eval_tensor->data.i16; \
                    for (size_t j = 0; j < output_dim; j++) { \
                        output_vec[j] = DequantizeInt16ToFloat(data[frame_idx * output_dim + j], output_scale, output_zero_point); \
                    } \
                } else if (eval_tensor->type == kTfLiteFloat32) { \
                    float* data = eval_tensor->data.f; \
                    memcpy(output_vec.data(), data + frame_idx * output_dim, output_dim * sizeof(float)); \
                } \
            } \
        } while(0)
    
    printf("[Consumer] Macros defined for quantized tensor operations\n");
    
    // ========================================
    // 步骤7：流式处理循环
    // ========================================
    std::vector<std::vector<float>> frame_buffer;  // 累积的帧
    std::vector<std::vector<float>> all_logits;    // 所有输出的logits
    std::vector<float> frame;
    size_t total_frames_received = 0;
    size_t chunk_count = 0;
    size_t output_dim = 0;  // 将在第一次推理后确定
    
    printf("[Consumer] Starting streaming processing...\n");
    printf("  Chunk size: %zu frames\n", CHUNK_SIZE);
    printf("  Right context: %zu frames\n", RIGHT_CTX);
    printf("  Feature dimension: %zu\n\n", FEATURE_DIM);
    
    while (true) {
        // 从队列取一帧
        if (queue->pop(frame)) {
            frame_buffer.push_back(frame);
            total_frames_received++;
            
            // 判断是否可以进行推理
            bool should_infer = false;
            
            if (frame_buffer.size() >= CHUNK_SIZE + RIGHT_CTX) {
                // 累积到12帧（10帧输入 + 2帧right_context）
                should_infer = true;
            }
            
            if (should_infer) {
                chunk_count++;
                printf("[Consumer] Chunk %zu: Processing %zu frames (total received: %zu)\n",
                       chunk_count, frame_buffer.size(), total_frames_received);
                
                // 准备输入数据（10帧）
                std::vector<float> input_data(CHUNK_SIZE * FEATURE_DIM);
                for (size_t i = 0; i < CHUNK_SIZE; i++) {
                    memcpy(input_data.data() + i * FEATURE_DIM,
                           frame_buffer[i].data(),
                           FEATURE_DIM * sizeof(float));
                }
                
                // 准备right_context（2帧）
                std::vector<float> right_context_data(RIGHT_CTX * FEATURE_DIM);
                for (size_t i = 0; i < RIGHT_CTX; i++) {
                    memcpy(right_context_data.data() + i * FEATURE_DIM,
                           frame_buffer[CHUNK_SIZE + i].data(),
                           FEATURE_DIM * sizeof(float));
                }
                
                // ========================================
                // 设置输入tensors
                // ========================================
                
                printf("[Consumer] Setting cache inputs...\n");
                fflush(stdout);
                
                // 1. 设置cache inputs
                for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
                    SET_CACHE_INPUT(i);
                }
                
                printf("[Consumer] Setting right_context...\n");
                fflush(stdout);
                
                // 2. 设置right_context
                SET_RIGHT_CONTEXT(right_context_data);
                
                printf("[Consumer] Setting main input...\n");
                fflush(stdout);
                
                // 3. 设置main input
                SET_MAIN_INPUT(input_data);
                
                printf("[Consumer] Running inference...\n");
                fflush(stdout);
                
                // ========================================
                // 4. 运行推理
                // ========================================
                status = interpreter.Invoke();
                printf("[Consumer] Inference done, status=%d\n", status);
                fflush(stdout);
                if (status != kTfLiteOk) {
                    printf("[Consumer] ERROR: Invoke() failed with status %d\n", status);
                    return;
                }
                
                // ========================================
                // 5. 获取输出logits
                // ========================================
                printf("[Consumer] Getting output tensor...\n");
                fflush(stdout);
                TfLiteTensor* output_tensor = interpreter.output(logits_output_idx);
                printf("[Consumer] output_tensor=%p\n", (void*)output_tensor);
                fflush(stdout);
                if (!output_tensor) {
                    printf("[Consumer] ERROR: Failed to get output tensor\n");
                    return;
                }
                
                printf("[Consumer] Checking output tensor dims...\n");
                fflush(stdout);
                // Fix output tensor dims if needed
                if (!output_tensor->dims) {
                    printf("[Consumer] output_tensor->dims is NULL, trying to fix...\n");
                    fflush(stdout);
                    int tensor_idx = subgraph->outputs()->Get(logits_output_idx);
                    TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0);
                    if (eval_tensor && eval_tensor->dims) {
                        output_tensor->dims = eval_tensor->dims;
                    }
                }
                
                printf("[Consumer] Determining output dimension...\n");
                fflush(stdout);
                // 确定输出维度（第一次推理时）- 使用 eval_tensor 而不是 TfLiteTensor
                if (output_dim == 0) {
                    printf("[Consumer] output_dim is 0, getting eval_tensor for output...\n");
                    fflush(stdout);
                    int output_tensor_idx = subgraph->outputs()->Get(logits_output_idx);
                    TfLiteEvalTensor* output_eval_tensor = interpreter.GetTensor(output_tensor_idx, 0);
                    printf("[Consumer] output_eval_tensor=%p\n", (void*)output_eval_tensor);
                    fflush(stdout);
                    if (output_eval_tensor && output_eval_tensor->dims) {
                        printf("[Consumer] eval dims exists, size=%d\n", output_eval_tensor->dims->size);
                        fflush(stdout);
                        if (output_eval_tensor->dims->size >= 3) {
                            output_dim = output_eval_tensor->dims->data[2];
                            printf("[Consumer] Output dimension: %zu\n", output_dim);
                        } else {
                            printf("[Consumer] WARNING: eval dims->size < 3\n");
                            fflush(stdout);
                            for (int d = 0; d < output_eval_tensor->dims->size; d++) {
                                printf("  dims->data[%d] = %d\n", d, output_eval_tensor->dims->data[d]);
                            }
                        }
                    } else {
                        printf("[Consumer] WARNING: eval_tensor or dims is NULL\n");
                        fflush(stdout);
                    }
                }
                
                // 保存全部10帧的logits
                // 注意：这里总是保存10帧，因为在正常处理中我们总是有完整的10帧
                for (size_t i = 0; i < CHUNK_SIZE; i++) {
                    std::vector<float> logit_frame(output_dim);
                    READ_OUTPUT_LOGIT_FRAME(logit_frame, i);
                    all_logits.push_back(logit_frame);
                }
                
                printf("[Consumer] Saved %zu logit frames (total: %zu)\n",
                       CHUNK_SIZE, all_logits.size());
                
                // ========================================
                // 6. 更新cache
                // ========================================
                for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
                    READ_CACHE_OUTPUT(i);
                }
                
                // 移除已处理的10帧，保留right_context的2帧
                std::vector<std::vector<float>> remaining(
                    frame_buffer.begin() + CHUNK_SIZE, frame_buffer.end());
                frame_buffer = remaining;
            }
        } else {
            // 队列已结束
            break;
        }
    }
    
    // ========================================
    // 步骤8：处理剩余的帧（flush）
    // ========================================
    if (!frame_buffer.empty()) {
        printf("[Consumer] Flushing %zu remaining frames\n", frame_buffer.size());
        
        size_t remaining_count = frame_buffer.size();
        
        // 如果剩余帧数>=12，可以正常处理
        while (frame_buffer.size() >= CHUNK_SIZE + RIGHT_CTX) {
            chunk_count++;
            
            // 准备输入数据（10帧）
            std::vector<float> input_data(CHUNK_SIZE * FEATURE_DIM);
            for (size_t i = 0; i < CHUNK_SIZE; i++) {
                memcpy(input_data.data() + i * FEATURE_DIM,
                       frame_buffer[i].data(),
                       FEATURE_DIM * sizeof(float));
            }
            
            // 准备right_context（2帧）
            std::vector<float> right_context_data(RIGHT_CTX * FEATURE_DIM);
            for (size_t i = 0; i < RIGHT_CTX; i++) {
                memcpy(right_context_data.data() + i * FEATURE_DIM,
                       frame_buffer[CHUNK_SIZE + i].data(),
                       FEATURE_DIM * sizeof(float));
            }
            
            // 设置输入tensors
            for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
                SET_CACHE_INPUT(i);
            }
            
            SET_RIGHT_CONTEXT(right_context_data);
            
            SET_MAIN_INPUT(input_data);
            
            // 运行推理
            status = interpreter.Invoke();
            if (status != kTfLiteOk) {
                printf("[Consumer] ERROR: Invoke() failed during flush\n");
                return;
            }
            
            // 获取输出
            TfLiteTensor* output_tensor = interpreter.output(logits_output_idx);
            if (output_tensor) {
                for (size_t i = 0; i < CHUNK_SIZE; i++) {
                    std::vector<float> logit_frame(output_dim);
                    READ_OUTPUT_LOGIT_FRAME(logit_frame, i);
                    all_logits.push_back(logit_frame);
                }
            }
            
            // 更新cache
            for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
                READ_CACHE_OUTPUT(i);
            }
            
            // 移除已处理的10帧
            std::vector<std::vector<float>> remaining(
                frame_buffer.begin() + CHUNK_SIZE, frame_buffer.end());
            frame_buffer = remaining;
        }
        
        // 处理最后不足12帧的情况
        if (!frame_buffer.empty()) {
            size_t final_count = frame_buffer.size();
            printf("[Consumer] Final flush: %zu frames\n", final_count);
            
            // 按照 run_encoder.cc 的方式处理：
            // 如果剩余帧数 >= 12，继续正常处理（10帧输入 + 2帧right_context）
            // 如果剩余帧数 < 12，padding 后处理，只输出实际的帧数
            
            while (frame_buffer.size() >= CHUNK_SIZE + RIGHT_CTX) {
                chunk_count++;
                
                // 准备输入数据（10帧）
                std::vector<float> input_data(CHUNK_SIZE * FEATURE_DIM);
                for (size_t i = 0; i < CHUNK_SIZE; i++) {
                    memcpy(input_data.data() + i * FEATURE_DIM,
                           frame_buffer[i].data(),
                           FEATURE_DIM * sizeof(float));
                }
                
                // 准备right_context（2帧）
                std::vector<float> right_context_data(RIGHT_CTX * FEATURE_DIM);
                for (size_t i = 0; i < RIGHT_CTX; i++) {
                    memcpy(right_context_data.data() + i * FEATURE_DIM,
                           frame_buffer[CHUNK_SIZE + i].data(),
                           FEATURE_DIM * sizeof(float));
                }
                
                // 设置输入tensors
                for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
                    SET_CACHE_INPUT(i);
                }
                
                SET_RIGHT_CONTEXT(right_context_data);
                
                SET_MAIN_INPUT(input_data);
                
                // 运行推理
                status = interpreter.Invoke();
                if (status != kTfLiteOk) {
                    printf("[Consumer] ERROR: Invoke() failed during flush\n");
                    return;
                }
                
                // 获取输出（保存全部10帧）
                TfLiteTensor* output_tensor = interpreter.output(logits_output_idx);
                if (output_tensor) {
                    for (size_t i = 0; i < CHUNK_SIZE; i++) {
                        std::vector<float> logit_frame(output_dim);
                        READ_OUTPUT_LOGIT_FRAME(logit_frame, i);
                        all_logits.push_back(logit_frame);
                    }
                }
                
                // 更新cache
                for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
                    READ_CACHE_OUTPUT(i);
                }
                
                printf("[Consumer] Flushed chunk: 10 logit frames (total: %zu)\n", all_logits.size());
                
                // 移除已处理的10帧
                std::vector<std::vector<float>> remaining(
                    frame_buffer.begin() + CHUNK_SIZE, frame_buffer.end());
                frame_buffer = remaining;
            }
            
            // 处理最后不足12帧的情况（< 12帧）
            // 按照 run_encoder.cc 的方式：每次处理最多10帧，如果还有剩余继续处理
            while (!frame_buffer.empty()) {
                size_t remaining_frames = frame_buffer.size();
                size_t current_chunk_size = std::min(remaining_frames, CHUNK_SIZE);
                
                printf("[Consumer] Final chunk: %zu frames remaining, processing %zu frames\n", 
                       remaining_frames, current_chunk_size);
                
                // 准备输入数据（10帧，不足的用0填充）
                std::vector<float> input_data(CHUNK_SIZE * FEATURE_DIM, 0.0f);  // 初始化为 0
                for (size_t i = 0; i < current_chunk_size; i++) {
                    memcpy(input_data.data() + i * FEATURE_DIM,
                           frame_buffer[i].data(),
                           FEATURE_DIM * sizeof(float));
                }
                
                // 准备right_context（2帧）
                // 如果有足够的剩余帧，使用实际的帧；否则用0填充
                std::vector<float> right_context_data(RIGHT_CTX * FEATURE_DIM, 0.0f);
                size_t available_rc_frames = (remaining_frames > current_chunk_size) 
                                              ? std::min(RIGHT_CTX, remaining_frames - current_chunk_size)
                                              : 0;
                for (size_t i = 0; i < available_rc_frames; i++) {
                    memcpy(right_context_data.data() + i * FEATURE_DIM,
                           frame_buffer[current_chunk_size + i].data(),
                           FEATURE_DIM * sizeof(float));
                }
                
                // 设置输入tensors
                for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
                    SET_CACHE_INPUT(i);
                }
                
                SET_RIGHT_CONTEXT(right_context_data);
                
                SET_MAIN_INPUT(input_data);
                
                // 运行推理
                printf("[Consumer] Running inference for final chunk (%zu frames, rc=%zu)...\n", 
                       current_chunk_size, available_rc_frames);
                status = interpreter.Invoke();
                if (status != kTfLiteOk) {
                    printf("[Consumer] ERROR: Invoke() failed during final chunk\n");
                    return;
                }
                
                // 获取输出（只保存 current_chunk_size 帧）
                TfLiteTensor* output_tensor = interpreter.output(logits_output_idx);
                if (output_tensor) {
                    // 只保存 current_chunk_size 帧（不是全部10帧）
                    printf("[Consumer] Saving %zu frames from inference output\n", current_chunk_size);
                    for (size_t i = 0; i < current_chunk_size; i++) {
                        std::vector<float> logit_frame(output_dim);
                        READ_OUTPUT_LOGIT_FRAME(logit_frame, i);
                        all_logits.push_back(logit_frame);
                    }
                } else {
                    printf("[Consumer] ERROR: output_tensor is NULL!\n");
                }
                
                // 更新cache（为下一次推理准备）
                for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
                    READ_CACHE_OUTPUT(i);
                }
                
                printf("[Consumer] Processed %zu frames (total: %zu)\n", 
                       current_chunk_size, all_logits.size());
                
                // 移除已处理的帧
                std::vector<std::vector<float>> remaining(
                    frame_buffer.begin() + current_chunk_size, frame_buffer.end());
                frame_buffer = remaining;
            }
        }
    }
    
    // 保存结果
    *output_logits = all_logits;
    frame_count->store(total_frames_received);
    
    printf("[Consumer] Thread finished\n");
    printf("  Total FBank frames received: %zu\n", total_frames_received);
    printf("  Total chunks processed: %zu\n", chunk_count);
    printf("  Total logit frames output: %zu\n", all_logits.size());
    printf("  ✅ Encoder inference completed successfully!\n");
}

// ========================================
// 保存特征到NumPy格式
// ========================================

bool save_npy(const std::string& filename,
              const std::vector<std::vector<float>>& features,
              size_t batch_size = 1) {
    if (features.empty()) {
        printf("Error: No features to save\n");
        return false;
    }
    
    size_t T = features.size();
    size_t D = features[0].size();
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        printf("Error: Cannot open file %s\n", filename.c_str());
        return false;
    }
    
    // Write .npy header
    uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    file.write(reinterpret_cast<char*>(magic), 6);
    
    uint8_t major_version = 1;
    uint8_t minor_version = 0;
    file.write(reinterpret_cast<char*>(&major_version), 1);
    file.write(reinterpret_cast<char*>(&minor_version), 1);
    
    char header[128];
    int header_len = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%zu, %zu, %zu), }",
        batch_size, T, D);
    
    while ((header_len + 10) % 16 != 0) {
        header[header_len++] = ' ';
    }
    header[header_len] = '\n';
    header_len++;
    
    uint16_t header_len_u16 = static_cast<uint16_t>(header_len);
    file.write(reinterpret_cast<char*>(&header_len_u16), 2);
    file.write(header, header_len);
    
    for (const auto& frame : features) {
        file.write(reinterpret_cast<const char*>(frame.data()), 
                   frame.size() * sizeof(float));
    }
    
    file.close();
    return true;
}

// ========================================
// 主函数
// ========================================

int main(int argc, char* argv[]) {
    printf("========================================\n");
    printf("Streaming FBank - 30ms Threaded Version (16x8 Quantized)\n");
    printf("========================================\n\n");
    
    // ========================================
    // 步骤1：配置参数
    // ========================================
    const char* wav_file = "/root/volume/ctc/ctc_tflite_micro/python/test_xiaoyun.wav";
    const char* model_file = "/root/volume/ctc/ctc_tflite_micro/python/tflite_models/fsmn_encoder_stateful_16x8.tflite";
    const char* output_file = "streaming_fbank_30ms_threaded_16x8_logits.npy";
    
    printf("Configuration:\n");
    printf("  Input WAV: %s\n", wav_file);
    printf("  Model file: %s\n", model_file);
    printf("  Output file: %s\n", output_file);
    printf("  Threading: Enabled (Producer-Consumer model)\n\n");
    
    // ========================================
    // 步骤2：加载音频文件
    // ========================================
    printf("Step 1: Loading WAV file...\n");
    WavData wav_data = WavReader::read_wav(wav_file);
    if (wav_data.samples.empty()) {
        printf("Error: Failed to load WAV file\n");
        return 1;
    }
    
    printf("  Sample rate: %d Hz\n", wav_data.sample_rate);
    printf("  Channels: %d\n", wav_data.num_channels);
    printf("  Total samples: %zu\n", wav_data.samples.size());
    printf("  Duration: %.2f seconds\n\n", 
           static_cast<float>(wav_data.samples.size()) / wav_data.sample_rate);
    
    // 转换为单声道
    std::vector<float> audio_samples;
    if (wav_data.num_channels == 1) {
        audio_samples = wav_data.samples;
    } else {
        printf("  Converting to mono...\n");
        audio_samples.resize(wav_data.samples.size() / wav_data.num_channels);
        for (size_t i = 0; i < audio_samples.size(); i++) {
            audio_samples[i] = wav_data.samples[i * wav_data.num_channels];
        }
    }
    printf("  Mono samples: %zu\n\n", audio_samples.size());
    
    // ========================================
    // 步骤3：配置FBank提取器
    // ========================================
    printf("Step 2: Configuring FBank extractor...\n");
    FBankConfig fbank_config;
    fbank_config.fs = 16000;
    fbank_config.n_mels = 80;
    fbank_config.frame_length = 25;
    fbank_config.frame_shift = 10;
    fbank_config.lfr_m = 5;
    fbank_config.lfr_n = 3;
    fbank_config.cmvn_file = "";
    
    printf("  Sample rate: %d Hz\n", fbank_config.fs);
    printf("  Mel bins: %d\n", fbank_config.n_mels);
    printf("  Frame length: %d ms\n", fbank_config.frame_length);
    printf("  Frame shift: %d ms\n", fbank_config.frame_shift);
    printf("  LFR: m=%d, n=%d\n", fbank_config.lfr_m, fbank_config.lfr_n);
    printf("  Output dimension: %d\n\n", fbank_config.n_mels * fbank_config.lfr_m);
    
    // ========================================
    // 步骤4：创建线程安全队列和消费者线程
    // ========================================
    printf("Step 3: Starting consumer thread...\n");
    
    // 创建特征帧队列
    FeatureQueue feature_queue;
    
    // 消费者线程的输出（logits而不是特征帧）
    std::vector<std::vector<float>> consumer_output_logits;
    std::atomic<size_t> consumer_frame_count(0);
    
    // 启动消费者线程（传入模型文件路径）
    std::thread consumer(consumer_thread, 
                        &feature_queue, 
                        &consumer_output_logits, 
                        &consumer_frame_count,
                        model_file);
    
    printf("  Consumer thread started\n\n");
    
    // ========================================
    // 步骤5：主线程处理音频（生产者）
    // ========================================
    printf("Step 4: Processing audio in main thread (Producer)...\n");
    printf("========================================\n");
    
    StreamingFBankExtractor fbank_extractor(fbank_config);
    
    size_t input_chunk_size = (30 * fbank_config.fs) / 1000;      // 480 samples
    size_t process_chunk_size = (10 * fbank_config.fs) / 1000;    // 160 samples
    
    printf("[Producer] Chunk sizes:\n");
    printf("  Input chunk: %zu samples (30ms)\n", input_chunk_size);
    printf("  Process chunk: %zu samples (10ms)\n\n", process_chunk_size);
    
    size_t total_input_chunks = 0;
    size_t total_frames_produced = 0;
    
    printf("[Producer] Processing:\n");
    
    // 主线程：按30ms块处理音频
    for (size_t i = 0; i < audio_samples.size(); i += input_chunk_size) {
        size_t remaining = audio_samples.size() - i;
        size_t current_input_size = (remaining < input_chunk_size) ? remaining : input_chunk_size;
        
        // 将30ms块分解为10ms子块处理
        for (size_t j = 0; j < current_input_size; j += process_chunk_size) {
            size_t sub_remaining = current_input_size - j;
            size_t current_process_size = (sub_remaining < process_chunk_size) ? 
                                          sub_remaining : process_chunk_size;
            
            // 提取FBank特征
            std::vector<std::vector<float>> lfr_frames;
            int num_lfr = fbank_extractor.process_chunk(
                audio_samples.data() + i + j,
                current_process_size,
                lfr_frames
            );
            
            // 将特征帧放入队列（供消费者线程处理）
            if (num_lfr > 0) {
                for (const auto& frame : lfr_frames) {
                    feature_queue.push(frame);
                    total_frames_produced++;
                }
            }
        }
        
        total_input_chunks++;
        
        // 每处理10个30ms块，打印一次进度
        if (total_input_chunks % 10 == 0) {
            printf("[Producer] Processed %zu input chunks, produced %zu frames, queue size: %zu\n", 
                   total_input_chunks, total_frames_produced, feature_queue.size());
        }
    }
    
    printf("\n[Producer] Processed all input chunks:\n");
    printf("  Total 30ms chunks: %zu\n", total_input_chunks);
    printf("  Frames produced so far: %zu\n\n", total_frames_produced);
    
    // ========================================
    // 步骤6：刷新缓冲区
    // ========================================
    printf("[Producer] Flushing remaining frames...\n");
    
    std::vector<std::vector<float>> final_frames;
    int final_count = fbank_extractor.flush(final_frames);
    
    if (final_count > 0) {
        for (const auto& frame : final_frames) {
            feature_queue.push(frame);
            total_frames_produced++;
        }
        printf("  Flushed %d final frames\n", final_count);
    }
    
    printf("[Producer] Total frames produced: %zu\n\n", total_frames_produced);
    
    // ========================================
    // 步骤7：通知消费者线程结束并等待
    // ========================================
    printf("Step 5: Finishing up...\n");
    
    // 标记生产者已完成
    feature_queue.finish();
    printf("  Notified consumer thread to finish\n");
    
    // 等待消费者线程完成
    printf("  Waiting for consumer thread...\n");
    consumer.join();
    printf("  Consumer thread joined\n\n");
    
    // ========================================
    // 步骤8：验证结果
    // ========================================
    printf("========================================\n");
    printf("Processing Complete!\n");
    printf("========================================\n");
    printf("Producer produced: %zu FBank frames\n", total_frames_produced);
    printf("Consumer processed: %zu FBank frames\n", consumer_frame_count.load());
    
    if (total_frames_produced == consumer_frame_count.load()) {
        printf("✅ Frame count matches!\n");
    } else {
        printf("⚠️  Frame count mismatch!\n");
    }
    
    // 注意：consumer_output_logits现在是logits而不是FBank特征
    if (!consumer_output_logits.empty()) {
        printf("Output logits shape: (1, %zu, %zu)\n\n", 
               consumer_output_logits.size(), consumer_output_logits[0].size());
    } else {
        printf("⚠️  No logits output\n\n");
    }
    
    // ========================================
    // 步骤9：保存结果
    // ========================================
    printf("Step 6: Saving results...\n");
    
    if (!consumer_output_logits.empty()) {
        if (save_npy(output_file, consumer_output_logits)) {
            printf("  ✅ Saved to %s\n", output_file);
            printf("  Shape: (1, %zu, %zu)\n", 
                   consumer_output_logits.size(), 
                   consumer_output_logits[0].size());
        } else {
            printf("  ❌ Failed to save file\n");
            return 1;
        }
    } else {
        printf("  ⚠️  Skipping save (no logits output)\n");
    }
    
    // ========================================
    // 步骤10：显示统计信息
    // ========================================
    printf("\nStatistics:\n");
    printf("  Input audio duration: %.2f seconds\n", 
           static_cast<float>(audio_samples.size()) / fbank_config.fs);
    printf("  30ms chunks processed: %zu\n", total_input_chunks);
    printf("  FBank frames produced: %zu\n", total_frames_produced);
    printf("  FBank frames consumed: %zu\n", consumer_frame_count.load());
    printf("  Feature dimension: %d\n", fbank_config.n_mels * fbank_config.lfr_m);
    if (!consumer_output_logits.empty()) {
        printf("  Logits dimension: %zu\n", consumer_output_logits[0].size());
    }
    
    printf("\n========================================\n");
    printf("Done!\n");
    printf("========================================\n");
    
    return 0;
}
