# 16x8 量化版本完整流程

## 概述

16x8 量化版本实现了流式音频处理 + FBank 特征提取 + Encoder 推理的完整流程，使用 INT16/INT8 量化模型以减少模型大小和提高推理速度。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      主线程 (生产者)                          │
│                                                               │
│  WAV 文件 → 30ms 音频块 → 3×10ms 子块 → FBank 提取          │
│                                    ↓                          │
│                              LFR 处理 (5帧→1帧)               │
│                                    ↓                          │
│                              CMVN 归一化                      │
│                                    ↓                          │
│                          线程安全队列 (特征帧)                 │
└─────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────┐
│                    消费者线程 (推理)                          │
│                                                               │
│  特征帧队列 → 累积到12帧 → 量化 → Encoder 推理 → 反量化      │
│                                    ↓                          │
│                              Logits 输出                      │
│                                    ↓                          │
│                          CTC 解码 (可选)                      │
└─────────────────────────────────────────────────────────────┘
```

## 详细流程

### 1. 初始化阶段

#### 1.1 主线程初始化
```cpp
// 加载 WAV 文件
WavData wav_data = WavReader::read_wav(wav_file);

// 配置 FBank 提取器
FBankConfig fbank_config;
fbank_config.fs = 16000;           // 采样率
fbank_config.n_mels = 80;          // Mel 滤波器数量
fbank_config.frame_length = 25;    // 帧长 25ms
fbank_config.frame_shift = 10;     // 帧移 10ms
fbank_config.lfr_m = 5;            // LFR: 5帧合并为1帧
fbank_config.lfr_n = 3;            // LFR: 跳过3帧

StreamingFBankExtractor fbank_extractor(fbank_config);
```

#### 1.2 消费者线程初始化
```cpp
// 加载 TFLite 模型
const tflite::Model* model = tflite::GetModel(model_data.data());

// 创建 Op Resolver (注册所需算子)
tflite::MicroMutableOpResolver<50> resolver;
resolver.AddFullyConnected();
resolver.AddConv2D();
// ... 添加其他算子

// 分配 Tensor Arena (3MB)
constexpr int kTensorArenaSize = 3 * 1024 * 1024;
static alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// 创建 Interpreter
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                   kTensorArenaSize, nullptr, nullptr, true);

// 分配 tensors
interpreter.AllocateTensors();
```

### 2. 量化参数获取

#### 2.1 获取输入 tensor 量化参数
```cpp
// 主输入 (10帧 × 400维)
float input_scale = 1.0f;
int32_t input_zero_point = 0;
TfLiteTensor* input_tensor = interpreter.input(input_tensor_idx);
GetQuantizationParams(input_tensor, input_scale, input_zero_point);

// Right context (2帧 × 400维)
float rc_scale = 1.0f;
int32_t rc_zero_point = 0;
TfLiteTensor* rc_tensor = interpreter.input(right_context_idx);
GetQuantizationParams(rc_tensor, rc_scale, rc_zero_point);

// Cache inputs (4层，每层 1×128×9)
float cache_in_scales[4];
int32_t cache_in_zero_points[4];
for (size_t i = 0; i < 4; i++) {
    TfLiteTensor* cache_tensor = interpreter.input(cache_input_indices[i]);
    GetQuantizationParams(cache_tensor, cache_in_scales[i], cache_in_zero_points[i]);
}
```

#### 2.2 获取输出 tensor 量化参数
```cpp
// Logits 输出 (10帧 × 2599维)
float output_scale = 1.0f;
int32_t output_zero_point = 0;
TfLiteTensor* output_tensor = interpreter.output(logits_output_idx);
GetQuantizationParams(output_tensor, output_scale, output_zero_point);

// Cache outputs (4层)
float cache_out_scales[4];
int32_t cache_out_zero_points[4];
for (size_t i = 0; i < 4; i++) {
    TfLiteTensor* cache_out_tensor = interpreter.output(cache_output_indices[i]);
    GetQuantizationParams(cache_out_tensor, cache_out_scales[i], cache_out_zero_points[i]);
}
```

### 3. 主线程处理流程 (生产者)

#### 3.1 音频块处理
```cpp
// 每次读取 30ms 音频块 (480 samples @ 16kHz)
size_t input_chunk_size = 480;
size_t process_chunk_size = 160;  // 10ms

for (size_t i = 0; i < audio_samples.size(); i += input_chunk_size) {
    // 将 30ms 块分解为 3×10ms 子块
    for (size_t j = 0; j < current_input_size; j += process_chunk_size) {
        // 提取 FBank 特征
        std::vector<std::vector<float>> lfr_frames;
        int num_lfr = fbank_extractor.process_chunk(
            audio_samples.data() + i + j,
            current_process_size,
            lfr_frames
        );
        
        // 将特征帧放入队列
        for (const auto& frame : lfr_frames) {
            feature_queue.push(frame);  // 线程安全
        }
    }
}

// 刷新剩余帧
std::vector<std::vector<float>> final_frames;
fbank_extractor.flush(final_frames);
for (const auto& frame : final_frames) {
    feature_queue.push(frame);
}

// 通知消费者线程结束
feature_queue.finish();
```

### 4. 消费者线程处理流程 (推理)

#### 4.1 累积特征帧
```cpp
std::vector<std::vector<float>> frame_buffer;

while (true) {
    std::vector<float> frame;
    if (!queue->pop(frame)) break;  // 队列结束
    
    frame_buffer.push_back(frame);
    
    // 判断是否可以推理
    if (frame_buffer.size() >= CHUNK_SIZE + RIGHT_CTX) {  // 12帧
        // 进行推理
        process_chunk(frame_buffer);
        
        // 移除已处理的10帧，保留2帧作为下次的 right_context
        frame_buffer.erase(frame_buffer.begin(), frame_buffer.begin() + CHUNK_SIZE);
    }
}
```

#### 4.2 量化输入数据
```cpp
// 准备输入数据 (10帧 × 400维 = 4000)
std::vector<float> input_data(CHUNK_SIZE * FEATURE_DIM);
for (size_t i = 0; i < CHUNK_SIZE; i++) {
    memcpy(input_data.data() + i * FEATURE_DIM,
           frame_buffer[i].data(),
           FEATURE_DIM * sizeof(float));
}

// 准备 right_context (2帧 × 400维 = 800)
std::vector<float> right_context_data(RIGHT_CTX * FEATURE_DIM);
for (size_t i = 0; i < RIGHT_CTX; i++) {
    memcpy(right_context_data.data() + i * FEATURE_DIM,
           frame_buffer[CHUNK_SIZE + i].data(),
           FEATURE_DIM * sizeof(float));
}

// 使用宏设置 tensors (自动量化)
for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
    SET_CACHE_INPUT(i);  // 量化 cache 输入
}
SET_RIGHT_CONTEXT(right_context_data);  // 量化 right_context
SET_MAIN_INPUT(input_data);  // 量化主输入
```

#### 4.3 量化宏定义
```cpp
// 宏：设置 cache 输入（自动检测类型并量化）
#define SET_CACHE_INPUT(cache_idx) \
    do { \
        int tensor_idx = subgraph->inputs()->Get(cache_input_indices[cache_idx]); \
        TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0); \
        if (eval_tensor->type == kTfLiteInt8) { \
            int8_t* data = eval_tensor->data.int8; \
            for (size_t j = 0; j < caches[cache_idx].size(); j++) { \
                data[j] = QuantizeFloatToInt8(caches[cache_idx][j], \
                    cache_in_scales[cache_idx], cache_in_zero_points[cache_idx]); \
            } \
        } else if (eval_tensor->type == kTfLiteInt16) { \
            int16_t* data = eval_tensor->data.i16; \
            for (size_t j = 0; j < caches[cache_idx].size(); j++) { \
                data[j] = QuantizeFloatToInt16(caches[cache_idx][j], \
                    cache_in_scales[cache_idx], cache_in_zero_points[cache_idx]); \
            } \
        } else if (eval_tensor->type == kTfLiteFloat32) { \
            float* data = eval_tensor->data.f; \
            memcpy(data, caches[cache_idx].data(), \
                   caches[cache_idx].size() * sizeof(float)); \
        } \
    } while(0)
```

#### 4.4 运行推理
```cpp
// 运行推理
TfLiteStatus status = interpreter.Invoke();
if (status != kTfLiteOk) {
    printf("ERROR: Invoke() failed\n");
    return;
}
```

#### 4.5 反量化输出数据
```cpp
// 获取输出维度（使用 TfLiteEvalTensor）
if (output_dim == 0) {
    int output_tensor_idx = subgraph->outputs()->Get(logits_output_idx);
    TfLiteEvalTensor* output_eval_tensor = interpreter.GetTensor(output_tensor_idx, 0);
    if (output_eval_tensor && output_eval_tensor->dims) {
        output_dim = output_eval_tensor->dims->data[2];  // 2599
    }
}

// 读取并反量化输出 logits
for (size_t i = 0; i < CHUNK_SIZE; i++) {
    std::vector<float> logit_frame(output_dim);
    READ_OUTPUT_LOGIT_FRAME(logit_frame, i);  // 自动反量化
    all_logits.push_back(logit_frame);
}

// 更新 cache 状态（反量化）
for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
    READ_CACHE_OUTPUT(i);  // 自动反量化
}
```

#### 4.6 反量化宏定义
```cpp
// 宏：读取输出 logits（自动检测类型并反量化）
#define READ_OUTPUT_LOGIT_FRAME(output_vec, frame_idx) \
    do { \
        int tensor_idx = subgraph->outputs()->Get(logits_output_idx); \
        TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0); \
        if (eval_tensor->type == kTfLiteInt8) { \
            int8_t* data = eval_tensor->data.int8; \
            for (size_t j = 0; j < output_dim; j++) { \
                output_vec[j] = DequantizeInt8ToFloat( \
                    data[frame_idx * output_dim + j], \
                    output_scale, output_zero_point); \
            } \
        } else if (eval_tensor->type == kTfLiteInt16) { \
            int16_t* data = eval_tensor->data.i16; \
            for (size_t j = 0; j < output_dim; j++) { \
                output_vec[j] = DequantizeInt16ToFloat( \
                    data[frame_idx * output_dim + j], \
                    output_scale, output_zero_point); \
            } \
        } else if (eval_tensor->type == kTfLiteFloat32) { \
            float* data = eval_tensor->data.f; \
            memcpy(output_vec.data(), data + frame_idx * output_dim, \
                   output_dim * sizeof(float)); \
        } \
    } while(0)
```

### 5. 量化/反量化辅助函数

#### 5.1 量化函数
```cpp
// Float32 → INT8
inline int8_t QuantizeFloatToInt8(float value, float scale, int32_t zero_point) {
    int32_t quantized = static_cast<int32_t>(std::round(value / scale + zero_point));
    quantized = std::max(-128, std::min(127, quantized));  // Clip
    return static_cast<int8_t>(quantized);
}

// Float32 → INT16
inline int16_t QuantizeFloatToInt16(float value, float scale, int32_t zero_point) {
    int32_t quantized = static_cast<int32_t>(std::round(value / scale + zero_point));
    quantized = std::max(-32768, std::min(32767, quantized));  // Clip
    return static_cast<int16_t>(quantized);
}
```

#### 5.2 反量化函数
```cpp
// INT8 → Float32
inline float DequantizeInt8ToFloat(int8_t value, float scale, int32_t zero_point) {
    return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}

// INT16 → Float32
inline float DequantizeInt16ToFloat(int16_t value, float scale, int32_t zero_point) {
    return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}
```

#### 5.3 获取量化参数
```cpp
bool GetQuantizationParams(const TfLiteTensor* tensor, float& scale, int32_t& zero_point) {
    if (!tensor || tensor->quantization.type != kTfLiteAffineQuantization) {
        return false;
    }
    
    const TfLiteAffineQuantization* quant_params =
        static_cast<const TfLiteAffineQuantization*>(tensor->quantization.params);
    
    if (!quant_params || !quant_params->scale || 
        quant_params->scale->size == 0 || !quant_params->scale->data) {
        return false;
    }
    
    scale = quant_params->scale->data[0];
    zero_point = (quant_params->zero_point && quant_params->zero_point->data) 
                 ? quant_params->zero_point->data[0] : 0;
    return true;
}
```

### 6. 最后一帧处理 (Flush)

```cpp
// 处理剩余帧
while (!frame_buffer.empty()) {
    size_t remaining_frames = frame_buffer.size();
    size_t current_chunk_size = std::min(remaining_frames, CHUNK_SIZE);
    
    // 准备输入（不足10帧用0填充）
    std::vector<float> input_data(CHUNK_SIZE * FEATURE_DIM, 0.0f);
    for (size_t i = 0; i < current_chunk_size; i++) {
        memcpy(input_data.data() + i * FEATURE_DIM,
               frame_buffer[i].data(),
               FEATURE_DIM * sizeof(float));
    }
    
    // 准备 right_context（不足用0填充）
    std::vector<float> right_context_data(RIGHT_CTX * FEATURE_DIM, 0.0f);
    size_t available_rc_frames = (remaining_frames > current_chunk_size) 
                                  ? std::min(RIGHT_CTX, remaining_frames - current_chunk_size)
                                  : 0;
    for (size_t i = 0; i < available_rc_frames; i++) {
        memcpy(right_context_data.data() + i * FEATURE_DIM,
               frame_buffer[current_chunk_size + i].data(),
               FEATURE_DIM * sizeof(float));
    }
    
    // 量化、推理、反量化
    SET_CACHE_INPUT(...);
    SET_RIGHT_CONTEXT(right_context_data);
    SET_MAIN_INPUT(input_data);
    interpreter.Invoke();
    
    // 只保存实际的帧数（不是全部10帧）
    for (size_t i = 0; i < current_chunk_size; i++) {
        std::vector<float> logit_frame(output_dim);
        READ_OUTPUT_LOGIT_FRAME(logit_frame, i);
        all_logits.push_back(logit_frame);
    }
    
    // 移除已处理的帧
    frame_buffer.erase(frame_buffer.begin(), 
                      frame_buffer.begin() + current_chunk_size);
}
```

### 7. 输出保存

```cpp
// 保存为 NumPy 格式
bool save_npy(const std::string& filename,
              const std::vector<std::vector<float>>& features,
              size_t batch_size = 1) {
    // 写入 .npy header
    // 写入数据
    // 形状: (batch_size, T, D) = (1, 151, 2599)
}
```

## 关键数据流

### 输入数据流
```
WAV (72462 samples, 4.53s)
    ↓
30ms 块 (480 samples) × 151
    ↓
10ms 子块 (160 samples) × 453
    ↓
FBank 帧 (80 dims) × 453
    ↓
LFR 处理 (5→1)
    ↓
LFR 帧 (400 dims) × 151
    ↓
CMVN 归一化
    ↓
特征帧队列 (400 dims) × 151
```

### 推理数据流
```
特征帧 (400 dims) × 12
    ↓
分离: 主输入 (10×400) + Right Context (2×400)
    ↓
量化: Float32 → INT16/INT8
    ↓
Encoder 推理 (16x8 量化模型)
    ↓
反量化: INT16/INT8 → Float32
    ↓
Logits (10×2599)
    ↓
累积所有 chunks
    ↓
最终输出 (151×2599)
```

## 性能指标

### 模型信息
- **模型文件**: fsmn_encoder_stateful_16x8.tflite
- **模型大小**: ~951KB
- **Tensor Arena**: 3MB
- **实际使用**: 417KB (13.27%)

### 处理统计
- **输入音频**: 4.53秒
- **30ms 块数**: 151
- **FBank 帧数**: 151
- **Encoder chunks**: 14
- **输出 logits**: 151 帧 × 2599 维

### 精度指标
- **与 PyTorch 参考对比**:
  - 最大差异: 396.49
  - 平均差异: 1.43
  - 关键词检测: ✅ 小云小云
  - 置信度: 97.46%

## 使用示例

### 编译
```bash
cd tflite_micro_cpp/streaming_fbank_only
bash run_30ms_threaded_16x8.sh
```

### 运行
```bash
cd build
./streaming_fbank_30ms_threaded_16x8
```

### 验证
```bash
python3 verify_encoder_output.py \
    features/test_xiaoyun_fbank.npy \
    tflite_micro_cpp/streaming_fbank_only/build/streaming_fbank_30ms_threaded_16x8_logits.npy \
    /root/volume/ctc/train/funasr_test/test_xiaoyun.wav
```

## 关键优化点

### 1. 使用宏简化代码
- 自动检测 tensor 类型
- 自动量化/反量化
- 减少代码重复

### 2. 使用 TfLiteEvalTensor
- 避免 TfLiteTensor 的 dims 指针问题
- 直接访问底层数据
- 更可靠的 tensor 访问

### 3. 量化参数 Fallback
- 优先从 TfLiteTensor 获取
- 失败时从 flatbuffer 获取
- 确保总能获取到量化参数

### 4. 线程安全
- 使用条件变量同步
- 队列操作加锁
- 优雅的结束处理

## 总结

16x8 量化版本实现了完整的流式音频处理流程，通过量化技术在保持高精度的同时减少了模型大小和内存占用。关键特性包括：

✅ **自动量化/反量化** - 宏定义简化操作  
✅ **多类型支持** - INT8/INT16/FLOAT32  
✅ **线程安全** - 生产者-消费者模式  
✅ **流式处理** - 低延迟实时处理  
✅ **高精度** - 97.46% 关键词检测置信度  

**状态**: 🎉 **完成并验证通过**
