# FBank INT16 输出实现指南

## 目标

修改 FBank 提取器，使其直接输出 INT16 格式的特征，供 16x8 Encoder 直接使用。

## 修改范围

### 需要修改的文件

```
tflite_micro_cpp/fbank_extractor_int16/
├── streaming_fbank_extractor_int16.h    # 新建：INT16 版本头文件
├── streaming_fbank_extractor_int16.cc   # 新建：INT16 版本实现
├── feature_queue_int16.h                # 新建：INT16 队列
└── quantization_utils.h                 # 新建：量化工具

tflite_micro_cpp/streaming_fbank_only/
└── src/
    └── streaming_fbank_30ms_threaded_16x8_optimized.cc  # 新建：优化版本
```

## 关键修改点

### 1. 从 Encoder 获取量化参数

**位置**: 消费者线程初始化时

**当前代码**:
```cpp
// 在 consumer_thread 中
float input_scale = 1.0f;
int32_t input_zero_point = 0;
TfLiteTensor* input_tensor = interpreter.input(input_tensor_idx);
GetQuantizationParams(input_tensor, input_scale, input_zero_point);
```

**修改为**:
```cpp
// 提前获取量化参数，传递给生产者
struct QuantizationParams {
    float input_scale;
    int32_t input_zero_point;
    float rc_scale;
    int32_t rc_zero_point;
};

QuantizationParams get_quantization_params(tflite::MicroInterpreter& interpreter) {
    QuantizationParams params;
    // 获取主输入量化参数
    TfLiteTensor* input_tensor = interpreter.input(input_tensor_idx);
    GetQuantizationParams(input_tensor, params.input_scale, params.input_zero_point);
    
    // 获取 right_context 量化参数
    TfLiteTensor* rc_tensor = interpreter.input(right_context_idx);
    GetQuantizationParams(rc_tensor, params.rc_scale, params.rc_zero_point);
    
    return params;
}
```

### 2. 修改队列为 INT16

**新建文件**: `feature_queue_int16.h`

```cpp
class FeatureQueueINT16 {
public:
    void push(const std::vector<int16_t>& frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(frame);
        cv_.notify_one();
    }
    
    bool pop(std::vector<int16_t>& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || finished_; });
        if (queue_.empty() && finished_) return false;
        frame = queue_.front();
        queue_.pop();
        return true;
    }
    
    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        cv_.notify_all();
    }
    
private:
    std::queue<std::vector<int16_t>> queue_;  // INT16 队列
    std::mutex mutex_;
    std::condition_variable cv_;
    bool finished_ = false;
};
```

### 3. 修改 FBank 提取器输出

**新建文件**: `streaming_fbank_extractor_int16.h`

```cpp
class StreamingFBankExtractorINT16 {
public:
    StreamingFBankExtractorINT16(const FBankConfig& config,
                                 float scale,
                                 int32_t zero_point)
        : config_(config), 
          scale_(scale), 
          zero_point_(zero_point) {
        // 初始化 float32 版本的提取器
        float_extractor_ = std::make_unique<StreamingFBankExtractor>(config);
    }
    
    // 处理音频块，直接输出 INT16
    int process_chunk(const float* audio_chunk,
                      size_t chunk_length,
                      std::vector<std::vector<int16_t>>& output_frames_int16) {
        
        // 1. 使用 float32 提取器处理
        std::vector<std::vector<float>> output_frames_float;
        int num_frames = float_extractor_->process_chunk(
            audio_chunk, chunk_length, output_frames_float);
        
        // 2. 立即量化为 INT16
        output_frames_int16.clear();
        for (const auto& frame_float : output_frames_float) {
            std::vector<int16_t> frame_int16(frame_float.size());
            quantize_frame(frame_float, frame_int16);
            output_frames_int16.push_back(frame_int16);
        }
        
        return num_frames;
    }
    
    int flush(std::vector<std::vector<int16_t>>& output_frames_int16) {
        std::vector<std::vector<float>> output_frames_float;
        int num_frames = float_extractor_->flush(output_frames_float);
        
        output_frames_int16.clear();
        for (const auto& frame_float : output_frames_float) {
            std::vector<int16_t> frame_int16(frame_float.size());
            quantize_frame(frame_float, frame_int16);
            output_frames_int16.push_back(frame_int16);
        }
        
        return num_frames;
    }
    
private:
    FBankConfig config_;
    float scale_;
    int32_t zero_point_;
    std::unique_ptr<StreamingFBankExtractor> float_extractor_;
    
    void quantize_frame(const std::vector<float>& frame_float,
                       std::vector<int16_t>& frame_int16) {
        for (size_t i = 0; i < frame_float.size(); i++) {
            frame_int16[i] = QuantizeFloatToInt16(
                frame_float[i], scale_, zero_point_);
        }
    }
};
```

### 4. 修改主程序

**新建文件**: `streaming_fbank_30ms_threaded_16x8_optimized.cc`

**关键修改**:

```cpp
// ========================================
// 步骤 1: 启动消费者线程并获取量化参数
// ========================================

// 创建队列（INT16）
FeatureQueueINT16 feature_queue_int16;

// 用于存储量化参数
std::atomic<bool> quant_params_ready(false);
QuantizationParams quant_params;

// 启动消费者线程
std::thread consumer([&]() {
    // ... 初始化 interpreter ...
    
    // 获取量化参数
    quant_params = get_quantization_params(interpreter);
    quant_params_ready.store(true);
    
    // ... 继续推理循环 ...
});

// 等待量化参数准备好
while (!quant_params_ready.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

// ========================================
// 步骤 2: 创建 INT16 FBank 提取器
// ========================================

StreamingFBankExtractorINT16 fbank_extractor_int16(
    fbank_config,
    quant_params.input_scale,
    quant_params.input_zero_point
);

// ========================================
// 步骤 3: 生产者处理（输出 INT16）
// ========================================

for (size_t i = 0; i < audio_samples.size(); i += input_chunk_size) {
    for (size_t j = 0; j < current_input_size; j += process_chunk_size) {
        // 提取 FBank 特征（直接输出 INT16）
        std::vector<std::vector<int16_t>> lfr_frames_int16;
        int num_lfr = fbank_extractor_int16.process_chunk(
            audio_samples.data() + i + j,
            current_process_size,
            lfr_frames_int16
        );
        
        // 放入 INT16 队列
        for (const auto& frame : lfr_frames_int16) {
            feature_queue_int16.push(frame);
        }
    }
}

// ========================================
// 步骤 4: 消费者处理（直接使用 INT16）
// ========================================

std::vector<std::vector<int16_t>> frame_buffer_int16;

while (true) {
    std::vector<int16_t> frame_int16;
    if (!queue->pop(frame_int16)) break;
    
    frame_buffer_int16.push_back(frame_int16);
    
    if (frame_buffer_int16.size() >= 12) {
        // 准备输入（已经是 INT16）
        std::vector<int16_t> input_data_int16(10 * 400);
        for (size_t i = 0; i < 10; i++) {
            memcpy(input_data_int16.data() + i * 400,
                   frame_buffer_int16[i].data(),
                   400 * sizeof(int16_t));
        }
        
        // 准备 right_context（已经是 INT16）
        std::vector<int16_t> right_context_int16(2 * 400);
        for (size_t i = 0; i < 2; i++) {
            memcpy(right_context_int16.data() + i * 400,
                   frame_buffer_int16[10 + i].data(),
                   400 * sizeof(int16_t));
        }
        
        // 直接复制到 tensor（无需量化！）
        int tensor_idx = subgraph->inputs()->Get(input_tensor_idx);
        TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0);
        int16_t* tensor_data = eval_tensor->data.i16;
        memcpy(tensor_data, input_data_int16.data(), 
               10 * 400 * sizeof(int16_t));
        
        // Right context 同样直接复制
        int rc_tensor_idx = subgraph->inputs()->Get(right_context_idx);
        TfLiteEvalTensor* rc_eval_tensor = interpreter.GetTensor(rc_tensor_idx, 0);
        int16_t* rc_tensor_data = rc_eval_tensor->data.i16;
        memcpy(rc_tensor_data, right_context_int16.data(),
               2 * 400 * sizeof(int16_t));
        
        // 推理...
    }
}
```

## 修改总结

### 文件修改清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `feature_queue_int16.h` | 新建 | INT16 队列 |
| `quantization_utils.h` | 新建 | 量化工具函数 |
| `streaming_fbank_extractor_int16.h` | 新建 | INT16 FBank 提取器头文件 |
| `streaming_fbank_extractor_int16.cc` | 新建 | INT16 FBank 提取器实现 |
| `streaming_fbank_30ms_threaded_16x8_optimized.cc` | 新建 | 优化版主程序 |
| `CMakeLists.txt` | 修改 | 添加新的编译目标 |
| `run_30ms_threaded_16x8_optimized.sh` | 新建 | 编译运行脚本 |

### 代码行数估算

```
feature_queue_int16.h:                    ~80 行
quantization_utils.h:                     ~50 行
streaming_fbank_extractor_int16.h:        ~60 行
streaming_fbank_extractor_int16.cc:       ~100 行
streaming_fbank_30ms_threaded_16x8_optimized.cc:  ~1500 行（基于现有代码修改）
CMakeLists.txt:                           ~10 行
run_30ms_threaded_16x8_optimized.sh:      ~20 行

总计: ~1820 行（其中大部分是复制现有代码）
```

### 预期收益

```
内存占用:
  当前: 38.4 KB (队列 + 缓冲区)
  优化: 19.2 KB (队列 + 缓冲区)
  节省: 50%

处理速度:
  生产者: 增加 ~5% 开销（量化）
  消费者: 减少 ~10% 开销（无需量化）
  总体: 提升 ~5-10%

精度:
  无变化（量化时机不影响精度）
```

## 下一步

1. 创建新文件夹和文件
2. 实现量化工具函数
3. 实现 INT16 队列
4. 实现 INT16 FBank 提取器
5. 修改主程序
6. 测试验证

准备好开始实现了吗？
