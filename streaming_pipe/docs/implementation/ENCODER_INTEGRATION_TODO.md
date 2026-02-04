# Encoder Integration - 真正的流式处理

## 关键约束

- ❌ **不能一次取出所有帧**（FBank可能还在生成中）
- ✅ **必须边生成边处理**（真正的流式）
- ✅ **每次累积到10帧后立即处理**
- ✅ **使用滑动窗口保持连续性**

## 流式处理策略

### 时间线示例

```
时间 | 队列状态 | 缓冲区 | 动作
-----|---------|--------|------
t1   | 1帧到达 | [1]    | 等待
t2   | 1帧到达 | [1,2]  | 等待
...  | ...     | ...    | ...
t12  | 1帧到达 | [1-12] | ✓ 第1次推理
     |         |        |   输入: 帧[1-10]
     |         |        |   Right context: 帧[11-12]（从buffer取）
     |         |        |   输出: logits[1-10]
     |         | [11,12]|   保留: 帧[11-12]
t13  | 1帧到达 | [11,12,13] | 等待
...  | ...     | ...    | ...
t20  | 1帧到达 | [11-20]| ✓ 第2次推理
     |         |        |   输入: 帧[11-20]（共10帧）
     |         |        |   Right context: 帧[21-22]（从队列peek）
     |         |        |   输出: logits[13-20]（只保存新的8帧）
     |         | [19,20]|   保留: 帧[19-20]
...  | ...     | ...    | ...
```

### 处理规则

**第一次推理**（累积到12帧时）：
- 触发条件：`buffer.size() >= 12`（10帧输入 + 2帧right_context）
- 输入：帧[0-9]（10帧）
- Right context：帧[10-11]（从buffer中取）
- 输出：保存全部10帧的logits
- 保留：帧[10-11]（最后2帧，注意是right_context的那2帧）

**后续推理**（每累积8帧新数据）：
- 触发条件：`buffer.size() >= 10`（包含上次保留的2帧）
- 输入：上次保留的2帧 + 新的8帧 = 10帧
- Right context：从队列peek接下来的2帧，如果没有则用零
- 输出：只保存新的8帧的logits（跳过前2帧，因为已经输出过）
- 保留：最后2帧

**最后推理**（队列结束时）：
- 触发条件：队列为空且buffer中还有帧
- 输入：剩余帧padding到10帧
- Right context：用零填充
- 输出：保存所有剩余帧的logits（跳过前2帧）

## 实现代码框架

```cpp
void consumer_thread(FeatureQueue* queue, 
                    std::vector<std::vector<float>>* output_logits,
                    std::atomic<size_t>* frame_count,
                    const char* model_file) {
    printf("[Consumer] Thread started\n");
    
    // 1. 加载TFLite模型和初始化interpreter
    // ... (参考run_encoder.cc)
    
    // 2. 初始化cache
    constexpr size_t NUM_CACHE_LAYERS = 4;
    constexpr size_t PROJ_DIM = 128;
    constexpr size_t LEFT_CTX = 9;
    constexpr size_t CHUNK_SIZE = 10;
    constexpr size_t RIGHT_CTX = 2;
    constexpr size_t FEATURE_DIM = 400;
    
    std::vector<std::vector<float>> caches(NUM_CACHE_LAYERS);
    for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
        caches[i].resize(1 * PROJ_DIM * LEFT_CTX, 0.0f);
    }
    
    // 3. 流式处理循环
    std::vector<std::vector<float>> frame_buffer;  // 累积的帧
    std::vector<std::vector<float>> all_logits;    // 所有输出的logits
    std::vector<float> frame;
    bool is_first_chunk = true;
    size_t total_frames_received = 0;
    size_t chunk_count = 0;
    
    while (true) {
        // 从队列取一帧
        if (queue->pop(frame)) {
            frame_buffer.push_back(frame);
            total_frames_received++;
            
            // 判断是否可以进行推理
            bool should_infer = false;
            
            if (is_first_chunk && frame_buffer.size() >= CHUNK_SIZE) {
                // 第一次：累积到10帧
                should_infer = true;
            } else if (!is_first_chunk && frame_buffer.size() >= CHUNK_SIZE) {
                // 后续：累积到10帧（包含上次保留的2帧）
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
                
                // 准备right_context（需要peek队列中的下2帧）
                std::vector<float> right_context(RIGHT_CTX * FEATURE_DIM, 0.0f);
                // TODO: 实现peek功能或使用其他方法获取right_context
                // 暂时用零填充
                
                // 设置输入tensors
                // 1. 设置cache inputs
                // 2. 设置right_context
                // 3. 设置main input
                // ... (参考run_encoder.cc)
                
                // 运行推理
                // TfLiteStatus status = interpreter.Invoke();
                
                // 获取输出logits
                // TfLiteTensor* output_tensor = interpreter.output(logits_output_idx);
                // float* output_data = tflite::GetTensorData<float>(output_tensor);
                // size_t output_dim = output_tensor->dims->data[2];
                
                // 保存输出
                size_t frames_to_save = is_first_chunk ? CHUNK_SIZE : (CHUNK_SIZE - 2);
                size_t start_frame = is_first_chunk ? 0 : 2;
                
                // for (size_t i = start_frame; i < CHUNK_SIZE; i++) {
                //     std::vector<float> logit_frame(output_dim);
                //     memcpy(logit_frame.data(),
                //            output_data + i * output_dim,
                //            output_dim * sizeof(float));
                //     all_logits.push_back(logit_frame);
                // }
                
                printf("[Consumer] Saved %zu logit frames (total: %zu)\n",
                       frames_to_save, all_logits.size());
                
                // 更新cache
                // for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
                //     TfLiteTensor* cache_out = interpreter.output(cache_output_indices[i]);
                //     float* cache_out_data = tflite::GetTensorData<float>(cache_out);
                //     memcpy(caches[i].data(), cache_out_data, caches[i].size() * sizeof(float));
                // }
                
                // 保留最后2帧
                std::vector<std::vector<float>> remaining(
                    frame_buffer.end() - 2, frame_buffer.end());
                frame_buffer = remaining;
                
                is_first_chunk = false;
            }
        } else {
            // 队列已结束
            break;
        }
    }
    
    // 4. 处理剩余的帧（flush）
    if (!frame_buffer.empty()) {
        printf("[Consumer] Flushing %zu remaining frames\n", frame_buffer.size());
        
        // Padding到10帧
        while (frame_buffer.size() < CHUNK_SIZE) {
            frame_buffer.push_back(frame_buffer.back());  // 重复最后一帧
        }
        
        // 进行最后一次推理
        // ... (类似上面的逻辑)
    }
    
    // 5. 保存结果
    *output_logits = all_logits;
    frame_count->store(total_frames_received);
    
    printf("[Consumer] Thread finished\n");
    printf("  Total frames received: %zu\n", total_frames_received);
    printf("  Total chunks processed: %zu\n", chunk_count);
    printf("  Total logit frames output: %zu\n", all_logits.size());
}
```

## 关键问题：Right Context

在流式处理中，获取right_context有两种方法：

### 方法1：Peek队列（推荐）

需要给`FeatureQueue`添加peek功能：

```cpp
class FeatureQueue {
public:
    // 新增：peek接下来的n帧（不移除）
    std::vector<std::vector<float>> peek(size_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::vector<float>> result;
        
        // 从队列中复制前n帧
        std::queue<std::vector<float>> temp_queue = queue_;
        for (size_t i = 0; i < n && !temp_queue.empty(); i++) {
            result.push_back(temp_queue.front());
            temp_queue.pop();
        }
        
        // 如果不足n帧，用零填充
        while (result.size() < n) {
            result.push_back(std::vector<float>(400, 0.0f));
        }
        
        return result;
    }
    
    // ... 其他方法
};
```

使用：
```cpp
// 获取right_context
auto rc_frames = queue->peek(RIGHT_CTX);
for (size_t i = 0; i < RIGHT_CTX; i++) {
    memcpy(right_context.data() + i * FEATURE_DIM,
           rc_frames[i].data(),
           FEATURE_DIM * sizeof(float));
}
```

### 方法2：延迟处理

保留更多帧在buffer中：

```cpp
// 保留最后4帧（2帧用于下次输入，2帧用于right_context）
if (frame_buffer.size() >= 12) {  // 10 + 2
    // 可以安全地获取right_context
    // ...
}
```

## 下一步

1. 实现`FeatureQueue::peek()`方法
2. 在`consumer_thread`中实现完整的encoder推理逻辑
3. 测试验证输出正确性

## 参考

- `run_encoder.cc` - 批处理版本的encoder推理
- `CHUNK_PROCESSING_EXPLAINED.md` - Chunk处理说明
