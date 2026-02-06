# INT16 优化实现修改清单

## ✅ 已完成的文件

### 新建文件（fbank_extractor_int16/）

- [x] `IMPLEMENTATION_GUIDE.md` - 实现指南
- [x] `quantization_utils.h` - 量化工具函数
- [x] `feature_queue_int16.h` - INT16 队列
- [x] `streaming_fbank_extractor_int16.h` - INT16 FBank 提取器头文件
- [x] `streaming_fbank_extractor_int16.cc` - INT16 FBank 提取器实现
- [x] `MODIFICATION_CHECKLIST.md` - 本文件

## 📋 待修改的文件

### 1. 主程序文件

**文件**: `tflite_micro_cpp/streaming_fbank_only/src/streaming_fbank_30ms_threaded_16x8_optimized.cc`

**操作**: 复制 `streaming_fbank_30ms_threaded_16x8.cc` 并修改

**关键修改点**:

```cpp
// ========== 修改 1: 包含新头文件 ==========
#include "../../fbank_extractor_int16/feature_queue_int16.h"
#include "../../fbank_extractor_int16/streaming_fbank_extractor_int16.h"
#include "../../fbank_extractor_int16/quantization_utils.h"

// ========== 修改 2: 量化参数结构 ==========
struct EncoderQuantParams {
    float input_scale;
    int32_t input_zero_point;
    float rc_scale;
    int32_t rc_zero_point;
    bool ready;
};

// ========== 修改 3: 消费者线程签名 ==========
void consumer_thread(
    FeatureQueueINT16* queue,                    // 改为 INT16 队列
    std::vector<std::vector<float>>* output_logits,
    std::atomic<size_t>* frame_count,
    const char* model_file,
    EncoderQuantParams* quant_params)            // 新增：量化参数输出

// ========== 修改 4: 消费者线程开始部分 ==========
// 在获取量化参数后，立即设置并通知
quant_params->input_scale = input_scale;
quant_params->input_zero_point = input_zero_point;
quant_params->rc_scale = rc_scale;
quant_params->rc_zero_point = rc_zero_point;
quant_params->ready = true;

// ========== 修改 5: 消费者处理循环 ==========
// 将所有 std::vector<float> 改为 std::vector<int16_t>
std::vector<std::vector<int16_t>> frame_buffer_int16;

while (true) {
    std::vector<int16_t> frame_int16;
    if (!queue->pop(frame_int16)) break;
    
    frame_buffer_int16.push_back(frame_int16);
    
    if (frame_buffer_int16.size() >= 12) {
        // 准备输入（已经是 INT16，无需量化）
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
        
        // 直接复制到 tensor（移除 SET_MAIN_INPUT 宏）
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
        
        // Cache 输入仍然需要量化（因为是 float32 存储）
        for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
            SET_CACHE_INPUT(i);
        }
        
        // 推理
        interpreter.Invoke();
        
        // 读取输出（不变）
        // ...
    }
}

// ========== 修改 6: 主函数 ==========
int main(int argc, char* argv[]) {
    // ... 加载音频 ...
    
    // 创建 INT16 队列
    FeatureQueueINT16 feature_queue_int16;
    
    // 量化参数（由消费者线程填充）
    EncoderQuantParams quant_params;
    quant_params.ready = false;
    
    // 启动消费者线程
    std::thread consumer(consumer_thread, 
                        &feature_queue_int16,
                        &consumer_output_logits,
                        &consumer_frame_count,
                        model_file,
                        &quant_params);
    
    // 等待量化参数准备好
    printf("Waiting for quantization parameters...\n");
    while (!quant_params.ready) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    printf("Quantization parameters ready:\n");
    printf("  Input: scale=%.6f, zero_point=%d\n", 
           quant_params.input_scale, quant_params.input_zero_point);
    
    // 创建 INT16 FBank 提取器
    StreamingFBankExtractorINT16 fbank_extractor_int16(
        fbank_config,
        quant_params.input_scale,
        quant_params.input_zero_point
    );
    
    // 生产者处理（输出 INT16）
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
                total_frames_produced++;
            }
        }
    }
    
    // 刷新
    std::vector<std::vector<int16_t>> final_frames_int16;
    int final_count = fbank_extractor_int16.flush(final_frames_int16);
    for (const auto& frame : final_frames_int16) {
        feature_queue_int16.push(frame);
        total_frames_produced++;
    }
    
    // 通知结束
    feature_queue_int16.finish();
    
    // 等待消费者
    consumer.join();
    
    // ... 保存结果 ...
}
```

### 2. CMakeLists.txt

**文件**: `tflite_micro_cpp/streaming_fbank_only/CMakeLists.txt`

**操作**: 添加新的编译目标

```cmake
# INT16 优化版本
add_executable(streaming_fbank_30ms_threaded_16x8_optimized
    src/streaming_fbank_30ms_threaded_16x8_optimized.cc
    src/streaming_fbank_extractor.cc
    ../fbank_extractor_int16/streaming_fbank_extractor_int16.cc
    ../fbank_extractor/fbank_extractor.cc
    ../fbank_extractor/wav_reader.cc
)

target_include_directories(streaming_fbank_30ms_threaded_16x8_optimized PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../fbank_extractor
    ${CMAKE_CURRENT_SOURCE_DIR}/../fbank_extractor_int16
    ${TFLITE_MICRO_INCLUDE_DIRS}
)

target_link_libraries(streaming_fbank_30ms_threaded_16x8_optimized
    ${TFLITE_MICRO_LIB}
    pthread
)

if(USE_CMVN_HEADER)
    target_compile_definitions(streaming_fbank_30ms_threaded_16x8_optimized PRIVATE USE_CMVN_HEADER)
endif()
```

### 3. 编译脚本

**文件**: `tflite_micro_cpp/streaming_fbank_only/run_30ms_threaded_16x8_optimized.sh`

**操作**: 新建

```bash
#!/bin/bash

echo "Building streaming FBank 30ms threaded (16x8 optimized)..."

# 创建 build 目录
mkdir -p build
cd build

# CMake 配置
cmake .. -DUSE_CMVN_HEADER=ON

# 编译
make streaming_fbank_30ms_threaded_16x8_optimized -j$(nproc)

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Running..."
    ./streaming_fbank_30ms_threaded_16x8_optimized
else
    echo "Build failed!"
    exit 1
fi
```

## 📊 修改统计

```
新建文件: 7 个
  - 头文件: 3 个
  - 源文件: 2 个
  - 文档: 2 个

修改文件: 2 个
  - 主程序: 1 个（复制并修改）
  - CMakeLists.txt: 1 个（添加）

新建脚本: 1 个

总代码行数: ~1900 行
  - 新代码: ~400 行
  - 修改代码: ~1500 行（基于现有代码）
```

## 🎯 预期效果

### 内存占用

```
当前版本:
  队列: 19.2 KB (float32)
  缓冲: 19.2 KB (float32)
  总计: 38.4 KB

优化版本:
  队列: 9.6 KB (INT16)
  缓冲: 9.6 KB (INT16)
  总计: 19.2 KB

节省: 50%
```

### 处理速度

```
生产者: +5% 开销（量化）
消费者: -10% 开销（无需量化）
总体: +5-10% 提升
```

### 精度

```
无变化（量化时机不影响精度）
识别率: 97.46% (保持不变)
```

## ✅ 验证步骤

1. **编译测试**
   ```bash
   cd tflite_micro_cpp/streaming_fbank_only
   bash run_30ms_threaded_16x8_optimized.sh
   ```

2. **功能测试**
   ```bash
   # 检查输出形状
   python3 -c "import numpy as np; x=np.load('build/streaming_fbank_30ms_threaded_16x8_optimized_logits.npy'); print(x.shape)"
   ```

3. **精度验证**
   ```bash
   python3 ../../verify_encoder_output.py \
       ../../features/test_xiaoyun_fbank.npy \
       build/streaming_fbank_30ms_threaded_16x8_optimized_logits.npy \
       /root/volume/ctc/train/funasr_test/test_xiaoyun.wav
   ```

4. **性能对比**
   ```bash
   # 对比优化前后的内存和速度
   time ./build/streaming_fbank_30ms_threaded_16x8
   time ./build/streaming_fbank_30ms_threaded_16x8_optimized
   ```

## 📝 下一步

1. [ ] 复制并修改主程序文件
2. [ ] 修改 CMakeLists.txt
3. [ ] 创建编译脚本
4. [ ] 编译测试
5. [ ] 功能验证
6. [ ] 性能测试
7. [ ] 文档更新

准备好开始实现了吗？
