# 实现16x8版本的完整指南

由于16x8版本需要修改约20-30处代码，手动逐个修改容易出错。这里提供完整的实现策略。

## 策略：使用辅助宏简化代码

在consumer_thread函数开始处添加以下宏定义：

```cpp
// 宏：设置cache输入（支持量化）
#define SET_CACHE_INPUT(cache_idx) \
    do { \
        TfLiteTensor* cache_tensor = interpreter.input(cache_input_indices[cache_idx]); \
        if (cache_tensor) { \
            int tensor_idx = subgraph->inputs()->Get(cache_input_indices[cache_idx]); \
            TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0); \
            if (eval_tensor && eval_tensor->data.raw) { \
                if (eval_tensor->type == kTfLiteInt8) { \
                    int8_t* data = eval_tensor->data.int8; \
                    for (size_t j = 0; j < caches[cache_idx].size(); j++) { \
                        data[j] = QuantizeFloatToInt8(caches[cache_idx][j], cache_in_scales[cache_idx], cache_in_zero_points[cache_idx]); \
                    } \
                } else if (eval_tensor->type == kTfLiteInt16) { \
                    int16_t* data = eval_tensor->data.i16; \
                    for (size_t j = 0; j < caches[cache_idx].size(); j++) { \
                        data[j] = QuantizeFloatToInt16(caches[cache_idx][j], cache_in_scales[cache_idx], cache_in_zero_points[cache_idx]); \
                    } \
                } else if (eval_tensor->type == kTfLiteFloat32) { \
                    float* data = eval_tensor->data.f; \
                    memcpy(data, caches[cache_idx].data(), caches[cache_idx].size() * sizeof(float)); \
                } \
            } \
        } \
    } while(0)

// 宏：读取cache输出（支持反量化）
#define READ_CACHE_OUTPUT(cache_idx) \
    do { \
        TfLiteTensor* cache_out_tensor = interpreter.output(cache_output_indices[cache_idx]); \
        if (cache_out_tensor) { \
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
        } \
    } while(0)

// 宏：设置主输入
#define SET_MAIN_INPUT(input_data_vec) \
    do { \
        int tensor_idx = subgraph->inputs()->Get(input_tensor_idx); \
        TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0); \
        if (eval_tensor && eval_tensor->data.raw) { \
            if (eval_tensor->type == kTfLiteInt8) { \
                int8_t* data = eval_tensor->data.int8; \
                for (size_t i = 0; i < input_data_vec.size(); i++) { \
                    data[i] = QuantizeFloatToInt8(input_data_vec[i], input_scale, input_zero_point); \
                } \
            } else if (eval_tensor->type == kTfLiteInt16) { \
                int16_t* data = eval_tensor->data.i16; \
                for (size_t i = 0; i < input_data_vec.size(); i++) { \
                    data[i] = QuantizeFloatToInt16(input_data_vec[i], input_scale, input_zero_point); \
                } \
            } else if (eval_tensor->type == kTfLiteFloat32) { \
                float* data = eval_tensor->data.f; \
                memcpy(data, input_data_vec.data(), input_data_vec.size() * sizeof(float)); \
            } \
        } \
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

// 宏：读取输出logits
#define READ_OUTPUT_LOGITS(output_vec, frame_idx) \
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
```

## 使用宏替换所有tensor操作

然后在代码中，将所有tensor操作替换为宏调用：

### 1. 设置cache inputs（4处）
```cpp
// 原代码
for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
    TfLiteTensor* cache_tensor = interpreter.input(cache_input_indices[i]);
    // ... 复杂的类型检查和memcpy
}

// 新代码
for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
    SET_CACHE_INPUT(i);
}
```

### 2. 读取cache outputs（4处）
```cpp
// 原代码
for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
    TfLiteTensor* cache_out_tensor = interpreter.output(cache_output_indices[i]);
    // ... 复杂的类型检查和memcpy
}

// 新代码
for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {
    READ_CACHE_OUTPUT(i);
}
```

### 3. 设置主输入（4处）
```cpp
// 原代码
TfLiteTensor* input_tensor = interpreter.input(input_tensor_idx);
// ... 复杂的类型检查和memcpy

// 新代码
SET_MAIN_INPUT(input_data);
```

### 4. 设置right_context（4处）
```cpp
// 原代码
if (right_context_idx >= 0) {
    TfLiteTensor* rc_tensor = interpreter.input(right_context_idx);
    // ... 复杂的类型检查和memcpy
}

// 新代码
SET_RIGHT_CONTEXT(right_context_data);
```

### 5. 读取输出logits（4处）
```cpp
// 原代码
TfLiteTensor* output_tensor = interpreter.output(logits_output_idx);
float* output_data = tflite::GetTensorData<float>(output_tensor);
for (size_t i = 0; i < CHUNK_SIZE; i++) {
    std::vector<float> logit_frame(output_dim);
    memcpy(logit_frame.data(), output_data + i * output_dim, output_dim * sizeof(float));
    all_logits.push_back(logit_frame);
}

// 新代码
for (size_t i = 0; i < CHUNK_SIZE; i++) {
    std::vector<float> logit_frame(output_dim);
    READ_OUTPUT_LOGITS(logit_frame, i);
    all_logits.push_back(logit_frame);
}
```

## 需要修改的具体位置

在`streaming_fbank_30ms_threaded_16x8.cc`中搜索以下模式并替换：

1. **搜索**: `for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {` 后面跟着 `TfLiteTensor* cache_tensor = interpreter.input`
   **替换**: 使用 `SET_CACHE_INPUT(i);`
   **数量**: 约4处

2. **搜索**: `for (size_t i = 0; i < NUM_CACHE_LAYERS; i++) {` 后面跟着 `TfLiteTensor* cache_out_tensor = interpreter.output`
   **替换**: 使用 `READ_CACHE_OUTPUT(i);`
   **数量**: 约4处

3. **搜索**: `TfLiteTensor* input_tensor = interpreter.input(input_tensor_idx);` 后面的整个设置逻辑
   **替换**: 使用 `SET_MAIN_INPUT(input_data);`
   **数量**: 约4处

4. **搜索**: `if (right_context_idx >= 0) { TfLiteTensor* rc_tensor = interpreter.input(right_context_idx);`
   **替换**: 使用 `SET_RIGHT_CONTEXT(right_context_data);`
   **数量**: 约4处

5. **搜索**: 读取输出的循环
   **替换**: 使用 `READ_OUTPUT_LOGITS(logit_frame, i);`
   **数量**: 约4处

## 自动化脚本

由于手动修改容易出错，建议使用以下Python脚本自动完成：

```python
#!/usr/bin/env python3
import re

# 读取文件
with open('src/streaming_fbank_30ms_threaded_16x8.cc', 'r') as f:
    content = f.read()

# TODO: 添加具体的替换逻辑
# 这需要仔细分析每个位置的上下文

# 写回文件
with open('src/streaming_fbank_30ms_threaded_16x8.cc', 'w') as f:
    f.write(content)
```

## 测试验证

修改完成后，需要：

1. 编译：`bash run_30ms_threaded_16x8.sh`
2. 运行：检查是否有段错误
3. 验证输出：与run_encoder_16x8.cc对比

## 预期结果

完成后，16x8版本应该：
- ✅ 不再段错误
- ✅ 正常运行完成
- ✅ 输出logits文件
- ✅ 与run_encoder_16x8.cc输出一致

---

**注意**: 由于修改点多且复杂，建议分步骤进行：
1. 先添加宏定义
2. 逐个位置测试替换
3. 每次修改后编译测试
4. 确保没有引入新错误
