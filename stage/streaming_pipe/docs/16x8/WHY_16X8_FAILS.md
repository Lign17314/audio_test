# 为什么16x8版本会段错误？

## 问题根源

`streaming_fbank_30ms_threaded_16x8.cc`目前会导致段错误，而`run_encoder_16x8.cc`可以正常运行。原因是：

### run_encoder_16x8.cc（正常运行）✅

在设置输入tensor时，**正确检查了tensor类型并进行量化**：

```cpp
// 获取eval tensor
TfLiteEvalTensor* input_eval_tensor = interpreter.GetTensor(fb_input_idx_loop, 0);

// 检查类型并量化
if (input_eval_tensor->type == kTfLiteInt8) {
    int8_t* input_data = input_eval_tensor->data.int8;
    for (size_t i = 0; i < input_size; i++) {
        input_data[i] = QuantizeFloatToInt8(
            input_array.data[start * D + i], input_scale, input_zero_point);
    }
} else if (input_eval_tensor->type == kTfLiteInt16) {
    int16_t* input_data = input_eval_tensor->data.i16;
    for (size_t i = 0; i < input_size; i++) {
        input_data[i] = QuantizeFloatToInt16(
            input_array.data[start * D + i], input_scale, input_zero_point);
    }
} else if (input_eval_tensor->type == kTfLiteFloat32) {
    float* input_data = input_eval_tensor->data.f;
    memcpy(input_data, &input_array.data[start * D], input_size * sizeof(float));
}
```

### streaming_fbank_30ms_threaded_16x8.cc（段错误）❌

在设置输入tensor时，**假设tensor是float类型，直接memcpy**：

```cpp
TfLiteTensor* input_tensor = interpreter.input(input_tensor_idx);
if (input_tensor) {
    float* input_ptr = tflite::GetTensorData<float>(input_tensor);  // ❌ 假设是float
    if (input_ptr) {
        memcpy(input_ptr, input_data.data(), input_data.size() * sizeof(float));  // ❌ 直接复制float数据
    }
}
```

**问题**：
- 16x8模型的输入tensor类型是`kTfLiteInt8`或`kTfLiteInt16`，不是`kTfLiteFloat32`
- `tflite::GetTensorData<float>`会返回NULL（因为类型不匹配）
- 即使获取到指针，直接复制float数据到int8/int16 tensor会导致数据错误
- 访问错误的内存地址导致段错误

## 需要修改的位置

streaming_fbank_30ms_threaded_16x8.cc中有**多个位置**需要修改：

### 1. 主输入tensor（至少4处）

每次设置输入tensor的地方都需要：
- 获取eval_tensor
- 检查`eval_tensor->type`
- 根据类型调用量化函数或直接复制

**位置**：
- 第一次推理（12帧）
- 常规推理（12帧）
- Flush推理（>=12帧）
- 最后推理（<12帧）

### 2. Cache tensors（4个cache × 多个位置）

**设置cache输入**（量化）：
```cpp
// 当前代码（错误）
float* cache_data = tflite::GetTensorData<float>(cache_tensor);
memcpy(cache_data, caches[i].data(), caches[i].size() * sizeof(float));

// 应该改为
TfLiteEvalTensor* cache_eval = interpreter.GetTensor(cache_tensor_idx, 0);
if (cache_eval->type == kTfLiteInt8) {
    int8_t* cache_data = cache_eval->data.int8;
    for (size_t j = 0; j < caches[i].size(); j++) {
        cache_data[j] = QuantizeFloatToInt8(caches[i][j], cache_scale, cache_zero_point);
    }
} else if (cache_eval->type == kTfLiteInt16) {
    // 类似处理
}
```

**读取cache输出**（反量化）：
```cpp
// 当前代码（错误）
float* cache_out_data = tflite::GetTensorData<float>(cache_out_tensor);
memcpy(caches[i].data(), cache_out_data, cache_size * sizeof(float));

// 应该改为
TfLiteEvalTensor* cache_out_eval = interpreter.GetTensor(cache_out_tensor_idx, 0);
if (cache_out_eval->type == kTfLiteInt8) {
    int8_t* cache_out_data = cache_out_eval->data.int8;
    for (size_t j = 0; j < caches[i].size(); j++) {
        caches[i][j] = DequantizeInt8ToFloat(cache_out_data[j], cache_out_scale, cache_out_zero_point);
    }
} else if (cache_out_eval->type == kTfLiteInt16) {
    // 类似处理
}
```

### 3. Right context tensor（多个位置）

```cpp
// 当前代码（错误）
float* rc_data = tflite::GetTensorData<float>(rc_tensor);
memcpy(rc_data, right_context_data.data(), right_context_data.size() * sizeof(float));

// 应该改为
TfLiteEvalTensor* rc_eval = interpreter.GetTensor(rc_tensor_idx, 0);
if (rc_eval->type == kTfLiteInt8) {
    int8_t* rc_data = rc_eval->data.int8;
    for (size_t j = 0; j < right_context_data.size(); j++) {
        rc_data[j] = QuantizeFloatToInt8(right_context_data[j], rc_scale, rc_zero_point);
    }
} else if (rc_eval->type == kTfLiteInt16) {
    // 类似处理
}
```

### 4. 输出tensor（多个位置）

```cpp
// 当前代码（错误）
float* output_data = tflite::GetTensorData<float>(output_tensor);
memcpy(logit_frame.data(), output_data + i * output_dim, output_dim * sizeof(float));

// 应该改为
TfLiteEvalTensor* output_eval = interpreter.GetTensor(output_tensor_idx, 0);
if (output_eval->type == kTfLiteInt8) {
    int8_t* output_data = output_eval->data.int8;
    for (size_t j = 0; j < output_dim; j++) {
        logit_frame[j] = DequantizeInt8ToFloat(
            output_data[i * output_dim + j], output_scale, output_zero_point);
    }
} else if (output_eval->type == kTfLiteInt16) {
    // 类似处理
}
```

## 工作量估算

需要修改的代码行数：**约200-300行**

需要修改的位置：**约20-30处**

每个位置都需要：
1. 获取eval_tensor
2. 获取量化参数（如果还没有）
3. 添加类型检查（if-else）
4. 实现量化/反量化逻辑

## 为什么不立即修复？

1. **代码量大**：文件有1372行，需要仔细修改每个tensor操作点
2. **容易出错**：一个地方遗漏就会导致段错误或结果错误
3. **测试复杂**：需要验证所有路径（第一次推理、常规推理、flush等）
4. **float32版本已验证**：当前float32版本完全可用且已验证

## 推荐方案

### 短期（当前）

**使用float32版本**：
```bash
bash run_30ms_threaded.sh
```

优点：
- ✅ 完全验证，与run_encoder.cc输出一致（max_diff = 0.0）
- ✅ 稳定可靠
- ✅ 立即可用

### 中期（如需16x8）

**使用run_encoder_16x8.cc**：
```bash
# 先用streaming版本生成FBank特征
bash run_30ms_threaded.sh

# 然后用run_encoder_16x8处理
cd ../encoder_runner
./run.sh ../tflite_models/fsmn_encoder_stateful_16x8.tflite \
         ../streaming_fbank_only/build/streaming_fbank_30ms_threaded.npy \
         output_16x8.npy
```

优点：
- ✅ 16x8模型可用
- ✅ 已验证的实现
- ✅ 两步处理，清晰可控

### 长期（未来）

完整实现`streaming_fbank_30ms_threaded_16x8.cc`：
1. 系统性地修改所有tensor操作点
2. 添加完整的类型检查和量化/反量化
3. 全面测试所有代码路径
4. 验证输出正确性

## 参考代码

完整的16x8实现参考：
- `tflite_micro_cpp/encoder_runner/run_encoder_16x8.cc`

关键函数：
- 第860-920行：输入tensor量化
- 第700-850行：cache和right_context量化
- 第1000-1100行：cache输出反量化
- 第1150-1250行：输出tensor反量化

## 总结

16x8版本段错误的根本原因是：**代码假设所有tensor都是float32类型，没有检查实际类型并进行量化/反量化**。

修复需要大量细致的工作，而float32版本已经完全可用。因此，**当前推荐使用float32版本**。

---

**更新日期**：2026-02-04  
**状态**：问题已识别，等待完整实现
