# 16x8 Encoder 输入规格说明

## 概述

16x8 量化版本的 Encoder 使用 **Stateful 模型**，包含多个输入 tensor，用于流式推理。本文档详细说明每个输入的规格、量化参数和数据流。

## 输入 Tensor 列表

16x8 Stateful Encoder 模型有 **6 个输入 tensor**：

| 索引 | 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|------|
| 0 | **主输入** (input) | `[1, 10, 400]` | INT16 | 10 帧 FBank 特征 |
| 1 | **Right Context** | `[1, 2, 400]` | INT16 | 2 帧右上下文 |
| 2 | **Cache 0** (in_cache_0) | `[1, 128, 9]` | INT16 | 第 0 层 cache 状态 |
| 3 | **Cache 1** (in_cache_1) | `[1, 128, 9]` | INT16 | 第 1 层 cache 状态 |
| 4 | **Cache 2** (in_cache_2) | `[1, 128, 9]` | INT16 | 第 2 层 cache 状态 |
| 5 | **Cache 3** (in_cache_3) | `[1, 128, 9]` | INT16 | 第 3 层 cache 状态 |

## 详细规格

### 1. 主输入 (Main Input)

**Tensor 信息**：
- **名称**: `input` 或包含 "input" 且不包含 "cache"/"right" 的 tensor
- **形状**: `[batch=1, frames=10, features=400]`
- **数据类型**: `INT16` (kTfLiteInt16)
- **量化参数**:
  - Scale: 从模型获取（例如：0.003906）
  - Zero point: 从模型获取（通常为 0）

**数据来源**：
- 10 帧 FBank 特征（每帧 400 维）
- 400 维 = 80 (Mel bins) × 5 (LFR_m)
- 经过 CMVN 归一化的 float32 数据

**量化过程**：
```cpp
// Float32 → INT16
int16_t quantized_value = QuantizeFloatToInt16(float_value, scale, zero_point);

// 公式：
// quantized = round(float_value / scale + zero_point)
// 然后 clip 到 [-32768, 32767]
```

**示例代码**：
```cpp
// 准备 10 帧输入数据
std::vector<float> input_data(10 * 400);  // 4000 个 float32
for (size_t i = 0; i < 10; i++) {
    memcpy(input_data.data() + i * 400,
           frame_buffer[i].data(),
           400 * sizeof(float));
}

// 使用宏自动量化并设置
SET_MAIN_INPUT(input_data);
```

### 2. Right Context

**Tensor 信息**：
- **名称**: `right_context` 或包含 "right_context" 的 tensor
- **形状**: `[batch=1, frames=2, features=400]`
- **数据类型**: `INT16` (kTfLiteInt16)
- **量化参数**:
  - Scale: 从模型获取（通常与主输入相同）
  - Zero point: 从模型获取（通常为 0）

**数据来源**：
- 2 帧未来的 FBank 特征
- 用于提供右上下文信息
- 如果没有足够的未来帧，用零填充

**量化过程**：
```cpp
// 准备 2 帧 right_context
std::vector<float> right_context_data(2 * 400);  // 800 个 float32
for (size_t i = 0; i < 2; i++) {
    memcpy(right_context_data.data() + i * 400,
           frame_buffer[10 + i].data(),  // 从第 11、12 帧获取
           400 * sizeof(float));
}

// 使用宏自动量化并设置
SET_RIGHT_CONTEXT(right_context_data);
```

### 3. Cache Inputs (4 层)

**Tensor 信息**：
- **名称**: `in_cache_0`, `in_cache_1`, `in_cache_2`, `in_cache_3`
- **形状**: `[batch=1, proj_dim=128, left_ctx=9]`
- **数据类型**: `INT16` (kTfLiteInt16)
- **量化参数**:
  - Scale: 每层独立（从模型获取）
  - Zero point: 每层独立（从模型获取）

**数据来源**：
- 上一次推理的 cache 输出
- 第一次推理时初始化为全零
- 用于保存历史上下文信息

**维度说明**：
- `proj_dim=128`: 投影维度（特征压缩后的维度）
- `left_ctx=9`: 左上下文长度（保存过去 9 帧的信息）

**量化过程**：
```cpp
// 初始化 cache（第一次推理）
std::vector<std::vector<float>> caches(4);
for (size_t i = 0; i < 4; i++) {
    caches[i].resize(1 * 128 * 9, 0.0f);  // 1152 个 float32
}

// 设置 cache 输入（自动量化）
for (size_t i = 0; i < 4; i++) {
    SET_CACHE_INPUT(i);
}
```

## 输出 Tensor 列表

16x8 Stateful Encoder 模型有 **5 个输出 tensor**：

| 索引 | 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|------|
| 0 | **Logits** | `[1, 10, 2599]` | INT16 | 10 帧输出 logits |
| 1 | **Cache 0 Out** | `[1, 128, 9]` | INT16 | 第 0 层更新后的 cache |
| 2 | **Cache 1 Out** | `[1, 128, 9]` | INT16 | 第 1 层更新后的 cache |
| 3 | **Cache 2 Out** | `[1, 128, 9]` | INT16 | 第 2 层更新后的 cache |
| 4 | **Cache 3 Out** | `[1, 128, 9]` | INT16 | 第 3 层更新后的 cache |

### Logits 输出

**Tensor 信息**：
- **形状**: `[batch=1, frames=10, vocab=2599]`
- **数据类型**: `INT16` (kTfLiteInt16)
- **量化参数**:
  - Scale: 从模型获取
  - Zero point: 从模型获取

**反量化过程**：
```cpp
// INT16 → Float32
float dequantized_value = DequantizeInt16ToFloat(int16_value, scale, zero_point);

// 公式：
// float_value = (int16_value - zero_point) * scale
```

**读取示例**：
```cpp
// 读取 10 帧 logits
for (size_t i = 0; i < 10; i++) {
    std::vector<float> logit_frame(2599);
    READ_OUTPUT_LOGIT_FRAME(logit_frame, i);  // 自动反量化
    all_logits.push_back(logit_frame);
}
```

### Cache 输出

**更新过程**：
```cpp
// 读取并更新 cache（自动反量化）
for (size_t i = 0; i < 4; i++) {
    READ_CACHE_OUTPUT(i);  // 更新 caches[i]
}

// 下次推理时，这些 cache 会作为输入
```

## 完整的推理流程

### 第一次推理

```cpp
// 1. 准备输入数据（12 帧：10 输入 + 2 right_context）
std::vector<float> input_data(10 * 400);        // 主输入
std::vector<float> right_context_data(2 * 400); // Right context

// 2. 初始化 cache（全零）
std::vector<std::vector<float>> caches(4);
for (size_t i = 0; i < 4; i++) {
    caches[i].resize(1 * 128 * 9, 0.0f);
}

// 3. 设置输入 tensors（自动量化）
for (size_t i = 0; i < 4; i++) {
    SET_CACHE_INPUT(i);              // 量化 cache 输入
}
SET_RIGHT_CONTEXT(right_context_data); // 量化 right_context
SET_MAIN_INPUT(input_data);            // 量化主输入

// 4. 运行推理
interpreter.Invoke();

// 5. 读取输出（自动反量化）
for (size_t i = 0; i < 10; i++) {
    std::vector<float> logit_frame(2599);
    READ_OUTPUT_LOGIT_FRAME(logit_frame, i);
    all_logits.push_back(logit_frame);
}

// 6. 更新 cache（自动反量化）
for (size_t i = 0; i < 4; i++) {
    READ_CACHE_OUTPUT(i);  // caches[i] 被更新
}
```

### 后续推理

```cpp
// 1. 准备新的输入数据（使用上次的 cache）
std::vector<float> input_data(10 * 400);
std::vector<float> right_context_data(2 * 400);

// 2. 设置输入 tensors（cache 已经是上次的输出）
for (size_t i = 0; i < 4; i++) {
    SET_CACHE_INPUT(i);              // 使用上次更新的 cache
}
SET_RIGHT_CONTEXT(right_context_data);
SET_MAIN_INPUT(input_data);

// 3-6. 同第一次推理
```

## 量化参数获取

### 自动获取流程

```cpp
// 1. 尝试从 TfLiteTensor 获取
TfLiteTensor* tensor = interpreter.input(tensor_idx);
bool success = GetQuantizationParams(tensor, scale, zero_point);

// 2. 如果失败，从 flatbuffer 获取（fallback）
if (!success) {
    int fb_idx = subgraph->inputs()->Get(tensor_idx);
    TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(fb_idx, 0);
    const tflite::Tensor* fb_tensor = subgraph->tensors()->Get(fb_idx);
    const tflite::QuantizationParameters* quant = fb_tensor->quantization();
    
    if (quant && quant->scale() && quant->scale()->size() > 0) {
        scale = quant->scale()->Get(0);
        zero_point = quant->zero_point()->Get(0);
    }
}
```

### 量化参数示例

典型的 16x8 模型量化参数：

```
主输入 (input):
  Scale: 0.003906
  Zero point: 0
  Range: [-128.0, 127.996] (float32)

Right Context:
  Scale: 0.003906
  Zero point: 0
  Range: [-128.0, 127.996] (float32)

Cache Inputs (各层可能不同):
  Layer 0: Scale=0.015625, Zero point=0
  Layer 1: Scale=0.015625, Zero point=0
  Layer 2: Scale=0.015625, Zero point=0
  Layer 3: Scale=0.015625, Zero point=0

Logits Output:
  Scale: 0.125
  Zero point: 0
  Range: [-4096.0, 4095.875] (float32)
```

## 宏定义说明

为了简化量化操作，代码中定义了以下宏：

### SET_CACHE_INPUT(i)
- **功能**: 设置第 i 层 cache 输入
- **自动**: 检测类型（INT8/INT16/FLOAT32）并量化
- **输入**: `caches[i]` (float32 vector)
- **输出**: 写入 tensor 内存（量化后）

### SET_MAIN_INPUT(data)
- **功能**: 设置主输入
- **自动**: 检测类型并量化
- **输入**: `input_data` (float32 vector, 10×400)
- **输出**: 写入 tensor 内存（量化后）

### SET_RIGHT_CONTEXT(data)
- **功能**: 设置 right_context
- **自动**: 检测类型并量化
- **输入**: `right_context_data` (float32 vector, 2×400)
- **输出**: 写入 tensor 内存（量化后）

### READ_OUTPUT_LOGIT_FRAME(vec, idx)
- **功能**: 读取第 idx 帧的 logits
- **自动**: 检测类型并反量化
- **输入**: frame index (0-9)
- **输出**: `vec` (float32 vector, 2599 维)

### READ_CACHE_OUTPUT(i)
- **功能**: 读取第 i 层 cache 输出
- **自动**: 检测类型并反量化
- **输入**: cache layer index (0-3)
- **输出**: 更新 `caches[i]` (float32 vector)

## 数据流图

```
┌─────────────────────────────────────────────────────────────┐
│                    输入准备（Float32）                        │
│                                                               │
│  FBank 特征 (12 帧 × 400 维)                                  │
│       ├─ 前 10 帧 → input_data                               │
│       └─ 后 2 帧  → right_context_data                       │
│                                                               │
│  Cache 状态 (4 层 × 1×128×9)                                 │
│       └─ caches[0..3] (上次推理的输出或初始零)                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    量化（Float32 → INT16）                    │
│                                                               │
│  使用宏自动量化:                                              │
│    SET_CACHE_INPUT(0..3)                                     │
│    SET_RIGHT_CONTEXT(right_context_data)                     │
│    SET_MAIN_INPUT(input_data)                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    TFLite Micro 推理                          │
│                                                               │
│  interpreter.Invoke()                                        │
│                                                               │
│  输入: 6 个 INT16 tensors                                     │
│  输出: 5 个 INT16 tensors                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  反量化（INT16 → Float32）                    │
│                                                               │
│  使用宏自动反量化:                                            │
│    READ_OUTPUT_LOGIT_FRAME(logit_frame, 0..9)               │
│    READ_CACHE_OUTPUT(0..3)                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    输出（Float32）                            │
│                                                               │
│  Logits: 10 帧 × 2599 维                                     │
│  Cache: 4 层 × 1×128×9 (用于下次推理)                        │
└─────────────────────────────────────────────────────────────┘
```

## 内存占用

### 输入内存

```
主输入:        1 × 10 × 400 × 2 bytes = 8,000 bytes   (8 KB)
Right Context: 1 × 2  × 400 × 2 bytes = 1,600 bytes   (1.6 KB)
Cache 0:       1 × 128 × 9 × 2 bytes = 2,304 bytes   (2.3 KB)
Cache 1:       1 × 128 × 9 × 2 bytes = 2,304 bytes   (2.3 KB)
Cache 2:       1 × 128 × 9 × 2 bytes = 2,304 bytes   (2.3 KB)
Cache 3:       1 × 128 × 9 × 2 bytes = 2,304 bytes   (2.3 KB)
─────────────────────────────────────────────────────────────
总计:                                  18,816 bytes  (18.4 KB)
```

### 输出内存

```
Logits:        1 × 10 × 2599 × 2 bytes = 51,980 bytes (50.8 KB)
Cache 0 Out:   1 × 128 × 9 × 2 bytes  = 2,304 bytes  (2.3 KB)
Cache 1 Out:   1 × 128 × 9 × 2 bytes  = 2,304 bytes  (2.3 KB)
Cache 2 Out:   1 × 128 × 9 × 2 bytes  = 2,304 bytes  (2.3 KB)
Cache 3 Out:   1 × 128 × 9 × 2 bytes  = 2,304 bytes  (2.3 KB)
─────────────────────────────────────────────────────────────
总计:                                   61,196 bytes  (59.8 KB)
```

### Tensor Arena

```
总分配: 3 MB (3,145,728 bytes)
实际使用: ~417 KB (13.27%)
剩余: ~2.6 MB
```

## 常见问题

### Q1: 为什么需要 right_context？

A: Right context 提供未来 2 帧的信息，帮助模型更准确地预测当前帧的输出。这在语音识别中很常见，因为后续的音素可以帮助识别当前的音素。

### Q2: Cache 的作用是什么？

A: Cache 保存了历史上下文信息（过去 9 帧的压缩表示），使得模型可以利用长期依赖关系。每次推理后，cache 会被更新，并在下次推理时作为输入。

### Q3: 如果没有足够的 right_context 怎么办？

A: 用零填充。例如，在音频末尾，可能只有 1 帧或 0 帧未来数据，此时用零填充到 2 帧。

### Q4: 量化参数从哪里来？

A: 量化参数在模型训练时确定，并保存在 TFLite 模型文件中。代码会自动从模型中读取这些参数。

### Q5: 为什么使用 INT16 而不是 INT8？

A: 16x8 表示权重使用 INT8，激活使用 INT16。这种混合精度可以在保持较高精度的同时减少模型大小。

## 总结

16x8 Encoder 的输入包括：
- ✅ **主输入**: 10 帧 FBank 特征 (1×10×400, INT16)
- ✅ **Right Context**: 2 帧未来特征 (1×2×400, INT16)
- ✅ **4 层 Cache**: 历史上下文 (4×1×128×9, INT16)

所有输入都是 **INT16 量化**，通过宏定义自动处理量化/反量化，简化了代码并确保了正确性。

---

**相关文档**:
- [16X8_WORKFLOW.md](16X8_WORKFLOW.md) - 完整流程说明
- [16X8_FLOWCHART.md](16X8_FLOWCHART.md) - 流程图和架构图
- [VERIFICATION_COMPARISON.md](../verification/VERIFICATION_COMPARISON.md) - 验证结果
