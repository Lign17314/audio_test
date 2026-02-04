# 16x8 命名解释

## 问题

**Q: "16x8" 中的 INT8 在哪里？输入输出都是 INT16？**

**A: 你的观察完全正确！** INT8 用在模型**内部**（权重），而 INT16 用在**输入/输出**（激活）。

## 16x8 量化详解

### 命名含义

**"16x8"** 表示：
- **16**: 激活(Activations)使用 **INT16** (16-bit)
- **8**: 权重(Weights)使用 **INT8** (8-bit)

这是一种**混合精度量化**策略。

### 数据类型分布

```
┌─────────────────────────────────────────────────────────┐
│                    用户可见部分                          │
│                                                          │
│  输入 Tensors  ──────────────────────→  INT16           │
│  输出 Tensors  ──────────────────────→  INT16           │
│  中间激活      ──────────────────────→  INT16           │
│                                                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    模型内部（不可见）                     │
│                                                          │
│  权重(Weights)  ─────────────────────→  INT8            │
│  偏置(Biases)   ─────────────────────→  INT32           │
│  内部计算       ─────────────────────→  INT8/INT16      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 详细说明

### INT16 - 激活（用户可见）

**使用位置**：
- ✅ 所有输入 tensors（主输入、right_context、cache）
- ✅ 所有输出 tensors（logits、cache 输出）
- ✅ 层与层之间的中间结果

**为什么使用 INT16**：
- 更高的精度范围：[-32768, 32767]
- 减少量化误差累积
- 保持较好的模型精度

**示例**：
```cpp
// 输入：float32 → INT16
int16_t quantized = QuantizeFloatToInt16(float_value, scale, zero_point);

// 输出：INT16 → float32
float dequantized = DequantizeInt16ToFloat(int16_value, scale, zero_point);
```

### INT8 - 权重（模型内部）

**使用位置**：
- ✅ 卷积层权重
- ✅ 全连接层权重
- ✅ 其他可学习参数

**为什么使用 INT8**：
- 减少模型大小（相比 INT16 减少 50%）
- 减少内存占用
- 加速计算（在支持 INT8 的硬件上）

**对用户透明**：
- 权重在模型文件中已经量化为 INT8
- TFLite Micro 内部自动处理
- 用户无需关心权重的量化

## 实际数据流

### 推理过程

```
用户输入 (float32)
    ↓
【量化】float32 → INT16
    ↓
输入 Tensor (INT16)
    ↓
┌─────────────────────────────────────┐
│      TFLite Micro 内部处理           │
│                                     │
│  INT16 激活 × INT8 权重 = INT32     │
│         ↓                           │
│  INT32 → INT16 (重新量化)           │
│         ↓                           │
│  下一层 (INT16 激活)                │
│                                     │
└─────────────────────────────────────┘
    ↓
输出 Tensor (INT16)
    ↓
【反量化】INT16 → float32
    ↓
用户输出 (float32)
```

### 内部计算示例

```
层 N:
  输入激活: INT16 [1, 10, 400]
  权重:     INT8  [400, 128]
  ↓
  矩阵乘法: INT16 × INT8 = INT32 (累加器)
  ↓
  重新量化: INT32 → INT16
  ↓
  输出激活: INT16 [1, 10, 128]
```

## 为什么选择 16x8？

### 优势

| 方面 | 16x8 | 纯 INT8 | 纯 INT16 |
|------|------|---------|----------|
| **模型大小** | ⭐⭐⭐ 小 | ⭐⭐⭐⭐ 最小 | ⭐⭐ 大 |
| **精度** | ⭐⭐⭐⭐ 高 | ⭐⭐ 中 | ⭐⭐⭐⭐⭐ 最高 |
| **速度** | ⭐⭐⭐ 快 | ⭐⭐⭐⭐ 最快 | ⭐⭐ 慢 |
| **内存占用** | ⭐⭐⭐ 中 | ⭐⭐⭐⭐ 最小 | ⭐⭐ 大 |

### 平衡点

16x8 是**精度**和**效率**的最佳平衡：
- ✅ 模型大小减少 ~50%（相比 float32）
- ✅ 精度损失很小（97.46% vs 97.93%，仅 0.47%）
- ✅ 内存占用减少
- ✅ 在支持的硬件上速度更快

## 代码中的体现

### 宏定义支持两种类型

代码中的宏同时支持 INT8 和 INT16，但实际运行时只会使用 INT16：

```cpp
// 宏定义（支持多种类型）
#define SET_MAIN_INPUT(input_data_vec) \
    do { \
        if (eval_tensor->type == kTfLiteInt8) {          // 不会执行
            // INT8 处理
        } else if (eval_tensor->type == kTfLiteInt16) {  // ✓ 实际执行
            // INT16 处理
        } else if (eval_tensor->type == kTfLiteFloat32) { // 不会执行
            // FLOAT32 处理
        } \
    } while(0)
```

**为什么保留 INT8 代码**：
- 代码通用性（可能有其他模型使用 INT8）
- 未来兼容性
- 调试和测试

### 实际运行时

```cpp
// 运行时检测（16x8 模型）
TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(tensor_idx, 0);

// 实际类型
eval_tensor->type == kTfLiteInt16  // ✓ TRUE（输入/输出）

// 不会是这些
eval_tensor->type == kTfLiteInt8   // ✗ FALSE
eval_tensor->type == kTfLiteFloat32 // ✗ FALSE
```

## 验证

### 模型文件分析

```bash
模型文件: fsmn_encoder_stateful_16x8.tflite
大小: 951,880 bytes (~930 KB)

输入 tensors: 6 个，全部 INT16
  - input:         [1, 10, 400]  INT16
  - right_context: [1, 2, 400]   INT16
  - cache_0:       [1, 128, 9]   INT16
  - cache_1:       [1, 128, 9]   INT16
  - cache_2:       [1, 128, 9]   INT16
  - cache_3:       [1, 128, 9]   INT16

输出 tensors: 5 个，全部 INT16
  - logits:        [1, 10, 2599] INT16
  - cache_0_out:   [1, 128, 9]   INT16
  - cache_1_out:   [1, 128, 9]   INT16
  - cache_2_out:   [1, 128, 9]   INT16
  - cache_3_out:   [1, 128, 9]   INT16

内部权重: INT8（不可见）
```

### 对比其他模型

| 模型 | 输入/输出 | 权重 | 大小 |
|------|-----------|------|------|
| **float32** | FLOAT32 | FLOAT32 | ~2 MB |
| **16x8** | INT16 | INT8 | ~930 KB |
| **int8** | INT8 | INT8 | ~500 KB |

## 总结

### 关键点

1. **"16x8" 中的 8 指的是权重**
   - INT8 权重存储在模型文件中
   - 对用户完全透明
   - TFLite Micro 内部自动处理

2. **"16x8" 中的 16 指的是激活**
   - INT16 用于所有输入/输出
   - 用户需要处理 float32 ↔ INT16 转换
   - 代码中的宏自动处理

3. **用户只需关心 INT16**
   - 输入：float32 → INT16（量化）
   - 输出：INT16 → float32（反量化）
   - INT8 完全不可见

### 命名建议

更准确的命名应该是：
- **"INT16 激活 + INT8 权重"**
- 或 **"16-bit 激活量化"**

但 **"16x8"** 是业界通用的简写。

## 相关文档

- [16X8_INPUT_SPECIFICATION.md](16X8_INPUT_SPECIFICATION.md) - 输入规格详解
- [16X8_WORKFLOW.md](16X8_WORKFLOW.md) - 完整流程说明
- [VERIFICATION_COMPARISON.md](../verification/VERIFICATION_COMPARISON.md) - 精度对比

---

**结论**: 你的观察完全正确！16x8 模型的输入/输出确实都是 INT16，INT8 只用在模型内部的权重，对用户是透明的。
