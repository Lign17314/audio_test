# 16x8 Encoder 输入快速指南

## 简要回答

16x8 Encoder 有 **6 个输入**，全部是 **INT16 量化**类型：

| 输入 | 形状 | 说明 |
|------|------|------|
| 1️⃣ **主输入** | `[1, 10, 400]` | 10 帧 FBank 特征 |
| 2️⃣ **Right Context** | `[1, 2, 400]` | 2 帧未来特征 |
| 3️⃣ **Cache 0** | `[1, 128, 9]` | 第 0 层历史状态 |
| 4️⃣ **Cache 1** | `[1, 128, 9]` | 第 1 层历史状态 |
| 5️⃣ **Cache 2** | `[1, 128, 9]` | 第 2 层历史状态 |
| 6️⃣ **Cache 3** | `[1, 128, 9]` | 第 3 层历史状态 |

## 数据准备

### 1. 主输入（最重要）

```cpp
// 准备 10 帧 FBank 特征
std::vector<float> input_data(10 * 400);  // 4000 个 float32

// 从 frame_buffer 复制 10 帧
for (size_t i = 0; i < 10; i++) {
    memcpy(input_data.data() + i * 400,
           frame_buffer[i].data(),
           400 * sizeof(float));
}

// 自动量化并设置（宏会自动处理 float32 → INT16）
SET_MAIN_INPUT(input_data);
```

**数据来源**：
- FBank 特征提取器输出
- 每帧 400 维 = 80 (Mel bins) × 5 (LFR)
- 已经过 CMVN 归一化

### 2. Right Context

```cpp
// 准备 2 帧未来特征
std::vector<float> right_context_data(2 * 400);  // 800 个 float32

// 从 frame_buffer 复制第 11、12 帧
for (size_t i = 0; i < 2; i++) {
    memcpy(right_context_data.data() + i * 400,
           frame_buffer[10 + i].data(),
           400 * sizeof(float));
}

// 自动量化并设置
SET_RIGHT_CONTEXT(right_context_data);
```

**注意**：如果没有足够的未来帧，用零填充。

### 3. Cache（4 层）

```cpp
// 第一次推理：初始化为全零
std::vector<std::vector<float>> caches(4);
for (size_t i = 0; i < 4; i++) {
    caches[i].resize(1 * 128 * 9, 0.0f);  // 1152 个 float32
}

// 后续推理：使用上次的输出
// （caches 会在每次推理后自动更新）

// 设置所有 cache 输入
for (size_t i = 0; i < 4; i++) {
    SET_CACHE_INPUT(i);  // 自动量化
}
```

## 完整示例

```cpp
// ========== 准备输入 ==========

// 1. 主输入（10 帧）
std::vector<float> input_data(10 * 400);
for (size_t i = 0; i < 10; i++) {
    memcpy(input_data.data() + i * 400,
           frame_buffer[i].data(),
           400 * sizeof(float));
}

// 2. Right context（2 帧）
std::vector<float> right_context_data(2 * 400);
for (size_t i = 0; i < 2; i++) {
    memcpy(right_context_data.data() + i * 400,
           frame_buffer[10 + i].data(),
           400 * sizeof(float));
}

// 3. Cache（4 层，第一次为零，后续使用上次输出）
// caches[0..3] 已经准备好

// ========== 设置输入 tensors ==========

// 设置 cache（自动量化 float32 → INT16）
for (size_t i = 0; i < 4; i++) {
    SET_CACHE_INPUT(i);
}

// 设置 right_context（自动量化）
SET_RIGHT_CONTEXT(right_context_data);

// 设置主输入（自动量化）
SET_MAIN_INPUT(input_data);

// ========== 运行推理 ==========

interpreter.Invoke();

// ========== 读取输出 ==========

// 读取 logits（自动反量化 INT16 → float32）
for (size_t i = 0; i < 10; i++) {
    std::vector<float> logit_frame(2599);
    READ_OUTPUT_LOGIT_FRAME(logit_frame, i);
    all_logits.push_back(logit_frame);
}

// 更新 cache（自动反量化，用于下次推理）
for (size_t i = 0; i < 4; i++) {
    READ_CACHE_OUTPUT(i);
}
```

## 关键点

### ✅ 自动量化
所有宏（`SET_*` 和 `READ_*`）都会自动处理量化/反量化：
- 输入：float32 → INT16（自动）
- 输出：INT16 → float32（自动）

### ✅ 累积 12 帧
需要累积 **12 帧**才能推理：
- 10 帧作为主输入
- 2 帧作为 right_context

### ✅ Cache 状态管理
- **第一次推理**：cache 初始化为全零
- **后续推理**：使用上次推理的 cache 输出
- **自动更新**：每次推理后 `READ_CACHE_OUTPUT` 会更新 cache

### ✅ 量化参数
量化参数（scale, zero_point）从模型自动获取，无需手动设置。

## 数据流

```
FBank 特征队列
    ↓
累积 12 帧
    ├─ 前 10 帧 → 主输入 (float32)
    └─ 后 2 帧  → Right Context (float32)
    
Cache 状态 (float32)
    ↓
    
【自动量化 float32 → INT16】
    ↓
    
TFLite Micro 推理
    ↓
    
【自动反量化 INT16 → float32】
    ↓
    
输出 Logits (float32, 10×2599)
更新 Cache (float32, 4×1×128×9)
```

## 内存占用

```
输入总计:  ~18.4 KB (6 个 INT16 tensors)
输出总计:  ~59.8 KB (5 个 INT16 tensors)
Tensor Arena: 3 MB (实际使用 ~417 KB)
```

## 常见问题

**Q: 为什么需要 12 帧？**  
A: 10 帧用于主输入，2 帧用于 right_context（提供未来信息）。

**Q: Cache 是什么？**  
A: Cache 保存历史上下文（过去 9 帧的压缩表示），使模型能利用长期依赖。

**Q: 量化参数从哪里来？**  
A: 从 TFLite 模型文件自动读取，无需手动设置。

**Q: 如何处理音频末尾？**  
A: 如果没有足够的 right_context，用零填充。

## 更多信息

详细文档请参考：
- **[16X8_INPUT_SPECIFICATION.md](16X8_INPUT_SPECIFICATION.md)** - 完整的输入规格说明
- **[16X8_WORKFLOW.md](16X8_WORKFLOW.md)** - 详细的处理流程
- **[16X8_FLOWCHART.md](16X8_FLOWCHART.md)** - 流程图和架构图

---

**总结**: 16x8 Encoder 需要 6 个 INT16 输入（主输入 + right_context + 4 层 cache），所有量化操作都通过宏自动处理，开发者只需准备 float32 数据即可。
