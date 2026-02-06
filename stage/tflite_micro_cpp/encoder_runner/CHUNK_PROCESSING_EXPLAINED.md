# Chunk Processing in run_encoder.cc

## Question: 151帧如何处理？不能被10整除怎么办？

## Answer: Padding + Selective Output

### 处理策略

`run_encoder.cc` 使用了 **"输入填充 + 选择性输出"** 的策略来处理不能被chunk_size(10)整除的帧数。

### 详细流程

#### 1. 输入处理（151帧的例子）

```
总帧数 T = 151
chunk_size = 10

循环处理：
- Chunk 0: frames [0:10]   → 10帧 (完整)
- Chunk 1: frames [10:20]  → 10帧 (完整)
- Chunk 2: frames [20:30]  → 10帧 (完整)
- ...
- Chunk 14: frames [140:150] → 10帧 (完整)
- Chunk 15: frames [150:151] → 1帧 (不完整) ⚠️
```

#### 2. 最后一个Chunk的处理

对于最后一个chunk（frames 150:151），只有1帧：

```cpp
size_t start = 150;
size_t end = std::min(start + chunk_size, T);  // min(160, 151) = 151
size_t current_chunk_size = end - start;        // 151 - 150 = 1

// 复制实际数据（1帧）
memcpy(input_data, &input_array.data[start * D], 
       current_chunk_size * D * sizeof(float));

// 填充剩余部分（9帧）用零
if (current_chunk_size < chunk_size) {
    memset(input_data + current_chunk_size * D, 0, 
           (chunk_size - current_chunk_size) * D * sizeof(float));
}

// 设置right_context（右帧）
if (end < T) {
    // 不是最后一个chunk：使用接下来的2帧
    size_t rc_size = std::min(RIGHT_CTX, T - end);
    memcpy(rc_data, &input_array.data[end * D], rc_size * D * sizeof(float));
    if (rc_size < RIGHT_CTX) {
        // 如果不足2帧，用零填充
        memset(rc_data + rc_size * D, 0, 
               (RIGHT_CTX - rc_size) * D * sizeof(float));
    }
} else {
    // 最后一个chunk：使用零填充 ⚠️
    memset(rc_data, 0, RIGHT_CTX * D * sizeof(float));
}
```

**结果**：
- 输入tensor始终是 `[1, 10, 400]`，最后一个chunk是 `[1帧实际数据 + 9帧零填充]`
- right_context tensor是 `[2, 400]`，最后一个chunk使用 **全零填充**

#### 3. 模型推理

```cpp
status = interpreter.Invoke();
```

模型接收 `[1, 10, 400]` 输入，输出 `[1, 10, 2599]`

**关键**：即使输入只有1帧实际数据，模型仍然输出10帧的logits

#### 4. 输出收集（选择性保存）

```cpp
// 只保存 current_chunk_size 帧（不是全部10帧）
size_t saved_frames = 0;
for (size_t i = 0; i < current_chunk_size; i++) {  // current_chunk_size = 1
    for (size_t j = 0; j < output_dim; j++) {
        size_t idx = i * output_dim + j;
        output_logits.push_back(output_data[idx]);
    }
    saved_frames++;
}
```

**关键**：虽然模型输出了10帧，但只保存前 `current_chunk_size` 帧（即1帧）

### 完整示例：151帧的处理

```
输入：151帧 FBank特征 [1, 151, 400]

处理过程：
┌─────────────────────────────────────────────────────────┐
│ Chunk 0-14: 每个chunk处理10帧，保存10帧                │
│   输入: [1, 10, 400] (完整)                             │
│   右帧: [2, 400] (从后续帧获取)                         │
│   输出: [1, 10, 2599] → 保存10帧                        │
│   累计: 15 chunks × 10 frames = 150 frames              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Chunk 15: 最后一个chunk，只有1帧                        │
│   输入准备:                                              │
│     - 复制1帧实际数据: frames[150:151]                  │
│     - 填充9帧零: frames[151:160] = 0                    │
│   输入tensor: [1, 10, 400] (1帧实际 + 9帧填充)         │
│                                                          │
│   右帧准备:                                              │
│     - end = 151, T = 151                                │
│     - end >= T，所以是最后一个chunk                     │
│     - 使用全零填充: [2, 400] = 0 ⚠️                     │
│   右帧tensor: [2, 400] (全零)                           │
│                                                          │
│   模型推理:                                              │
│     Invoke() → 输出 [1, 10, 2599]                       │
│                                                          │
│   输出收集:                                              │
│     只保存前1帧: output[0, 0:1, :]                      │
│     丢弃后9帧: output[0, 1:10, :] (基于填充数据)        │
│   累计: 150 + 1 = 151 frames                            │
└─────────────────────────────────────────────────────────┘

最终输出：[1, 151, 2599] ✅
```

### 为什么这样设计？

#### 1. 模型输入形状固定
TFLite模型通常要求固定的输入形状 `[1, 10, 400]`，不能动态改变

#### 2. 避免重新编译
如果每个chunk使用不同的输入形状，需要重新分配tensor和重新编译模型，开销很大

#### 3. 填充不影响有效输出
- 零填充的帧不会影响前面实际帧的encoder输出
- 因为encoder是因果的（causal），每帧的输出只依赖于当前帧和之前的帧
- 填充帧的输出被丢弃，不会影响最终结果

### Right Context（右帧）的处理

#### 什么是Right Context？

Right context是stateful encoder模型的一个输入，表示当前chunk之后的2帧（RIGHT_CTX=2）。这些帧用于提供未来的上下文信息，帮助模型更好地理解当前帧。

#### Right Context的处理逻辑

```cpp
constexpr size_t RIGHT_CTX = 2;  // 右帧数量

if (end < T) {
    // 不是最后一个chunk：使用接下来的实际帧
    size_t rc_size = std::min(RIGHT_CTX, T - end);
    memcpy(rc_data, &input_array.data[end * D], rc_size * D * sizeof(float));
    
    // 如果剩余帧不足2帧，用零填充
    if (rc_size < RIGHT_CTX) {
        memset(rc_data + rc_size * D, 0, 
               (RIGHT_CTX - rc_size) * D * sizeof(float));
    }
} else {
    // 最后一个chunk：使用全零填充
    memset(rc_data, 0, RIGHT_CTX * D * sizeof(float));
}
```

#### 具体例子：151帧的Right Context

| Chunk | 输入帧范围 | end | T | 条件 | Right Context来源 |
|-------|-----------|-----|---|------|------------------|
| 0 | [0:10] | 10 | 151 | end < T | frames[10:12] (实际帧) |
| 1 | [10:20] | 20 | 151 | end < T | frames[20:22] (实际帧) |
| ... | ... | ... | 151 | end < T | ... |
| 14 | [140:150] | 150 | 151 | end < T | frames[150:151] + 1帧零填充 |
| 15 | [150:151] | 151 | 151 | **end >= T** | **全零填充** ⚠️ |

**关键点**：
- Chunk 14: end=150 < T=151，right_context使用frame[150]（1帧实际）+ 1帧零填充
- Chunk 15: end=151 >= T=151，right_context使用全零填充（2帧零）

#### 为什么最后一个chunk使用零填充？

1. **没有未来帧**：最后一个chunk之后没有更多的音频帧了
2. **保持一致性**：模型期望固定形状的right_context输入 `[2, 400]`
3. **匹配批处理**：Python批处理实现也是这样处理的（最后一个chunk的right_context为零）

```python
# Python实现（test_ctc_decode.py）
if end < T:
    rc = input_tensor[:, end:end + RIGHT_CTX, :]
    if rc.shape[1] < RIGHT_CTX:
        rc = torch.nn.functional.pad(rc, (0, 0, 0, RIGHT_CTX - rc.shape[1]))
else:
    # 最后一个chunk：使用零
    rc = torch.zeros(B, RIGHT_CTX, D, device=device, dtype=torch.float32)
```

### 代码关键变量

```cpp
size_t T = 151;                    // 总帧数
size_t chunk_size = 10;            // 固定chunk大小
size_t start = 150;                // 最后一个chunk的起始位置
size_t end = 151;                  // 最后一个chunk的结束位置
size_t current_chunk_size = 1;     // 实际帧数（end - start）

// 输入：填充到chunk_size
// [1帧实际数据 + 9帧零填充] → [1, 10, 400]

// 输出：只保存current_chunk_size帧
// [1, 10, 2599] → 只保存 [1, 1, 2599]
```

### 与Python实现的对比

#### Python (a2.py)
```python
# Python可以动态改变输入形状
for start in range(0, T, chunk_size):
    end = min(start + chunk_size, T)
    chunk = input_fbank[:, start:end, :]  # 可以是任意长度
    
    # 如果是最后一个chunk且不足10帧，需要填充
    if chunk.shape[1] < chunk_size:
        chunk = F.pad(chunk, (0, 0, 0, chunk_size - chunk.shape[1]))
    
    # 推理
    output_chunk = encoder(chunk, ...)
    
    # 只保存实际帧数
    actual_frames = end - start
    output_list.append(output_chunk[:, :actual_frames, :])
```

#### C++ (run_encoder.cc)
```cpp
// C++使用固定输入形状，手动填充和截取
for (size_t start = 0; start < T; start += chunk_size) {
    size_t end = std::min(start + chunk_size, T);
    size_t current_chunk_size = end - start;
    
    // 复制实际数据
    memcpy(input_data, &input_array.data[start * D], 
           current_chunk_size * D * sizeof(float));
    
    // 填充到chunk_size
    if (current_chunk_size < chunk_size) {
        memset(input_data + current_chunk_size * D, 0, 
               (chunk_size - current_chunk_size) * D * sizeof(float));
    }
    
    // 推理（固定输入形状 [1, 10, 400]）
    interpreter.Invoke();
    
    // 只保存实际帧数
    for (size_t i = 0; i < current_chunk_size; i++) {
        for (size_t j = 0; j < output_dim; j++) {
            output_logits.push_back(output_data[i * output_dim + j]);
        }
    }
}
```

### 总结

**问题**：151帧不能被10整除，最后一个chunk只有1帧

**解决方案**：
1. ✅ **输入填充**：将1帧填充到10帧（用零填充）
2. ✅ **固定推理**：模型始终接收 `[1, 10, 400]` 输入
3. ✅ **选择性输出**：只保存前1帧的输出，丢弃后9帧（基于填充的输出）
4. ✅ **最终结果**：输出形状正确 `[1, 151, 2599]`

这种方法既保持了模型输入形状的固定性（提高效率），又正确处理了不规则的输入长度。
