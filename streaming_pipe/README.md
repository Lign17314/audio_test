# Streaming FBank Extractor with Encoder

独立的流式FBank特征提取器和TFLite Micro编码器推理，用于实时音频特征提取和关键词识别。

## 目录结构

```
streaming_pipe/
├── inc/                          # 头文件
│   ├── streaming_fbank_extractor.h
│   └── cmvn_data.h
├── src/                          # 源文件
│   ├── streaming_fbank_main.cc       # 10ms版本（仅FBank）
│   ├── streaming_fbank_30ms_threaded.cc  # 30ms多线程版本（FBank + Encoder, float32）
│   ├── streaming_fbank_30ms_threaded_16x8.cc  # 30ms多线程版本（FBank + Encoder, 16x8量化）✅
│   └── streaming_fbank_extractor.cc
├── scripts/                      # Python测试脚本
│   ├── test_ctc_decode.py            # CTC解码测试
│   └── analyze_diff.py               # 差异分析
├── docs/                         # 📚 文档目录
│   ├── INDEX.md                      # 文档索引
│   ├── FINAL_SUMMARY.md              # 项目总结
│   ├── 16x8/                         # 16x8 量化相关文档
│   │   ├── 16X8_STATUS.md            # 16x8 实现状态
│   │   ├── 16X8_WORKFLOW.md          # 16x8 完整流程说明
│   │   ├── 16X8_FLOWCHART.md         # 16x8 流程图和架构图
│   │   ├── IMPLEMENT_16X8.md         # 16x8 实现指南
│   │   └── WHY_16X8_FAILS.md         # 16x8 问题分析（历史记录）
│   ├── verification/                 # 验证测试相关文档
│   │   ├── VERIFICATION_REPORT.md    # Float32 版本验证报告
│   │   └── VERIFICATION_COMPARISON.md # Float32 vs 16x8 对比
│   └── implementation/               # 实现细节文档
│       ├── ENCODER_INTEGRATION_TODO.md
│       ├── THREADED_ENCODER_COMPLETE.md
│       └── SUMMARY.md
├── build/                        # 编译输出
├── CMakeLists.txt
├── run.sh                        # 运行10ms版本（仅FBank）
├── run_30ms_threaded.sh          # 运行30ms多线程版本（FBank + Encoder, float32）
├── run_30ms_threaded_16x8.sh     # 运行30ms多线程版本（FBank + Encoder, 16x8量化）✅
├── compare_with_run_encoder.py   # 与run_encoder.cc对比
└── README.md                     # 本文件
```

## 📚 文档

详细文档请查看 **[docs/INDEX.md](docs/INDEX.md)**，包含：

- **16x8 量化版本** - 完整的流程说明、架构图、实现指南
- **验证测试** - Float32 和 16x8 版本的验证报告和对比分析
- **实现细节** - 开发过程中的技术决策和实现方案

**快速链接**：
- [项目总结](docs/FINAL_SUMMARY.md)
- [16x8 完整流程](docs/16x8/16X8_WORKFLOW.md)
- [Float32 vs 16x8 对比](docs/verification/VERIFICATION_COMPARISON.md)

## 快速开始

### 编译并运行（10ms版本 - 仅FBank提取）

```bash
bash run.sh
```

输出：`build/streaming_fbank_output.npy` (形状: 1×151×400)

### 编译并运行（30ms多线程版本 - FBank + Encoder推理）⭐

**Float32版本（推荐）**：

```bash
bash run_30ms_threaded.sh
```

输出：
- `build/streaming_fbank_30ms_threaded.npy` (FBank特征: 1×151×400)
- `build/streaming_fbank_30ms_threaded_logits.npy` (Encoder输出: 1×151×2599)

**16x8量化版本（已完成）**：✅

```bash
bash run_30ms_threaded_16x8.sh
```

输出：
- `build/streaming_fbank_30ms_threaded_16x8_logits.npy` (Encoder输出: 1×151×2599)

✅ **验证通过**：
- 关键词检测: "小云小云"
- 置信度: 97.46%
- 与 PyTorch 参考差异: 1.21%
- 与 Float32 版本差异: 0.47%

详见：[16x8 完整流程](docs/16x8/16X8_WORKFLOW.md) | [Float32 vs 16x8 对比](docs/verification/VERIFICATION_COMPARISON.md)

**说明**：多线程版本使用生产者-消费者模型：
- **生产者线程**（主线程）：
  - 以30ms为单位读取音频
  - 提取FBank特征（10ms帧移）
  - 将特征帧放入线程安全队列
- **消费者线程**：
  - 从队列中取出特征帧
  - 累积12帧（10帧输入 + 2帧right_context）
  - 运行TFLite Micro编码器推理
  - 输出logits用于CTC解码

**优势**：
- 解耦特征提取和编码器推理
- 真实模拟实时流式处理场景
- 与非线程版本run_encoder.cc输出**完全一致**（max_diff = 0.0）

### 验证输出

```bash
# 与run_encoder.cc对比（使用相同FBank特征）
cd tflite_micro_cpp/streaming_pipe
python3 compare_with_run_encoder.py
```

预期结果：✅ 所有151帧完全匹配（max_diff = 0.0）

详细验证报告见：`VERIFICATION_REPORT.md`

## 功能特性

- ✅ **流式处理**：支持10ms和30ms音频块实时处理
- ✅ **多线程架构**：生产者-消费者模型，解耦特征提取和编码器推理
- ✅ **TFLite Micro集成**：实时编码器推理（float32 和 16x8 量化模型）
- ✅ **16x8量化支持**：支持 INT16/INT8 量化模型，减少模型大小和内存占用
- ✅ **FBank特征**：80维Mel滤波器组
- ✅ **LFR降采样**：5帧拼接，每3帧输出1个 (输出400维)
- ✅ **CMVN归一化**：倒谱均值方差归一化
- ✅ **自包含**：无外部依赖
- ✅ **已验证**：Float32 和 16x8 版本都已验证通过

## 验证结果

### Float32 版本 ✅

#### 与run_encoder.cc对比

使用相同的FBank特征，线程版本和非线程版本的编码器输出**完全一致**：

```bash
python3 compare_with_run_encoder.py
```

结果：
- 所有151帧：max_diff = 0.0
- CTC解码：都识别为"小云小云"
- 置信度：97.93%

#### 与PyTorch参考对比

```bash
python3 ../../verify_encoder_output.py \
  build/streaming_fbank_30ms_threaded.npy \
  build/streaming_fbank_30ms_threaded_logits.npy \
  /root/volume/ctc/train/funasr_test/test_xiaoyun.wav
```

结果：
- Max diff: 7.095337e-04 (< 0.001阈值)
- CTC解码：都识别为"小云小云"
- 置信度差异：1.23e-07

详细验证报告见：[docs/verification/VERIFICATION_REPORT.md](docs/verification/VERIFICATION_REPORT.md)

### 16x8 量化版本 ✅

```bash
python3 ../../verify_encoder_output.py \
  ../../features/test_xiaoyun_fbank.npy \
  build/streaming_fbank_30ms_threaded_16x8_logits.npy \
  /root/volume/ctc/train/funasr_test/test_xiaoyun.wav
```

结果：
- 输出形状: (1, 151, 2599) ✅
- 关键词检测: "小云小云" ✅
- 置信度: 97.46%
- 与 PyTorch 参考差异: 1.21%
- 与 Float32 版本差异: 0.47%

详细对比见：[docs/verification/VERIFICATION_COMPARISON.md](docs/verification/VERIFICATION_COMPARISON.md)

### 版本对比总结

| 版本 | 模型大小 | 置信度 | 精度损失 | 推荐场景 |
|------|---------|--------|---------|---------|
| **Float32** | ~2MB | 97.93% | - | 精度要求高的场景 |
| **16x8 量化** | ~1MB | 97.46% | 0.47% | 嵌入式设备、资源受限场景 |

## 配置

### 修改输入文件

编辑相应的源文件：
- 10ms版本（仅FBank）：`src/streaming_fbank_main.cc`
- 30ms多线程版本（FBank + Encoder）：`src/streaming_fbank_30ms_threaded.cc`

```cpp
const char* wav_file = "/path/to/audio.wav";
const char* model_file = "/path/to/model.tflite";  // 仅多线程版本
```

### FBank参数

```cpp
FBankConfig config;
config.fs = 16000;           // 采样率
config.n_mels = 80;          // Mel滤波器数
config.frame_length = 25;    // 帧长(ms)
config.frame_shift = 10;     // 帧移(ms)
config.lfr_m = 5;            // LFR拼接数
config.lfr_n = 3;            // LFR降采样率
```

### 启用/禁用CMVN

在 `CMakeLists.txt` 中：

```cmake
add_definitions(-DUSE_CMVN_HEADER)  # 启用
# add_definitions(-DUSE_CMVN_HEADER)  # 禁用（注释掉）
```

## 技术细节

### 处理流程

#### 10ms版本（streaming_fbank_main.cc）- 仅FBank提取
```
音频输入 (WAV)
    ↓
[10ms块] → 音频缓冲
    ↓
提取FBank帧 → FBank缓冲
    ↓
应用LFR (5帧→1帧)
    ↓
应用CMVN
    ↓
输出特征 (400维)
```

#### 30ms多线程版本（streaming_fbank_30ms_threaded.cc）- FBank + Encoder
```
【生产者线程】
音频输入 (WAV)
    ↓
[30ms块] → 音频缓冲
    ↓
[10ms子块] → 内部处理
    ↓
提取FBank帧 → 线程安全队列
    ↓
【消费者线程】
从队列取出特征帧
    ↓
累积12帧 (10输入 + 2右上下文)
    ↓
TFLite Micro推理
    ↓
输出logits (2599维)
```

**关键特性**：
- 生产者和消费者并行运行
- 队列解耦两个线程
- 消费者等待足够的帧后才开始推理
- 最后一帧特殊处理（与run_encoder.cc完全一致）

### 输出格式

#### FBank特征
- **形状**：`(1, T, 400)`
  - T: LFR帧数
  - 400: 80 × 5 (n_mels × lfr_m)
- **格式**：NumPy .npy
- **类型**：float32

#### Encoder Logits（仅多线程版本）
- **形状**：`(1, T, 2599)`
  - T: 帧数（与FBank相同）
  - 2599: 输出维度（token数）
- **格式**：NumPy .npy
- **类型**：float32

## 测试结果

### 编译测试 ✅
- 编译成功，无错误
- 输出文件正常生成

### 功能测试 ✅
- FBank输出形状：(1, 151, 400) ✓
- Encoder输出形状：(1, 151, 2599) ✓
- 无NaN/Inf值 ✓
- CMVN正确应用 ✓

### 与run_encoder.cc对比 ✅

使用相同FBank特征：
- 所有151帧：max_diff = 0.0（完全一致）
- CTC解码：都识别为"小云小云"
- 置信度：97.93%

### 与PyTorch参考对比 ✅

- Max diff: 7.095337e-04 (< 0.001阈值)
- Mean diff: 4.372082e-06
- CTC解码：都识别为"小云小云"
- 置信度差异：1.23e-07

**结论**：线程版本的编码器推理与参考实现完全一致，可用于生产环境。

## 依赖

### 内部依赖
- `../fbank_extractor/` - FBank提取和WAV读取
- `../tflite_micro_install/` - TFLite Micro库

### 系统依赖
- CMake >= 3.10
- C++17编译器
- pthread（多线程版本）
- Python 3.x (仅测试脚本)

## 性能

- **处理速度**：实时 (30ms/块)
- **编码器推理**：~50ms/chunk（10帧）
- **内存占用**：
  - Tensor arena: 3MB
  - 总内存：< 10MB
- **输出延迟**：30-50ms

## 更新记录

### 2026-02-04
- ✅ 完成 16x8 量化版本实现和验证
- ✅ 整理文档到 docs 目录，创建分类结构
- ✅ 添加 Float32 vs 16x8 对比文档
- ✅ 集成TFLite Micro编码器推理
- ✅ 实现生产者-消费者多线程架构
- ✅ 验证与run_encoder.cc完全一致（max_diff = 0.0）
- ✅ 删除非线程版本，统一使用多线程实现
- ✅ 更新文档和验证报告

## 常见问题

### Q: 找不到WAV文件
A: 检查 `src/streaming_fbank_main.cc` 中的路径，使用绝对路径

### Q: 输出形状不对
A: 检查LFR参数 (lfr_m, lfr_n) 是否正确

### Q: 编译错误
A: 确保依赖文件存在：
- `../fbank_extractor/fbank_extractor.h`
- `../tflite_micro_install/lib/libtensorflow-microlite.a`

### Q: CTC解码失败
A: 确保CMVN已启用 (`-DUSE_CMVN_HEADER`)

## 手动编译

```bash
mkdir -p build
cd build
cmake ..
make
./streaming_pipe
```

## 许可

Copyright 2025
