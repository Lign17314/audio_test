# Streaming FBank + Encoder 项目总结

## 项目概述

本项目实现了实时流式音频处理系统，包括FBank特征提取和TFLite Micro编码器推理，用于关键词识别（KWS）。

## 实现版本

### 1. Float32版本（主要实现）✅

**文件**：`src/streaming_fbank_30ms_threaded.cc`

**状态**：✅ 完全验证，生产就绪

**特性**：
- 多线程架构（生产者-消费者模型）
- 实时FBank特征提取（30ms音频块）
- TFLite Micro编码器推理（float32模型）
- 与run_encoder.cc输出完全一致（max_diff = 0.0）

**运行**：
```bash
bash run_30ms_threaded.sh
```

**验证**：
```bash
python3 compare_with_run_encoder.py
python3 ../../verify_encoder_output.py \
  build/streaming_fbank_30ms_threaded.npy \
  build/streaming_fbank_30ms_threaded_logits.npy \
  /path/to/test.wav
```

### 2. 16x8量化版本（开发中）🚧

**文件**：`src/streaming_fbank_30ms_threaded_16x8.cc`

**状态**：⚠️ 不可用（段错误）

**问题**：
- 代码假设所有tensor都是float32类型
- 没有实现量化/反量化逻辑
- 直接复制float数据到int8/int16 tensor导致段错误

**详细分析**：
- `16X8_STATUS.md` - 开发状态
- `WHY_16X8_FAILS.md` - 技术分析

**替代方案**：
使用两步处理：
1. Streaming版本生成FBank
2. run_encoder_16x8.cc处理（已验证）

## 验证结果

### Float32版本

#### 与run_encoder.cc对比
- ✅ 所有151帧：max_diff = 0.0（完全一致）
- ✅ CTC解码：都识别为"小云小云"
- ✅ 置信度：97.93%

#### 与PyTorch参考对比
- ✅ Max diff: 7.095337e-04 (< 0.001阈值)
- ✅ Mean diff: 4.372082e-06
- ✅ CTC解码：都识别为"小云小云"
- ✅ 置信度差异：1.23e-07

## 技术架构

### 多线程模型

```
┌─────────────────────────────────────────────────────────┐
│                    主线程（生产者）                        │
│                                                           │
│  读取30ms音频 → 提取FBank → 应用LFR/CMVN → 放入队列      │
└─────────────────────────────────────────────────────────┘
                            ↓
                    线程安全队列
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   消费者线程                              │
│                                                           │
│  从队列取帧 → 累积12帧 → TFLite推理 → 输出logits        │
└─────────────────────────────────────────────────────────┘
```

### 处理流程

1. **音频输入**：30ms块（480 samples @ 16kHz）
2. **FBank提取**：10ms帧移，80维Mel滤波器
3. **LFR降采样**：5帧拼接，每3帧输出1个（400维）
4. **CMVN归一化**：倒谱均值方差归一化
5. **Encoder推理**：
   - 累积12帧（10输入 + 2右上下文）
   - TFLite Micro推理
   - 输出10帧logits（2599维）
6. **CTC解码**：识别关键词

### 关键特性

- ✅ **实时处理**：30ms延迟
- ✅ **线程安全**：生产者-消费者解耦
- ✅ **状态管理**：正确处理cache和right_context
- ✅ **边界处理**：正确处理最后一帧
- ✅ **内存效率**：3MB tensor arena

## 文件结构

```
streaming_fbank_only/
├── src/
│   ├── streaming_fbank_30ms_threaded.cc       # Float32版本 ✅
│   ├── streaming_fbank_30ms_threaded_16x8.cc  # 16x8版本 🚧
│   └── streaming_fbank_extractor.cc
├── inc/
│   ├── streaming_fbank_extractor.h
│   └── cmvn_data.h
├── scripts/
│   ├── test_ctc_decode.py
│   └── analyze_diff.py
├── run_30ms_threaded.sh                       # Float32运行脚本
├── run_30ms_threaded_16x8.sh                  # 16x8运行脚本
├── compare_with_run_encoder.py                # 对比脚本
├── VERIFICATION_REPORT.md                     # 验证报告
├── 16X8_STATUS.md                             # 16x8状态
├── WHY_16X8_FAILS.md                          # 16x8技术分析
├── SUMMARY.md                                 # 本文件
└── README.md
```

## 性能指标

### Float32版本

- **处理速度**：实时（30ms/块）
- **编码器推理**：~50ms/chunk（10帧）
- **内存占用**：
  - Tensor arena: 3MB
  - 总内存：< 10MB
- **输出延迟**：30-50ms
- **准确率**：与PyTorch参考一致

## 使用建议

### 生产环境

**推荐使用Float32版本**：
```bash
bash run_30ms_threaded.sh
```

理由：
- ✅ 完全验证
- ✅ 稳定可靠
- ✅ 性能足够（实时处理）
- ✅ 准确率高

### 如需16x8模型

**使用两步方案**：
```bash
# 步骤1：生成FBank特征
bash run_30ms_threaded.sh

# 步骤2：用run_encoder_16x8处理
cd ../encoder_runner
./run.sh ../tflite_models/fsmn_encoder_stateful_16x8.tflite \
         ../streaming_fbank_only/build/streaming_fbank_30ms_threaded.npy \
         output_16x8.npy
```

## 未来工作

### 短期
- [ ] 完成16x8版本的量化/反量化逻辑
- [ ] 测试和验证16x8版本
- [ ] 性能优化

### 中期
- [ ] 支持更多量化格式（int4, int16）
- [ ] 添加更多模型支持
- [ ] 优化内存占用

### 长期
- [ ] 硬件加速支持
- [ ] 多模型并行推理
- [ ] 端到端优化

## 参考文档

- `README.md` - 快速开始指南
- `VERIFICATION_REPORT.md` - 详细验证报告
- `16X8_STATUS.md` - 16x8开发状态
- `WHY_16X8_FAILS.md` - 16x8技术分析
- `THREADED_ENCODER_COMPLETE.md` - 实现完成记录

## 贡献者

- 实现：基于FunASR和TFLite Micro
- 验证：与run_encoder.cc和PyTorch参考对比
- 文档：完整的技术文档和验证报告

---

**更新日期**：2026-02-04  
**版本**：1.0（Float32完成，16x8开发中）  
**状态**：✅ Float32生产就绪，🚧 16x8开发中
