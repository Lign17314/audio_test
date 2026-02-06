# Streaming FBank 文档索引

本目录包含了 Streaming FBank + Encoder 项目的所有文档，按类别组织。

## 📁 文档结构

```
docs/
├── INDEX.md                    # 本文件 - 文档索引
├── FINAL_SUMMARY.md           # 项目总结
├── 16x8/                      # 16x8 量化相关文档
│   ├── 16X8_INPUT_SPECIFICATION.md  # ⭐ 输入规格详解
│   ├── 16X8_STATUS.md         # 16x8 实现状态
│   ├── 16X8_WORKFLOW.md       # 16x8 完整流程说明
│   ├── 16X8_FLOWCHART.md      # 16x8 流程图和架构图
│   ├── IMPLEMENT_16X8.md      # 16x8 实现指南
│   └── WHY_16X8_FAILS.md      # 16x8 问题分析（历史记录）
├── verification/              # 验证测试相关文档
│   ├── VERIFICATION_REPORT.md      # Float32 版本验证报告
│   └── VERIFICATION_COMPARISON.md  # Float32 vs 16x8 对比
└── implementation/            # 实现细节文档
    ├── ENCODER_INTEGRATION_TODO.md      # Encoder 集成待办事项
    ├── THREADED_ENCODER_COMPLETE.md     # 多线程 Encoder 完成报告
    └── SUMMARY.md                       # 实现总结
```

## 📚 文档分类说明

### 🎯 快速开始
- **[README.md](../README.md)** - 项目主文档，包含编译和运行说明
- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - 项目总结，了解整体进展

### 🔢 16x8 量化版本
16x8 量化版本使用 INT16/INT8 量化模型，减少模型大小和内存占用。

#### 基础文档
- **[16X8_NAMING_EXPLAINED.md](16x8/16X8_NAMING_EXPLAINED.md)** - ⭐ **16x8 命名解释**
  - 为什么叫 "16x8"？
  - INT8 在哪里？INT16 在哪里？
  - 混合精度量化详解
  - 数据流和内部计算
- **[16X8_INPUT_SPECIFICATION.md](16x8/16X8_INPUT_SPECIFICATION.md)** - ⭐ **输入规格详解**
  - 6 个输入 tensor 的详细规格
  - 量化参数说明
  - 数据流图和内存占用
  - 完整的推理流程示例
  - 宏定义使用说明
- **[QUICK_INPUT_GUIDE.md](16x8/QUICK_INPUT_GUIDE.md)** - 快速输入指南（简洁版）

#### INT16 优化（推荐）⭐
- **[QUICK_START_INT16.md](16x8/QUICK_START_INT16.md)** - 🚀 **INT16 优化快速开始**
  - 一分钟快速开始
  - 版本对比表
  - 核心优化说明
  - 使用场景推荐
  - 常见问题解答
- **[INT16_OPTIMIZATION_COMPLETE.md](16x8/INT16_OPTIMIZATION_COMPLETE.md)** - ⭐ **INT16 优化完成报告**
  - 实现方案详解
  - 文件结构说明
  - 编译运行指南
  - 性能对比数据
  - 技术细节和验证
- **[FBANK_INT16_OPTIMIZATION_ANALYSIS.md](16x8/FBANK_INT16_OPTIMIZATION_ANALYSIS.md)** - FBank INT16 优化可行性分析
  - 三种优化方案对比
  - 可行性分析
  - 性能和精度评估
  - 实现建议

#### 全流程 INT16 方案（高级）
- **[FULL_INT16_SUMMARY.md](16x8/FULL_INT16_SUMMARY.md)** - ⭐ **全流程 INT16 方案总结**
  - 完整的定点实现方案
  - 分 5 个部分详细说明
  - 实现路线图（9-13 周）
  - 风险和收益分析
- **[FULL_INT16_PART1_OVERVIEW.md](16x8/FULL_INT16_PART1_OVERVIEW.md)** - Part 1: 概述和架构
- **[FULL_INT16_PART2_FFT.md](16x8/FULL_INT16_PART2_FFT.md)** - Part 2: 定点 FFT
- **[FULL_INT16_PART3_MEL_LOG.md](16x8/FULL_INT16_PART3_MEL_LOG.md)** - Part 3: Mel 滤波和对数
- **[FULL_INT16_PART4_LFR_CMVN.md](16x8/FULL_INT16_PART4_LFR_CMVN.md)** - Part 4: LFR 和 CMVN
- **[FULL_INT16_PART5_IMPLEMENTATION.md](16x8/FULL_INT16_PART5_IMPLEMENTATION.md)** - Part 5: 实现路线图

#### 实现文档
- **[16X8_STATUS.md](16x8/16X8_STATUS.md)** - 当前实现状态和测试结果
- **[16X8_WORKFLOW.md](16x8/16X8_WORKFLOW.md)** - 详细的流程说明
- **[16X8_FLOWCHART.md](16x8/16X8_FLOWCHART.md)** - 流程图和架构图
- **[IMPLEMENT_16X8.md](16x8/IMPLEMENT_16X8.md)** - 实现指南和技术细节
- **[WHY_16X8_FAILS.md](16x8/WHY_16X8_FAILS.md)** - 历史问题分析（已解决）

### ✅ 验证测试
验证测试文档记录了不同版本的测试结果和对比分析。

- **[VERIFICATION_REPORT.md](verification/VERIFICATION_REPORT.md)** - Float32 版本验证报告
  - 输出形状验证
  - 与 PyTorch 参考对比
  - CTC 解码结果
  - 关键词检测结果
- **[VERIFICATION_COMPARISON.md](verification/VERIFICATION_COMPARISON.md)** - Float32 vs 16x8 对比
  - 详细的精度对比
  - 逐 Chunk 差异分析
  - CTC 解码详细结果
  - 性能优势分析
  - 推荐使用场景

### 🛠️ 实现细节
实现细节文档记录了开发过程中的技术决策和实现方案。

- **[ENCODER_INTEGRATION_TODO.md](implementation/ENCODER_INTEGRATION_TODO.md)** - Encoder 集成待办事项
- **[THREADED_ENCODER_COMPLETE.md](implementation/THREADED_ENCODER_COMPLETE.md)** - 多线程 Encoder 完成报告
- **[SUMMARY.md](implementation/SUMMARY.md)** - 实现总结

## 🎯 推荐阅读顺序

### 新用户
1. [README.md](../README.md) - 了解项目基本信息
2. [FINAL_SUMMARY.md](FINAL_SUMMARY.md) - 了解项目整体进展
3. **[QUICK_START_INT16.md](16x8/QUICK_START_INT16.md)** - 🚀 **快速开始 INT16 优化版本**
4. [16X8_INPUT_SPECIFICATION.md](16x8/16X8_INPUT_SPECIFICATION.md) - 了解 16x8 输入规格
5. [VERIFICATION_COMPARISON.md](verification/VERIFICATION_COMPARISON.md) - 了解不同版本的性能对比

### 开发者
1. [16X8_INPUT_SPECIFICATION.md](16x8/16X8_INPUT_SPECIFICATION.md) - 输入规格详解
2. [16X8_WORKFLOW.md](16x8/16X8_WORKFLOW.md) - 详细的实现流程
3. [16X8_FLOWCHART.md](16x8/16X8_FLOWCHART.md) - 流程图和架构图
4. [IMPLEMENT_16X8.md](16x8/IMPLEMENT_16X8.md) - 实现指南
5. [implementation/](implementation/) - 实现细节文档

### 测试验证
1. [VERIFICATION_REPORT.md](verification/VERIFICATION_REPORT.md) - Float32 验证报告
2. [VERIFICATION_COMPARISON.md](verification/VERIFICATION_COMPARISON.md) - 版本对比
3. [16X8_STATUS.md](16x8/16X8_STATUS.md) - 16x8 测试结果

## 📊 版本对比总结

| 版本 | 模型大小 | 置信度 | 精度损失 | 队列内存 | 速度 | 推荐场景 |
|------|---------|--------|---------|---------|------|---------|
| **Float32** | ~2MB | 97.93% | - | 100% | 100% | 精度要求高的场景 |
| **16x8 量化** | ~1MB | 97.46% | 0.47% | 100% | 100% | 嵌入式设备、资源受限场景 |
| **16x8 INT16** ⭐ | ~1MB | 97.46% | 0.47% | **50%** | **105-110%** | **推荐：嵌入式 + 实时** |

## 🔗 相关链接

- **源代码**: `../src/`
- **编译脚本**: `../run_30ms_threaded.sh`, `../run_30ms_threaded_16x8.sh`
- **验证脚本**: `../../../verify_encoder_output.py`

## 📝 更新日志

- **2026-02-04**: 整理文档到 docs 目录，创建分类结构
- **2026-02-04**: 完成 16x8 量化版本实现和验证
- **2026-02-04**: 完成 Float32 版本实现和验证

---

**状态**: ✅ 项目完成并验证通过  
**最后更新**: 2026-02-04
