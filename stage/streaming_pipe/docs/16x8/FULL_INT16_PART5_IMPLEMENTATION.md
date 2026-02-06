# 全流程 INT16 实现方案 - Part 5: 实现路线图

## 实现步骤

### 阶段 1: 基础设施（1-2 周）

#### 1.1 定点数学库
```cpp
// fixed_point_math.h
namespace FixedPoint {
    // Q15 基本运算
    int16_t mul_q15(int16_t a, int16_t b);
    int16_t div_q15(int16_t a, int16_t b);
    int16_t add_q15(int16_t a, int16_t b);
    int16_t sub_q15(int16_t a, int16_t b);
    
    // 饱和运算
    int16_t saturate_int16(int32_t value);
    
    // 转换
    int16_t float_to_q15(float x);
    float q15_to_float(int16_t x);
}
```

#### 1.2 单元测试框架
```cpp
// 测试定点运算精度
TEST(FixedPointMath, Multiplication) {
    float a = 0.5f, b = 0.3f;
    int16_t a_q15 = float_to_q15(a);
    int16_t b_q15 = float_to_q15(b);
    int16_t result_q15 = mul_q15(a_q15, b_q15);
    float result = q15_to_float(result_q15);
    
    EXPECT_NEAR(result, a * b, 0.001f);  // 0.1% 误差
}
```

### 阶段 2: FFT 实现（2-3 周）

#### 2.1 选择 FFT 库
- **选项 A**: CMSIS-DSP（ARM 平台，推荐）
- **选项 B**: KissFFT 定点版本（跨平台）
- **选项 C**: 自己实现（学习用途）

#### 2.2 集成和测试
```cpp
// 对比 float32 FFT 和 Q15 FFT 的输出
void test_fft_accuracy() {
    // 生成测试信号
    std::vector<float> signal_float(512);
    std::vector<int16_t> signal_q15(512);
    
    for (int i = 0; i < 512; i++) {
        signal_float[i] = sinf(2 * M_PI * 10 * i / 512);
        signal_q15[i] = float_to_q15(signal_float[i]);
    }
    
    // Float32 FFT
    std::vector<float> fft_float_real, fft_float_imag;
    fft_float(signal_float, fft_float_real, fft_float_imag);
    
    // Q15 FFT
    std::vector<int16_t> fft_q15_real, fft_q15_imag;
    fft_q15(signal_q15, fft_q15_real, fft_q15_imag);
    
    // 对比误差
    for (int i = 0; i < 257; i++) {
        float real_float = fft_float_real[i];
        float real_q15 = q15_to_float(fft_q15_real[i]);
        float error = fabsf(real_float - real_q15);
        
        printf("Bin %d: float=%.6f, q15=%.6f, error=%.6f\n",
               i, real_float, real_q15, error);
    }
}
```

### 阶段 3: Mel 滤波器和对数（1-2 周）

#### 3.1 Mel 滤波器预计算
```bash
# 生成 Mel 滤波器系数（Q15）
python3 generate_mel_filters_q15.py \
    --n_fft 512 \
    --n_mels 80 \
    --sample_rate 16000 \
    --output mel_filters_q15.h
```

#### 3.2 对数查找表生成
```bash
# 生成对数查找表
python3 generate_log_table_q15.py \
    --method interpolated \
    --step 256 \
    --output log_table_q15.h
```

### 阶段 4: LFR 和 CMVN（1 周）

#### 4.1 CMVN 数据转换
```python
# 将 float32 CMVN 转换为 Q15
import numpy as np

means = np.load('cmvn_means.npy')
vars = np.load('cmvn_vars.npy')

means_q15 = (means * 32768).astype(np.int16)
stds = np.sqrt(vars + 1e-10)
inv_stds_q15 = ((1.0 / stds) * 32768).astype(np.int16)

# 生成头文件
with open('cmvn_data_q15.h', 'w') as f:
    f.write('const int16_t cmvn_means_q15[] = {\n')
    f.write(','.join(map(str, means_q15)))
    f.write('\n};\n')
    # ...
```

### 阶段 5: 集成和优化（2-3 周）

#### 5.1 完整流程测试
```cpp
void test_full_pipeline() {
    // 加载测试音频
    WavData wav = WavReader::read_wav("test.wav");
    
    // Float32 版本
    StreamingFBankExtractor extractor_float(config);
    std::vector<std::vector<float>> output_float;
    extractor_float.process_chunk(wav.samples.data(), 
                                  wav.samples.size(), 
                                  output_float);
    
    // Q15 版本
    StreamingFBankExtractorQ15 extractor_q15(config);
    std::vector<std::vector<int16_t>> output_q15;
    extractor_q15.process_chunk(wav.samples_int16.data(),
                                wav.samples.size(),
                                output_q15);
    
    // 对比输出
    compare_outputs(output_float, output_q15);
}
```

#### 5.2 性能优化
- SIMD 优化（NEON, SSE）
- 内存对齐
- 缓存优化
- 并行化

### 阶段 6: 验证和调优（2-3 周）

#### 6.1 精度验证
```bash
# 运行完整的识别测试
./test_recognition_accuracy \
    --float32_model fsmn_encoder_stateful_float32.tflite \
    --int16_model fsmn_encoder_stateful_16x8.tflite \
    --test_set test_xiaoyun.wav \
    --compare_fbank
```

#### 6.2 性能测试
```bash
# 性能基准测试
./benchmark_fbank \
    --float32 \
    --int16 \
    --iterations 1000 \
    --report performance_report.txt
```

## 预期时间线

```
总计: 9-13 周

Week 1-2:   基础设施 + 单元测试
Week 3-5:   FFT 实现和验证
Week 6-7:   Mel 滤波器 + 对数
Week 8:     LFR + CMVN
Week 9-11:  集成和优化
Week 12-13: 验证和调优
```

## 风险和挑战

### 高风险项

1. **FFT 精度**
   - 风险: 定点 FFT 误差累积
   - 缓解: 使用成熟库（CMSIS-DSP）
   - 备选: 增加位宽（Q31）

2. **对数运算精度**
   - 风险: 查找表精度不足
   - 缓解: 使用插值或更密集的表
   - 备选: 使用近似算法

3. **CMVN 溢出**
   - 风险: 除法可能溢出
   - 缓解: 预计算 1/std，使用乘法
   - 备选: 动态范围调整

### 中风险项

1. **内存占用**
   - 对数查找表: 64 KB（可优化到 256 B）
   - Mel 滤波器系数: ~10 KB
   - FFT 旋转因子: ~1 KB

2. **性能瓶颈**
   - FFT 可能比 float32 慢（无 SIMD 优化时）
   - 需要针对目标平台优化

## 成功标准

### 必须达到

- ✅ 识别精度下降 < 1%（97.46% → 96.5%+）
- ✅ 内存占用减少 > 30%
- ✅ 代码可维护性良好

### 期望达到

- ⭐ 识别精度下降 < 0.5%
- ⭐ 处理速度提升 > 20%
- ⭐ 功耗降低 > 30%

### 理想达到

- 🎯 识别精度下降 < 0.2%
- 🎯 处理速度提升 > 50%
- 🎯 功耗降低 > 50%

