# 全流程 INT16 实现方案 - Part 3: Mel 滤波和对数

## 1. 功率谱计算（Q15）

```cpp
// 输入: FFT 输出（复数，Q15）
// 输出: 功率谱（Q15）
void compute_power_spectrum_q15(
    const int16_t* fft_real,  // Q15
    const int16_t* fft_imag,  // Q15
    int16_t* power_spectrum,  // Q15
    int fft_size) {
    
    for (int i = 0; i < fft_size / 2 + 1; i++) {
        // |X|^2 = real^2 + imag^2
        int32_t real_sq = ((int32_t)fft_real[i] * fft_real[i]) >> 15;
        int32_t imag_sq = ((int32_t)fft_imag[i] * fft_imag[i]) >> 15;
        
        int32_t power = real_sq + imag_sq;
        
        // 归一化到 Q15
        power_spectrum[i] = saturate_int16(power);
    }
}
```

## 2. Mel 滤波器组（Q15）

### 预计算滤波器系数

```cpp
class MelFilterBankQ15 {
public:
    MelFilterBankQ15(int n_fft, int n_mels, int sample_rate) {
        // 预计算 Mel 滤波器系数（Q15 格式）
        compute_mel_filters_q15(n_fft, n_mels, sample_rate);
    }
    
    // 应用 Mel 滤波器
    void apply(const int16_t* power_spectrum,  // Q15
               int16_t* mel_energies) {        // Q15
        
        for (int m = 0; m < n_mels_; m++) {
            int64_t energy = 0;  // 使用 int64 避免溢出
            
            // 对每个 Mel bin 进行加权求和
            for (int k = mel_start_[m]; k <= mel_end_[m]; k++) {
                int32_t weighted = ((int32_t)power_spectrum[k] * 
                                   mel_weights_[m][k - mel_start_[m]]) >> 15;
                energy += weighted;
            }
            
            // 归一化并饱和
            mel_energies[m] = saturate_int16(energy >> 15);
        }
    }
    
private:
    int n_mels_;
    std::vector<int> mel_start_;  // 每个 Mel bin 的起始频率索引
    std::vector<int> mel_end_;    // 每个 Mel bin 的结束频率索引
    std::vector<std::vector<int16_t>> mel_weights_;  // Q15 权重
    
    void compute_mel_filters_q15(int n_fft, int n_mels, int sample_rate) {
        // 1. 计算 Mel 频率点（float）
        std::vector<float> mel_points = compute_mel_points(n_mels, sample_rate);
        
        // 2. 转换为 FFT bin 索引
        std::vector<int> fft_bins = mel_to_fft_bins(mel_points, n_fft, sample_rate);
        
        // 3. 计算三角滤波器权重并量化为 Q15
        mel_weights_.resize(n_mels);
        mel_start_.resize(n_mels);
        mel_end_.resize(n_mels);
        
        for (int m = 0; m < n_mels; m++) {
            int start = fft_bins[m];
            int peak = fft_bins[m + 1];
            int end = fft_bins[m + 2];
            
            mel_start_[m] = start;
            mel_end_[m] = end;
            mel_weights_[m].resize(end - start + 1);
            
            // 上升斜坡
            for (int k = start; k < peak; k++) {
                float weight = (float)(k - start) / (peak - start);
                mel_weights_[m][k - start] = (int16_t)(weight * 32768.0f);
            }
            
            // 下降斜坡
            for (int k = peak; k <= end; k++) {
                float weight = (float)(end - k) / (end - peak);
                mel_weights_[m][k - start] = (int16_t)(weight * 32768.0f);
            }
        }
    }
};
```

## 3. 对数运算（查找表 + 插值）

### 方案 A: 直接查找表

```cpp
class LogLookupTableQ15 {
public:
    LogLookupTableQ15() {
        // 预计算对数表
        // 输入范围: [1, 32767] (Q15 正数)
        // 输出范围: log(x/32768) in Q15
        
        log_table_.resize(32768);
        for (int i = 1; i < 32768; i++) {
            float x = (float)i / 32768.0f;
            float log_x = logf(x + 1e-10f);  // 避免 log(0)
            log_table_[i] = (int16_t)(log_x * 32768.0f);
        }
    }
    
    int16_t compute(int16_t x) {
        if (x <= 0) return log_table_[1];  // 最小值
        if (x >= 32767) return log_table_[32767];
        return log_table_[x];
    }
    
private:
    std::vector<int16_t> log_table_;  // 64 KB
};
```

### 方案 B: 分段线性插值（节省内存）

```cpp
class LogLookupTableInterpolated {
public:
    LogLookupTableInterpolated() {
        // 只存储关键点，使用线性插值
        // 例如：每 256 个点存储一个
        
        const int step = 256;
        log_table_.resize(32768 / step + 1);
        
        for (int i = 0; i < log_table_.size(); i++) {
            int x = i * step;
            if (x == 0) x = 1;
            float log_x = logf((float)x / 32768.0f + 1e-10f);
            log_table_[i] = (int16_t)(log_x * 32768.0f);
        }
    }
    
    int16_t compute(int16_t x) {
        if (x <= 0) return log_table_[0];
        
        const int step = 256;
        int idx = x / step;
        int remainder = x % step;
        
        if (idx >= log_table_.size() - 1) {
            return log_table_[log_table_.size() - 1];
        }
        
        // 线性插值
        int32_t y0 = log_table_[idx];
        int32_t y1 = log_table_[idx + 1];
        int32_t interpolated = y0 + ((y1 - y0) * remainder) / step;
        
        return saturate_int16(interpolated);
    }
    
private:
    std::vector<int16_t> log_table_;  // 只需 256 bytes
};
```

### 方案 C: 快速对数近似（最快）

```cpp
// 基于 IEEE 754 浮点表示的快速对数
int16_t fast_log2_q15(int16_t x) {
    if (x <= 0) return -32768;  // -inf
    
    // 计算前导零个数
    int leading_zeros = __builtin_clz((uint32_t)x) - 16;
    
    // log2(x) ≈ 15 - leading_zeros + fraction
    int16_t log2_approx = (15 - leading_zeros) << 15;
    
    // 可以添加小数部分的近似
    // ...
    
    return log2_approx;
}

// log(x) = log2(x) * ln(2)
int16_t fast_log_q15(int16_t x) {
    int16_t log2_x = fast_log2_q15(x);
    // ln(2) ≈ 0.693147 in Q15 = 22713
    return (int16_t)(((int32_t)log2_x * 22713) >> 15);
}
```

