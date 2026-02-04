# 全流程 INT16 实现方案 - Part 2: 定点 FFT

## FFT 定点实现

### 1. 算法选择

**推荐**: Radix-2 Cooley-Tukey FFT（定点版本）

**原因**:
- ✅ 实现相对简单
- ✅ 适合 2 的幂次长度（512）
- ✅ 有成熟的定点实现参考

### 2. 定点 FFT 实现

```cpp
// Q15 定点 FFT
class FixedPointFFT {
public:
    FixedPointFFT(int fft_size) : fft_size_(fft_size) {
        // 预计算旋转因子（twiddle factors）
        precompute_twiddle_factors();
    }
    
    // 输入: Q15 实数
    // 输出: Q15 复数（实部和虚部）
    void compute(const int16_t* input, 
                 int16_t* real_output, 
                 int16_t* imag_output) {
        // 1. 位反转排列
        bit_reverse_copy(input, real_output);
        
        // 2. FFT 蝶形运算
        for (int stage = 1; stage <= log2(fft_size_); stage++) {
            int m = 1 << stage;
            int m2 = m >> 1;
            
            for (int k = 0; k < fft_size_; k += m) {
                for (int j = 0; j < m2; j++) {
                    int idx = k + j;
                    int idx_m2 = idx + m2;
                    
                    // 蝶形运算（Q15）
                    butterfly_q15(real_output[idx], imag_output[idx],
                                 real_output[idx_m2], imag_output[idx_m2],
                                 twiddle_real_[j * (fft_size_ / m)],
                                 twiddle_imag_[j * (fft_size_ / m)]);
                }
            }
        }
    }
    
private:
    int fft_size_;
    std::vector<int16_t> twiddle_real_;  // Q15
    std::vector<int16_t> twiddle_imag_;  // Q15
    
    void precompute_twiddle_factors() {
        twiddle_real_.resize(fft_size_ / 2);
        twiddle_imag_.resize(fft_size_ / 2);
        
        for (int k = 0; k < fft_size_ / 2; k++) {
            float angle = -2.0f * M_PI * k / fft_size_;
            twiddle_real_[k] = (int16_t)(cosf(angle) * 32768.0f);
            twiddle_imag_[k] = (int16_t)(sinf(angle) * 32768.0f);
        }
    }
    
    // Q15 蝶形运算
    void butterfly_q15(int16_t& ar, int16_t& ai,
                      int16_t& br, int16_t& bi,
                      int16_t wr, int16_t wi) {
        // 复数乘法: (br + j*bi) * (wr + j*wi)
        // = (br*wr - bi*wi) + j*(br*wi + bi*wr)
        
        int32_t tr = ((int32_t)br * wr - (int32_t)bi * wi) >> 15;
        int32_t ti = ((int32_t)br * wi + (int32_t)bi * wr) >> 15;
        
        // 蝶形更新
        int32_t new_ar = ar + tr;
        int32_t new_ai = ai + ti;
        int32_t new_br = ar - tr;
        int32_t new_bi = ai - ti;
        
        // 饱和到 INT16
        ar = saturate_int16(new_ar);
        ai = saturate_int16(new_ai);
        br = saturate_int16(new_br);
        bi = saturate_int16(new_bi);
    }
    
    int16_t saturate_int16(int32_t value) {
        if (value > 32767) return 32767;
        if (value < -32768) return -32768;
        return (int16_t)value;
    }
};
```

### 3. 使用开源库（推荐）

**CMSIS-DSP** (ARM):
```cpp
#include "arm_math.h"

// 初始化
arm_rfft_instance_q15 fft_instance;
arm_rfft_init_q15(&fft_instance, 512, 0, 1);

// 执行 FFT
int16_t input_q15[512];
int16_t output_q15[512];
arm_rfft_q15(&fft_instance, input_q15, output_q15);
```

**KissFFT** (跨平台):
```cpp
// KissFFT 有定点版本
#include "kiss_fftr.h"

kiss_fftr_cfg cfg = kiss_fftr_alloc(512, 0, NULL, NULL);
kiss_fft_scalar input[512];  // int16_t
kiss_fft_cpx output[257];    // 复数输出
kiss_fftr(cfg, input, output);
```

