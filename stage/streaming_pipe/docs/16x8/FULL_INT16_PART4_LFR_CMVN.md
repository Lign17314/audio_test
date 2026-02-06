# 全流程 INT16 实现方案 - Part 4: LFR 和 CMVN

## 1. LFR (Low Frame Rate) 处理

### 定点 LFR 实现

```cpp
// LFR: 将 5 帧拼接，每 3 帧输出 1 帧
// 输入: 80 维 × N 帧 (Q15)
// 输出: 400 维 × M 帧 (Q15)
class LFRQ15 {
public:
    LFRQ15(int lfr_m, int lfr_n) 
        : lfr_m_(lfr_m), lfr_n_(lfr_n) {}
    
    void process(const std::vector<std::vector<int16_t>>& input_frames,
                 std::vector<std::vector<int16_t>>& output_frames) {
        
        output_frames.clear();
        
        for (size_t i = 0; i + lfr_m_ <= input_frames.size(); i += lfr_n_) {
            std::vector<int16_t> lfr_frame;
            lfr_frame.reserve(input_frames[0].size() * lfr_m_);
            
            // 拼接 lfr_m 帧
            for (int j = 0; j < lfr_m_; j++) {
                const auto& frame = input_frames[i + j];
                lfr_frame.insert(lfr_frame.end(), frame.begin(), frame.end());
            }
            
            output_frames.push_back(lfr_frame);
        }
    }
    
private:
    int lfr_m_;  // 拼接帧数（5）
    int lfr_n_;  // 跳过帧数（3）
};
```

**说明**: LFR 只是数据重排，不涉及运算，INT16 实现与 float32 完全相同。

## 2. CMVN (倒谱均值方差归一化)

### 定点 CMVN 实现

```cpp
class CMVNQ15 {
public:
    CMVNQ15(const std::vector<float>& means, 
            const std::vector<float>& vars) {
        
        // 将 float32 的 mean 和 var 转换为 Q15
        means_q15_.resize(means.size());
        inv_stds_q15_.resize(vars.size());
        
        for (size_t i = 0; i < means.size(); i++) {
            // Mean 转换为 Q15
            means_q15_[i] = float_to_q15(means[i]);
            
            // 1/std 转换为 Q15（预计算倒数）
            float std = sqrtf(vars[i] + 1e-10f);
            inv_stds_q15_[i] = float_to_q15(1.0f / std);
        }
    }
    
    // 应用 CMVN: (x - mean) / std
    void apply(std::vector<int16_t>& frame) {
        for (size_t i = 0; i < frame.size(); i++) {
            // (x - mean)
            int32_t centered = (int32_t)frame[i] - means_q15_[i];
            
            // (x - mean) * (1/std)
            int32_t normalized = (centered * inv_stds_q15_[i]) >> 15;
            
            // 饱和
            frame[i] = saturate_int16(normalized);
        }
    }
    
private:
    std::vector<int16_t> means_q15_;     // Q15
    std::vector<int16_t> inv_stds_q15_;  // Q15
    
    int16_t float_to_q15(float x) {
        int32_t q15 = (int32_t)(x * 32768.0f);
        return saturate_int16(q15);
    }
    
    int16_t saturate_int16(int32_t value) {
        if (value > 32767) return 32767;
        if (value < -32768) return -32768;
        return (int16_t)value;
    }
};
```

### CMVN 数据预处理

```cpp
// 从 cmvn_data.h 读取并转换
void load_cmvn_q15(std::vector<int16_t>& means_q15,
                   std::vector<int16_t>& inv_stds_q15) {
    
    #ifdef USE_CMVN_HEADER
    // 从 cmvn_data.h 读取 float32 数据
    extern const float cmvn_means[];
    extern const float cmvn_vars[];
    extern const int cmvn_size;
    
    means_q15.resize(cmvn_size);
    inv_stds_q15.resize(cmvn_size);
    
    for (int i = 0; i < cmvn_size; i++) {
        // Mean
        means_q15[i] = (int16_t)(cmvn_means[i] * 32768.0f);
        
        // 1/std
        float std = sqrtf(cmvn_vars[i] + 1e-10f);
        inv_stds_q15[i] = (int16_t)((1.0f / std) * 32768.0f);
    }
    #endif
}
```

## 3. 完整的定点 FBank 流程

```cpp
class StreamingFBankExtractorQ15 {
public:
    StreamingFBankExtractorQ15(const FBankConfig& config) 
        : config_(config) {
        
        // 初始化各个组件
        fft_ = std::make_unique<FixedPointFFT>(512);
        mel_filter_ = std::make_unique<MelFilterBankQ15>(
            512, config.n_mels, config.fs);
        log_table_ = std::make_unique<LogLookupTableInterpolated>();
        lfr_ = std::make_unique<LFRQ15>(config.lfr_m, config.lfr_n);
        
        // 加载 CMVN
        load_cmvn_q15(cmvn_means_, cmvn_inv_stds_);
        cmvn_ = std::make_unique<CMVNQ15>(cmvn_means_, cmvn_inv_stds_);
    }
    
    // 处理音频块（输入已经是 int16）
    int process_chunk(const int16_t* audio_chunk,
                      size_t chunk_length,
                      std::vector<std::vector<int16_t>>& output_frames) {
        
        // 1. 添加到缓冲区（无需归一化，保持 int16）
        audio_buffer_.insert(audio_buffer_.end(),
                            audio_chunk,
                            audio_chunk + chunk_length);
        
        // 2. 提取帧并处理
        while (audio_buffer_.size() >= frame_length_) {
            // 提取一帧
            std::vector<int16_t> frame(
                audio_buffer_.begin(),
                audio_buffer_.begin() + frame_length_);
            
            // 应用窗函数（Q15）
            apply_window_q15(frame);
            
            // FFT（Q15）
            std::vector<int16_t> fft_real(fft_size_);
            std::vector<int16_t> fft_imag(fft_size_);
            fft_->compute(frame.data(), fft_real.data(), fft_imag.data());
            
            // 功率谱（Q15）
            std::vector<int16_t> power_spectrum(fft_size_ / 2 + 1);
            compute_power_spectrum_q15(fft_real.data(), fft_imag.data(),
                                      power_spectrum.data(), fft_size_);
            
            // Mel 滤波（Q15）
            std::vector<int16_t> mel_energies(config_.n_mels);
            mel_filter_->apply(power_spectrum.data(), mel_energies.data());
            
            // 对数（Q15）
            for (auto& energy : mel_energies) {
                energy = log_table_->compute(energy);
            }
            
            // 存储到帧缓冲
            fbank_frames_.push_back(mel_energies);
            
            // 移动缓冲区
            audio_buffer_.erase(audio_buffer_.begin(),
                               audio_buffer_.begin() + frame_shift_);
        }
        
        // 3. LFR 处理
        std::vector<std::vector<int16_t>> lfr_frames;
        lfr_->process(fbank_frames_, lfr_frames);
        
        // 4. CMVN 归一化
        for (auto& frame : lfr_frames) {
            cmvn_->apply(frame);
        }
        
        // 5. 输出
        output_frames = lfr_frames;
        
        // 清理已处理的帧
        // ...
        
        return lfr_frames.size();
    }
    
private:
    FBankConfig config_;
    std::vector<int16_t> audio_buffer_;
    std::vector<std::vector<int16_t>> fbank_frames_;
    
    std::unique_ptr<FixedPointFFT> fft_;
    std::unique_ptr<MelFilterBankQ15> mel_filter_;
    std::unique_ptr<LogLookupTableInterpolated> log_table_;
    std::unique_ptr<LFRQ15> lfr_;
    std::unique_ptr<CMVNQ15> cmvn_;
    
    std::vector<int16_t> cmvn_means_;
    std::vector<int16_t> cmvn_inv_stds_;
    
    int frame_length_;
    int frame_shift_;
    int fft_size_;
};
```

