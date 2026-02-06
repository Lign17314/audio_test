#include "fbank_extractor.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <limits>

// Use KissFFT from tflite-micro
#include "signal/src/kiss_fft_wrappers/kiss_fft_float.h"

FBankExtractor::FBankExtractor(const FBankConfig& config)
    : config_(config), has_cmvn_(false), fft_plan_(nullptr), fft_size_(0) {
    // Calculate FFT size (next power of 2 >= frame_length_samples)
    int frame_length_samples = (config_.frame_length * config_.fs) / 1000;
    fft_size_ = 1;
    while (fft_size_ < frame_length_samples) {
        fft_size_ <<= 1;
    }
    
    // Initialize KissFFT plan
    size_t state_size = 0;
    kiss_fft_float::kiss_fftr_alloc(fft_size_, 0, nullptr, &state_size);
    if (state_size > 0) {
        fft_plan_ = malloc(state_size);
        if (fft_plan_) {
            void* result = kiss_fft_float::kiss_fftr_alloc(fft_size_, 0, fft_plan_, &state_size);
            if (result != fft_plan_) {
                std::cerr << "Warning: KissFFT allocation returned different pointer" << std::endl;
                free(fft_plan_);
                fft_plan_ = nullptr;
            } else {
                std::cerr << "KissFFT initialized: fft_size=" << fft_size_ << ", state_size=" << state_size << std::endl;
            }
        } else {
            std::cerr << "Warning: Failed to allocate memory for KissFFT plan" << std::endl;
        }
    } else {
        std::cerr << "Warning: KissFFT state_size is 0" << std::endl;
    }
    
    // Load CMVN if provided
    if (!config_.cmvn_file.empty()) {
        load_cmvn(config_.cmvn_file);
    }
}

FBankExtractor::~FBankExtractor() {
    // Cleanup KissFFT plan
    if (fft_plan_) {
        free(fft_plan_);
        fft_plan_ = nullptr;
    }
}

void FBankExtractor::load_cmvn(const std::string& cmvn_file) {
    std::ifstream file(cmvn_file);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open CMVN file: " << cmvn_file << std::endl;
        return;
    }
    
    std::string line;
    std::vector<float> means_list;
    std::vector<float> vars_list;
    bool reading_means = false;
    bool reading_vars = false;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        
        if (token == "<AddShift>") {
            reading_means = true;
            reading_vars = false;
            continue;
        } else if (token == "<Rescale>") {
            reading_means = false;
            reading_vars = true;
            continue;
        } else if (token == "<LearnRateCoef>" && (reading_means || reading_vars)) {
            // Values are in the SAME line as <LearnRateCoef>
            // Format: <LearnRateCoef> 0 [ value1 value2 ... ]
            // Python code: line_item[3 : (len(line_item) - 1)]  (skip first 3, skip last)
            // Match Python implementation: read from current line
            
            std::istringstream iss(line);
            std::string val_token;
            
            // Skip first 3 tokens: "<LearnRateCoef>", "0", "["
            for (int j = 0; j < 3 && iss >> val_token; j++) {
                // Skip
            }
            
            // Read values until closing bracket "]"
            float value;
            while (iss >> value) {
                if (reading_means) {
                    means_list.push_back(value);
                } else if (reading_vars) {
                    vars_list.push_back(value);
                }
            }
            
            // Check if closing bracket is in the line (to stop reading)
            if (line.find("]") != std::string::npos) {
                reading_means = false;
                reading_vars = false;
            }
        }
    }
    
    if (!means_list.empty() && !vars_list.empty()) {
        // Reshape: CMVN file has 400 values (80 mels * 5 lfr_m), but we need to handle it
        // The means/vars are for the LFR-spliced features (400 dim)
        int feature_dim = config_.n_mels * config_.lfr_m;
        
        cmvn_means_.resize(1);
        cmvn_vars_.resize(1);
        cmvn_means_[0].resize(feature_dim);
        cmvn_vars_[0].resize(feature_dim);
        
        // Copy values (assuming they match the feature dimension)
        int copy_size = std::min(feature_dim, static_cast<int>(means_list.size()));
        std::copy(means_list.begin(), means_list.begin() + copy_size, cmvn_means_[0].begin());
        std::copy(vars_list.begin(), vars_list.begin() + copy_size, cmvn_vars_[0].begin());
        
        has_cmvn_ = true;
        std::cout << "Loaded CMVN: " << feature_dim << " dimensions" << std::endl;
    }
}

std::vector<std::vector<float>> FBankExtractor::extract(
    const float* waveform,
    size_t waveform_length) {
    
    // Step 1: Scale samples (upsacle_samples)
    std::vector<float> scaled_waveform(waveform_length);
    if (config_.upsacle_samples) {
        for (size_t i = 0; i < waveform_length; ++i) {
            scaled_waveform[i] = waveform[i] * 32768.0f;  // (1 << 15)
        }
    } else {
        std::copy(waveform, waveform + waveform_length, scaled_waveform.begin());
    }
    
    // Step 2: Compute fbank features
    std::vector<std::vector<float>> fbank_features = compute_fbank(
        scaled_waveform.data(),
        waveform_length
    );
    
    // Step 3: Apply LFR (Low Frame Rate) BEFORE CMVN
    // This matches Python implementation: LFR first, then CMVN
    if (config_.lfr_m != 1 || config_.lfr_n != 1) {
        fbank_features = apply_lfr(fbank_features);
    }
    
    // Step 4: Apply CMVN AFTER LFR (matching Python implementation)
    if (has_cmvn_) {
        apply_cmvn(fbank_features);
    }
    
    return fbank_features;
}

std::vector<std::vector<float>> FBankExtractor::compute_fbank(
    const float* waveform,
    size_t waveform_length) {
    
    int frame_length_samples = (config_.frame_length * config_.fs) / 1000;
    int frame_shift_samples = (config_.frame_shift * config_.fs) / 1000;
    
    // Create window
    std::vector<float> window = create_hamming_window(frame_length_samples);
    
    // Create mel filterbank
    std::vector<std::vector<float>> mel_filters = create_mel_filterbank();
    
    // Compute number of frames
    int num_frames;
    if (config_.snip_edges) {
        num_frames = (waveform_length - frame_length_samples) / frame_shift_samples + 1;
        if (num_frames < 0) num_frames = 0;
    } else {
        num_frames = (waveform_length + frame_shift_samples / 2) / frame_shift_samples;
    }
    
    std::vector<std::vector<float>> fbank_features;
    fbank_features.reserve(num_frames);
    
    for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        int start_sample = frame_idx * frame_shift_samples;
        
        // Extract frame
        std::vector<float> frame(frame_length_samples, 0.0f);
        int valid_samples = std::min(frame_length_samples, 
                                     static_cast<int>(waveform_length - start_sample));
        std::copy(waveform + start_sample, 
                 waveform + start_sample + valid_samples,
                 frame.begin());
        
        // Match torchaudio's processing order:
        // 1. dither (first)
        if (config_.dither > 0.0f) {
            // Simple dithering: add small random noise
            for (int i = 0; i < frame_length_samples; ++i) {
                float dither_val = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * config_.dither;
                frame[i] += dither_val;
            }
        }
        
        // 2. remove_dc_offset: subtract mean from each frame
        float frame_mean = 0.0f;
        for (int i = 0; i < frame_length_samples; ++i) {
            frame_mean += frame[i];
        }
        frame_mean /= frame_length_samples;
        for (int i = 0; i < frame_length_samples; ++i) {
            frame[i] -= frame_mean;
        }
        
        // 3. preemphasis: Match torchaudio's implementation
        // torchaudio: strided_input[i,j] -= preemphasis_coefficient * offset_strided_input[i, j]
        // where offset_strided_input = pad(strided_input, (1,0), mode="replicate")
        // offset_strided_input[:, :-1] means offset_strided_input[:, 0:window_size]
        // For j=0: frame[0] -= 0.97 * offset_strided_input[0,0] = frame[0] - 0.97 * frame[0] = frame[0] * 0.03
        // For j>0: frame[j] -= 0.97 * offset_strided_input[j] = frame[j] - 0.97 * frame[j-1]
        // Process from front to back, using previous value before modification
        float prev_val = frame[0];
        frame[0] = frame[0] * (1.0f - 0.97f);  // j=0: replicate mode
        for (int i = 1; i < frame_length_samples; ++i) {
            float current = frame[i];
            frame[i] = frame[i] - 0.97f * prev_val;  // Use previous value (before modification)
            prev_val = current;
        }
        
        // 4. Apply window
        apply_window(frame.data(), frame_length_samples, window);
        
        // Compute FFT magnitude
        std::vector<float> magnitude(fft_size_);
        compute_fft(frame.data(), frame_length_samples, magnitude);
        
        // Apply mel filterbank
        std::vector<float> mel_energies(config_.n_mels);
        apply_mel_filterbank(magnitude, mel_filters, mel_energies);
        
        // Apply log (log10)
        apply_log(mel_energies);
        
        fbank_features.push_back(mel_energies);
    }
    
    return fbank_features;
}

std::vector<std::vector<float>> FBankExtractor::apply_lfr(
    const std::vector<std::vector<float>>& fbank_features) {
    
    int T = static_cast<int>(fbank_features.size());
    int T_lfr = static_cast<int>(std::ceil(static_cast<float>(T) / config_.lfr_n));
    
    // Left padding: repeat first frame (lfr_m - 1) / 2 times
    int left_padding = (config_.lfr_m - 1) / 2;
    std::vector<std::vector<float>> padded_features;
    padded_features.reserve(T + left_padding);
    
    // Add left padding
    for (int i = 0; i < left_padding; ++i) {
        padded_features.push_back(fbank_features[0]);
    }
    
    // Add original features
    padded_features.insert(padded_features.end(), 
                          fbank_features.begin(), 
                          fbank_features.end());
    
    T = static_cast<int>(padded_features.size());
    
    // Apply LFR splicing
    std::vector<std::vector<float>> lfr_features;
    lfr_features.reserve(T_lfr);
    
    for (int i = 0; i < T_lfr; ++i) {
        int start_idx = i * config_.lfr_n;
        
        if (start_idx + config_.lfr_m <= T) {
            // Normal case: enough frames available
            std::vector<float> spliced_frame;
            spliced_frame.reserve(config_.n_mels * config_.lfr_m);
            
            for (int j = 0; j < config_.lfr_m; ++j) {
                spliced_frame.insert(spliced_frame.end(),
                                    padded_features[start_idx + j].begin(),
                                    padded_features[start_idx + j].end());
            }
            
            lfr_features.push_back(spliced_frame);
        } else {
            // Last frame: pad with last available frame
            std::vector<float> spliced_frame;
            spliced_frame.reserve(config_.n_mels * config_.lfr_m);
            
            // Copy available frames
            for (int j = start_idx; j < T; ++j) {
                spliced_frame.insert(spliced_frame.end(),
                                    padded_features[j].begin(),
                                    padded_features[j].end());
            }
            
            // Pad with last frame
            int num_padding = config_.lfr_m - (T - start_idx);
            for (int j = 0; j < num_padding; ++j) {
                spliced_frame.insert(spliced_frame.end(),
                                    padded_features[T - 1].begin(),
                                    padded_features[T - 1].end());
            }
            
            lfr_features.push_back(spliced_frame);
        }
    }
    
    return lfr_features;
}

void FBankExtractor::apply_cmvn(
    std::vector<std::vector<float>>& features) {
    
    if (!has_cmvn_ || features.empty()) {
        return;
    }
    
    int feature_dim = static_cast<int>(features[0].size());
    int cmvn_dim = static_cast<int>(cmvn_means_[0].size());
    int dim = std::min(feature_dim, cmvn_dim);
    
    for (auto& frame : features) {
        for (int i = 0; i < dim; ++i) {
            // Apply CMVN: (x + mean) * var
            frame[i] = (frame[i] + cmvn_means_[0][i]) * cmvn_vars_[0][i];
        }
    }
}

std::vector<float> FBankExtractor::create_hamming_window(int window_size) {
    std::vector<float> window(window_size);
    for (int i = 0; i < window_size; ++i) {
        window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (window_size - 1));
    }
    return window;
}

std::vector<std::vector<float>> FBankExtractor::create_mel_filterbank() {
    // Create mel filterbank (80 filters for 80 mel bins)
    // This is a simplified version - in practice, you'd use proper mel scale conversion
    std::vector<std::vector<float>> mel_filters(config_.n_mels);
    
    int fft_bins = fft_size_ / 2 + 1;
    float nyquist = config_.fs / 2.0f;
    float mel_nyquist = 2595.0f * std::log10(1.0f + nyquist / 700.0f);
    
    // Create triangular mel filters
    for (int i = 0; i < config_.n_mels; ++i) {
        mel_filters[i].resize(fft_bins, 0.0f);
        
        float mel_low = (mel_nyquist / (config_.n_mels + 1)) * i;
        float mel_center = (mel_nyquist / (config_.n_mels + 1)) * (i + 1);
        float mel_high = (mel_nyquist / (config_.n_mels + 1)) * (i + 2);
        
        float hz_low = 700.0f * (std::pow(10.0f, mel_low / 2595.0f) - 1.0f);
        float hz_center = 700.0f * (std::pow(10.0f, mel_center / 2595.0f) - 1.0f);
        float hz_high = 700.0f * (std::pow(10.0f, mel_high / 2595.0f) - 1.0f);
        
        int bin_low = static_cast<int>(hz_low * fft_size_ / config_.fs);
        int bin_center = static_cast<int>(hz_center * fft_size_ / config_.fs);
        int bin_high = static_cast<int>(hz_high * fft_size_ / config_.fs);
        
        bin_low = std::max(0, std::min(bin_low, fft_bins - 1));
        bin_center = std::max(0, std::min(bin_center, fft_bins - 1));
        bin_high = std::max(0, std::min(bin_high, fft_bins - 1));
        
        // Rising edge
        if (bin_center > bin_low) {
            for (int j = bin_low; j <= bin_center; ++j) {
                mel_filters[i][j] = static_cast<float>(j - bin_low) / (bin_center - bin_low);
            }
        }
        
        // Falling edge
        if (bin_high > bin_center) {
            for (int j = bin_center; j <= bin_high; ++j) {
                mel_filters[i][j] = static_cast<float>(bin_high - j) / (bin_high - bin_center);
            }
        }
    }
    
    return mel_filters;
}

void FBankExtractor::apply_window(
    float* frame,
    int frame_size,
    const std::vector<float>& window) {
    for (int i = 0; i < frame_size; ++i) {
        frame[i] *= window[i];
    }
}

void FBankExtractor::compute_fft(
    const float* frame,
    int frame_size,
    std::vector<float>& magnitude) {
    
    if (!fft_plan_) {
        // Fallback to DFT if KissFFT not initialized
        static bool warned_once = false;
        if (!warned_once) {
            std::cerr << "Warning: Using DFT fallback (KissFFT not initialized)" << std::endl;
            warned_once = true;
        }
        magnitude.assign(fft_size_, 0.0f);
        for (int k = 0; k < fft_size_ / 2 + 1; ++k) {
            float real = 0.0f;
            float imag = 0.0f;
            for (int n = 0; n < frame_size; ++n) {
                float angle = -2.0f * M_PI * k * n / fft_size_;
                real += frame[n] * std::cos(angle);
                imag += frame[n] * std::sin(angle);
            }
            float mag = std::sqrt(real * real + imag * imag);
            magnitude[k] = mag * mag;  // Convert to power (magnitude^2)
        }
        for (int k = fft_size_ / 2 + 1; k < fft_size_; ++k) {
            magnitude[k] = magnitude[fft_size_ - k];
        }
        return;
    }
    
    // Prepare input: pad frame to fft_size with zeros
    std::vector<float> fft_input(fft_size_, 0.0f);
    for (int i = 0; i < frame_size; ++i) {
        fft_input[i] = frame[i];
    }
    
    // Allocate output buffer for complex FFT result
    std::vector<kiss_fft_float::kiss_fft_cpx> fft_output(fft_size_ / 2 + 1);
    
    // Perform FFT using KissFFT
    kiss_fft_float::kiss_fftr(
        static_cast<kiss_fft_float::kiss_fftr_cfg>(fft_plan_),
        fft_input.data(),
        fft_output.data());
    
    // Extract magnitude from complex output, then convert to power (magnitude^2)
    // Matching torchaudio: use_power=True -> spectrum.pow(2.0)
    magnitude.assign(fft_size_, 0.0f);
    for (int k = 0; k < fft_size_ / 2 + 1; ++k) {
        float real = fft_output[k].r;
        float imag = fft_output[k].i;
        float mag = std::sqrt(real * real + imag * imag);
        magnitude[k] = mag * mag;  // Convert to power (magnitude^2)
    }
    
    // Mirror the spectrum for full fft_size
    for (int k = fft_size_ / 2 + 1; k < fft_size_; ++k) {
        magnitude[k] = magnitude[fft_size_ - k];
    }
}

void FBankExtractor::apply_mel_filterbank(
    const std::vector<float>& magnitude,
    const std::vector<std::vector<float>>& mel_filters,
    std::vector<float>& mel_energies) {
    
    for (int i = 0; i < config_.n_mels; ++i) {
        mel_energies[i] = 0.0f;
        for (size_t j = 0; j < magnitude.size() && j < mel_filters[i].size(); ++j) {
            mel_energies[i] += magnitude[j] * mel_filters[i][j];
        }
    }
}

void FBankExtractor::apply_log(std::vector<float>& mel_energies) {
    // Match torchaudio: use log() (natural logarithm) instead of log10()
    // torchaudio: mel_energies = torch.max(mel_energies, epsilon).log()
    const float energy_floor = std::numeric_limits<float>::epsilon();  // Match torchaudio's epsilon
    for (float& energy : mel_energies) {
        energy = std::max(energy, energy_floor);
        energy = std::log(energy);  // Use natural logarithm (log) instead of log10
    }
}

std::vector<std::vector<float>> FBankExtractor::extract_from_file(const std::string& wav_file) {
    // TODO: Implement WAV file reading
    // For now, return empty vector
    std::cerr << "extract_from_file not yet implemented" << std::endl;
    return {};
}
