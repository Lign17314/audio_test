#ifndef FBANK_EXTRACTOR_H
#define FBANK_EXTRACTOR_H

#include <vector>
#include <string>
#include <memory>

/**
 * FBank Feature Extractor (C++ implementation matching Python wav_frontend.py)
 * 
 * Configuration (matching a2.py WavFrontend_config):
 *   - fs: 16000 Hz
 *   - window: "hamming"
 *   - n_mels: 80
 *   - frame_length: 25 ms
 *   - frame_shift: 10 ms
 *   - lfr_m: 5 (splice 5 frames)
 *   - lfr_n: 3 (take every 3rd frame)
 *   - cmvn_file: path to CMVN file
 */

struct FBankConfig {
    int fs = 16000;                    // Sample rate
    std::string window = "hamming";    // Window type
    int n_mels = 80;                   // Number of mel bins
    int frame_length = 25;             // Frame length in ms
    int frame_shift = 10;              // Frame shift in ms
    int lfr_m = 5;                     // LFR splice frames
    int lfr_n = 3;                     // LFR skip frames
    float dither = 1.0f;                // Dithering factor
    bool snip_edges = true;            // Snip edges
    bool upsacle_samples = true;       // Scale samples by 32768
    std::string cmvn_file;             // CMVN file path (optional)
};

class FBankExtractor {
public:
    explicit FBankExtractor(const FBankConfig& config);
    ~FBankExtractor();

    /**
     * Extract fbank features from audio waveform
     * @param waveform: Input audio samples (normalized to [-1, 1])
     * @param waveform_length: Number of samples
     * @return: Feature matrix (num_frames, feature_dim)
     *          feature_dim = n_mels * lfr_m = 80 * 5 = 400
     */
    std::vector<std::vector<float>> extract(
        const float* waveform,
        size_t waveform_length
    );

    /**
     * Extract fbank features from WAV file
     * @param wav_file: Path to WAV file
     * @return: Feature matrix (num_frames, feature_dim)
     */
    std::vector<std::vector<float>> extract_from_file(const std::string& wav_file);

    /**
     * Get output feature dimension
     */
    int get_output_dim() const { return config_.n_mels * config_.lfr_m; }

private:
    FBankConfig config_;
    
    // CMVN data: [means, vars] shape (2, feature_dim)
    std::vector<std::vector<float>> cmvn_means_;
    std::vector<std::vector<float>> cmvn_vars_;
    bool has_cmvn_;

    // Internal processing functions
    std::vector<std::vector<float>> compute_fbank(
        const float* waveform,
        size_t waveform_length
    );
    
    std::vector<std::vector<float>> apply_lfr(
        const std::vector<std::vector<float>>& fbank_features
    );
    
    void apply_cmvn(
        std::vector<std::vector<float>>& features
    );
    
    void load_cmvn(const std::string& cmvn_file);
    
    // Helper functions
    std::vector<float> create_hamming_window(int window_size);
    std::vector<std::vector<float>> create_mel_filterbank();
    void apply_window(float* frame, int frame_size, const std::vector<float>& window);
    void compute_fft(const float* frame, int frame_size, std::vector<float>& magnitude);
    void apply_mel_filterbank(
        const std::vector<float>& magnitude,
        const std::vector<std::vector<float>>& mel_filters,
        std::vector<float>& mel_energies
    );
    void apply_log(std::vector<float>& mel_energies);
    
    // FFT-related (using KissFFT or similar)
    void* fft_plan_;
    int fft_size_;
};

#endif // FBANK_EXTRACTOR_H
