#ifndef STREAMING_FBANK_EXTRACTOR_H
#define STREAMING_FBANK_EXTRACTOR_H

#include <vector>
#include <string>
#include "../fbank_extractor/fbank_extractor.h"

/**
 * Streaming FBank Feature Extractor
 * 
 * Processes audio in 10ms chunks (160 samples at 16kHz)
 * Maintains internal buffer for frame extraction
 * Outputs LFR features when enough frames are accumulated
 */
class StreamingFBankExtractor {
public:
    explicit StreamingFBankExtractor(const FBankConfig& config);
    ~StreamingFBankExtractor();

    /**
     * Process 10ms audio chunk (160 samples at 16kHz)
     * @param audio_chunk: Audio samples (normalized to [-1, 1])
     * @param chunk_length: Number of samples (should be 160 for 10ms at 16kHz)
     * @param output_lfr_frames: Output LFR frames (if any ready)
     * @return: Number of LFR frames output (0 if not enough frames yet)
     */
    int process_chunk(
        const float* audio_chunk,
        size_t chunk_length,
        std::vector<std::vector<float>>& output_lfr_frames
    );

    /**
     * Flush remaining frames and output final LFR frames
     * @param output_lfr_frames: Output LFR frames
     * @return: Number of LFR frames output
     */
    int flush(std::vector<std::vector<float>>& output_lfr_frames);

    /**
     * Reset internal state (clear buffers)
     */
    void reset();

    /**
     * Get output feature dimension
     */
    int get_output_dim() const { return config_.n_mels * config_.lfr_m; }
    
    /**
     * Check if CMVN is loaded
     */
    bool has_cmvn() const { return has_cmvn_; }
    
    /**
     * Get CMVN means (for INT16 CMVN initialization)
     * @return CMVN means vector (empty if not loaded)
     */
    const std::vector<float>& get_cmvn_means() const {
        return has_cmvn_ && !cmvn_means_.empty() ? cmvn_means_[0] : empty_vector_;
    }
    
    /**
     * Get CMVN vars (for INT16 CMVN initialization)
     * @return CMVN vars vector (empty if not loaded)
     */
    const std::vector<float>& get_cmvn_vars() const {
        return has_cmvn_ && !cmvn_vars_.empty() ? cmvn_vars_[0] : empty_vector_;
    }

private:
    FBankConfig config_;
    
    // Empty vector for returning when CMVN not loaded
    static const std::vector<float> empty_vector_;
    
    // Internal buffer for audio samples
    std::vector<float> audio_buffer_;
    
    // Buffer for raw fbank frames (before LFR, after CMVN)
    std::vector<std::vector<float>> fbank_frame_buffer_;
    
    // Raw fbank extractor (without LFR, without CMVN)
    // This is used to extract raw fbank frames
    // CMVN will be applied AFTER LFR (matching batch processing order)
    std::unique_ptr<FBankExtractor> raw_fbank_extractor_;
    
    // CMVN data (loaded from config if available)
    // Applied after LFR to match batch processing
    std::vector<std::vector<float>> cmvn_means_;
    std::vector<std::vector<float>> cmvn_vars_;
    bool has_cmvn_;
    
    // Track if this is the first LFR output (for left padding)
    bool is_first_lfr_output_;
    
    // Track the global LFR output index (across all calls)
    // This is needed to calculate which frames to keep for overlap
    int global_lfr_output_idx_;
    
    // Track the starting raw frame index of the buffer (relative to batch processing)
    // This is needed to correctly map buffer indices to batch processing indices
    // Example: If buffer_start_raw_idx_ = 25, then buffer[0] corresponds to raw[25] in batch processing
    int buffer_start_raw_idx_;
    
    // Helper: Extract fbank frames from buffered audio
    void extract_fbank_frames_from_buffer();
    
    // Helper: Apply LFR to buffered fbank frames and output ready frames
    int apply_lfr_and_output(std::vector<std::vector<float>>& output_lfr_frames);
    
    // Helper: Load CMVN from header file (if USE_CMVN_HEADER is defined)
    void load_cmvn_from_header();
    
    // Helper: Load CMVN from file
    void load_cmvn(const std::string& cmvn_file);
    
    // Helper: Apply CMVN to LFR frames (after LFR, matching batch processing)
    void apply_cmvn_to_lfr_frames(std::vector<std::vector<float>>& lfr_frames);
};

#endif // STREAMING_FBANK_EXTRACTOR_H
