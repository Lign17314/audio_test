#include "streaming_fbank_extractor.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>

// Include CMVN data header if available
#ifdef USE_CMVN_HEADER
#include "cmvn_data.h"
#endif

// Define static member
const std::vector<float> StreamingFBankExtractor::empty_vector_;

StreamingFBankExtractor::StreamingFBankExtractor(const FBankConfig& config)
    : config_(config), has_cmvn_(false), is_first_lfr_output_(true), global_lfr_output_idx_(0), buffer_start_raw_idx_(0) {
    // Create raw fbank extractor (without LFR, without CMVN) for extracting raw frames
    // CMVN will be applied AFTER LFR (matching batch processing order: LFR -> CMVN)
    FBankConfig raw_config = config;
    raw_config.lfr_m = 1;
    raw_config.lfr_n = 1;
    raw_config.cmvn_file = "";  // Don't apply CMVN in raw extractor
    raw_fbank_extractor_ = std::make_unique<FBankExtractor>(raw_config);
    
    // Load CMVN separately (will be applied after LFR)
    // Always use header file if available (cmvn_data.h), fall back to file loading if needed
#ifdef USE_CMVN_HEADER
    // Always try to load from header file first
    load_cmvn_from_header();
    // If header loading failed and cmvn_file is specified, try file loading
    if (!has_cmvn_ && !config_.cmvn_file.empty()) {
        load_cmvn(config_.cmvn_file);
    }
#else
    // Header file not available, use file loading
    if (!config_.cmvn_file.empty()) {
        load_cmvn(config_.cmvn_file);
    }
#endif
    
    reset();
}

StreamingFBankExtractor::~StreamingFBankExtractor() {
    // Cleanup handled by member destructors
}

int StreamingFBankExtractor::process_chunk(
    const float* audio_chunk,
    size_t chunk_length,
    std::vector<std::vector<float>>& output_lfr_frames) {
    
    // Append audio chunk to buffer
    audio_buffer_.insert(audio_buffer_.end(), audio_chunk, audio_chunk + chunk_length);
    
    // Extract fbank frames from buffered audio
    extract_fbank_frames_from_buffer();
    
    // Apply LFR and output ready frames
    return apply_lfr_and_output(output_lfr_frames);
}

void StreamingFBankExtractor::extract_fbank_frames_from_buffer() {
    int frame_length_samples = (config_.frame_length * config_.fs) / 1000;  // 400 samples for 25ms
    int frame_shift_samples = (config_.frame_shift * config_.fs) / 1000;     // 160 samples for 10ms
    
    // For streaming fbank extraction, we need to:
    // 1. Accumulate audio samples
    // 2. Extract fbank frames when we have enough samples
    // 3. The challenge is that FBankExtractor::extract applies LFR automatically
    // 
    // Solution: We'll accumulate audio and extract when buffer is large enough
    // We'll extract raw fbank frames by using FBankExtractor on chunks
    // but we need to handle LFR separately
    
    // Minimum samples needed: frame_length_samples for one frame
    // But for proper extraction, we want multiple frames
    // Let's extract when we have enough for at least 3 frames (for LFR)
    int min_samples_needed = frame_length_samples + 2 * frame_shift_samples;  // ~720 samples
    
    while (static_cast<int>(audio_buffer_.size()) >= min_samples_needed) {
        // Extract a chunk that's large enough for multiple frames
        // We'll extract enough for at least 3 raw fbank frames (for one LFR frame)
        size_t extract_size = std::min(static_cast<size_t>(frame_length_samples + 10 * frame_shift_samples),
                                       audio_buffer_.size());
        
        // Extract chunk
        std::vector<float> chunk(audio_buffer_.begin(),
                                 audio_buffer_.begin() + extract_size);
        
        // Extract raw fbank frames (without LFR, without CMVN)
        // The raw extractor has lfr_m=1, lfr_n=1, cmvn_file=""
        // So it extracts raw fbank frames only
        // Then we'll apply LFR, then CMVN (matching batch processing order)
        auto raw_fbank_frames = raw_fbank_extractor_->extract(chunk.data(), chunk.size());
        
        // Add to buffer
        fbank_frame_buffer_.insert(fbank_frame_buffer_.end(),
                                  raw_fbank_frames.begin(),
                                  raw_fbank_frames.end());
        
        // Shift buffer by the number of samples we've processed
        // We've processed extract_size samples, but we should shift by frame_shift_samples
        // for each frame extracted
        int num_frames_extracted = static_cast<int>(raw_fbank_frames.size());
        int samples_to_shift = num_frames_extracted * frame_shift_samples;
        
        if (static_cast<int>(audio_buffer_.size()) >= samples_to_shift) {
            audio_buffer_.erase(audio_buffer_.begin(), 
                               audio_buffer_.begin() + samples_to_shift);
        } else {
            // Shift remaining samples
            audio_buffer_.clear();
        }
    }
}

int StreamingFBankExtractor::apply_lfr_and_output(
    std::vector<std::vector<float>>& output_lfr_frames) {
    
    output_lfr_frames.clear();
    
    int lfr_n = config_.lfr_n;  // 3
    int lfr_m = config_.lfr_m;  // 5
    int left_padding = (lfr_m - 1) / 2;  // 2
    
    // We need at least lfr_n raw fbank frames to output one LFR frame
    int num_raw_frames = static_cast<int>(fbank_frame_buffer_.size());
    if (num_raw_frames < lfr_n) {
        return 0;
    }
    
    // Apply LFR processing matching batch processing exactly
    // Key insight: In batch processing, LFR output i uses frames starting at index i*lfr_n
    // in padded_features (which includes left padding).
    // In streaming, we need to simulate this by:
    // 1. First call: Add left padding, then process normally
    // 2. Subsequent calls: Buffer already contains overlap frames, process from buffer start
    
    std::vector<std::vector<float>> padded_features;
    
    // IMPORTANT: Only add left padding when we're actually going to output the FIRST LFR frame
    // Use global_lfr_output_idx_ to determine if this is the first LFR output
    // global_lfr_output_idx_ == 0 means we haven't output any LFR frames yet
    bool is_first_output = (global_lfr_output_idx_ == 0);
    
    if (is_first_output && !fbank_frame_buffer_.empty()) {
        // First LFR output: Add left padding (repeat first frame)
        // This matches batch processing: padded = [pad0, pad1, frame0, frame1, ...]
        padded_features.reserve(num_raw_frames + left_padding);
        for (int i = 0; i < left_padding; ++i) {
            padded_features.push_back(fbank_frame_buffer_[0]);
        }
    } else {
        // Subsequent LFR outputs: No left padding needed
        // Buffer already contains overlap frames from previous call
        padded_features.reserve(num_raw_frames);
    }
    
    // Add buffer frames to padded_features
    padded_features.insert(padded_features.end(),
                          fbank_frame_buffer_.begin(),
                          fbank_frame_buffer_.end());
    
    // Calculate how many LFR frames we can output
    // IMPORTANT: Use padded_features.size() for T_lfr calculation (matching batch processing)
    // Batch processing: T_lfr = ceil(T / lfr_n) where T is original frames (before padding)
    // But we need to account for padding in the loop condition
    int T = static_cast<int>(padded_features.size());
    // Use original num_raw_frames for T_lfr calculation (matching batch processing)
    // Batch: T_lfr = ceil(T_raw / lfr_n), then uses padded_features[i*lfr_n : i*lfr_n + lfr_m]
    int T_lfr = static_cast<int>(std::ceil(static_cast<float>(num_raw_frames) / lfr_n));
    
    int num_output_lfr = 0;
    
    // Generate LFR outputs matching batch processing
    // In batch: LFR output i uses padded_features[i*lfr_n : i*lfr_n + lfr_m]
    // In streaming: 
    //   - First call: Add left padding, output from i=0
    //   - Subsequent calls: Buffer has overlap, skip i=0 to avoid duplication
    int start_i = (global_lfr_output_idx_ > 0) ? 1 : 0;
    
    for (int i = start_i; i < T_lfr; ++i) {
        // IMPORTANT: Calculate start_idx considering:
        // 1. Batch processing: LFR[i] uses padded[i*lfr_n : i*lfr_n + lfr_m] where padded includes left_padding
        // 2. Streaming: After first call, padded does NOT include left_padding
        // 3. Buffer starting position: buffer[0] corresponds to raw[buffer_start_raw_idx_] in batch processing
        
        // Calculate what raw frame index this LFR output should use in batch processing
        int target_global_lfr_idx = global_lfr_output_idx_ + (i - start_i);
        
        // For batch processing:
        // - LFR[0]: padded[0:5] -> raw[0:3] (padded[0:1] are padding, padded[2:5] are raw[0:3])
        // - LFR[1]: padded[3:8] -> raw[1:6] (padded[3:4] are padding+raw[0:1], padded[5:8] are raw[2:5])
        // So: target_raw_start_batch = target_global_lfr_idx * lfr_n - left_padding
        // But for LFR[0], this gives -2, which means we need to use left padding
        int target_raw_start_batch = target_global_lfr_idx * lfr_n - left_padding;
        
        // For the first LFR output (target_global_lfr_idx == 0), we have left padding in padded_features
        // So the mapping is different
        int start_idx;
        if (target_global_lfr_idx == 0) {
            // First LFR output: padded includes left_padding
            // padded[0:5] -> raw[0:3], so start_idx = 0
            start_idx = 0;
        } else {
            // Subsequent LFR outputs: no left padding in padded_features
            // Calculate start_idx: buffer_start_raw_idx_ + start_idx = target_raw_start_batch
            start_idx = target_raw_start_batch - buffer_start_raw_idx_;
            
            // Ensure start_idx is valid
            if (start_idx < 0) {
                std::cerr << "ERROR: start_idx < 0: " << start_idx 
                          << " (target_raw_start_batch=" << target_raw_start_batch 
                          << ", buffer_start_raw_idx_=" << buffer_start_raw_idx_ << ")" << std::endl;
                start_idx = 0;
            }
        }
        
        // Check if we have enough frames for this LFR output
        if (start_idx + lfr_m <= T) {
            // Extract LFR frame
            std::vector<float> lfr_frame;
            lfr_frame.reserve(config_.n_mels * lfr_m);
            
            for (int j = 0; j < lfr_m; ++j) {
                lfr_frame.insert(lfr_frame.end(),
                                padded_features[start_idx + j].begin(),
                                padded_features[start_idx + j].end());
            }
            
            output_lfr_frames.push_back(lfr_frame);
            num_output_lfr++;
            global_lfr_output_idx_++;
        } else {
            // Not enough frames for this LFR output, break
            break;
        }
    }
    
    // Apply CMVN AFTER LFR (matching batch processing order: LFR -> CMVN)
    if (!output_lfr_frames.empty() && has_cmvn_) {
        apply_cmvn_to_lfr_frames(output_lfr_frames);
    }
    
    // Remove processed raw frames from buffer
    // Keep frames needed for future LFR outputs (overlap)
    if (num_output_lfr > 0) {
        // Calculate which raw frames were actually used
        // Key: We need to track which raw frames (not padded) were consumed
        
        // IMPORTANT: Buffer management strategy
        // The last LFR output uses raw frames [last_start : last_start + lfr_m]
        // We need to keep these lfr_m frames for the next call
        // This ensures that the next call's first LFR (with start_i=1) can correctly
        // align with batch processing's corresponding LFR output
        
        // Example: First call outputs LFR[0] to LFR[9]
        //   LFR[9] uses padded[27:32] -> raw[25:30] (5 frames)
        //   Keep raw[25:30] (lfr_m = 5 frames)
        // Second call needs to output LFR[10] (with start_i=1, it's LFR[1] in the call)
        //   LFR[10] needs raw[28:33]
        //   buffer[3] = raw[28], buffer[8] = raw[33]
        //   LFR[1] uses buffer[3:8] -> raw[28:33], which is correct!
        
        int frames_to_keep = lfr_m;  // Keep lfr_m frames (not left_padding + lfr_m)
        
        // DEBUG: Calculate which frames were used
        // Find the actual start_idx used for the last LFR output
        int last_output_i = start_i + num_output_lfr - 1;
        int last_global_idx = global_lfr_output_idx_ - 1;  // Last output's global index
        
        // Calculate raw frame indices for last LFR output (in batch processing coordinates)
        int last_raw_start, last_raw_end, last_start_idx_debug;
        if (last_global_idx == 0) {
            // First LFR output: padded includes left_padding
            // padded[0:5] -> raw[0:3] (padded[0:1] are padding, padded[2:5] are raw[0:3])
            // Actually uses raw[0:3], so we need to keep raw[0:3] for next call
            // But LFR[1] needs raw[1:6], so we should keep raw[0:3] (all frames)
            last_raw_start = 0;
            last_start_idx_debug = 0;  // For debug output
            // For first call, we typically have 3 raw frames, so last_raw_end = 3
            // But we need to keep frames for LFR[1] which needs raw[1:6]
            // So we keep raw[0:3] (all frames), and buffer_start_raw_idx_ stays at 0
            last_raw_end = static_cast<int>(fbank_frame_buffer_.size());  // Use all available raw frames
        } else {
            // Subsequent LFR outputs: no left padding
            int last_target_raw_start_batch = last_global_idx * lfr_n - left_padding;
            last_start_idx_debug = last_target_raw_start_batch - buffer_start_raw_idx_;
            if (last_start_idx_debug < 0) last_start_idx_debug = 0;
            last_raw_start = buffer_start_raw_idx_ + last_start_idx_debug;
            last_raw_end = last_raw_start + lfr_m;
        }
        
        // Keep only the last frames_to_keep frames
        if (static_cast<int>(fbank_frame_buffer_.size()) > frames_to_keep) {
            std::vector<std::vector<float>> kept_frames(
                fbank_frame_buffer_.end() - frames_to_keep,
                fbank_frame_buffer_.end()
            );
            fbank_frame_buffer_ = kept_frames;
        }
        
        // Update buffer_start_raw_idx_ for next call
        buffer_start_raw_idx_ = last_raw_start;
    }
    
    return num_output_lfr;
}

int StreamingFBankExtractor::flush(std::vector<std::vector<float>>& output_lfr_frames) {
    // Process remaining audio
    if (!audio_buffer_.empty()) {
        // Extract remaining frames
        extract_fbank_frames_from_buffer();
    }
    
    // Output remaining LFR frames
    output_lfr_frames.clear();
    
    if (fbank_frame_buffer_.empty()) {
        return 0;
    }
    
    // Apply LFR to remaining frames
    int lfr_n = config_.lfr_n;
    int lfr_m = config_.lfr_m;
    int left_padding = (lfr_m - 1) / 2;
    
    // Create padded features
    std::vector<std::vector<float>> padded_features;
    padded_features.reserve(fbank_frame_buffer_.size() + left_padding);
    
    // Add left padding
    if (!fbank_frame_buffer_.empty()) {
        for (int i = 0; i < left_padding; ++i) {
            padded_features.push_back(fbank_frame_buffer_[0]);
        }
    }
    
    // Add original frames
    padded_features.insert(padded_features.end(),
                          fbank_frame_buffer_.begin(),
                          fbank_frame_buffer_.end());
    
    // Apply LFR (handle last frame padding)
    // IMPORTANT: Calculate T_lfr based on the actual number of raw frames (before padding)
    // This matches batch processing: T_lfr = ceil(T_raw / lfr_n)
    // where T_raw is the number of raw fbank frames (before left padding)
    int T = static_cast<int>(padded_features.size());
    int T_raw = static_cast<int>(fbank_frame_buffer_.size());
    
    // Calculate expected total LFR frames (matching batch processing)
    // In batch processing: T_lfr_total = ceil(T_total_raw / lfr_n)
    // where T_total_raw = buffer_start_raw_idx_ + T_raw (the total raw frames processed)
    // This ensures we output the same number of LFR frames as batch processing
    int T_total_raw = buffer_start_raw_idx_ + T_raw;
    int T_lfr_total_expected = static_cast<int>(std::ceil(static_cast<float>(T_total_raw) / lfr_n));
    int remaining_lfr_frames = T_lfr_total_expected - global_lfr_output_idx_;
    
    // Calculate how many LFR frames can be produced from the remaining buffer
    // Even if buffer has fewer than lfr_m frames, we can still output 1 frame with padding
    int T_lfr_from_buffer = (T_raw > 0) ? static_cast<int>(std::ceil(static_cast<float>(T_raw) / lfr_n)) : 0;
    
    // CRITICAL: Output the remaining frames we should output, but not more
    // Use remaining_lfr_frames as the target, but ensure we can actually produce that many
    // If remaining_lfr_frames > T_lfr_from_buffer, we need to output remaining_lfr_frames frames
    // (with padding for the last frame if needed)
    int T_lfr = remaining_lfr_frames;
    
    // Ensure T_lfr is non-negative and doesn't exceed what we can produce from buffer
    if (T_lfr < 0) {
        T_lfr = 0;
    }
    
    // If remaining_lfr_frames is 0 but we have frames, we might still need to output 1 frame
    // This handles edge cases where buffer management didn't perfectly align
    if (T_lfr == 0 && T_raw > 0 && remaining_lfr_frames == 0) {
        // Check if we should output one more frame based on buffer content
        // If buffer has enough frames for at least 1 LFR frame (with padding), output it
        if (T_raw >= 1) {
            T_lfr = 1;
        }
    }
    
    for (int i = 0; i < T_lfr; ++i) {
        int start_idx = i * lfr_n;
        
        std::vector<float> lfr_frame;
        lfr_frame.reserve(config_.n_mels * lfr_m);
        
        if (start_idx + lfr_m <= T) {
            // Normal case
            for (int j = 0; j < lfr_m; ++j) {
                lfr_frame.insert(lfr_frame.end(),
                                padded_features[start_idx + j].begin(),
                                padded_features[start_idx + j].end());
            }
        } else {
            // Last frame: pad with last available frame
            for (int j = start_idx; j < T; ++j) {
                lfr_frame.insert(lfr_frame.end(),
                                padded_features[j].begin(),
                                padded_features[j].end());
            }
            // Pad with last frame
            int num_padding = lfr_m - (T - start_idx);
            for (int j = 0; j < num_padding; ++j) {
                lfr_frame.insert(lfr_frame.end(),
                                padded_features[T - 1].begin(),
                                padded_features[T - 1].end());
            }
        }
        
        output_lfr_frames.push_back(lfr_frame);
        global_lfr_output_idx_++;
    }
    
    // Apply CMVN AFTER LFR (matching batch processing order: LFR -> CMVN)
    if (!output_lfr_frames.empty() && has_cmvn_) {
        apply_cmvn_to_lfr_frames(output_lfr_frames);
    }
    
    // Clear buffers
    fbank_frame_buffer_.clear();
    
    return static_cast<int>(output_lfr_frames.size());
}

void StreamingFBankExtractor::load_cmvn_from_header() {
#ifdef USE_CMVN_HEADER
    int feature_dim = config_.n_mels * config_.lfr_m;  // 80 * 5 = 400
    
    if (feature_dim != CMVN_DIM) {
        std::cerr << "Warning: CMVN dimension mismatch: expected " << feature_dim 
                  << ", got " << CMVN_DIM << ". CMVN will not be applied." << std::endl;
        has_cmvn_ = false;
        return;
    }
    
    cmvn_means_.resize(1);
    cmvn_vars_.resize(1);
    cmvn_means_[0].resize(feature_dim);
    cmvn_vars_[0].resize(feature_dim);
    
    std::copy(cmvn_means, cmvn_means + feature_dim, cmvn_means_[0].begin());
    std::copy(cmvn_vars, cmvn_vars + feature_dim, cmvn_vars_[0].begin());
    
    has_cmvn_ = true;
    std::cout << "Loaded CMVN from header (cmvn_data.h): " << feature_dim << " dimensions" << std::endl;
    std::cout << "  Means first 5: " 
              << cmvn_means_[0][0] << " " << cmvn_means_[0][1] << " " 
              << cmvn_means_[0][2] << " " << cmvn_means_[0][3] << " " 
              << cmvn_means_[0][4] << std::endl;
    std::cout << "  Vars first 5: " 
              << cmvn_vars_[0][0] << " " << cmvn_vars_[0][1] << " " 
              << cmvn_vars_[0][2] << " " << cmvn_vars_[0][3] << " " 
              << cmvn_vars_[0][4] << std::endl;
#else
    // Header file not available, mark as not loaded
    has_cmvn_ = false;
#endif
}

void StreamingFBankExtractor::load_cmvn(const std::string& cmvn_file) {
    // Load CMVN file (same format as FBankExtractor)
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
        int feature_dim = config_.n_mels * config_.lfr_m;  // 80 * 5 = 400
        cmvn_means_.resize(1);
        cmvn_vars_.resize(1);
        cmvn_means_[0].resize(feature_dim);
        cmvn_vars_[0].resize(feature_dim);
        
        int copy_size = std::min(feature_dim, static_cast<int>(means_list.size()));
        std::copy(means_list.begin(), means_list.begin() + copy_size, cmvn_means_[0].begin());
        std::copy(vars_list.begin(), vars_list.begin() + copy_size, cmvn_vars_[0].begin());
        
        has_cmvn_ = true;
        std::cout << "Loaded CMVN: " << feature_dim << " dimensions" << std::endl;
        std::cout << "  Means: " << means_list.size() << " values, first 5: " 
                  << means_list[0] << " " << means_list[1] << " " << means_list[2] << " " 
                  << means_list[3] << " " << means_list[4] << std::endl;
        std::cout << "  Vars: " << vars_list.size() << " values, first 5: " 
                  << vars_list[0] << " " << vars_list[1] << " " << vars_list[2] << " " 
                  << vars_list[3] << " " << vars_list[4] << std::endl;
    } else {
        std::cerr << "Warning: Failed to load CMVN data" << std::endl;
    }
}

void StreamingFBankExtractor::apply_cmvn_to_lfr_frames(
    std::vector<std::vector<float>>& lfr_frames) {
    if (!has_cmvn_ || lfr_frames.empty()) {
        if (!has_cmvn_) {
            std::cerr << "Warning: CMVN not loaded, skipping CMVN application" << std::endl;
        }
        return;
    }
    
    int feature_dim = config_.n_mels * config_.lfr_m;  // 400
    
    // Debug: print first frame before and after CMVN (only for first call)
    static bool first_call = true;
    if (first_call && !lfr_frames.empty()) {
        std::vector<float> before_cmvn = lfr_frames[0];
        
        std::cout << "Applying CMVN to LFR frames:" << std::endl;
        std::cout << "  First frame before CMVN (first 5): " 
                  << before_cmvn[0] << " " << before_cmvn[1] << " " << before_cmvn[2] << " " 
                  << before_cmvn[3] << " " << before_cmvn[4] << std::endl;
    }
    
    for (auto& frame : lfr_frames) {
        if (frame.size() != static_cast<size_t>(feature_dim)) {
            continue;
        }
        
        // Apply CMVN: (x + mean) * var (matching Python implementation)
        // Python: inputs += means; inputs *= vars
        int dim = std::min(feature_dim, static_cast<int>(cmvn_means_[0].size()));
        for (int i = 0; i < dim; i++) {
            frame[i] = (frame[i] + cmvn_means_[0][i]) * cmvn_vars_[0][i];
        }
    }
    
    // Debug: print first frame after CMVN (only for first call)
    if (first_call && !lfr_frames.empty()) {
        std::cout << "  First frame after CMVN (first 5): " 
                  << lfr_frames[0][0] << " " << lfr_frames[0][1] << " " << lfr_frames[0][2] << " " 
                  << lfr_frames[0][3] << " " << lfr_frames[0][4] << std::endl;
        first_call = false;  // Only print once
    }
}

void StreamingFBankExtractor::reset() {
    audio_buffer_.clear();
    fbank_frame_buffer_.clear();
    is_first_lfr_output_ = true;  // Reset flag for new audio stream
    global_lfr_output_idx_ = 0;   // Reset global LFR output index
    buffer_start_raw_idx_ = 0;    // Reset buffer start position
}
