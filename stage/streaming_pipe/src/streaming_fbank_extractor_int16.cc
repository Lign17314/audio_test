#include "streaming_fbank_extractor_int16.h"
#include <cstring>

StreamingFBankExtractorINT16::StreamingFBankExtractorINT16(
    const FBankConfig& config,
    float scale,
    int32_t zero_point)
    : scale_(scale), zero_point_(zero_point) {
    
    // 创建内部的 float32 提取器
    float_extractor_ = std::make_unique<StreamingFBankExtractor>(config);
}

StreamingFBankExtractorINT16::~StreamingFBankExtractorINT16() {
    // unique_ptr 会自动清理
}

int StreamingFBankExtractorINT16::process_chunk(
    const float* audio_chunk,
    size_t chunk_length,
    std::vector<std::vector<int16_t>>& output_lfr_frames) {
    
    // 1. 使用 float32 提取器处理
    std::vector<std::vector<float>> output_frames_float;
    int num_frames = float_extractor_->process_chunk(
        audio_chunk, chunk_length, output_frames_float);
    
    // 2. 立即量化为 INT16
    quantize_frames(output_frames_float, output_lfr_frames);
    
    return num_frames;
}

int StreamingFBankExtractorINT16::flush(
    std::vector<std::vector<int16_t>>& output_lfr_frames) {
    
    // 1. 刷新 float32 提取器
    std::vector<std::vector<float>> output_frames_float;
    int num_frames = float_extractor_->flush(output_frames_float);
    
    // 2. 量化为 INT16
    quantize_frames(output_frames_float, output_lfr_frames);
    
    return num_frames;
}

void StreamingFBankExtractorINT16::reset() {
    float_extractor_->reset();
}

void StreamingFBankExtractorINT16::quantize_frame(
    const std::vector<float>& frame_float,
    std::vector<int16_t>& frame_int16) {
    
    frame_int16.resize(frame_float.size());
    
    for (size_t i = 0; i < frame_float.size(); i++) {
        frame_int16[i] = QuantizeFloatToInt16(
            frame_float[i], scale_, zero_point_);
    }
}

void StreamingFBankExtractorINT16::quantize_frames(
    const std::vector<std::vector<float>>& frames_float,
    std::vector<std::vector<int16_t>>& frames_int16) {
    
    frames_int16.clear();
    frames_int16.reserve(frames_float.size());
    
    for (const auto& frame_float : frames_float) {
        std::vector<int16_t> frame_int16;
        quantize_frame(frame_float, frame_int16);
        frames_int16.push_back(std::move(frame_int16));
    }
}
