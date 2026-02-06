#ifndef STREAMING_FBANK_EXTRACTOR_INT16_H
#define STREAMING_FBANK_EXTRACTOR_INT16_H

#include <vector>
#include <memory>
#include <cstdint>
#include "streaming_fbank_extractor.h"
#include "quantization_utils.h"

/**
 * Streaming FBank Feature Extractor with INT16 Output
 * 
 * 基于 float32 版本的 FBank 提取器，但直接输出 INT16 量化后的特征
 * 适用于 16x8 量化模型，减少内存占用和后续量化开销
 */
class StreamingFBankExtractorINT16 {
public:
    /**
     * 构造函数
     * @param config FBank 配置
     * @param scale 量化 scale 参数（从模型获取）
     * @param zero_point 量化 zero_point 参数（从模型获取）
     */
    explicit StreamingFBankExtractorINT16(const FBankConfig& config,
                                          float scale,
                                          int32_t zero_point);
    
    ~StreamingFBankExtractorINT16();

    /**
     * 处理音频块，直接输出 INT16 特征
     * @param audio_chunk 音频样本（归一化到 [-1, 1]）
     * @param chunk_length 样本数量（10ms @ 16kHz = 160）
     * @param output_lfr_frames 输出 INT16 LFR 帧
     * @return 输出的 LFR 帧数
     */
    int process_chunk(
        const float* audio_chunk,
        size_t chunk_length,
        std::vector<std::vector<int16_t>>& output_lfr_frames
    );

    /**
     * 刷新剩余帧并输出最终的 INT16 LFR 帧
     * @param output_lfr_frames 输出 INT16 LFR 帧
     * @return 输出的 LFR 帧数
     */
    int flush(std::vector<std::vector<int16_t>>& output_lfr_frames);

    /**
     * 重置内部状态（清空缓冲区）
     */
    void reset();

    /**
     * 获取输出特征维度
     */
    int get_output_dim() const { 
        return float_extractor_->get_output_dim(); 
    }
    
    /**
     * 获取量化参数
     */
    QuantizationParams get_quant_params() const {
        return QuantizationParams(scale_, zero_point_);
    }

private:
    // 量化参数
    float scale_;
    int32_t zero_point_;
    
    // 内部使用 float32 版本的提取器
    std::unique_ptr<StreamingFBankExtractor> float_extractor_;
    
    // 辅助函数：量化单帧
    void quantize_frame(const std::vector<float>& frame_float,
                       std::vector<int16_t>& frame_int16);
    
    // 辅助函数：批量量化多帧
    void quantize_frames(const std::vector<std::vector<float>>& frames_float,
                        std::vector<std::vector<int16_t>>& frames_int16);
};

#endif // STREAMING_FBANK_EXTRACTOR_INT16_H
