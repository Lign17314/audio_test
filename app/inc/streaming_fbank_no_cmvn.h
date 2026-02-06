/* Copyright 2025
 * Streaming FBank Extractor without CMVN
 * 
 * 用于 V2 版本，只做 FBank 和 LFR，不做 CMVN
 */

#ifndef STREAMING_FBANK_NO_CMVN_H_
#define STREAMING_FBANK_NO_CMVN_H_

#include "../streaming_pipe/inc/streaming_fbank_extractor.h"

/**
 * 创建一个不带 CMVN 的 FBank 提取器
 * 通过修改 config 来禁用 CMVN
 */
inline std::unique_ptr<StreamingFBankExtractor> CreateFBankExtractorNoCMVN(
    const FBankConfig& config) {
    
    FBankConfig config_no_cmvn = config;
    config_no_cmvn.cmvn_file = "";  // 禁用 CMVN 文件加载
    
    return std::make_unique<StreamingFBankExtractor>(config_no_cmvn);
}

#endif  // STREAMING_FBANK_NO_CMVN_H_
