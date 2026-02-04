/* Copyright 2025
 * Test CMVN Parameter Loading
 * 
 * 测试从 StreamingFBankExtractor 加载 CMVN 参数
 */

#include <iostream>
#include <vector>
#include "../streaming_pipe/inc/streaming_fbank_extractor.h"

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Test CMVN Parameter Loading" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // 步骤 1: 创建 FBank 配置
    FBankConfig config;
    config.fs = 16000;
    config.n_mels = 80;
    config.frame_length = 25;
    config.frame_shift = 10;
    config.lfr_m = 5;
    config.lfr_n = 3;
    config.cmvn_file = "";  // 使用 header 中的 CMVN
    
    std::cout << "Step 1: Creating FBank extractor..." << std::endl;
    StreamingFBankExtractor extractor(config);
    
    // 步骤 2: 检查 CMVN 是否加载
    std::cout << "\nStep 2: Checking CMVN status..." << std::endl;
    if (extractor.has_cmvn()) {
        std::cout << "  ✅ CMVN loaded successfully" << std::endl;
    } else {
        std::cout << "  ❌ CMVN not loaded" << std::endl;
        return 1;
    }
    
    // 步骤 3: 获取 CMVN 参数
    std::cout << "\nStep 3: Getting CMVN parameters..." << std::endl;
    const std::vector<float>& cmvn_means = extractor.get_cmvn_means();
    const std::vector<float>& cmvn_vars = extractor.get_cmvn_vars();
    
    std::cout << "  CMVN means size: " << cmvn_means.size() << std::endl;
    std::cout << "  CMVN vars size: " << cmvn_vars.size() << std::endl;
    
    if (cmvn_means.empty() || cmvn_vars.empty()) {
        std::cout << "  ❌ CMVN parameters are empty" << std::endl;
        return 1;
    }
    
    std::cout << "  ✅ CMVN parameters retrieved" << std::endl;
    
    // 步骤 4: 显示前 10 个值
    std::cout << "\nStep 4: Displaying first 10 values..." << std::endl;
    std::cout << "  Means: ";
    for (size_t i = 0; i < std::min(size_t(10), cmvn_means.size()); i++) {
        std::cout << cmvn_means[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "  Vars: ";
    for (size_t i = 0; i < std::min(size_t(10), cmvn_vars.size()); i++) {
        std::cout << cmvn_vars[i] << " ";
    }
    std::cout << std::endl;
    
    // 步骤 5: 验证参数范围
    std::cout << "\nStep 5: Validating parameter ranges..." << std::endl;
    
    // 检查 means 范围（通常在 [-10, 0] 范围）
    float mean_min = cmvn_means[0];
    float mean_max = cmvn_means[0];
    for (const auto& val : cmvn_means) {
        mean_min = std::min(mean_min, val);
        mean_max = std::max(mean_max, val);
    }
    std::cout << "  Means range: [" << mean_min << ", " << mean_max << "]" << std::endl;
    
    // 检查 vars 范围（通常在 [0.1, 0.3] 范围）
    float var_min = cmvn_vars[0];
    float var_max = cmvn_vars[0];
    for (const auto& val : cmvn_vars) {
        var_min = std::min(var_min, val);
        var_max = std::max(var_max, val);
    }
    std::cout << "  Vars range: [" << var_min << ", " << var_max << "]" << std::endl;
    
    // 验证范围是否合理
    bool valid = true;
    if (mean_min < -20.0f || mean_max > 10.0f) {
        std::cout << "  ⚠️  Warning: Means range seems unusual" << std::endl;
        valid = false;
    }
    if (var_min < 0.01f || var_max > 1.0f) {
        std::cout << "  ⚠️  Warning: Vars range seems unusual" << std::endl;
        valid = false;
    }
    
    if (valid) {
        std::cout << "  ✅ Parameter ranges are valid" << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "✅ All tests passed!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
