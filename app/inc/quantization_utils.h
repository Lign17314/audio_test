#ifndef QUANTIZATION_UTILS_H
#define QUANTIZATION_UTILS_H

#include <cstdint>
#include <cmath>
#include <algorithm>

/**
 * 量化工具函数
 * 用于 float32 和 INT16 之间的转换
 */

// 量化参数结构
struct QuantizationParams {
    float scale;
    int32_t zero_point;
    
    QuantizationParams() : scale(1.0f), zero_point(0) {}
    QuantizationParams(float s, int32_t zp) : scale(s), zero_point(zp) {}
};

// Float32 → INT16 量化
inline int16_t QuantizeFloatToInt16(float value, float scale, int32_t zero_point) {
    int32_t quantized = static_cast<int32_t>(std::round(value / scale + zero_point));
    // Clip to INT16 range [-32768, 32767]
    quantized = std::max(-32768, std::min(32767, quantized));
    return static_cast<int16_t>(quantized);
}

// INT16 → Float32 反量化
inline float DequantizeInt16ToFloat(int16_t value, float scale, int32_t zero_point) {
    return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}

// 批量量化（向量）
inline void QuantizeVectorToInt16(const float* input, 
                                  int16_t* output,
                                  size_t size,
                                  float scale,
                                  int32_t zero_point) {
    for (size_t i = 0; i < size; i++) {
        output[i] = QuantizeFloatToInt16(input[i], scale, zero_point);
    }
}

// 批量反量化（向量）
inline void DequantizeVectorToFloat(const int16_t* input,
                                    float* output,
                                    size_t size,
                                    float scale,
                                    int32_t zero_point) {
    for (size_t i = 0; i < size; i++) {
        output[i] = DequantizeInt16ToFloat(input[i], scale, zero_point);
    }
}

#endif // QUANTIZATION_UTILS_H
