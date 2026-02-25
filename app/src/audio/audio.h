// c_functions.h - 兼容 C/C++ 的头文件
#ifndef __AUDIO_H
#define __AUDIO_H

// 核心：如果是 C++ 编译器，用 extern "C" 包裹 C 函数声明
#ifdef __cplusplus
extern "C" {
#endif

// 声明 C 函数（和 .c 文件中的实现对应）
int audio_cap_main(void *callback(uint8_t *data, uint32_t len));
#ifdef __cplusplus
} // 结束 extern "C"
#endif

#endif // AUDIO_H