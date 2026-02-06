# ******************************************
# RISCV64交叉编译CMake工具链文件（玄铁C906 + musl-libc）
# 使用方式：cmake -DCMAKE_TOOLCHAIN_FILE=toolchain-riscv64.cmake ..
# ******************************************
cmake_minimum_required(VERSION 3.15)

# 1. 交叉编译核心：指定目标系统/架构（不可随意修改）
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_SYSTEM_VERSION 1) # 版本无实际意义，仅为CMake规范

# 2. 工具链配置（***************** 仅需修改这部分适配你的环境 *****************）
# 交叉编译工具链根目录（你的玄铁musl工具链路径，保持不变）
set(TOOLCHAIN_ROOT "/root/volume/ctc/ctc_tflite_micro/host-tools-1.6/gcc/riscv64-linux-musl-x86_64")
# 交叉编译前缀（musl工具链为riscv64-unknown-linux-musl-，glibc一般为riscv64-linux-gnu-）
set(CROSS_PREFIX "riscv64-unknown-linux-musl-")
# 玄铁C906专属架构参数（无需修改，其他RISCV64平台需替换为对应参数）
set(ARCH_FLAGS "-mcpu=c906fdv -march=rv64imafdcv0p7xthead -mabi=lp64d -mcmodel=medany -mno-ldd")
# C/C++标准（根据项目需求修改，如C99/C17，C++11/C20）
set(C_STANDARD "gnu11")
set(CXX_STANDARD "c++17")

# 3. 工具链有效性检查（路径错误直接报错，避免后续编译踩坑）
set(TC_GCC "${TOOLCHAIN_ROOT}/bin/${CROSS_PREFIX}gcc")
set(TC_GXX "${TOOLCHAIN_ROOT}/bin/${CROSS_PREFIX}g++")
if(NOT EXISTS ${TC_GCC})
    message(FATAL_ERROR "RISCV64交叉编译器不存在！检查路径：${TC_GCC}")
endif()
if(NOT EXISTS ${TC_GXX})
    message(FATAL_ERROR "RISCV64交叉C++编译器不存在！检查路径：${TC_GXX}")
endif()

# 4. 指定交叉编译器（C/C++/汇编）
set(CMAKE_C_COMPILER   ${TC_GCC})
set(CMAKE_CXX_COMPILER ${TC_GXX})
set(CMAKE_ASM_COMPILER ${TC_GCC}) # 汇编器复用GCC，通用写法
# 工具链辅助工具（objcopy/objdump/strip，用于烧录/调试/瘦身）
set(CMAKE_OBJCOPY ${TOOLCHAIN_ROOT}/bin/${CROSS_PREFIX}objcopy CACHE FILEPATH "" FORCE)
set(CMAKE_OBJDUMP ${TOOLCHAIN_ROOT}/bin/${CROSS_PREFIX}objdump CACHE FILEPATH "" FORCE)
set(CMAKE_STRIP   ${TOOLCHAIN_ROOT}/bin/${CROSS_PREFIX}strip   CACHE FILEPATH "" FORCE)

# 5. 交叉编译查找规则（关键：避免链接主机库/头文件）
set(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_ROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)  # 程序（如cmake）仅在主机查找
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)  # 库仅在目标工具链查找
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)  # 头文件仅在目标工具链查找
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)  # 第三方包仅在目标工具链查找

# 6. 通用编译标志（编译优化、段拆分、警告等，通用所有RISCV64项目）
set(COMMON_FLAGS "-ffunction-sections -fdata-sections -fsigned-char -O2 -Wall")

# 7. 分语言赋值编译标志（架构参数+通用参数+语言标准）
set(CMAKE_C_FLAGS        "${ARCH_FLAGS} -std=${C_STANDARD} ${COMMON_FLAGS} -Wno-pointer-to-int-cast" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS      "${ARCH_FLAGS} -std=${CXX_STANDARD} ${COMMON_FLAGS} -fPIC -fexceptions -frtti" CACHE STRING "" FORCE)
set(CMAKE_ASM_FLAGS      "${ARCH_FLAGS} -x assembler-with-cpp" CACHE STRING "" FORCE) # 汇编支持C预处理（如#include/#define）

# 8. 链接器标志（单独拆分，避免编译阶段识别错误，核心！）
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--gc-sections -lm -lpthread -lstdc++" CACHE STRING "" FORCE)

# 注意：不设置 CMAKE_STATIC_LINKER_FLAGS
# 静态库归档器(ar)只是打包目标文件，不需要链接器标志
# 垃圾回收会在最终链接可执行文件时由 CMAKE_EXE_LINKER_FLAGS 处理

# 可选：静态链接（musl-libc推荐，生成无依赖可执行文件，取消注释即可）
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static" CACHE STRING "" FORCE)
# 可选：调试标志（GDB调试时添加，发布时注释）
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -ggdb")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -ggdb")