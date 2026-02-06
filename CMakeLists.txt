# /tmp/tflm-tree/CMakeLists.txt 核心配置
# CMake最低版本要求
cmake_minimum_required(VERSION 3.15)
# 项目名+语言（C/C++）
project(tflm_riscv64 LANGUAGES C CXX)
# 设置C/C++标准（TFLM要求C11/C++11）
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/toolchain-riscv64.cmake)

add_subdirectory(thirdpart/tflm-tree)