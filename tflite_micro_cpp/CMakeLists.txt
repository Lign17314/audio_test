# ============================================================================
# CMakeLists.txt for TFLite Micro C++ Programs
# ============================================================================
#
# 这个 CMakeLists.txt 用于编译使用 TFLite Micro 库的 C++ 程序
#
# 编译步骤：
#   1. 安装 TFLite Micro 库和头文件到本地（推荐）：
#      cd tflite_micro_cpp
#      ./install_tflite_micro.sh
#
#      或者手动编译外部库：
#      cd ../tflite-micro
#      make -f tensorflow/lite/micro/tools/make/Makefile TARGET=linux microlite
#
#   2. 创建构建目录并配置：
#      mkdir -p build
#      cd build
#      cmake ..
#
#   3. 编译：
#      make -j$(nproc)
#
#   4. 运行程序：
#      ./run_encoder ../tflite_models/fsmn_encoder_stateful_float32.tflite input.npy output.npy
#      ./test_tflite_micro ../tflite_models/fsmn_encoder_float32.tflite
#
# ============================================================================

cmake_minimum_required(VERSION 3.16)
project(tflite_micro_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# 路径配置
# ============================================================================
# 优先使用本地安装的 TFLite Micro（如果存在）
set(TFLITE_MICRO_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tflite_micro_install")
set(TFLITE_MICRO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tflite-micro")

# 检查本地安装是否存在
if(EXISTS "${TFLITE_MICRO_INSTALL_DIR}/lib/libtensorflow-microlite.a")
    # 使用本地安装
    set(TFLITE_MICRO_LIB_DIR "${TFLITE_MICRO_INSTALL_DIR}/lib")
    set(TFLITE_MICRO_LIB "${TFLITE_MICRO_INSTALL_DIR}/lib/libtensorflow-microlite.a")
    set(TFLITE_MICRO_GENFILES "${TFLITE_MICRO_INSTALL_DIR}/genfiles")
    set(TFLITE_MICRO_INCLUDE_DIR "${TFLITE_MICRO_INSTALL_DIR}/include")
    set(USE_LOCAL_INSTALL TRUE)
    message(STATUS "Using local TFLite Micro installation: ${TFLITE_MICRO_INSTALL_DIR}")
else()
    # 回退到外部 tflite-micro 目录
    set(TFLITE_MICRO_LIB_DIR "${TFLITE_MICRO_DIR}/gen/linux_x86_64_default_gcc/lib")
    set(TFLITE_MICRO_LIB "${TFLITE_MICRO_LIB_DIR}/libtensorflow-microlite.a")
    set(TFLITE_MICRO_GENFILES "${TFLITE_MICRO_DIR}/gen/linux_x86_64_default_gcc/genfiles")
    set(TFLITE_MICRO_INCLUDE_DIR "${TFLITE_MICRO_DIR}")
    set(USE_LOCAL_INSTALL FALSE)
    message(STATUS "Using external TFLite Micro: ${TFLITE_MICRO_DIR}")
    message(STATUS "  (To use local installation, run: ./install_tflite_micro.sh)")
endif()

# ============================================================================
# 检查 TFLite Micro 库是否存在
# ============================================================================
if(NOT EXISTS "${TFLITE_MICRO_LIB}")
    message(FATAL_ERROR
        "TFLite Micro library not found: ${TFLITE_MICRO_LIB}\n"
        "Please either:\n"
        "  1. Run the installation script:\n"
        "     cd ${CMAKE_CURRENT_SOURCE_DIR}\n"
        "     ./install_tflite_micro.sh\n"
        "  OR\n"
        "  2. Compile the library manually:\n"
        "     cd ${TFLITE_MICRO_DIR}\n"
        "     make -f tensorflow/lite/micro/tools/make/Makefile TARGET=linux microlite"
    )
endif()

message(STATUS "Found TFLite Micro library: ${TFLITE_MICRO_LIB}")

# ============================================================================
# 包含路径配置
# ============================================================================
if(USE_LOCAL_INSTALL)
    # 使用本地安装的路径
    # Note: For fixedpoint.h, we need to include the gemmlowp directory itself
    # because headers reference "fixedpoint/fixedpoint.h" relative to gemmlowp root
    include_directories(
        ${TFLITE_MICRO_INCLUDE_DIR}
        ${TFLITE_MICRO_INCLUDE_DIR}/downloads/gemmlowp
        ${TFLITE_MICRO_INCLUDE_DIR}/downloads/flatbuffers/include
        ${TFLITE_MICRO_INCLUDE_DIR}/downloads/kissfft
        ${TFLITE_MICRO_INCLUDE_DIR}/downloads/ruy
        ${TFLITE_MICRO_GENFILES}
        # Add gemmlowp parent directory for fixedpoint includes
        ${TFLITE_MICRO_INCLUDE_DIR}/downloads
    )
else()
    # 使用外部 tflite-micro 的路径
    include_directories(
        ${TFLITE_MICRO_DIR}
        ${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads
        ${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/gemmlowp
        ${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
        ${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft
        ${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/ruy
        ${TFLITE_MICRO_GENFILES}
    )
endif()

# ============================================================================
# 可执行文件 1: run_encoder
# ============================================================================
# 用于运行 encoder 推理的程序，支持 stateful 模型和 cache 处理
add_executable(run_encoder encoder_runner/run_encoder.cc)

target_link_libraries(run_encoder PRIVATE
    ${TFLITE_MICRO_LIB}
    m
    pthread
)

# ============================================================================
# 可执行文件 2: run_encoder_16x8
# ============================================================================
# 用于运行 INT8/16x8 量化 encoder 推理的程序
add_executable(run_encoder_16x8 encoder_runner/run_encoder_16x8.cc)

target_link_libraries(run_encoder_16x8 PRIVATE
    ${TFLITE_MICRO_LIB}
    m
    pthread
)

# ============================================================================
# 可执行文件 3: test_tflite_micro
# ============================================================================
# 用于测试 TFLite Micro 库的程序
add_executable(test_tflite_micro test_tflite_micro.cc)

target_link_libraries(test_tflite_micro PRIVATE
    ${TFLITE_MICRO_LIB}
    m
    pthread
)

# ============================================================================
# 测试目标（可选）
# ============================================================================
# 运行 test_tflite_micro 的测试目标
# 使用方法: make test 或 ctest
enable_testing()

# 测试 float32 模型
add_test(NAME test_float32
    COMMAND test_tflite_micro
    ${CMAKE_CURRENT_SOURCE_DIR}/../tflite_models/fsmn_encoder_float32.tflite
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# 测试 int8 模型（如果存在）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tflite_models/fsmn_encoder_int8.tflite")
    add_test(NAME test_int8
        COMMAND test_tflite_micro
        ${CMAKE_CURRENT_SOURCE_DIR}/../tflite_models/fsmn_encoder_int8.tflite
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

# 测试 16x8 模型（如果存在）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tflite_models/fsmn_encoder_16x8.tflite")
    add_test(NAME test_16x8
        COMMAND test_tflite_micro
        ${CMAKE_CURRENT_SOURCE_DIR}/../tflite_models/fsmn_encoder_16x8.tflite
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

# ============================================================================
# 可执行文件 4: test_fbank_extractor
# ============================================================================
# FBank 特征提取器（不需要 TFLite Micro 库）
# 用于从 WAV 文件提取 fbank 特征，匹配 Python wav_frontend.py 的行为
add_executable(test_fbank_extractor
    fbank_extractor/test_fbank_extractor.cc
    fbank_extractor/fbank_extractor.cc
    fbank_extractor/wav_reader.cc
    ${TFLITE_MICRO_DIR}/signal/src/kiss_fft_wrappers/kiss_fft_float.cc
)

target_include_directories(test_fbank_extractor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fbank_extractor
    ${TFLITE_MICRO_DIR}
    ${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft
)

target_link_libraries(test_fbank_extractor PRIVATE
    m
)

# ============================================================================
# 可执行文件 5: test_ctc_decoder
# ============================================================================
# CTC 解码器（不需要 TFLite Micro 库）
# 用于从 logits 解码关键词，匹配 Python ctc.py KwsCtcPrefixDecoder 的行为
add_executable(test_ctc_decoder
    ctc_decoder/test_ctc_decoder.cc
    ctc_decoder/ctc_decoder.cc
)

target_include_directories(test_ctc_decoder PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ctc_decoder
)

target_link_libraries(test_ctc_decoder PRIVATE
    m
)

# ============================================================================
# 安装配置（可选）
# ============================================================================
# 如果需要安装可执行文件，取消下面的注释
# install(TARGETS run_encoder test_tflite_micro test_fbank_extractor
#     RUNTIME DESTINATION bin
# )
