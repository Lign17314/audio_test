cmake_minimum_required(VERSION 3.10)
project(StreamingFBankOnly)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compile options
add_compile_options(-Wall -Wextra)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fbank_extractor
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite_micro_install/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite_micro_install/include/downloads/flatbuffers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite_micro_install/include/downloads/gemmlowp
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite_micro_install/include/downloads/ruy
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite_micro_install/include/signal/src/kiss_fft_wrappers
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite_micro_install/include/downloads/kissfft
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite_micro_install/include/tensorflow/lite/micro/tools/make/downloads/kissfft
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite_micro_install/genfiles
)

# Source files
set(SOURCES
    src/streaming_fbank_main.cc
    src/streaming_fbank_extractor.cc
    src/fbank_extractor/fbank_extractor.cc
    src/fbank_extractor/wav_reader.cc
)

# Create executable
add_executable(streaming_pipe ${SOURCES})

# Link libraries
target_link_libraries(streaming_pipe
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite_micro_install/lib/libtensorflow-microlite.a
    m  # Math library
)

# Create 30ms threaded version executable (main implementation)
set(SOURCES_30MS_THREADED
    src/streaming_fbank_30ms_threaded.cc
    src/streaming_fbank_extractor.cc
    src/fbank_extractor/fbank_extractor.cc
    src/fbank_extractor/wav_reader.cc
)

add_executable(streaming_fbank_30ms_threaded ${SOURCES_30MS_THREADED})

target_link_libraries(streaming_fbank_30ms_threaded
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite_micro_install/lib/libtensorflow-microlite.a
    pthread  # Threading library
    m        # Math library
)

# Create 30ms threaded 16x8 quantized version executable (experimental)
set(SOURCES_30MS_THREADED_16X8
    src/streaming_fbank_30ms_threaded_16x8.cc
    src/streaming_fbank_extractor.cc
    src/fbank_extractor/fbank_extractor.cc
    src/fbank_extractor/wav_reader.cc
)

add_executable(streaming_fbank_30ms_threaded_16x8 ${SOURCES_30MS_THREADED_16X8})

target_link_libraries(streaming_fbank_30ms_threaded_16x8
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite_micro_install/lib/libtensorflow-microlite.a
    pthread  # Threading library
    m        # Math library
)

# Create 30ms threaded 16x8 INT16 optimized version executable (INT16 FBank output)
set(SOURCES_30MS_THREADED_16X8_INT16
    src/streaming_fbank_30ms_threaded_16x8_int16.cc
    src/streaming_fbank_extractor.cc
    src/streaming_fbank_extractor_int16.cc
    src/fbank_extractor/fbank_extractor.cc
    src/fbank_extractor/wav_reader.cc
)

add_executable(streaming_fbank_30ms_threaded_16x8_int16 ${SOURCES_30MS_THREADED_16X8_INT16})

target_include_directories(streaming_fbank_30ms_threaded_16x8_int16 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_link_libraries(streaming_fbank_30ms_threaded_16x8_int16
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite_micro_install/lib/libtensorflow-microlite.a
    pthread  # Threading library
    m        # Math library
)

# Test CMVN loading
set(SOURCES_TEST_CMVN
    src/test_cmvn_loading.cc
    src/streaming_fbank_extractor.cc
    src/fbank_extractor/fbank_extractor.cc
)

add_executable(test_cmvn_loading ${SOURCES_TEST_CMVN})

target_include_directories(test_cmvn_loading PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_link_libraries(test_cmvn_loading
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite_micro_install/lib/libtensorflow-microlite.a
    m  # Math library
)

# Optional: Enable CMVN from header
# Uncomment the following line to use CMVN data from cmvn_data.h
add_definitions(-DUSE_CMVN_HEADER)

message(STATUS "Configured Streaming FBank with Threaded Encoder")
message(STATUS "  Float32 version: ${SOURCES_30MS_THREADED}")
message(STATUS "  16x8 version: ${SOURCES_30MS_THREADED_16X8}")
message(STATUS "  16x8 INT16 optimized version: ${SOURCES_30MS_THREADED_16X8_INT16}")
