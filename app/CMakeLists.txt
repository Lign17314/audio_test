
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compile options
add_compile_options(-Wall -Wextra)

# Optional: Enable CMVN from header
# Uncomment the following line to use CMVN data from cmvn_data.h
add_definitions(-DUSE_CMVN_HEADER)

# Create 30ms threaded 16x8 INT16 with KWS decoder (Keyword Spotting)
set(SOURCES_30MS_THREADED_16X8_INT16_KWS
    src/streaming_fbank_30ms_threaded_16x8_int16_kws.cc
    src/streaming_fbank_extractor.cc
    src/streaming_fbank_extractor_int16.cc
    src/fbank_extractor.cc
    src/wav_reader.cc
)

add_executable(streaming_fbank_30ms_threaded_16x8_int16_kws ${SOURCES_30MS_THREADED_16X8_INT16_KWS})

target_include_directories(streaming_fbank_30ms_threaded_16x8_int16_kws PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdpart/tflm-tree
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdpart/tflm-tree/tensorflow
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdpart/tflm-tree/tensorflow/lite/micro
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdpart/tflm-tree/tensorflow/lite/micro/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdpart/tflm-tree/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdpart/tflm-tree/third_party/flatbuffers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdpart/tflm-tree/third_party/ruy
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdpart/tflm-tree/third_party/gemmlowp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdpart/tflm-tree/third_party/kissfft
)

target_link_libraries(streaming_fbank_30ms_threaded_16x8_int16_kws
    tflm
    pthread  # Threading library
    m        # Math library
)
