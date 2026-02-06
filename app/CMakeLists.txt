
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

add_custom_command(
    TARGET streaming_fbank_30ms_threaded_16x8_int16_kws          # 关联目标：编译完 audio_test 后执行
    POST_BUILD                 # 时机：目标构建完成后（编译+链接后）
    COMMAND ${CMAKE_STRIP}     # strip 工具（CMake 内置变量，自动找系统 strip）
    ARGS ${CMAKE_BINARY_DIR}/app/streaming_fbank_30ms_threaded_16x8_int16_kws  # 参数：要剥离的文件路径（编译生成的可执行文件）
    COMMENT "Stripping debug symbols from streaming_fbank_30ms_threaded_16x8_int16_kws"  # 执行时的提示信息
)

add_executable(audio_test src/audio/audio.c)
target_include_directories(audio_test PRIVATE
    ../platform/cvi_mpi/include)
target_link_directories(audio_test  PRIVATE
    ../platform/cvi_mpi/lib
    ../platform/cvi_mpi/lib/3rd
)
target_link_libraries(
    audio_test
    cvi_vqe.a 
    cvi_audio.a 
    sbc.a 
    cvi_VoiceEngine.a 
    cvi_RES1.a 
    cvi_ssp.a 
    tinyalsa.a 
    aacdec2.a 
    aacsbrdec2.a 
    aacenc2.a 
    aaccomm2.a 
    aacsbrenc2.a 
    sys.a 
    msg.a 
    cvilink.a 
    vi.a 
    vpss.a 
    vo.a 
    rgn.a 
    gdc.a 
    dl 
    pthread  # Threading library
    m        # Math library
)

add_custom_command(
    TARGET audio_test          # 关联目标：编译完 audio_test 后执行
    POST_BUILD                 # 时机：目标构建完成后（编译+链接后）
    COMMAND ${CMAKE_STRIP}     # strip 工具（CMake 内置变量，自动找系统 strip）
    ARGS ${CMAKE_BINARY_DIR}/app/audio_test  # 参数：要剥离的文件路径（编译生成的可执行文件）
    COMMENT "Stripping debug symbols from audio_test"  # 执行时的提示信息
)