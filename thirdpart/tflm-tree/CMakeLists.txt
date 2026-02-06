
# ========================= TFLM必要宏定义（官方推荐） =========================
add_definitions(
    -DTF_LITE_STATIC_MEMORY        # 静态内存分配（嵌入式必备）
    -DTF_LITE_MCU_DEBUG_LOG        # 调试日志支持
    -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS  # 全局数学函数
    -DTF_LITE_USE_GLOBAL_MIN       # 全局MIN函数
    -DTF_LITE_USE_GLOBAL_MAX       # 全局MAX函数
)

# ========================= 头文件目录（TFLM核心+第三方依赖） =========================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/lite/micro
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/lite/micro/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flatbuffers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ruy
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gemmlowp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kissfft
)

# ========================= 收集TFLM核心源码（递归查找C/C++文件） =========================
# 递归查找tensorflow/下所有C/C++源码（核心库）
file(GLOB_RECURSE TFLM_CORE_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/**/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/**/*.cc
)
# 递归查找third_party/下所有C/C++源码（剔除测试/示例，避免冗余）
file(GLOB_RECURSE THIRD_PARTY_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/**/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/**/*.cc
)
list(FILTER THIRD_PARTY_SRCS EXCLUDE REGEX "test|example|demo")

# ========================= 生成TFLM静态库 libtflm.a =========================
add_library(tflm STATIC
    ${TFLM_CORE_SRCS}
    ${THIRD_PARTY_SRCS}
)
# 链接必要系统库（数学库、线程库）
target_link_libraries(tflm m pthread)

# ========================= 可选：编译hello_world示例（验证库可用性） =========================
# 收集hello_world示例源码
file(GLOB_RECURSE HELLO_WORLD_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/hello_world/**/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/hello_world/**/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/hello_world/*.cc
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}examples/hello_world/)
# 生成hello_world可执行文件（依赖libtflm）
add_executable(tflm_hello_world ${HELLO_WORLD_SRCS})
target_link_libraries(tflm_hello_world tflm)