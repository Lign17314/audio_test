#!/bin/bash
# TFLite Micro C++ 一键运行脚本（包含编译、运行和结果对比）
# 功能：编译（如需要）、运行 encoder、对比测试结果
#
# 使用方法:
#   ./run.sh [model.tflite] [input.npy] [output.npy] [--compare-with REF_OUTPUT] [--skip-compare]
#
# 示例:
#   ./run.sh
#   ./run.sh ../tflite_models/fsmn_encoder_stateful_float32.tflite ../features/test_xiaoyun_fbank.npy output.npy
#   ./run.sh --compare-with ../tflite_cpp/build/output_logits_no_xnnpack.npy
#   ./run.sh --skip-compare  # 跳过对比测试

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PARENT_DIR}/build"
ROOT_DIR="$(cd "${PARENT_DIR}/.." && pwd)"

# 解析参数
SKIP_COMPARE=false
COMPARE_WITH=""
WAV_FILE=""
MODEL_FILE=""
INPUT_FILE=""
OUTPUT_FILE=""

# 解析参数
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-compare)
            SKIP_COMPARE=true
            shift
            ;;
        --compare-with)
            COMPARE_WITH="$2"
            shift 2
            ;;
        --compare-with=*)
            COMPARE_WITH="${1#*=}"
            shift
            ;;
        --wav)
            WAV_FILE="$2"
            shift 2
            ;;
        --wav=*)
            WAV_FILE="${1#*=}"
            shift
            ;;
        -h|--help)
            # 帮助信息会在后面显示
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# 设置位置参数
set -- "${POSITIONAL_ARGS[@]}"

MODEL_FILE="${1:-../tflite_models/fsmn_encoder_stateful_float32.tflite}"
INPUT_FILE="${2:-../features/test_xiaoyun_fbank.npy}"
OUTPUT_FILE="${3:-output_logits_no_xnnpack.npy}"

# 检查帮助参数（在解析之前）
for arg in "$@"; do
    if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
        echo "TFLite Micro C++ Encoder Runner (with Comparison)"
        echo ""
        echo "Usage: $0 [model.tflite] [input.npy] [output.npy] [OPTIONS]"
        echo ""
        echo "Arguments:"
        echo "  model.tflite  - TFLite 模型文件路径 (默认: ../tflite_models/fsmn_encoder_stateful_float32.tflite)"
        echo "  input.npy     - 输入 fbank 特征文件 (默认: ../features/test_xiaoyun_fbank.npy)"
        echo "  output.npy    - 输出 logits 文件 (默认: output_logits_no_xnnpack.npy)"
        echo ""
        echo "Options:"
        echo "  --compare-with FILE  - 与指定参考输出文件对比（TFLite C++ 或 PyTorch）"
        echo "  --skip-compare       - 跳过对比测试"
        echo "  --wav FILE           - WAV 文件路径（用于 CTC 解码验证）"
        echo ""
        echo "Examples:"
        echo "  $0"
        echo "  $0 ../tflite_models/model.tflite ../features/input.npy output.npy"
        echo "  $0 --compare-with ../tflite_cpp/build/output_logits_no_xnnpack.npy"
        echo "  $0 --skip-compare"
        exit 0
    fi
done

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "TFLite Micro C++ Encoder Runner"
echo "=========================================="
echo "Model: ${MODEL_FILE}"
echo "Input: ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"
if [ -n "${COMPARE_WITH}" ]; then
    echo "Compare with: ${COMPARE_WITH}"
fi
if [ -n "${WAV_FILE}" ]; then
    echo "WAV file: ${WAV_FILE}"
fi
echo ""

# 检查文件是否存在
if [ ! -f "${MODEL_FILE}" ]; then
    echo -e "${RED}ERROR: Model file not found: ${MODEL_FILE}${NC}"
    exit 1
fi

if [ ! -f "${INPUT_FILE}" ]; then
    echo -e "${RED}ERROR: Input file not found: ${INPUT_FILE}${NC}"
    exit 1
fi

# 创建 build 目录
if [ ! -d "${BUILD_DIR}" ]; then
    echo -e "${YELLOW}Creating build directory...${NC}"
    mkdir -p "${BUILD_DIR}"
fi

# ==========================================
# 1. 编译
# ==========================================
echo -e "${BLUE}=========================================="
echo "Step 1: Building"
echo "==========================================${NC}"

# 检查是否需要编译
NEED_BUILD=false
if [ ! -f "${BUILD_DIR}/run_encoder" ]; then
    NEED_BUILD=true
    echo -e "${YELLOW}Executable not found, will build...${NC}"
elif [ "${SCRIPT_DIR}/run_encoder.cc" -nt "${BUILD_DIR}/run_encoder" ] || \
     [ "${PARENT_DIR}/CMakeLists.txt" -nt "${BUILD_DIR}/run_encoder" ]; then
    NEED_BUILD=true
    echo -e "${YELLOW}Source file updated, will rebuild...${NC}"
fi

# 编译
if [ "${NEED_BUILD}" = true ]; then
    echo ""
    echo -e "${GREEN}Building...${NC}"
    cd "${BUILD_DIR}"
    
    # 检查是否有 CMakeLists.txt
    if [ ! -f "${PARENT_DIR}/CMakeLists.txt" ]; then
        echo -e "${RED}ERROR: CMakeLists.txt not found${NC}"
        echo -e "${RED}Please ensure CMakeLists.txt exists in ${PARENT_DIR}${NC}"
        exit 1
    fi
    
    # 检查是否有 CMakeCache.txt，如果没有则运行 cmake
    # 或者如果 CMakeLists.txt 更新了，也需要重新运行 cmake
    if [ ! -f "CMakeCache.txt" ] || [ "${PARENT_DIR}/CMakeLists.txt" -nt "CMakeCache.txt" ]; then
        echo "Running cmake..."
        cmake "${PARENT_DIR}"
    fi
    
    echo "Running make run_encoder..."
    make -j$(nproc) run_encoder
    
    cd "${PARENT_DIR}"
    echo -e "${GREEN}Build completed!${NC}"
    echo ""
else
    echo -e "${GREEN}Build skipped (executable is up to date)${NC}"
    echo ""
fi

# 检查可执行文件
if [ ! -f "${BUILD_DIR}/run_encoder" ]; then
    echo -e "${RED}ERROR: Executable not found: ${BUILD_DIR}/run_encoder${NC}"
    exit 1
fi

# ==========================================
# 2. 运行
# ==========================================
echo -e "${BLUE}=========================================="
echo "Step 2: Running Encoder"
echo "==========================================${NC}"

cd "${BUILD_DIR}"

# 转换相对路径为绝对路径（从 build 目录的角度）
if [[ "${MODEL_FILE}" != /* ]]; then
    MODEL_FILE_ABS="${PARENT_DIR}/${MODEL_FILE}"
else
    MODEL_FILE_ABS="${MODEL_FILE}"
fi

if [[ "${INPUT_FILE}" != /* ]]; then
    INPUT_FILE_ABS="${PARENT_DIR}/${INPUT_FILE}"
else
    INPUT_FILE_ABS="${INPUT_FILE}"
fi

# 输出文件路径处理
if [[ "${OUTPUT_FILE}" != /* ]]; then
    # 相对路径：保存在 build 目录
    OUTPUT_FILE_ABS="${BUILD_DIR}/${OUTPUT_FILE}"
    OUTPUT_FILE_FOR_CMD="${OUTPUT_FILE}"
else
    # 绝对路径：直接使用
    OUTPUT_FILE_ABS="${OUTPUT_FILE}"
    OUTPUT_FILE_FOR_CMD="${OUTPUT_FILE}"
fi

echo -e "${GREEN}Running encoder...${NC}"
./run_encoder "${MODEL_FILE_ABS}" "${INPUT_FILE_ABS}" "${OUTPUT_FILE_FOR_CMD}"

# 检查输出文件
cd "${PARENT_DIR}"
if [ ! -f "${OUTPUT_FILE_ABS}" ]; then
    echo -e "${RED}ERROR: Output file not found: ${OUTPUT_FILE_ABS}${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Success! Output saved to: ${OUTPUT_FILE_ABS}"
echo "==========================================${NC}"

# 显示输出文件信息
if command -v python3 &> /dev/null; then
    python3 -c "
import numpy as np
import sys
try:
    arr = np.load('${OUTPUT_FILE_ABS}')
    print(f'Output shape: {arr.shape}')
    print(f'Output dtype: {arr.dtype}')
    print(f'Output size: {arr.nbytes / 1024 / 1024:.2f} MB')
except Exception as e:
    print(f'Could not read output file: {e}')
" 2>/dev/null || true
fi

# ==========================================
# 3. 结果对比测试
# ==========================================
if [ "${SKIP_COMPARE}" = true ]; then
    echo ""
    echo -e "${YELLOW}Skipping comparison test (--skip-compare)${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}=========================================="
echo "Step 3: Comparison Test"
echo "==========================================${NC}"

# 检查是否有 verify_encoder_output.py
VERIFY_SCRIPT="${ROOT_DIR}/verify_encoder_output.py"
if [ ! -f "${VERIFY_SCRIPT}" ]; then
    echo -e "${YELLOW}WARNING: verify_encoder_output.py not found, skipping comparison${NC}"
    exit 0
fi

# 准备对比参数
INPUT_FILE_ABS_FOR_COMPARE="${INPUT_FILE_ABS}"
OUTPUT_FILE_ABS_FOR_COMPARE="${OUTPUT_FILE_ABS}"

# 如果指定了参考输出文件，先对比数值
if [ -n "${COMPARE_WITH}" ]; then
    echo ""
    echo -e "${GREEN}Comparing with reference output: ${COMPARE_WITH}${NC}"
    
    # 转换参考文件路径为绝对路径
    if [[ "${COMPARE_WITH}" != /* ]]; then
        # 如果是相对路径，从 ROOT_DIR 开始解析
        if [[ "${COMPARE_WITH}" == ../* ]]; then
            # 从 tflite_micro_cpp 目录解析
            COMPARE_WITH_ABS="$(cd "${PARENT_DIR}" && readlink -f "${COMPARE_WITH}" 2>/dev/null || echo "${PARENT_DIR}/${COMPARE_WITH}")"
        else
            COMPARE_WITH_ABS="${ROOT_DIR}/${COMPARE_WITH}"
        fi
    else
        COMPARE_WITH_ABS="${COMPARE_WITH}"
    fi
    
    if [ ! -f "${COMPARE_WITH_ABS}" ]; then
        echo -e "${RED}ERROR: Reference output file not found: ${COMPARE_WITH_ABS}${NC}"
        exit 1
    fi
    
    # 使用 Python 进行简单数值对比
    python3 -c "
import numpy as np
import sys

try:
    micro_output = np.load('${OUTPUT_FILE_ABS_FOR_COMPARE}')
    ref_output = np.load('${COMPARE_WITH_ABS}')
    
    print(f'Micro output shape: {micro_output.shape}')
    print(f'Reference output shape: {ref_output.shape}')
    
    # 确保形状一致
    min_T = min(micro_output.shape[1], ref_output.shape[1])
    min_D = min(micro_output.shape[2], ref_output.shape[2])
    
    micro_trimmed = micro_output[:, :min_T, :min_D]
    ref_trimmed = ref_output[:, :min_T, :min_D]
    
    diff = np.abs(micro_trimmed - ref_trimmed)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f'Max difference: {max_diff:.6e}')
    print(f'Mean difference: {mean_diff:.6e}')
    
    # 计算相对误差
    ref_abs = np.abs(ref_trimmed)
    relative_diff = diff / (ref_abs + 1e-10)
    max_relative_diff = np.max(relative_diff)
    mean_relative_diff = np.mean(relative_diff)
    
    print(f'Max relative difference: {max_relative_diff:.6e}')
    print(f'Mean relative difference: {mean_relative_diff:.6e}')
    
    # 统计差异分布
    large_diff_count = np.sum(diff > 1e-3)
    total_elements = diff.size
    large_diff_ratio = large_diff_count / total_elements * 100
    
    print(f'Elements with diff > 1e-3: {large_diff_count}/{total_elements} ({large_diff_ratio:.2f}%)')
    
    if max_diff < 1e-3:
        print('✅ PASS: Max difference < 1e-3')
    elif max_diff < 1e-2:
        print('⚠️  WARNING: Max difference < 1e-2 but >= 1e-3')
    else:
        print('❌ FAIL: Max difference >= 1e-2')
        
except Exception as e:
    print(f'Error during comparison: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" || exit 1
fi

# 使用 verify_encoder_output.py 进行完整验证（对比 PyTorch）
echo ""
echo -e "${GREEN}Running full verification with PyTorch reference...${NC}"

VERIFY_CMD="python3 ${VERIFY_SCRIPT} \"${INPUT_FILE_ABS_FOR_COMPARE}\" \"${OUTPUT_FILE_ABS_FOR_COMPARE}\""
if [ -n "${WAV_FILE}" ]; then
    VERIFY_CMD="${VERIFY_CMD} \"${WAV_FILE}\""
fi

cd "${ROOT_DIR}"
eval "${VERIFY_CMD}"

echo ""
echo -e "${GREEN}=========================================="
echo "All steps completed!"
echo "==========================================${NC}"
