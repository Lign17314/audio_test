#!/bin/bash
# FBank Extractor 一键运行和测试脚本
# 功能：编译（如需要）、运行 fbank 提取、对比测试结果
#
# 使用方法:
#   ./run_fbank_extractor.sh [input.wav] [output.npy] [cmvn_file]
#
# 示例:
#   ./run_fbank_extractor.sh
#   ./run_fbank_extractor.sh input.wav output_fbank.npy
#   ./run_fbank_extractor.sh input.wav output_fbank.npy /path/to/cmvn_file

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PARENT_DIR}/build"

# 解析参数
SKIP_BUILD=false
POSITIONAL_ARGS=()

# 检查帮助参数（在解析之前）
for arg in "$@"; do
    if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
    echo "FBank Extractor Runner and Tester"
    echo ""
    echo "Usage: $0 [input.wav] [output.npy] [cmvn_file]"
    echo ""
    echo "Arguments:"
    echo "  input.wav  - 输入 WAV 文件路径 (默认: ../../example_kws/wav/20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_21.wav)"
    echo "  output.npy - 输出 fbank 特征文件 (默认: output_fbank.npy)"
    echo "  cmvn_file  - CMVN 归一化文件路径 (默认: /root/volume/ctc/speech_charctc_kws_phone-xiaoyun/funasr/am.mvn.dim80_l2r2)"
    echo ""
    echo "Options:"
    echo "  --skip-build  - 跳过编译步骤（即使可执行文件不存在）"
    echo "  --force-build - 强制重新编译"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 input.wav output_fbank.npy"
    echo "  $0 input.wav output_fbank.npy /path/to/cmvn_file"
    echo "  $0 --skip-build input.wav output_fbank.npy"
    exit 0
    fi
done

# 解析选项参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --force-build)
            SKIP_BUILD=false
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

# 解析位置参数
# 默认使用项目根目录下的测试音频
DEFAULT_WAV="../../example_kws/wav/20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_21.wav"
WAV_FILE="${1:-${DEFAULT_WAV}}"
OUTPUT_FILE="${2:-output_fbank.npy}"
CMVN_FILE="${3:-/root/volume/ctc/speech_charctc_kws_phone-xiaoyun/funasr/am.mvn.dim80_l2r2}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "FBank Extractor Runner and Tester"
echo "=========================================="
echo "Input WAV: ${WAV_FILE}"
echo "Output: ${OUTPUT_FILE}"
if [ -n "${CMVN_FILE}" ]; then
    echo "CMVN file: ${CMVN_FILE}"
fi
echo ""

# 检查输入文件是否存在
if [ ! -f "${WAV_FILE}" ]; then
    # 尝试从项目根目录解析路径
    if [ ! -f "${PARENT_DIR}/${WAV_FILE}" ]; then
        echo -e "${RED}ERROR: Input WAV file not found: ${WAV_FILE}${NC}"
        echo "  Also tried: ${PARENT_DIR}/${WAV_FILE}"
        exit 1
    else
        WAV_FILE="${PARENT_DIR}/${WAV_FILE}"
    fi
fi

# 创建 build 目录
if [ ! -d "${BUILD_DIR}" ]; then
    echo -e "${YELLOW}Creating build directory...${NC}"
    mkdir -p "${BUILD_DIR}"
fi

# ==========================================
# 1. 编译
# ==========================================
if [ "${SKIP_BUILD}" = false ]; then
    # 检查是否需要编译
    NEED_BUILD=false
    EXECUTABLE="${BUILD_DIR}/test_fbank_extractor"
    
    if [ ! -f "${EXECUTABLE}" ]; then
        NEED_BUILD=true
    else
        # 检查源文件和头文件是否有更新
        SRC_FILES=(
            "${SCRIPT_DIR}/test_fbank_extractor.cc"
            "${SCRIPT_DIR}/fbank_extractor.cc"
            "${SCRIPT_DIR}/fbank_extractor.h"
            "${SCRIPT_DIR}/wav_reader.cc"
            "${SCRIPT_DIR}/wav_reader.h"
        )
        
        # 检查 CMakeLists.txt 是否更新（需要重新运行 cmake）
        if [ -f "${PARENT_DIR}/CMakeLists.txt" ] && [ "${PARENT_DIR}/CMakeLists.txt" -nt "${EXECUTABLE}" ]; then
            NEED_BUILD=true
        fi
        
        # 检查源文件是否有更新
        if [ "${NEED_BUILD}" = false ]; then
            for src_file in "${SRC_FILES[@]}"; do
                if [ -f "${src_file}" ] && [ "${src_file}" -nt "${EXECUTABLE}" ]; then
                    NEED_BUILD=true
                    break
                fi
            done
        fi
    fi
    
    # 只在需要编译时才显示编译步骤
    if [ "${NEED_BUILD}" = true ]; then
        echo -e "${BLUE}=========================================="
        echo "Step 1: Building"
        echo "==========================================${NC}"
        
        if [ ! -f "${EXECUTABLE}" ]; then
            echo -e "${YELLOW}Executable not found, will build...${NC}"
        else
            echo -e "${YELLOW}Source files updated, will rebuild...${NC}"
        fi
        
        echo ""
        echo -e "${GREEN}Building with CMake...${NC}"
        cd "${BUILD_DIR}"
        
        # 检查是否有 CMakeCache.txt，如果没有则运行 cmake
        # 或者如果 CMakeLists.txt 更新了，也需要重新运行 cmake
        if [ ! -f "CMakeCache.txt" ] || [ "${PARENT_DIR}/CMakeLists.txt" -nt "CMakeCache.txt" ]; then
            echo "Running cmake..."
            cmake "${PARENT_DIR}"
        fi
        
        echo "Running make..."
        make -j$(nproc) test_fbank_extractor
        
        cd "${SCRIPT_DIR}"
        echo -e "${GREEN}Build completed!${NC}"
        echo ""
    fi
    # 如果不需要编译，不显示任何编译相关的输出
else
    echo -e "${YELLOW}Build step skipped (--skip-build)${NC}"
    echo ""
fi

# 检查可执行文件
if [ ! -f "${BUILD_DIR}/test_fbank_extractor" ]; then
    echo -e "${RED}ERROR: Executable not found: ${BUILD_DIR}/test_fbank_extractor${NC}"
    exit 1
fi

# ==========================================
# 2. 运行 FBank 提取
# ==========================================
echo -e "${BLUE}=========================================="
echo "Step 2: Running FBank Extraction"
echo "==========================================${NC}"

cd "${BUILD_DIR}"

# 转换相对路径为绝对路径
if [[ "${WAV_FILE}" != /* ]]; then
    WAV_FILE_ABS="$(cd "${SCRIPT_DIR}" && readlink -f "${WAV_FILE}" 2>/dev/null || echo "${SCRIPT_DIR}/${WAV_FILE}")"
else
    WAV_FILE_ABS="${WAV_FILE}"
fi

if [[ "${OUTPUT_FILE}" != /* ]]; then
    OUTPUT_FILE_ABS="${BUILD_DIR}/${OUTPUT_FILE}"
else
    OUTPUT_FILE_ABS="${OUTPUT_FILE}"
fi

echo -e "${GREEN}Running fbank extractor...${NC}"
echo "Command: ./test_fbank_extractor \"${WAV_FILE_ABS}\" \"${OUTPUT_FILE_ABS}\" ${CMVN_FILE:+${CMVN_FILE}}"

if [ -n "${CMVN_FILE}" ]; then
    ./test_fbank_extractor "${WAV_FILE_ABS}" "${OUTPUT_FILE_ABS}" "${CMVN_FILE}"
else
    ./test_fbank_extractor "${WAV_FILE_ABS}" "${OUTPUT_FILE_ABS}"
fi

# 检查输出文件
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
    # Try to read as numpy array (simplified format)
    with open('${OUTPUT_FILE_ABS}', 'rb') as f:
        batch_size = np.frombuffer(f.read(4), dtype=np.int32)[0]
        num_frames = np.frombuffer(f.read(4), dtype=np.int32)[0]
        feature_dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.frombuffer(f.read(), dtype=np.float32)
        arr = data.reshape(batch_size, num_frames, feature_dim)
    
    print(f'Output shape: {arr.shape}')
    print(f'Output dtype: {arr.dtype}')
    print(f'Output size: {arr.nbytes / 1024 / 1024:.2f} MB')
    print(f'Number of frames: {num_frames}')
    print(f'Feature dimension: {feature_dim}')
    print(f'Duration: {num_frames * 0.03:.2f} seconds (assuming 30ms per frame after LFR)')
    
    # Print statistics
    print(f'')
    print(f'Statistics:')
    print(f'  Min value: {np.min(arr):.6f}')
    print(f'  Max value: {np.max(arr):.6f}')
    print(f'  Mean value: {np.mean(arr):.6f}')
    print(f'  Std value: {np.std(arr):.6f}')
except Exception as e:
    print(f'Could not read output file: {e}')
    import traceback
    traceback.print_exc()
" 2>/dev/null || true
fi

# ==========================================
# 3. 对比测试（如果 Python 版本存在）
# ==========================================
echo ""
echo -e "${BLUE}=========================================="
echo "Step 3: Comparison Test (Optional)"
echo "==========================================${NC}"

# 检查是否有 Python 版本的输出可以对比
PYTHON_OUTPUT="${BUILD_DIR}/python_fbank.npy"
if [ -f "${PYTHON_OUTPUT}" ]; then
    echo ""
    echo -e "${GREEN}Comparing with Python output...${NC}"
    
    python3 -c "
import numpy as np
import sys

try:
    # Read C++ output
    with open('${OUTPUT_FILE_ABS}', 'rb') as f:
        batch_size_cpp = np.frombuffer(f.read(4), dtype=np.int32)[0]
        num_frames_cpp = np.frombuffer(f.read(4), dtype=np.int32)[0]
        feature_dim_cpp = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data_cpp = np.frombuffer(f.read(), dtype=np.float32)
        cpp_output = data_cpp.reshape(batch_size_cpp, num_frames_cpp, feature_dim_cpp)
    
    # Read Python output
    python_output = np.load('${PYTHON_OUTPUT}')
    
    print(f'C++ output shape: {cpp_output.shape}')
    print(f'Python output shape: {python_output.shape}')
    
    # Ensure same shape
    min_T = min(cpp_output.shape[1], python_output.shape[1])
    min_D = min(cpp_output.shape[2], python_output.shape[2])
    
    cpp_trimmed = cpp_output[:, :min_T, :min_D]
    python_trimmed = python_output[:, :min_T, :min_D]
    
    diff = np.abs(cpp_trimmed - python_trimmed)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f'')
    print(f'Comparison Results:')
    print(f'  Max difference: {max_diff:.6e}')
    print(f'  Mean difference: {mean_diff:.6e}')
    
    # Relative error
    ref_abs = np.abs(python_trimmed)
    relative_diff = diff / (ref_abs + 1e-10)
    max_relative_diff = np.max(relative_diff)
    mean_relative_diff = np.mean(relative_diff)
    
    print(f'  Max relative difference: {max_relative_diff:.6e}')
    print(f'  Mean relative difference: {mean_relative_diff:.6e}')
    
    if max_diff < 1e-3:
        print('  ✅ PASS: Max difference < 1e-3')
    elif max_diff < 1e-2:
        print('  ⚠️  WARNING: Max difference < 1e-2 but >= 1e-3')
    else:
        print('  ❌ FAIL: Max difference >= 1e-2')
        
except Exception as e:
    print(f'Error during comparison: {e}')
    import traceback
    traceback.print_exc()
" || echo -e "${YELLOW}Comparison skipped (Python output not found or error)${NC}"
else
    echo ""
    echo -e "${YELLOW}Python output not found for comparison.${NC}"
    echo "To generate Python output for comparison, run:"
    echo "  python prepare_fbank_features.py <input.wav> <output_dir>"
    echo "  Then copy the output to: ${PYTHON_OUTPUT}"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "All steps completed!"
echo "==========================================${NC}"
