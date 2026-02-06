#!/bin/bash
# CTC Decoder 一键运行和测试脚本
# 功能：编译（如需要）、运行 CTC 解码、显示识别结果
#
# 使用方法:
#   ./run_ctc_decoder.sh <logits.npy> <keywords> [--skip-build]
#
# 示例:
#   ./run_ctc_decoder.sh output_logits.npy "小云小云"
#   ./run_ctc_decoder.sh output_logits.npy "小云小云" --skip-build

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PARENT_DIR}/build"

# 解析参数
SKIP_BUILD=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -h|--help)
            echo "CTC Decoder Runner and Tester"
            echo ""
            echo "Usage: $0 <logits.npy> <keywords> [token_list.txt] [seg_dict.txt] [--skip-build]"
            echo ""
            echo "Arguments:"
            echo "  logits.npy    - 输入 logits 文件路径（numpy 格式）"
            echo "  keywords      - 关键词（逗号分隔，如 \"小云小云\"）"
            echo "  token_list.txt - Token 列表文件（可选，默认: token_list.txt）"
            echo "  seg_dict.txt   - 分词字典文件（可选，默认: seg_dict.txt）"
            echo ""
            echo "Options:"
            echo "  --skip-build - 跳过编译步骤"
            echo ""
            echo "Examples:"
            echo "  $0 output_logits.npy \"小云小云\""
            echo "  $0 output_logits.npy \"小云小云\" token_list.txt seg_dict.txt"
            echo "  $0 output_logits.npy \"小云小云\" --skip-build"
            exit 0
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

set -- "${POSITIONAL_ARGS[@]}"

if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <logits.npy> <keywords> [token_list.txt] [seg_dict.txt] [--skip-build]"
    exit 1
fi

LOGITS_FILE="$1"
KEYWORDS="$2"
TOKEN_LIST_FILE="${3:-token_list.txt}"
SEG_DICT_FILE="${4:-seg_dict.txt}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "CTC Decoder Runner and Tester"
echo "=========================================="
echo "Logits file: ${LOGITS_FILE}"
echo "Keywords: ${KEYWORDS}"
echo "Token list: ${TOKEN_LIST_FILE}"
echo "Seg dict: ${SEG_DICT_FILE}"
echo ""

# 检查输入文件是否存在
if [ ! -f "${LOGITS_FILE}" ]; then
    # 尝试从 build 目录查找
    if [ -f "${BUILD_DIR}/${LOGITS_FILE}" ]; then
        LOGITS_FILE="${BUILD_DIR}/${LOGITS_FILE}"
    else
        echo -e "${RED}ERROR: Logits file not found: ${LOGITS_FILE}${NC}"
        exit 1
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
    EXECUTABLE="${BUILD_DIR}/test_ctc_decoder"
    
    if [ ! -f "${EXECUTABLE}" ]; then
        NEED_BUILD=true
    else
        SRC_FILES=(
            "${SCRIPT_DIR}/test_ctc_decoder.cc"
            "${SCRIPT_DIR}/ctc_decoder.cc"
            "${SCRIPT_DIR}/ctc_decoder.h"
            "${PARENT_DIR}/CMakeLists.txt"
        )
        
        for src_file in "${SRC_FILES[@]}"; do
            if [ -f "${src_file}" ] && [ "${src_file}" -nt "${EXECUTABLE}" ]; then
                NEED_BUILD=true
                break
            fi
        done
    fi
    
    if [ "${NEED_BUILD}" = true ]; then
        echo -e "${BLUE}=========================================="
        echo "Step 1: Building"
        echo "==========================================${NC}"
        echo ""
        echo -e "${GREEN}Building with CMake...${NC}"
        cd "${BUILD_DIR}"
        
        if [ ! -f "CMakeCache.txt" ] || [ "${PARENT_DIR}/CMakeLists.txt" -nt "CMakeCache.txt" ]; then
            echo "Running cmake..."
            cmake "${PARENT_DIR}"
        fi
        
        echo "Running make..."
        make -j$(nproc) test_ctc_decoder
        
        cd "${SCRIPT_DIR}"
        echo -e "${GREEN}Build completed!${NC}"
        echo ""
    fi
fi

# 检查可执行文件
if [ ! -f "${BUILD_DIR}/test_ctc_decoder" ]; then
    echo -e "${RED}ERROR: Executable not found: ${BUILD_DIR}/test_ctc_decoder${NC}"
    echo "Please build first or remove --skip-build option"
    exit 1
fi

# ==========================================
# 2. 运行 CTC 解码
# ==========================================
echo -e "${BLUE}=========================================="
echo "Step 2: Running CTC Decoding"
echo "==========================================${NC}"

cd "${BUILD_DIR}"

# 转换相对路径为绝对路径
if [[ "${LOGITS_FILE}" != /* ]]; then
    LOGITS_FILE_ABS="$(cd "${SCRIPT_DIR}" && readlink -f "${LOGITS_FILE}" 2>/dev/null || echo "${SCRIPT_DIR}/${LOGITS_FILE}")"
else
    LOGITS_FILE_ABS="${LOGITS_FILE}"
fi

echo -e "${GREEN}Running CTC decoder...${NC}"

# Convert relative paths to absolute paths for token files
if [[ "${TOKEN_LIST_FILE}" != /* ]]; then
    TOKEN_LIST_FILE_ABS="$(cd "${SCRIPT_DIR}" && readlink -f "${TOKEN_LIST_FILE}" 2>/dev/null || echo "${SCRIPT_DIR}/${TOKEN_LIST_FILE}")"
else
    TOKEN_LIST_FILE_ABS="${TOKEN_LIST_FILE}"
fi

if [[ "${SEG_DICT_FILE}" != /* ]]; then
    SEG_DICT_FILE_ABS="$(cd "${SCRIPT_DIR}" && readlink -f "${SEG_DICT_FILE}" 2>/dev/null || echo "${SCRIPT_DIR}/${SEG_DICT_FILE}")"
else
    SEG_DICT_FILE_ABS="${SEG_DICT_FILE}"
fi

echo "Command: ./test_ctc_decoder \"${LOGITS_FILE_ABS}\" \"${KEYWORDS}\" \"${TOKEN_LIST_FILE_ABS}\" \"${SEG_DICT_FILE_ABS}\""

./test_ctc_decoder "${LOGITS_FILE_ABS}" "${KEYWORDS}" "${TOKEN_LIST_FILE_ABS}" "${SEG_DICT_FILE_ABS}"

echo ""
echo -e "${GREEN}=========================================="
echo "Decoding completed!"
echo "==========================================${NC}"
