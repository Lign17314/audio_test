#!/bin/bash
# Install TFLite Micro library and headers to local directory
# This script copies the compiled library and necessary headers from tflite-micro
# to tflite_micro_cpp/tflite_micro_install, making the project self-contained.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_DIR="${SCRIPT_DIR}/tflite_micro_install"
TFLITE_MICRO_DIR="${PARENT_DIR}/tflite-micro"
TFLITE_MICRO_GEN_DIR="${TFLITE_MICRO_DIR}/gen/linux_x86_64_default_gcc"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================="
echo "Installing TFLite Micro Library and Headers"
echo "==========================================${NC}"

# Check if tflite-micro directory exists
if [ ! -d "${TFLITE_MICRO_DIR}" ]; then
    echo -e "${RED}ERROR: TFLite Micro directory not found: ${TFLITE_MICRO_DIR}${NC}"
    exit 1
fi

# Check if library is compiled
TFLITE_MICRO_LIB="${TFLITE_MICRO_GEN_DIR}/lib/libtensorflow-microlite.a"
if [ ! -f "${TFLITE_MICRO_LIB}" ]; then
    echo -e "${RED}ERROR: TFLite Micro library not found: ${TFLITE_MICRO_LIB}${NC}"
    echo "Please compile the library first:"
    echo "  cd ${TFLITE_MICRO_DIR}"
    echo "  make -f tensorflow/lite/micro/tools/make/Makefile TARGET=linux microlite"
    exit 1
fi

echo -e "${GREEN}Found TFLite Micro library: ${TFLITE_MICRO_LIB}${NC}"

# Create install directory structure
echo ""
echo -e "${GREEN}Creating install directory structure...${NC}"
mkdir -p "${INSTALL_DIR}/lib"
mkdir -p "${INSTALL_DIR}/include/tensorflow/lite"
mkdir -p "${INSTALL_DIR}/include/tensorflow/lite/micro"
mkdir -p "${INSTALL_DIR}/include/tensorflow/lite/schema"
mkdir -p "${INSTALL_DIR}/include/tensorflow/lite/core"
mkdir -p "${INSTALL_DIR}/include/tensorflow/lite/core/api"
mkdir -p "${INSTALL_DIR}/include/tensorflow/lite/core/c"
mkdir -p "${INSTALL_DIR}/include/tensorflow/lite/kernels"
mkdir -p "${INSTALL_DIR}/include/tensorflow/lite/kernels/internal"
# Note: portable_type_to_tflitetype.h is a file, not a directory
mkdir -p "${INSTALL_DIR}/include/downloads/gemmlowp"
mkdir -p "${INSTALL_DIR}/include/downloads/flatbuffers/include"
mkdir -p "${INSTALL_DIR}/include/downloads/kissfft"
mkdir -p "${INSTALL_DIR}/include/downloads/ruy"
mkdir -p "${INSTALL_DIR}/genfiles"

# Copy library
echo ""
echo -e "${GREEN}Copying library...${NC}"
rsync -av "${TFLITE_MICRO_LIB}" "${INSTALL_DIR}/lib/" || cp "${TFLITE_MICRO_LIB}" "${INSTALL_DIR}/lib/"

# Copy generated files
echo ""
echo -e "${GREEN}Copying generated files...${NC}"
if [ -d "${TFLITE_MICRO_GEN_DIR}/genfiles" ]; then
    rsync -av "${TFLITE_MICRO_GEN_DIR}/genfiles/" "${INSTALL_DIR}/genfiles/" || \
    cp -r "${TFLITE_MICRO_GEN_DIR}/genfiles/"* "${INSTALL_DIR}/genfiles/" 2>/dev/null || true
fi

# Copy tensorflow/lite headers (recursively, preserving directory structure)
echo ""
echo -e "${GREEN}Copying TensorFlow Lite headers...${NC}"
# Use find to copy all .h files while preserving directory structure
find "${TFLITE_MICRO_DIR}/tensorflow/lite" -name "*.h" -type f | while read file; do
    rel_path="${file#${TFLITE_MICRO_DIR}/}"
    dest_file="${INSTALL_DIR}/include/${rel_path}"
    dest_dir=$(dirname "${dest_file}")
    mkdir -p "${dest_dir}"
    cp "${file}" "${dest_file}"
done

# Copy tensorflow/compiler headers if they exist (for c_api_types.h dependency)
if [ -d "${TFLITE_MICRO_DIR}/tensorflow/compiler" ]; then
    echo -e "${GREEN}Copying TensorFlow compiler headers...${NC}"
    find "${TFLITE_MICRO_DIR}/tensorflow/compiler" -name "*.h" -type f | while read file; do
        rel_path="${file#${TFLITE_MICRO_DIR}/}"
        dest_file="${INSTALL_DIR}/include/${rel_path}"
        dest_dir=$(dirname "${dest_file}")
        mkdir -p "${dest_dir}"
        cp "${file}" "${dest_file}"
    done
fi

# Copy signal headers if they exist (for micro_mutable_op_resolver.h dependency)
if [ -d "${TFLITE_MICRO_DIR}/signal" ]; then
    echo -e "${GREEN}Copying signal headers...${NC}"
    find "${TFLITE_MICRO_DIR}/signal" -name "*.h" -type f | while read file; do
        rel_path="${file#${TFLITE_MICRO_DIR}/}"
        dest_file="${INSTALL_DIR}/include/${rel_path}"
        dest_dir=$(dirname "${dest_file}")
        mkdir -p "${dest_dir}"
        cp "${file}" "${dest_file}"
    done
fi

# Copy downloads (dependencies)
echo ""
echo -e "${GREEN}Copying dependency headers...${NC}"

# gemmlowp (including fixedpoint)
if [ -d "${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/gemmlowp" ]; then
    echo -e "${GREEN}  Copying gemmlowp headers...${NC}"
    find "${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/gemmlowp" -name "*.h" -type f | while read file; do
        rel_path="${file#${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/}"
        dest_file="${INSTALL_DIR}/include/downloads/${rel_path}"
        dest_dir=$(dirname "${dest_file}")
        mkdir -p "${dest_dir}"
        cp "${file}" "${dest_file}"
    done
fi

# flatbuffers
if [ -d "${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include" ]; then
    rsync -av "${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include/" \
        "${INSTALL_DIR}/include/downloads/flatbuffers/include/" || true
fi

# kissfft
if [ -d "${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft" ]; then
    rsync -av --include="*.h" --exclude="*" \
        "${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/" \
        "${INSTALL_DIR}/include/downloads/kissfft/" || true
fi

# ruy
if [ -d "${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/ruy" ]; then
    rsync -av --include="*.h" --exclude="*" \
        "${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/ruy/" \
        "${INSTALL_DIR}/include/downloads/ruy/" || true
fi

# Create a version info file
echo ""
echo -e "${GREEN}Creating version info...${NC}"
cat > "${INSTALL_DIR}/VERSION.txt" << EOF
TFLite Micro Installation
========================
Installed from: ${TFLITE_MICRO_DIR}
Install date: $(date)
Library: libtensorflow-microlite.a
Library size: $(du -h "${INSTALL_DIR}/lib/libtensorflow-microlite.a" | cut -f1)
EOF

# Summary
echo ""
echo -e "${GREEN}=========================================="
echo "Installation Summary"
echo "==========================================${NC}"
echo "Install directory: ${INSTALL_DIR}"
echo "Library: ${INSTALL_DIR}/lib/libtensorflow-microlite.a"
echo "Headers: ${INSTALL_DIR}/include/"
echo "Generated files: ${INSTALL_DIR}/genfiles/"
echo ""
echo -e "${GREEN}Installation completed successfully!${NC}"
echo ""
echo "You can now use the local installation by updating CMakeLists.txt"
echo "or running cmake with the updated paths."
