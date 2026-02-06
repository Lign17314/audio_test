#!/bin/bash
# 运行脚本：Streaming FBank + Encoder (30ms threaded, 16x8 quantized)

set -e

echo "========================================"
echo "Streaming FBank - 30ms Threaded Version (16x8 Quantized)"
echo "========================================"
echo ""

# 创建build目录
if [ ! -d "build" ]; then
    mkdir -p build
fi

# 编译
echo "Creating build directory..."
cd build
echo "Configuring with CMake..."
cmake .. > /dev/null 2>&1
echo "Building..."
make streaming_fbank_30ms_threaded_16x8 2>&1 | grep -E "error|warning|Building|Linking|Built" || true

if [ ! -f "streaming_fbank_30ms_threaded_16x8" ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "Build successful!"
echo ""

# 运行
echo "========================================"
echo "Running Streaming FBank (Threaded, 16x8)..."
echo "========================================"
echo ""

./streaming_fbank_30ms_threaded_16x8

echo ""
echo "========================================"
echo "Success!"
echo "========================================"
echo "Output files:"
ls -lh streaming_fbank_30ms_threaded*.npy 2>/dev/null || echo "No output files generated"
echo ""
