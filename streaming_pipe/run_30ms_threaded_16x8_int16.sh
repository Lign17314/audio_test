#!/bin/bash
# 运行脚本：Streaming FBank + Encoder (30ms threaded, 16x8 quantized, INT16 optimized)

set -e

echo "========================================"
echo "Streaming FBank - 30ms Threaded Version"
echo "16x8 Quantized + INT16 Optimized"
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
echo "Building streaming_fbank_30ms_threaded_16x8_int16..."
make streaming_fbank_30ms_threaded_16x8_int16 2>&1 | grep -E "error|warning|Building|Linking|Built" || true

if [ ! -f "streaming_fbank_30ms_threaded_16x8_int16" ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build successful!"
echo ""

# 运行
echo "========================================"
echo "Running Streaming FBank (Threaded, 16x8, INT16)..."
echo "========================================"
echo ""

./streaming_fbank_30ms_threaded_16x8_int16

echo ""
echo "========================================"
echo "Success!"
echo "========================================"
echo "Output files:"
ls -lh streaming_fbank_30ms_threaded_16x8_int16*.npy 2>/dev/null || echo "No output files generated"
echo ""

# 可选：验证输出
if [ -f "streaming_fbank_30ms_threaded_16x8_int16_logits.npy" ]; then
    echo "Verifying output shape..."
    python3 -c "import numpy as np; x=np.load('streaming_fbank_30ms_threaded_16x8_int16_logits.npy'); print(f'Shape: {x.shape}')" 2>/dev/null || echo "Python verification skipped"
    python3 ../decode_cpp_output.py streaming_fbank_30ms_threaded_16x8_int16_logits.npy
fi
