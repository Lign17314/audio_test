#!/bin/bash

# Streaming FBank Extractor - Build and Run Script
# This script builds and runs the streaming fbank extractor without any parameters

set -e  # Exit on error

echo "========================================"
echo "Streaming FBank Extractor"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create build directory
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Building..."
make -j$(nproc)

# Check if build succeeded
if [ ! -f "streaming_pipe" ]; then
    echo ""
    echo "ERROR: Build failed - executable not found"
    exit 1
fi

echo ""
echo "Build successful!"
echo ""

# Run the program
echo "========================================"
echo "Running Streaming FBank Extractor..."
echo "========================================"
echo ""

./streaming_pipe

# Check if output file was created
if [ -f "streaming_fbank_output.npy" ]; then
    echo ""
    echo "========================================"
    echo "Success!"
    echo "========================================"
    echo "Output file: streaming_fbank_output.npy"
    
    # Show file size
    FILE_SIZE=$(du -h streaming_fbank_output.npy | cut -f1)
    echo "File size: $FILE_SIZE"
    
    # Optional: Compare with batch processing result if available
    BATCH_FILE="../../features/20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_21_fbank.npy"
    if [ -f "$BATCH_FILE" ]; then
        echo ""
        echo "Batch processing reference file found:"
        echo "  $BATCH_FILE"
        echo ""
        echo "You can compare the results using Python:"
        echo "  python3 -c \"import numpy as np; s=np.load('streaming_fbank_output.npy'); b=np.load('$BATCH_FILE'); print(f'Streaming: {s.shape}, Batch: {b.shape}'); print(f'Max diff: {np.abs(s-b).max()}')\""
    fi
else
    echo ""
    echo "WARNING: Output file not created"
    exit 1
fi

echo ""
