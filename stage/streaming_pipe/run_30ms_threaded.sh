#!/bin/bash

echo "========================================"
echo "Streaming FBank - 30ms Threaded Version"
echo "========================================"
echo ""

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. > /dev/null 2>&1

# Build
echo "Building..."
make streaming_fbank_30ms_threaded

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful!"
echo ""

# Run
echo "========================================"
echo "Running Streaming FBank (Threaded)..."
echo "========================================"
echo ""

./streaming_fbank_30ms_threaded

echo ""
echo "========================================"
echo "Success!"
echo "========================================"
echo "Output file:"
ls -lh streaming_fbank_30ms_threaded.npy 2>/dev/null || echo "No output file found"
echo ""
