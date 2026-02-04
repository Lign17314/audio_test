# Encoder Runner Directory

This directory contains the TFLite Micro encoder runner programs and scripts.

## Files

- **run_encoder.cc**: Float32 encoder runner (supports stateful models with cache)
- **run_encoder_16x8.cc**: INT8/16x8 quantized encoder runner
- **run.sh**: One-click script for float32 encoder (build, run, compare)
- **run16x8.sh**: One-click script for 16x8 quantized encoder (build, run, compare)

## Building

### Using CMake (Recommended)

From the parent directory (`tflite_micro_cpp`):

```bash
mkdir -p build
cd build
cmake ..
make run_encoder run_encoder_16x8
```

The build scripts (`run.sh` and `run16x8.sh`) will automatically use CMake if available.

## Usage

### Quick Start

```bash
cd encoder_runner

# Float32 model
./run.sh [model.tflite] [input.npy] [output.npy]

# 16x8 quantized model
./run16x8.sh [model.tflite] [input.npy] [output.npy]
```

### Manual Usage

```bash
cd ../build
./run_encoder model.tflite input_fbank.npy output_logits.npy
./run_encoder_16x8 model.tflite input_fbank.npy output_logits.npy
```

## Features

- **Stateful Model Support**: Handles cache inputs/outputs for streaming inference
- **Quantization Support**: 16x8 quantized model support with proper dequantization
- **Memory Management**: Uses TFLite Micro arena allocator
- **Comparison Tools**: Integrated comparison with Python TFLite and PyTorch

## Output

- Saves logits to `.npy` format (shape: `(1, T, vocab_size)`)
- Supports comparison with reference outputs
- Includes CTC decoding verification
