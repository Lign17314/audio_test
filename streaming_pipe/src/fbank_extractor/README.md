# FBank Extractor Directory

This directory contains the C++ implementation of fbank feature extraction, matching the Python `wav_frontend.py` behavior.

## Files

- **fbank_extractor.h/cpp**: Main fbank extractor class
- **wav_reader.h/cpp**: WAV file I/O utilities  
- **test_fbank_extractor.cc**: Test/example program
- **FBANK_EXTRACTOR_README.md**: Detailed documentation

## Building

### Using CMake (Recommended)

From the parent directory (`tflite_micro_cpp`):

```bash
mkdir -p build
cd build
cmake ..
make test_fbank_extractor
```

Or build all targets:

```bash
make -j$(nproc)
```

The executable will be in `build/test_fbank_extractor`.


## Usage

### Quick Start (Recommended)

使用一键运行脚本（自动编译和运行）：

```bash
cd tflite_micro_cpp/fbank_extractor
./run_fbank_extractor.sh [input.wav] [output.npy] [cmvn_file]
```

示例：
```bash
# 使用默认参数
./run_fbank_extractor.sh

# 指定输入输出文件
./run_fbank_extractor.sh input.wav output_fbank.npy

# 完整参数
./run_fbank_extractor.sh input.wav output_fbank.npy /path/to/cmvn_file
```

### Manual Usage

#### With CMake build:

```bash
cd tflite_micro_cpp/build
./test_fbank_extractor input.wav output_fbank.npy [cmvn_file]
```


## Output

- Shape: `(1, num_frames, 400)`
- Feature dimension: `80 * 5 = 400` (after LFR processing)
- Data type: `float32`

See `FBANK_EXTRACTOR_README.md` for detailed documentation.
