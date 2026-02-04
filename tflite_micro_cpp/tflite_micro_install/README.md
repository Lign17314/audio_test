# TFLite Micro Local Installation

This directory contains a local installation of the TFLite Micro library and headers, making the project self-contained and independent of the external `tflite-micro` directory.

## Directory Structure

```
tflite_micro_install/
├── lib/
│   └── libtensorflow-microlite.a    # Compiled library (2.2MB)
├── include/
│   ├── tensorflow/lite/             # TFLite headers
│   ├── tensorflow/compiler/         # Compiler headers (for c_api_types.h)
│   ├── signal/                      # Signal processing headers
│   └── downloads/                   # Dependency headers
│       ├── gemmlowp/                # Including fixedpoint/
│       ├── flatbuffers/include/
│       ├── kissfft/
│       └── ruy/
├── genfiles/                         # Generated files
└── VERSION.txt                       # Installation info
```

## Installation

To install/update the TFLite Micro library and headers:

```bash
cd tflite_micro_cpp
./install_tflite_micro.sh
```

This script will:
1. Copy the compiled library from `../tflite-micro/gen/linux_x86_64_default_gcc/lib/`
2. Copy necessary headers from `../tflite-micro/tensorflow/lite/`
3. Copy compiler headers from `../tflite-micro/tensorflow/compiler/`
4. Copy signal headers from `../tflite-micro/signal/`
5. Copy dependency headers (gemmlowp, flatbuffers, kissfft, ruy)
6. Copy generated files

**Total size**: ~29MB (1782 files)

## Usage

The CMakeLists.txt automatically detects and uses this local installation if it exists. If not found, it falls back to using the external `tflite-micro` directory.

When using local installation, CMake will show:
```
-- Using local TFLite Micro installation: /path/to/tflite_micro_install
-- Found TFLite Micro library: /path/to/tflite_micro_install/lib/libtensorflow-microlite.a
```

## Benefits

- **Self-contained**: No need to maintain external tflite-micro directory
- **Portable**: Can be easily packaged and distributed
- **Version control**: Can track the exact version of TFLite Micro used
- **Faster builds**: Reduced dependency on external paths
- **Independent**: Project can be moved/copied without external dependencies

## Updating

To update the installation after recompiling tflite-micro:

```bash
cd tflite_micro_cpp
./install_tflite_micro.sh
```

The script uses `rsync`/`cp` for efficient incremental updates.

## Version Control

By default, this directory is **not** ignored by git (see `.gitignore`). You can:

1. **Include in git** (recommended for reproducibility):
   - Commit the entire `tflite_micro_install/` directory
   - Ensures everyone uses the same TFLite Micro version

2. **Exclude from git** (to reduce repo size):
   - Uncomment `tflite_micro_install/` in `.gitignore`
   - Each developer runs `./install_tflite_micro.sh` after cloning
