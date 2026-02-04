# CTC Decoder Directory

This directory contains the C++ implementation of CTC decoding for keyword spotting, matching the Python `ctc.py` behavior.

## Files

- **ctc_decoder.h/cpp**: CTC prefix beam search decoder implementation
- **test_ctc_decoder.cc**: Test/example program
- **run_ctc_decoder.sh**: One-click run and test script

## Overview

The CTC decoder implements:
1. **CTC Prefix Beam Search**: Beam search algorithm for CTC decoding
2. **Keyword Detection**: Detects keywords in decoded sequences
3. **Confidence Scoring**: Computes confidence scores for detected keywords

## Building

### Using CMake

From the parent directory (`tflite_micro_cpp`):

```bash
mkdir -p build
cd build
cmake ..
make test_ctc_decoder
```


## Usage

### Quick Start

```bash
cd tflite_micro_cpp/ctc_decoder

# First, export token_list and seg_dict (if not already done)
python3 export_token_data.py .

# Then run decoder
./run_ctc_decoder.sh <logits.npy> <keywords> [token_list.txt] [seg_dict.txt]
```

### Manual Usage

```bash
cd tflite_micro_cpp/build
./test_ctc_decoder logits.npy "小云小云" token_list.txt seg_dict.txt
```

### Export Token Data

Before using the decoder, you need to export token_list and seg_dict:

```bash
cd tflite_micro_cpp/ctc_decoder
python3 export_token_data.py [output_dir]
```

This will create:
- `token_list.txt`: One token per line
- `seg_dict.txt`: Format: `word token1 token2 ...`

## Configuration

- **Vocab size**: 2599 (default, can be inferred from logits)
- **Blank ID**: 0 (default)
- **Score beam size**: 3
- **Path beam size**: 20

## Output

Returns a `DecodeResult` structure:
- `hit`: Whether keyword was detected (bool)
- `keyword`: Detected keyword string (if hit=true)
- `score`: Confidence score (if hit=true)

## Implementation Notes

- Matches Python `KwsCtcPrefixDecoder` behavior
- Uses CTC prefix beam search algorithm
- Supports multiple keywords (comma-separated)
- **Token list and segmentation dictionary loading**: Implemented
  - Loads `token_list.txt` (one token per line)
  - Loads `seg_dict.txt` (format: `word token1 token2 ...`)
  - Use `export_token_data.py` to generate these files from FunASR model

## References

- Python implementation: `ctc.py` `KwsCtcPrefixDecoder`
- CTC algorithm: Connectionist Temporal Classification
