# FBank Feature Extractor (C++)

C++ implementation of fbank feature extraction matching Python `wav_frontend.py` behavior.

## Overview

This implementation replicates the fbank feature extraction pipeline from `a2.py` and `wav_frontend.py`:

1. **Audio Loading**: Read WAV file (16kHz, mono)
2. **Sample Scaling**: Multiply by 32768 (1 << 15) if `upsacle_samples=true`
3. **FBank Extraction**: Extract 80 mel-filterbank features per frame
   - Frame length: 25ms
   - Frame shift: 10ms
   - Window: Hamming
   - Dither: 1.0
4. **LFR (Low Frame Rate) Processing**: Splice 5 frames, take every 3rd
   - Input: (T, 80)
   - Output: (T_lfr, 400) where T_lfr = ceil(T/3)
5. **CMVN (Cepstral Mean and Variance Normalization)**: Apply mean and variance normalization

## Configuration

Matches `a2.py` `WavFrontend_config`:

```cpp
FBankConfig config;
config.fs = 16000;              // Sample rate
config.window = "hamming";      // Window type
config.n_mels = 80;             // Number of mel bins
config.frame_length = 25;       // Frame length in ms
config.frame_shift = 10;        // Frame shift in ms
config.lfr_m = 5;               // LFR splice frames
config.lfr_n = 3;               // LFR skip frames
config.dither = 1.0f;           // Dithering factor
config.snip_edges = true;       // Snip edges
config.upsacle_samples = true;  // Scale samples by 32768
config.cmvn_file = "...";       // CMVN file path (optional)
```

## Files

- `fbank_extractor.h/cpp`: Main fbank extractor class
- `wav_reader.h/cpp`: WAV file I/O utilities
- `test_fbank_extractor.cc`: Example/test program

## Building

### Using CMake (Recommended)

Add to `CMakeLists.txt`:

```cmake
add_executable(test_fbank_extractor
    test_fbank_extractor.cc
    fbank_extractor.cc
    wav_reader.cc
)
target_link_libraries(test_fbank_extractor m)
```

## Usage

### Basic Usage

```bash
./test_fbank_extractor input.wav output_fbank.npy
```

### With CMVN

```bash
./test_fbank_extractor input.wav output_fbank.npy \
    /path/to/am.mvn.dim80_l2r2
```

## Output Format

- Shape: `(1, num_frames, 400)`
- Feature dimension: `80 * 5 = 400` (after LFR)
- Data type: `float32`

## Comparison with Python

To verify correctness, compare outputs:

```python
import numpy as np

# C++ output
cpp_features = np.load("output_fbank.npy")

# Python output (from prepare_fbank_features.py)
python_features = np.load("python_fbank.npy")

# Compare
diff = np.abs(cpp_features - python_features)
print(f"Max diff: {np.max(diff)}")
print(f"Mean diff: {np.mean(diff)}")
```

## Implementation Notes

### FFT Implementation

The current implementation uses a simple DFT (slow but no dependencies). For production use:

1. **FFTW** (recommended for performance):
   ```cpp
   #define USE_FFTW
   // Link with -lfftw3f
   ```

2. **KissFFT** (lightweight):
   ```cpp
   #define USE_KISSFFT
   // Include kiss_fft.h and link
   ```

### Mel Filterbank

The mel filterbank creation matches Kaldi's implementation:
- Mel scale: `mel = 2595 * log10(1 + hz / 700)`
- Triangular filters with proper frequency bin mapping

### LFR Processing

Matches Python `apply_lfr`:
- Left padding: repeat first frame `(lfr_m - 1) / 2` times
- Splice `lfr_m` consecutive frames
- Take every `lfr_n`-th spliced frame
- Right padding: repeat last frame if needed

### CMVN

CMVN file format parsing matches Python `load_cmvn`:
- Reads `<AddShift>` (means) and `<Rescale>` (vars) sections
- Applies: `(x + mean) * var`

## Limitations

1. **Resampling**: WAV file must be 16kHz (resampling not implemented)
2. **FFT**: Uses simple DFT (slow for long audio)
3. **WAV Format**: Supports 16-bit and 32-bit PCM only
4. **Numpy Format**: Uses simplified binary format (not standard .npy)

## Future Improvements

1. Add proper resampling (e.g., using libsamplerate)
2. Integrate FFTW or KissFFT for faster FFT
3. Support standard numpy .npy format
4. Add streaming/online processing support
5. Optimize for embedded systems (TFLite Micro)

## References

- Python implementation: `wav_frontend.py`
- Configuration: `a2.py` `WavFrontend_config`
- Kaldi fbank: `torchaudio.compliance.kaldi.fbank`
