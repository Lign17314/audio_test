#ifndef WAV_READER_H
#define WAV_READER_H

#include <vector>
#include <string>

struct WavData {
    std::vector<float> samples;  // Normalized to [-1, 1]
    int sample_rate;
    int num_channels;
    int bits_per_sample;
};

class WavReader {
public:
    static WavData read_wav(const std::string& filename);
    static bool write_wav(const std::string& filename, 
                         const std::vector<float>& samples,
                         int sample_rate,
                         int num_channels = 1);
};

#endif // WAV_READER_H
