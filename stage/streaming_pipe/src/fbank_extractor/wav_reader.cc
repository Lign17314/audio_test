#include "wav_reader.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

// Helper function to read little-endian uint32
static uint32_t read_le32(std::ifstream& file) {
    uint8_t bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
}

// Helper function to read little-endian uint16
static uint16_t read_le16(std::ifstream& file) {
    uint8_t bytes[2];
    file.read(reinterpret_cast<char*>(bytes), 2);
    return bytes[0] | (bytes[1] << 8);
}

WavData WavReader::read_wav(const std::string& filename) {
    WavData wav_data;
    wav_data.sample_rate = 0;
    wav_data.num_channels = 0;
    wav_data.bits_per_sample = 0;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open WAV file: " << filename << std::endl;
        return wav_data;
    }
    
    // Read RIFF header
    char riff[4];
    file.read(riff, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0) {
        std::cerr << "Error: Not a valid RIFF file" << std::endl;
        return wav_data;
    }
    
    uint32_t file_size = read_le32(file);  // File size - 8
    
    char wave[4];
    file.read(wave, 4);
    if (std::strncmp(wave, "WAVE", 4) != 0) {
        std::cerr << "Error: Not a valid WAVE file" << std::endl;
        return wav_data;
    }
    
    // Find fmt chunk
    char chunk_id[4];
    uint32_t chunk_size;
    bool found_fmt = false;
    
    while (file.read(chunk_id, 4)) {
        chunk_size = read_le32(file);
        
        if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t audio_format = read_le16(file);
            wav_data.num_channels = read_le16(file);
            wav_data.sample_rate = read_le32(file);
            uint32_t byte_rate = read_le32(file);
            uint16_t block_align = read_le16(file);
            wav_data.bits_per_sample = read_le16(file);
            
            // Skip remaining fmt chunk data if any
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
            found_fmt = true;
            break;
        } else {
            // Skip this chunk
            file.seekg(chunk_size, std::ios::cur);
        }
    }
    
    if (!found_fmt) {
        std::cerr << "Error: fmt chunk not found" << std::endl;
        return wav_data;
    }
    
    // Find data chunk
    uint32_t data_size = 0;
    bool found_data = false;
    
    while (file.read(chunk_id, 4)) {
        chunk_size = read_le32(file);
        
        if (std::strncmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
            break;
        } else {
            // Skip this chunk
            file.seekg(chunk_size, std::ios::cur);
        }
    }
    
    if (!found_data) {
        std::cerr << "Error: data chunk not found" << std::endl;
        return wav_data;
    }
    
    // Read audio data
    int bytes_per_sample = wav_data.bits_per_sample / 8;
    int num_samples = data_size / bytes_per_sample / wav_data.num_channels;
    
    if (num_samples <= 0 || num_samples > 100000000) {  // Sanity check
        std::cerr << "Error: Invalid number of samples: " << num_samples << std::endl;
        std::cerr << "  data_size: " << data_size << std::endl;
        std::cerr << "  bytes_per_sample: " << bytes_per_sample << std::endl;
        std::cerr << "  num_channels: " << wav_data.num_channels << std::endl;
        return wav_data;
    }
    
    wav_data.samples.resize(num_samples * wav_data.num_channels);
    
    if (wav_data.bits_per_sample == 16) {
        std::vector<int16_t> raw_samples(num_samples * wav_data.num_channels);
        file.read(reinterpret_cast<char*>(raw_samples.data()), 
                 num_samples * wav_data.num_channels * sizeof(int16_t));
        
        // Convert to float and normalize to [-1, 1]
        for (size_t i = 0; i < raw_samples.size(); ++i) {
            wav_data.samples[i] = static_cast<float>(raw_samples[i]) / 32768.0f;
        }
    } else if (wav_data.bits_per_sample == 32) {
        std::vector<int32_t> raw_samples(num_samples * wav_data.num_channels);
        file.read(reinterpret_cast<char*>(raw_samples.data()),
                 num_samples * wav_data.num_channels * sizeof(int32_t));
        
        for (size_t i = 0; i < raw_samples.size(); ++i) {
            wav_data.samples[i] = static_cast<float>(raw_samples[i]) / 2147483648.0f;
        }
    } else {
        std::cerr << "Error: Unsupported bits per sample: " << wav_data.bits_per_sample << std::endl;
        wav_data.samples.clear();
        return wav_data;
    }
    
    // Convert to mono if stereo
    if (wav_data.num_channels == 2) {
        std::vector<float> mono_samples(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            mono_samples[i] = (wav_data.samples[i * 2] + wav_data.samples[i * 2 + 1]) / 2.0f;
        }
        wav_data.samples = mono_samples;
        wav_data.num_channels = 1;
    }
    
    return wav_data;
}

bool WavReader::write_wav(const std::string& filename,
                          const std::vector<float>& samples,
                          int sample_rate,
                          int num_channels) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create WAV file: " << filename << std::endl;
        return false;
    }
    
    // Write RIFF header
    file.write("RIFF", 4);
    
    uint32_t data_size = samples.size() * sizeof(int16_t);
    uint32_t file_size = 36 + data_size;
    
    // Write file size (little-endian)
    uint8_t file_size_bytes[4];
    file_size_bytes[0] = file_size & 0xFF;
    file_size_bytes[1] = (file_size >> 8) & 0xFF;
    file_size_bytes[2] = (file_size >> 16) & 0xFF;
    file_size_bytes[3] = (file_size >> 24) & 0xFF;
    file.write(reinterpret_cast<const char*>(file_size_bytes), 4);
    
    file.write("WAVE", 4);
    
    // Write fmt chunk
    file.write("fmt ", 4);
    
    uint32_t fmt_size = 16;
    uint8_t fmt_size_bytes[4];
    fmt_size_bytes[0] = fmt_size & 0xFF;
    fmt_size_bytes[1] = (fmt_size >> 8) & 0xFF;
    fmt_size_bytes[2] = (fmt_size >> 16) & 0xFF;
    fmt_size_bytes[3] = (fmt_size >> 24) & 0xFF;
    file.write(reinterpret_cast<const char*>(fmt_size_bytes), 4);
    
    uint16_t audio_format = 1;  // PCM
    uint8_t audio_format_bytes[2] = {audio_format & 0xFF, (audio_format >> 8) & 0xFF};
    file.write(reinterpret_cast<const char*>(audio_format_bytes), 2);
    
    uint8_t num_channels_bytes[2] = {num_channels & 0xFF, (num_channels >> 8) & 0xFF};
    file.write(reinterpret_cast<const char*>(num_channels_bytes), 2);
    
    uint8_t sample_rate_bytes[4];
    sample_rate_bytes[0] = sample_rate & 0xFF;
    sample_rate_bytes[1] = (sample_rate >> 8) & 0xFF;
    sample_rate_bytes[2] = (sample_rate >> 16) & 0xFF;
    sample_rate_bytes[3] = (sample_rate >> 24) & 0xFF;
    file.write(reinterpret_cast<const char*>(sample_rate_bytes), 4);
    
    uint32_t byte_rate = sample_rate * num_channels * 2;
    uint8_t byte_rate_bytes[4];
    byte_rate_bytes[0] = byte_rate & 0xFF;
    byte_rate_bytes[1] = (byte_rate >> 8) & 0xFF;
    byte_rate_bytes[2] = (byte_rate >> 16) & 0xFF;
    byte_rate_bytes[3] = (byte_rate >> 24) & 0xFF;
    file.write(reinterpret_cast<const char*>(byte_rate_bytes), 4);
    
    uint16_t block_align = num_channels * 2;
    uint8_t block_align_bytes[2] = {block_align & 0xFF, (block_align >> 8) & 0xFF};
    file.write(reinterpret_cast<const char*>(block_align_bytes), 2);
    
    uint16_t bits_per_sample = 16;
    uint8_t bits_per_sample_bytes[2] = {bits_per_sample & 0xFF, (bits_per_sample >> 8) & 0xFF};
    file.write(reinterpret_cast<const char*>(bits_per_sample_bytes), 2);
    
    // Write data chunk
    file.write("data", 4);
    
    uint8_t data_size_bytes[4];
    data_size_bytes[0] = data_size & 0xFF;
    data_size_bytes[1] = (data_size >> 8) & 0xFF;
    data_size_bytes[2] = (data_size >> 16) & 0xFF;
    data_size_bytes[3] = (data_size >> 24) & 0xFF;
    file.write(reinterpret_cast<const char*>(data_size_bytes), 4);
    
    // Write samples as int16
    for (float sample : samples) {
        int16_t int_sample = static_cast<int16_t>(
            std::max(-1.0f, std::min(1.0f, sample)) * 32767.0f
        );
        uint8_t sample_bytes[2] = {int_sample & 0xFF, (int_sample >> 8) & 0xFF};
        file.write(reinterpret_cast<const char*>(sample_bytes), 2);
    }
    
    return true;
}
