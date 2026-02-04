/* Copyright 2025
 * Streaming FBank Extractor - Standalone Version
 * 
 * This program extracts FBank features from audio in streaming mode (10ms chunks)
 * and saves the results to a numpy file for comparison with batch processing.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <memory>
#include <iomanip>

#include "streaming_fbank_extractor.h"
#include "../fbank_extractor/wav_reader.h"

// Save features to numpy .npy format
bool save_npy(const std::string& filename,
              const std::vector<std::vector<float>>& data) {
    if (data.empty() || data[0].empty()) {
        fprintf(stderr, "Error: Empty data\n");
        return false;
    }
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to create file: %s\n", filename.c_str());
        return false;
    }
    
    // Write magic
    file.write("\x93NUMPY", 6);
    
    // Write version (1.0)
    uint8_t major = 1, minor = 0;
    file.write(reinterpret_cast<const char*>(&major), 1);
    file.write(reinterpret_cast<const char*>(&minor), 1);
    
    // Build header: shape is (1, T, D)
    size_t T = data.size();
    size_t D = data[0].size();
    
    char header[256];
    memset(header, 0, sizeof(header));
    int header_len = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (1, %zu, %zu), }",
        T, D);
    
    // Pad header to be divisible by 64
    int padded_len = ((header_len + 1 + 63) / 64) * 64;
    for (int i = header_len; i < padded_len - 1; i++) {
        header[i] = ' ';
    }
    header[padded_len - 1] = '\n';
    
    // Write header length
    uint16_t header_len_16 = padded_len;
    file.write(reinterpret_cast<const char*>(&header_len_16), 2);
    
    // Write header
    file.write(header, padded_len);
    
    // Write data: (1, T, D) format
    for (size_t t = 0; t < T; t++) {
        file.write(reinterpret_cast<const char*>(data[t].data()),
                   D * sizeof(float));
    }
    
    printf("Saved to %s: shape=(1, %zu, %zu)\n", filename.c_str(), T, D);
    return true;
}

int main(int argc, char* argv[]) {
    // Default parameters (can be overridden by command line)
    const char* wav_file = "/root/volume/ctc/train/funasr_test/test_xiaoyun.wav";
    const char* output_file = "streaming_fbank_output.npy";
    const char* cmvn_file = nullptr;  // Optional CMVN file
    
    printf("========================================\n");
    printf("Streaming FBank Extractor\n");
    printf("========================================\n");
    printf("WAV file: %s\n", wav_file);
    printf("Output file: %s\n", output_file);
    printf("\n");
    
    // 1. Load WAV file
    printf("Loading WAV file...\n");
    WavData wav_data = WavReader::read_wav(wav_file);
    
    if (wav_data.samples.empty()) {
        fprintf(stderr, "ERROR: Failed to load WAV file: %s\n", wav_file);
        return 1;
    }
    
    int sample_rate = wav_data.sample_rate;
    int num_channels = wav_data.num_channels;
    const std::vector<float>& samples = wav_data.samples;
    
    printf("WAV info:\n");
    printf("  Sample rate: %d Hz\n", sample_rate);
    printf("  Channels: %d\n", num_channels);
    printf("  Samples: %zu\n", samples.size());
    printf("  Duration: %.2f seconds\n", 
           static_cast<float>(samples.size()) / num_channels / sample_rate);
    printf("\n");
    
    if (sample_rate != 16000) {
        fprintf(stderr, "WARNING: Sample rate is %d, expected 16000\n", sample_rate);
    }
    
    // Convert to mono if needed
    std::vector<float> audio_samples;
    if (num_channels == 1) {
        audio_samples = samples;
    } else {
        printf("Converting stereo to mono...\n");
        audio_samples.reserve(samples.size() / num_channels);
        for (size_t i = 0; i < samples.size(); i += num_channels) {
            float mono = 0.0f;
            for (int ch = 0; ch < num_channels; ch++) {
                mono += samples[i + ch];
            }
            audio_samples.push_back(mono / num_channels);
        }
    }
    
    printf("Mono audio samples: %zu\n\n", audio_samples.size());
    
    // 2. Initialize FBank config
    FBankConfig fbank_config;
    fbank_config.fs = 16000;
    fbank_config.window = "hamming";
    fbank_config.n_mels = 80;
    fbank_config.frame_length = 25;  // ms
    fbank_config.frame_shift = 10;   // ms
    fbank_config.lfr_m = 5;
    fbank_config.lfr_n = 3;
    fbank_config.dither = 1.0f;
    fbank_config.snip_edges = true;
    fbank_config.upsacle_samples = true;
    if (cmvn_file) {
        fbank_config.cmvn_file = cmvn_file;
    }
    
    printf("FBank configuration:\n");
    printf("  Sample rate: %d Hz\n", fbank_config.fs);
    printf("  Window: %s\n", fbank_config.window.c_str());
    printf("  Mel bins: %d\n", fbank_config.n_mels);
    printf("  Frame length: %d ms\n", fbank_config.frame_length);
    printf("  Frame shift: %d ms\n", fbank_config.frame_shift);
    printf("  LFR: m=%d, n=%d\n", fbank_config.lfr_m, fbank_config.lfr_n);
    printf("  Output dimension: %d (n_mels * lfr_m)\n", 
           fbank_config.n_mels * fbank_config.lfr_m);
    printf("\n");
    
    // 3. Initialize streaming FBank extractor
    StreamingFBankExtractor fbank_extractor(fbank_config);
    
    // 4. Process audio in 10ms chunks (160 samples at 16kHz)
    constexpr size_t chunk_samples = 160;  // 10ms at 16kHz
    size_t total_lfr_frames = 0;
    
    std::vector<std::vector<float>> all_lfr_features;
    
    printf("Processing audio in 10ms chunks...\n");
    printf("Chunk size: %zu samples (10ms)\n", chunk_samples);
    printf("\n");
    
    size_t chunk_count = 0;
    for (size_t i = 0; i < audio_samples.size(); i += chunk_samples) {
        size_t chunk_size = std::min(chunk_samples, audio_samples.size() - i);
        
        // Process audio chunk
        std::vector<std::vector<float>> lfr_frames;
        int num_lfr = fbank_extractor.process_chunk(
            audio_samples.data() + i,
            chunk_size,
            lfr_frames
        );
        
        if (num_lfr > 0) {
            total_lfr_frames += num_lfr;
            
            // Collect LFR features
            all_lfr_features.insert(all_lfr_features.end(), 
                                   lfr_frames.begin(), lfr_frames.end());
            
            // Print progress every 100 chunks
            if (chunk_count % 100 == 0) {
                printf("Chunk %zu: Output %d LFR frames (total: %zu)\n",
                       chunk_count, num_lfr, total_lfr_frames);
            }
        }
        
        chunk_count++;
    }
    
    printf("\nProcessed %zu chunks\n", chunk_count);
    
    // 5. Flush remaining frames
    printf("\nFlushing remaining frames...\n");
    
    std::vector<std::vector<float>> final_lfr_frames;
    int num_final_lfr = fbank_extractor.flush(final_lfr_frames);
    
    if (num_final_lfr > 0) {
        total_lfr_frames += num_final_lfr;
        printf("Final LFR frames: %d\n", num_final_lfr);
        
        // Collect final LFR features
        all_lfr_features.insert(all_lfr_features.end(),
                               final_lfr_frames.begin(), final_lfr_frames.end());
    }
    
    printf("\n");
    printf("========================================\n");
    printf("Processing complete!\n");
    printf("========================================\n");
    printf("Total LFR frames: %zu\n", total_lfr_frames);
    
    if (!all_lfr_features.empty()) {
        printf("Output shape: (1, %zu, %zu)\n", 
               all_lfr_features.size(), all_lfr_features[0].size());
    }
    
    // 6. Save results
    if (!all_lfr_features.empty()) {
        printf("\nSaving results...\n");
        if (save_npy(output_file, all_lfr_features)) {
            printf("✓ Successfully saved to: %s\n", output_file);
        } else {
            fprintf(stderr, "✗ Failed to save results\n");
            return 1;
        }
    } else {
        fprintf(stderr, "WARNING: No LFR features extracted\n");
        return 1;
    }
    
    printf("\n");
    printf("========================================\n");
    printf("Done!\n");
    printf("========================================\n");
    
    return 0;
}
