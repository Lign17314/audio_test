#include "fbank_extractor.h"
#include "wav_reader.h"
#include <iostream>
#include <iomanip>
#include <fstream>

// Save features to numpy .npy format (simplified version)
void save_features_npy(const std::string& filename,
                      const std::vector<std::vector<float>>& features) {
    if (features.empty()) {
        std::cerr << "Error: Empty features" << std::endl;
        return;
    }
    
    // For simplicity, we'll save as a simple binary format
    // In production, use proper numpy .npy format
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create output file: " << filename << std::endl;
        return;
    }
    
    // Write shape: (1, num_frames, feature_dim)
    int32_t batch_size = 1;
    int32_t num_frames = static_cast<int32_t>(features.size());
    int32_t feature_dim = static_cast<int32_t>(features[0].size());
    
    file.write(reinterpret_cast<const char*>(&batch_size), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&num_frames), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&feature_dim), sizeof(int32_t));
    
    // Write data
    for (const auto& frame : features) {
        file.write(reinterpret_cast<const char*>(frame.data()),
                  frame.size() * sizeof(float));
    }
    
    std::cout << "Saved features: shape=(1, " << num_frames << ", " << feature_dim << ")" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.wav> <output.npy> [cmvn_file]" << std::endl;
        std::cerr << "Example: " << argv[0] 
                  << " input.wav output_fbank.npy /path/to/cmvn_file" << std::endl;
        return 1;
    }
    
    std::string wav_file = argv[1];
    std::string output_file = argv[2];
    std::string cmvn_file = (argc > 3) ? argv[3] : "";
    
    // Configure fbank extractor (matching a2.py WavFrontend_config)
    FBankConfig config;
    config.fs = 16000;
    config.window = "hamming";
    config.n_mels = 80;
    config.frame_length = 25;  // ms
    config.frame_shift = 10;   // ms
    config.lfr_m = 5;
    config.lfr_n = 3;
    config.dither = 1.0f;
    config.snip_edges = true;
    config.upsacle_samples = true;
    config.cmvn_file = cmvn_file;
    
    std::cout << "==========================================" << std::endl;
    std::cout << "FBank Feature Extractor (C++)" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Input WAV: " << wav_file << std::endl;
    std::cout << "Output: " << output_file << std::endl;
    if (!cmvn_file.empty()) {
        std::cout << "CMVN file: " << cmvn_file << std::endl;
    }
    std::cout << std::endl;
    
    // Read WAV file
    std::cout << "Reading WAV file..." << std::endl;
    WavData wav_data = WavReader::read_wav(wav_file);
    
    if (wav_data.samples.empty()) {
        std::cerr << "Error: Failed to read WAV file or file is empty" << std::endl;
        return 1;
    }
    
    std::cout << "  Sample rate: " << wav_data.sample_rate << " Hz" << std::endl;
    std::cout << "  Channels: " << wav_data.num_channels << std::endl;
    std::cout << "  Samples: " << wav_data.samples.size() << std::endl;
    std::cout << "  Duration: " << std::fixed << std::setprecision(2)
              << static_cast<float>(wav_data.samples.size()) / wav_data.sample_rate 
              << " seconds" << std::endl;
    
    // Resample if needed (simple implementation - use proper resampler in production)
    if (wav_data.sample_rate != config.fs) {
        std::cerr << "Warning: Sample rate mismatch. Resampling not implemented." << std::endl;
        std::cerr << "  Expected: " << config.fs << " Hz, Got: " << wav_data.sample_rate << " Hz" << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    std::cout << "Extracting fbank features..." << std::endl;
    
    // Create extractor
    FBankExtractor extractor(config);
    
    // Extract features
    std::vector<std::vector<float>> features = extractor.extract(
        wav_data.samples.data(),
        wav_data.samples.size()
    );
    
    if (features.empty()) {
        std::cerr << "Error: Failed to extract features" << std::endl;
        return 1;
    }
    
    std::cout << "  Extracted " << features.size() << " frames" << std::endl;
    std::cout << "  Feature dimension: " << features[0].size() << std::endl;
    std::cout << "  Expected dimension: " << extractor.get_output_dim() << std::endl;
    
    // Print first frame statistics
    if (!features.empty()) {
        float min_val = features[0][0];
        float max_val = features[0][0];
        float sum_val = 0.0f;
        
        for (float val : features[0]) {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum_val += val;
        }
        
        std::cout << std::endl;
        std::cout << "First frame statistics:" << std::endl;
        std::cout << "  Min: " << std::fixed << std::setprecision(6) << min_val << std::endl;
        std::cout << "  Max: " << std::fixed << std::setprecision(6) << max_val << std::endl;
        std::cout << "  Mean: " << std::fixed << std::setprecision(6) 
                  << sum_val / features[0].size() << std::endl;
    }
    
    // Save features
    std::cout << std::endl;
    std::cout << "Saving features to " << output_file << "..." << std::endl;
    save_features_npy(output_file, features);
    
    std::cout << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Extraction completed!" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}
