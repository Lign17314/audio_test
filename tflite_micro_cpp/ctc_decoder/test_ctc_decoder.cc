#include "ctc_decoder.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <algorithm>

// NumPy .npy file reader (standard format)
bool load_logits_npy(const std::string& filename,
                     std::vector<std::vector<float>>& logits) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }
    
    // Read magic string
    char magic[6];
    file.read(magic, 6);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) {
        std::cerr << "Error: Invalid numpy file format (magic mismatch)" << std::endl;
        return false;
    }
    
    // Read version
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    
    // Read header length
    uint16_t header_len = 0;
    if (major == 1) {
        file.read(reinterpret_cast<char*>(&header_len), 2);
    } else {
        uint32_t header_len_32 = 0;
        file.read(reinterpret_cast<char*>(&header_len_32), 4);
        header_len = header_len_32;
    }
    
    // Read header
    std::vector<char> header(header_len);
    file.read(header.data(), header_len);
    header.push_back('\0');
    
    // Parse header (simplified - assumes float32, C-order)
    char* shape_start = strstr(header.data(), "'shape'");
    char* descr_start = strstr(header.data(), "'descr'");
    
    if (!shape_start || !descr_start) {
        std::cerr << "Error: Failed to parse numpy header" << std::endl;
        return false;
    }
    
    // Parse shape: (1, T, vocab_size) or (T, vocab_size)
    std::vector<size_t> shape;
    char* shape_data = strstr(shape_start, "(");
    if (shape_data) {
        shape_data++;
        while (*shape_data != ')') {
            size_t dim = strtoul(shape_data, &shape_data, 10);
            shape.push_back(dim);
            while (*shape_data == ',' || *shape_data == ' ') shape_data++;
        }
    }
    
    // Verify dtype is float32
    if (strstr(descr_start, "f4") == nullptr && strstr(descr_start, "<f4") == nullptr) {
        std::cerr << "Error: Only float32 numpy arrays are supported" << std::endl;
        return false;
    }
    
    // Calculate data size
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    
    // Read data
    std::vector<float> data(total_size);
    file.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(float));
    
    if (file.gcount() != total_size * sizeof(float)) {
        std::cerr << "Error: Failed to read all data from file" << std::endl;
        return false;
    }
    
    // Reshape: handle (1, T, vocab_size) or (T, vocab_size)
    logits.clear();
    size_t num_frames, vocab_size;
    
    if (shape.size() == 3) {
        // Shape: (batch_size, num_frames, vocab_size)
        size_t batch_size = shape[0];
        num_frames = shape[1];
        vocab_size = shape[2];
        
        // Use first batch (index 0)
        logits.reserve(num_frames);
        for (size_t t = 0; t < num_frames; ++t) {
            std::vector<float> frame(vocab_size);
            for (size_t v = 0; v < vocab_size; ++v) {
                // Data layout: [batch0_frame0_vocab0, batch0_frame0_vocab1, ..., batch0_frame1_vocab0, ...]
                size_t idx = 0 * num_frames * vocab_size + t * vocab_size + v;
                frame[v] = data[idx];
            }
            logits.push_back(frame);
        }
    } else if (shape.size() == 2) {
        // Shape: (num_frames, vocab_size)
        num_frames = shape[0];
        vocab_size = shape[1];
        
        logits.reserve(num_frames);
        for (size_t t = 0; t < num_frames; ++t) {
            std::vector<float> frame(vocab_size);
            for (size_t v = 0; v < vocab_size; ++v) {
                frame[v] = data[t * vocab_size + v];
            }
            logits.push_back(frame);
        }
    } else {
        std::cerr << "Error: Unsupported shape dimension: " << shape.size() << std::endl;
        return false;
    }
    
    return true;
}

// Load token_list from file (one token per line)
bool load_token_list(const std::string& filename,
                     std::vector<std::string>& token_list) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open token_list file: " << filename << std::endl;
        return false;
    }
    
    token_list.clear();
    std::string line;
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);
        if (!line.empty()) {
            token_list.push_back(line);
        }
    }
    
    return true;
}

// Load seg_dict from file (format: word token1 token2 ...)
bool load_seg_dict(const std::string& filename,
                   std::unordered_map<std::string, std::vector<std::string>>& seg_dict) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open seg_dict file: " << filename << std::endl;
        return false;
    }
    
    seg_dict.clear();
    std::string line;
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);
        if (line.empty()) continue;
        
        // Split by whitespace
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 2) {
            std::string word = tokens[0];
            std::vector<std::string> seg_tokens(tokens.begin() + 1, tokens.end());
            seg_dict[word] = seg_tokens;
        }
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <logits.npy> <keywords> [token_list_file] [seg_dict_file]" << std::endl;
        std::cerr << "Example: " << argv[0] 
                  << " output_logits.npy \"小云小云\" token_list.txt seg_dict.txt" << std::endl;
        std::cerr << "Note: If token_list_file or seg_dict_file are not provided," << std::endl;
        std::cerr << "      will try to load from current directory." << std::endl;
        return 1;
    }
    
    std::string logits_file = argv[1];
    std::string keywords = argv[2];
    std::string token_list_file = (argc > 3) ? argv[3] : "token_list.txt";
    std::string seg_dict_file = (argc > 4) ? argv[4] : "seg_dict.txt";
    
    std::cout << "==========================================" << std::endl;
    std::cout << "CTC Decoder Test" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Logits file: " << logits_file << std::endl;
    std::cout << "Keywords: " << keywords << std::endl;
    std::cout << std::endl;
    
    // Load logits
    std::cout << "Loading logits..." << std::endl;
    std::vector<std::vector<float>> logits;
    if (!load_logits_npy(logits_file, logits)) {
        return 1;
    }
    
    std::cout << "  Shape: (" << logits.size() << ", " 
              << (logits.empty() ? 0 : logits[0].size()) << ")" << std::endl;
    std::cout << std::endl;
    
    // Load token_list and seg_dict
    std::vector<std::string> token_list;
    std::unordered_map<std::string, std::vector<std::string>> seg_dict;
    
    std::cout << "Loading token_list from: " << token_list_file << std::endl;
    if (!load_token_list(token_list_file, token_list)) {
        std::cerr << "Warning: Failed to load token_list, using empty list" << std::endl;
        std::cerr << "  This may cause keyword detection to fail" << std::endl;
    } else {
        std::cout << "  Loaded " << token_list.size() << " tokens" << std::endl;
    }
    
    std::cout << "Loading seg_dict from: " << seg_dict_file << std::endl;
    if (!load_seg_dict(seg_dict_file, seg_dict)) {
        std::cerr << "Warning: Failed to load seg_dict, using empty dict" << std::endl;
        std::cerr << "  This may cause keyword detection to fail" << std::endl;
    } else {
        std::cout << "  Loaded " << seg_dict.size() << " entries" << std::endl;
    }
    std::cout << std::endl;
    
    // Get vocab_size from logits or token_list
    int vocab_size = logits.empty() ? 
        (token_list.empty() ? 2599 : static_cast<int>(token_list.size())) :
        static_cast<int>(logits[0].size());
    
    // Initialize decoder
    std::cout << "Initializing CTC decoder..." << std::endl;
    std::cout << "  Vocab size: " << vocab_size << std::endl;
    std::cout << "  Keywords: " << keywords << std::endl;
    if (!token_list.empty()) {
        std::cout << "  Token list size: " << token_list.size() << std::endl;
    }
    if (!seg_dict.empty()) {
        std::cout << "  Seg dict size: " << seg_dict.size() << std::endl;
    }
    std::cout << std::endl;
    
    KwsCtcPrefixDecoder decoder(vocab_size, keywords, token_list, seg_dict);
    
    // Decode
    std::cout << "Decoding..." << std::endl;
    DecodeResult result = decoder.decode(logits);
    
    // Print result
    std::cout << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Decode Result" << std::endl;
    std::cout << "==========================================" << std::endl;
    if (result.hit) {
        std::cout << "  ✅ Keyword detected: " << result.keyword << std::endl;
        std::cout << "  Score: " << std::fixed << std::setprecision(6) << result.score << std::endl;
    } else {
        std::cout << "  ❌ No keyword detected" << std::endl;
    }
    std::cout << std::endl;
    
    return 0;
}
