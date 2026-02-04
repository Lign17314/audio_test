#ifndef CTC_DECODER_H
#define CTC_DECODER_H

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>

/**
 * CTC Prefix Beam Search Decoder for Keyword Spotting (C++ implementation)
 * 
 * Matches Python KwsCtcPrefixDecoder behavior from ctc.py
 * 
 * Features:
 *   - CTC prefix beam search
 *   - Keyword detection with confidence scoring
 *   - Support for multiple keywords
 */

struct DecodeResult {
    bool hit;              // Whether keyword was detected
    std::string keyword;   // Detected keyword (if hit=true)
    float score;           // Confidence score (if hit=true)
    
    DecodeResult() : hit(false), score(0.0f) {}
    DecodeResult(bool h, const std::string& kw, float s) 
        : hit(h), keyword(kw), score(s) {}
};

struct PrefixNode {
    int token;
    int frame;
    float prob;
    
    PrefixNode(int t, int f, float p) : token(t), frame(f), prob(p) {}
};

struct Hypothesis {
    std::vector<int> prefix_ids;      // Token sequence
    float score;                      // Total score (pb + pnb)
    std::vector<PrefixNode> nodes;    // Node information for each token
    
    Hypothesis() : score(0.0f) {}
};

class CTCDecoder {
public:
    /**
     * Initialize CTC decoder
     * @param vocab_size: Vocabulary size (e.g., 2599)
     * @param blank_id: Blank token ID (usually 0)
     */
    CTCDecoder(int vocab_size, int blank_id = 0);
    ~CTCDecoder();
    
    /**
     * Apply softmax to logits
     * @param logits: Input logits (T, vocab_size)
     * @param output: Output probabilities (T, vocab_size)
     */
    void softmax(const std::vector<std::vector<float>>& logits,
                 std::vector<std::vector<float>>& output);
    
private:
    int vocab_size_;
    int blank_id_;
};

class KwsCtcPrefixDecoder {
public:
    /**
     * Initialize KWS CTC Prefix Decoder
     * @param vocab_size: Vocabulary size (e.g., 2599)
     * @param keywords: Comma-separated keywords (e.g., "小云小云")
     * @param token_list: List of tokens (strings)
     * @param seg_dict: Segmentation dictionary (word -> tokens mapping)
     * @param blank_id: Blank token ID (usually 0)
     */
    KwsCtcPrefixDecoder(
        int vocab_size,
        const std::string& keywords,
        const std::vector<std::string>& token_list,
        const std::unordered_map<std::string, std::vector<std::string>>& seg_dict,
        int blank_id = 0
    );
    
    ~KwsCtcPrefixDecoder();
    
    /**
     * Decode logits to detect keywords
     * @param logits: Input logits (T, vocab_size) - raw logits, will apply softmax internally
     * @return: DecodeResult with hit status, keyword, and score
     */
    DecodeResult decode(const std::vector<std::vector<float>>& logits);
    
    /**
     * Decode logits from numpy array (simplified format)
     * @param logits_file: Path to .npy file containing logits (1, T, vocab_size)
     * @return: DecodeResult
     */
    DecodeResult decode_from_file(const std::string& logits_file);
    
private:
    int vocab_size_;
    int blank_id_;
    std::string keywords_str_;
    std::vector<std::string> token_list_;
    std::unordered_map<std::string, int> token_table_;
    std::unordered_set<int> keywords_idxset_;
    std::unordered_map<std::string, std::vector<int>> keywords_token_;
    std::vector<std::string> keywords_list_;
    
    CTCDecoder ctc_;
    
    /**
     * CTC prefix beam search
     * @param probs: Probabilities (T, vocab_size) - NOT log probabilities!
     * @param keywords_tokenset: Set of keyword token IDs to filter
     * @param score_beam_size: Beam size for score pruning
     * @param path_beam_size: Beam size for path pruning
     * @return: List of hypotheses
     */
    std::vector<Hypothesis> beam_search(
        const std::vector<std::vector<float>>& probs,
        const std::unordered_set<int>& keywords_tokenset,
        int score_beam_size = 3,
        int path_beam_size = 20
    );
    
    /**
     * Check if check_list is a sublist of main_list
     * @return: Starting index if found, -1 otherwise
     */
    int is_sublist(const std::vector<int>& main_list, 
                   const std::vector<int>& check_list);
    
    /**
     * Query token set for keyword tokenization
     * Converts keyword string to token IDs
     */
    std::pair<std::vector<std::string>, std::vector<int>> query_token_set(
        const std::string& txt,
        const std::unordered_map<std::string, int>& symbol_table,
        const std::unordered_map<std::string, std::vector<std::string>>& lexicon_table
    );
    
    /**
     * Internal decode function
     * @param probs: Probabilities (T, vocab_size) - already softmax applied
     */
    DecodeResult decode_inside(const std::vector<std::vector<float>>& probs);
};

#endif // CTC_DECODER_H
