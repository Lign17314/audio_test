#include "ctc_decoder.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <tuple>
#include <map>

// Helper function to split string
static std::vector<std::string> split_string(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

// Helper function to trim whitespace
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

CTCDecoder::CTCDecoder(int vocab_size, int blank_id)
    : vocab_size_(vocab_size), blank_id_(blank_id) {
}

CTCDecoder::~CTCDecoder() {
}

void CTCDecoder::softmax(const std::vector<std::vector<float>>& logits,
                         std::vector<std::vector<float>>& output) {
    output.clear();
    output.reserve(logits.size());
    
    for (const auto& frame_logits : logits) {
        std::vector<float> probs(vocab_size_);
        
        // Find max for numerical stability
        float max_val = *std::max_element(frame_logits.begin(), frame_logits.end());
        
        // Compute exp and sum
        float sum = 0.0f;
        for (size_t i = 0; i < frame_logits.size() && i < static_cast<size_t>(vocab_size_); ++i) {
            probs[i] = std::exp(frame_logits[i] - max_val);
            sum += probs[i];
        }
        
        // Normalize
        if (sum > 0.0f) {
            for (float& prob : probs) {
                prob /= sum;
            }
        }
        
        output.push_back(probs);
    }
}

KwsCtcPrefixDecoder::KwsCtcPrefixDecoder(
    int vocab_size,
    const std::string& keywords,
    const std::vector<std::string>& token_list,
    const std::unordered_map<std::string, std::vector<std::string>>& seg_dict,
    int blank_id)
    : vocab_size_(vocab_size), blank_id_(blank_id), token_list_(token_list), ctc_(vocab_size, blank_id) {
    
    // Build token table
    for (size_t i = 0; i < token_list.size(); ++i) {
        token_table_[token_list[i]] = static_cast<int>(i);
    }
    
    // Initialize keywords
    keywords_idxset_.insert(blank_id);
    keywords_str_ = keywords;
    
    // Parse keywords (comma-separated)
    std::string trimmed_keywords = trim(keywords);
    // Remove spaces
    trimmed_keywords.erase(std::remove(trimmed_keywords.begin(), trimmed_keywords.end(), ' '), 
                          trimmed_keywords.end());
    keywords_list_ = split_string(trimmed_keywords, ',');
    
    // Process each keyword
    for (const auto& keyword : keywords_list_) {
        auto token_result = query_token_set(keyword, token_table_, seg_dict);
        std::vector<int> token_ids = token_result.second;
        
        keywords_token_[keyword] = token_ids;
        
        // Add token IDs to keywords_idxset
        for (int id : token_ids) {
            keywords_idxset_.insert(id);
        }
    }
}

KwsCtcPrefixDecoder::~KwsCtcPrefixDecoder() {
}

std::pair<std::vector<std::string>, std::vector<int>> 
KwsCtcPrefixDecoder::query_token_set(
    const std::string& txt,
    const std::unordered_map<std::string, int>& symbol_table,
    const std::unordered_map<std::string, std::vector<std::string>>& lexicon_table) {
    
    std::vector<std::string> tokens_str;
    std::vector<int> tokens_idx;
    
    // Check if keyword is directly in symbol table
    auto it = symbol_table.find(txt);
    if (it != symbol_table.end()) {
        tokens_str.push_back(txt);
        tokens_idx.push_back(it->second);
        return {tokens_str, tokens_idx};
    }
    
    // Simple character-based tokenization (simplified version)
    // In production, should use proper segmentation
    for (size_t i = 0; i < txt.length(); ) {
        // Try to match longest possible token
        bool found = false;
        for (int len = static_cast<int>(txt.length() - i); len > 0; --len) {
            std::string substr = txt.substr(i, len);
            
            // Check symbol table
            auto sym_it = symbol_table.find(substr);
            if (sym_it != symbol_table.end()) {
                tokens_str.push_back(substr);
                tokens_idx.push_back(sym_it->second);
                i += len;
                found = true;
                break;
            }
            
            // Check lexicon table
            auto lex_it = lexicon_table.find(substr);
            if (lex_it != lexicon_table.end()) {
                for (const auto& token : lex_it->second) {
                    auto token_it = symbol_table.find(token);
                    if (token_it != symbol_table.end()) {
                        tokens_str.push_back(token);
                        tokens_idx.push_back(token_it->second);
                    }
                }
                i += len;
                found = true;
                break;
            }
        }
        
        if (!found) {
            // Character-by-character fallback
            std::string char_str = txt.substr(i, 1);
            auto char_it = symbol_table.find(char_str);
            if (char_it != symbol_table.end()) {
                tokens_str.push_back(char_str);
                tokens_idx.push_back(char_it->second);
            } else {
                // Use blank as fallback
                tokens_str.push_back("<blank>");
                tokens_idx.push_back(blank_id_);
            }
            ++i;
        }
    }
    
    return {tokens_str, tokens_idx};
}

int KwsCtcPrefixDecoder::is_sublist(const std::vector<int>& main_list,
                                    const std::vector<int>& check_list) {
    if (main_list.size() < check_list.size()) {
        return -1;
    }
    
    if (main_list.size() == check_list.size()) {
        return (main_list == check_list) ? 0 : -1;
    }
    
    for (size_t i = 0; i <= main_list.size() - check_list.size(); ++i) {
        if (main_list[i] == check_list[0]) {
            bool match = true;
            for (size_t j = 0; j < check_list.size(); ++j) {
                if (main_list[i + j] != check_list[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return static_cast<int>(i);
            }
        }
    }
    
    return -1;
}

std::vector<Hypothesis> KwsCtcPrefixDecoder::beam_search(
    const std::vector<std::vector<float>>& probs,
    const std::unordered_set<int>& keywords_tokenset,
    int score_beam_size,
    int path_beam_size) {
    
    int maxlen = static_cast<int>(probs.size());
    
    // Current hypotheses: (prefix_tuple, (pb, pnb, nodes))
    // Using vector<pair> instead of tuple for easier manipulation
    struct HypState {
        std::vector<int> prefix;
        float pb;  // Probability with blank
        float pnb; // Probability without blank
        std::vector<PrefixNode> nodes;
    };
    
    std::vector<HypState> cur_hyps;
    cur_hyps.push_back({{}, 1.0f, 0.0f, {}});
    
    // CTC beam search step by step
    for (int t = 0; t < maxlen; ++t) {
        const std::vector<float>& frame_probs = probs[t];
        
        // Next hypotheses: key = prefix, value = (pb, pnb, nodes)
        // Use map instead of unordered_map for simplicity (vector as key)
        std::map<std::vector<int>, std::tuple<float, float, std::vector<PrefixNode>>> next_hyps;
        
        // First beam prune: select top-k best
        std::vector<std::pair<float, int>> prob_idx;
        for (size_t i = 0; i < frame_probs.size(); ++i) {
            prob_idx.push_back({frame_probs[i], static_cast<int>(i)});
        }
        
        std::sort(prob_idx.begin(), prob_idx.end(), 
                 [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                     return a.first > b.first;
                 });
        
        // Filter probabilities
        std::vector<int> filter_index;
        for (size_t i = 0; i < std::min(static_cast<size_t>(score_beam_size), prob_idx.size()); ++i) {
            float prob = prob_idx[i].first;
            int idx = prob_idx[i].second;
            
            if (prob > 0.05f) {
                if (keywords_tokenset.empty() || keywords_tokenset.find(idx) != keywords_tokenset.end()) {
                    filter_index.push_back(idx);
                }
            }
        }
        
        if (filter_index.empty()) {
            continue;
        }
        
        // Process each filtered token
        for (int s : filter_index) {
            float ps = frame_probs[s];
            
            if (s != blank_id_) {
                std::cout << "frame:" << t << ", token:" << s << ", score:" << ps << std::endl;
            }
            
            for (const auto& hyp : cur_hyps) {
                const std::vector<int>& prefix = hyp.prefix;
                float pb = hyp.pb;
                float pnb = hyp.pnb;
                const std::vector<PrefixNode>& cur_nodes = hyp.nodes;
                
                int last = prefix.empty() ? -1 : prefix.back();
                
                if (s == blank_id_) {
                    // Blank token
                    auto& state = next_hyps[prefix];
                    float& n_pb = std::get<0>(state);
                    std::vector<PrefixNode>& nodes = std::get<2>(state);
                    n_pb = n_pb + pb * ps + pnb * ps;
                    nodes = cur_nodes;
                } else if (s == last) {
                    // Same token as last
                    // Update *ss -> *s
                    if (std::abs(pnb) > 1e-6f) {
                        auto& state = next_hyps[prefix];
                        float& n_pnb = std::get<1>(state);
                        std::vector<PrefixNode>& nodes = std::get<2>(state);
                        n_pnb = n_pnb + pnb * ps;
                        nodes = cur_nodes;
                        if (!nodes.empty() && ps > nodes.back().prob) {
                            nodes.back().prob = ps;
                            nodes.back().frame = t;
                        }
                    }
                    
                    // Update *s-s -> *ss
                    if (std::abs(pb) > 1e-6f) {
                        std::vector<int> n_prefix = prefix;
                        n_prefix.push_back(s);
                        auto& state = next_hyps[n_prefix];
                        float& n_pnb = std::get<1>(state);
                        std::vector<PrefixNode>& nodes = std::get<2>(state);
                        n_pnb = n_pnb + pb * ps;
                        nodes = cur_nodes;
                        nodes.push_back(PrefixNode(s, t, ps));
                    }
                } else {
                    // Different token
                    std::vector<int> n_prefix = prefix;
                    n_prefix.push_back(s);
                    auto& state = next_hyps[n_prefix];
                    float& n_pnb = std::get<1>(state);
                    std::vector<PrefixNode>& nodes = std::get<2>(state);
                    
                    if (!nodes.empty() && ps > nodes.back().prob) {
                        nodes.back().prob = ps;
                        nodes.back().frame = t;
                    } else {
                        nodes = cur_nodes;
                        nodes.push_back(PrefixNode(s, t, ps));
                    }
                    n_pnb = n_pnb + pb * ps + pnb * ps;
                }
            }
        }
        
        // Second beam prune: select top path_beam_size
        std::vector<std::pair<std::vector<int>, std::tuple<float, float, std::vector<PrefixNode>>>> 
            next_hyps_list(next_hyps.begin(), next_hyps.end());
        
        std::sort(next_hyps_list.begin(), next_hyps_list.end(),
                 [](const auto& a, const auto& b) {
                     float score_a = std::get<0>(a.second) + std::get<1>(a.second);
                     float score_b = std::get<0>(b.second) + std::get<1>(b.second);
                     return score_a > score_b;
                 });
        
        cur_hyps.clear();
        int num_hyps = std::min(path_beam_size, static_cast<int>(next_hyps_list.size()));
        for (int i = 0; i < num_hyps; ++i) {
            HypState hyp;
            hyp.prefix = next_hyps_list[i].first;
            hyp.pb = std::get<0>(next_hyps_list[i].second);
            hyp.pnb = std::get<1>(next_hyps_list[i].second);
            hyp.nodes = std::get<2>(next_hyps_list[i].second);
            cur_hyps.push_back(hyp);
        }
    }
    
    // Convert to Hypothesis format
    std::vector<Hypothesis> hyps;
    for (const auto& hyp : cur_hyps) {
        Hypothesis h;
        h.prefix_ids = hyp.prefix;
        h.score = hyp.pb + hyp.pnb;
        h.nodes = hyp.nodes;
        hyps.push_back(h);
    }
    
    return hyps;
}

DecodeResult KwsCtcPrefixDecoder::decode_inside(
    const std::vector<std::vector<float>>& probs) {
    
    // probs is probabilities (not log probabilities), matching Python behavior
    std::vector<Hypothesis> hyps = beam_search(probs, keywords_idxset_);
    
    std::string hit_keyword;
    float hit_score = 1.0f;
    
    for (const auto& hyp : hyps) {
        const std::vector<int>& prefix_ids = hyp.prefix_ids;
        const std::vector<PrefixNode>& prefix_nodes = hyp.nodes;
        
        if (prefix_ids.size() != prefix_nodes.size()) {
            continue;
        }
        
        // Check each keyword
        for (const auto& kw_pair : keywords_token_) {
            const std::string& word = kw_pair.first;
            const std::vector<int>& lab = kw_pair.second;
            
            int offset = is_sublist(prefix_ids, lab);
            if (offset != -1) {
                hit_keyword = word;
                hit_score = 1.0f;
                
                // Multiply probabilities
                for (int idx = offset; idx < offset + static_cast<int>(lab.size()); ++idx) {
                    if (idx < static_cast<int>(prefix_nodes.size())) {
                        hit_score *= prefix_nodes[idx].prob;
                    }
                }
                
                // Square root normalization
                hit_score = std::sqrt(hit_score);
                break;
            }
        }
        
        if (!hit_keyword.empty()) {
            break;
        }
    }
    
    if (!hit_keyword.empty()) {
        return DecodeResult(true, hit_keyword, hit_score);
    } else {
        return DecodeResult(false, "", 0.0f);
    }
}

DecodeResult KwsCtcPrefixDecoder::decode(
    const std::vector<std::vector<float>>& logits) {
    
    // Apply softmax to get probabilities (matching Python behavior)
    // Python: raw_logp = self.ctc.softmax(x.unsqueeze(0)).detach().squeeze(0).cpu()
    // Then beam_search uses probabilities directly, not log probabilities
    std::vector<std::vector<float>> probs;
    ctc_.softmax(logits, probs);
    
    // Pass probabilities directly to decode_inside (not log probabilities)
    return decode_inside(probs);
}

DecodeResult KwsCtcPrefixDecoder::decode_from_file(const std::string& logits_file) {
    // TODO: Implement numpy file reading
    std::cerr << "decode_from_file not yet implemented" << std::endl;
    return DecodeResult();
}
