#ifndef KWS_DECODER_H
#define KWS_DECODER_H

#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

/**
 * @brief 简单的关键词唤醒解码器（状态机方法）
 * 
 * 使用有限状态自动机跟踪关键词匹配进度
 * 内存占用极小，速度极快，适合嵌入式设备
 */
class SimpleKwsDecoder {
public:
    /**
     * @brief 构造函数
     * 
     * @param keyword_seq 关键词的 token 序列，例如 {1462, 976, 1462, 976} 表示 "小云小云"
     * @param prob_threshold 概率阈值，低于此值的 token 会被忽略
     * @param blank_id CTC blank token 的 ID，通常为 0
     */
    SimpleKwsDecoder(const std::vector<int>& keyword_seq, 
                     float prob_threshold = 0.05f,
                     int blank_id = 0)
        : keyword_seq_(keyword_seq),
          keyword_len_(keyword_seq.size()),
          prob_threshold_(prob_threshold),
          blank_id_(blank_id) {
        reset();
    }
    
    /**
     * @brief 重置解码器状态
     */
    void reset() {
        states_.clear();
        // 初始状态：状态 0，概率 1.0，无 token
        states_[0] = {1.0f, {}};
        frame_idx_ = 0;
        detected_ = false;
        detected_score_ = 0.0f;
    }
    
    /**
     * @brief 处理单帧概率分布（流式接口）
     * 
     * @param frame_probs 当前帧的概率分布数组
     * @param vocab_size 词汇表大小
     * @return std::pair<bool, float> (是否检测到, 置信度分数)
     */
    std::pair<bool, float> process_frame(const float* frame_probs, int vocab_size) {
        if (detected_) {
            return {true, detected_score_};
        }
        
        int t = frame_idx_;
        std::map<int, StateData> new_states;
        
        float blank_prob = frame_probs[blank_id_];
        
        // 遍历所有当前状态
        for (const auto& [state_idx, state_data] : states_) {
            float state_prob = state_data.prob;
            const auto& token_probs = state_data.token_probs;
            
            // 转移 1: Blank 转移（保持当前状态）
            float new_prob = state_prob * blank_prob;
            if (new_states.find(state_idx) == new_states.end() || 
                new_prob > new_states[state_idx].prob) {
                new_states[state_idx] = {new_prob, token_probs};
            }
            
            // 转移 2: Token 转移（进入下一个状态）
            if (state_idx < keyword_len_) {
                int next_token = keyword_seq_[state_idx];
                
                // 边界检查
                if (next_token >= 0 && next_token < vocab_size) {
                    float next_token_prob = frame_probs[next_token];
                    
                    // 只有概率足够高才转移
                    if (next_token_prob > prob_threshold_) {
                        int next_state = state_idx + 1;
                        new_prob = state_prob * next_token_prob;
                        
                        if (new_states.find(next_state) == new_states.end() ||
                            new_prob > new_states[next_state].prob) {
                            auto new_token_probs = token_probs;
                            new_token_probs.push_back(next_token_prob);
                            new_states[next_state] = {new_prob, new_token_probs};
                        }
                    }
                }
            }
        }
        
        states_ = std::move(new_states);
        frame_idx_++;
        
        // 检查是否完成匹配
        if (states_.find(keyword_len_) != states_.end()) {
            const auto& final_state = states_[keyword_len_];
            const auto& token_probs = final_state.token_probs;
            
            // 计算置信度：所有 token 概率的平方根（与 Python 版本一致）
            float score = 1.0f;
            for (float p : token_probs) {
                score *= p;
            }
            score = std::sqrt(score);
            
            detected_ = true;
            detected_score_ = score;
            
            return {true, score};
        }
        
        return {false, 0.0f};
    }
    
    /**
     * @brief 获取当前帧索引
     */
    int get_frame_idx() const { return frame_idx_; }
    
    /**
     * @brief 是否已检测到关键词
     */
    bool is_detected() const { return detected_; }
    
    /**
     * @brief 获取检测到的置信度
     */
    float get_score() const { return detected_score_; }

private:
    // 状态数据结构
    struct StateData {
        float prob;                      // 到达该状态的概率
        std::vector<float> token_probs;  // 已匹配 token 的概率列表
    };
    
    std::vector<int> keyword_seq_;       // 关键词 token 序列
    int keyword_len_;                    // 关键词长度
    float prob_threshold_;               // 概率阈值
    int blank_id_;                       // blank token ID
    
    std::map<int, StateData> states_;    // 当前状态表
    int frame_idx_;                      // 当前帧索引
    bool detected_;                      // 是否已检测到
    float detected_score_;               // 检测到的置信度
};


/**
 * @brief 鲁棒的关键词唤醒解码器（带时间约束）
 * 
 * 在 SimpleKwsDecoder 基础上添加：
 * 1. 时间窗口限制：防止跨越长时间停顿的误拼接
 * 2. Blank 帧限制：限制连续 blank 帧数量
 * 3. 冷却期机制：防止重复触发
 */
class RobustKwsDecoder {
public:
    /**
     * @brief 构造函数
     * 
     * @param keyword_seq 关键词的 token 序列
     * @param prob_threshold 概率阈值
     * @param max_frames_per_token 每个 token 最多允许的帧数
     * @param max_consecutive_blanks 最多连续 blank 帧数
     * @param cooldown_frames 检测后的冷却期（帧数）
     * @param blank_id blank token ID
     */
    RobustKwsDecoder(const std::vector<int>& keyword_seq,
                     float prob_threshold = 0.05f,
                     int max_frames_per_token = 10,
                     int max_consecutive_blanks = 5,
                     int cooldown_frames = 30,
                     int blank_id = 0)
        : keyword_seq_(keyword_seq),
          keyword_len_(keyword_seq.size()),
          prob_threshold_(prob_threshold),
          max_frames_per_token_(max_frames_per_token),
          max_consecutive_blanks_(max_consecutive_blanks),
          cooldown_frames_(cooldown_frames),
          blank_id_(blank_id) {
        reset();
    }
    
    void reset() {
        states_.clear();
        // 初始状态：状态 0，概率 1.0，开始帧 0，blank 计数 0
        states_[0] = {1.0f, {}, 0, 0};
        frame_idx_ = 0;
        last_detection_frame_ = -999;
        detected_ = false;
        detected_score_ = 0.0f;
        detected_duration_ = 0;
    }
    
    /**
     * @brief 处理单帧
     * 
     * @return std::tuple<bool, float, int, std::string> 
     *         (是否检测到, 置信度, 持续帧数, 状态)
     */
    std::tuple<bool, float, int, std::string> process_frame(const float* frame_probs, int vocab_size) {
        int t = frame_idx_;
        
        // 冷却期检查
        if (t - last_detection_frame_ < cooldown_frames_) {
            frame_idx_++;
            return {false, 0.0f, 0, "cooldown"};
        }
        
        std::map<int, StateData> new_states;
        float blank_prob = frame_probs[blank_id_];
        
        // 遍历所有当前状态
        for (const auto& [state_idx, state_data] : states_) {
            float state_prob = state_data.prob;
            const auto& token_probs = state_data.token_probs;
            int start_frame = state_data.start_frame;
            int blank_count = state_data.blank_count;
            
            // 时间窗口检查
            int elapsed_frames = t - start_frame;
            int max_allowed = max_frames_per_token_ * (state_idx + 1);
            
            if (elapsed_frames > max_allowed) {
                continue;  // 超时，丢弃
            }
            
            // Blank 转移
            if (blank_count < max_consecutive_blanks_) {
                float new_prob = state_prob * blank_prob;
                int new_blank_count = blank_count + 1;
                
                if (new_states.find(state_idx) == new_states.end() ||
                    new_prob > new_states[state_idx].prob) {
                    new_states[state_idx] = {new_prob, token_probs, start_frame, new_blank_count};
                }
            }
            
            // Token 转移
            if (state_idx < keyword_len_) {
                int next_token = keyword_seq_[state_idx];
                
                if (next_token >= 0 && next_token < vocab_size) {
                    float next_token_prob = frame_probs[next_token];
                    
                    if (next_token_prob > prob_threshold_) {
                        int next_state = state_idx + 1;
                        float new_prob = state_prob * next_token_prob;
                        
                        if (new_states.find(next_state) == new_states.end() ||
                            new_prob > new_states[next_state].prob) {
                            auto new_token_probs = token_probs;
                            new_token_probs.push_back(next_token_prob);
                            int new_start_frame = (state_idx > 0) ? start_frame : t;
                            new_states[next_state] = {new_prob, new_token_probs, new_start_frame, 0};
                        }
                    }
                }
            }
        }
        
        states_ = std::move(new_states);
        frame_idx_++;
        
        // 检查是否完成
        if (states_.find(keyword_len_) != states_.end()) {
            const auto& final_state = states_[keyword_len_];
            const auto& token_probs = final_state.token_probs;
            int start_frame = final_state.start_frame;
            
            int total_frames = t - start_frame;
            
            // 计算置信度
            float score = 1.0f;
            for (float p : token_probs) {
                score *= p;
            }
            score = std::sqrt(score);
            
            // 进入冷却期
            last_detection_frame_ = t;
            states_.clear();
            states_[0] = {1.0f, {}, t, 0};
            
            detected_ = true;
            detected_score_ = score;
            detected_duration_ = total_frames;
            
            return {true, score, total_frames, "detected"};
        }
        
        return {false, 0.0f, 0, "processing"};
    }
    
    int get_frame_idx() const { return frame_idx_; }
    bool is_detected() const { return detected_; }
    float get_score() const { return detected_score_; }
    int get_duration() const { return detected_duration_; }

private:
    struct StateData {
        float prob;
        std::vector<float> token_probs;
        int start_frame;
        int blank_count;
    };
    
    std::vector<int> keyword_seq_;
    int keyword_len_;
    float prob_threshold_;
    int max_frames_per_token_;
    int max_consecutive_blanks_;
    int cooldown_frames_;
    int blank_id_;
    
    std::map<int, StateData> states_;
    int frame_idx_;
    int last_detection_frame_;
    bool detected_;
    float detected_score_;
    int detected_duration_;
};

#endif // KWS_DECODER_H
