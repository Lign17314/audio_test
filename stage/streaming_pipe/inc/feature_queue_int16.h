#ifndef FEATURE_QUEUE_INT16_H
#define FEATURE_QUEUE_INT16_H

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cstdint>

/**
 * 线程安全的 INT16 特征帧队列
 * 用于在生产者线程和消费者线程之间传递 INT16 特征帧
 */
class FeatureQueueINT16 {
public:
    FeatureQueueINT16() : finished_(false) {}
    
    /**
     * 将 INT16 特征帧放入队列（生产者调用）
     * @param frame INT16 特征帧（400维向量）
     */
    void push(const std::vector<int16_t>& frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(frame);
        cv_.notify_one();  // 通知消费者有新数据
    }
    
    /**
     * 从队列中取出 INT16 特征帧（消费者调用）
     * @param frame 输出 INT16 特征帧
     * @return true=成功取出，false=队列已结束且为空
     */
    bool pop(std::vector<int16_t>& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // 等待直到队列非空或生产者结束
        cv_.wait(lock, [this] { 
            return !queue_.empty() || finished_; 
        });
        
        // 如果队列为空且生产者已结束，返回false
        if (queue_.empty() && finished_) {
            return false;
        }
        
        // 取出一帧
        frame = queue_.front();
        queue_.pop();
        return true;
    }
    
    /**
     * 标记生产者已完成（不再有新数据）
     */
    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        cv_.notify_all();  // 通知所有等待的消费者
    }
    
    /**
     * 获取当前队列大小
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    /**
     * 检查队列是否为空
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    /**
     * 检查生产者是否已完成
     */
    bool is_finished() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return finished_;
    }
    
private:
    std::queue<std::vector<int16_t>> queue_;  // INT16 特征帧队列
    mutable std::mutex mutex_;                 // 互斥锁
    std::condition_variable cv_;               // 条件变量
    bool finished_;                            // 生产者是否已完成
};

#endif // FEATURE_QUEUE_INT16_H
