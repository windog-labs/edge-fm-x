#pragma once

#include "utils/non_copyable.h"
#include <chrono>
#include <vector>
#include <string>
#include <cstdint>

namespace edge_fm {

struct MetricStats {
    double prefill_time_ms = 0.0;           // Prefill 阶段耗时（毫秒）
    double total_decode_time_ms = 0.0;       // 总 decode 耗时（毫秒）
    double avg_decode_step_time_ms = 0.0;   // 平均每个 decode_step 耗时（毫秒）
    double min_decode_step_time_ms = 0.0;   // 最小 decode_step 耗时（毫秒）
    double max_decode_step_time_ms = 0.0;   // 最大 decode_step 耗时（毫秒）
    
    int32_t total_tokens = 0;               // 总生成 token 数
    double tokens_per_second = 0.0;         // 吞吐量（tokens/sec）
    
    int32_t prefill_count = 0;              // Prefill 调用次数
    int32_t decode_step_count = 0;          // Decode step 调用次数
    
    std::vector<double> decode_step_times_ms;  // 每次 decode_step 的耗时记录
};

class Metric : public NonCopyable {
public:
    Metric();
    ~Metric() = default;

    void start_prefill();
    void end_prefill();
    
    void start_decode_step();
    void end_decode_step(int32_t tokens_generated = 1);

    void reset();

    MetricStats get_stats() const;
    
    double get_prefill_time_ms() const;
    double get_total_decode_time_ms() const;
    double get_avg_decode_step_time_ms() const;
    double get_tokens_per_second() const;
    int32_t get_total_tokens() const;

private:
    using TimePoint = std::chrono::high_resolution_clock::time_point;
    
    TimePoint prefill_start_;
    bool prefill_active_;
    
    TimePoint decode_step_start_;
    bool decode_step_active_;
    
    double prefill_time_ms_;
    double total_decode_time_ms_;
    std::vector<double> decode_step_times_ms_;
    
    int32_t total_tokens_;
    int32_t prefill_count_;
    int32_t decode_step_count_;
    
    double calculate_tokens_per_second() const;
    double calculate_avg_decode_step_time() const;
    double calculate_min_decode_step_time() const;
    double calculate_max_decode_step_time() const;
    
    double time_diff_ms(const TimePoint& start, const TimePoint& end) const;
};

} // namespace edge_fm

