#pragma once

#include <string>
#include <memory>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <vector>
#include "utils/non_copyable.h"

namespace edge_fm {

enum class LogLevel {
    LOG_TRACE = 0,
    LOG_DEBUG = 1,
    LOG_INFO = 2,
    LOG_WARN = 3,
    LOG_ERROR = 4,
    LOG_CRITICAL = 5,
    LOG_OFF = 6
};

class Logging : public NonCopyableNonMovable {
public:
    static Logging& instance();

    void init(LogLevel log_level = LogLevel::LOG_INFO, bool log_to_file = false, const std::string& log_file_path = "");
    void configure(LogLevel log_level, bool log_to_file = false, const std::string& log_file_path = "");

    void set_level(LogLevel level);
    LogLevel get_level() const;

    template<typename... Args>
    void log_trace(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        ensure_initialized();
        spdlog::trace(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void log_debug(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        ensure_initialized();
        spdlog::debug(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void log_info(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        ensure_initialized();
        spdlog::info(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void log_warn(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        ensure_initialized();
        spdlog::warn(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void log_error(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        ensure_initialized();
        spdlog::error(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void log_critical(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        ensure_initialized();
        spdlog::critical(fmt, std::forward<Args>(args)...);
    }

private:
    Logging() = default;
    ~Logging();

    void ensure_initialized();
    void shutdown();
    spdlog::level::level_enum to_spdlog_level(LogLevel level) const;
    
    LogLevel level_ = LogLevel::LOG_INFO;
    bool initialized_ = false;
};

} // namespace edge_fm