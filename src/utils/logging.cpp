#include "logging.h"

namespace edge_fm {

Logging& Logging::instance() {
    static Logging instance;
    return instance;
}

Logging::~Logging() {
    initialized_ = false;
}

void Logging::ensure_initialized() {
    if (!initialized_) {
        init();
    }
}

void Logging::init(LogLevel log_level, bool log_to_file, const std::string& log_file_path) {
    if (initialized_) {
        return;
    }

    level_ = log_level;
    
    std::vector<spdlog::sink_ptr> sinks;
    
    if (log_to_file && !log_file_path.empty()) {
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_file_path, 10 * 1024 * 1024, 5);
        sinks.push_back(file_sink);
    } else {
        auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        sinks.push_back(stdout_sink);
    }
    
    auto logger = std::make_shared<spdlog::logger>("edge_fm", sinks.begin(), sinks.end());
    
    logger->set_level(to_spdlog_level(log_level));
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    logger->flush_on(spdlog::level::warn);
    
    spdlog::set_default_logger(logger);
    initialized_ = true;
}

void Logging::configure(LogLevel log_level, bool log_to_file, const std::string& log_file_path) {
    if (initialized_) {
        shutdown();
    }
    init(log_level, log_to_file, log_file_path);
}

void Logging::set_level(LogLevel level) {
    ensure_initialized();
    level_ = level;
    spdlog::set_level(to_spdlog_level(level));
}

LogLevel Logging::get_level() const {
    return level_;
}

void Logging::shutdown() {
    if (initialized_) {
        spdlog::shutdown();
        initialized_ = false;
    }
}

spdlog::level::level_enum Logging::to_spdlog_level(LogLevel level) const {
    switch (level) {
        case LogLevel::LOG_TRACE:
            return spdlog::level::trace;
        case LogLevel::LOG_DEBUG:
            return spdlog::level::debug;
        case LogLevel::LOG_INFO:
            return spdlog::level::info;
        case LogLevel::LOG_WARN:
            return spdlog::level::warn;
        case LogLevel::LOG_ERROR:
            return spdlog::level::err;
        case LogLevel::LOG_CRITICAL:
            return spdlog::level::critical;
        case LogLevel::LOG_OFF:
            return spdlog::level::off;
        default:
            return spdlog::level::info;
    }
}

} // namespace edge_fm

