#include "logger.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace skin {
namespace {

const char* levelName(LogLevel level) {
    switch (level) {
        case LogLevel::Debug:
            return "DEBUG";
        case LogLevel::Info:
            return "INFO";
        case LogLevel::Warning:
            return "WARN";
        case LogLevel::Error:
            return "ERROR";
        case LogLevel::Stat:
            return "STAT";
    }
    return "INFO";
}

}  // namespace

Logger::~Logger() = default;

bool Logger::open(const std::filesystem::path& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    path_ = path;
    stream_.open(path_, std::ios::out | std::ios::trunc);
    return stream_.is_open();
}

bool Logger::isOpen() const {
    return stream_.is_open();
}

const std::filesystem::path& Logger::path() const {
    return path_;
}

std::string Logger::timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &time);
#else
    localtime_r(&time, &local);
#endif
    std::ostringstream output;
    output << std::put_time(&local, "%Y-%m-%dT%H:%M:%S%z");
    return output.str();
}

void Logger::log(
    LogLevel level,
    const std::string& group,
    const std::string& message
) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stream_.is_open()) {
        return;
    }
    const std::string groupValue = group.empty() ? "-" : group;
    stream_ << timestamp()
            << " [" << levelName(level) << "]"
            << " group=\"" << groupValue << "\" "
            << message << '\n';
    stream_.flush();
}

void Logger::debug(const std::string& group, const std::string& message) {
    log(LogLevel::Debug, group, message);
}

void Logger::info(const std::string& group, const std::string& message) {
    log(LogLevel::Info, group, message);
}

void Logger::warning(
    const std::string& group,
    int code,
    const std::string& message
) {
    log(
        LogLevel::Warning,
        group,
        "code=" + std::to_string(code) + " message=" + message
    );
}

void Logger::error(
    const std::string& group,
    int code,
    const std::string& message
) {
    log(
        LogLevel::Error,
        group,
        "code=" + std::to_string(code) + " message=" + message
    );
}

void Logger::stat(const std::string& group, const std::string& message) {
    log(LogLevel::Stat, group, message);
}

}  // namespace skin

