#pragma once

#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace skin {

enum class LogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Stat,
};

class Logger {
public:
    Logger() = default;
    ~Logger();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    bool open(const std::filesystem::path& path);
    [[nodiscard]] bool isOpen() const;
    [[nodiscard]] const std::filesystem::path& path() const;

    void log(LogLevel level, const std::string& group, const std::string& message);
    void debug(const std::string& group, const std::string& message);
    void info(const std::string& group, const std::string& message);
    void warning(const std::string& group, int code, const std::string& message);
    void error(const std::string& group, int code, const std::string& message);
    void stat(const std::string& group, const std::string& message);

    static std::string timestamp();

private:
    std::ofstream stream_;
    std::filesystem::path path_;
    std::mutex mutex_;
};

}  // namespace skin

