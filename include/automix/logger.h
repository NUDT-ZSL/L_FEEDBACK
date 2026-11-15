#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace automix {

class Logger {
public:
    static constexpr std::size_t kDefaultMaxBytes = 10u * 1024u * 1024u;
    static constexpr std::size_t kDefaultMaxFiles = 3;

    Logger(std::filesystem::path directory,
           std::string filename,
           std::size_t maxBytes = kDefaultMaxBytes,
           std::size_t maxFiles = kDefaultMaxFiles);

    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);

    bool degraded() const noexcept { return degraded_; }
    const std::filesystem::path& directory() const noexcept { return directory_; }

private:
    void write(const char* level, const std::string& message);
    bool ensureDirectory();
    bool rotate();
    void degrade(const std::string& reason);

    std::filesystem::path directory_;
    std::string filename_;
    std::size_t maxBytes_;
    std::size_t maxFiles_;
    bool degraded_ = false;
    bool warned_ = false;
};

}  // namespace automix
