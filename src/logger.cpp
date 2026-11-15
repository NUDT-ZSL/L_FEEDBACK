#include "automix/logger.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <system_error>

namespace automix {

namespace {

std::string utcTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::filesystem::path archivePath(const std::filesystem::path& active, std::size_t index) {
    std::filesystem::path result = active;
    result += "." + std::to_string(index);
    return result;
}

}  // namespace

Logger::Logger(std::filesystem::path directory,
               std::string filename,
               std::size_t maxBytes,
               std::size_t maxFiles)
    : directory_(std::move(directory)),
      filename_(std::move(filename)),
      maxBytes_(maxBytes),
      maxFiles_(maxFiles < 1 ? 1 : maxFiles) {
    if (!ensureDirectory()) {
        return;
    }
    const std::filesystem::path active = directory_ / filename_;
    std::ofstream probe(active, std::ios::app | std::ios::binary);
    if (!probe) {
        degrade("cannot open log file '" + (directory_ / filename_).string() + "'");
    }
}

void Logger::info(const std::string& message) {
    write("INFO", message);
}

void Logger::warning(const std::string& message) {
    write("WARN", message);
}

void Logger::error(const std::string& message) {
    write("ERROR", message);
}

void Logger::write(const char* level, const std::string& message) {
    if (degraded_) {
        std::cerr << "[log:" << level << "] " << message << '\n';
        return;
    }
    if (!ensureDirectory()) {
        return;
    }

    const std::filesystem::path active = directory_ / filename_;
    std::error_code ec;
    const auto size = std::filesystem::exists(active, ec)
                          ? std::filesystem::file_size(active, ec)
                          : 0;
    if (ec) {
        degrade("cannot inspect log file: " + ec.message());
        return;
    }

    const std::string record = utcTimestamp() + " [" + level + "] " + message + "\n";
    if (size > 0 && size + record.size() > maxBytes_) {
        if (!rotate()) {
            return;
        }
    }

    std::ofstream out(active, std::ios::app | std::ios::binary);
    if (!out) {
        degrade("cannot write log file '" + active.string() + "'");
        return;
    }
    out.write(record.data(), static_cast<std::streamsize>(record.size()));
    out.flush();
    if (!out) {
        degrade("log write failed for '" + active.string() + "'");
    }
}

bool Logger::ensureDirectory() {
    std::error_code ec;
    if (std::filesystem::exists(directory_, ec)) {
        if (ec || !std::filesystem::is_directory(directory_, ec)) {
            degrade("log path is not a directory: '" + directory_.string() + "'");
            return false;
        }
        return true;
    }
    ec.clear();
    std::filesystem::create_directories(directory_, ec);
    if (ec) {
        degrade("cannot create log directory '" + directory_.string() + "': " + ec.message());
        return false;
    }
    return true;
}

bool Logger::rotate() {
    const std::filesystem::path active = directory_ / filename_;
    std::error_code ec;

    if (maxFiles_ <= 1) {
        std::filesystem::remove(active, ec);
        if (ec) {
            degrade("log rotation failed: " + ec.message());
            return false;
        }
        return true;
    }

    const std::filesystem::path oldest = archivePath(active, maxFiles_ - 1);
    std::filesystem::remove(oldest, ec);
    if (ec) {
        degrade("log rotation failed while removing '" + oldest.string() + "': " + ec.message());
        return false;
    }

    for (std::size_t index = maxFiles_ - 1; index > 1; --index) {
        const std::filesystem::path from = archivePath(active, index - 1);
        const std::filesystem::path to = archivePath(active, index);
        ec.clear();
        if (std::filesystem::exists(from, ec)) {
            std::filesystem::rename(from, to, ec);
            if (ec) {
                degrade("log rotation failed while renaming '" + from.string() + "': " +
                        ec.message());
                return false;
            }
        }
    }

    ec.clear();
    if (std::filesystem::exists(active, ec)) {
        std::filesystem::rename(active, archivePath(active, 1), ec);
        if (ec) {
            degrade("log rotation failed while archiving active log: " + ec.message());
            return false;
        }
    }
    return true;
}

void Logger::degrade(const std::string& reason) {
    degraded_ = true;
    if (!warned_) {
        warned_ = true;
        std::cerr << "warning: logging disabled; continuing without file logs: " << reason
                  << '\n';
    }
}

}  // namespace automix
