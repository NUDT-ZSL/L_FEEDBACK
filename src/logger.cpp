#include "renderer/logger.hpp"

#include "renderer/error.hpp"

#include <chrono>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace renderer {
namespace {

template <typename T>
std::string format_number(T value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

bool needs_quotes(const std::string& value) {
    if (value.empty()) {
        return true;
    }
    for (const unsigned char character : value) {
        if (!(std::isalnum(character) || character == '.' || character == '_'
              || character == '-' || character == '+' || character == '/'
              || character == ':')) {
            return true;
        }
    }
    return false;
}

void append_field(std::ostringstream& stream, const LogField& field) {
    stream << ' ' << field.key << '=' << Logger::quote(field.value);
}

}  // namespace

LogField log_field(std::string key, std::string value) {
    return {std::move(key), std::move(value)};
}

LogField log_field(std::string key, const char* value) {
    return {std::move(key), value == nullptr ? std::string{} : std::string{value}};
}

LogField log_field(std::string key, int value) {
    return {std::move(key), format_number(value)};
}

LogField log_field(std::string key, unsigned int value) {
    return {std::move(key), format_number(value)};
}

LogField log_field(std::string key, float value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << value;
    return {std::move(key), stream.str()};
}

LogField log_field(std::string key, double value, int precision) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return {std::move(key), stream.str()};
}

LogField log_field(std::string key, bool value) {
    return {std::move(key), value ? "1" : "0"};
}

Logger::Logger(const std::string& path) {
    open(path);
}

Logger::~Logger() {
    flush();
}

void Logger::open(const std::string& path) {
    flush();
    stream_.close();

    const std::filesystem::path outputPath(path);
    if (outputPath.has_parent_path()) {
        std::error_code error;
        std::filesystem::create_directories(outputPath.parent_path(), error);
        if (error) {
            fail("E_LOG_IO", "cannot create log directory: " + error.message());
        }
    }
    stream_.open(outputPath, std::ios::out | std::ios::trunc);
    if (!stream_) {
        fail("E_LOG_IO", "cannot open log file: " + path);
    }
}

bool Logger::isOpen() const {
    return stream_.is_open();
}

void Logger::info(std::string_view event, const std::vector<LogField>& fields) {
    write_line("INFO", event, "OK", fields);
}

void Logger::error(std::string_view errorCode, std::string_view event,
                   const std::vector<LogField>& fields) {
    write_line("ERROR", event, errorCode, fields);
}

void Logger::flush() {
    if (stream_.is_open()) {
        stream_.flush();
    }
}

std::string Logger::timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - seconds);
    const std::time_t time = std::chrono::system_clock::to_time_t(seconds);

    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif

    std::ostringstream stream;
    stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << milliseconds.count()
           << 'Z';
    return stream.str();
}

void Logger::write_line(const std::string& level, std::string_view event,
                        std::string_view errorCode, const std::vector<LogField>& fields) {
    std::ostringstream line;
    line << "timestamp=" << timestamp()
         << " level=" << level
         << " event=" << quote(std::string(event))
         << " error=" << quote(std::string(errorCode));
    for (const LogField& item : fields) {
        append_field(line, item);
    }
    line << '\n';

    if (stream_.is_open()) {
        stream_ << line.str();
        if (!stream_) {
            fail("E_LOG_IO", "failed while writing to the log file");
        }
        if (level == "ERROR") {
            stream_.flush();
        }
    }

    if (level == "ERROR") {
        std::cerr << line.str();
        std::cerr.flush();
    }
}

std::string Logger::quote(const std::string& value) {
    if (!needs_quotes(value)) {
        return value;
    }
    std::ostringstream stream;
    stream << '"';
    for (const char character : value) {
        switch (character) {
        case '\\':
            stream << "\\\\";
            break;
        case '"':
            stream << "\\\"";
            break;
        case '\n':
            stream << "\\n";
            break;
        case '\r':
            stream << "\\r";
            break;
        case '\t':
            stream << "\\t";
            break;
        default:
            stream << character;
            break;
        }
    }
    stream << '"';
    return stream.str();
}

void write_stderr_error(std::string_view errorCode, std::string_view message) {
    std::cerr << "timestamp=" << Logger::timestamp()
              << " level=ERROR event=fatal error=" << Logger::quote(std::string(errorCode))
              << " message=" << Logger::quote(std::string(message)) << '\n';
    std::cerr.flush();
}

}  // namespace renderer
