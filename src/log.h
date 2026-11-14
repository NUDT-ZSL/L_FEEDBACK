#pragma once

#include "json.h"

#include <fstream>
#include <mutex>
#include <string>

namespace cloud {

class Logger {
public:
    explicit Logger(const std::string& path);

    void info(const std::string& event, const std::string& message, json::Value fields = json::Value::Object{});
    void warning(const std::string& code, const std::string& message,
                 json::Value fields = json::Value::Object{});
    void error(const std::string& code, int exitCode, const std::string& message,
               json::Value fields = json::Value::Object{});

    const std::string& path() const { return path_; }

private:
    void write(const char* level, const std::string& code, const std::string& message, json::Value fields);

    std::string path_;
    std::ofstream file_;
    std::mutex mutex_;
};

void writeErrorToStderr(const std::string& code, int exitCode, const std::string& message);

}  // namespace cloud

