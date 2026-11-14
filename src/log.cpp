#include "log.h"

#include <iostream>

namespace cloud {

Logger::Logger(const std::string& path) : path_(path) {
    file_.open(path_, std::ios::binary | std::ios::trunc);
    if (!file_) {
        throw std::runtime_error("cannot open log file: " + path_);
    }
}

void Logger::info(const std::string& event, const std::string& message, json::Value fields) {
    write("info", event, message, std::move(fields));
}

void Logger::warning(const std::string& code, const std::string& message, json::Value fields) {
    write("warning", code, message, std::move(fields));
}

void Logger::error(const std::string& code, int exitCode, const std::string& message, json::Value fields) {
    if (fields.isObject()) {
        fields["exitCode"] = exitCode;
    }
    write("error", code, message, std::move(fields));
}

void Logger::write(const char* level, const std::string& code, const std::string& message,
                   json::Value fields) {
    if (!fields.isObject()) {
        fields = json::Value::Object{};
    }
    json::Value::Object& object = fields.asObject();
    object["level"] = level;
    if (!code.empty()) {
        object["code"] = code;
        object["event"] = code;
    }
    object["message"] = message;
    const std::string line = json::stringify(fields);

    std::lock_guard<std::mutex> lock(mutex_);
    file_ << line << '\n';
    file_.flush();
    std::cout << line << '\n';
}

void writeErrorToStderr(const std::string& code, int exitCode, const std::string& message) {
    json::Value::Object object;
    object["level"] = "error";
    object["code"] = code;
    object["message"] = message;
    object["exitCode"] = exitCode;
    std::cerr << json::stringify(object) << '\n';
}

}  // namespace cloud

