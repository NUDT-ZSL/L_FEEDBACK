#pragma once

#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace renderer {

struct LogField {
    std::string key;
    std::string value;
};

LogField log_field(std::string key, std::string value);
LogField log_field(std::string key, const char* value);
LogField log_field(std::string key, int value);
LogField log_field(std::string key, unsigned int value);
LogField log_field(std::string key, float value);
LogField log_field(std::string key, double value, int precision = 6);
LogField log_field(std::string key, bool value);

class Logger {
public:
    Logger() = default;
    explicit Logger(const std::string& path);
    ~Logger();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

    void open(const std::string& path);
    bool isOpen() const;
    void info(std::string_view event, const std::vector<LogField>& fields = {});
    void error(std::string_view errorCode, std::string_view event,
               const std::vector<LogField>& fields = {});
    void flush();

    static std::string timestamp();
    static std::string quote(const std::string& value);

private:
    void write_line(const std::string& level, std::string_view event,
                    std::string_view errorCode, const std::vector<LogField>& fields);

    std::ofstream stream_;
};

void write_stderr_error(std::string_view errorCode, std::string_view message);

}  // namespace renderer
