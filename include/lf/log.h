#pragma once

#include "lf/error_code.h"

#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace lf {

class Logger {
 public:
  static Logger& instance();

  void open(const std::filesystem::path& logFile);
  void close();

  void info(const std::string& message);
  void warn(ErrorCode code, const std::string& message);
  void error(ErrorCode code, const std::string& message);

  [[nodiscard]] std::filesystem::path file_path() const;

 private:
  Logger() = default;
  void write(const char* level, ErrorCode code, const std::string& message);

  std::mutex mutex_;
  std::ofstream file_;
  std::filesystem::path path_;
};

inline void log_info(const std::string& message) { Logger::instance().info(message); }
inline void log_warn(ErrorCode code, const std::string& message) {
  Logger::instance().warn(code, message);
}
inline void log_error(ErrorCode code, const std::string& message) {
  Logger::instance().error(code, message);
}

}  // namespace lf

