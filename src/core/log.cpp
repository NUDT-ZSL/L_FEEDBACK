#include "lf/log.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace lf {

namespace {

std::string timestamp_now() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &time);
#else
  localtime_r(&time, &tm);
#endif
  std::ostringstream out;
  out << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
  return out.str();
}

}  // namespace

Logger& Logger::instance() {
  static Logger logger;
  return logger;
}

void Logger::open(const std::filesystem::path& logFile) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (file_.is_open()) {
    file_.close();
  }
  path_ = logFile;
  std::error_code ec;
  if (!path_.parent_path().empty()) {
    std::filesystem::create_directories(path_.parent_path(), ec);
  }
  file_.open(path_, std::ios::out | std::ios::app);
  if (!file_.is_open()) {
    std::cerr << "[WARN] E_LOG_FILE_OPEN cannot open log file: " << path_ << '\n';
  }
}

void Logger::close() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (file_.is_open()) {
    file_.flush();
    file_.close();
  }
}

std::filesystem::path Logger::file_path() const { return path_; }

void Logger::info(const std::string& message) { write("INFO", ErrorCode::Ok, message); }

void Logger::warn(ErrorCode code, const std::string& message) {
  write("WARN", code, message);
}

void Logger::error(ErrorCode code, const std::string& message) {
  write("ERROR", code, message);
}

void Logger::write(const char* level, ErrorCode code, const std::string& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::ostringstream line;
  line << '[' << timestamp_now() << "] [" << level << ']';
  if (code != ErrorCode::Ok) {
    line << " [" << error_code_name(code) << ':' << error_code_number(code) << ']';
  }
  line << ' ' << message;

  const std::string text = line.str();
  std::cerr << text << '\n';
  if (file_.is_open()) {
    file_ << text << '\n';
    file_.flush();
  }
}

}  // namespace lf

