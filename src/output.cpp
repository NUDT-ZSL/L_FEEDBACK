#include "output.hpp"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "json.hpp"

namespace orv {
namespace {

std::string boolJson(bool value) { return value ? "true" : "false"; }

std::filesystem::path uniqueTempPath(const std::filesystem::path& target) {
  static std::uint64_t counter = 0;
#ifdef _WIN32
  const auto pid = GetCurrentProcessId();
#else
  const auto pid = static_cast<unsigned long>(getpid());
#endif
  return std::filesystem::path(target.string() + ".tmp." + std::to_string(pid) + "." +
                               std::to_string(++counter));
}

bool flushFile(const std::filesystem::path& path, AppError& error,
               const std::string& errorCode) {
#ifdef _WIN32
  HANDLE handle = CreateFileW(path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE,
                              FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    error = ioError(errorCode, 6, "failed to reopen temporary file for flush",
                    path.string());
    return false;
  }
  const BOOL flushed = FlushFileBuffers(handle);
  CloseHandle(handle);
  if (!flushed) {
    error = ioError(errorCode, 6, "failed to flush temporary file", path.string());
    return false;
  }
#else
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    error = ioError(errorCode, 6, "failed to reopen temporary file for flush",
                    path.string());
    return false;
  }
  const int result = ::fsync(fd);
  ::close(fd);
  if (result != 0) {
    error = ioError(errorCode, 6, "failed to flush temporary file", path.string());
    return false;
  }
#endif
  return true;
}

bool atomicReplace(const std::filesystem::path& source,
                   const std::filesystem::path& target, AppError& error,
                   const std::string& errorCode) {
#ifdef _WIN32
  if (!MoveFileExW(source.wstring().c_str(), target.wstring().c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    error = ioError(errorCode, 6, "failed to atomically replace output file",
                    target.string());
    return false;
  }
#else
  if (::rename(source.c_str(), target.c_str()) != 0) {
    error = ioError(errorCode, 6,
                    "failed to atomically replace output file: " +
                        std::string(std::strerror(errno)),
                    target.string());
    return false;
  }
  const auto directory = target.parent_path().empty() ? std::filesystem::path(".")
                                                       : target.parent_path();
  const int dirFd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
  if (dirFd >= 0) {
    ::fsync(dirFd);
    ::close(dirFd);
  }
#endif
  return true;
}

bool writeAtomicBytes(const std::filesystem::path& target, const std::string& bytes,
                      const std::string& errorCode, const std::string& description,
                      AppError& error) {
  std::error_code ec;
  const auto parent = target.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      error = ioError(errorCode, 6,
                      "failed to create output directory: " + ec.message(),
                      parent.string());
      return false;
    }
  }

  const auto temporary = uniqueTempPath(target);
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      error = ioError(errorCode, 6, "failed to create temporary " + description,
                      temporary.string());
      return false;
    }
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    stream.flush();
    if (!stream) {
      stream.close();
      std::filesystem::remove(temporary, ec);
      error = ioError(errorCode, 6, "failed to write temporary " + description,
                      temporary.string());
      return false;
    }
  }

  if (!flushFile(temporary, error, errorCode)) {
    std::filesystem::remove(temporary, ec);
    return false;
  }
  if (!atomicReplace(temporary, target, error, errorCode)) {
    std::filesystem::remove(temporary, ec);
    return false;
  }
  return true;
}

}  // namespace

std::filesystem::path makePpmPath(const std::filesystem::path& prefix,
                                  const std::string& passName) {
  return std::filesystem::path(prefix.string() + "_" + passName + ".ppm");
}

std::filesystem::path makeStatsPath(const std::filesystem::path& prefix) {
  return std::filesystem::path(prefix.string() + "_stats.json");
}

std::string makeStatsJson(const RenderStats& stats) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << "{\n";
  stream << "  \"ok\": true,\n";
  stream << "  \"resolution\": {\"width\": " << stats.width
         << ", \"height\": " << stats.height << "},\n";
  stream << "  \"width\": " << stats.width << ",\n";
  stream << "  \"height\": " << stats.height << ",\n";
  stream << "  \"render_time_ms\": " << jsonNumber(stats.renderTimeMs, 9) << ",\n";
  stream << "  \"frame_time_ms\": " << jsonNumber(stats.renderTimeMs, 9) << ",\n";
  stream << "  \"visible_pixels\": " << stats.visiblePixels << ",\n";
  stream << "  \"average_luminance\": " << jsonNumber(stats.averageLuminance, 9) << ",\n";
  stream << "  \"maximum_luminance\": " << jsonNumber(stats.maximumLuminance, 9) << ",\n";
  stream << "  \"average_luminance_clamped\": "
         << boolJson(stats.averageLuminanceClamped) << ",\n";
  stream << "  \"maximum_luminance_clamped\": "
         << boolJson(stats.maximumLuminanceClamped) << ",\n";
  stream << "  \"shadow_method\": \"" << jsonEscape(stats.shadowMethod) << "\",\n";
  stream << "  \"light_count\": " << stats.lightCount << ",\n";
  stream << "  \"pass\": \"" << jsonEscape(stats.pass) << "\"\n";
  stream << "}\n";
  return stream.str();
}

std::string makeFinalStateJson(bool ok, const std::string& state, const std::string& code,
                               int exitCode, const std::string& pass,
                               const std::filesystem::path& outputPath,
                               const std::filesystem::path& statsPath) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << "{\"ok\":" << boolJson(ok) << ",\"state\":\"" << jsonEscape(state)
         << "\",\"code\":\"" << jsonEscape(code) << "\",\"exit_code\":" << exitCode
         << ",\"pass\":\"" << jsonEscape(pass) << "\"";
  if (!outputPath.empty()) {
    stream << ",\"output\":\"" << jsonEscape(outputPath.string()) << "\"";
  }
  if (!statsPath.empty()) {
    stream << ",\"stats\":\"" << jsonEscape(statsPath.string()) << "\"";
  }
  stream << "}";
  return stream.str();
}

std::string makeRecoveryJson(std::size_t removedFiles) {
  return "{\"ok\":true,\"code\":\"RECOVERY_CLEANED\",\"recovered\":true,"
         "\"removed_temp_files\":" +
         std::to_string(removedFiles) + "}";
}

void emitErrorJson(const AppError& error) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << "{\"ok\":false,\"code\":\"" << jsonEscape(error.code)
         << "\",\"message\":\"" << jsonEscape(error.message) << "\"";
  if (!error.path.empty()) {
    stream << ",\"path\":\"" << jsonEscape(error.path) << "\"";
  }
  if (!error.extra_json.empty()) {
    stream << ",\"details\":" << error.extra_json;
  }
  stream << "}\n";
  std::cerr << stream.str();
  std::cerr.flush();
}

bool writePpmAtomically(const std::filesystem::path& path, const Image8& image,
                        AppError& error) {
  if (image.width <= 0 || image.height <= 0 ||
      image.rgb.size() != static_cast<std::size_t>(image.width) *
                              static_cast<std::size_t>(image.height) * 3U) {
    error = ioError("IO_ERR_PPM_WRITE", 6, "invalid RGB image buffer", path.string());
    return false;
  }
  std::string bytes;
  const std::string header = "P6\n" + std::to_string(image.width) + " " +
                             std::to_string(image.height) + "\n255\n";
  bytes.reserve(header.size() + image.rgb.size());
  bytes.append(header);
  bytes.append(reinterpret_cast<const char*>(image.rgb.data()), image.rgb.size());
  return writeAtomicBytes(path, bytes, "IO_ERR_PPM_WRITE", "PPM file", error);
}

bool writeStatsAtomically(const std::filesystem::path& path, const RenderStats& stats,
                          AppError& error) {
  return writeAtomicBytes(path, makeStatsJson(stats), "IO_ERR_STATS_WRITE",
                          "statistics JSON", error);
}

}  // namespace orv
