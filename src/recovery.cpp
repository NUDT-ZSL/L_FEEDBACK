#include "recovery.hpp"

#include <chrono>
#include <fstream>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "json.hpp"
#include "output.hpp"

namespace orv {

std::filesystem::path markerPathForPrefix(const std::filesystem::path& prefix) {
  return std::filesystem::path(prefix.string() + ".render-in-progress.json");
}

namespace {

bool looksLikeTempForPrefix(const std::filesystem::path& candidate,
                            const std::filesystem::path& prefix) {
  const std::string filename = candidate.filename().string();
  const std::string base = prefix.filename().string();
  return filename.rfind(base + "_", 0) == 0 &&
         filename.find(".tmp.") != std::string::npos;
}

}  // namespace

RecoveryReport recoverIncompleteSession(const std::filesystem::path& prefix,
                                        AppError& cleanupError) {
  RecoveryReport report;
  const auto marker = markerPathForPrefix(prefix);
  std::error_code ec;
  if (!std::filesystem::exists(marker, ec)) {
    return report;
  }
  report.foundMarker = true;

  const auto directory =
      prefix.parent_path().empty() ? std::filesystem::path(".") : prefix.parent_path();
  if (std::filesystem::exists(directory, ec)) {
    for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
      if (ec) {
        break;
      }
      if (!entry.is_regular_file(ec)) {
        continue;
      }
      if (looksLikeTempForPrefix(entry.path(), prefix)) {
        std::error_code removeError;
        if (std::filesystem::remove(entry.path(), removeError)) {
          report.removedTempFiles.push_back(entry.path());
        }
      }
    }
  }

  std::error_code removeError;
  if (!std::filesystem::remove(marker, removeError)) {
    cleanupError = ioError("RECOVERY_CLEAN_FAILED", 0,
                           "unable to remove stale render marker: " +
                               removeError.message(),
                           marker.string());
  }
  return report;
}

SessionMarker::SessionMarker(std::filesystem::path prefix) : prefix_(std::move(prefix)) {}

SessionMarker::~SessionMarker() {
  if (active_) {
    std::error_code ec;
    std::filesystem::remove(markerPathForPrefix(prefix_), ec);
  }
}

bool SessionMarker::begin(AppError& error) {
  std::error_code ec;
  const auto marker = markerPathForPrefix(prefix_);
  const auto parent = marker.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      error = ioError("IO_ERR_MARKER_WRITE", 6,
                      "unable to create output directory: " + ec.message(),
                      parent.string());
      return false;
    }
  }
  std::ofstream stream(marker, std::ios::binary | std::ios::trunc);
  if (!stream) {
    error = ioError("IO_ERR_MARKER_WRITE", 6, "unable to create render marker",
                    marker.string());
    return false;
  }
#ifdef _WIN32
  const auto pid = static_cast<unsigned long>(GetCurrentProcessId());
#else
  const auto pid = static_cast<unsigned long>(getpid());
#endif
  const auto started = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
  stream << "{\"version\":1,\"pid\":" << pid << ",\"prefix\":\""
         << jsonEscape(prefix_.string()) << "\",\"started_unix_ms\":" << started
         << "}\n";
  stream.flush();
  if (!stream) {
    stream.close();
    std::filesystem::remove(marker, ec);
    error = ioError("IO_ERR_MARKER_WRITE", 6, "unable to write render marker",
                    marker.string());
    return false;
  }
  active_ = true;
  return true;
}

bool SessionMarker::complete(AppError& error) {
  if (!active_) {
    return true;
  }
  std::error_code ec;
  if (!std::filesystem::remove(markerPathForPrefix(prefix_), ec)) {
    error = ioError("IO_ERR_MARKER_REMOVE", 0,
                    "unable to remove render marker: " + ec.message(),
                    markerPathForPrefix(prefix_).string());
    return false;
  }
  active_ = false;
  return true;
}

void SessionMarker::abandon() {
  std::error_code ec;
  std::filesystem::remove(markerPathForPrefix(prefix_), ec);
  active_ = false;
}

}  // namespace orv
