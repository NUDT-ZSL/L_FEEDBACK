#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include "errors.hpp"

namespace orv {

std::filesystem::path markerPathForPrefix(const std::filesystem::path& prefix);

struct RecoveryReport {
  bool foundMarker = false;
  std::vector<std::filesystem::path> removedTempFiles;
};

RecoveryReport recoverIncompleteSession(const std::filesystem::path& prefix,
                                        AppError& cleanupError);

class SessionMarker {
 public:
  explicit SessionMarker(std::filesystem::path prefix);
  ~SessionMarker();

  SessionMarker(const SessionMarker&) = delete;
  SessionMarker& operator=(const SessionMarker&) = delete;

  bool begin(AppError& error);
  bool complete(AppError& error);
  void abandon();
  bool active() const { return active_; }

 private:
  std::filesystem::path prefix_;
  bool active_ = false;
};

}  // namespace orv
