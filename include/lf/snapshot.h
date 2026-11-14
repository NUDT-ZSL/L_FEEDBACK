#pragma once

#include "lf/config.h"
#include "lf/error_code.h"

#include <filesystem>
#include <string>
#include <vector>

namespace lf {

constexpr int kSnapshotFormatVersion = 2;

struct Snapshot {
  int formatVersion = kSnapshotFormatVersion;
  std::string savedAt;
  LightingState state;
};

struct SnapshotLoadResult {
  Snapshot snapshot;
  std::vector<std::string> migrationLog;
};

[[nodiscard]] std::string utc_timestamp_now();
[[nodiscard]] Snapshot make_snapshot(const LightingState& state);

void save_snapshot(const std::filesystem::path& path, const Snapshot& snapshot);
[[nodiscard]] SnapshotLoadResult load_snapshot(const std::filesystem::path& path);

}  // namespace lf

