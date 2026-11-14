#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "baker/model.h"

namespace baker {

struct Snapshot {
    std::uint32_t version = 0;
    std::uint64_t input_hash = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t layer_count = 0;
    std::uint32_t completed_layers = 0;
    std::uint32_t time_bits = 0;
    std::uint32_t degenerate_layers = 0;
    bool legacy_stats = false;
    bool encoded_degenerate_count = false;
    std::vector<std::uint64_t> layer_pixels;
    std::vector<float> pixels;
};

enum class SnapshotStatus {
    kMissing,
    kLoaded,
    kVersionMismatch,
    kCorrupt,
};

struct SnapshotReadResult {
    SnapshotStatus status = SnapshotStatus::kMissing;
    Snapshot snapshot;
    std::string detail;
};

SnapshotReadResult read_snapshot_file(const std::filesystem::path& path);
std::vector<std::uint8_t> encode_snapshot(const Snapshot& snapshot);
Snapshot make_snapshot(const Scene& scene, std::uint32_t completed_layers, float time,
                       const Stats& stats, const PixelBuffer& pixels);
const char* snapshot_status_name(SnapshotStatus status) noexcept;

}  // namespace baker
