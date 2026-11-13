#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace skinbake {

struct FrameRecord {
    std::uint32_t frameIndex = 0;
    std::uint32_t timeMs = 0;
    std::vector<float> boneMatrices;
    std::vector<float> vertices;
};

struct Snapshot {
    std::uint32_t version = 0;
    std::uint32_t fps = 0;
    std::uint64_t fingerprint = 0;
    std::uint32_t totalFrames = 0;
    std::uint32_t boneCount = 0;
    std::uint32_t vertexCount = 0;
    bool complete = false;
    std::vector<FrameRecord> frames;
};

constexpr std::uint32_t kSnapshotFormatHeaderSize = 64;
constexpr std::uint32_t kSnapshotFlagComplete = 1u;

std::vector<std::uint8_t> serialize_snapshot(const Snapshot& snapshot);
void write_snapshot_atomic(const std::string& path, const Snapshot& snapshot);

Snapshot load_snapshot(const std::string& path,
                       std::optional<std::uint32_t> expectedVersion = std::nullopt,
                       std::optional<std::uint64_t> expectedFingerprint = std::nullopt);

std::uint32_t snapshot_checksum(const std::vector<std::uint8_t>& bytes);
std::size_t frame_record_size(std::uint32_t boneCount,
                              std::uint32_t vertexCount);

}

