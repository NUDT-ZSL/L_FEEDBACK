#pragma once

#include "math.h"
#include "renderer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cloud {

struct SnapshotData {
    std::uint32_t version = 2;
    std::uint64_t seed = 0;
    std::uint32_t cloudType = 0;
    std::uint32_t densityResolution = 128;
    std::uint32_t frameIndex = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t historyLength = 8;
    bool hasLight = true;
    bool upgradedFromV1 = false;
    Light nextLight = defaultLight();
    Mat4 view = Mat4::identity();
    Mat4 projection = Mat4::identity();
    std::vector<Vec3> accumulatedColor;
    std::vector<float> depth;
    std::vector<std::uint32_t> counts;
};

SnapshotData loadSnapshot(const std::string& path);
std::uint32_t saveSnapshot(const std::string& path, const SnapshotData& data, std::uint32_t version);

std::uint32_t snapshotVersionFromDisk(const std::string& path);

}  // namespace cloud

