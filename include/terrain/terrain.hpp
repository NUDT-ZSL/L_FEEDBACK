#pragma once

#include "terrain/math.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace terrain {

constexpr int kTerrainGridSize = 512;
constexpr int kPatchQuads = 16;
constexpr int kPatchVertexCount = kPatchQuads + 1;
constexpr uint32_t kDefaultSeed = 1337u;
constexpr float kDefaultAmplitude = 48.0f;
constexpr float kDefaultFrequency = 2.4f;
constexpr int kDefaultOctaves = 5;

struct TerrainParameters {
    uint32_t seed = kDefaultSeed;
    float amplitude = kDefaultAmplitude;
    float frequency = kDefaultFrequency;
    int octaves = kDefaultOctaves;
};

class Terrain {
public:
    explicit Terrain(const TerrainParameters& parameters);

    const TerrainParameters& parameters() const { return parameters_; }
    int gridSize() const { return kTerrainGridSize; }
    float extent() const { return static_cast<float>(kTerrainGridSize - 1); }
    int maxDepth() const { return maxDepth_; }
    uint64_t version() const { return version_; }
    uint64_t heightChecksum() const { return heightChecksum_; }

    float sampleHeight(float x, float z) const;
    const std::vector<float>& heights() const { return heights_; }

    std::pair<float, float> heightBounds(
        float minX, float minZ, float maxX, float maxZ) const;

private:
    float sampleLattice(float x, float z) const;
    void generateHeights();
    void computeVersion();

    TerrainParameters parameters_{};
    std::vector<float> heights_;
    int maxDepth_ = 0;
    uint64_t version_ = 0;
    uint64_t heightChecksum_ = 0;
};

}  // namespace terrain

