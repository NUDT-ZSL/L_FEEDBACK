#include "terrain/terrain.hpp"

#include "terrain/version.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace terrain {
namespace {

uint32_t mix32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float latticeValue(int32_t x, int32_t z, uint32_t seed, int octave) {
    uint32_t hash = seed ^ 0x9e3779b9u;
    hash = mix32(hash ^ static_cast<uint32_t>(x) * 0x85ebca6bu);
    hash = mix32(hash ^ static_cast<uint32_t>(z) * 0xc2b2ae35u);
    hash = mix32(hash ^ static_cast<uint32_t>(octave + 1) * 0x27d4eb2du);

    const float zeroToOne = static_cast<float>(hash & 0x00ffffffu) /
                            static_cast<float>(0x01000000u);
    return zeroToOne * 2.0f - 1.0f;
}

float valueNoise(float x, float z, uint32_t seed, int octave) {
    const int32_t x0 = static_cast<int32_t>(std::floor(x));
    const int32_t z0 = static_cast<int32_t>(std::floor(z));
    const int32_t x1 = x0 + 1;
    const int32_t z1 = z0 + 1;

    const float tx = x - static_cast<float>(x0);
    const float tz = z - static_cast<float>(z0);
    const float sx = tx * tx * (3.0f - 2.0f * tx);
    const float sz = tz * tz * (3.0f - 2.0f * tz);

    const float a = latticeValue(x0, z0, seed, octave);
    const float b = latticeValue(x1, z0, seed, octave);
    const float c = latticeValue(x0, z1, seed, octave);
    const float d = latticeValue(x1, z1, seed, octave);

    const float ab = a + (b - a) * sx;
    const float cd = c + (d - c) * sx;
    return ab + (cd - ab) * sz;
}

}  // namespace

Terrain::Terrain(const TerrainParameters& parameters)
    : parameters_(parameters) {
    if (parameters_.amplitude <= 0.0f ||
        parameters_.frequency <= 0.0f ||
        parameters_.octaves < 1) {
        throw std::invalid_argument("invalid terrain parameters");
    }

    maxDepth_ = 0;
    while ((kPatchQuads << maxDepth_) < (kTerrainGridSize - 1)) {
        ++maxDepth_;
    }

    generateHeights();
    computeVersion();
}

void Terrain::generateHeights() {
    heights_.resize(static_cast<std::size_t>(kTerrainGridSize) * kTerrainGridSize);

    const float inverseExtent = 1.0f / extent();
    uint64_t checksum = 1469598103934665603ull;
    for (int z = 0; z < kTerrainGridSize; ++z) {
        for (int x = 0; x < kTerrainGridSize; ++x) {
            const float nx = static_cast<float>(x) * inverseExtent;
            const float nz = static_cast<float>(z) * inverseExtent;

            float sum = 0.0f;
            float weight = 1.0f;
            float weightSum = 0.0f;
            float octaveFrequency = parameters_.frequency;
            for (int octave = 0; octave < parameters_.octaves; ++octave) {
                sum += weight * valueNoise(
                    nx * octaveFrequency,
                    nz * octaveFrequency,
                    parameters_.seed,
                    octave);
                weightSum += weight;
                weight *= 0.5f;
                octaveFrequency *= 2.0f;
            }

            const float height = parameters_.amplitude * (sum / weightSum);
            const std::size_t index =
                static_cast<std::size_t>(z) * kTerrainGridSize +
                static_cast<std::size_t>(x);
            heights_[index] = height;
            checksum = fnv1aBytes(&height, sizeof(height), checksum);
        }
    }
    heightChecksum_ = checksum;
}

void Terrain::computeVersion() {
    version_ = computeTerrainVersion(parameters_);
    version_ ^= heightChecksum_ >> 17;
    version_ ^= static_cast<uint64_t>(maxDepth_) << 48;
}

float Terrain::sampleLattice(float x, float z) const {
    const int ix = std::clamp(static_cast<int>(std::lround(x)), 0, kTerrainGridSize - 1);
    const int iz = std::clamp(static_cast<int>(std::lround(z)), 0, kTerrainGridSize - 1);
    return heights_[static_cast<std::size_t>(iz) * kTerrainGridSize +
                    static_cast<std::size_t>(ix)];
}

float Terrain::sampleHeight(float x, float z) const {
    const float maxCoordinate = extent();
    const float clampedX = clampf(x, 0.0f, maxCoordinate);
    const float clampedZ = clampf(z, 0.0f, maxCoordinate);

    const float gx = clampedX / maxCoordinate * static_cast<float>(kTerrainGridSize - 1);
    const float gz = clampedZ / maxCoordinate * static_cast<float>(kTerrainGridSize - 1);

    const int ix0 = std::clamp(static_cast<int>(std::floor(gx)), 0, kTerrainGridSize - 1);
    const int iz0 = std::clamp(static_cast<int>(std::floor(gz)), 0, kTerrainGridSize - 1);
    const int ix1 = std::min(ix0 + 1, kTerrainGridSize - 1);
    const int iz1 = std::min(iz0 + 1, kTerrainGridSize - 1);
    const float fx = gx - static_cast<float>(ix0);
    const float fz = gz - static_cast<float>(iz0);

    const float h00 = sampleLattice(static_cast<float>(ix0), static_cast<float>(iz0));
    const float h10 = sampleLattice(static_cast<float>(ix1), static_cast<float>(iz0));
    const float h01 = sampleLattice(static_cast<float>(ix0), static_cast<float>(iz1));
    const float h11 = sampleLattice(static_cast<float>(ix1), static_cast<float>(iz1));

    const float hx0 = h00 + (h10 - h00) * fx;
    const float hx1 = h01 + (h11 - h01) * fx;
    return hx0 + (hx1 - hx0) * fz;
}

std::pair<float, float> Terrain::heightBounds(
    float minX, float minZ, float maxX, float maxZ) const {
    const float maxCoordinate = extent();
    const int ix0 = std::clamp(
        static_cast<int>(std::floor(minX / maxCoordinate * (kTerrainGridSize - 1))),
        0,
        kTerrainGridSize - 1);
    const int ix1 = std::clamp(
        static_cast<int>(std::ceil(maxX / maxCoordinate * (kTerrainGridSize - 1))),
        0,
        kTerrainGridSize - 1);
    const int iz0 = std::clamp(
        static_cast<int>(std::floor(minZ / maxCoordinate * (kTerrainGridSize - 1))),
        0,
        kTerrainGridSize - 1);
    const int iz1 = std::clamp(
        static_cast<int>(std::ceil(maxZ / maxCoordinate * (kTerrainGridSize - 1))),
        0,
        kTerrainGridSize - 1);

    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    for (int z = iz0; z <= iz1; ++z) {
        for (int x = ix0; x <= ix1; ++x) {
            const float height =
                heights_[static_cast<std::size_t>(z) * kTerrainGridSize +
                         static_cast<std::size_t>(x)];
            minimum = std::min(minimum, height);
            maximum = std::max(maximum, height);
        }
    }

    if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
        return {0.0f, 0.0f};
    }
    return {minimum - 0.001f, maximum + 0.001f};
}

}  // namespace terrain

