#include "terrain/version.hpp"

#include <cstddef>
#include <cstring>

namespace terrain {

uint64_t fnv1aBytes(const void* data, std::size_t size, uint64_t seed) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    uint64_t value = seed;
    for (std::size_t i = 0; i < size; ++i) {
        value ^= static_cast<uint64_t>(bytes[i]);
        value *= 1099511628211ull;
    }
    return value;
}

uint64_t computeTerrainVersion(const TerrainParameters& parameters) {
    struct Canonical {
        uint32_t seed;
        float amplitude;
        float frequency;
        int32_t octaves;
        int32_t gridSize;
        int32_t patchQuads;
    };

    const Canonical canonical{
        parameters.seed,
        parameters.amplitude,
        parameters.frequency,
        parameters.octaves,
        kTerrainGridSize,
        kPatchQuads,
    };
    return fnv1aBytes(&canonical, sizeof(canonical));
}

}  // namespace terrain

