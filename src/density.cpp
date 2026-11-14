#include "density.h"

#include "math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace cloud {
namespace {

constexpr std::uint32_t kFullSamples =
    static_cast<std::uint32_t>(DensityField::kFullResolution * DensityField::kFullResolution *
                               DensityField::kFullResolution);

std::uint32_t hashU32(std::uint32_t value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

std::uint32_t hash3(int x, int y, int z, std::uint64_t seed) {
    std::uint32_t value = static_cast<std::uint32_t>(seed);
    value ^= hashU32(static_cast<std::uint32_t>(x) + 0x9e3779b9u);
    value ^= hashU32(static_cast<std::uint32_t>(y) + 0x85ebca6bu);
    value ^= hashU32(static_cast<std::uint32_t>(z) + 0xc2b2ae35u);
    value ^= static_cast<std::uint32_t>(seed >> 32u);
    return hashU32(value);
}

float gradientDot(std::uint32_t hash, float x, float y, float z) {
    switch (hash & 15u) {
        case 0: return x + y;
        case 1: return -x + y;
        case 2: return x - y;
        case 3: return -x - y;
        case 4: return x + z;
        case 5: return -x + z;
        case 6: return x - z;
        case 7: return -x - z;
        case 8: return y + z;
        case 9: return -y + z;
        case 10: return y - z;
        case 11: return -y - z;
        case 12: return x + y + z;
        case 13: return -x + y - z;
        case 14: return x - y + z;
        default: return -x - y - z;
    }
}

float quintic(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float perlin3(float x, float y, float z, std::uint64_t seed) {
    const int ix = static_cast<int>(std::floor(x));
    const int iy = static_cast<int>(std::floor(y));
    const int iz = static_cast<int>(std::floor(z));
    const float fx = x - static_cast<float>(ix);
    const float fy = y - static_cast<float>(iy);
    const float fz = z - static_cast<float>(iz);
    const float u = quintic(fx);
    const float v = quintic(fy);
    const float w = quintic(fz);

    const float n000 = gradientDot(hash3(ix, iy, iz, seed), fx, fy, fz);
    const float n100 = gradientDot(hash3(ix + 1, iy, iz, seed), fx - 1.0f, fy, fz);
    const float n010 = gradientDot(hash3(ix, iy + 1, iz, seed), fx, fy - 1.0f, fz);
    const float n110 = gradientDot(hash3(ix + 1, iy + 1, iz, seed), fx - 1.0f, fy - 1.0f, fz);
    const float n001 = gradientDot(hash3(ix, iy, iz + 1, seed), fx, fy, fz - 1.0f);
    const float n101 = gradientDot(hash3(ix + 1, iy, iz + 1, seed), fx - 1.0f, fy, fz - 1.0f);
    const float n011 = gradientDot(hash3(ix, iy + 1, iz + 1, seed), fx, fy - 1.0f, fz - 1.0f);
    const float n111 =
        gradientDot(hash3(ix + 1, iy + 1, iz + 1, seed), fx - 1.0f, fy - 1.0f, fz - 1.0f);

    const float x00 = lerpf(n000, n100, u);
    const float x10 = lerpf(n010, n110, u);
    const float x01 = lerpf(n001, n101, u);
    const float x11 = lerpf(n011, n111, u);
    return lerpf(lerpf(x00, x10, v), lerpf(x01, x11, v), w);
}

float perlinFbm(float x, float y, float z, std::uint64_t seed) {
    float sum = 0.0f;
    float amplitude = 0.5f;
    float normalization = 0.0f;
    float frequency = 4.0f;
    for (int octave = 0; octave < 4; ++octave) {
        sum += amplitude * perlin3(x * frequency, y * frequency, z * frequency,
                                   seed + static_cast<std::uint64_t>(octave) * 0x10001u);
        normalization += amplitude;
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    return (sum / normalization) * 0.5f + 0.5f;
}

float worleyF1(float x, float y, float z, float frequency, std::uint64_t seed) {
    const float px = x * frequency;
    const float py = y * frequency;
    const float pz = z * frequency;
    const int cx = static_cast<int>(std::floor(px));
    const int cy = static_cast<int>(std::floor(py));
    const int cz = static_cast<int>(std::floor(pz));
    float minimumDistanceSquared = 4.0f;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = cx + dx;
                const int ny = cy + dy;
                const int nz = cz + dz;
                const std::uint32_t h = hash3(nx, ny, nz, seed);
                const float offX = static_cast<float>(h & 0x3ffu) / 1024.0f;
                const float offY = static_cast<float>((h >> 10u) & 0x3ffu) / 1024.0f;
                const float offZ = static_cast<float>((h >> 20u) & 0x3ffu) / 1024.0f;
                const float featureX = static_cast<float>(nx) + offX;
                const float featureY = static_cast<float>(ny) + offY;
                const float featureZ = static_cast<float>(nz) + offZ;
                const float deltaX = px - featureX;
                const float deltaY = py - featureY;
                const float deltaZ = pz - featureZ;
                const float distanceSquared =
                    deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ;
                minimumDistanceSquared = std::min(minimumDistanceSquared, distanceSquared);
            }
        }
    }
    return std::sqrt(minimumDistanceSquared) / frequency;
}

float worleyFbm(float x, float y, float z, std::uint64_t seed) {
    float sum = 0.0f;
    float amplitude = 0.5f;
    float normalization = 0.0f;
    float frequency = 4.0f;
    for (int octave = 0; octave < 3; ++octave) {
        const float f1 = clampf(worleyF1(x, y, z, frequency,
                                        seed + static_cast<std::uint64_t>(octave) * 0x9e37u),
                                0.0f, 1.0f);
        sum += amplitude * (1.0f - f1);
        normalization += amplitude;
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    return sum / normalization;
}

std::uint8_t makeDensity(CloudType type, float x, float y, float z, std::uint64_t seed) {
    const float perlin = perlinFbm(x, y, z, seed);
    const float worley = worleyFbm(x, y, z, seed ^ 0xa5a5a5a5u);
    const float field = clampf(0.56f * perlin + 0.44f * worley, 0.0f, 1.0f);

    float density = 0.0f;
    if (type == CloudType::Cumulus) {
        const float radiusX = 2.0f * x - 1.0f;
        const float radiusZ = 2.0f * z - 1.0f;
        const float radialDistance = std::sqrt(radiusX * radiusX + radiusZ * radiusZ);
        const float top = 0.69f + 0.17f * perlin + 0.035f * worley;
        const float normalizedHeight = clampf(y / top, 0.0f, 1.0f);
        const float radiusLimit =
            0.58f + 0.38f * std::pow(std::sin(normalizedHeight * kPi), 0.78f);
        const float horizontal =
            1.0f - smoothstep(radiusLimit - 0.16f, radiusLimit, radialDistance);
        const float vertical = smoothstep(0.025f, 0.16f, y) *
                               (1.0f - smoothstep(top - 0.11f, top + 0.025f, y));
        const float base = smoothstep(0.28f, 0.82f, y);
        const float threshold = 0.445f + 0.10f * (1.0f - base);
        const float shaped = clampf((field - threshold) / (1.0f - threshold), 0.0f, 1.0f);
        density = horizontal * vertical * shaped;
        density *= 0.88f + 0.12f * perlin;
    } else {
        const float horizontal = smoothstep(0.0f, 0.14f, x) *
                                 smoothstep(0.0f, 0.14f, 1.0f - x) *
                                 smoothstep(0.0f, 0.14f, z) *
                                 smoothstep(0.0f, 0.14f, 1.0f - z);
        const float vertical = smoothstep(0.28f, 0.36f, y) * (1.0f - smoothstep(0.66f, 0.74f, y));
        const float threshold = 0.39f;
        const float shaped = clampf((field - threshold) / (1.0f - threshold), 0.0f, 1.0f);
        density = horizontal * vertical * shaped;
        density *= 0.9f + 0.1f * worley;
    }

    const float expanded = clampf(std::pow(density, 0.82f), 0.0f, 1.0f);
    return static_cast<std::uint8_t>(std::floor(expanded * 255.0f + 0.5f));
}

}  // namespace

CloudType parseCloudType(const std::string& value) {
    if (value == "cumulus") {
        return CloudType::Cumulus;
    }
    if (value == "stratus") {
        return CloudType::Stratus;
    }
    throw std::invalid_argument("cloudType must be 'cumulus' or 'stratus'");
}

const char* cloudTypeName(CloudType type) {
    return type == CloudType::Cumulus ? "cumulus" : "stratus";
}

DensityField::DensityField(std::uint64_t seed, CloudType type, int resolution,
                           std::vector<std::uint8_t> data)
    : seed_(seed), type_(type), resolution_(resolution), data_(std::move(data)) {
    refreshStatistics();
}

DensityField DensityField::build(std::uint64_t seed, CloudType type, int targetResolution,
                                 WorkerPool* pool, bool asynchronous) {
    if (targetResolution != 128 && targetResolution != 64 && targetResolution != 32) {
        throw std::invalid_argument("density resolution must be 128, 64, or 32");
    }

    const std::uint32_t sampleCount = kFullSamples;
    std::vector<std::uint8_t> full(sampleCount, 0);
    DensityField field(seed, type, kFullResolution, std::move(full));

    WorkerPool* usablePool = asynchronous ? pool : nullptr;
    field.generateFull(seed, type, usablePool);
    while (field.resolution_ > targetResolution) {
        field.downsampleOne(usablePool);
    }
    field.refreshStatistics();
    return field;
}

const std::uint8_t* DensityField::voxel(int x, int y, int z) const {
    if (x < 0 || y < 0 || z < 0 || x >= resolution_ || y >= resolution_ || z >= resolution_) {
        return nullptr;
    }
    const std::size_t index = static_cast<std::size_t>((z * resolution_ + y) * resolution_ + x);
    return &data_[index];
}

void DensityField::generateFull(std::uint64_t seed, CloudType type, WorkerPool* pool) {
    constexpr int resolution = kFullResolution;
    const int taskRows = std::max(1, resolution / 8);

    const auto generateRange = [this, seed, type](int zBegin, int zEnd) {
        for (int z = zBegin; z < zEnd; ++z) {
            const float zCoordinate = static_cast<float>(z) / static_cast<float>(resolution - 1);
            for (int y = 0; y < resolution; ++y) {
                const float yCoordinate = static_cast<float>(y) / static_cast<float>(resolution - 1);
                for (int x = 0; x < resolution; ++x) {
                    const float xCoordinate = static_cast<float>(x) / static_cast<float>(resolution - 1);
                    const std::size_t index =
                        static_cast<std::size_t>((z * resolution + y) * resolution + x);
                    data_[index] = makeDensity(type, xCoordinate, yCoordinate, zCoordinate, seed);
                }
            }
        }
    };

    if (pool == nullptr) {
        generateRange(0, resolution);
        return;
    }
    for (int zBegin = 0; zBegin < resolution; zBegin += taskRows) {
        const int zEnd = std::min(resolution, zBegin + taskRows);
        pool->submit([generateRange, zBegin, zEnd] { generateRange(zBegin, zEnd); });
    }
    pool->waitIdle();
}

void DensityField::downsampleOne(WorkerPool* pool) {
    if (resolution_ <= 32) {
        throw std::runtime_error("cannot downsample density below 32^3");
    }
    const int sourceResolution = resolution_;
    const int targetResolution = sourceResolution / 2;
    std::vector<std::uint8_t> target(
        static_cast<std::size_t>(targetResolution * targetResolution * targetResolution), 0);
    const std::uint8_t* source = data_.data();

    const auto downsampleRange = [source, sourceResolution, targetResolution, &target](int zBegin,
                                                                                      int zEnd) {
        for (int z = zBegin; z < zEnd; ++z) {
            for (int y = 0; y < targetResolution; ++y) {
                for (int x = 0; x < targetResolution; ++x) {
                    unsigned sum = 0;
                    for (int dz = 0; dz < 2; ++dz) {
                        for (int dy = 0; dy < 2; ++dy) {
                            for (int dx = 0; dx < 2; ++dx) {
                                const int sx = x * 2 + dx;
                                const int sy = y * 2 + dy;
                                const int sz = z * 2 + dz;
                                const std::size_t sourceIndex =
                                    static_cast<std::size_t>((sz * sourceResolution + sy) *
                                                                 sourceResolution +
                                                             sx);
                                sum += source[sourceIndex];
                            }
                        }
                    }
                    const std::size_t targetIndex =
                        static_cast<std::size_t>((z * targetResolution + y) *
                                                 targetResolution +
                                                 x);
                    target[targetIndex] = static_cast<std::uint8_t>((sum + 4u) / 8u);
                }
            }
        }
    };

    if (pool == nullptr) {
        downsampleRange(0, targetResolution);
    } else {
        const int taskRows = std::max(1, targetResolution / 4);
        for (int zBegin = 0; zBegin < targetResolution; zBegin += taskRows) {
            const int zEnd = std::min(targetResolution, zBegin + taskRows);
            pool->submit([downsampleRange, zBegin, zEnd] { downsampleRange(zBegin, zEnd); });
        }
        pool->waitIdle();
    }

    data_ = std::move(target);
    resolution_ = targetResolution;
}

float DensityField::sampleNormalized(float x, float y, float z) const {
    if (data_.empty() || resolution_ == 0) {
        return 0.0f;
    }
    const float gx = clampf(x, 0.0f, 1.0f) * static_cast<float>(resolution_ - 1);
    const float gy = clampf(y, 0.0f, 1.0f) * static_cast<float>(resolution_ - 1);
    const float gz = clampf(z, 0.0f, 1.0f) * static_cast<float>(resolution_ - 1);
    const int ix = std::min(static_cast<int>(std::floor(gx)), resolution_ - 2);
    const int iy = std::min(static_cast<int>(std::floor(gy)), resolution_ - 2);
    const int iz = std::min(static_cast<int>(std::floor(gz)), resolution_ - 2);
    const float fx = gx - static_cast<float>(ix);
    const float fy = gy - static_cast<float>(iy);
    const float fz = gz - static_cast<float>(iz);

    const auto at = [this](int x, int y, int z) {
        return static_cast<float>(
            data_[static_cast<std::size_t>((z * resolution_ + y) * resolution_ + x)]);
    };
    const float c000 = at(ix, iy, iz);
    const float c100 = at(ix + 1, iy, iz);
    const float c010 = at(ix, iy + 1, iz);
    const float c110 = at(ix + 1, iy + 1, iz);
    const float c001 = at(ix, iy, iz + 1);
    const float c101 = at(ix + 1, iy, iz + 1);
    const float c011 = at(ix, iy + 1, iz + 1);
    const float c111 = at(ix + 1, iy + 1, iz + 1);
    const float x00 = lerpf(c000, c100, fx);
    const float x10 = lerpf(c010, c110, fx);
    const float x01 = lerpf(c001, c101, fx);
    const float x11 = lerpf(c011, c111, fx);
    return lerpf(lerpf(x00, x10, fy), lerpf(x01, x11, fy), fz) / 255.0f;
}

float DensityField::sampleAt(int x, int y, int z) const {
    const std::uint8_t* value = voxel(x, y, z);
    return value == nullptr ? 0.0f : static_cast<float>(*value) / 255.0f;
}

void DensityField::refreshStatistics() {
    if (data_.empty()) {
        minimum_ = maximum_ = 0;
        mean_ = 0.0;
        return;
    }
    minimum_ = 255;
    maximum_ = 0;
    std::uint64_t sum = 0;
    for (const std::uint8_t value : data_) {
        minimum_ = std::min(minimum_, value);
        maximum_ = std::max(maximum_, value);
        sum += value;
    }
    mean_ = static_cast<double>(sum) / static_cast<double>(data_.size());
}

DensityField DensityField::makeTestPattern(int resolution, std::uint64_t seed, CloudType type) {
    if (resolution <= 0) {
        throw std::invalid_argument("test pattern resolution must be positive");
    }
    std::vector<std::uint8_t> data(static_cast<std::size_t>(resolution * resolution * resolution));
    for (int z = 0; z < resolution; ++z) {
        for (int y = 0; y < resolution; ++y) {
            for (int x = 0; x < resolution; ++x) {
                const std::uint32_t h = hash3(x, y, z, seed);
                std::uint8_t value = static_cast<std::uint8_t>(h & 0xffu);
                if (type == CloudType::Stratus) {
                    value = static_cast<std::uint8_t>((value + y * 17u) & 0xffu);
                }
                data[static_cast<std::size_t>((z * resolution + y) * resolution + x)] = value;
            }
        }
    }
    return DensityField(seed, type, resolution, std::move(data));
}

int densityResolutionRequiredMB(int resolution) {
    switch (resolution) {
        case 128: return 128;
        case 64: return 64;
        case 32: return 32;
        default: return 1'000'000;
    }
}

int chooseDensityResolution(double availableMemoryMB, std::string* reason) {
    int resolution = 0;
    if (availableMemoryMB >= 128.0) {
        resolution = 128;
    } else if (availableMemoryMB >= 64.0) {
        resolution = 64;
    } else if (availableMemoryMB >= 32.0) {
        resolution = 32;
    } else {
        if (reason != nullptr) {
            std::ostringstream out;
            out << "available=" << availableMemoryMB
                << " MiB is below the minimum 32 MiB density footprint";
            *reason = out.str();
        }
        throw std::runtime_error("insufficient memory for 32^3 density path");
    }
    if (reason != nullptr) {
        std::ostringstream out;
        if (resolution == 128) {
            out << "available=" << availableMemoryMB
                << " MiB meets the 128 MiB full-resolution preset";
        } else {
            out << "available=" << availableMemoryMB << " MiB is below the 128 MiB full-resolution preset; "
                << "selected " << resolution << "^3 density";
        }
        *reason = out.str();
    }
    return resolution;
}

}  // namespace cloud

