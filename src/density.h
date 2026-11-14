#pragma once

#include "thread_pool.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cloud {

enum class CloudType : std::uint32_t {
    Cumulus = 0,
    Stratus = 1,
};

CloudType parseCloudType(const std::string& value);
const char* cloudTypeName(CloudType type);

class DensityField {
public:
    static constexpr int kFullResolution = 128;

    DensityField() = default;

    static DensityField build(std::uint64_t seed, CloudType type, int targetResolution,
                              WorkerPool* pool, bool asynchronous);

    int resolution() const { return resolution_; }
    std::uint64_t seed() const { return seed_; }
    CloudType type() const { return type_; }
    const std::vector<std::uint8_t>& data() const { return data_; }
    const std::uint8_t* voxel(int x, int y, int z) const;

    float sampleNormalized(float x, float y, float z) const;
    float sampleAt(int x, int y, int z) const;

    std::uint8_t minimumDensity() const { return minimum_; }
    std::uint8_t maximumDensity() const { return maximum_; }
    double meanDensity() const { return mean_; }

    static DensityField makeTestPattern(int resolution, std::uint64_t seed, CloudType type);

private:
    DensityField(std::uint64_t seed, CloudType type, int resolution, std::vector<std::uint8_t> data);
    void refreshStatistics();
    void generateFull(std::uint64_t seed, CloudType type, WorkerPool* pool);
    void downsampleOne(WorkerPool* pool);

    std::uint64_t seed_ = 0;
    CloudType type_ = CloudType::Cumulus;
    int resolution_ = 0;
    std::vector<std::uint8_t> data_;
    std::uint8_t minimum_ = 0;
    std::uint8_t maximum_ = 0;
    double mean_ = 0.0;
};

int chooseDensityResolution(double availableMemoryMB, std::string* reason = nullptr);
int densityResolutionRequiredMB(int resolution);

}  // namespace cloud

