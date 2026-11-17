#pragma once

#include "math.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace terrain {

constexpr int kDefaultTerrainDimension = 256;
constexpr float kTerrainWorldSize = 256.0f;

struct DataConfig {
  int seed = 1337;
  int width = kDefaultTerrainDimension;
  int height = kDefaultTerrainDimension;
};

struct HeightWeightData {
  int width = 0;
  int height = 0;
  std::vector<float> heights;
  std::vector<uint8_t> weights;
  int zeroWeightTexels = 0;
  int firstZeroU = -1;
  int firstZeroV = -1;
  uint64_t contentHash = 0;

  float heightAt(int u, int v) const;
  Vec3 weightsAt(int u, int v) const;
};

struct DataPaths {
  std::filesystem::path heightPath;
  std::filesystem::path weightPath;
};

HeightWeightData generateHeightWeightData(const DataConfig& config);
void writeHeightWeightData(const HeightWeightData& data, const std::filesystem::path& directory);
HeightWeightData loadHeightWeightData(const std::filesystem::path& directory,
                                      int expectedWidth,
                                      int expectedHeight);
HeightWeightData ensureHeightWeightData(const std::filesystem::path& directory,
                                        const DataConfig& config,
                                        bool allowGeneration,
                                        bool* generated);
DataPaths dataPathsForDirectory(const std::filesystem::path& directory);
std::string hashToHex(uint64_t value);

}  // namespace terrain

