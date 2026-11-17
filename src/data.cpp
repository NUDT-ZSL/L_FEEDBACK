#include "data.hpp"

#include "errors.hpp"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace terrain {
namespace {

constexpr uint32_t kDataVersion = 1;
constexpr char kHeightMagic[4] = {'H', 'F', '3', '2'};
constexpr char kWeightMagic[4] = {'W', 'G', 'H', 'T'};

uint32_t hash2D(int x, int y, uint32_t seed) {
  uint64_t value = static_cast<uint64_t>(static_cast<uint32_t>(x)) * 0x9e3779b185ebca87ULL;
  value ^= static_cast<uint64_t>(static_cast<uint32_t>(y)) * 0xc2b2ae3d27d4eb4fULL;
  value ^= static_cast<uint64_t>(seed) * 0x165667b19e3779f9ULL;
  return static_cast<uint32_t>(hashMix(value));
}

float valueNoise(float x, float y, uint32_t seed) {
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const float tx = smoothstep(0.0f, 1.0f, x - static_cast<float>(x0));
  const float ty = smoothstep(0.0f, 1.0f, y - static_cast<float>(y0));
  const auto sample = [seed](int ix, int iy) {
    return static_cast<float>(hash2D(ix, iy, seed) & 0x00ffffffu) *
               (2.0f / 16777215.0f) -
           1.0f;
  };
  const float n00 = sample(x0, y0);
  const float n10 = sample(x1, y0);
  const float n01 = sample(x0, y1);
  const float n11 = sample(x1, y1);
  const float nx0 = n00 + (n10 - n00) * tx;
  const float nx1 = n01 + (n11 - n01) * tx;
  return nx0 + (nx1 - nx0) * ty;
}

float fbm(float x, float y, uint32_t seed, int octaves) {
  float value = 0.0f;
  float amplitude = 0.5f;
  float frequency = 1.0f;
  float normalization = 0.0f;
  for (int octave = 0; octave < octaves; ++octave) {
    value += valueNoise(x * frequency, y * frequency,
                        seed + static_cast<uint32_t>(octave) * 1013u) *
             amplitude;
    normalization += amplitude;
    amplitude *= 0.5f;
    frequency *= 2.03f;
  }
  return normalization > 0.0f ? value / normalization : 0.0f;
}

float calculateHeight(int u, int v, int width, int height, uint32_t seed) {
  const float nx = static_cast<float>(u) / static_cast<float>(width - 1);
  const float nz = static_cast<float>(v) / static_cast<float>(height - 1);

  const float macro = fbm(nx * 3.1f + 11.0f, nz * 3.1f - 7.0f, seed, 5);
  const float detail = fbm(nx * 13.0f, nz * 13.0f, seed ^ 0x91e10da5u, 3);
  const float ridgeNoise = fbm(nx * 5.0f + 3.0f, nz * 5.0f + 19.0f,
                               seed ^ 0x7f4a7c15u, 4);

  float h = 10.0f + 17.0f * (macro * 0.5f + 0.5f) +
            1.2f * (detail * 0.5f + 0.5f);

  const float ridgeDx = nx - 0.31f;
  const float ridgeDz = nz - 0.47f;
  const float ridgeMask =
      std::exp(-(ridgeDx * ridgeDx / 0.025f + ridgeDz * ridgeDz / 0.085f));
  h += (20.0f + 13.0f * (ridgeNoise * 0.5f + 0.5f)) * ridgeMask;

  const float hillDx = nx - 0.71f;
  const float hillDz = nz - 0.69f;
  h += 10.0f * std::exp(-(hillDx * hillDx + hillDz * hillDz) / 0.018f);
  h += nx * 3.0f;

  const float edgeFade =
      smoothstep(0.0f, 0.055f, nx) * smoothstep(0.0f, 0.055f, 1.0f - nx) *
      smoothstep(0.0f, 0.055f, nz) * smoothstep(0.0f, 0.055f, 1.0f - nz);
  h = 7.0f + (h - 7.0f) * (0.72f + 0.28f * edgeFade);

  return clampf(h, 1.5f, 68.0f);
}

void setLittleU32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
  bytes[offset + 0] = static_cast<uint8_t>(value & 0xffu);
  bytes[offset + 1] = static_cast<uint8_t>((value >> 8u) & 0xffu);
  bytes[offset + 2] = static_cast<uint8_t>((value >> 16u) & 0xffu);
  bytes[offset + 3] = static_cast<uint8_t>((value >> 24u) & 0xffu);
}

uint32_t getLittleU32(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8u) |
         (static_cast<uint32_t>(bytes[2]) << 16u) |
         (static_cast<uint32_t>(bytes[3]) << 24u);
}

uint32_t floatBits(float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "Unexpected float layout");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float bitsFloat(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::vector<uint8_t> readWholeFile(const std::filesystem::path& path,
                                   const std::string& code,
                                   const std::string& description) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw businessError(code, "Cannot open " + description + ": " + path.string());
  }
  const std::streampos end = stream.tellg();
  if (end < 0 || end > static_cast<std::streampos>(512LL * 1024LL * 1024LL)) {
    throw businessError(code, "Invalid or oversized " + description + ": " + path.string());
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  stream.seekg(0, std::ios::beg);
  if (!bytes.empty() &&
      !stream.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()))) {
    throw businessError(code, "Cannot read " + description + ": " + path.string());
  }
  return bytes;
}

void writeWholeFile(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      throw businessError("DATA_WRITE", "Cannot create data file: " + temporary.string());
    }
    if (!bytes.empty()) {
      stream.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
    }
    stream.flush();
    if (!stream) {
      throw businessError("DATA_WRITE", "Failed to write data file: " + temporary.string());
    }
  }
  std::error_code error;
  std::filesystem::remove(path, error);
  error.clear();
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary);
    throw businessError("DATA_WRITE", "Cannot publish data file: " + path.string());
  }
}

}  // namespace

float HeightWeightData::heightAt(int u, int v) const {
  u = std::max(0, std::min(width - 1, u));
  v = std::max(0, std::min(height - 1, v));
  return heights[static_cast<size_t>(v) * static_cast<size_t>(width) +
                 static_cast<size_t>(u)];
}

Vec3 HeightWeightData::weightsAt(int u, int v) const {
  u = std::max(0, std::min(width - 1, u));
  v = std::max(0, std::min(height - 1, v));
  const size_t offset =
      (static_cast<size_t>(v) * static_cast<size_t>(width) + static_cast<size_t>(u)) *
      3u;
  return {
      static_cast<float>(weights[offset + 0]) / 255.0f,
      static_cast<float>(weights[offset + 1]) / 255.0f,
      static_cast<float>(weights[offset + 2]) / 255.0f,
  };
}

HeightWeightData generateHeightWeightData(const DataConfig& config) {
  if (config.width < 2 || config.height < 2) {
    throw usageError("Data width and height must both be at least 2");
  }
  if (static_cast<int64_t>(config.width) * static_cast<int64_t>(config.height) >
      16 * 1024 * 1024) {
    throw usageError("Requested data dimensions are too large");
  }

  HeightWeightData data;
  data.width = config.width;
  data.height = config.height;
  data.heights.resize(static_cast<size_t>(data.width) * static_cast<size_t>(data.height));
  data.weights.resize(data.heights.size() * 3u);

  const uint32_t seed = static_cast<uint32_t>(config.seed);
  float minimumHeight = std::numeric_limits<float>::max();
  float maximumHeight = std::numeric_limits<float>::lowest();

  for (int v = 0; v < data.height; ++v) {
    for (int u = 0; u < data.width; ++u) {
      const float h = calculateHeight(u, v, data.width, data.height, seed);
      data.heights[static_cast<size_t>(v) * static_cast<size_t>(data.width) +
                   static_cast<size_t>(u)] = h;
      minimumHeight = std::min(minimumHeight, h);
      maximumHeight = std::max(maximumHeight, h);
    }
  }

  const float heightRange = std::max(0.001f, maximumHeight - minimumHeight);
  const int zeroStartU = std::max(2, data.width / 2 + 5);
  const int zeroStartV = std::max(2, data.height / 2 - 8);
  const int zeroEndU = std::min(data.width - 2, zeroStartU + 5);
  const int zeroEndV = std::min(data.height - 2, zeroStartV + 5);

  for (int v = 0; v < data.height; ++v) {
    for (int u = 0; u < data.width; ++u) {
      const size_t index =
          static_cast<size_t>(v) * static_cast<size_t>(data.width) +
          static_cast<size_t>(u);
      const float h = data.heights[index];
      const float hn = clamp01((h - minimumHeight) / heightRange);
      const float dx = data.heightAt(u + 1, v) - data.heightAt(u - 1, v);
      const float dz = data.heightAt(u, v + 1) - data.heightAt(u, v - 1);
      const float slope = std::sqrt(dx * dx + dz * dz);
      const float grain =
          static_cast<float>(hash2D(u, v, seed ^ 0xa511e9b3u) & 0xffffu) / 65535.0f;

      float snow = clamp01((hn - 0.60f) * 3.2f) *
                   clamp01(1.0f - slope * 0.72f);
      snow = clamp01(snow * (0.82f + 0.18f * grain));

      float rock = clamp01((slope - 0.20f) * 1.65f + (hn - 0.55f) * 0.55f);
      rock = clamp01(rock * (0.85f + 0.15f * grain));

      float grass = std::max(0.08f, 1.0f - rock - snow);
      const float total = grass + rock + snow;
      const int r = static_cast<int>(std::lround(255.0f * grass / total));
      const int g = static_cast<int>(std::lround(255.0f * rock / total));
      const int b = static_cast<int>(std::lround(255.0f * snow / total));

      if (u >= zeroStartU && u <= zeroEndU && v >= zeroStartV && v <= zeroEndV) {
        data.weights[index * 3u + 0u] = 0;
        data.weights[index * 3u + 1u] = 0;
        data.weights[index * 3u + 2u] = 0;
        if (data.firstZeroU < 0) {
          data.firstZeroU = u;
          data.firstZeroV = v;
        }
        ++data.zeroWeightTexels;
      } else {
        data.weights[index * 3u + 0u] = static_cast<uint8_t>(std::clamp(r, 0, 255));
        data.weights[index * 3u + 1u] = static_cast<uint8_t>(std::clamp(g, 0, 255));
        data.weights[index * 3u + 2u] = static_cast<uint8_t>(std::clamp(b, 0, 255));
      }
    }
  }

  uint64_t hash = 0xcbf29ce484222325ULL;
  for (float height : data.heights) {
    hash ^= static_cast<uint64_t>(floatBits(height));
    hash *= 0x100000001b3ULL;
  }
  for (uint8_t weight : data.weights) {
    hash ^= static_cast<uint64_t>(weight);
    hash *= 0x100000001b3ULL;
  }
  data.contentHash = hashMix(hash);
  return data;
}

DataPaths dataPathsForDirectory(const std::filesystem::path& directory) {
  return {directory / "heightfield.hf32", directory / "material_weights.wgt8"};
}

void writeHeightWeightData(const HeightWeightData& data,
                           const std::filesystem::path& directory) {
  const DataPaths paths = dataPathsForDirectory(directory);
  std::vector<uint8_t> heightBytes(16u + data.heights.size() * 4u);
  std::memcpy(heightBytes.data(), kHeightMagic, 4);
  setLittleU32(heightBytes, 4, kDataVersion);
  setLittleU32(heightBytes, 8, static_cast<uint32_t>(data.width));
  setLittleU32(heightBytes, 12, static_cast<uint32_t>(data.height));
  for (size_t i = 0; i < data.heights.size(); ++i) {
    setLittleU32(heightBytes, 16u + i * 4u, floatBits(data.heights[i]));
  }
  writeWholeFile(paths.heightPath, heightBytes);

  std::vector<uint8_t> weightBytes(16u + data.weights.size());
  std::memcpy(weightBytes.data(), kWeightMagic, 4);
  setLittleU32(weightBytes, 4, kDataVersion);
  setLittleU32(weightBytes, 8, static_cast<uint32_t>(data.width));
  setLittleU32(weightBytes, 12, static_cast<uint32_t>(data.height));
  std::copy(data.weights.begin(), data.weights.end(), weightBytes.begin() + 16);
  writeWholeFile(paths.weightPath, weightBytes);
}

HeightWeightData loadHeightWeightData(const std::filesystem::path& directory,
                                      int expectedWidth,
                                      int expectedHeight) {
  const DataPaths paths = dataPathsForDirectory(directory);
  const std::vector<uint8_t> heightBytes =
      readWholeFile(paths.heightPath, "DATA_MISSING", "heightfield");
  const std::vector<uint8_t> weightBytes =
      readWholeFile(paths.weightPath, "DATA_MISSING", "material weight map");

  if (heightBytes.size() < 16u || std::memcmp(heightBytes.data(), kHeightMagic, 4) != 0) {
    throw businessError("DATA_INVALID", "Heightfield header is invalid");
  }
  if (weightBytes.size() < 16u || std::memcmp(weightBytes.data(), kWeightMagic, 4) != 0) {
    throw businessError("DATA_INVALID", "Material weight header is invalid");
  }
  if (getLittleU32(heightBytes.data() + 4) != kDataVersion ||
      getLittleU32(weightBytes.data() + 4) != kDataVersion) {
    throw businessError("DATA_VERSION", "Unsupported data file version");
  }

  const int width = static_cast<int>(getLittleU32(heightBytes.data() + 8));
  const int height = static_cast<int>(getLittleU32(heightBytes.data() + 12));
  const int weightWidth = static_cast<int>(getLittleU32(weightBytes.data() + 8));
  const int weightHeight = static_cast<int>(getLittleU32(weightBytes.data() + 12));

  if (width != weightWidth || height != weightHeight) {
    throw businessError(
        "DIM_MISMATCH",
        "Heightfield and material weight dimensions differ: " +
            std::to_string(width) + "x" + std::to_string(height) + " vs " +
            std::to_string(weightWidth) + "x" + std::to_string(weightHeight));
  }
  if (width != expectedWidth || height != expectedHeight) {
    throw businessError(
        "DIM_MISMATCH",
        "Expected " + std::to_string(expectedWidth) + "x" +
            std::to_string(expectedHeight) + " data, received " +
            std::to_string(width) + "x" + std::to_string(height));
  }
  if (width < 2 || height < 2) {
    throw businessError("DIM_MISMATCH", "Terrain dimensions must be at least 2x2");
  }

  const uint64_t heightCount = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  const uint64_t expectedHeightBytes = 16u + heightCount * 4u;
  const uint64_t expectedWeightBytes = 16u + heightCount * 3u;
  if (heightBytes.size() != expectedHeightBytes || weightBytes.size() != expectedWeightBytes) {
    throw businessError(
        "DIM_MISMATCH",
        "Data payload size does not match declared dimensions");
  }

  HeightWeightData data;
  data.width = width;
  data.height = height;
  data.heights.resize(static_cast<size_t>(heightCount));
  data.weights.resize(static_cast<size_t>(heightCount) * 3u);
  for (size_t i = 0; i < data.heights.size(); ++i) {
    data.heights[i] = bitsFloat(getLittleU32(heightBytes.data() + 16u + i * 4u));
    if (!std::isfinite(data.heights[i])) {
      throw businessError("DATA_INVALID", "Heightfield contains a non-finite height");
    }
  }
  std::copy(weightBytes.begin() + 16, weightBytes.end(), data.weights.begin());

  uint64_t hash = 0xcbf29ce484222325ULL;
  for (float heightValue : data.heights) {
    hash ^= static_cast<uint64_t>(floatBits(heightValue));
    hash *= 0x100000001b3ULL;
  }
  for (int v = 0; v < data.height; ++v) {
    for (int u = 0; u < data.width; ++u) {
      const Vec3 weight = data.weightsAt(u, v);
      if (weight.x + weight.y + weight.z <= 1.0e-6f) {
        if (data.firstZeroU < 0) {
          data.firstZeroU = u;
          data.firstZeroV = v;
        }
        ++data.zeroWeightTexels;
      }
    }
  }
  for (uint8_t weight : data.weights) {
    hash ^= static_cast<uint64_t>(weight);
    hash *= 0x100000001b3ULL;
  }
  data.contentHash = hashMix(hash);
  return data;
}

HeightWeightData ensureHeightWeightData(const std::filesystem::path& directory,
                                        const DataConfig& config,
                                        bool allowGeneration,
                                        bool* generated) {
  if (generated != nullptr) {
    *generated = false;
  }
  const DataPaths paths = dataPathsForDirectory(directory);
  const bool heightExists = std::filesystem::exists(paths.heightPath);
  const bool weightExists = std::filesystem::exists(paths.weightPath);
  if (heightExists && weightExists) {
    return loadHeightWeightData(directory, config.width, config.height);
  }
  if (heightExists != weightExists) {
    throw businessError(
        "DATA_SET_INCOMPLETE",
        "Heightfield and material weight map must either both exist or both be absent");
  }
  if (!allowGeneration) {
    throw businessError("DATA_MISSING", "No generated input data is present");
  }
  HeightWeightData data = generateHeightWeightData(config);
  writeHeightWeightData(data, directory);
  if (generated != nullptr) {
    *generated = true;
  }
  return data;
}

std::string hashToHex(uint64_t value) {
  std::ostringstream stream;
  stream << std::hex << value;
  return stream.str();
}

}  // namespace terrain
