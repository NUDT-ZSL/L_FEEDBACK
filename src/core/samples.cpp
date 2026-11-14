#include "lf/samples.h"

#include "lf/error_code.h"
#include "lf/log.h"

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <vector>

namespace lf {

namespace {

constexpr std::array<const char*, 3> kLayerPrefixes = {"diffuse", "ggx", "cloth"};
constexpr std::array<const char*, 4> kTextureKinds = {
    "basecolor", "normal", "roughness", "metallic"};

class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

  std::uint64_t next() {
    std::uint64_t value = (state_ += 0x9E3779B97F4A7C15ULL);
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
  }

  float unit_float() {
    return static_cast<float>(next() >> 40U) * (1.0F / 16777216.0F);
  }

 private:
  std::uint64_t state_;
};

float hash_noise(int x, int y, std::uint64_t seed) {
  std::uint64_t value = static_cast<std::uint64_t>(static_cast<std::uint32_t>(x));
  value ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(y)) << 32U;
  value ^= seed;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  value ^= value >> 31U;
  return static_cast<float>(value >> 40U) * (1.0F / 16777216.0F);
}

float smooth_value_noise(float x, float y, std::uint64_t seed) {
  const int floorX = static_cast<int>(std::floor(x));
  const int floorY = static_cast<int>(std::floor(y));
  const float fx = x - static_cast<float>(floorX);
  const float fy = y - static_cast<float>(floorY);
  const float sx = fx * fx * (3.0F - 2.0F * fx);
  const float sy = fy * fy * (3.0F - 2.0F * fy);
  const float a = hash_noise(floorX, floorY, seed);
  const float b = hash_noise(floorX + 1, floorY, seed);
  const float c = hash_noise(floorX, floorY + 1, seed);
  const float d = hash_noise(floorX + 1, floorY + 1, seed);
  return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy;
}

float height_field(int x, int y, std::uint64_t seed) {
  const float xf = static_cast<float>(x) / static_cast<float>(kSampleTextureSize);
  const float yf = static_cast<float>(y) / static_cast<float>(kSampleTextureSize);
  float value = 0.0F;
  float amplitude = 0.55F;
  float frequency = 7.0F;
  for (int octave = 0; octave < 4; ++octave) {
    value += amplitude * smooth_value_noise(xf * frequency, yf * frequency,
                                             seed + static_cast<std::uint64_t>(octave) * 131U);
    amplitude *= 0.5F;
    frequency *= 2.03F;
  }
  return value;
}

std::uint8_t to_byte(float value) {
  const float clamped = glm::clamp(value, 0.0F, 1.0F);
  return static_cast<std::uint8_t>(std::lround(clamped * 255.0F));
}

struct Rgba8 {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

std::vector<Rgba8> generate_texture(std::size_t layerIndex, std::size_t textureKind) {
  const std::uint64_t seed =
      kSampleSeed ^ (static_cast<std::uint64_t>(layerIndex + 1U) * 0x9E3779B97F4A7C15ULL) ^
      (static_cast<std::uint64_t>(textureKind + 1U) * 0xD1B54A32D192ED03ULL);
  std::vector<Rgba8> pixels(static_cast<std::size_t>(kSampleTextureSize) * kSampleTextureSize);

  const std::array<glm::vec3, 3> diffuseTints = {
      glm::vec3(0.72F, 0.46F, 0.36F), glm::vec3(0.53F, 0.60F, 0.68F),
      glm::vec3(0.67F, 0.31F, 0.22F)};
  const float normalStrength = 2.4F + static_cast<float>(layerIndex) * 0.6F;

  auto heights_at = [&](int x, int y) {
    return height_field((x + kSampleTextureSize) % kSampleTextureSize,
                        (y + kSampleTextureSize) % kSampleTextureSize, seed);
  };

  for (int y = 0; y < kSampleTextureSize; ++y) {
    for (int x = 0; x < kSampleTextureSize; ++x) {
      const std::size_t index =
          static_cast<std::size_t>(y * kSampleTextureSize + x);
      const float u = static_cast<float>(x) / static_cast<float>(kSampleTextureSize);
      const float v = static_cast<float>(y) / static_cast<float>(kSampleTextureSize);
      const float weave =
          0.5F + 0.5F * std::sin((u * 17.0F + v * 11.0F) * 6.2831853F);
      const float grain = hash_noise(x, y, seed ^ 0xA5A5A5A5ULL);

      Rgba8 color{};
      if (textureKind == 0U) {
        const glm::vec3 tint = diffuseTints[layerIndex];
        const float detail = 0.79F + 0.10F * weave + 0.11F * grain;
        color.r = to_byte(tint.r * detail);
        color.g = to_byte(tint.g * detail);
        color.b = to_byte(tint.b * detail);
        color.a = 255;
      } else if (textureKind == 1U) {
        const float dx = (heights_at(x + 1, y) - heights_at(x - 1, y)) * normalStrength;
        const float dy = (heights_at(x, y + 1) - heights_at(x, y - 1)) * normalStrength;
        const glm::vec3 normal = glm::normalize(glm::vec3(-dx, -dy, 1.0F));
        color.r = to_byte(normal.x * 0.5F + 0.5F);
        color.g = to_byte(normal.y * 0.5F + 0.5F);
        color.b = to_byte(normal.z * 0.5F + 0.5F);
        color.a = 255;
      } else if (textureKind == 2U) {
        const float layerBase = 0.34F + static_cast<float>(layerIndex) * 0.18F;
        const float roughness =
            glm::clamp(layerBase + (weave - 0.5F) * 0.22F + (grain - 0.5F) * 0.08F, 0.05F,
                       0.96F);
        const std::uint8_t byte = to_byte(roughness);
        color = {byte, byte, byte, 255};
      } else {
        const float metal = layerIndex == 0U ? 0.03F : (layerIndex == 1U ? 0.82F : 0.08F);
        const std::uint8_t byte = to_byte(metal + (grain - 0.5F) * 0.03F);
        color = {byte, byte, byte, 255};
      }
      pixels[index] = color;
    }
  }
  return pixels;
}

void write_tga(const std::filesystem::path& path, const std::vector<Rgba8>& pixels) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw AppError(ErrorCode::E_TEXTURE_LOAD,
                   "cannot create deterministic texture: " + path.string());
  }

  const std::array<std::uint8_t, 18> header = {
      0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
      0, 0,
      static_cast<std::uint8_t>(kSampleTextureSize & 0xFF),
      static_cast<std::uint8_t>((kSampleTextureSize >> 8) & 0xFF),
      static_cast<std::uint8_t>(kSampleTextureSize & 0xFF),
      static_cast<std::uint8_t>((kSampleTextureSize >> 8) & 0xFF),
      32, 0x28};
  output.write(reinterpret_cast<const char*>(header.data()),
               static_cast<std::streamsize>(header.size()));
  for (const Rgba8& pixel : pixels) {
    const std::array<std::uint8_t, 4> bgra = {pixel.b, pixel.g, pixel.r, pixel.a};
    output.write(reinterpret_cast<const char*>(bgra.data()), 4);
  }
  if (!output) {
    throw AppError(ErrorCode::E_TEXTURE_LOAD,
                   "failed writing deterministic texture: " + path.string());
  }
}

void write_config(const std::filesystem::path& configFile, const LightingConfig& config) {
  nlohmann::json root;
  root["formatVersion"] = config.formatVersion;
  root["material"]["name"] = config.material.name;
  root["material"]["layers"] = nlohmann::json::array();
  for (const MaterialLayer& layer : config.material.layers) {
    root["material"]["layers"].push_back({
        {"type", brdf_type_name(layer.brdf)},
        {"weight", layer.weight},
        {"baseColorTexture", layer.baseColorTexture},
        {"normalTexture", layer.normalTexture},
        {"roughnessTexture", layer.roughnessTexture},
        {"metallicTexture", layer.metallicTexture},
    });
  }
  root["lights"] = nlohmann::json::array();
  for (const Light& light : config.lights) {
    root["lights"].push_back(light_to_json(light));
  }

  std::error_code ec;
  std::filesystem::create_directories(configFile.parent_path(), ec);
  std::ofstream output(configFile, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw AppError(ErrorCode::E_CONFIG_INVALID,
                   "cannot create default configuration: " + configFile.string());
  }
  output << root.dump(2) << '\n';
  if (!output) {
    throw AppError(ErrorCode::E_CONFIG_INVALID,
                   "failed writing default configuration: " + configFile.string());
  }
}

std::string texture_relative_path(std::size_t layer, std::size_t kind) {
  std::ostringstream path;
  path << "textures/" << kLayerPrefixes[layer] << '_' << kTextureKinds[kind] << ".tga";
  return path.str();
}

}  // namespace

std::vector<std::string> expected_texture_files() {
  std::vector<std::string> result;
  result.reserve(12U);
  for (std::size_t layer = 0; layer < kLayerPrefixes.size(); ++layer) {
    for (std::size_t kind = 0; kind < kTextureKinds.size(); ++kind) {
      result.push_back(texture_relative_path(layer, kind));
    }
  }
  return result;
}

LightingConfig make_default_config() {
  LightingConfig config;
  config.formatVersion = 1;
  config.material.name = "deterministic_layered_character";
  const std::array<const char*, 3> types = {"diffuse", "ggx_specular", "cloth_sheen"};
  const std::array<float, 3> weights = {0.55F, 0.28F, 0.17F};
  for (std::size_t layer = 0; layer < 3U; ++layer) {
    MaterialLayer materialLayer;
    materialLayer.brdf = parse_brdf_type(types[layer]);
    materialLayer.weight = weights[layer];
    materialLayer.baseColorTexture = texture_relative_path(layer, 0);
    materialLayer.normalTexture = texture_relative_path(layer, 1);
    materialLayer.roughnessTexture = texture_relative_path(layer, 2);
    materialLayer.metallicTexture = texture_relative_path(layer, 3);
    config.material.layers.push_back(std::move(materialLayer));
  }

  Light key;
  key.id = "warm_key";
  key.type = LightType::Point;
  key.position = glm::vec3(2.6F, 3.7F, 2.9F);
  key.color = glm::vec3(1.0F, 0.73F, 0.46F);
  key.intensity = 44.0F;
  key.range = 12.0F;
  key.constantAttenuation = 1.0F;
  key.linearAttenuation = 0.09F;
  key.quadraticAttenuation = 0.032F;
  config.lights.push_back(key);

  Light fill;
  fill.id = "cool_fill";
  fill.type = LightType::Directional;
  fill.direction = glm::normalize(glm::vec3(-0.45F, -1.0F, -0.35F));
  fill.color = glm::vec3(0.42F, 0.62F, 1.0F);
  fill.intensity = 1.65F;
  config.lights.push_back(fill);

  Light rim;
  rim.id = "cloth_rim";
  rim.type = LightType::Spot;
  rim.position = glm::vec3(-2.3F, 2.9F, -2.6F);
  rim.direction = glm::normalize(glm::vec3(0.41F, -0.62F, 0.58F));
  rim.color = glm::vec3(1.0F, 0.43F, 0.71F);
  rim.intensity = 62.0F;
  rim.range = 10.0F;
  rim.constantAttenuation = 1.0F;
  rim.linearAttenuation = 0.08F;
  rim.quadraticAttenuation = 0.028F;
  rim.innerConeDegrees = 18.0F;
  rim.outerConeDegrees = 31.0F;
  config.lights.push_back(rim);

  return config;
}

SampleGenerationResult generate_samples(const std::filesystem::path& assetRoot,
                                        const std::filesystem::path& configFile,
                                        bool overwrite) {
  SampleGenerationResult result;
  std::error_code ec;
  std::filesystem::create_directories(assetRoot / "textures", ec);
  if (ec) {
    throw AppError(ErrorCode::E_RESOURCE_MISSING,
                   "cannot create asset directory: " + (assetRoot / "textures").string());
  }

  std::vector<std::filesystem::path> textures;
  textures.reserve(12U);
  for (std::size_t layer = 0; layer < kLayerPrefixes.size(); ++layer) {
    for (std::size_t kind = 0; kind < kTextureKinds.size(); ++kind) {
      const std::filesystem::path path = assetRoot / texture_relative_path(layer, kind);
      textures.push_back(path);
      if (overwrite || !std::filesystem::is_regular_file(path, ec) || ec) {
        write_tga(path, generate_texture(layer, kind));
        result.texturesGenerated = true;
      }
    }
  }
  result.textureFiles = textures;

  if (overwrite || !std::filesystem::is_regular_file(configFile, ec) || ec) {
    write_config(configFile, make_default_config());
    result.configGenerated = true;
  }
  return result;
}

SampleGenerationResult ensure_samples(const std::filesystem::path& assetRoot,
                                      const std::filesystem::path& configFile) {
  SampleGenerationResult result = generate_samples(assetRoot, configFile, false);
  if (result.texturesGenerated || result.configGenerated) {
    log_info("deterministic samples generated (textures=" +
             std::string(result.texturesGenerated ? "yes" : "no") +
             ", config=" + (result.configGenerated ? "yes" : "no") + ")");
  }
  return result;
}

}  // namespace lf

