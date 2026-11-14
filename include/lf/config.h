#pragma once

#include "lf/error_code.h"

#include <glm/vec3.hpp>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace lf {

enum class BrdfType { Diffuse, GgxSpecular, ClothSheen };
enum class LightType { Point, Directional, Spot };

struct MaterialLayer {
  BrdfType brdf = BrdfType::Diffuse;
  float weight = 0.0F;
  std::string baseColorTexture;
  std::string normalTexture;
  std::string roughnessTexture;
  std::string metallicTexture;
};

struct Material {
  std::string name;
  std::vector<MaterialLayer> layers;
};

struct Light {
  std::string id;
  LightType type = LightType::Point;
  bool enabled = true;
  glm::vec3 position{0.0F};
  glm::vec3 direction{0.0F, -1.0F, 0.0F};
  glm::vec3 color{1.0F};
  float intensity = 1.0F;
  float range = 10.0F;
  float innerConeDegrees = 0.0F;
  float outerConeDegrees = 0.0F;
  float constantAttenuation = 1.0F;
  float linearAttenuation = 0.0F;
  float quadraticAttenuation = 0.0F;
};

struct LightingConfig {
  int formatVersion = 1;
  Material material;
  std::vector<Light> lights;
};

struct LightingState {
  std::vector<Light> lights;
  glm::vec3 materialWeights{0.55F, 0.30F, 0.15F};
  float animationTime = 0.0F;
};

class JsonValidationError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

const char* brdf_type_name(BrdfType type) noexcept;
const char* light_type_name(LightType type) noexcept;
BrdfType parse_brdf_type(const std::string& value);
LightType parse_light_type(const std::string& value);

Light parse_light_json(const nlohmann::json& value, const std::string& context);
nlohmann::json light_to_json(const Light& light);
void validate_light(const Light& light, const std::string& context);

glm::vec3 material_layer_weights(const Material& material);
void normalize_material_weights(glm::vec3& weights);

class ConfigLoader {
 public:
  explicit ConfigLoader(std::filesystem::path assetRoot);

  [[nodiscard]] LightingConfig load(const std::filesystem::path& configFile) const;

 private:
  void validate_material(const Material& material) const;

  std::filesystem::path assetRoot_;
};

}  // namespace lf

