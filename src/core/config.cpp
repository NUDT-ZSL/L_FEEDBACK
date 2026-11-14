#include "lf/config.h"

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>

namespace lf {

namespace {

using json = nlohmann::json;

[[noreturn]] void reject(const std::string& context, const std::string& message) {
  throw JsonValidationError(context + ": " + message);
}

const json& require_object(const json& value, const char* context) {
  if (!value.is_object()) {
    reject(std::string(context), "expected JSON object");
  }
  return value;
}

const json& require_array(const json& parent, const char* key, const char* context) {
  if (!parent.contains(key)) {
    reject(std::string(context), std::string("missing field '") + key + "'");
  }
  if (!parent.at(key).is_array()) {
    reject(std::string(context) + "." + key, "expected array");
  }
  return parent.at(key);
}

const json& require_field(const json& parent, const char* key, const char* context) {
  if (!parent.contains(key)) {
    reject(std::string(context), std::string("missing field '") + key + "'");
  }
  return parent.at(key);
}

std::string require_string(const json& parent, const char* key, const std::string& context) {
  const json& value = require_field(parent, key, context.c_str());
  if (!value.is_string()) {
    reject(context + "." + key, "expected string");
  }
  const std::string result = value.get<std::string>();
  if (result.empty()) {
    reject(context + "." + key, "must not be empty");
  }
  return result;
}

float require_number(const json& parent, const char* key, const std::string& context) {
  const json& value = require_field(parent, key, context.c_str());
  if (!value.is_number()) {
    reject(context + "." + key, "expected number");
  }
  const double number = value.get<double>();
  if (!std::isfinite(number)) {
    reject(context + "." + key, "must be finite");
  }
  return static_cast<float>(number);
}

float require_number_in_range(const json& parent, const char* key, const std::string& context,
                              float minimum, float maximum) {
  const float value = require_number(parent, key, context);
  if (value < minimum || value > maximum) {
    std::ostringstream message;
    message << "must be in [" << minimum << ", " << maximum << "], got " << value;
    reject(context + "." + key, message.str());
  }
  return value;
}

glm::vec3 parse_vec3(const json& value, const std::string& context, float minimum,
                     float maximum) {
  if (!value.is_array() || value.size() != 3U) {
    reject(context, "expected [x, y, z] array");
  }
  glm::vec3 result{0.0F};
  for (std::size_t index = 0; index < 3U; ++index) {
    if (!value.at(index).is_number()) {
      reject(context, "vector components must be numbers");
    }
    const double component = value.at(index).get<double>();
    if (!std::isfinite(component)) {
      reject(context, "vector components must be finite");
    }
    if (component < static_cast<double>(minimum) ||
        component > static_cast<double>(maximum)) {
      reject(context, "vector component outside allowed range");
    }
    result[static_cast<glm::length_t>(index)] = static_cast<float>(component);
  }
  return result;
}

glm::vec3 require_vec3(const json& parent, const char* key, const std::string& context,
                       float minimum, float maximum) {
  const json& value = require_field(parent, key, context.c_str());
  return parse_vec3(value, context + "." + key, minimum, maximum);
}

bool optional_bool(const json& parent, const char* key, bool fallback,
                   const std::string& context) {
  if (!parent.contains(key)) {
    return fallback;
  }
  if (!parent.at(key).is_boolean()) {
    reject(context + "." + key, "expected boolean");
  }
  return parent.at(key).get<bool>();
}

void validate_texture_reference(const std::filesystem::path& assetRoot,
                                const std::string& textureFile,
                                const std::string& context) {
  const std::filesystem::path relativePath(textureFile);
  if (relativePath.is_absolute()) {
    reject(context, "texture path must be relative to the asset root");
  }
  const std::filesystem::path normalized = relativePath.lexically_normal();
  if (normalized.empty() || normalized.begin()->string() == "..") {
    reject(context, "texture path escapes the asset root");
  }
  const std::filesystem::path resolved = assetRoot / normalized;
  std::error_code ec;
  if (!std::filesystem::is_regular_file(resolved, ec) || ec) {
    reject(context, "texture reference does not exist: " + resolved.string());
  }
}

}  // namespace

const char* brdf_type_name(BrdfType type) noexcept {
  switch (type) {
    case BrdfType::Diffuse:
      return "diffuse";
    case BrdfType::GgxSpecular:
      return "ggx_specular";
    case BrdfType::ClothSheen:
      return "cloth_sheen";
  }
  return "unknown";
}

const char* light_type_name(LightType type) noexcept {
  switch (type) {
    case LightType::Point:
      return "point";
    case LightType::Directional:
      return "directional";
    case LightType::Spot:
      return "spot";
  }
  return "unknown";
}

BrdfType parse_brdf_type(const std::string& value) {
  if (value == "diffuse") {
    return BrdfType::Diffuse;
  }
  if (value == "ggx_specular") {
    return BrdfType::GgxSpecular;
  }
  if (value == "cloth_sheen") {
    return BrdfType::ClothSheen;
  }
  throw JsonValidationError("unsupported BRDF type '" + value + "'");
}

LightType parse_light_type(const std::string& value) {
  if (value == "point") {
    return LightType::Point;
  }
  if (value == "directional") {
    return LightType::Directional;
  }
  if (value == "spot") {
    return LightType::Spot;
  }
  throw JsonValidationError("unsupported light type '" + value + "'");
}

Light parse_light_json(const json& value, const std::string& context) {
  const json& object = require_object(value, context.c_str());
  Light light;
  light.id = require_string(object, "id", context);
  if (light.id.size() > 64U) {
    reject(context + ".id", "must contain at most 64 characters");
  }
  if (!std::all_of(light.id.begin(), light.id.end(), [](char character) {
        return (character >= 'a' && character <= 'z') ||
               (character >= 'A' && character <= 'Z') ||
               (character >= '0' && character <= '9') || character == '_' ||
               character == '-' || character == '.';
      })) {
    reject(context + ".id", "contains unsupported characters");
  }

  light.type = parse_light_type(require_string(object, "type", context));
  light.enabled = optional_bool(object, "enabled", true, context);
  light.color = require_vec3(object, "color", context, 0.0F, 8.0F);
  light.intensity = require_number_in_range(object, "intensity", context, 0.0001F, 10000.0F);

  if (light.type == LightType::Point || light.type == LightType::Spot) {
    light.position = require_vec3(object, "position", context, -1000.0F, 1000.0F);
    light.range = require_number_in_range(object, "range", context, 0.001F, 1000.0F);
    light.constantAttenuation =
        require_number_in_range(object, "constantAttenuation", context, 0.0001F, 1000.0F);
    light.linearAttenuation =
        require_number_in_range(object, "linearAttenuation", context, 0.0F, 1000.0F);
    light.quadraticAttenuation =
        require_number_in_range(object, "quadraticAttenuation", context, 0.0F, 1000.0F);
  }

  if (light.type == LightType::Directional || light.type == LightType::Spot) {
    light.direction = require_vec3(object, "direction", context, -1.0F, 1.0F);
    const float length = glm::length(light.direction);
    if (length < 1.0e-5F) {
      reject(context + ".direction", "must be a non-zero vector");
    }
  }

  if (light.type == LightType::Spot) {
    light.innerConeDegrees =
        require_number_in_range(object, "innerConeDegrees", context, 0.1F, 89.0F);
    light.outerConeDegrees =
        require_number_in_range(object, "outerConeDegrees", context, 0.2F, 89.5F);
    if (light.outerConeDegrees <= light.innerConeDegrees) {
      reject(context + ".outerConeDegrees", "must be greater than innerConeDegrees");
    }
  }

  validate_light(light, context);
  return light;
}

nlohmann::json light_to_json(const Light& light) {
  json value = {
      {"id", light.id},
      {"type", light_type_name(light.type)},
      {"enabled", light.enabled},
      {"color", {light.color.r, light.color.g, light.color.b}},
      {"intensity", light.intensity},
  };
  if (light.type == LightType::Point || light.type == LightType::Spot) {
    value["position"] = {light.position.r, light.position.g, light.position.b};
    value["range"] = light.range;
    value["constantAttenuation"] = light.constantAttenuation;
    value["linearAttenuation"] = light.linearAttenuation;
    value["quadraticAttenuation"] = light.quadraticAttenuation;
  }
  if (light.type == LightType::Directional || light.type == LightType::Spot) {
    value["direction"] = {light.direction.r, light.direction.g, light.direction.b};
  }
  if (light.type == LightType::Spot) {
    value["innerConeDegrees"] = light.innerConeDegrees;
    value["outerConeDegrees"] = light.outerConeDegrees;
  }
  return value;
}

void validate_light(const Light& light, const std::string& context) {
  auto finite = [](const glm::vec3& value) {
    return std::isfinite(value.r) && std::isfinite(value.g) && std::isfinite(value.b);
  };
  if (light.id.empty() || light.id.size() > 64U) {
    reject(context, "invalid light id length");
  }
  if (!finite(light.color) || !finite(light.position) || !finite(light.direction)) {
    reject(context, "light vectors must be finite");
  }
  if (!std::isfinite(light.intensity) || light.intensity <= 0.0F ||
      light.intensity > 10000.0F) {
    reject(context, "intensity outside (0, 10000]");
  }
  if (light.type == LightType::Point || light.type == LightType::Spot) {
    if (!std::isfinite(light.range) || light.range <= 0.0F || light.range > 1000.0F) {
      reject(context, "range outside (0, 1000]");
    }
    if (!std::isfinite(light.constantAttenuation) || light.constantAttenuation <= 0.0F ||
        light.constantAttenuation > 1000.0F || !std::isfinite(light.linearAttenuation) ||
        light.linearAttenuation < 0.0F || light.linearAttenuation > 1000.0F ||
        !std::isfinite(light.quadraticAttenuation) ||
        light.quadraticAttenuation < 0.0F || light.quadraticAttenuation > 1000.0F) {
      reject(context, "invalid attenuation parameters");
    }
  }
  if (light.type == LightType::Directional || light.type == LightType::Spot) {
    if (glm::length(light.direction) < 1.0e-5F) {
      reject(context, "direction must be non-zero");
    }
  }
  if (light.type == LightType::Spot) {
    if (light.innerConeDegrees <= 0.0F || light.outerConeDegrees <= light.innerConeDegrees ||
        light.outerConeDegrees > 89.5F) {
      reject(context, "invalid spot cone angles");
    }
  }
}

glm::vec3 material_layer_weights(const Material& material) {
  glm::vec3 weights{0.0F};
  for (std::size_t index = 0; index < std::min<std::size_t>(3U, material.layers.size());
       ++index) {
    weights[static_cast<glm::length_t>(index)] = material.layers[index].weight;
  }
  return weights;
}

void normalize_material_weights(glm::vec3& weights) {
  weights = glm::max(weights, glm::vec3(0.0F));
  const float total = weights.x + weights.y + weights.z;
  if (total > 1.0e-6F) {
    weights /= total;
  } else {
    weights = glm::vec3(1.0F, 0.0F, 0.0F);
  }
}

ConfigLoader::ConfigLoader(std::filesystem::path assetRoot) : assetRoot_(std::move(assetRoot)) {}

LightingConfig ConfigLoader::load(const std::filesystem::path& configFile) const {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(configFile, ec) || ec) {
    throw AppError(ErrorCode::E_RESOURCE_MISSING,
                   "configuration file is missing: " + configFile.string());
  }

  std::ifstream input(configFile, std::ios::binary);
  if (!input) {
    throw AppError(ErrorCode::E_CONFIG_INVALID,
                   "cannot open configuration file: " + configFile.string());
  }

  json document;
  try {
    input >> document;
  } catch (const std::exception& error) {
    throw AppError(ErrorCode::E_CONFIG_INVALID,
                   "JSON parse error in " + configFile.string() + ": " + error.what());
  }

  try {
    const json& root = require_object(document, "$");
    const json& versionValue = require_field(root, "formatVersion", "$");
    if (!versionValue.is_number_integer()) {
      reject("$.formatVersion", "expected an integer");
    }
    const int formatVersion = versionValue.get<int>();
    if (formatVersion != 1) {
      reject("$.formatVersion", "only configuration version 1 is supported");
    }

    LightingConfig config;
    config.formatVersion = formatVersion;
    const json& materialValue =
        require_object(require_field(root, "material", "$"), "$.material");
    if (materialValue.contains("name")) {
      if (!materialValue.at("name").is_string()) {
        reject("$.material.name", "expected string");
      }
      config.material.name = materialValue.at("name").get<std::string>();
    }

    const json& layers = require_array(materialValue, "layers", "$.material");
    if (layers.size() != 3U) {
      reject("$.material.layers", "exactly three BRDF layers are required");
    }
    constexpr std::array<BrdfType, 3> kExpectedOrder = {
        BrdfType::Diffuse, BrdfType::GgxSpecular, BrdfType::ClothSheen};
    for (std::size_t index = 0; index < layers.size(); ++index) {
      const std::string context = "$.material.layers[" + std::to_string(index) + "]";
      const json& layerValue = require_object(layers.at(index), context.c_str());
      MaterialLayer layer;
      layer.brdf = parse_brdf_type(require_string(layerValue, "type", context));
      if (layer.brdf != kExpectedOrder[index]) {
        reject(context + ".type", "layers must use fixed order: diffuse, ggx_specular, cloth_sheen");
      }
      layer.weight = require_number_in_range(layerValue, "weight", context, 0.0F, 1.0F);
      layer.baseColorTexture = require_string(layerValue, "baseColorTexture", context);
      layer.normalTexture = require_string(layerValue, "normalTexture", context);
      layer.roughnessTexture = require_string(layerValue, "roughnessTexture", context);
      layer.metallicTexture = require_string(layerValue, "metallicTexture", context);
      config.material.layers.push_back(std::move(layer));
    }

    const float weightSum = config.material.layers[0].weight +
                            config.material.layers[1].weight +
                            config.material.layers[2].weight;
    if (!std::isfinite(weightSum) || weightSum <= 1.0e-4F) {
      reject("$.material.layers", "material weights must have a positive sum");
    }

    const json& lights = require_array(root, "lights", "$");
    if (lights.empty() || lights.size() > 32U) {
      reject("$.lights", "light count must be in [1, 32]");
    }
    std::set<std::string> lightIds;
    for (std::size_t index = 0; index < lights.size(); ++index) {
      const std::string context = "$.lights[" + std::to_string(index) + "]";
      Light light = parse_light_json(lights.at(index), context);
      if (!lightIds.insert(light.id).second) {
        reject(context + ".id", "duplicate light id '" + light.id + "'");
      }
      config.lights.push_back(std::move(light));
    }

    validate_material(config.material);
    return config;
  } catch (const JsonValidationError& error) {
    throw AppError(ErrorCode::E_CONFIG_INVALID,
                   "configuration validation failed: " + std::string(error.what()));
  } catch (const nlohmann::json::exception& error) {
    throw AppError(ErrorCode::E_CONFIG_INVALID,
                   "configuration schema error: " + std::string(error.what()));
  }
}

void ConfigLoader::validate_material(const Material& material) const {
  if (material.layers.size() != 3U) {
    reject("$.material.layers", "exactly three layers are required");
  }
  for (std::size_t index = 0; index < material.layers.size(); ++index) {
    const MaterialLayer& layer = material.layers[index];
    const std::string context = "$.material.layers[" + std::to_string(index) + "]";
    validate_texture_reference(assetRoot_, layer.baseColorTexture, context + ".baseColorTexture");
    validate_texture_reference(assetRoot_, layer.normalTexture, context + ".normalTexture");
    validate_texture_reference(assetRoot_, layer.roughnessTexture,
                               context + ".roughnessTexture");
    validate_texture_reference(assetRoot_, layer.metallicTexture,
                               context + ".metallicTexture");
  }
}

}  // namespace lf

