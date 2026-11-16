#include "scene.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

#include "json.hpp"

namespace orv {
namespace {

const JsonValue& requireMember(const JsonValue& object, const std::string& key,
                               const std::string& path) {
  if (!object.isObject()) {
    throw sceneError("SCENE_ERR_INVALID_TYPE", "expected object", path);
  }
  const JsonValue* value = object.find(key);
  if (value == nullptr) {
    throw sceneError("SCENE_ERR_MISSING_FIELD", "missing required field", path + "." + key);
  }
  return *value;
}

const JsonValue::Array& requireArray(const JsonValue& object, const std::string& key,
                                     const std::string& path) {
  const JsonValue& value = requireMember(object, key, path);
  if (!value.isArray()) {
    throw sceneError("SCENE_ERR_INVALID_TYPE", "expected array", path + "." + key);
  }
  return value.asArray();
}

double requireNumber(const JsonValue& object, const std::string& key,
                     const std::string& path) {
  const JsonValue& value = requireMember(object, key, path);
  if (!value.isNumber()) {
    throw sceneError("SCENE_ERR_INVALID_TYPE", "expected number", path + "." + key);
  }
  if (!std::isfinite(value.asNumber())) {
    throw sceneError("SCENE_ERR_INVALID_VALUE", "number must be finite", path + "." + key);
  }
  return value.asNumber();
}

std::string requireString(const JsonValue& object, const std::string& key,
                          const std::string& path) {
  const JsonValue& value = requireMember(object, key, path);
  if (!value.isString()) {
    throw sceneError("SCENE_ERR_INVALID_TYPE", "expected string", path + "." + key);
  }
  return value.asString();
}

bool optionalBool(const JsonValue& object, const std::string& key, bool defaultValue,
                  const std::string& path) {
  const JsonValue* value = object.find(key);
  if (value == nullptr) {
    return defaultValue;
  }
  if (!value->isBool()) {
    throw sceneError("SCENE_ERR_INVALID_TYPE", "expected boolean", path + "." + key);
  }
  return value->asBool();
}

float checkedFloat(double value, float minimum, float maximum, const std::string& path) {
  if (!std::isfinite(value) || value < minimum || value > maximum) {
    std::ostringstream message;
    message << "value must be in [" << minimum << ", " << maximum << "]";
    throw sceneError("SCENE_ERR_INVALID_VALUE", message.str(), path);
  }
  return static_cast<float>(value);
}

float checkedPositiveFloat(double value, const std::string& path) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw sceneError("SCENE_ERR_INVALID_VALUE", "value must be finite and positive", path);
  }
  return static_cast<float>(value);
}

Vec3 parseVec3(const JsonValue& value, const std::string& path) {
  if (!value.isArray() || value.asArray().size() != 3) {
    throw sceneError("SCENE_ERR_INVALID_TYPE", "expected array of 3 numbers", path);
  }
  Vec3 result;
  float* components[3] = {&result.x, &result.y, &result.z};
  for (std::size_t i = 0; i < 3; ++i) {
    const JsonValue& component = value.asArray()[i];
    if (!component.isNumber()) {
      throw sceneError("SCENE_ERR_INVALID_TYPE", "expected number",
                       path + "[" + std::to_string(i) + "]");
    }
    if (!std::isfinite(component.asNumber())) {
      throw sceneError("SCENE_ERR_INVALID_VALUE", "number must be finite",
                       path + "[" + std::to_string(i) + "]");
    }
    *components[i] = static_cast<float>(component.asNumber());
  }
  return result;
}

Vec2 parseVec2(const JsonValue& value, const std::string& path) {
  if (!value.isArray() || value.asArray().size() != 2) {
    throw sceneError("SCENE_ERR_INVALID_TYPE", "expected array of 2 numbers", path);
  }
  Vec2 result;
  float* components[2] = {&result.x, &result.y};
  for (std::size_t i = 0; i < 2; ++i) {
    const JsonValue& component = value.asArray()[i];
    if (!component.isNumber()) {
      throw sceneError("SCENE_ERR_INVALID_TYPE", "expected number",
                       path + "[" + std::to_string(i) + "]");
    }
    if (!std::isfinite(component.asNumber())) {
      throw sceneError("SCENE_ERR_INVALID_VALUE", "number must be finite",
                       path + "[" + std::to_string(i) + "]");
    }
    *components[i] = static_cast<float>(component.asNumber());
  }
  return result;
}

Mat4 parseMat4(const JsonValue& value, const std::string& path) {
  if (!value.isArray() || value.asArray().size() != 16) {
    throw sceneError("SCENE_ERR_INVALID_TYPE", "expected array of 16 numbers", path);
  }
  Mat4 result;
  for (std::size_t i = 0; i < 16; ++i) {
    const JsonValue& component = value.asArray()[i];
    if (!component.isNumber() || !std::isfinite(component.asNumber())) {
      throw sceneError("SCENE_ERR_INVALID_TYPE", "expected finite number",
                       path + "[" + std::to_string(i) + "]");
    }
    result.m[i] = static_cast<float>(component.asNumber());
  }
  return result;
}

Camera parseCamera(const JsonValue& root) {
  const JsonValue& camera = requireMember(root, "camera", "scene");
  Camera result;
  result.position = parseVec3(requireMember(camera, "position", "scene.camera"),
                              "scene.camera.position");
  result.target = parseVec3(requireMember(camera, "target", "scene.camera"),
                            "scene.camera.target");
  result.up = parseVec3(requireMember(camera, "up", "scene.camera"), "scene.camera.up");
  if (lengthSquared(result.up) <= 1.0e-10f) {
    throw sceneError("SCENE_ERR_INVALID_VALUE", "camera.up must not be zero",
                     "scene.camera.up");
  }
  if (lengthSquared(result.target - result.position) <= 1.0e-10f) {
    throw sceneError("SCENE_ERR_INVALID_VALUE",
                     "camera.target must differ from camera.position",
                     "scene.camera.target");
  }
  const std::string projection = requireString(camera, "projection", "scene.camera");
  if (projection != "perspective") {
    throw sceneError("SCENE_ERR_INVALID_ENUM",
                     "unsupported camera.projection; expected 'perspective'",
                     "scene.camera.projection");
  }
  result.fovYDegrees = checkedFloat(requireNumber(camera, "fov_y_deg", "scene.camera"),
                                    1.0, 179.0, "scene.camera.fov_y_deg");
  result.nearPlane = checkedPositiveFloat(requireNumber(camera, "near", "scene.camera"),
                                          "scene.camera.near");
  result.farPlane = checkedPositiveFloat(requireNumber(camera, "far", "scene.camera"),
                                         "scene.camera.far");
  if (result.farPlane <= result.nearPlane) {
    throw sceneError("SCENE_ERR_INVALID_VALUE", "camera.far must exceed camera.near",
                     "scene.camera.far");
  }
  return result;
}

Material parseMaterial(const JsonValue& mesh, const std::string& meshPath) {
  const JsonValue& material = requireMember(mesh, "material", meshPath);
  const std::string path = meshPath + ".material";
  Material result;
  result.baseColor = parseVec3(requireMember(material, "base_color", path),
                               path + ".base_color");
  result.baseColor.x = checkedFloat(result.baseColor.x, 0.0f, 1.0f, path + ".base_color[0]");
  result.baseColor.y = checkedFloat(result.baseColor.y, 0.0f, 1.0f, path + ".base_color[1]");
  result.baseColor.z = checkedFloat(result.baseColor.z, 0.0f, 1.0f, path + ".base_color[2]");
  result.roughness = checkedFloat(requireNumber(material, "roughness", path), 0.0, 1.0,
                                  path + ".roughness");
  result.metallic = checkedFloat(requireNumber(material, "metallic", path), 0.0, 1.0,
                                 path + ".metallic");
  result.anisotropyStrength =
      checkedFloat(requireNumber(material, "anisotropy_strength", path), 0.0, 1.0,
                   path + ".anisotropy_strength");
  const double rotation = requireNumber(material, "anisotropy_rotation_rad", path);
  if (!std::isfinite(rotation)) {
    throw sceneError("SCENE_ERR_INVALID_VALUE", "rotation must be finite",
                     path + ".anisotropy_rotation_rad");
  }
  result.anisotropyRotationRadians = static_cast<float>(rotation);
  return result;
}

std::vector<Vec3> parseVec3Array(const JsonValue& parent, const std::string& key,
                                 const std::string& path) {
  const JsonValue::Array& values = requireArray(parent, key, path);
  std::vector<Vec3> result;
  result.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    result.push_back(parseVec3(values[i], path + "." + key + "[" + std::to_string(i) + "]"));
  }
  return result;
}

std::vector<Vec2> parseVec2Array(const JsonValue& parent, const std::string& key,
                                 const std::string& path) {
  const JsonValue::Array& values = requireArray(parent, key, path);
  std::vector<Vec2> result;
  result.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    result.push_back(parseVec2(values[i], path + "." + key + "[" + std::to_string(i) + "]"));
  }
  return result;
}

Mesh parseMesh(const JsonValue& meshValue, std::size_t meshIndex) {
  const std::string meshPath = "scene.meshes[" + std::to_string(meshIndex) + "]";
  if (!meshValue.isObject()) {
    throw sceneError("SCENE_ERR_INVALID_TYPE", "mesh must be an object", meshPath);
  }

  Mesh mesh;
  mesh.name = requireString(meshValue, "name", meshPath);
  const JsonValue* model = meshValue.find("model");
  if (model == nullptr) {
    mesh.model = Mat4::identity();
  } else {
    mesh.model = parseMat4(*model, meshPath + ".model");
  }

  const JsonValue& vertices = requireMember(meshValue, "vertices", meshPath);
  const std::string verticesPath = meshPath + ".vertices";
  mesh.positions = parseVec3Array(vertices, "positions", verticesPath);
  mesh.normals = parseVec3Array(vertices, "normals", verticesPath);

  const JsonValue* tangentValue = vertices.find("tangents");
  if (tangentValue == nullptr) {
    throw sceneError("SCENE_ERR_INVALID_TANGENT",
                     "every vertex requires a non-zero tangent",
                     verticesPath + ".tangents");
  }
  mesh.tangents = parseVec3Array(vertices, "tangents", verticesPath);
  mesh.uvs = parseVec2Array(vertices, "uvs", verticesPath);

  if (mesh.positions.empty()) {
    throw sceneError("SCENE_ERR_INVALID_VALUE", "mesh must contain at least one vertex",
                     verticesPath + ".positions");
  }
  const std::size_t vertexCount = mesh.positions.size();
  if (mesh.normals.size() != vertexCount) {
    throw sceneError("SCENE_ERR_INVALID_VALUE",
                     "positions and normals must have the same length",
                     verticesPath + ".normals");
  }
  if (mesh.tangents.size() != vertexCount) {
    throw sceneError("SCENE_ERR_INVALID_TANGENT",
                     "positions and tangents must have the same length",
                     verticesPath + ".tangents");
  }
  if (mesh.uvs.size() != vertexCount) {
    throw sceneError("SCENE_ERR_INVALID_VALUE",
                     "positions and uvs must have the same length", verticesPath + ".uvs");
  }
  for (std::size_t i = 0; i < vertexCount; ++i) {
    if (!finite(mesh.positions[i])) {
      throw sceneError("SCENE_ERR_INVALID_VALUE", "position must be finite",
                       verticesPath + ".positions[" + std::to_string(i) + "]");
    }
    if (!finite(mesh.normals[i]) || lengthSquared(mesh.normals[i]) <= 1.0e-12f) {
      throw sceneError("SCENE_ERR_INVALID_VALUE", "normal must not be zero",
                       verticesPath + ".normals[" + std::to_string(i) + "]");
    }
    if (!finite(mesh.tangents[i]) || lengthSquared(mesh.tangents[i]) <= 1.0e-12f) {
      throw sceneError("SCENE_ERR_INVALID_TANGENT",
                       "tangent must be finite and have non-zero length",
                       verticesPath + ".tangents[" + std::to_string(i) + "]");
    }
    const Vec3 normalizedNormal = normalize(mesh.normals[i]);
    const Vec3 normalizedTangent = normalize(mesh.tangents[i]);
    if (lengthSquared(cross(normalizedNormal, normalizedTangent)) <= 1.0e-10f) {
      throw sceneError("SCENE_ERR_INVALID_TANGENT",
                       "tangent must not be parallel to its vertex normal",
                       verticesPath + ".tangents[" + std::to_string(i) + "]");
    }
  }

  const JsonValue::Array& indexValues = requireArray(meshValue, "indices", meshPath);
  mesh.indices.reserve(indexValues.size());
  for (std::size_t i = 0; i < indexValues.size(); ++i) {
    const JsonValue& index = indexValues[i];
    if (!index.isNumber() || index.asNumber() < 0.0 ||
        std::floor(index.asNumber()) != index.asNumber()) {
      throw sceneError("SCENE_ERR_INVALID_INDEX",
                       "indices must be non-negative integers",
                       meshPath + ".indices[" + std::to_string(i) + "]");
    }
    if (index.asNumber() >
        static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
      throw sceneError("SCENE_ERR_INVALID_INDEX",
                       "index is outside the vertex array",
                       meshPath + ".indices[" + std::to_string(i) + "]");
    }
    const auto parsed = static_cast<std::uint64_t>(index.asNumber());
    if (parsed >= vertexCount ||
        parsed > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
      throw sceneError("SCENE_ERR_INVALID_INDEX",
                       "index is outside the vertex array",
                       meshPath + ".indices[" + std::to_string(i) + "]");
    }
    mesh.indices.push_back(static_cast<std::uint32_t>(parsed));
  }
  if (mesh.indices.size() % 3 != 0) {
    throw sceneError("SCENE_ERR_INVALID_INDEX",
                     "triangle index count must be divisible by 3", meshPath + ".indices");
  }
  mesh.material = parseMaterial(meshValue, meshPath);
  return mesh;
}

DirectionalLight parseDirectionalLight(const JsonValue& value) {
  if (!value.isObject()) {
    throw sceneError("SCENE_ERR_INVALID_TYPE", "directional light must be an object",
                     "scene.lights.directional");
  }
  const std::string path = "scene.lights.directional";
  const std::string type = requireString(value, "type", path);
  if (type != "directional") {
    throw sceneError("SCENE_ERR_INVALID_ENUM",
                     "directional light type must be 'directional'", path + ".type");
  }
  DirectionalLight light;
  light.direction = parseVec3(requireMember(value, "direction", path), path + ".direction");
  if (lengthSquared(light.direction) <= 1.0e-10f) {
    throw sceneError("SCENE_ERR_INVALID_VALUE", "direction must not be zero",
                     path + ".direction");
  }
  light.direction = normalize(light.direction);
  light.color = parseVec3(requireMember(value, "color", path), path + ".color");
  light.color.x = checkedFloat(light.color.x, 0.0f, 1.0f, path + ".color[0]");
  light.color.y = checkedFloat(light.color.y, 0.0f, 1.0f, path + ".color[1]");
  light.color.z = checkedFloat(light.color.z, 0.0f, 1.0f, path + ".color[2]");
  light.intensity =
      checkedFloat(requireNumber(value, "intensity", path), 0.0, 100000.0,
                   path + ".intensity");
  optionalBool(value, "casts_shadow", true, path);
  return light;
}

PointLight parsePointLight(const JsonValue& value, std::size_t index) {
  const std::string path = "scene.lights.point_lights[" + std::to_string(index) + "]";
  if (!value.isObject()) {
    throw sceneError("SCENE_ERR_INVALID_TYPE", "point light must be an object", path);
  }
  const std::string type = requireString(value, "type", path);
  if (type != "point") {
    throw sceneError("SCENE_ERR_INVALID_ENUM", "point light type must be 'point'",
                     path + ".type");
  }
  PointLight light;
  light.position = parseVec3(requireMember(value, "position", path), path + ".position");
  light.color = parseVec3(requireMember(value, "color", path), path + ".color");
  light.color.x = checkedFloat(light.color.x, 0.0f, 1.0f, path + ".color[0]");
  light.color.y = checkedFloat(light.color.y, 0.0f, 1.0f, path + ".color[1]");
  light.color.z = checkedFloat(light.color.z, 0.0f, 1.0f, path + ".color[2]");
  light.intensity =
      checkedFloat(requireNumber(value, "intensity", path), 0.0, 100000.0,
                   path + ".intensity");
  light.range =
      checkedPositiveFloat(requireNumber(value, "range", path), path + ".range");
  optionalBool(value, "casts_shadow", true, path);
  return light;
}

Scene parseScene(const JsonValue& root) {
  if (!root.isObject()) {
    throw sceneError("SCENE_ERR_INVALID_TYPE", "scene root must be an object", "scene");
  }
  Scene scene;
  scene.camera = parseCamera(root);

  const JsonValue::Array& meshes = requireArray(root, "meshes", "scene");
  if (meshes.empty()) {
    throw sceneError("SCENE_ERR_INVALID_VALUE", "scene must contain at least one mesh",
                     "scene.meshes");
  }
  scene.meshes.reserve(meshes.size());
  for (std::size_t i = 0; i < meshes.size(); ++i) {
    scene.meshes.push_back(parseMesh(meshes[i], i));
  }

  const JsonValue& lights = requireMember(root, "lights", "scene");
  if (!lights.isObject()) {
    throw sceneError("SCENE_ERR_INVALID_TYPE", "lights must be an object", "scene.lights");
  }
  const JsonValue& directional =
      requireMember(lights, "directional", "scene.lights");
  if (!directional.isNull()) {
    scene.directionalLight = parseDirectionalLight(directional);
    scene.hasDirectionalLight = true;
  }

  const JsonValue& pointLights =
      requireMember(lights, "point_lights", "scene.lights");
  if (!pointLights.isArray()) {
    throw sceneError("SCENE_ERR_INVALID_TYPE", "point_lights must be an array",
                     "scene.lights.point_lights");
  }
  for (std::size_t i = 0; i < pointLights.asArray().size(); ++i) {
    scene.pointLights.push_back(parsePointLight(pointLights.asArray()[i], i));
  }
  if (scene.pointLights.size() > 4) {
    throw sceneError("SCENE_ERR_TOO_MANY_LIGHTS",
                     "at most four point lights are supported",
                     "scene.lights.point_lights");
  }
  return scene;
}

}  // namespace

bool parseSceneJson(const std::string& text, Scene& scene, AppError& error) {
  JsonValue root;
  JsonParseError parseError;
  if (!parseJson(text, root, parseError)) {
    error = sceneError("SCENE_ERR_INVALID_JSON",
                       parseError.message + " at line " + std::to_string(parseError.line) +
                           ", column " + std::to_string(parseError.column),
                       "$");
    return false;
  }
  try {
    scene = parseScene(root);
    return true;
  } catch (const AppError& caught) {
    error = caught;
    return false;
  }
}

bool loadScene(const std::filesystem::path& path, Scene& scene, AppError& error) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    error = sceneError("SCENE_ERR_READ", "unable to open scene file", path.string());
    return false;
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    error = sceneError("SCENE_ERR_READ", "unable to read scene file", path.string());
    return false;
  }
  return parseSceneJson(buffer.str(), scene, error);
}

}  // namespace orv
