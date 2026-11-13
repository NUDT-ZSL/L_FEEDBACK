#include "scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <unordered_map>

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 13
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif

namespace {

[[noreturn]] void sceneError(const std::string& message) {
    throw RenderError("E_SCENE_INVALID", message);
}

[[noreturn]] void lightError(const std::string& message) {
    throw RenderError("E_LIGHT_INVALID", message);
}

[[noreturn]] void transformError(const std::string& message) {
    throw RenderError("E_TRANSFORM_INVALID", message);
}

const JsonValue& requireObject(const JsonValue& parent, const std::string& key,
                               const std::string& path) {
    const JsonValue* value = parent.find(key);
    if (!value) sceneError("missing field: " + path + "." + key);
    if (!value->isObject()) sceneError("expected object: " + path + "." + key);
    return *value;
}

const JsonValue::Array& requireArray(const JsonValue& parent, const std::string& key,
                                     const std::string& path) {
    const JsonValue* value = parent.find(key);
    if (!value) sceneError("missing field: " + path + "." + key);
    if (!value->isArray()) sceneError("expected array: " + path + "." + key);
    return value->asArray();
}

double requireNumber(const JsonValue& parent, const std::string& key, const std::string& path) {
    const JsonValue* value = parent.find(key);
    if (!value) sceneError("missing field: " + path + "." + key);
    if (!value->isNumber()) sceneError("expected number: " + path + "." + key);
    return value->asNumber();
}

int requireInt(const JsonValue& parent, const std::string& key, const std::string& path) {
    const double value = requireNumber(parent, key, path);
    if (!std::isfinite(value) || std::floor(value) != value ||
        value < static_cast<double>(std::numeric_limits<int>::min()) ||
        value > static_cast<double>(std::numeric_limits<int>::max())) {
        sceneError("expected integer: " + path + "." + key);
    }
    return static_cast<int>(value);
}

std::string requireString(const JsonValue& parent, const std::string& key,
                          const std::string& path) {
    const JsonValue* value = parent.find(key);
    if (!value) sceneError("missing field: " + path + "." + key);
    if (!value->isString()) sceneError("expected string: " + path + "." + key);
    return value->asString();
}

Vec3 readVec3Array(const JsonValue& value, const std::string& path, const std::string& errorCode) {
    if (!value.isArray() || value.asArray().size() != 3) {
        if (errorCode == "E_LIGHT_INVALID") lightError("expected vec3: " + path);
        if (errorCode == "E_TRANSFORM_INVALID") transformError("expected vec3: " + path);
        sceneError("expected vec3: " + path);
    }
    Vec3 result;
    const auto& array = value.asArray();
    for (int i = 0; i < 3; ++i) {
        if (!array[static_cast<size_t>(i)].isNumber()) {
            if (errorCode == "E_LIGHT_INVALID") lightError("expected numeric vec3: " + path);
            if (errorCode == "E_TRANSFORM_INVALID") transformError("expected numeric vec3: " + path);
            sceneError("expected numeric vec3: " + path);
        }
        const double component = array[static_cast<size_t>(i)].asNumber();
        if (i == 0) result.x = component;
        if (i == 1) result.y = component;
        if (i == 2) result.z = component;
    }
    return result;
}

Vec3 requireVec3(const JsonValue& parent, const std::string& key, const std::string& path,
                 const std::string& errorCode = "E_SCENE_INVALID") {
    const JsonValue* value = parent.find(key);
    if (!value) {
        if (errorCode == "E_LIGHT_INVALID") lightError("missing field: " + path + "." + key);
        if (errorCode == "E_TRANSFORM_INVALID") transformError("missing field: " + path + "." + key);
        sceneError("missing field: " + path + "." + key);
    }
    return readVec3Array(*value, path + "." + key, errorCode);
}

uint32_t requireSeed(const JsonValue& root) {
    const double value = requireNumber(root, "seed", "scene");
    if (!std::isfinite(value) || std::floor(value) != value || value < 0.0 ||
        value > 4294967295.0) {
        sceneError("seed must be an integer in [0, 2^32-1]");
    }
    return static_cast<uint32_t>(value);
}

void validateColor(const Vec3& color, const std::string& path, bool requireUnitRange) {
    if (!finite(color) || color.x < 0.0 || color.y < 0.0 || color.z < 0.0) {
        sceneError("invalid color: " + path);
    }
    if (requireUnitRange &&
        (color.x > 1.0 || color.y > 1.0 || color.z > 1.0)) {
        sceneError("color components must be in [0,1]: " + path);
    }
}

uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

uint64_t hash2(int x, int y, uint32_t seed) {
    const uint64_t a = static_cast<uint32_t>(x) * 0x9e3779b185ebca87ULL;
    const uint64_t b = static_cast<uint32_t>(y) * 0xc2b2ae3d27d4eb4fULL;
    return mix64(a ^ b ^ (static_cast<uint64_t>(seed) << 32) ^ 0xa0761d6478bd642fULL);
}

double hashUnit(int x, int y, uint32_t seed) {
    return static_cast<double>(hash2(x, y, seed) >> 11) * (1.0 / 9007199254740992.0);
}

double smoothStep(double t) {
    t = std::max(0.0, std::min(1.0, t));
    return t * t * (3.0 - 2.0 * t);
}

double valueNoise(double x, double y, uint32_t seed) {
    const int ix = static_cast<int>(std::floor(x));
    const int iy = static_cast<int>(std::floor(y));
    const double fx = x - static_cast<double>(ix);
    const double fy = y - static_cast<double>(iy);
    const double ux = smoothStep(fx);
    const double uy = smoothStep(fy);
    const double v00 = hashUnit(ix, iy, seed);
    const double v10 = hashUnit(ix + 1, iy, seed);
    const double v01 = hashUnit(ix, iy + 1, seed);
    const double v11 = hashUnit(ix + 1, iy + 1, seed);
    const double vx0 = v00 + (v10 - v00) * ux;
    const double vx1 = v01 + (v11 - v01) * ux;
    return vx0 + (vx1 - vx0) * uy;
}

double fbm(double x, double y, uint32_t seed, int octaves) {
    double sum = 0.0;
    double amplitude = 0.5;
    double norm = 0.0;
    for (int octave = 0; octave < octaves; ++octave) {
        sum += amplitude * valueNoise(x, y, seed + static_cast<uint32_t>(octave) * 0x9e3779b9U);
        norm += amplitude;
        x *= 2.03;
        y *= 2.03;
        amplitude *= 0.5;
    }
    return norm > 0.0 ? sum / norm : 0.0;
}

double terrainHeightAt(const HeightfieldParams& params, uint32_t seed, double x, double z) {
    const double absX = std::abs(x);
    const double wallT = smoothStep((absX - params.wallStart) /
                                    std::max(1e-9, params.wallEnd - params.wallStart));
    const double floorNoise = fbm(x * 0.095 + 17.0, z * 0.085 - 9.0, seed, params.noiseOctaves);
    const double wallNoise = fbm(x * 0.16 - 3.0, z * 0.12 + 23.0, seed + 101U,
                                 params.noiseOctaves);
    const double erosion = fbm(x * 0.29 + 7.0, z * 0.17 + 4.0, seed + 211U,
                               std::min(6, params.noiseOctaves + 1));
    const double span = params.heightMax - params.heightMin;
    const double floorHeight = params.heightMin + span * (0.055 + floorNoise * 0.10);
    const double wallHeight = params.heightMin + span *
        (0.34 + wallNoise * 0.38 + (erosion - 0.5) * 0.10);
    const double gullies = (0.5 - std::abs(erosion - 0.5)) * params.erosionStrength *
                           (0.25 + 0.75 * wallT);
    return floorHeight + wallT * (wallHeight - floorHeight) - gullies * wallT;
}

void buildTerrain(Scene& scene) {
    const HeightfieldParams& p = scene.heightfield;
    const int nx = p.resolutionX;
    const int nz = p.resolutionZ;
    std::vector<Vec3> positions(static_cast<size_t>(nx) * static_cast<size_t>(nz));
    for (int zIndex = 0; zIndex < nz; ++zIndex) {
        const double z = p.minZ + (p.maxZ - p.minZ) *
            (static_cast<double>(zIndex) / static_cast<double>(nz - 1));
        for (int xIndex = 0; xIndex < nx; ++xIndex) {
            const double x = p.minX + (p.maxX - p.minX) *
                (static_cast<double>(xIndex) / static_cast<double>(nx - 1));
            positions[static_cast<size_t>(zIndex) * nx + xIndex] =
                {x, terrainHeightAt(p, scene.seed, x, z), z};
        }
    }

    std::vector<Vec3> normals(positions.size());
    const double dx = (p.maxX - p.minX) / static_cast<double>(nx - 1);
    const double dz = (p.maxZ - p.minZ) / static_cast<double>(nz - 1);
    for (int zIndex = 0; zIndex < nz; ++zIndex) {
        for (int xIndex = 0; xIndex < nx; ++xIndex) {
            const int xm = std::max(0, xIndex - 1);
            const int xp = std::min(nx - 1, xIndex + 1);
            const int zm = std::max(0, zIndex - 1);
            const int zp = std::min(nz - 1, zIndex + 1);
            const Vec3 dxv = positions[static_cast<size_t>(zIndex) * nx + xp] -
                             positions[static_cast<size_t>(zIndex) * nx + xm];
            const Vec3 dzv = positions[static_cast<size_t>(zp) * nx + xIndex] -
                             positions[static_cast<size_t>(zm) * nx + xIndex];
            Vec3 n = normalized(cross(dzv, dxv));
            if (!finite(n) || lengthSquared(n) < 0.5) {
                n = {0.0, 1.0, 0.0};
            }
            normals[static_cast<size_t>(zIndex) * nx + xIndex] = n;
            (void)dx;
            (void)dz;
        }
    }

    scene.terrainTriangles.clear();
    scene.terrainTriangles.reserve(static_cast<size_t>(nx - 1) * (nz - 1) * 2);
    uint32_t primitive = 1;
    auto vertex = [&](int xIndex, int zIndex) -> size_t {
        return static_cast<size_t>(zIndex) * nx + xIndex;
    };
    for (int zIndex = 0; zIndex < nz - 1; ++zIndex) {
        for (int xIndex = 0; xIndex < nx - 1; ++xIndex) {
            const size_t a = vertex(xIndex, zIndex);
            const size_t b = vertex(xIndex, zIndex + 1);
            const size_t c = vertex(xIndex + 1, zIndex);
            const size_t d = vertex(xIndex + 1, zIndex + 1);

            TerrainTriangle t1;
            t1.p[0] = positions[a];
            t1.p[1] = positions[b];
            t1.p[2] = positions[c];
            t1.n[0] = normals[a];
            t1.n[1] = normals[b];
            t1.n[2] = normals[c];
            t1.primitiveId = primitive++;
            scene.terrainTriangles.push_back(t1);

            TerrainTriangle t2;
            t2.p[0] = positions[c];
            t2.p[1] = positions[b];
            t2.p[2] = positions[d];
            t2.n[0] = normals[c];
            t2.n[1] = normals[b];
            t2.n[2] = normals[d];
            t2.primitiveId = primitive++;
            scene.terrainTriangles.push_back(t2);
        }
    }
}

RockLocalMesh makeIcosahedron(double radius, int subdivisions, uint32_t seed) {
    const double t = (1.0 + std::sqrt(5.0)) * 0.5;
    std::vector<Vec3> base = {
        {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
        {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
        {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1},
    };
    for (Vec3& v : base) v = normalized(v);
    std::vector<std::array<int, 3>> faces = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
        {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
        {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1},
    };

    for (int level = 0; level < subdivisions; ++level) {
        std::map<std::pair<int, int>, int> midpointCache;
        std::vector<std::array<int, 3>> next;
        next.reserve(faces.size() * 4);
        auto midpoint = [&](int a, int b) {
            if (a > b) std::swap(a, b);
            const auto key = std::make_pair(a, b);
            const auto it = midpointCache.find(key);
            if (it != midpointCache.end()) return it->second;
            base.push_back(normalized(base[static_cast<size_t>(a)] +
                                      base[static_cast<size_t>(b)]));
            const int index = static_cast<int>(base.size() - 1);
            midpointCache[key] = index;
            return index;
        };
        for (const auto& face : faces) {
            const int a = face[0], b = face[1], c = face[2];
            const int ab = midpoint(a, b);
            const int bc = midpoint(b, c);
            const int ca = midpoint(c, a);
            next.push_back({a, ab, ca});
            next.push_back({b, bc, ab});
            next.push_back({c, ca, bc});
            next.push_back({ab, bc, ca});
        }
        faces.swap(next);
    }

    RockLocalMesh mesh;
    mesh.vertices.reserve(base.size());
    for (size_t i = 0; i < base.size(); ++i) {
        const uint64_t h = mix64(static_cast<uint64_t>(seed) ^ (i * 0x9e3779b97f4a7c15ULL));
        const double unit = static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0);
        const double radial = radius * (0.86 + 0.28 * unit);
        mesh.vertices.push_back(base[i] * radial);
    }
    for (const auto& face : faces) {
        int a = face[0], b = face[1], c = face[2];
        Vec3 n = cross(mesh.vertices[static_cast<size_t>(b)] - mesh.vertices[static_cast<size_t>(a)],
                       mesh.vertices[static_cast<size_t>(c)] - mesh.vertices[static_cast<size_t>(a)]);
        const Vec3 centroid = (mesh.vertices[static_cast<size_t>(a)] +
                               mesh.vertices[static_cast<size_t>(b)] +
                               mesh.vertices[static_cast<size_t>(c)]) / 3.0;
        if (dot(n, centroid) < 0.0) std::swap(b, c);
        mesh.indices.push_back(a);
        mesh.indices.push_back(b);
        mesh.indices.push_back(c);
    }
    return mesh;
}

Transform parseTransform(const JsonValue& value, const std::string& path) {
    if (!value.isObject()) transformError("transform must be an object: " + path);
    Transform transform;
    transform.translation = requireVec3(value, "translation", path, "E_TRANSFORM_INVALID");
    transform.rotationDeg = requireVec3(value, "rotationDeg", path, "E_TRANSFORM_INVALID");
    transform.scale = requireVec3(value, "scale", path, "E_TRANSFORM_INVALID");
    if (!finite(transform.translation) || !finite(transform.rotationDeg) ||
        !finite(transform.scale)) {
        transformError("non-finite transform value: " + path);
    }
    if (!(transform.scale.x > 0.0) || !(transform.scale.y > 0.0) ||
        !(transform.scale.z > 0.0)) {
        transformError("transform scale must be finite and positive: " + path);
    }
    return transform;
}

RockMaterial parseRockMaterial(const JsonValue& value, const std::string& path) {
    if (!value.isObject()) sceneError("rock material must be an object: " + path);
    RockMaterial material;
    material.baseColor = requireVec3(value, "baseColor", path);
    validateColor(material.baseColor, path + ".baseColor", true);
    material.roughness = requireNumber(value, "roughness", path);
    if (!std::isfinite(material.roughness) || material.roughness < 0.0 ||
        material.roughness > 1.0) {
        sceneError("roughness must be in [0,1]: " + path);
    }
    return material;
}

void validateCamera(const Camera& camera) {
    if (!finite(camera.position) || !finite(camera.target) || !finite(camera.up)) {
        sceneError("camera vectors must be finite");
    }
    if (lengthSquared(camera.target - camera.position) < 1e-12) {
        sceneError("camera target must differ from position");
    }
    const Vec3 forward = normalized(camera.target - camera.position);
    const Vec3 up = normalized(camera.up);
    if (lengthSquared(forward) < 0.5 || lengthSquared(up) < 0.5 ||
        std::abs(dot(forward, up)) > 0.999) {
        sceneError("invalid camera basis");
    }
    if (!std::isfinite(camera.fovYDeg) || camera.fovYDeg <= 1.0 || camera.fovYDeg >= 179.0) {
        sceneError("camera fovYDeg must be in (1,179)");
    }
}

void validateLights(const std::vector<Light>& lights) {
    int directional = 0;
    int point = 0;
    for (size_t i = 0; i < lights.size(); ++i) {
        const Light& light = lights[i];
        if (!std::isfinite(light.intensity) || light.intensity <= 0.0) {
            lightError("light intensity must be finite and positive: lights[" +
                       std::to_string(i) + "].intensity");
        }
        if (!finite(light.color) || light.color.x < 0.0 || light.color.y < 0.0 ||
            light.color.z < 0.0) {
            lightError("light color must be finite and non-negative: lights[" +
                       std::to_string(i) + "].color");
        }
        if (light.type == LightType::Directional) {
            ++directional;
            if (!finite(light.direction) || lengthSquared(light.direction) < 1e-12) {
                lightError("directional light direction must be finite and non-zero: lights[" +
                           std::to_string(i) + "]");
            }
        } else {
            ++point;
            if (!finite(light.position) || !std::isfinite(light.range) || light.range <= 0.0) {
                lightError("point light position/range must be finite and range positive: lights[" +
                           std::to_string(i) + "]");
            }
        }
    }
    if (lights.size() != 3 || directional != 2 || point != 1) {
        lightError("scene requires exactly two directional lights and one point light");
    }
}

void validateHierarchy(const Scene& scene) {
    std::vector<int> state(scene.rocks.size(), 0);
    std::function<void(int)> visit = [&](int index) {
        if (state[static_cast<size_t>(index)] == 1) {
            sceneError("rock parent-child cycle detected at '" +
                       scene.rocks[static_cast<size_t>(index)].id + "'");
        }
        if (state[static_cast<size_t>(index)] == 2) return;
        state[static_cast<size_t>(index)] = 1;
        if (scene.rocks[static_cast<size_t>(index)].parent >= 0) {
            visit(scene.rocks[static_cast<size_t>(index)].parent);
        }
        state[static_cast<size_t>(index)] = 2;
    };
    for (size_t i = 0; i < scene.rocks.size(); ++i) {
        visit(static_cast<int>(i));
    }
}

} // namespace

uint32_t Scene::depthMax() const {
    if (depthBits >= 32) return 0xffffffffU;
    return (1U << static_cast<unsigned>(depthBits)) - 1U;
}

Mat4 Scene::viewMatrix() const {
    const Vec3 eye = camera.position;
    const Vec3 forward = normalized(camera.target - camera.position);
    const Vec3 right = normalized(cross(forward, camera.up));
    const Vec3 up = cross(right, forward);

    Mat4 view = Mat4::identity();
    view.at(0, 0) = right.x;
    view.at(0, 1) = right.y;
    view.at(0, 2) = right.z;
    view.at(0, 3) = -dot(right, eye);
    view.at(1, 0) = up.x;
    view.at(1, 1) = up.y;
    view.at(1, 2) = up.z;
    view.at(1, 3) = -dot(up, eye);
    view.at(2, 0) = -forward.x;
    view.at(2, 1) = -forward.y;
    view.at(2, 2) = -forward.z;
    view.at(2, 3) = dot(forward, eye);
    return view;
}

Mat4 Scene::projectionMatrix() const {
    const double aspect = static_cast<double>(width) / static_cast<double>(height);
    const double f = 1.0 / std::tan(camera.fovYDeg * 3.14159265358979323846 / 360.0);
    Mat4 projection;
    projection.at(0, 0) = f / aspect;
    projection.at(1, 1) = f;
    projection.at(2, 2) = (farPlane + nearPlane) / (nearPlane - farPlane);
    projection.at(2, 3) = (2.0 * farPlane * nearPlane) / (nearPlane - farPlane);
    projection.at(3, 2) = -1.0;
    return projection;
}

Mat4 Scene::viewProjectionMatrix() const {
    return projectionMatrix() * viewMatrix();
}

std::vector<int> Scene::subtree(int rootIndex) const {
    std::vector<int> result;
    if (rootIndex < 0 || rootIndex >= static_cast<int>(rocks.size())) return result;
    std::vector<int> stack{rootIndex};
    while (!stack.empty()) {
        const int index = stack.back();
        stack.pop_back();
        result.push_back(index);
        const auto& children = rocks[static_cast<size_t>(index)].children;
        for (auto it = children.rbegin(); it != children.rend(); ++it) {
            stack.push_back(*it);
        }
    }
    return result;
}

Scene loadScene(const std::string& path) {
    const JsonValue root = JsonValue::parseFile(path);
    if (!root.isObject()) sceneError("scene root must be an object");
    Scene scene;
    scene.seed = requireSeed(root);

    const JsonValue& render = requireObject(root, "render", "scene");
    scene.width = requireInt(render, "width", "render");
    scene.height = requireInt(render, "height", "render");
    scene.tileSize = requireInt(render, "tileSize", "render");
    scene.depthBits = requireInt(render, "depthBits", "render");
    scene.stencilBits = requireInt(render, "stencilBits", "render");
    scene.nearPlane = requireNumber(render, "near", "render");
    scene.farPlane = requireNumber(render, "far", "render");
    scene.shadowExtrusion = requireNumber(render, "shadowExtrusion", "render");
    const double bias = requireNumber(render, "shadowDepthBiasUnits", "render");
    if (scene.width < 640 || scene.height < 360 || scene.width > 4096 || scene.height > 4096) {
        sceneError("render resolution must be at least 640x360 and at most 4096x4096");
    }
    if (scene.tileSize < 1 || scene.tileSize > 256) sceneError("invalid tileSize");
    if (scene.depthBits < 16 || scene.depthBits > 30) {
        sceneError("depthBits must be in [16,30]");
    }
    if (scene.stencilBits < 1 || scene.stencilBits > 16) {
        sceneError("stencilBits must be in [1,16]");
    }
    if (!std::isfinite(scene.nearPlane) || !std::isfinite(scene.farPlane) ||
        scene.nearPlane <= 0.0 || scene.farPlane <= scene.nearPlane) {
        sceneError("near/far planes are invalid");
    }
    if (!std::isfinite(scene.shadowExtrusion) || scene.shadowExtrusion <= 0.0) {
        sceneError("shadowExtrusion must be finite and positive");
    }
    if (!std::isfinite(bias) || bias < 0.0 || bias > 4294967295.0 ||
        std::floor(bias) != bias) {
        sceneError("shadowDepthBiasUnits must be a non-negative integer");
    }
    scene.shadowDepthBiasUnits = static_cast<uint32_t>(bias);

    const JsonValue& camera = requireObject(root, "camera", "scene");
    scene.camera.position = requireVec3(camera, "position", "camera");
    scene.camera.target = requireVec3(camera, "target", "camera");
    scene.camera.up = requireVec3(camera, "up", "camera");
    scene.camera.fovYDeg = requireNumber(camera, "fovYDeg", "camera");
    validateCamera(scene.camera);

    const JsonValue& hf = requireObject(root, "heightfield", "scene");
    const auto& resolution = requireArray(hf, "resolution", "heightfield");
    if (resolution.size() != 2 || !resolution[0].isNumber() || !resolution[1].isNumber()) {
        sceneError("heightfield.resolution must contain two integers");
    }
    for (const JsonValue& component : resolution) {
        const double value = component.asNumber();
        if (!std::isfinite(value) || std::floor(value) != value) {
            sceneError("heightfield.resolution must contain two finite integers");
        }
    }
    scene.heightfield.resolutionX = static_cast<int>(resolution[0].asNumber());
    scene.heightfield.resolutionZ = static_cast<int>(resolution[1].asNumber());
    if (scene.heightfield.resolutionX < 8 || scene.heightfield.resolutionZ < 8 ||
        scene.heightfield.resolutionX > 512 || scene.heightfield.resolutionZ > 512) {
        sceneError("heightfield resolution must be in [8,512]");
    }
    const auto& extent = requireArray(hf, "extent", "heightfield");
    if (extent.size() != 4) {
        sceneError("heightfield.extent must contain [minX,maxX,minZ,maxZ]");
    }
    for (const JsonValue& component : extent) {
        if (!component.isNumber()) sceneError("heightfield.extent must be numeric");
    }
    scene.heightfield.minX = extent[0].asNumber();
    scene.heightfield.maxX = extent[1].asNumber();
    scene.heightfield.minZ = extent[2].asNumber();
    scene.heightfield.maxZ = extent[3].asNumber();
    const auto& heightRange = requireArray(hf, "heightRange", "heightfield");
    if (heightRange.size() != 2 || !heightRange[0].isNumber() ||
        !heightRange[1].isNumber()) {
        sceneError("heightfield.heightRange must contain [min,max]");
    }
    scene.heightfield.heightMin = heightRange[0].asNumber();
    scene.heightfield.heightMax = heightRange[1].asNumber();
    scene.heightfield.wallStart = requireNumber(hf, "wallStart", "heightfield");
    scene.heightfield.wallEnd = requireNumber(hf, "wallEnd", "heightfield");
    scene.heightfield.noiseOctaves = requireInt(hf, "noiseOctaves", "heightfield");
    scene.heightfield.erosionStrength = requireNumber(hf, "erosionStrength", "heightfield");
    if (!std::isfinite(scene.heightfield.minX) || !std::isfinite(scene.heightfield.maxX) ||
        !std::isfinite(scene.heightfield.minZ) || !std::isfinite(scene.heightfield.maxZ) ||
        scene.heightfield.minX >= scene.heightfield.maxX ||
        scene.heightfield.minZ >= scene.heightfield.maxZ) {
        sceneError("heightfield extent is invalid");
    }
    if (!std::isfinite(scene.heightfield.heightMin) ||
        !std::isfinite(scene.heightfield.heightMax) ||
        scene.heightfield.heightMin < 0.0 ||
        scene.heightfield.heightMax <= scene.heightfield.heightMin) {
        sceneError("heightfield heightRange is invalid (must be non-negative and increasing)");
    }
    if (!std::isfinite(scene.heightfield.wallStart) ||
        !std::isfinite(scene.heightfield.wallEnd) ||
        scene.heightfield.wallStart < 0.0 ||
        scene.heightfield.wallEnd <= scene.heightfield.wallStart) {
        sceneError("heightfield wall range is invalid");
    }
    if (scene.heightfield.noiseOctaves < 1 || scene.heightfield.noiseOctaves > 8) {
        sceneError("heightfield.noiseOctaves must be in [1,8]");
    }
    if (!std::isfinite(scene.heightfield.erosionStrength) ||
        scene.heightfield.erosionStrength < 0.0) {
        sceneError("heightfield.erosionStrength must be non-negative");
    }

    const JsonValue& materials = requireObject(root, "materials", "scene");
    scene.ambient = requireVec3(materials, "ambient", "materials");
    validateColor(scene.ambient, "materials.ambient", true);
    const auto& layerArray = requireArray(materials, "layers", "materials");
    if (layerArray.size() < 3) sceneError("at least three terrain material layers are required");
    for (size_t i = 0; i < layerArray.size(); ++i) {
        const JsonValue& layerValue = layerArray[i];
        const std::string path = "materials.layers[" + std::to_string(i) + "]";
        if (!layerValue.isObject()) sceneError("material layer must be an object: " + path);
        MaterialLayer layer;
        layer.name = requireString(layerValue, "name", path);
        layer.color = requireVec3(layerValue, "color", path);
        layer.slopeTint = requireVec3(layerValue, "slopeTint", path);
        validateColor(layer.color, path + ".color", true);
        validateColor(layer.slopeTint, path + ".slopeTint", true);
        layer.heightStart = requireNumber(layerValue, "heightStart", path);
        layer.heightEnd = requireNumber(layerValue, "heightEnd", path);
        if (!std::isfinite(layer.heightStart) || !std::isfinite(layer.heightEnd) ||
            layer.heightStart < 0.0 || layer.heightEnd <= layer.heightStart) {
            sceneError("invalid material layer height range: " + path);
        }
        unsigned colorSignature = static_cast<unsigned>((layer.color.x * 255.0 + 0.5)) << 16U |
                                  static_cast<unsigned>((layer.color.y * 255.0 + 0.5)) << 8U |
                                  static_cast<unsigned>((layer.color.z * 255.0 + 0.5));
        for (size_t j = 0; j < scene.layers.size(); ++j) {
            unsigned previous = static_cast<unsigned>((scene.layers[j].color.x * 255.0 + 0.5)) << 16U |
                                static_cast<unsigned>((scene.layers[j].color.y * 255.0 + 0.5)) << 8U |
                                static_cast<unsigned>((scene.layers[j].color.z * 255.0 + 0.5));
            if (colorSignature == previous) {
                sceneError("material layer colors must be distinguishable: " + path);
            }
        }
        scene.layers.push_back(layer);
    }

    const auto& lightArray = requireArray(root, "lights", "scene");
    for (size_t i = 0; i < lightArray.size(); ++i) {
        const JsonValue& lightValue = lightArray[i];
        const std::string path = "lights[" + std::to_string(i) + "]";
        if (!lightValue.isObject()) lightError("light must be an object: " + path);
        Light light;
        const std::string type = requireString(lightValue, "type", path);
        if (type == "directional") {
            light.type = LightType::Directional;
            light.direction = requireVec3(lightValue, "direction", path, "E_LIGHT_INVALID");
        } else if (type == "point") {
            light.type = LightType::Point;
            light.position = requireVec3(lightValue, "position", path, "E_LIGHT_INVALID");
            light.range = requireNumber(lightValue, "range", path);
        } else {
            lightError("unknown light type: " + path);
        }
        light.color = requireVec3(lightValue, "color", path, "E_LIGHT_INVALID");
        light.intensity = requireNumber(lightValue, "intensity", path);
        scene.lights.push_back(light);
    }
    validateLights(scene.lights);

    const auto& rockArray = requireArray(root, "rocks", "scene");
    if (rockArray.empty()) sceneError("scene must contain at least one floating rock");
    std::unordered_map<std::string, int> ids;
    std::vector<std::string> parentIds;
    for (size_t i = 0; i < rockArray.size(); ++i) {
        const JsonValue& rockValue = rockArray[i];
        const std::string path = "rocks[" + std::to_string(i) + "]";
        if (!rockValue.isObject()) sceneError("rock must be an object: " + path);
        RockNode rock;
        rock.id = requireString(rockValue, "id", path);
        if (rock.id.empty()) sceneError("rock id cannot be empty: " + path);
        if (ids.count(rock.id) != 0) sceneError("duplicate rock id: " + rock.id);
        ids[rock.id] = static_cast<int>(i);

        const JsonValue* parentValue = rockValue.find("parent");
        if (parentValue && !parentValue->isNull()) {
            if (!parentValue->isString()) sceneError("rock parent must be a string or null: " + path);
            parentIds.push_back(parentValue->asString());
        } else {
            parentIds.push_back(std::string());
        }

        const JsonValue* transformValue = rockValue.find("transform");
        if (!transformValue) transformError("missing rock transform: " + path);
        rock.transform = parseTransform(*transformValue, path + ".transform");

        const JsonValue* materialValue = rockValue.find("material");
        if (!materialValue) sceneError("missing rock material: " + path);
        rock.material = parseRockMaterial(*materialValue, path + ".material");

        const JsonValue* meshValue = rockValue.find("mesh");
        if (!meshValue || !meshValue->isObject()) sceneError("missing rock mesh: " + path);
        rock.mesh.radius = requireNumber(*meshValue, "radius", path + ".mesh");
        rock.mesh.subdivisions = requireInt(*meshValue, "subdivisions", path + ".mesh");
        if (!std::isfinite(rock.mesh.radius) || rock.mesh.radius <= 0.0) {
            sceneError("rock mesh radius must be finite and positive: " + path);
        }
        if (rock.mesh.subdivisions < 0 || rock.mesh.subdivisions > 2) {
            sceneError("rock mesh subdivisions must be in [0,2]: " + path);
        }
        scene.rocks.push_back(std::move(rock));
    }
    for (size_t i = 0; i < parentIds.size(); ++i) {
        if (parentIds[i].empty()) continue;
        const auto it = ids.find(parentIds[i]);
        if (it == ids.end()) sceneError("unknown parent '" + parentIds[i] + "' for rock '" +
                                        scene.rocks[i].id + "'");
        if (it->second == static_cast<int>(i)) sceneError("rock cannot parent itself: " +
                                                           scene.rocks[i].id);
        scene.rocks[i].parent = it->second;
        scene.rocks[static_cast<size_t>(it->second)].children.push_back(static_cast<int>(i));
    }
    for (size_t i = 0; i < scene.rocks.size(); ++i) {
        std::sort(scene.rocks[i].children.begin(), scene.rocks[i].children.end());
    }
    validateHierarchy(scene);

    for (size_t i = 0; i < scene.rocks.size(); ++i) {
        const uint32_t rockSeed = static_cast<uint32_t>(
            mix64(static_cast<uint64_t>(scene.seed) ^ (i * 0x9e3779b97f4a7c15ULL)));
        scene.rocks[i].localMesh = makeIcosahedron(
            scene.rocks[i].mesh.radius, scene.rocks[i].mesh.subdivisions, rockSeed);
    }

    const JsonValue* changesValue = root.find("changes");
    if (changesValue) {
        if (!changesValue->isArray()) sceneError("changes must be an array");
        const auto& changes = changesValue->asArray();
        for (size_t i = 0; i < changes.size(); ++i) {
            const JsonValue& changeValue = changes[i];
            const std::string path = "changes[" + std::to_string(i) + "]";
            if (!changeValue.isObject()) sceneError("change must be an object: " + path);
            Change change;
            const std::string target = requireString(changeValue, "target", path);
            const auto targetIt = ids.find(target);
            if (targetIt == ids.end()) sceneError("change target does not exist: " + target);
            change.target = targetIt->second;
            const std::string kind = requireString(changeValue, "kind", path);
            if (kind == "transform") {
                change.kind = ChangeKind::Transform;
                const JsonValue* patchValue = changeValue.find("transform");
                if (!patchValue || !patchValue->isObject()) {
                    transformError("transform change requires a transform object: " + path);
                }
                const JsonValue& patch = *patchValue;
                if (const JsonValue* v = patch.find("translation")) {
                    change.transform.hasTranslation = true;
                    change.transform.translation = readVec3Array(*v, path + ".translation",
                                                                 "E_TRANSFORM_INVALID");
                    if (!finite(change.transform.translation)) {
                        transformError("non-finite transform translation: " + path);
                    }
                }
                if (const JsonValue* v = patch.find("rotationDeg")) {
                    change.transform.hasRotation = true;
                    change.transform.rotationDeg = readVec3Array(*v, path + ".rotationDeg",
                                                                 "E_TRANSFORM_INVALID");
                    if (!finite(change.transform.rotationDeg)) {
                        transformError("non-finite transform rotation: " + path);
                    }
                }
                if (const JsonValue* v = patch.find("scale")) {
                    change.transform.hasScale = true;
                    change.transform.scale = readVec3Array(*v, path + ".scale",
                                                           "E_TRANSFORM_INVALID");
                    if (!finite(change.transform.scale) ||
                        change.transform.scale.x <= 0.0 ||
                        change.transform.scale.y <= 0.0 ||
                        change.transform.scale.z <= 0.0) {
                        transformError("transform scale must be finite and positive: " + path);
                    }
                }
                if (!change.transform.hasTranslation && !change.transform.hasRotation &&
                    !change.transform.hasScale) {
                    transformError("transform change does not specify any component: " + path);
                }
            } else if (kind == "material") {
                change.kind = ChangeKind::Material;
                const JsonValue* patchValue = changeValue.find("material");
                if (!patchValue || !patchValue->isObject()) {
                    sceneError("material change requires a material object: " + path);
                }
                const JsonValue& patch = *patchValue;
                if (const JsonValue* v = patch.find("baseColor")) {
                    change.material.hasBaseColor = true;
                    change.material.baseColor = readVec3Array(*v, path + ".baseColor",
                                                              "E_SCENE_INVALID");
                    validateColor(change.material.baseColor, path + ".baseColor", true);
                }
                if (const JsonValue* v = patch.find("roughness")) {
                    if (!v->isNumber()) sceneError("roughness must be numeric: " + path);
                    change.material.hasRoughness = true;
                    change.material.roughness = v->asNumber();
                    if (!std::isfinite(change.material.roughness) ||
                        change.material.roughness < 0.0 ||
                        change.material.roughness > 1.0) {
                        sceneError("roughness must be in [0,1]: " + path);
                    }
                }
                if (!change.material.hasBaseColor && !change.material.hasRoughness) {
                    sceneError("material change does not specify any field: " + path);
                }
            } else {
                sceneError("unknown change kind '" + kind + "': " + path);
            }
            scene.changes.push_back(change);
        }
    }

    buildTerrain(scene);
    rebuildWorldTransforms(scene);
    rebuildAllRockGeometry(scene);
    return scene;
}

void rebuildWorldTransforms(Scene& scene) {
    std::vector<int> visiting(scene.rocks.size(), 0);
    std::function<void(int, const Mat4&)> compute = [&](int index, const Mat4& parentWorld) {
        if (visiting[static_cast<size_t>(index)] != 0) {
            sceneError("rock hierarchy contains a cycle");
        }
        visiting[static_cast<size_t>(index)] = 1;
        RockNode& rock = scene.rocks[static_cast<size_t>(index)];
        const Mat4 local = makeTRS(rock.transform.translation, rock.transform.rotationDeg,
                                   rock.transform.scale);
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                if (!std::isfinite(local.at(row, col))) {
                    transformError("non-finite local transform matrix for rock '" + rock.id + "'");
                }
            }
        }
        rock.world = parentWorld * local;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                if (!std::isfinite(rock.world.at(row, col))) {
                    transformError("non-finite world transform matrix for rock '" + rock.id + "'");
                }
            }
        }
        for (const int child : rock.children) {
            compute(child, rock.world);
        }
        visiting[static_cast<size_t>(index)] = 2;
    };
    for (size_t i = 0; i < scene.rocks.size(); ++i) {
        if (scene.rocks[i].parent < 0) {
            compute(static_cast<int>(i), Mat4::identity());
        }
    }
    for (size_t i = 0; i < scene.rocks.size(); ++i) {
        if (visiting[i] != 2) {
            sceneError("rock hierarchy is not connected to a root: " +
                       scene.rocks[i].id);
        }
    }
}

void rebuildRockGeometry(Scene& scene, int rockIndex) {
    if (rockIndex < 0 || rockIndex >= static_cast<int>(scene.rocks.size())) return;
    RockNode& rock = scene.rocks[static_cast<size_t>(rockIndex)];
    rock.worldTriangles.clear();
    rock.worldTriangles.reserve(rock.localMesh.indices.size() / 3);
    for (size_t i = 0; i + 2 < rock.localMesh.indices.size(); i += 3) {
        RockTriangle triangle;
        for (int k = 0; k < 3; ++k) {
            const int localIndex = rock.localMesh.indices[i + static_cast<size_t>(k)];
            triangle.p[k] = transformPoint(rock.world,
                                           rock.localMesh.vertices[static_cast<size_t>(localIndex)]);
        }
        triangle.normal = normalized(cross(triangle.p[1] - triangle.p[0],
                                           triangle.p[2] - triangle.p[0]));
        triangle.primitiveId = static_cast<uint32_t>(100000 + rockIndex * 200000 +
                                                     static_cast<int>(i / 3));
        rock.worldTriangles.push_back(triangle);
    }
}

void rebuildAllRockGeometry(Scene& scene) {
    for (int i = 0; i < static_cast<int>(scene.rocks.size()); ++i) {
        rebuildRockGeometry(scene, i);
    }
}

void applyChange(Scene& scene, const Change& change) {
    if (change.target < 0 || change.target >= static_cast<int>(scene.rocks.size())) {
        sceneError("change target index is out of range");
    }
    RockNode& rock = scene.rocks[static_cast<size_t>(change.target)];
    if (change.kind == ChangeKind::Transform) {
        if (change.transform.hasTranslation) rock.transform.translation = change.transform.translation;
        if (change.transform.hasRotation) rock.transform.rotationDeg = change.transform.rotationDeg;
        if (change.transform.hasScale) rock.transform.scale = change.transform.scale;
        if (!finite(rock.transform.translation) || !finite(rock.transform.rotationDeg) ||
            !finite(rock.transform.scale)) {
            transformError("non-finite transform value for rock '" + rock.id + "'");
        }
    } else {
        if (change.material.hasBaseColor) rock.material.baseColor = change.material.baseColor;
        if (change.material.hasRoughness) rock.material.roughness = change.material.roughness;
    }
}

void applyAllChanges(Scene& scene) {
    for (const Change& change : scene.changes) {
        applyChange(scene, change);
    }
    rebuildWorldTransforms(scene);
    rebuildAllRockGeometry(scene);
}

std::string changeKindName(ChangeKind kind) {
    return kind == ChangeKind::Transform ? "transform" : "material";
}
