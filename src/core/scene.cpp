#include "core/scene.hpp"

#include "core/error.hpp"
#include "core/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace uid166 {

namespace {

class DeterministicRng {
public:
    explicit DeterministicRng(uint64_t seed) : state_(seed ? seed : 0x9e3779b97f4a7c15ull) {}

    uint64_t nextU64() {
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        return state_ * 2685821657736338717ull;
    }

    float uniform(float lo, float hi) {
        const uint32_t bits = static_cast<uint32_t>(nextU64() >> 40);
        const float unit = static_cast<float>(bits) / static_cast<float>(0xffffff);
        return lo + (hi - lo) * unit;
    }

    uint32_t index(uint32_t count) {
        return count ? static_cast<uint32_t>(nextU64() % count) : 0;
    }

private:
    uint64_t state_;
};

Vec3 readVec3(const JsonValue& value, std::string_view where) {
    const auto& arr = requireArray(value, where, ErrorCode::sceneSchema);
    if (arr.array.size() != 3) throw AppError(ErrorCode::sceneSchema, std::string(where) + " must have 3 elements");
    Vec3 v{
        static_cast<float>(requireNumber(arr.array[0], where, ErrorCode::sceneSchema)),
        static_cast<float>(requireNumber(arr.array[1], where, ErrorCode::sceneSchema)),
        static_cast<float>(requireNumber(arr.array[2], where, ErrorCode::sceneSchema)),
    };
    if (!finite(v.x) || !finite(v.y) || !finite(v.z) || std::abs(v.x) > 1.0e6f ||
        std::abs(v.y) > 1.0e6f || std::abs(v.z) > 1.0e6f) {
        throw AppError(ErrorCode::sceneSchema, std::string(where) + " is out of range");
    }
    return v;
}

float readFloat(const JsonValue& object, std::string_view name, ErrorCode code = ErrorCode::sceneSchema) {
    return static_cast<float>(requireNumber(requireMember(object, name, code), name, code));
}

uint32_t readU32(const JsonValue& object, std::string_view name, ErrorCode code = ErrorCode::sceneSchema) {
    const int64_t value = requireInteger(requireMember(object, name, code), name, code);
    if (value < 0 || value > std::numeric_limits<uint32_t>::max()) {
        throw AppError(code, std::string(name) + " is out of uint32 range");
    }
    return static_cast<uint32_t>(value);
}

std::string lightTypeName(LightType type) {
    switch (type) {
    case LightType::directional: return "directional";
    case LightType::point: return "point";
    case LightType::spot: return "spot";
    }
    return "unknown";
}

LightType parseLightType(const std::string& text) {
    if (text == "directional") return LightType::directional;
    if (text == "point") return LightType::point;
    if (text == "spot") return LightType::spot;
    throw AppError(ErrorCode::sceneSchema, "unknown light type: " + text);
}

void writeVec3(JsonWriter& w, const Vec3& v) {
    w.beginArray();
    w.number(v.x);
    w.number(v.y);
    w.number(v.z);
    w.endArray();
}

void writeMotion(JsonWriter& w, const Motion& motion) {
    w.beginObject();
    w.key("enabled"); w.boolean(motion.enabled);
    w.key("center"); writeVec3(w, motion.center);
    w.key("radius"); w.number(motion.radius);
    w.key("speed"); w.number(motion.speed);
    w.key("phase"); w.number(motion.phase);
    w.endObject();
}

Motion readMotion(const JsonValue& object) {
    Motion motion;
    if (const JsonValue* value = object.find("motion")) {
        (void)requireObject(*value, "motion", ErrorCode::sceneSchema);
        const JsonValue* enabled = value->find("enabled");
        if (!enabled || enabled->type != JsonType::booleanValue) {
            throw AppError(ErrorCode::sceneSchema, "motion.enabled must be boolean");
        }
        motion.enabled = enabled->boolean;
        motion.center = readVec3(requireMember(*value, "center", ErrorCode::sceneSchema),
                                 "motion.center");
        motion.radius = readFloat(*value, "radius");
        motion.speed = readFloat(*value, "speed");
        motion.phase = readFloat(*value, "phase");
        if (motion.radius < 0.0f || motion.radius > 1.0e5f || std::abs(motion.speed) > 100.0f) {
            throw AppError(ErrorCode::sceneSchema, "motion values out of range");
        }
    }
    return motion;
}

} // namespace

AABB localCubeBounds() {
    AABB box;
    box.mn = {-0.5f, -0.5f, -0.5f};
    box.mx = {0.5f, 0.5f, 0.5f};
    return box;
}

Mat4 objectWorldMatrix(const SceneObject& object) {
    return Mat4::scale(object.scale) * Mat4::rotationY(object.yaw) *
           Mat4::translation(object.position);
}

void updateObjectWorldBounds(SceneObject& object) {
    object.bounds = transformedAABB(localCubeBounds(), objectWorldMatrix(object));
}

void updateMovingObjects(SceneDesc& scene, float timeSeconds) {
    for (SceneObject& object : scene.objects) {
        if (!object.motion.enabled) continue;
        const float angle = object.motion.phase + timeSeconds * object.motion.speed;
        object.position.x = object.motion.center.x + std::cos(angle) * object.motion.radius;
        object.position.z = object.motion.center.z + std::sin(angle) * object.motion.radius;
        updateObjectWorldBounds(object);
    }
}

void updateMovingLights(SceneDesc& scene, float timeSeconds) {
    for (Light& light : scene.lights) {
        if (!light.motion.enabled || light.type == LightType::directional) continue;
        const float angle = light.motion.phase + timeSeconds * light.motion.speed;
        light.position.x = light.motion.center.x + std::cos(angle) * light.motion.radius;
        light.position.z = light.motion.center.z + std::sin(angle) * light.motion.radius;
        light.position.y = light.motion.center.y +
                           std::sin(angle * 1.7f + light.motion.phase) * light.motion.radius * 0.18f;
    }
}

Camera cameraAtTime(const CameraConfig& config, float timeSeconds) {
    Camera camera;
    const float angle = config.startAngle + config.angularSpeed * timeSeconds;
    camera.target = {0.0f, config.targetHeight, 0.0f};
    camera.eye = {std::cos(angle) * config.orbitRadius,
                  config.orbitHeight + config.targetHeight,
                  std::sin(angle) * config.orbitRadius};
    camera.fovY = config.fovY;
    camera.aspect = config.aspect;
    camera.nearZ = config.nearZ;
    camera.farZ = config.farZ;
    return camera;
}

Camera cameraAtFrame(const CameraConfig& config, uint32_t frame, float frameSeconds) {
    uint32_t effectiveFrame = frame;
    if (config.holdEndFrame > config.holdStartFrame &&
        frame >= config.holdStartFrame && frame < config.holdEndFrame) {
        effectiveFrame = config.holdStartFrame;
    } else if (config.holdEndFrame > config.holdStartFrame && frame >= config.holdEndFrame) {
        effectiveFrame = config.holdStartFrame + (frame - config.holdEndFrame);
    }
    return cameraAtTime(config, static_cast<float>(effectiveFrame) * frameSeconds);
}

SceneDesc generateScene(const GenerateOptions& options) {
    if (options.objectCount == 0 || options.objectCount > 200000) {
        throw AppError(ErrorCode::usage, "object count must be in [1, 200000]");
    }

    SceneDesc scene;
    scene.seed = options.seed;
    scene.name = "uid166-deterministic-city";
    scene.objects.reserve(options.objectCount);
    const uint32_t grid = static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<double>(options.objectCount))));
    const float spacing = 3.05f;
    const float half = (static_cast<float>(grid) - 1.0f) * spacing * 0.5f;

    DeterministicRng rng(options.seed ^ 0xa5a55a5af00dcafeull);
    for (uint32_t i = 0; i < options.objectCount; ++i) {
        const uint32_t row = i / grid;
        const uint32_t col = i % grid;
        SceneObject object;
        object.id = i;
        const float width = rng.uniform(1.15f, 2.55f);
        const float depth = rng.uniform(1.15f, 2.55f);
        float height = rng.uniform(1.2f, 7.5f);
        if ((i % 337) == 0) height = rng.uniform(12.0f, 20.0f);
        object.scale = {width, height, depth};
        object.position.x = static_cast<float>(col) * spacing - half + rng.uniform(-0.72f, 0.72f);
        object.position.z = static_cast<float>(row) * spacing - half + rng.uniform(-0.72f, 0.72f);
        object.position.y = height * 0.5f;
        const float centerDistance = std::sqrt(object.position.x * object.position.x +
                                               object.position.z * object.position.z);
        if (centerDistance < 20.0f) {
            const uint32_t hash = i * 2654435761u;
            int dx = static_cast<int>((hash >> 7) % 401u) - 200;
            int dz = static_cast<int>((hash >> 17) % 401u) - 200;
            if (dx == 0 && dz == 0) dx = 1;
            const float inverseLength = 1.0f / std::sqrt(
                static_cast<float>(dx * dx + dz * dz));
            object.position.x = 20.0f * static_cast<float>(dx) * inverseLength;
            object.position.z = 20.0f * static_cast<float>(dz) * inverseLength;
        }
        object.yaw = rng.uniform(-0.35f, 0.35f);
        object.material = static_cast<uint32_t>((i * 2654435761u) % 8u);

        if ((i % 211u) == 0u && scene.objects.size() < 40u) {
            object.motion.enabled = true;
            object.motion.center = object.position;
            object.motion.radius = rng.uniform(1.0f, 2.6f);
            object.motion.speed = rng.uniform(0.35f, 0.85f) * ((i & 1u) ? 1.0f : -1.0f);
            object.motion.phase = rng.uniform(0.0f, 2.0f * kPi);
        }
        updateObjectWorldBounds(object);
        scene.objects.push_back(object);
    }

    Light sun;
    sun.id = 0;
    sun.type = LightType::directional;
    sun.direction = normalize({-0.45f, -0.82f, 0.35f});
    sun.color = {1.0f, 0.93f, 0.82f};
    sun.intensity = 2.4f;
    sun.range = 500.0f;
    scene.lights.push_back(sun);

    Light p1;
    p1.id = 1;
    p1.type = LightType::point;
    p1.position = {-8.0f, 6.0f, 8.0f};
    p1.color = {1.0f, 0.42f, 0.18f};
    p1.intensity = 22.0f;
    p1.range = 28.0f;
    p1.motion = {true, p1.position, 3.8f, 0.61f, 0.3f};
    scene.lights.push_back(p1);

    Light p2;
    p2.id = 2;
    p2.type = LightType::point;
    p2.position = {9.0f, 4.5f, -7.0f};
    p2.color = {0.25f, 0.55f, 1.0f};
    p2.intensity = 20.0f;
    p2.range = 24.0f;
    p2.motion = {true, p2.position, 3.1f, -0.74f, 2.1f};
    scene.lights.push_back(p2);

    Light s1;
    s1.id = 3;
    s1.type = LightType::spot;
    s1.position = {0.0f, 10.0f, 4.0f};
    s1.direction = normalize({0.0f, -1.0f, 0.18f});
    s1.color = {0.55f, 1.0f, 0.62f};
    s1.intensity = 30.0f;
    s1.range = 34.0f;
    s1.innerCos = std::cos(23.0f * kPi / 180.0f);
    s1.outerCos = std::cos(31.0f * kPi / 180.0f);
    s1.motion = {true, s1.position, 2.3f, 0.46f, 4.4f};
    scene.lights.push_back(s1);

    Light s2;
    s2.id = 4;
    s2.type = LightType::spot;
    s2.position = {-10.0f, 7.0f, -9.0f};
    s2.direction = normalize({0.45f, -1.0f, 0.28f});
    s2.color = {1.0f, 0.28f, 0.35f};
    s2.intensity = 26.0f;
    s2.range = 30.0f;
    s2.innerCos = std::cos(20.0f * kPi / 180.0f);
    s2.outerCos = std::cos(29.0f * kPi / 180.0f);
    s2.motion = {true, s2.position, 2.8f, -0.53f, 1.2f};
    scene.lights.push_back(s2);

    scene.camera.orbitRadius = 16.0f;
    scene.camera.orbitHeight = 8.0f;
    scene.camera.targetHeight = 1.8f;
    scene.camera.farZ = 340.0f;
    scene.camera.angularSpeed = 0.75f;
    return scene;
}

void writeSceneJson(const SceneDesc& scene, const std::filesystem::path& path) {
    JsonWriter w(true);
    w.beginObject();
    w.key("version"); w.integer(scene.version);
    w.key("seed"); w.integer(static_cast<int64_t>(scene.seed));
    w.key("name"); w.string(scene.name);
    w.key("objectCount"); w.integer(scene.objects.size());
    w.key("objects");
    w.beginArray();
    for (const SceneObject& object : scene.objects) {
        w.beginObject();
        w.key("id"); w.integer(object.id);
        w.key("position"); writeVec3(w, object.position);
        w.key("yaw"); w.number(object.yaw);
        w.key("scale"); writeVec3(w, object.scale);
        w.key("material"); w.integer(object.material);
        w.key("motion"); writeMotion(w, object.motion);
        w.endObject();
    }
    w.endArray();

    w.key("lights");
    w.beginArray();
    for (const Light& light : scene.lights) {
        w.beginObject();
        w.key("id"); w.integer(light.id);
        w.key("type"); w.string(lightTypeName(light.type));
        w.key("position"); writeVec3(w, light.position);
        w.key("direction"); writeVec3(w, light.direction);
        w.key("color"); writeVec3(w, light.color);
        w.key("intensity"); w.number(light.intensity);
        w.key("range"); w.number(light.range);
        w.key("innerCos"); w.number(light.innerCos);
        w.key("outerCos"); w.number(light.outerCos);
        w.key("motion"); writeMotion(w, light.motion);
        w.endObject();
    }
    w.endArray();

    w.key("camera");
    w.beginObject();
    w.key("fovY"); w.number(scene.camera.fovY);
    w.key("aspect"); w.number(scene.camera.aspect);
    w.key("nearZ"); w.number(scene.camera.nearZ);
    w.key("farZ"); w.number(scene.camera.farZ);
    w.key("orbitRadius"); w.number(scene.camera.orbitRadius);
    w.key("orbitHeight"); w.number(scene.camera.orbitHeight);
    w.key("targetHeight"); w.number(scene.camera.targetHeight);
    w.key("angularSpeed"); w.number(scene.camera.angularSpeed);
    w.key("startAngle"); w.number(scene.camera.startAngle);
    w.key("holdStartFrame"); w.integer(scene.camera.holdStartFrame);
    w.key("holdEndFrame"); w.integer(scene.camera.holdEndFrame);
    w.endObject();
    w.endObject();
    writeTextFile(path, w.str());
}

SceneDesc loadSceneJson(const std::filesystem::path& path) {
    const JsonValue root = parseJsonFile(path, ErrorCode::sceneMissing, ErrorCode::sceneRead,
                                         ErrorCode::sceneParse, ErrorCode::sceneSchema);
    (void)requireObject(root, "root", ErrorCode::sceneSchema);
    SceneDesc scene;
    scene.version = readU32(root, "version");
    if (scene.version != 1) {
        throw AppError(ErrorCode::sceneSchema, "unsupported scene version");
    }
    const int64_t seed = requireInteger(requireMember(root, "seed", ErrorCode::sceneSchema),
                                        "seed", ErrorCode::sceneSchema);
    if (seed < 0) throw AppError(ErrorCode::sceneSchema, "seed must be non-negative");
    scene.seed = static_cast<uint64_t>(seed);
    scene.name = requireString(requireMember(root, "name", ErrorCode::sceneSchema), "name",
                               ErrorCode::sceneSchema);
    const auto& objects = requireArray(requireMember(root, "objects", ErrorCode::sceneSchema),
                                       "objects", ErrorCode::sceneSchema);
    if (objects.array.empty() || objects.array.size() > 200000) {
        throw AppError(ErrorCode::sceneSchema, "objects array has invalid size");
    }
    const uint32_t declared = readU32(root, "objectCount");
    if (declared != objects.array.size()) {
        throw AppError(ErrorCode::sceneSchema, "objectCount does not match objects array");
    }
    scene.objects.reserve(objects.array.size());
    std::unordered_set<uint32_t> ids;
    ids.reserve(objects.array.size() * 2);
    for (size_t i = 0; i < objects.array.size(); ++i) {
        const JsonValue& value = requireObject(objects.array[i], "object", ErrorCode::sceneSchema);
        SceneObject object;
        object.id = readU32(value, "id");
        if (!ids.insert(object.id).second) {
            throw AppError(ErrorCode::sceneSchema, "duplicate object id");
        }
        object.position = readVec3(requireMember(value, "position", ErrorCode::sceneSchema),
                                   "object.position");
        object.yaw = readFloat(value, "yaw");
        object.scale = readVec3(requireMember(value, "scale", ErrorCode::sceneSchema),
                                "object.scale");
        if (object.scale.x <= 0.0f || object.scale.y <= 0.0f || object.scale.z <= 0.0f ||
            object.scale.x > 10000.0f || object.scale.y > 10000.0f || object.scale.z > 10000.0f) {
            throw AppError(ErrorCode::sceneSchema, "object scale must be positive and bounded");
        }
        object.material = readU32(value, "material");
        if (object.material > 64) throw AppError(ErrorCode::sceneSchema, "material is out of range");
        object.motion = readMotion(value);
        updateObjectWorldBounds(object);
        scene.objects.push_back(object);
    }

    const auto& lights = requireArray(requireMember(root, "lights", ErrorCode::sceneSchema),
                                      "lights", ErrorCode::sceneSchema);
    if (lights.array.size() < 5 || lights.array.size() > 17) {
        throw AppError(ErrorCode::sceneSchema, "expected one directional light and at least four local lights");
    }
    uint32_t directionalCount = 0;
    uint32_t localCount = 0;
    std::unordered_set<uint32_t> lightIds;
    for (const JsonValue& value : lights.array) {
        (void)requireObject(value, "light", ErrorCode::sceneSchema);
        Light light;
        light.id = readU32(value, "id");
        if (!lightIds.insert(light.id).second) {
            throw AppError(ErrorCode::sceneSchema, "duplicate light id");
        }
        light.type = parseLightType(requireString(requireMember(value, "type", ErrorCode::sceneSchema),
                                                  "light.type", ErrorCode::sceneSchema));
        light.position = readVec3(requireMember(value, "position", ErrorCode::sceneSchema),
                                  "light.position");
        light.direction = normalize(readVec3(requireMember(value, "direction", ErrorCode::sceneSchema),
                                             "light.direction"));
        light.color = readVec3(requireMember(value, "color", ErrorCode::sceneSchema),
                               "light.color");
        light.intensity = readFloat(value, "intensity");
        light.range = readFloat(value, "range");
        light.innerCos = readFloat(value, "innerCos");
        light.outerCos = readFloat(value, "outerCos");
        light.motion = readMotion(value);
        if (light.intensity < 0.0f || light.intensity > 10000.0f || light.range <= 0.0f ||
            light.range > 5000.0f || light.innerCos < -1.0f || light.innerCos > 1.0f ||
            light.outerCos < -1.0f || light.outerCos > 1.0f) {
            throw AppError(ErrorCode::sceneSchema, "light values out of range");
        }
        if (light.type == LightType::directional) ++directionalCount;
        else ++localCount;
        scene.lights.push_back(light);
    }
    if (directionalCount != 1 || localCount < 4) {
        throw AppError(ErrorCode::sceneSchema, "scene requires exactly 1 directional and >= 4 local lights");
    }

    const JsonValue& camera = requireObject(requireMember(root, "camera", ErrorCode::sceneSchema),
                                            "camera", ErrorCode::sceneSchema);
    scene.camera.fovY = readFloat(camera, "fovY");
    scene.camera.aspect = readFloat(camera, "aspect");
    scene.camera.nearZ = readFloat(camera, "nearZ");
    scene.camera.farZ = readFloat(camera, "farZ");
    scene.camera.orbitRadius = readFloat(camera, "orbitRadius");
    scene.camera.orbitHeight = readFloat(camera, "orbitHeight");
    scene.camera.targetHeight = readFloat(camera, "targetHeight");
    scene.camera.angularSpeed = readFloat(camera, "angularSpeed");
    scene.camera.startAngle = readFloat(camera, "startAngle");
    scene.camera.holdStartFrame = readU32(camera, "holdStartFrame");
    scene.camera.holdEndFrame = readU32(camera, "holdEndFrame");
    if (scene.camera.fovY <= 0.02f || scene.camera.fovY >= 3.0f ||
        scene.camera.aspect <= 0.1f || scene.camera.aspect > 20.0f ||
        scene.camera.nearZ <= 0.0f || scene.camera.farZ <= scene.camera.nearZ ||
        scene.camera.orbitRadius <= 0.0f || scene.camera.orbitRadius > 100000.0f ||
        std::abs(scene.camera.angularSpeed) > 20.0f) {
        throw AppError(ErrorCode::sceneSchema, "camera values out of range");
    }
    return scene;
}

AABB lightInfluenceBounds(const Light& light) {
    AABB box;
    if (light.type == LightType::directional) {
        box.mn = {-10000.0f, -10000.0f, -10000.0f};
        box.mx = {10000.0f, 10000.0f, 10000.0f};
        return box;
    }
    if (light.type == LightType::point) {
        box.mn = light.position - Vec3{light.range, light.range, light.range};
        box.mx = light.position + Vec3{light.range, light.range, light.range};
        return box;
    }
    const Vec3 tip = light.position;
    Vec3 dir = normalize(light.direction);
    Vec3 up = std::abs(dir.y) > 0.95f ? Vec3{1.0f, 0.0f, 0.0f} : Vec3{0.0f, 1.0f, 0.0f};
    Vec3 right = normalize(cross(up, dir));
    up = normalize(cross(dir, right));
    const float angle = std::acos(std::clamp(light.outerCos, -1.0f, 1.0f));
    const float coneRadius = std::tan(angle) * light.range;
    const Vec3 end = tip + dir * light.range;
    box.expand(tip);
    for (int i = 0; i < 8; ++i) {
        const float sx = (i & 1) ? 1.0f : -1.0f;
        const float sy = (i & 2) ? 1.0f : -1.0f;
        box.expand(end + right * (coneRadius * sx) + up * (coneRadius * sy));
    }
    return box;
}

Mat4 lightViewProjection(const Light& light) {
    if (light.type == LightType::directional) {
        const Vec3 center{0.0f, 4.0f, 0.0f};
        const Mat4 view = lookAtLH(center - normalize(light.direction) * 120.0f,
                                   center, {0.0f, 1.0f, 0.0f});
        return view * orthoLH(-110.0f, 110.0f, -110.0f, 110.0f, 0.1f, 300.0f);
    }
    const Vec3 dir = normalize(light.direction);
    Vec3 up = std::abs(dir.y) > 0.95f ? Vec3{1.0f, 0.0f, 0.0f} : Vec3{0.0f, 1.0f, 0.0f};
    const Mat4 view = lookAtLH(light.position, light.position + dir, up);
    if (light.type == LightType::spot) {
        const float fov = 2.0f * std::acos(std::clamp(light.outerCos, -1.0f, 1.0f));
        return view * perspectiveLH(std::clamp(fov, 0.05f, 2.9f), 1.0f, 0.1f, light.range);
    }
    return view * orthoLH(-light.range, light.range, -light.range, light.range,
                          0.1f, light.range);
}

} // namespace uid166

