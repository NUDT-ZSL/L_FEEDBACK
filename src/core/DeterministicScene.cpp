#include "indoor/Scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>

#include <glm/gtc/matrix_transform.hpp>

namespace indoor {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kRoomHalfX = 26.0f;
constexpr float kRoomHalfZ = 18.0f;
constexpr float kRoomHeight = 8.0f;

class DeterministicRng {
public:
    explicit DeterministicRng(std::uint64_t seed) : state_(seed + 0x9e3779b97f4a7c15ULL) {}

    std::uint32_t nextU32() {
        std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return static_cast<std::uint32_t>((z ^ (z >> 31)) >> 32U);
    }

    float uniform(float minimum, float maximum) {
        const float unit = static_cast<float>(nextU32()) /
                           static_cast<float>(std::numeric_limits<std::uint32_t>::max());
        return minimum + (maximum - minimum) * unit;
    }

    std::uint32_t index(std::uint32_t count) {
        return count == 0 ? 0 : nextU32() % count;
    }

private:
    std::uint64_t state_ = 0;
};

const std::array<glm::vec3, 10> kLightPalette = {
    glm::vec3(1.00f, 0.78f, 0.56f), glm::vec3(0.62f, 0.80f, 1.00f),
    glm::vec3(1.00f, 0.55f, 0.38f), glm::vec3(0.52f, 1.00f, 0.82f),
    glm::vec3(0.96f, 0.88f, 0.68f), glm::vec3(0.70f, 0.64f, 1.00f),
    glm::vec3(1.00f, 0.95f, 0.86f), glm::vec3(0.42f, 0.72f, 1.00f),
    glm::vec3(1.00f, 0.42f, 0.58f), glm::vec3(0.70f, 1.00f, 0.52f)};

MaterialParams makeMaterial(DeterministicRng& rng, bool dynamic, std::uint32_t objectId) {
    MaterialParams material;
    const glm::vec3 wood(0.56f, 0.33f, 0.18f);
    const glm::vec3 concrete(0.52f, 0.55f, 0.58f);
    const glm::vec3 painted(0.30f, 0.42f, 0.58f);
    const glm::vec3 accent(0.86f, 0.34f, 0.16f);
    const std::array<glm::vec3, 4> palette = {wood, concrete, painted, accent};

    material.albedo = palette[rng.index(static_cast<std::uint32_t>(palette.size()))];
    material.albedo += glm::vec3(rng.uniform(-0.08f, 0.08f),
                                 rng.uniform(-0.08f, 0.08f),
                                 rng.uniform(-0.08f, 0.08f));
    material.albedo = glm::clamp(material.albedo, glm::vec3(0.08f), glm::vec3(0.95f));
    material.roughness = rng.uniform(0.24f, 0.92f);
    material.metallic = rng.uniform(0.0f, 0.75f) < 0.22f ? rng.uniform(0.35f, 0.92f) : 0.03f;
    material.uvScale = rng.uniform(0.55f, 2.1f);
    material.normalStrength = rng.uniform(0.35f, 1.25f);
    material.baseLayer = rng.index(3);
    material.normalLayer = 3 + rng.index(2);
    material.roughMetalLayer = 5 + rng.index(3);
    if (dynamic) {
        material.albedo = glm::mix(material.albedo, glm::vec3(1.0f, 0.45f, 0.18f),
                                   rng.uniform(0.25f, 0.55f));
        material.roughness = rng.uniform(0.18f, 0.55f);
        material.metallic = rng.uniform(0.1f, 0.8f);
    }
    (void)objectId;
    return material;
}

void addBox(Scene& scene, const ObjectTransform& transform, MaterialParams material,
            bool dynamic = false, Motion motion = {}) {
    SceneObject object;
    object.id = static_cast<std::uint32_t>(scene.objects.size());
    object.dynamic = dynamic;
    object.initialTransform = transform;
    object.transform = transform;
    object.motion = motion;
    object.material = std::move(material);
    scene.objects.push_back(std::move(object));
}

ObjectTransform boxTransform(glm::vec3 center, glm::vec3 halfExtents,
                             float rotationY = 0.0f) {
    ObjectTransform transform;
    transform.position = center;
    transform.rotationEuler = glm::vec3(0.0f, rotationY, 0.0f);
    transform.halfExtents = halfExtents;
    return transform;
}

std::uint64_t hashBytes(std::uint64_t hash, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

template <typename T>
std::uint64_t hashValue(std::uint64_t hash, const T& value) {
    return hashBytes(hash, &value, sizeof(T));
}

} // namespace

const char* materialPresetName(MaterialPresetId preset) {
    switch (preset) {
    case MaterialPresetId::BaseOnly:
        return "0 BaseOnly";
    case MaterialPresetId::NormalDetail:
        return "1 NormalDetail";
    case MaterialPresetId::RoughMetal:
        return "2 RoughMetal";
    case MaterialPresetId::BaseNormal:
        return "3 BaseNormal";
    case MaterialPresetId::FullLayered:
        return "4 FullLayered";
    default:
        return "?";
    }
}

MaterialPresetId nextMaterialPreset(MaterialPresetId preset) {
    const int next = (static_cast<int>(preset) + 1) %
                     static_cast<int>(MaterialPresetId::Count);
    return static_cast<MaterialPresetId>(next);
}

glm::mat4 ObjectTransform::matrix() const {
    glm::mat4 model(1.0f);
    model = glm::translate(model, position);
    model = glm::rotate(model, rotationEuler.y, glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, rotationEuler.x, glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, rotationEuler.z, glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::scale(model, halfExtents);
    return model;
}

float ObjectTransform::boundingRadius() const {
    return glm::length(halfExtents);
}

glm::vec3 Camera::position() const {
    const float cosPitch = std::cos(pitch);
    const glm::vec3 direction(cosPitch * std::sin(yaw),
                              std::sin(pitch),
                              cosPitch * std::cos(yaw));
    return target + direction * distance;
}

glm::mat4 Camera::view() const {
    return glm::lookAt(position(), target, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 Camera::projection(float aspect) const {
    return glm::perspective(glm::radians(58.0f), std::max(aspect, 0.01f),
                            0.05f, 150.0f);
}

Scene buildDeterministicScene(std::uint64_t seed, std::uint32_t lightCount,
                              std::uint64_t generation) {
    Scene scene;
    scene.seed = seed;
    scene.generation = generation;
    scene.animationTime = 0.0;

    DeterministicRng rng(seed ^ (generation * 0x9e3779b97f4a7c15ULL));

    addBox(scene, boxTransform({0.0f, -0.25f, 0.0f},
                               {kRoomHalfX, 0.25f, kRoomHalfZ}),
           makeMaterial(rng, false, 0));
    addBox(scene, boxTransform({0.0f, kRoomHeight + 0.25f, 0.0f},
                               {kRoomHalfX, 0.25f, kRoomHalfZ}),
           makeMaterial(rng, false, 1));
    addBox(scene, boxTransform({0.0f, kRoomHeight * 0.5f, -kRoomHalfZ - 0.25f},
                               {kRoomHalfX, kRoomHeight * 0.5f, 0.25f}),
           makeMaterial(rng, false, 2));
    addBox(scene, boxTransform({0.0f, kRoomHeight * 0.5f, kRoomHalfZ + 0.25f},
                               {kRoomHalfX, kRoomHeight * 0.5f, 0.25f}),
           makeMaterial(rng, false, 3));
    addBox(scene, boxTransform({-kRoomHalfX - 0.25f, kRoomHeight * 0.5f, 0.0f},
                               {0.25f, kRoomHeight * 0.5f, kRoomHalfZ}),
           makeMaterial(rng, false, 4));
    addBox(scene, boxTransform({kRoomHalfX + 0.25f, kRoomHeight * 0.5f, 0.0f},
                               {0.25f, kRoomHeight * 0.5f, kRoomHalfZ}),
           makeMaterial(rng, false, 5));

    for (int column = 0; column < 8; ++column) {
        const float x = (column % 4 - 1.5f) * 12.0f;
        const float z = (column / 4 == 0 ? -10.5f : 10.5f);
        addBox(scene, boxTransform({x, 2.0f, z}, {0.36f, 2.0f, 0.36f},
                                   rng.uniform(-0.04f, 0.04f)),
               makeMaterial(rng, false, 6 + column));
    }

    constexpr int kCrateColumns = 14;
    constexpr int kCrateRows = 10;
    for (int row = 0; row < kCrateRows; ++row) {
        for (int column = 0; column < kCrateColumns; ++column) {
            const float baseX = -22.0f + static_cast<float>(column) * 3.35f;
            const float baseZ = -14.6f + static_cast<float>(row) * 3.18f;
            if (std::abs(baseX) < 5.2f && std::abs(baseZ) < 5.0f) {
                continue;
            }
            const float x = baseX + rng.uniform(-0.48f, 0.48f);
            const float z = baseZ + rng.uniform(-0.42f, 0.42f);
            const float hx = rng.uniform(0.36f, 1.05f);
            const float hy = rng.uniform(0.30f, 1.55f);
            const float hz = rng.uniform(0.36f, 1.05f);
            addBox(scene, boxTransform({x, hy, z}, {hx, hy, hz},
                                       rng.uniform(-0.35f, 0.35f)),
                   makeMaterial(rng, false, static_cast<std::uint32_t>(scene.objects.size())));
        }
    }

    constexpr int kDynamicCount = 16;
    for (int i = 0; i < kDynamicCount; ++i) {
        const float angle = (2.0f * kPi * static_cast<float>(i)) /
                            static_cast<float>(kDynamicCount);
        const float radius = 4.2f + static_cast<float>(i % 4) * 1.45f;
        const glm::vec3 basePosition(std::sin(angle) * radius,
                                     rng.uniform(0.55f, 1.35f),
                                     std::cos(angle) * radius);
        const glm::vec3 halfExtents(rng.uniform(0.30f, 0.72f),
                                    rng.uniform(0.30f, 0.82f),
                                    rng.uniform(0.30f, 0.72f));
        Motion motion;
        motion.amplitude = glm::vec3(rng.uniform(0.55f, 1.45f),
                                     rng.uniform(0.08f, 0.38f),
                                     rng.uniform(0.55f, 1.45f));
        motion.frequency = glm::vec3(rng.uniform(0.22f, 0.48f),
                                     rng.uniform(0.35f, 0.70f),
                                     rng.uniform(0.18f, 0.42f));
        motion.phase = glm::vec3(rng.uniform(0.0f, 2.0f * kPi),
                                 rng.uniform(0.0f, 2.0f * kPi),
                                 rng.uniform(0.0f, 2.0f * kPi));
        addBox(scene,
               boxTransform(basePosition, halfExtents,
                            rng.uniform(-0.8f, 0.8f)),
               makeMaterial(rng, true,
                            static_cast<std::uint32_t>(scene.objects.size())),
               true, motion);
    }

    const std::uint32_t safeLightCount = std::max<std::uint32_t>(lightCount, 1U);
    const std::uint32_t side = static_cast<std::uint32_t>(
        std::ceil(std::sqrt(static_cast<float>(safeLightCount))));
    scene.lights.reserve(safeLightCount);
    for (std::uint32_t i = 0; i < safeLightCount; ++i) {
        const std::uint32_t gx = side <= 1 ? 0 : i % side;
        const std::uint32_t gz = side <= 1 ? 0 : i / side;
        const float tx = side <= 1 ? 0.5f :
            static_cast<float>(gx) / static_cast<float>(side - 1);
        const float tz = side <= 1 ? 0.5f :
            static_cast<float>(gz) / static_cast<float>(side - 1);
        Light light;
        light.id = i;
        light.position = glm::vec3(
            glm::mix(-kRoomHalfX + 2.0f, kRoomHalfX - 2.0f, tx) +
                rng.uniform(-0.72f, 0.72f),
            rng.uniform(kRoomHeight - 1.65f, kRoomHeight - 0.55f),
            glm::mix(-kRoomHalfZ + 2.0f, kRoomHalfZ - 2.0f, tz) +
                rng.uniform(-0.72f, 0.72f));
        light.color = kLightPalette[rng.index(
            static_cast<std::uint32_t>(kLightPalette.size()))];
        light.color += glm::vec3(rng.uniform(-0.04f, 0.04f),
                                 rng.uniform(-0.04f, 0.04f),
                                 rng.uniform(-0.04f, 0.04f));
        light.color = glm::clamp(light.color, glm::vec3(0.08f), glm::vec3(1.0f));
        light.intensity = rng.uniform(4.5f, 14.0f);
        light.range = rng.uniform(5.5f, 11.5f);
        scene.lights.push_back(light);
    }

    return scene;
}

std::vector<MovedObject> setAnimationTime(Scene& scene, double seconds) {
    std::vector<MovedObject> moved;
    const float time = static_cast<float>(seconds);
    for (SceneObject& object : scene.objects) {
        if (!object.dynamic) {
            continue;
        }
        const glm::vec3 oldPosition = object.transform.position;
        const glm::vec3 motion(
            object.motion.amplitude.x *
                std::sin(time * object.motion.frequency.x + object.motion.phase.x),
            object.motion.amplitude.y *
                std::sin(time * object.motion.frequency.y + object.motion.phase.y),
            object.motion.amplitude.z *
                std::sin(time * object.motion.frequency.z + object.motion.phase.z));
        object.transform.position = object.initialTransform.position + motion;

        const glm::vec3 displacement = oldPosition - object.transform.position;
        if (glm::dot(displacement, displacement) > 1.0e-10f) {
            moved.push_back({object.id, oldPosition, object.transform.position,
                             object.transform.boundingRadius()});
        }
    }
    scene.animationTime = seconds;
    return moved;
}

std::vector<MovedObject> advanceAnimation(Scene& scene, double deltaSeconds) {
    return setAnimationTime(scene, scene.animationTime + deltaSeconds);
}

std::vector<std::uint32_t> findAffectedLights(
    const std::vector<Light>& lights,
    const std::vector<MovedObject>& movedObjects) {
    std::unordered_set<std::uint32_t> dirty;
    for (const MovedObject& moved : movedObjects) {
        const glm::vec3 segment = moved.newCenter - moved.oldCenter;
        const float segmentLengthSq = glm::dot(segment, segment);
        for (const Light& light : lights) {
            float t = 0.0f;
            if (segmentLengthSq > 1.0e-12f) {
                t = glm::dot(light.position - moved.oldCenter, segment) /
                    segmentLengthSq;
                t = glm::clamp(t, 0.0f, 1.0f);
            }
            const glm::vec3 closest = moved.oldCenter + segment * t;
            const float reach = light.range + moved.boundingRadius + 0.30f;
            const glm::vec3 delta = light.position - closest;
            if (glm::dot(delta, delta) <= reach * reach) {
                dirty.insert(light.id);
            }
        }
    }
    std::vector<std::uint32_t> result(dirty.begin(), dirty.end());
    std::sort(result.begin(), result.end());
    return result;
}

bool removeLight(Scene& scene, std::uint32_t lightId) {
    const auto it = std::find_if(scene.lights.begin(), scene.lights.end(),
                                 [lightId](const Light& light) {
                                     return light.id == lightId;
                                 });
    if (it == scene.lights.end()) {
        return false;
    }
    scene.lights.erase(it);
    return true;
}

const Light* findLight(const Scene& scene, std::uint32_t lightId) {
    const auto it = std::find_if(scene.lights.begin(), scene.lights.end(),
                                 [lightId](const Light& light) {
                                     return light.id == lightId;
                                 });
    return it == scene.lights.end() ? nullptr : &*it;
}

std::uint64_t sceneDigest(const Scene& scene) {
    std::uint64_t hash = 1469598103934665603ULL;
    hash = hashValue(hash, scene.seed);
    hash = hashValue(hash, scene.generation);
    hash = hashValue(hash, scene.animationTime);
    const std::uint64_t lightCount = scene.lights.size();
    const std::uint64_t objectCount = scene.objects.size();
    hash = hashValue(hash, lightCount);
    hash = hashValue(hash, objectCount);
    for (const Light& light : scene.lights) {
        hash = hashValue(hash, light.id);
        hash = hashValue(hash, light.position);
        hash = hashValue(hash, light.range);
        hash = hashValue(hash, light.color);
        hash = hashValue(hash, light.intensity);
    }
    for (const SceneObject& object : scene.objects) {
        hash = hashValue(hash, object.id);
        const std::uint8_t dynamic = object.dynamic ? 1U : 0U;
        hash = hashValue(hash, dynamic);
        hash = hashValue(hash, object.transform.position);
        hash = hashValue(hash, object.transform.rotationEuler);
        hash = hashValue(hash, object.transform.halfExtents);
        hash = hashValue(hash, object.material.albedo);
        hash = hashValue(hash, object.material.roughness);
        hash = hashValue(hash, object.material.metallic);
        hash = hashValue(hash, object.material.uvScale);
        hash = hashValue(hash, object.material.normalStrength);
        hash = hashValue(hash, object.material.baseLayer);
        hash = hashValue(hash, object.material.normalLayer);
        hash = hashValue(hash, object.material.roughMetalLayer);
    }
    return hash;
}

std::string sceneDigestHex(const Scene& scene) {
    static constexpr char kHex[] = "0123456789abcdef";
    const std::uint64_t digest = sceneDigest(scene);
    std::string result(16, '0');
    for (int i = 15; i >= 0; --i) {
        result[static_cast<std::size_t>(i)] = kHex[digest >> ((15 - i) * 4U) & 0xfU];
    }
    return result;
}

} // namespace indoor
