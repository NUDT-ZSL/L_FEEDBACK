#pragma once

#include <cstdint>
#include <vector>

namespace lightcluster {

enum class LightType : std::uint32_t {
    Point = 0,
    Directional = 1,
};

struct Vec3 {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};

struct Light {
    std::uint32_t id = 0;
    LightType type = LightType::Point;
    Vec3 position{};
    Vec3 direction{0.0F, -1.0F, 0.0F};
    Vec3 color{1.0F, 1.0F, 1.0F};
    float intensity = 1.0F;
    float range = 10.0F;
};

struct LightCounts {
    std::uint32_t total = 0;
    std::uint32_t point = 0;
    std::uint32_t directional = 0;
};

struct SceneSnapshot {
    std::uint64_t version = 0;
    std::vector<Light> lights;

    [[nodiscard]] LightCounts counts() const noexcept;
};

[[nodiscard]] const char* toString(LightType type) noexcept;
[[nodiscard]] Vec3 add(Vec3 a, Vec3 b) noexcept;
[[nodiscard]] Vec3 subtract(Vec3 a, Vec3 b) noexcept;
[[nodiscard]] Vec3 scale(Vec3 value, float factor) noexcept;
[[nodiscard]] float length(Vec3 value) noexcept;
[[nodiscard]] Vec3 normalizeOr(Vec3 value, Vec3 fallback) noexcept;

}  // namespace lightcluster
