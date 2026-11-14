#include "lightcluster/types.hpp"

#include <cmath>

namespace lightcluster {

LightCounts SceneSnapshot::counts() const noexcept {
    LightCounts result;
    result.total = static_cast<std::uint32_t>(lights.size());
    for (const Light& light : lights) {
        if (light.type == LightType::Point) {
            ++result.point;
        } else {
            ++result.directional;
        }
    }
    return result;
}

const char* toString(LightType type) noexcept {
    return type == LightType::Point ? "point" : "directional";
}

Vec3 add(Vec3 a, Vec3 b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 subtract(Vec3 a, Vec3 b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 scale(Vec3 value, float factor) noexcept {
    return {value.x * factor, value.y * factor, value.z * factor};
}

float length(Vec3 value) noexcept {
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

Vec3 normalizeOr(Vec3 value, Vec3 fallback) noexcept {
    const float magnitude = length(value);
    if (magnitude <= 1.0e-8F) {
        return fallback;
    }
    return scale(value, 1.0F / magnitude);
}

}  // namespace lightcluster
