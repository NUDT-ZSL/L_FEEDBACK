#include "environment.h"

#include <cmath>

namespace pbr {
namespace {

using Vec3d = std::array<double, 3>;

double smoothstep(double edge0, double edge1, double x) {
    const double t = std::max(0.0, std::min(1.0, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0 - 2.0 * t);
}

Vec3d evaluateEnvironmentDouble(const Environment& environment, const Vec3d& direction) {
    const double y = direction[1];
    Vec3d color;
    if (y >= 0.0) {
        const double t = std::pow(std::max(0.0, std::min(1.0, y)), 0.5);
        color = {
            environment.horizon.x + (environment.zenith.x - environment.horizon.x) * t,
            environment.horizon.y + (environment.zenith.y - environment.horizon.y) * t,
            environment.horizon.z + (environment.zenith.z - environment.horizon.z) * t,
        };
    } else {
        const double t = std::pow(std::min(1.0, -y), 0.45);
        color = {
            environment.horizon.x + (environment.ground.x - environment.horizon.x) * t,
            environment.horizon.y + (environment.ground.y - environment.horizon.y) * t,
            environment.horizon.z + (environment.ground.z - environment.horizon.z) * t,
        };
    }

    const Vec3 sun = normalize(environment.sunDirection);
    const double cosine = direction[0] * sun.x + direction[1] * sun.y + direction[2] * sun.z;
    const double inner = std::cos(environment.sunAngularRadiusDegrees * kPi / 180.0);
    const double outer = std::cos(environment.sunAngularRadiusDegrees * 1.35 * kPi / 180.0);
    const double sunMask = smoothstep(outer, inner, cosine);
    color[0] += environment.sunRadiance.x * sunMask;
    color[1] += environment.sunRadiance.y * sunMask;
    color[2] += environment.sunRadiance.z * sunMask;
    return color;
}

std::array<double, 9> shBasis(double x, double y, double z) {
    return {
        0.28209479177387814,
        0.4886025119029199 * y,
        0.4886025119029199 * z,
        0.4886025119029199 * x,
        1.0925484305920792 * x * y,
        1.0925484305920792 * y * z,
        0.31539156525252005 * (3.0 * z * z - 1.0),
        1.0925484305920792 * x * z,
        0.5462742152960396 * (x * x - y * y),
    };
}

}  // namespace

EnvironmentSH computeEnvironmentSH(const Environment& environment, int sampleCount) {
    EnvironmentSH result;
    const int count = std::max(sampleCount, 256);
    const double goldenAngle = kPi * (3.0 - std::sqrt(5.0));
    std::array<Vec3d, 9> integral{};
    for (int i = 0; i < count; ++i) {
        const double z = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(count);
        const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double phi = goldenAngle * static_cast<double>(i);
        const Vec3d direction = {radius * std::cos(phi), radius * std::sin(phi), z};
        const std::array<double, 9> basis = shBasis(direction[0], direction[1], direction[2]);
        const Vec3d radiance = evaluateEnvironmentDouble(environment, direction);
        for (int band = 0; band < 9; ++band) {
            integral[static_cast<size_t>(band)][0] += radiance[0] * basis[static_cast<size_t>(band)];
            integral[static_cast<size_t>(band)][1] += radiance[1] * basis[static_cast<size_t>(band)];
            integral[static_cast<size_t>(band)][2] += radiance[2] * basis[static_cast<size_t>(band)];
        }
    }

    const double weight = 4.0 * kPi / static_cast<double>(count);
    const std::array<double, 3> bandFactors = {
        kPi,
        2.0 * kPi / 3.0,
        kPi / 4.0,
    };
    for (int index = 0; index < 9; ++index) {
        const int band = index == 0 ? 0 : index <= 3 ? 1 : 2;
        const Vec3d radiance = {
            integral[static_cast<size_t>(index)][0] * weight,
            integral[static_cast<size_t>(index)][1] * weight,
            integral[static_cast<size_t>(index)][2] * weight,
        };
        result.radiance[static_cast<size_t>(index)] =
            {static_cast<float>(radiance[0]), static_cast<float>(radiance[1]), static_cast<float>(radiance[2])};
        const Vec3d irradiance = {
            radiance[0] * bandFactors[static_cast<size_t>(band)],
            radiance[1] * bandFactors[static_cast<size_t>(band)],
            radiance[2] * bandFactors[static_cast<size_t>(band)],
        };
        result.irradiance[static_cast<size_t>(index)] = {
            static_cast<float>(irradiance[0]),
            static_cast<float>(irradiance[1]),
            static_cast<float>(irradiance[2]),
        };
    }
    return result;
}

Vec3 evaluateEnvironment(const Environment& environment, const Vec3& direction) {
    const Vec3 dir = normalize(direction);
    const Vec3d value = evaluateEnvironmentDouble(environment, {dir.x, dir.y, dir.z});
    return {static_cast<float>(value[0]), static_cast<float>(value[1]), static_cast<float>(value[2])};
}

}  // namespace pbr
