#include "noise.h"

#include <cmath>

namespace df {

uint32_t hash3D(int x, int y, int z, uint32_t seed) {
    uint32_t h = seed;
    h ^= static_cast<uint32_t>(x) * 0x8da6b343u;
    h ^= static_cast<uint32_t>(y) * 0xd8163841u;
    h ^= static_cast<uint32_t>(z) * 0xcb1ab31fu;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

float valueNoise3D(const Vec3& p, uint32_t seed) {
    const int xi = static_cast<int>(std::floor(p.x));
    const int yi = static_cast<int>(std::floor(p.y));
    const int zi = static_cast<int>(std::floor(p.z));
    const float fx = p.x - static_cast<float>(xi);
    const float fy = p.y - static_cast<float>(yi);
    const float fz = p.z - static_cast<float>(zi);
    const float u = fx * fx * (3.0f - 2.0f * fx);
    const float v = fy * fy * (3.0f - 2.0f * fy);
    const float w = fz * fz * (3.0f - 2.0f * fz);

    auto lattice = [&](int dx, int dy, int dz) {
        const uint32_t h = hash3D(xi + dx, yi + dy, zi + dz, seed);
        return static_cast<float>(h & 0x00ffffffu) / 16777215.0f;
    };
    const float x00 = lerp(lattice(0, 0, 0), lattice(1, 0, 0), u);
    const float x10 = lerp(lattice(0, 1, 0), lattice(1, 1, 0), u);
    const float x01 = lerp(lattice(0, 0, 1), lattice(1, 0, 1), u);
    const float x11 = lerp(lattice(0, 1, 1), lattice(1, 1, 1), u);
    const float y0 = lerp(x00, x10, v);
    const float y1 = lerp(x01, x11, v);
    return lerp(y0, y1, w);
}

float fbm3D(const Vec3& p, int octaves, uint32_t seed) {
    float value = 0.0f;
    float amplitude = 0.5f;
    Vec3 sample = p;
    for (int i = 0; i < octaves; ++i) {
        value += amplitude * valueNoise3D(sample, seed + static_cast<uint32_t>(i) * 1013u);
        sample = sample * (2.03f) + Vec3{17.1f, 9.2f, 13.7f};
        amplitude *= 0.5f;
    }
    return value;
}

float densityAt(const Vec3& world, float densityScale, float noiseScale) {
    const Vec3 p = world * noiseScale + Vec3{3.7f, 5.1f, 1.9f};
    const float base = fbm3D(p, 4, 0x51ed270bu);
    const float wisps = std::max(0.0f, base - 0.43f) * 2.7f;
    const float height = std::clamp(1.0f - world.y / 10.0f, 0.08f, 1.0f);
    const float floorFade = smoothstep(0.02f, 0.45f, world.y);
    return densityScale * wisps * height * floorFade;
}

} // namespace df
