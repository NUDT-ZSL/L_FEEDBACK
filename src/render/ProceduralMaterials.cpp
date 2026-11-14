#include "indoor/RenderTypes.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace indoor {
namespace {

class TextureRng {
public:
    explicit TextureRng(std::uint64_t seed) : state_(seed ^ 0x6a09e667f3bcc909ULL) {}

    std::uint32_t next() {
        std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return static_cast<std::uint32_t>((z ^ (z >> 31)) >> 32U);
    }

    float unit() {
        return static_cast<float>(next()) / 4294967295.0f;
    }

private:
    std::uint64_t state_;
};

float hashNoise(std::uint32_t x, std::uint32_t y, std::uint32_t seed) {
    std::uint32_t value = x * 374761393U + y * 668265263U + seed * 2246822519U;
    value = (value ^ (value >> 13U)) * 1274126177U;
    value ^= value >> 16U;
    return static_cast<float>(value & 0x00ffffffU) / 16777215.0f;
}

float smoothStep(float edge0, float edge1, float x) {
    const float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float valueNoise(float x, float y, std::uint32_t seed) {
    const float floorX = std::floor(x);
    const float floorY = std::floor(y);
    const auto ix = static_cast<std::int32_t>(floorX);
    const auto iy = static_cast<std::int32_t>(floorY);
    const float fx = x - floorX;
    const float fy = y - floorY;
    const float ux = smoothStep(0.0f, 1.0f, fx);
    const float uy = smoothStep(0.0f, 1.0f, fy);
    const float a = hashNoise(static_cast<std::uint32_t>(ix),
                              static_cast<std::uint32_t>(iy), seed);
    const float b = hashNoise(static_cast<std::uint32_t>(ix + 1),
                              static_cast<std::uint32_t>(iy), seed);
    const float c = hashNoise(static_cast<std::uint32_t>(ix),
                              static_cast<std::uint32_t>(iy + 1), seed);
    const float d = hashNoise(static_cast<std::uint32_t>(ix + 1),
                              static_cast<std::uint32_t>(iy + 1), seed);
    return glm::mix(glm::mix(a, b, ux), glm::mix(c, d, ux), uy);
}

float fbm(float x, float y, std::uint32_t seed, int octaves) {
    float value = 0.0f;
    float amplitude = 0.55f;
    float total = 0.0f;
    for (int octave = 0; octave < octaves; ++octave) {
        value += valueNoise(x, y, seed + static_cast<std::uint32_t>(octave) * 17U) *
                 amplitude;
        total += amplitude;
        x *= 2.03f;
        y *= 2.03f;
        amplitude *= 0.5f;
    }
    return value / std::max(total, 0.0001f);
}

std::uint8_t toByte(float value) {
    return static_cast<std::uint8_t>(
        std::clamp(std::round(value * 255.0f), 0.0f, 255.0f));
}

void store(std::vector<std::uint8_t>& image, int size, int layer, int x, int y,
           glm::vec4 value) {
    const std::size_t index =
        (static_cast<std::size_t>(layer) * static_cast<std::size_t>(size) *
             static_cast<std::size_t>(size) +
         static_cast<std::size_t>(y) * static_cast<std::size_t>(size) +
         static_cast<std::size_t>(x)) *
        4U;
    image[index + 0] = toByte(value.r);
    image[index + 1] = toByte(value.g);
    image[index + 2] = toByte(value.b);
    image[index + 3] = toByte(value.a);
}

float heightSample(int layer, int x, int y, int size, std::uint32_t seed) {
    const float u = static_cast<float>(x) / static_cast<float>(size);
    const float v = static_cast<float>(y) / static_cast<float>(size);
    if (layer == 3) {
        return fbm(u * 38.0f, v * 38.0f, seed, 5);
    }
    return fbm(u * 11.0f, v * 11.0f, seed + 91U, 4);
}

} // namespace

const char* debugViewName(DebugView view) {
    switch (view) {
    case DebugView::Final:
        return "Final";
    case DebugView::WorldPosition:
        return "F2 WorldPos";
    case DebugView::ViewNormal:
        return "F3 ViewNormal";
    case DebugView::Albedo:
        return "F4 Albedo";
    case DebugView::RoughMetal:
        return "F5 RoughMetal";
    case DebugView::ShadowVersions:
        return "F6 ShadowVersions";
    case DebugView::CacheOccupancy:
        return "F7 CacheOccupancy";
    default:
        return "?";
    }
}

MaterialTextureData generateMaterialTextures(std::uint64_t seed) {
    constexpr int size = kMaterialTextureSize;
    constexpr int layers = kMaterialTextureLayers;
    MaterialTextureData data;
    data.size = size;
    data.layers = layers;
    data.rgba.resize(static_cast<std::size_t>(size) * size * layers * 4U);
    TextureRng rng(seed);

    for (int layer = 0; layer < layers; ++layer) {
        const std::uint32_t layerSeed =
            static_cast<std::uint32_t>(seed) ^ static_cast<std::uint32_t>(layer * 1013 + 7);
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                const float u = static_cast<float>(x) / static_cast<float>(size);
                const float v = static_cast<float>(y) / static_cast<float>(size);
                glm::vec4 value(0.5f, 0.5f, 0.5f, 1.0f);
                if (layer <= 2) {
                    const float broad = fbm(u * 7.0f, v * 7.0f, layerSeed, 4);
                    const float fine = fbm(u * 41.0f, v * 41.0f,
                                           layerSeed + 23U, 3);
                    if (layer == 0) {
                        const float rings = 0.5f + 0.5f *
                            std::sin((v * 31.0f + broad * 4.0f) * 2.0f);
                        value = glm::vec4(
                            0.23f + broad * 0.35f + rings * 0.12f,
                            0.12f + broad * 0.23f + rings * 0.07f,
                            0.045f + broad * 0.13f, 1.0f);
                    } else if (layer == 1) {
                        const float aggregate = std::pow(broad, 2.2f) * 0.32f +
                                                fine * 0.18f;
                        value = glm::vec4(
                            0.34f + aggregate, 0.37f + aggregate,
                            0.40f + aggregate, 1.0f);
                    } else {
                        const float brushed = 0.5f + 0.5f *
                            std::sin((u * 110.0f + broad * 2.5f) * 2.0f);
                        value = glm::vec4(
                            0.08f + brushed * 0.09f + broad * 0.05f,
                            0.21f + brushed * 0.13f + broad * 0.08f,
                            0.34f + brushed * 0.18f + broad * 0.12f, 1.0f);
                    }
                    value.a = 0.82f + broad * 0.18f;
                } else if (layer <= 4) {
                    const float strength = layer == 3 ? 1.0f : 2.25f;
                    const float hx =
                        heightSample(layer, (x + 1) % size, y, size, layerSeed) -
                        heightSample(layer, (x + size - 1) % size, y, size, layerSeed);
                    const float hy =
                        heightSample(layer, x, (y + 1) % size, size, layerSeed) -
                        heightSample(layer, x, (y + size - 1) % size, size, layerSeed);
                    glm::vec3 normal(-hx * 9.0f * strength,
                                     -hy * 9.0f * strength, 1.0f);
                    normal = glm::normalize(normal);
                    value = glm::vec4(normal * 0.5f + 0.5f, 1.0f);
                } else {
                    const float fine = fbm(u * 55.0f, v * 55.0f,
                                           layerSeed, 3);
                    const float broad = fbm(u * 9.0f, v * 9.0f,
                                            layerSeed + 11U, 4);
                    const float scratches = std::pow(
                        std::max(0.0f, std::sin((u * 12.0f + v * 2.0f +
                                                 fine * 2.0f) * 2.0f)), 18.0f);
                    const float roughness = 0.38f + broad * 0.42f +
                                            scratches * 0.15f;
                    const float metal = 0.06f + std::pow(fine, 2.0f) * 0.28f +
                                        (layer == 7 ? broad * 0.55f : 0.0f);
                    value = glm::vec4(roughness, metal,
                                      0.8f + fine * 0.2f, 1.0f);
                }
                if (layer <= 2) {
                    value.r += (rng.unit() - 0.5f) * 0.025f;
                    value.g += (rng.unit() - 0.5f) * 0.025f;
                    value.b += (rng.unit() - 0.5f) * 0.025f;
                }
                store(data.rgba, size, layer, x, y, value);
            }
        }
    }
    return data;
}

std::uint64_t pixelDigest(const std::vector<std::uint8_t>& pixels) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::uint8_t byte : pixels) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

int countPixelDifferences(const std::vector<std::uint8_t>& a,
                          const std::vector<std::uint8_t>& b) {
    const std::size_t length = std::min(a.size(), b.size());
    int differences = static_cast<int>(a.size() > b.size() ? a.size() - b.size() : 0);
    differences += static_cast<int>(b.size() > a.size() ? b.size() - a.size() : 0);
    for (std::size_t i = 0; i + 3 < length; i += 4) {
        if (std::memcmp(&a[i], &b[i], 4) != 0) {
            ++differences;
        }
    }
    return differences;
}

} // namespace indoor
