#include "sample_generator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace cr {
namespace {

constexpr int kSampleFormatVersion = 4;
constexpr float kPi = 3.14159265358979323846F;

enum class LayerPattern {
    SkinBase,
    SkinBlush,
    SkinFreckles,
    ClothBase,
    ClothEmbroidery,
    ClothGrime,
    MetalBase,
    MetalTrim,
    MetalPatina,
    HairBase,
    HairSheen,
    EyeBase,
    EyeGlint,
};

struct GeneratedLayer {
    std::string name;
    std::string blendMode;
    float opacity = 1.0F;
    LayerPattern pattern = LayerPattern::SkinBase;
    Vec3 colorA;
    Vec3 colorB;
    float metallic = 0.0F;
    float roughness = 0.5F;
    float normalStrength = 1.0F;
    std::uint32_t seed = 1;
};

struct GeneratedMaterial {
    std::string name;
    std::vector<GeneratedLayer> layers;
};

struct LayerSample {
    Vec3 color;
    float coverage = 1.0F;
    float metallic = 0.0F;
    float roughness = 0.5F;
    float height = 0.0F;
};

std::uint32_t hashU32(std::uint32_t x) {
    x ^= x >> 16U;
    x *= 0x7FEB352DU;
    x ^= x >> 15U;
    x *= 0x846CA68BU;
    x ^= x >> 16U;
    return x;
}

float hash01(std::uint32_t x, std::uint32_t y, std::uint32_t seed) {
    const std::uint32_t value =
        hashU32(x + 0x9E3779B9U) ^ hashU32(y + 0x85EBCA6BU) ^ hashU32(seed + 0xC2B2AE35U);
    return static_cast<float>(value & 0x00FFFFFFU) / 16777215.0F;
}

float smoothStep(float edge0, float edge1, float value) {
    const float t = clamp((value - edge0) / (edge1 - edge0), 0.0F, 1.0F);
    return t * t * (3.0F - 2.0F * t);
}

float valueNoise(float u, float v, float frequency, std::uint32_t seed) {
    const float x = u * frequency;
    const float y = v * frequency;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float sx = tx * tx * (3.0F - 2.0F * tx);
    const float sy = ty * ty * (3.0F - 2.0F * ty);
    const auto ix = static_cast<std::uint32_t>(x0);
    const auto iy = static_cast<std::uint32_t>(y0);
    const float a = hash01(ix, iy, seed);
    const float b = hash01(ix + 1U, iy, seed);
    const float c = hash01(ix, iy + 1U, seed);
    const float d = hash01(ix + 1U, iy + 1U, seed);
    return mix(mix(a, b, sx), mix(c, d, sx), sy);
}

float fbm(float u, float v, float frequency, std::uint32_t seed) {
    float value = 0.0F;
    float amplitude = 0.55F;
    float total = 0.0F;
    for (int octave = 0; octave < 4; ++octave) {
        value += amplitude * valueNoise(u, v, frequency, seed + static_cast<std::uint32_t>(octave) * 101U);
        total += amplitude;
        frequency *= 2.03F;
        amplitude *= 0.5F;
    }
    return total > 0.0F ? value / total : 0.0F;
}

float radialMask(float u, float v, float centerU, float centerV, float radius) {
    const float du = u - centerU;
    const float dv = (v - centerV) * 0.9F;
    const float distance = std::sqrt(du * du + dv * dv);
    return 1.0F - smoothStep(radius * 0.3F, radius, distance);
}

float threadMask(float u, float v, float frequency, std::uint32_t seed) {
    const float diagonal =
        0.5F + 0.5F * std::sin((u + v) * frequency * 2.0F * kPi +
                               valueNoise(u, v, 18.0F, seed) * 2.0F);
    const float crossHat =
        0.5F + 0.5F * std::sin((u - v) * frequency * 2.0F * kPi);
    return clamp(diagonal * 0.75F + crossHat * 0.25F, 0.0F, 1.0F);
}

Vec3 tint(const Vec3& a, const Vec3& b, float t) {
    return mix(a, b, clamp(t, 0.0F, 1.0F));
}

LayerSample evaluateLayer(const GeneratedLayer& layer, float u, float v) {
    const float fineNoise = fbm(u, v, 42.0F, layer.seed);
    const float broadNoise = fbm(u, v, 8.0F, layer.seed ^ 0xA53C9E1U);
    LayerSample result;
    result.metallic = layer.metallic;
    result.roughness = layer.roughness;

    switch (layer.pattern) {
        case LayerPattern::SkinBase: {
            const float pores = fbm(u, v, 180.0F, layer.seed);
            const float veins = valueNoise(u, v, 22.0F, layer.seed ^ 0x71U);
            result.color = tint(layer.colorA, layer.colorB, broadNoise * 0.35F + veins * 0.15F);
            result.color = result.color * (0.94F + pores * 0.10F);
            result.coverage = 1.0F;
            result.height = (pores - 0.5F) * 0.8F + (fineNoise - 0.5F) * 0.25F;
            break;
        }
        case LayerPattern::SkinBlush: {
            const float cheeks =
                std::max(radialMask(u, v, 0.34F, 0.55F, 0.20F),
                         radialMask(u, v, 0.66F, 0.55F, 0.20F));
            result.color = tint(layer.colorA, layer.colorB, fineNoise * 0.4F);
            result.coverage = clamp(cheeks * (0.60F + 0.40F * fineNoise), 0.0F, 1.0F);
            result.roughness = layer.roughness + (fineNoise - 0.5F) * 0.08F;
            result.height = (fineNoise - 0.5F) * 0.2F;
            break;
        }
        case LayerPattern::SkinFreckles: {
            const float cellSize = 52.0F;
            const int cellX = static_cast<int>(std::floor(u * cellSize));
            const int cellY = static_cast<int>(std::floor(v * cellSize));
            const float localU = u * cellSize - static_cast<float>(cellX);
            const float localV = v * cellSize - static_cast<float>(cellY);
            const float centerU = 0.2F + 0.6F * hash01(static_cast<std::uint32_t>(cellX),
                                                        static_cast<std::uint32_t>(cellY),
                                                        layer.seed);
            const float centerV = 0.2F + 0.6F * hash01(static_cast<std::uint32_t>(cellY),
                                                        static_cast<std::uint32_t>(cellX),
                                                        layer.seed ^ 0xBEEFU);
            const float distance =
                std::sqrt((localU - centerU) * (localU - centerU) +
                          (localV - centerV) * (localV - centerV));
            const float freckle = 1.0F - smoothStep(0.035F, 0.085F, distance);
            result.color = tint(layer.colorA, layer.colorB, fineNoise);
            result.coverage = freckle *
                              smoothStep(0.18F, 0.32F, v) *
                              (1.0F - smoothStep(0.68F, 0.80F, v));
            result.height = freckle * 0.25F;
            break;
        }
        case LayerPattern::ClothBase: {
            const float weave = threadMask(u, v, 58.0F, layer.seed);
            const float slub = fbm(u, v, 33.0F, layer.seed ^ 0x4A5U);
            result.color = tint(layer.colorA, layer.colorB, broadNoise * 0.5F + slub * 0.25F);
            result.color = result.color * (0.86F + weave * 0.22F);
            result.coverage = 1.0F;
            result.height = (weave - 0.5F) * 0.9F + (slub - 0.5F) * 0.35F;
            break;
        }
        case LayerPattern::ClothEmbroidery: {
            const float lineA =
                1.0F - smoothStep(0.018F, 0.052F,
                                  std::abs(std::sin((u + v) * kPi * 10.0F)));
            const float lineB =
                1.0F - smoothStep(0.012F, 0.040F,
                                  std::abs(std::sin((u - v) * kPi * 8.0F + 0.7F)));
            const float diamond = std::max(lineA, lineB * 0.75F);
            result.color = tint(layer.colorA, layer.colorB, fineNoise * 0.35F);
            result.coverage = clamp(diamond * (0.65F + 0.35F * fineNoise), 0.0F, 1.0F);
            result.metallic = layer.metallic * (0.75F + 0.25F * broadNoise);
            result.roughness = layer.roughness + (fineNoise - 0.5F) * 0.12F;
            result.height = diamond * 0.8F;
            break;
        }
        case LayerPattern::ClothGrime: {
            const float grime = smoothStep(0.43F, 0.76F, fbm(u, v, 13.0F, layer.seed));
            result.color = tint(layer.colorA, layer.colorB, fineNoise * 0.5F);
            result.coverage = grime * (0.55F + 0.45F * fineNoise);
            result.roughness = layer.roughness + fineNoise * 0.1F;
            result.height = (fineNoise - 0.5F) * 0.18F;
            break;
        }
        case LayerPattern::MetalBase: {
            const float brushed =
                0.5F + 0.5F * std::sin(v * 1500.0F + valueNoise(u, v, 30.0F, layer.seed) * 8.0F);
            const float scratches =
                smoothStep(0.72F, 0.95F, valueNoise(u * 0.3F, v * 7.0F, 1.0F, layer.seed));
            result.color = tint(layer.colorA, layer.colorB, broadNoise * 0.35F + brushed * 0.18F);
            result.metallic = clamp(layer.metallic - scratches * 0.10F, 0.0F, 1.0F);
            result.roughness = clamp(layer.roughness + brushed * 0.08F + scratches * 0.18F,
                                     0.03F, 1.0F);
            result.coverage = 1.0F;
            result.height = (brushed - 0.5F) * 0.35F + (fineNoise - 0.5F) * 0.1F;
            break;
        }
        case LayerPattern::MetalTrim: {
            const float border =
                1.0F - smoothStep(0.025F, 0.055F,
                                  std::abs(std::sin((u * 7.0F + v * 3.0F) * kPi)));
            const float channels =
                1.0F - smoothStep(0.020F, 0.042F,
                                  std::abs(std::sin((u - v * 2.0F) * kPi * 12.0F)));
            const float pattern = std::max(border, channels * 0.55F);
            result.color = tint(layer.colorA, layer.colorB, fineNoise * 0.35F);
            result.coverage = clamp(pattern * (0.75F + broadNoise * 0.25F), 0.0F, 1.0F);
            result.metallic = layer.metallic;
            result.roughness = layer.roughness + pattern * 0.04F;
            result.height = pattern * 0.9F;
            break;
        }
        case LayerPattern::MetalPatina: {
            const float patina = smoothStep(0.40F, 0.72F, fbm(u, v, 12.0F, layer.seed));
            result.color = tint(layer.colorA, layer.colorB, fineNoise * 0.6F);
            result.coverage = patina * (0.45F + 0.55F * broadNoise);
            result.metallic = layer.metallic * (1.0F - patina * 0.4F);
            result.roughness = layer.roughness + patina * 0.12F;
            result.height = (fineNoise - 0.5F) * 0.3F;
            break;
        }
        case LayerPattern::HairBase: {
            const float strands =
                0.5F + 0.5F * std::sin((u * 120.0F + std::sin(v * 28.0F) * 2.1F) * kPi);
            result.color = tint(layer.colorA, layer.colorB, broadNoise * 0.45F + strands * 0.2F);
            result.coverage = 1.0F;
            result.roughness = layer.roughness + strands * 0.10F;
            result.height = (strands - 0.5F) * 0.75F;
            break;
        }
        case LayerPattern::HairSheen: {
            const float strand =
                std::pow(0.5F + 0.5F * std::sin((u * 95.0F + v * 5.0F) * kPi * 2.0F), 4.0F);
            result.color = tint(layer.colorA, layer.colorB, fineNoise * 0.3F);
            result.coverage = strand * (0.45F + broadNoise * 0.55F);
            result.height = strand * 0.2F;
            break;
        }
        case LayerPattern::EyeBase: {
            const float iris =
                smoothStep(0.28F, 0.50F, radialMask(u, v, 0.5F, 0.5F, 0.46F));
            result.color = tint(layer.colorA, layer.colorB, iris * 0.45F + fineNoise * 0.08F);
            result.roughness = layer.roughness + fineNoise * 0.06F;
            result.coverage = 1.0F;
            result.height = iris * 0.15F;
            break;
        }
        case LayerPattern::EyeGlint: {
            const float glintA = radialMask(u, v, 0.38F, 0.40F, 0.13F);
            const float glintB = radialMask(u, v, 0.64F, 0.62F, 0.055F);
            result.color = tint(layer.colorA, layer.colorB, fineNoise * 0.15F);
            result.coverage = clamp(glintA + glintB * 0.7F, 0.0F, 1.0F);
            result.roughness = layer.roughness;
            result.height = result.coverage * 0.25F;
            break;
        }
    }
    result.color = clamp(result.color, 0.0F, 1.0F);
    result.metallic = clamp(result.metallic, 0.0F, 1.0F);
    result.roughness = clamp(result.roughness, 0.02F, 1.0F);
    result.coverage = clamp(result.coverage, 0.0F, 1.0F);
    return result;
}

std::uint8_t toByte(float value) {
    return static_cast<std::uint8_t>(
        std::lround(clamp(value, 0.0F, 1.0F) * 255.0F));
}

void setPixel(Image& image, int x, int y, float r, float g, float b, float a) {
    std::uint8_t* pixel =
        image.pixels.data() +
        (static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width) +
         static_cast<std::size_t>(x)) *
            4U;
    pixel[0] = toByte(r);
    pixel[1] = toByte(g);
    pixel[2] = toByte(b);
    pixel[3] = toByte(a);
}

std::array<Image, 4> generateLayerTextures(const GeneratedLayer& layer, int size) {
    std::array<Image, 4> images = {
        Image(size, size, 4), Image(size, size, 4), Image(size, size, 4),
        Image(size, size, 4)};
    const float inverseSize = 1.0F / static_cast<float>(size);
    for (int y = 0; y < size; ++y) {
        const float v = (static_cast<float>(y) + 0.5F) * inverseSize;
        for (int x = 0; x < size; ++x) {
            const float u = (static_cast<float>(x) + 0.5F) * inverseSize;
            const LayerSample sample = evaluateLayer(layer, u, v);
            const LayerSample left =
                evaluateLayer(layer, u - inverseSize, v);
            const LayerSample right =
                evaluateLayer(layer, u + inverseSize, v);
            const LayerSample up =
                evaluateLayer(layer, u, v - inverseSize);
            const LayerSample down =
                evaluateLayer(layer, u, v + inverseSize);
            const float dx = (right.height - left.height) * layer.normalStrength * 2.2F;
            const float dy = (down.height - up.height) * layer.normalStrength * 2.2F;
            const Vec3 normal = normalize(Vec3(-dx, -dy, 1.0F));
            const float metalNoise = (fbm(u, v, 88.0F, layer.seed ^ 0x91U) - 0.5F) *
                                     (layer.pattern == LayerPattern::SkinBase ? 0.01F : 0.055F);
            const float roughNoise = (fbm(u, v, 70.0F, layer.seed ^ 0x31U) - 0.5F) * 0.11F;
            setPixel(images[0], x, y, sample.color.x, sample.color.y, sample.color.z,
                     sample.coverage);
            setPixel(images[1], x, y, sample.metallic + metalNoise,
                     sample.metallic + metalNoise, sample.metallic + metalNoise,
                     sample.coverage);
            setPixel(images[2], x, y, sample.roughness + roughNoise,
                     sample.roughness + roughNoise, sample.roughness + roughNoise,
                     sample.coverage);
            setPixel(images[3], x, y, normal.x * 0.5F + 0.5F, normal.y * 0.5F + 0.5F,
                     normal.z * 0.5F + 0.5F, sample.coverage);
        }
    }
    return images;
}

std::vector<GeneratedMaterial> buildMaterialDefinitions() {
    return {
        {"skin",
         {
             {"skin_base", "override", 1.0F, LayerPattern::SkinBase,
              Vec3(0.58F, 0.35F, 0.25F), Vec3(0.74F, 0.50F, 0.37F), 0.01F, 0.52F,
              0.22F, 0x1201U},
             {"skin_blush", "linear", 0.42F, LayerPattern::SkinBlush,
              Vec3(0.78F, 0.30F, 0.27F), Vec3(0.92F, 0.49F, 0.40F), 0.0F, 0.46F,
              0.25F, 0x1202U},
             {"skin_freckles", "mask", 0.85F, LayerPattern::SkinFreckles,
              Vec3(0.35F, 0.18F, 0.11F), Vec3(0.49F, 0.29F, 0.18F), 0.0F, 0.62F,
              0.10F, 0x1203U},
         }},
        {"cloth",
         {
             {"cloth_base", "override", 1.0F, LayerPattern::ClothBase,
              Vec3(0.045F, 0.10F, 0.22F), Vec3(0.10F, 0.20F, 0.38F), 0.02F, 0.82F,
              0.35F, 0x2201U},
             {"cloth_embroidery", "linear", 0.88F, LayerPattern::ClothEmbroidery,
              Vec3(0.72F, 0.46F, 0.10F), Vec3(1.00F, 0.76F, 0.28F), 0.72F, 0.34F,
              0.45F, 0x2202U},
             {"cloth_grime", "mask", 0.72F, LayerPattern::ClothGrime,
              Vec3(0.10F, 0.12F, 0.10F), Vec3(0.23F, 0.22F, 0.17F), 0.02F, 0.95F,
              0.15F, 0x2203U},
         }},
        {"metal",
         {
             {"metal_base", "override", 1.0F, LayerPattern::MetalBase,
              Vec3(0.42F, 0.48F, 0.56F), Vec3(0.68F, 0.73F, 0.78F), 0.93F, 0.27F,
              0.65F, 0x3201U},
             {"metal_trim", "linear", 0.82F, LayerPattern::MetalTrim,
              Vec3(0.72F, 0.48F, 0.12F), Vec3(1.00F, 0.76F, 0.30F), 0.96F, 0.18F,
              0.70F, 0x3202U},
             {"metal_patina", "mask", 0.78F, LayerPattern::MetalPatina,
              Vec3(0.10F, 0.36F, 0.31F), Vec3(0.24F, 0.54F, 0.42F), 0.20F, 0.72F,
              0.30F, 0x3203U},
         }},
        {"hair",
         {
             {"hair_base", "override", 1.0F, LayerPattern::HairBase,
              Vec3(0.055F, 0.033F, 0.022F), Vec3(0.20F, 0.12F, 0.07F), 0.02F, 0.38F,
              0.35F, 0x4201U},
             {"hair_sheen", "linear", 0.55F, LayerPattern::HairSheen,
              Vec3(0.35F, 0.20F, 0.10F), Vec3(0.68F, 0.44F, 0.23F), 0.04F, 0.24F,
              0.20F, 0x4202U},
         }},
        {"eyes",
         {
             {"eye_base", "override", 1.0F, LayerPattern::EyeBase,
              Vec3(0.01F, 0.025F, 0.028F), Vec3(0.03F, 0.16F, 0.18F), 0.05F, 0.16F,
              0.05F, 0x5201U},
             {"eye_glint", "linear", 0.95F, LayerPattern::EyeGlint,
              Vec3(0.55F, 0.93F, 0.91F), Vec3(1.00F, 1.00F, 1.00F), 0.08F, 0.06F,
              0.05F, 0x5202U},
         }},
    };
}

Json generatedLayerJson(const GeneratedLayer& layer, std::size_t materialIndex,
                        std::size_t layerIndex) {
    std::ostringstream prefix;
    prefix << "../textures/m" << materialIndex << "_l" << layerIndex << "_" << layer.name;
    const std::string base = prefix.str();
    return Json{{"name", layer.name},
                {"baseColor", base + "_basecolor.png"},
                {"metallic", base + "_metallic.png"},
                {"roughness", base + "_roughness.png"},
                {"normal", base + "_normal.png"},
                {"blendMode", layer.blendMode},
                {"opacity", layer.opacity}};
}

void emitSamples(const fs::path& dataDirectory) {
    const fs::path texturesDirectory = dataDirectory / "textures";
    const fs::path materialsDirectory = dataDirectory / "materials";
    const fs::path sceneDirectory = dataDirectory / "scene";
    ensureDirectory(texturesDirectory);
    ensureDirectory(materialsDirectory);
    ensureDirectory(sceneDirectory);

    const std::vector<GeneratedMaterial> materials = buildMaterialDefinitions();
    for (std::size_t materialIndex = 0; materialIndex < materials.size(); ++materialIndex) {
        const GeneratedMaterial& material = materials[materialIndex];
        Json materialJson;
        materialJson["name"] = material.name;
        materialJson["layers"] = Json::array();
        for (std::size_t layerIndex = 0; layerIndex < material.layers.size(); ++layerIndex) {
            const GeneratedLayer& layer = material.layers[layerIndex];
            const std::array<Image, 4> images =
                generateLayerTextures(layer, kMaterialTextureSize);
            std::ostringstream prefix;
            prefix << "m" << materialIndex << "_l" << layerIndex << "_" << layer.name;
            const std::array<const char*, 4> suffixes = {
                "_basecolor.png", "_metallic.png", "_roughness.png", "_normal.png"};
            for (std::size_t channel = 0; channel < images.size(); ++channel) {
                savePngAtomic(texturesDirectory / (prefix.str() + suffixes[channel]),
                              images[channel]);
            }
            materialJson["layers"].push_back(
                generatedLayerJson(layer, materialIndex, layerIndex));
        }
        writeTextAtomic(materialsDirectory / (material.name + ".json"),
                        materialJson.dump(2) + "\n");
    }

    const Json cameraJson = {
        {"viewport", {{"width", kDefaultViewportWidth}, {"height", kDefaultViewportHeight}}},
        {"camera",
         {{"position", vec3ToJson(Vec3(0.0F, 0.08F, 4.80F))},
          {"target", vec3ToJson(Vec3(0.0F, -0.18F, 0.0F))},
          {"up", vec3ToJson(Vec3(0.0F, 1.0F, 0.0F))},
          {"fovYDegrees", 38.0F},
          {"near", 0.1F},
          {"far", 20.0F}}}};
    writeTextAtomic(sceneDirectory / "camera.json", cameraJson.dump(2) + "\n");

    Json lightsArray = Json::array();
    lightsArray.push_back({{"id", "key"},
                           {"position", vec3ToJson(Vec3(-2.4F, 2.1F, 3.0F))},
                           {"color", vec3ToJson(Vec3(1.0F, 0.80F, 0.62F))},
                           {"intensity", 28.0F},
                           {"radius", 10.0F}});
    lightsArray.push_back({{"id", "fill"},
                           {"position", vec3ToJson(Vec3(2.7F, 0.8F, 2.4F))},
                           {"color", vec3ToJson(Vec3(0.46F, 0.66F, 1.0F))},
                           {"intensity", 17.0F},
                           {"radius", 9.0F}});
    lightsArray.push_back({{"id", "rim"},
                           {"position", vec3ToJson(Vec3(1.0F, 2.2F, -3.0F))},
                           {"color", vec3ToJson(Vec3(0.70F, 0.35F, 1.0F))},
                           {"intensity", 24.0F},
                           {"radius", 9.0F}});
    lightsArray.push_back({{"id", "bounce"},
                           {"position", vec3ToJson(Vec3(0.0F, -1.6F, 2.2F))},
                           {"color", vec3ToJson(Vec3(1.0F, 0.42F, 0.18F))},
                           {"intensity", 8.0F},
                           {"radius", 4.5F}});
    const Json lightsJson = {{"lights", lightsArray}};
    writeTextAtomic(sceneDirectory / "lights.json", lightsJson.dump(2) + "\n");

    const Json manifest = {
        {"formatVersion", kSampleFormatVersion},
        {"seed", kSampleSeed},
        {"textureSize", kMaterialTextureSize},
        {"camera", "scene/camera.json"},
        {"defaultLights", "scene/lights.json"},
        {"materials", Json::array({"skin", "cloth", "metal", "hair", "eyes"})}};
    writeTextAtomic(dataDirectory / "samples.manifest.json", manifest.dump(2) + "\n");
}

bool samplesAreCurrent(const fs::path& dataDirectory) {
    const fs::path manifestPath = dataDirectory / "samples.manifest.json";
    if (!pathExists(manifestPath)) {
        return false;
    }
    try {
        const Json manifest = Json::parse(readTextFile(manifestPath));
        return manifest.at("formatVersion").get<int>() == kSampleFormatVersion &&
               manifest.at("seed").get<std::uint64_t>() == kSampleSeed &&
               manifest.at("textureSize").get<int>() == kMaterialTextureSize;
    } catch (const AppError& error) {
        if (std::string(error.code()) == E_IO_DENIED) {
            throw;
        }
        return false;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace

void ensureGeneratedSamples(const fs::path& dataDirectory, bool forceRegenerate) {
    ensureDirectory(dataDirectory);
    if (forceRegenerate) {
        std::error_code error;
        for (const char* child : {"textures", "materials", "scene"}) {
            fs::remove_all(dataDirectory / child, error);
            if (error) {
                throw AppError(error == std::errc::permission_denied ? E_IO_DENIED : E_IO,
                               "cannot reset generated sample directory: " +
                                   error.message());
            }
        }
        std::error_code manifestError;
        fs::remove(dataDirectory / "samples.manifest.json", manifestError);
        if (manifestError) {
            throw AppError(E_IO, "cannot reset sample manifest: " + manifestError.message());
        }
    }
    if (!forceRegenerate && samplesAreCurrent(dataDirectory)) {
        return;
    }
    emitSamples(dataDirectory);
}

Camera loadCamera(const fs::path& dataDirectory) {
    const fs::path path = dataDirectory / "scene" / "camera.json";
    Json document;
    try {
        document = Json::parse(readTextFile(path));
    } catch (const AppError&) {
        throw;
    } catch (const std::exception& error) {
        throw AppError(E_DATA_INVALID,
                       "invalid camera JSON '" + path.string() + "': " + error.what());
    }
    Camera camera;
    camera.width = document.at("viewport").at("width").get<int>();
    camera.height = document.at("viewport").at("height").get<int>();
    camera.position = vec3FromJson(document.at("camera").at("position"), "camera.position");
    camera.target = vec3FromJson(document.at("camera").at("target"), "camera.target");
    camera.up = vec3FromJson(document.at("camera").at("up"), "camera.up");
    camera.fovYDegrees = document.at("camera").at("fovYDegrees").get<float>();
    camera.nearPlane = document.at("camera").at("near").get<float>();
    camera.farPlane = document.at("camera").at("far").get<float>();
    if (camera.width <= 0 || camera.height <= 0 || camera.fovYDegrees <= 0.0F ||
        camera.fovYDegrees >= 179.0F) {
        throw AppError(E_DATA_INVALID, "camera parameters are outside valid ranges");
    }
    return camera;
}

std::vector<Light> loadDefaultLights(const fs::path& dataDirectory) {
    const fs::path path = dataDirectory / "scene" / "lights.json";
    Json document;
    try {
        document = Json::parse(readTextFile(path));
    } catch (const AppError&) {
        throw;
    } catch (const std::exception& error) {
        throw AppError(E_DATA_INVALID,
                       "invalid default lights JSON '" + path.string() + "': " +
                           error.what());
    }
    const Json& lightsJson = document.at("lights");
    if (!lightsJson.is_array() || lightsJson.empty()) {
        throw AppError(E_DATA_INVALID, "default light array must not be empty");
    }
    std::vector<Light> lights;
    for (const Json& lightJson : lightsJson) {
        lights.push_back(lightFromJson(lightJson, "default light"));
    }
    return lights;
}

}  // namespace cr

