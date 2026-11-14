#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "baker/error.h"

namespace baker {

constexpr std::uint32_t kSceneVersion = 1;
constexpr std::uint32_t kOutputVersion = 1;
constexpr std::uint32_t kSnapshotVersion = 2;
constexpr std::uint32_t kLegacySnapshotVersion = 1;
constexpr std::uint32_t kSceneMagic = 0x43534251u;  // bytes "QBSC" when stored LE
constexpr std::uint32_t kOutputMagic = 0x41425251u;  // bytes "QRBA" when stored LE
constexpr std::uint32_t kSnapshotMagic = 0x504e5351u;  // bytes "QSNP" when stored LE

constexpr std::uint32_t kMaxDimension = 4096;
constexpr std::uint32_t kMaxTextureSize = 512;
constexpr std::uint32_t kMaxLayers = 256;
constexpr std::uint32_t kMaxKeys = 4096;
constexpr std::uint64_t kMaxPixels = 64ull * 1024ull * 1024ull;
constexpr std::uint64_t kMaxSceneBytes = 512ull * 1024ull * 1024ull;

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

enum class Playback : std::uint32_t {
    kClamp = 0,
    kLoop = 1,
};

struct VKey {
    float time = 0.0f;
    std::array<std::array<float, 2>, 4> displacement{};
};

struct OKey {
    float time = 0.0f;
    float opacity = 1.0f;
};

struct Layer {
    std::vector<VKey> vertex_keys;
    std::vector<OKey> opacity_keys;
    Playback playback = Playback::kClamp;
    std::array<Vec3, 4> base{};
    std::array<Vec2, 4> uv{};
    Vec3 albedo{1.0f, 1.0f, 1.0f};
    float roughness = 0.5f;
    float specular = 0.5f;
    bool premultiplied = false;
};

struct Scene {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t tex_size = 0;
    std::vector<float> texels;  // tex_size * tex_size * RGBA
    std::vector<Layer> layers;
    std::uint64_t input_hash = 0;
};

struct Stats {
    std::vector<std::uint64_t> layer_pixels;
    std::uint32_t degenerate_layers = 0;
};

struct PixelBuffer {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> rgba;  // premultiplied linear-ish RGBA, row-major
};

std::uint64_t fnv1a64(std::span<const std::uint8_t> bytes) noexcept;
bool is_finite(float value) noexcept;
bool nearly_equal(float lhs, float rhs) noexcept;

}  // namespace baker
