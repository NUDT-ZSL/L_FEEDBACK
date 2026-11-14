#include "baker/scene_generator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "baker/byte_io.h"
#include "baker/model.h"

namespace baker {
namespace {

class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next() noexcept {
        state_ += 0x9e3779b97f4a7c15ull;
        std::uint64_t value = state_;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
        return value ^ (value >> 31);
    }

    float unit() noexcept {
        const std::uint64_t bits = next() >> 40;
        return static_cast<float>(static_cast<double>(bits) * (1.0 / 16777216.0));
    }

    float range(float low, float high) noexcept {
        return low + (high - low) * unit();
    }

private:
    std::uint64_t state_;
};

void append_vec3(std::vector<std::uint8_t>& out, float x, float y, float z) {
    append_f32(out, x);
    append_f32(out, y);
    append_f32(out, z);
}

void append_vec2(std::vector<std::uint8_t>& out, float x, float y) {
    append_f32(out, x);
    append_f32(out, y);
}

void append_vkey(std::vector<std::uint8_t>& out,
                 float time,
                 const std::array<std::array<float, 2>, 4>& displacement) {
    append_f32(out, time);
    for (const auto& vertex : displacement) {
        append_f32(out, vertex[0]);
        append_f32(out, vertex[1]);
    }
}

}  // namespace

std::vector<std::uint8_t> generate_scene_bytes(const GeneratorOptions& options) {
    const GeneratorOptions safe{
        std::clamp(options.width, 1u, kMaxDimension),
        std::clamp(options.height, 1u, kMaxDimension),
        std::clamp(options.layers, 1u, kMaxLayers),
        std::clamp(options.tex_size, 1u, kMaxTextureSize),
        options.seed,
    };

    std::vector<std::uint8_t> out;
    out.reserve(24 + static_cast<std::size_t>(safe.tex_size) * safe.tex_size * 16 +
                static_cast<std::size_t>(safe.layers) * 512);

    append_u32(out, kSceneMagic);
    append_u32(out, kSceneVersion);
    append_u32(out, safe.width);
    append_u32(out, safe.height);
    append_u32(out, safe.layers);
    append_u32(out, safe.tex_size);

    SplitMix64 random(safe.seed ^ 0x51c3a9f4d2e607b1ull);
    for (std::uint32_t i = 0; i < safe.tex_size * safe.tex_size; ++i) {
        append_f32(out, random.unit());  // albedo noise R
        append_f32(out, random.unit());  // albedo noise G
        append_f32(out, random.unit());  // albedo noise B
        append_f32(out, random.unit());  // alpha noise A
    }

    for (std::uint32_t layer_index = 0; layer_index < safe.layers; ++layer_index) {
        const bool background = layer_index == 0;
        const bool degenerate = safe.layers > 1 && layer_index + 1 == safe.layers;
        const std::uint32_t vertex_key_count = background ? 1u : 3u;
        const std::uint32_t opacity_key_count = background ? 1u : 2u;
        const std::uint32_t playback =
            (layer_index % 3 == 2) ? static_cast<std::uint32_t>(Playback::kLoop)
                                   : static_cast<std::uint32_t>(Playback::kClamp);

        append_u32(out, vertex_key_count);
        for (std::uint32_t key_index = 0; key_index < vertex_key_count; ++key_index) {
            const float time = vertex_key_count == 1
                                   ? 0.0f
                                   : static_cast<float>(key_index) /
                                         static_cast<float>(vertex_key_count - 1);
            std::array<std::array<float, 2>, 4> displacement{};
            if (!background && !degenerate) {
                for (auto& vertex : displacement) {
                    vertex[0] = random.range(-0.32f, 0.32f);
                    vertex[1] = random.range(-0.32f, 0.32f);
                }
            }
            append_vkey(out, time, displacement);
        }

        append_u32(out, opacity_key_count);
        for (std::uint32_t key_index = 0; key_index < opacity_key_count; ++key_index) {
            const float time = opacity_key_count == 1
                                   ? 0.0f
                                   : static_cast<float>(key_index) /
                                         static_cast<float>(opacity_key_count - 1);
            const float opacity = opacity_key_count == 1
                                      ? 1.0f
                                      : (key_index == 0 ? random.range(0.55f, 0.95f)
                                                        : random.range(0.45f, 0.90f));
            append_f32(out, time);
            append_f32(out, opacity);
        }
        append_u32(out, playback);

        float cx = 0.0f;
        float cy = 0.0f;
        float half = 0.0f;
        float z = 0.0f;
        if (background) {
            half = 7.0f;
            z = 4.2f;
        } else if (degenerate) {
            half = 1.5f;
            z = 2.0f;
        } else {
            cx = random.range(-1.5f, 1.5f);
            cy = random.range(-1.0f, 1.0f);
            half = random.range(0.75f, 1.75f);
            z = random.range(1.6f, 3.7f);
        }

        if (degenerate) {
            append_vec3(out, -half, -0.5f * half, z);
            append_vec3(out, 0.0f, 0.0f, z);
            append_vec3(out, half, 0.5f * half, z);
            append_vec3(out, 2.0f * half, half, z);
        } else {
            append_vec3(out, cx - half, cy - half, z);
            append_vec3(out, cx + half, cy - half, z);
            append_vec3(out, cx + half, cy + half, z);
            append_vec3(out, cx - half, cy + half, z);
        }

        append_vec2(out, 0.0f, 0.0f);
        append_vec2(out, 1.4f, 0.0f);
        append_vec2(out, 1.4f, 1.4f);
        append_vec2(out, 0.0f, 1.4f);

        if (degenerate) {
            append_vec3(out, 1.0f, 0.05f, 0.05f);
        } else if (background) {
            append_vec3(out, 0.18f, 0.20f, 0.26f);
        } else {
            append_vec3(out, random.range(0.28f, 0.92f), random.range(0.22f, 0.92f),
                        random.range(0.22f, 0.98f));
        }
        append_f32(out, degenerate ? 0.5f : random.range(0.08f, 0.92f));
        append_f32(out, degenerate ? 0.5f : random.range(0.02f, 0.76f));
        append_u32(out, (layer_index % 3 == 2) ? 1u : 0u);
    }

    return out;
}

}  // namespace baker
