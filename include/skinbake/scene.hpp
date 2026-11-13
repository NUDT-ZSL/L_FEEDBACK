#pragma once

#include "skinbake/curve.hpp"
#include "skinbake/math.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace skinbake {

enum Channel : std::size_t {
    TranslationX = 0,
    TranslationY = 1,
    Rotation = 2,
    ScaleX = 3,
    ScaleY = 4,
    ChannelCount = 5
};

struct Bone {
    std::string name;
    int parent = -1;
};

struct BoneCurves {
    std::array<Curve, ChannelCount> channels;
};

struct Clip {
    std::string name;
    std::int64_t durationMs = 0;
    bool loop = true;
    std::vector<BoneCurves> bones;
};

struct Vertex {
    Vec2 position;
    std::array<int, 4> boneIndices{{-1, -1, -1, -1}};
    std::array<double, 4> weights{{0.0, 0.0, 0.0, 0.0}};
    std::uint32_t influenceCount = 0;
};

struct Scene {
    std::vector<Bone> bones;
    std::vector<Vertex> vertices;
    std::vector<Clip> clips;
    std::string defaultClip;
    std::uint32_t fps = 60;
};

struct PreparedScene {
    Scene scene;
    std::vector<int> topologicalOrder;
    std::unordered_map<std::string, std::size_t> clipByName;
    std::uint64_t fingerprint = 0;
};

std::shared_ptr<const PreparedScene> prepare_scene(const Scene& scene);

std::uint64_t scene_fingerprint(const Scene& scene);
void validate_scene(const Scene& scene);

std::size_t clip_index(const PreparedScene& prepared, const std::string& name);

std::int64_t rounded_frame_time_ms(std::uint32_t frameIndex, std::uint32_t fps);
std::int64_t wrap_playback_time(std::int64_t absoluteTimeMs,
                                std::int64_t durationMs, bool loop);

}

