#include "skinbake/scenarios.hpp"

#include "skinbake/error.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace skinbake {

namespace {

Curve make_curve(std::int64_t durationMs, double base, double amplitude,
                 double skew) {
    const std::int64_t mid = durationMs / 2;
    std::vector<Keyframe> keys;
    keys.resize(3);
    keys[0].timeMs = 0;
    keys[0].value = base + amplitude;
    keys[1].timeMs = mid;
    keys[1].value = base - amplitude * (0.55 + skew);
    keys[2].timeMs = durationMs;
    keys[2].value = base + amplitude * (0.20 - skew * 0.25);
    for (std::size_t i = 0; i + 1 < keys.size(); ++i) {
        const double dt =
            static_cast<double>(keys[i + 1].timeMs - keys[i].timeMs);
        const double delta = keys[i + 1].value - keys[i].value;
        keys[i].outTangent = Tangent{dt / 3.0, delta * 0.45};
        keys[i + 1].inTangent = Tangent{dt / 3.0, delta * 0.35};
    }
    return Curve(std::move(keys));
}

Clip make_clip(const std::string& name, std::int64_t durationMs,
               std::size_t boneCount, double direction) {
    Clip clip;
    clip.name = name;
    clip.durationMs = durationMs;
    clip.loop = true;
    clip.bones.resize(boneCount);
    for (std::size_t bone = 0; bone < boneCount; ++bone) {
        const double boneNumber = static_cast<double>(bone);
        const double baseX = 0.0;
        const double baseY = bone == 0 ? 0.0 : 18.0;
        const double baseRotation = -4.0 + 1.35 * boneNumber * direction;
        const double baseScale = 1.0;
        const double skew = 0.035 * static_cast<double>(bone % 3);

        clip.bones[bone].channels[TranslationX] =
            make_curve(durationMs, baseX,
                       1.1 + 0.17 * boneNumber, skew * direction);
        clip.bones[bone].channels[TranslationY] =
            make_curve(durationMs, baseY,
                       0.45 + 0.11 * boneNumber, -skew * direction);
        clip.bones[bone].channels[Rotation] =
            make_curve(durationMs, baseRotation,
                       1.6 + 0.22 * boneNumber, skew);
        clip.bones[bone].channels[ScaleX] =
            make_curve(durationMs, baseScale,
                       0.025 + 0.002 * boneNumber, skew * 0.5);
        clip.bones[bone].channels[ScaleY] =
            make_curve(durationMs, baseScale,
                       0.020 + 0.003 * boneNumber, -skew * 0.5);
    }
    return clip;
}

Scene make_base_scene(std::uint32_t fps) {
    Scene scene;
    scene.fps = fps;
    scene.defaultClip = "idle";

    const std::size_t boneCount = 7;
    scene.bones.resize(boneCount);
    for (std::size_t i = 0; i < boneCount; ++i) {
        scene.bones[i].name = "bone_" + std::to_string(i);
        scene.bones[i].parent = i == 0 ? -1 : static_cast<int>(i - 1);
    }

    const std::array<Vec2, 11> positions{{
        {-22.0, 0.0},   {22.0, 0.0},   {-30.0, 22.0},
        {30.0, 22.0},   {-20.0, 45.0}, {20.0, 45.0},
        {-13.0, 68.0},  {13.0, 68.0},  {0.0, 92.0},
        {0.0, 12.0},    {0.0, 108.0},
    }};
    scene.vertices.resize(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        Vertex& vertex = scene.vertices[i];
        vertex.position = positions[i];
        int primary = static_cast<int>(std::floor(positions[i].y / 18.0));
        primary = std::max(0, std::min(6, primary));
        const int secondary = primary < 6 ? primary + 1 : 5;
        vertex.influenceCount = 2;
        vertex.boneIndices = {{primary, secondary, -1, -1}};
        vertex.weights = {{0.72, 0.28, 0.0, 0.0}};
    }

    scene.clips.push_back(make_clip("idle", 1000, boneCount, 1.0));
    scene.clips.push_back(make_clip("action", 600, boneCount, -0.75));
    return scene;
}

Scenario normal_scenario(const std::string& name) {
    Scenario scenario;
    std::uint32_t fps = 60;
    if (name == "parallel") {
        fps = 120;
        scenario.defaultFrames = 512;
    } else if (name == "transition") {
        fps = 60;
        scenario.defaultFrames = 48;
    } else if (name == "checksum" || name == "version") {
        fps = 60;
        scenario.defaultFrames = 8;
    } else if (name == "long-loop") {
        fps = 60;
        scenario.defaultFrames = 10000;
    } else {
        scenario.defaultFrames = 180;
    }
    scenario.fps = fps;
    scenario.definition.scene = make_base_scene(fps);
    if (name == "transition") {
        scenario.definition.transitions.push_back(
            TransitionSpec{12, "action", 200});
    }
    scenario.description =
        "deterministic seven-bone, eleven-vertex two-clip sample";
    return scenario;
}

Scenario error_scenario(const std::string& name) {
    Scenario scenario = normal_scenario("checksum");
    scenario.description = "intentional validation error: " + name;
    Scene& scene = scenario.definition.scene;
    if (name == "curve-time") {
        auto& curve = scene.clips[0].bones[0].channels[0];
        std::vector<Keyframe> keys(2);
        keys[0].timeMs = 0;
        keys[0].value = 0.0;
        keys[1].timeMs = 0;
        keys[1].value = 1.0;
        curve.keys() = std::move(keys);
    } else if (name == "bezier-denominator") {
        std::vector<Keyframe> keys(2);
        keys[0].timeMs = 0;
        keys[0].value = 0.0;
        keys[1].timeMs = 300;
        keys[1].value = 1.0;
        keys[0].outTangent = Tangent{0.0, 0.4};
        keys[1].inTangent = Tangent{100.0, 0.3};
        scene.clips[0].bones[0].channels[0].keys() = std::move(keys);
    } else if (name == "weight-zero") {
        scene.vertices[0].influenceCount = 1;
        scene.vertices[0].boneIndices = {{0, -1, -1, -1}};
        scene.vertices[0].weights = {{0.0, 0.0, 0.0, 0.0}};
    } else if (name == "weight-not-one") {
        scene.vertices[0].influenceCount = 1;
        scene.vertices[0].boneIndices = {{0, -1, -1, -1}};
        scene.vertices[0].weights = {{0.8, 0.0, 0.0, 0.0}};
    } else if (name == "skeleton-cycle") {
        scene.bones[0].parent = 0;
    } else if (name == "curve-empty") {
        scene.clips[0].bones[0].channels[0].keys().clear();
    } else {
        throw_error(ErrorCode::InvalidScenario,
                    "unknown error scenario: " + name,
                    {{"scenario", name}});
    }
    return scenario;
}

bool is_error_scenario(const std::string& name) {
    return name == "curve-time" || name == "bezier-denominator" ||
           name == "weight-zero" || name == "weight-not-one" ||
           name == "skeleton-cycle" || name == "curve-empty";
}

}

Scenario make_scenario(const std::string& name, bool variant) {
    Scenario scenario;
    if (is_error_scenario(name)) {
        scenario = error_scenario(name);
    } else if (name == "idle" || name == "parallel" ||
               name == "transition" || name == "checksum" ||
               name == "version" || name == "long-loop") {
        scenario = normal_scenario(name);
    } else {
        throw_error(ErrorCode::InvalidScenario,
                    "unknown scenario: " + name, {{"scenario", name}});
    }

    if (variant && !is_error_scenario(name)) {
        auto& curve = scenario.definition.scene.clips[1].bones[2].channels[0];
        if (curve.keys().size() >= 2) {
            curve.keys()[1].value += 0.125;
        }
    }
    return scenario;
}

const char* const* error_scenario_names(std::size_t& count) {
    static const char* const names[] = {
        "curve-time", "bezier-denominator", "weight-zero",
        "weight-not-one", "skeleton-cycle", "curve-empty"};
    count = sizeof(names) / sizeof(names[0]);
    return names;
}

}

