#pragma once

#include "core/math.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace uid166 {

struct SkinnedVertex {
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    float nx = 0.0f, ny = 0.0f, nz = 0.0f;
    float u = 0.0f, v = 0.0f;
    uint8_t bone[4]{0, 0, 0, 0};
    float weight[4]{1.0f, 0.0f, 0.0f, 0.0f};
};

struct Bone {
    std::string name;
    int32_t parent = -1;
    Mat4 inverseBind = Mat4::identity();
    Vec3 bindWorld{};
};

struct Keyframe {
    float time = 0.0f;
    Vec3 translation{};
    Quat rotation{};
};

struct BoneTrack {
    uint32_t boneIndex = 0;
    std::vector<Keyframe> keys;
};

struct CharacterAsset {
    std::vector<SkinnedVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Bone> bones;
    std::vector<BoneTrack> tracks;
    float duration = 2.0f;
    AABB bindBounds{};
    uint64_t payloadHash = 0;
};

[[nodiscard]] CharacterAsset generateCharacterAsset();
void writeCharacterBinary(const CharacterAsset& asset, const std::filesystem::path& path);
[[nodiscard]] CharacterAsset loadCharacterBinary(const std::filesystem::path& path);

[[nodiscard]] std::vector<Mat4> evaluatePose(const CharacterAsset& asset, float timeSeconds);
[[nodiscard]] AABB transformedCharacterBounds(const AABB& bindBounds,
                                              const std::vector<Mat4>& palette,
                                              const Mat4& world);

} // namespace uid166

