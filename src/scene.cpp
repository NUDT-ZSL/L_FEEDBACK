#include "skinbake/scene.hpp"

#include "skinbake/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>

namespace skinbake {

namespace {

std::string fixed6(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6) << value;
    return out.str();
}

class FingerprintBuilder {
public:
    void u8(std::uint8_t value) { mix(&value, sizeof(value)); }
    void u32(std::uint32_t value) {
        for (int i = 0; i < 4; ++i) {
            u8(static_cast<std::uint8_t>((value >> (i * 8)) & 0xffu));
        }
    }
    void u64(std::uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            u8(static_cast<std::uint8_t>((value >> (i * 8)) & 0xffu));
        }
    }
    void i64(std::int64_t value) {
        u64(static_cast<std::uint64_t>(value));
    }
    void f64(double value) {
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "unexpected double width");
        std::memcpy(&bits, &value, sizeof(bits));
        u64(bits);
    }
    void str(const std::string& value) {
        u32(static_cast<std::uint32_t>(value.size()));
        mix(value.data(), value.size());
    }
    std::uint64_t finish() const { return hash_; }

private:
    void mix(const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash_ ^= bytes[i];
            hash_ *= 1099511628211ull;
        }
    }

    std::uint64_t hash_ = 1469598103934665603ull;
};

void validate_vertex(const Vertex& vertex, std::size_t vertexIndex,
                     std::size_t boneCount) {
    if (!is_finite(vertex.position)) {
        throw_error(ErrorCode::MeshInvalid,
                    "vertex position is not finite",
                    {{"vertex", std::to_string(vertexIndex)}});
    }
    if (vertex.influenceCount > 4) {
        throw_error(ErrorCode::VertexTooManyInfluences,
                    "vertex has more than four skin influences",
                    {{"vertex", std::to_string(vertexIndex)},
                     {"influences", std::to_string(vertex.influenceCount)}});
    }

    double sum = 0.0;
    for (std::uint32_t i = 0; i < vertex.influenceCount; ++i) {
        const int bone = vertex.boneIndices[i];
        const double weight = vertex.weights[i];
        if (bone < 0 || static_cast<std::size_t>(bone) >= boneCount) {
            throw_error(ErrorCode::VertexBoneIndexInvalid,
                        "vertex references an invalid bone index",
                        {{"vertex", std::to_string(vertexIndex)},
                         {"influence", std::to_string(i)},
                         {"boneIndex", std::to_string(bone)}});
        }
        for (std::uint32_t j = 0; j < i; ++j) {
            if (vertex.boneIndices[j] == bone) {
                throw_error(ErrorCode::VertexBoneIndexInvalid,
                            "vertex contains a duplicate bone influence",
                            {{"vertex", std::to_string(vertexIndex)},
                             {"boneIndex", std::to_string(bone)}});
            }
        }
        if (!std::isfinite(weight) || weight < 0.0) {
            throw_error(ErrorCode::WeightInvalid,
                        "vertex skin weight is negative or not finite",
                        {{"vertex", std::to_string(vertexIndex)},
                         {"weight", fixed6(weight)}});
        }
        sum += weight;
    }

    if (sum == 0.0) {
        throw_error(ErrorCode::WeightSumZero,
                    "vertex skin-weight sum is zero",
                    {{"vertex", std::to_string(vertexIndex)},
                     {"weightSum", fixed6(sum)}});
    }
    if (std::fabs(sum - 1.0) > 1e-4) {
        throw_error(
            ErrorCode::WeightSumNotOne,
            "vertex skin-weight sum differs from 1 by more than 0.0001: " +
                fixed6(sum),
            {{"vertex", std::to_string(vertexIndex)},
             {"weightSum", fixed6(sum)}});
    }
}

void build_topological_order(const Scene& scene,
                             std::vector<int>& order) {
    const std::size_t count = scene.bones.size();
    for (std::size_t i = 0; i < count; ++i) {
        const int parent = scene.bones[i].parent;
        if (parent < -1 || parent >= static_cast<int>(count)) {
            throw_error(ErrorCode::SkeletonParentInvalid,
                        "skeleton parent index is out of range",
                        {{"boneIndex", std::to_string(i)},
                         {"parent", std::to_string(parent)}});
        }
        if (static_cast<std::size_t>(parent) == i) {
            throw_error(ErrorCode::SkeletonCycle,
                        "cycle detected in skeleton at bone " + std::to_string(i),
                        {{"boneIndex", std::to_string(i)}});
        }
    }

    std::vector<std::uint8_t> state(count, 0);
    order.clear();
    order.reserve(count);
    std::function<void(std::size_t)> visit = [&](std::size_t index) {
        if (state[index] == 2) {
            return;
        }
        if (state[index] == 1) {
            throw_error(ErrorCode::SkeletonCycle,
                        "cycle detected in skeleton at bone " +
                            std::to_string(index),
                        {{"boneIndex", std::to_string(index)}});
        }
        state[index] = 1;
        const int parent = scene.bones[index].parent;
        if (parent >= 0) {
            visit(static_cast<std::size_t>(parent));
        }
        state[index] = 2;
        order.push_back(static_cast<int>(index));
    };
    for (std::size_t i = 0; i < count; ++i) {
        visit(i);
    }
}

}

void validate_scene(const Scene& scene) {
    if (scene.fps == 0) {
        throw_error(ErrorCode::InvalidArgument, "scene fps must be positive");
    }
    if (scene.bones.empty()) {
        throw_error(ErrorCode::MeshInvalid, "skeleton contains no bones");
    }
    if (scene.vertices.empty()) {
        throw_error(ErrorCode::MeshInvalid, "mesh contains no vertices");
    }
    if (scene.clips.empty()) {
        throw_error(ErrorCode::ClipNotFound, "scene contains no clips");
    }

    std::vector<int> order;
    build_topological_order(scene, order);

    for (std::size_t i = 0; i < scene.vertices.size(); ++i) {
        validate_vertex(scene.vertices[i], i, scene.bones.size());
    }

    std::unordered_map<std::string, std::size_t> names;
    for (std::size_t c = 0; c < scene.clips.size(); ++c) {
        const Clip& clip = scene.clips[c];
        if (clip.name.empty()) {
            throw_error(ErrorCode::ClipNotFound, "clip name is empty",
                        {{"clipIndex", std::to_string(c)}});
        }
        if (!names.emplace(clip.name, c).second) {
            throw_error(ErrorCode::ClipNotFound,
                        "duplicate clip name: " + clip.name,
                        {{"clip", clip.name}});
        }
        if (clip.durationMs <= 0) {
            throw_error(ErrorCode::ClipDurationInvalid,
                        "clip duration must be positive",
                        {{"clip", clip.name},
                         {"durationMs", std::to_string(clip.durationMs)}});
        }
        if (clip.bones.size() != scene.bones.size()) {
            throw_error(ErrorCode::ClipNotFound,
                        "clip bone curve count does not match skeleton",
                        {{"clip", clip.name},
                         {"expectedBones", std::to_string(scene.bones.size())},
                         {"actualBones", std::to_string(clip.bones.size())}});
        }
        for (std::size_t b = 0; b < clip.bones.size(); ++b) {
            for (std::size_t channel = 0; channel < ChannelCount; ++channel) {
                CurveContext context{clip.name, b, channel};
                clip.bones[b].channels[channel].validate(context);
            }
        }
    }

    if (scene.defaultClip.empty() ||
        names.find(scene.defaultClip) == names.end()) {
        throw_error(ErrorCode::ClipNotFound,
                    "default clip does not exist: " + scene.defaultClip,
                    {{"clip", scene.defaultClip}});
    }
}

std::uint64_t scene_fingerprint(const Scene& scene) {
    FingerprintBuilder builder;
    builder.str("SKINBAKE-SCENE-v1");
    builder.u32(scene.fps);
    builder.str(scene.defaultClip);
    builder.u32(static_cast<std::uint32_t>(scene.bones.size()));
    for (const Bone& bone : scene.bones) {
        builder.str(bone.name);
        builder.u32(static_cast<std::uint32_t>(bone.parent));
    }
    builder.u32(static_cast<std::uint32_t>(scene.vertices.size()));
    for (const Vertex& vertex : scene.vertices) {
        builder.f64(vertex.position.x);
        builder.f64(vertex.position.y);
        builder.u32(vertex.influenceCount);
        for (std::uint32_t i = 0; i < vertex.influenceCount; ++i) {
            builder.u32(static_cast<std::uint32_t>(vertex.boneIndices[i]));
            builder.f64(vertex.weights[i]);
        }
    }
    builder.u32(static_cast<std::uint32_t>(scene.clips.size()));
    for (const Clip& clip : scene.clips) {
        builder.str(clip.name);
        builder.i64(clip.durationMs);
        builder.u8(clip.loop ? 1u : 0u);
        builder.u32(static_cast<std::uint32_t>(clip.bones.size()));
        for (const BoneCurves& bone : clip.bones) {
            for (const Curve& curve : bone.channels) {
                builder.u32(static_cast<std::uint32_t>(curve.keys().size()));
                for (const Keyframe& key : curve.keys()) {
                    builder.i64(key.timeMs);
                    builder.f64(key.value);
                    builder.f64(key.inTangent.timeMs);
                    builder.f64(key.inTangent.value);
                    builder.f64(key.outTangent.timeMs);
                    builder.f64(key.outTangent.value);
                }
            }
        }
    }
    return builder.finish();
}

std::shared_ptr<const PreparedScene> prepare_scene(const Scene& scene) {
    validate_scene(scene);
    auto prepared = std::make_shared<PreparedScene>();
    prepared->scene = scene;
    for (Clip& clip : prepared->scene.clips) {
        for (BoneCurves& bone : clip.bones) {
            for (Curve& curve : bone.channels) {
                curve.prepare();
            }
        }
    }
    build_topological_order(scene, prepared->topologicalOrder);
    for (std::size_t i = 0; i < scene.clips.size(); ++i) {
        prepared->clipByName.emplace(scene.clips[i].name, i);
    }
    prepared->fingerprint = scene_fingerprint(scene);
    return prepared;
}

std::size_t clip_index(const PreparedScene& prepared, const std::string& name) {
    auto it = prepared.clipByName.find(name);
    if (it == prepared.clipByName.end()) {
        throw_error(ErrorCode::ClipNotFound, "clip not found: " + name,
                    {{"clip", name}});
    }
    return it->second;
}

std::int64_t rounded_frame_time_ms(std::uint32_t frameIndex,
                                   std::uint32_t fps) {
    if (fps == 0) {
        throw_error(ErrorCode::InvalidArgument, "fps must be positive");
    }
    return static_cast<std::int64_t>(
        std::llround(static_cast<double>(frameIndex) * 1000.0 /
                     static_cast<double>(fps)));
}

std::int64_t wrap_playback_time(std::int64_t absoluteTimeMs,
                                std::int64_t durationMs, bool loop) {
    if (durationMs <= 0) {
        return 0;
    }
    if (!loop) {
        if (absoluteTimeMs < 0) {
            return 0;
        }
        return std::min(absoluteTimeMs, durationMs);
    }
    std::int64_t wrapped = absoluteTimeMs % durationMs;
    if (wrapped < 0) {
        wrapped += durationMs;
    }
    return wrapped;
}

}

