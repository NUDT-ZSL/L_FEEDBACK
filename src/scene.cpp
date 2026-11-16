#include "scene.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace skel {
namespace {

constexpr float kQuaternionEpsilon = 1.0e-8F;
constexpr float kWeightUnitTolerance = 1.0e-6F;

glm::mat4 translationMatrix(const glm::vec3& translation) {
    glm::mat4 matrix(1.0F);
    matrix[3] = glm::vec4(translation, 1.0F);
    return matrix;
}

glm::mat4 scaleMatrix(const glm::vec3& scale) {
    glm::mat4 matrix(1.0F);
    matrix[0][0] = scale.x;
    matrix[1][1] = scale.y;
    matrix[2][2] = scale.z;
    return matrix;
}

bool sampleTrack(const Asset& asset, const Node& node, float time,
                 AnimationSample& sample, Error& error) {
    if (node.sampleCount == 0U) {
        sample = AnimationSample{};
        return true;
    }
    const std::uint64_t endIndex =
        static_cast<std::uint64_t>(node.firstSample) + node.sampleCount;
    if (endIndex > asset.samples.size()) {
        error = makeError(ErrorCode::InvalidTrackNode,
                          "node sample range exceeds the sample table");
        return false;
    }

    const auto begin =
        asset.samples.begin() + static_cast<std::ptrdiff_t>(node.firstSample);
    const auto end = begin + static_cast<std::ptrdiff_t>(node.sampleCount);
    if (time <= begin->time) {
        sample = *begin;
    } else if (time >= (end - 1)->time) {
        sample = *(end - 1);
    } else {
        const auto upper = std::upper_bound(
            begin, end, time,
            [](float value, const AnimationSample& candidate) {
                return value < candidate.time;
            });
        const AnimationSample& next = *upper;
        const AnimationSample& previous = *(upper - 1);
        const float duration = next.time - previous.time;
        if (!(duration > 0.0F)) {
            error = makeError(ErrorCode::NonMonotonicTime,
                              "sampled track times are not strictly "
                              "increasing");
            return false;
        }
        const float alpha = (time - previous.time) / duration;
        sample.time = time;
        sample.translation =
            glm::mix(previous.translation, next.translation, alpha);
        sample.scale = glm::mix(previous.scale, next.scale, alpha);

        const float q0Length = glm::length(previous.rotation);
        const float q1Length = glm::length(next.rotation);
        if (!std::isfinite(q0Length) ||
            q0Length < kQuaternionEpsilon ||
            !std::isfinite(q1Length) ||
            q1Length < kQuaternionEpsilon) {
            error = makeError(ErrorCode::ZeroQuaternion,
                              "sample track contains a zero-length "
                              "quaternion");
            return false;
        }
        glm::quat q0 = previous.rotation / q0Length;
        glm::quat q1 = next.rotation / q1Length;
        sample.rotation = glm::normalize(glm::slerp(q0, q1, alpha));
    }

    if (!isFinite(sample.translation) || !isFinite(sample.scale) ||
        !isFinite(sample.rotation)) {
        error = makeError(ErrorCode::InvalidNumericValue,
                          "sampling produced a non-finite TRS component");
        return false;
    }
    return true;
}

bool validateNormalizationInput(const glm::vec3& translation,
                                const glm::quat& rotation,
                                const glm::vec3& scale, Error& error) {
    if (!isFinite(translation) || !isFinite(scale) ||
        !isFinite(rotation)) {
        error = makeError(ErrorCode::InvalidNumericValue,
                          "local transform contains a non-finite value");
        return false;
    }
    const float length = glm::length(rotation);
    if (!std::isfinite(length) || length < kQuaternionEpsilon) {
        error = makeError(ErrorCode::ZeroQuaternion,
                          "cannot compose a zero-length quaternion");
        return false;
    }
    return true;
}

} // namespace

bool composeLocalTransform(CompositionOrder order,
                           const glm::vec3& translation,
                           const glm::quat& rotation,
                           const glm::vec3& scale,
                           glm::mat4& result, Error& error) {
    if (!validateNormalizationInput(translation, rotation, scale, error)) {
        return false;
    }

    const glm::quat normalized = glm::normalize(rotation);
    const glm::mat4 t = translationMatrix(translation);
    const glm::mat4 r = glm::mat4_cast(normalized);
    const glm::mat4 s = scaleMatrix(scale);
    result = glm::mat4(1.0F);

    const glm::mat4* sequence[3]{};
    switch (order) {
    case CompositionOrder::TRS:
        sequence[0] = &t; sequence[1] = &r; sequence[2] = &s;
        break;
    case CompositionOrder::TSR:
        sequence[0] = &t; sequence[1] = &s; sequence[2] = &r;
        break;
    case CompositionOrder::RTS:
        sequence[0] = &r; sequence[1] = &t; sequence[2] = &s;
        break;
    case CompositionOrder::RST:
        sequence[0] = &r; sequence[1] = &s; sequence[2] = &t;
        break;
    case CompositionOrder::STR:
        sequence[0] = &s; sequence[1] = &t; sequence[2] = &r;
        break;
    case CompositionOrder::SRT:
        sequence[0] = &s; sequence[1] = &r; sequence[2] = &t;
        break;
    default:
        error = makeError(ErrorCode::InvalidCompositionOrder,
                          "unknown local transform composition order");
        return false;
    }

    for (const glm::mat4* operation : sequence) {
        result = result * *operation;
    }
    if (!isFinite(result)) {
        error = makeError(ErrorCode::InvalidNumericValue,
                          "local transform composition overflowed");
        return false;
    }
    if (glm::length(normalized) < kQuaternionEpsilon) {
        error = makeError(ErrorCode::ZeroQuaternion,
                          "normalized quaternion became zero");
        return false;
    }
    return true;
}

bool evaluateFrame(const Asset& asset, std::uint32_t frame,
                   BakedFrame& baked, Error& error) {
    baked = BakedFrame{};
    if (frame >= asset.frameCount) {
        error = makeError(ErrorCode::InvalidFrameRange,
                          "frame " + std::to_string(frame) +
                              " is outside [0, " +
                              std::to_string(asset.frameCount - 1U) + "]");
        return false;
    }

    const float frameTime =
        static_cast<float>(frame) / asset.framesPerSecond;
    const std::size_t nodeCount = asset.nodes.size();
    std::vector<glm::mat4> local(nodeCount, glm::mat4(1.0F));
    std::vector<glm::mat4> world(nodeCount, glm::mat4(1.0F));
    std::vector<std::vector<std::uint32_t>> children(nodeCount);
    std::vector<std::uint32_t> roots;

    for (std::size_t index = 0; index < nodeCount; ++index) {
        AnimationSample sample;
        if (!sampleTrack(asset, asset.nodes[index], frameTime, sample, error) ||
            !composeLocalTransform(asset.compositionOrder, sample.translation,
                                   sample.rotation, sample.scale, local[index],
                                   error)) {
            return false;
        }
        const std::int32_t parent = asset.nodes[index].parent;
        if (parent < 0) {
            roots.push_back(static_cast<std::uint32_t>(index));
        } else {
            children[static_cast<std::size_t>(parent)].push_back(
                static_cast<std::uint32_t>(index));
        }
    }

    std::queue<std::uint32_t> pending;
    for (const std::uint32_t root : roots) {
        pending.push(root);
    }
    std::size_t visited = 0;
    while (!pending.empty()) {
        const std::uint32_t index = pending.front();
        pending.pop();
        ++visited;
        const std::int32_t parent = asset.nodes[index].parent;
        world[index] = parent < 0
                           ? local[index]
                           : world[static_cast<std::size_t>(parent)] *
                                 local[index];
        if (!isFinite(world[index])) {
            error = makeError(ErrorCode::InvalidNumericValue,
                              "world transform overflowed at node " +
                                  std::to_string(index));
            return false;
        }
        for (const std::uint32_t child : children[index]) {
            pending.push(child);
        }
    }
    if (visited != nodeCount) {
        error = makeError(ErrorCode::ParentCycle,
                          "node hierarchy cannot be evaluated topologically");
        return false;
    }

    baked.frame = frame;
    baked.skinMatrices.resize(nodeCount);
    for (std::size_t index = 0; index < nodeCount; ++index) {
        baked.skinMatrices[index] =
            world[index] * asset.nodes[index].inverseBind;
    }

    baked.positions.reserve(asset.vertexCount());
    baked.normals.reserve(asset.vertexCount());
    baked.colors.reserve(asset.vertexCount());
    baked.indices.reserve(asset.triangleCount() * 3U);

    for (const Mesh& mesh : asset.meshes) {
        const std::uint32_t base =
            static_cast<std::uint32_t>(baked.positions.size());
        for (const Vertex& vertex : mesh.vertices) {
            float weightSum = 0.0F;
            for (std::uint32_t slot = 0; slot < vertex.weightCount; ++slot) {
                weightSum += vertex.weights[slot];
            }
            if (!std::isfinite(weightSum) || weightSum <= 0.0F) {
                error = makeError(ErrorCode::WeightSumZero,
                                  "vertex weight sum became non-positive");
                return false;
            }

            const bool needsNormalization =
                std::abs(weightSum - 1.0F) > kWeightUnitTolerance;
            if (needsNormalization) {
                ++baked.stats.normalizedVertices;
            }
            const float inverseSum = 1.0F / weightSum;

            glm::vec3 position(0.0F);
            glm::vec3 normal(0.0F);
            for (std::uint32_t slot = 0; slot < vertex.weightCount; ++slot) {
                const std::uint32_t bone = vertex.boneIndices[slot];
                const float weight =
                    needsNormalization ? vertex.weights[slot] * inverseSum
                                       : vertex.weights[slot];
                const glm::mat4& skin = baked.skinMatrices[bone];

                position += weight * glm::vec3(skin * glm::vec4(vertex.position,
                                                                1.0F));

                // Normals require the inverse-transpose of the skin matrix.
                // A singular skin matrix intentionally propagates NaN/inf so
                // that it can be counted and surfaced by offline rendering.
                const glm::mat3 normalMatrix =
                    glm::transpose(glm::inverse(glm::mat3(skin)));
                normal += weight * (normalMatrix * vertex.normal);
            }

            const float normalLength = glm::length(normal);
            if (std::isfinite(normalLength) && normalLength > 0.0F) {
                normal /= normalLength;
            }

            if (!isFinite(position)) {
                ++baked.stats.nanPositions;
            }
            if (!isFinite(normal)) {
                ++baked.stats.nanNormals;
            }
            baked.positions.push_back(position);
            baked.normals.push_back(normal);
            baked.colors.push_back(vertex.color);
        }

        for (const std::uint32_t index : mesh.indices) {
            baked.indices.push_back(base + index);
        }
    }
    baked.stats.frame = frame;
    error = {};
    return true;
}

std::vector<InterleavedVertex> makeInterleavedVertices(
    const BakedFrame& baked) {
    std::vector<InterleavedVertex> vertices;
    vertices.resize(baked.positions.size());
    for (std::size_t index = 0; index < vertices.size(); ++index) {
        InterleavedVertex& output = vertices[index];
        output.position[0] = baked.positions[index].x;
        output.position[1] = baked.positions[index].y;
        output.position[2] = baked.positions[index].z;
        output.normal[0] = baked.normals[index].x;
        output.normal[1] = baked.normals[index].y;
        output.normal[2] = baked.normals[index].z;
        output.color[0] = baked.colors[index].x;
        output.color[1] = baked.colors[index].y;
        output.color[2] = baked.colors[index].z;
    }
    return vertices;
}

} // namespace skel
