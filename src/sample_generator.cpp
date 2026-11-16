#include "sample_generator.hpp"

#include "binary_io.hpp"
#include "scene.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace skel {
namespace {

constexpr float kFps = 24.0F;
constexpr std::uint32_t kFrameCount = 24;
constexpr std::uint32_t kAnimationSamples = kFrameCount;

struct RestTransform {
    glm::vec3 translation;
    glm::quat rotation;
    glm::vec3 scale;
};

AnimationSample makeSample(float time, const glm::vec3& translation,
                           const glm::quat& rotation, const glm::vec3& scale) {
    AnimationSample sample;
    sample.time = time;
    sample.translation = translation;
    sample.rotation = rotation;
    sample.scale = scale;
    return sample;
}

glm::quat axisAngle(glm::vec3 axis, float radians) {
    return glm::angleAxis(radians, glm::normalize(axis));
}

void appendCube(std::vector<glm::vec3>& positions,
                std::vector<glm::vec3>& normals,
                std::vector<std::uint32_t>& indices,
                const glm::vec3& minimum, const glm::vec3& maximum) {
    const std::uint32_t base =
        static_cast<std::uint32_t>(positions.size());
    const std::array<glm::vec3, 8> corners{
        glm::vec3(minimum.x, minimum.y, minimum.z),
        glm::vec3(maximum.x, minimum.y, minimum.z),
        glm::vec3(maximum.x, maximum.y, minimum.z),
        glm::vec3(minimum.x, maximum.y, minimum.z),
        glm::vec3(minimum.x, minimum.y, maximum.z),
        glm::vec3(maximum.x, minimum.y, maximum.z),
        glm::vec3(maximum.x, maximum.y, maximum.z),
        glm::vec3(minimum.x, maximum.y, maximum.z),
    };
    const std::array<glm::vec3, 8> vertexNormals{
        glm::normalize(glm::vec3(-1.0F, -1.0F, -1.0F)),
        glm::normalize(glm::vec3(1.0F, -1.0F, -1.0F)),
        glm::normalize(glm::vec3(1.0F, 1.0F, -1.0F)),
        glm::normalize(glm::vec3(-1.0F, 1.0F, -1.0F)),
        glm::normalize(glm::vec3(-1.0F, -1.0F, 1.0F)),
        glm::normalize(glm::vec3(1.0F, -1.0F, 1.0F)),
        glm::normalize(glm::vec3(1.0F, 1.0F, 1.0F)),
        glm::normalize(glm::vec3(-1.0F, 1.0F, 1.0F)),
    };
    positions.insert(positions.end(), corners.begin(), corners.end());
    normals.insert(normals.end(), vertexNormals.begin(), vertexNormals.end());

    const std::array<std::uint32_t, 36> cubeIndices{
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        3, 7, 6, 3, 6, 2,
        0, 4, 7, 0, 7, 3,
        1, 2, 6, 1, 6, 5,
    };
    for (const std::uint32_t index : cubeIndices) {
        indices.push_back(base + index);
    }
}

Asset makeNormalAsset(CompositionOrder order) {
    Asset asset;
    asset.compositionOrder = order;
    asset.framesPerSecond = kFps;
    asset.frameCount = kFrameCount;

    const std::array<RestTransform, 5> rest{{
        {glm::vec3(0.0F), glm::quat(1.0F, 0.0F, 0.0F, 0.0F),
         glm::vec3(1.0F)},
        {glm::vec3(0.0F, 1.0F, 0.0F),
         glm::quat(1.0F, 0.0F, 0.0F, 0.0F), glm::vec3(1.0F)},
        {glm::vec3(0.0F, 1.5F, 0.0F),
         glm::quat(1.0F, 0.0F, 0.0F, 0.0F), glm::vec3(1.0F)},
        {glm::vec3(0.7F, 0.8F, 0.0F),
         glm::quat(1.0F, 0.0F, 0.0F, 0.0F), glm::vec3(1.0F)},
        {glm::vec3(0.8F, 0.0F, 0.0F),
         glm::quat(1.0F, 0.0F, 0.0F, 0.0F), glm::vec3(1.0F)},
    }};
    const std::array<std::string, 5> names{
        "root", "pelvis", "chest", "arm", "tip"};
    const std::array<std::int32_t, 5> parents{-1, 0, 1, 2, 3};

    std::vector<glm::mat4> restWorld(rest.size(), glm::mat4(1.0F));
    for (std::size_t nodeIndex = 0; nodeIndex < rest.size(); ++nodeIndex) {
        glm::mat4 local;
        Error error;
        if (!composeLocalTransform(order, rest[nodeIndex].translation,
                                   rest[nodeIndex].rotation,
                                   rest[nodeIndex].scale, local, error)) {
            return {};
        }
        restWorld[nodeIndex] =
            parents[nodeIndex] < 0
                ? local
                : restWorld[static_cast<std::size_t>(parents[nodeIndex])] *
                      local;
    }

    asset.samples.reserve(rest.size() * kAnimationSamples);
    for (std::size_t nodeIndex = 0; nodeIndex < rest.size(); ++nodeIndex) {
        Node node;
        node.name = names[nodeIndex];
        node.parent = parents[nodeIndex];
        node.firstSample =
            static_cast<std::uint32_t>(asset.samples.size());
        node.sampleCount = kAnimationSamples;
        node.inverseBind = glm::inverse(restWorld[nodeIndex]);

        for (std::uint32_t frame = 0; frame < kAnimationSamples; ++frame) {
            const float phase =
                static_cast<float>(frame) / static_cast<float>(kFrameCount);
            const float angle =
                glm::two_pi<float>() * phase;
            glm::vec3 translation = rest[nodeIndex].translation;
            glm::quat rotation = rest[nodeIndex].rotation;
            glm::vec3 scale = rest[nodeIndex].scale;

            if (nodeIndex == 0) {
                translation.y += 0.08F * std::sin(angle);
                rotation = axisAngle(glm::vec3(0.0F, 1.0F, 0.0F),
                                     0.10F * std::sin(angle));
            } else if (nodeIndex == 1) {
                rotation = axisAngle(glm::vec3(0.0F, 0.0F, 1.0F),
                                     0.08F * std::sin(angle + 0.4F));
            } else if (nodeIndex == 2) {
                rotation = axisAngle(glm::vec3(0.0F, 1.0F, 0.0F),
                                     0.30F * std::sin(angle));
                scale = glm::vec3(1.0F + 0.15F * std::sin(angle),
                                  1.0F - 0.10F * std::sin(angle),
                                  1.0F);
            } else if (nodeIndex == 3) {
                rotation = axisAngle(glm::vec3(0.0F, 0.0F, 1.0F),
                                     0.55F * std::sin(angle + 0.8F));
            } else {
                rotation = axisAngle(glm::vec3(1.0F, 0.0F, 0.0F),
                                     0.35F * std::sin(angle + 1.2F));
            }

            asset.samples.push_back(makeSample(
                static_cast<float>(frame) / kFps, translation, rotation,
                scale));
        }
        asset.nodes.push_back(node);
    }

    Mesh torso;
    torso.name = "torso";
    {
        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> normals;
        appendCube(positions, normals, torso.indices,
                   glm::vec3(-0.7F, 0.5F, -0.35F),
                   glm::vec3(0.7F, 2.4F, 0.35F));
        for (std::size_t index = 0; index < positions.size(); ++index) {
            Vertex vertex;
            vertex.position = positions[index];
            vertex.normal = normals[index];
            vertex.color = index < 4U ? glm::vec3(0.35F, 0.62F, 0.92F)
                                      : glm::vec3(0.20F, 0.42F, 0.72F);
            const float blend =
                glm::clamp((positions[index].y - 0.5F) / 1.9F, 0.0F, 1.0F);
            vertex.weightCount = 2;
            vertex.boneIndices = {1, 2, 0, 0};
            vertex.weights = {1.0F - blend, blend, 0.0F, 0.0F};
            torso.vertices.push_back(vertex);
        }
    }

    Mesh arm;
    arm.name = "arm";
    {
        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> normals;
        appendCube(positions, normals, arm.indices,
                   glm::vec3(0.55F, 1.55F, -0.22F),
                   glm::vec3(2.05F, 2.15F, 0.22F));
        for (std::size_t index = 0; index < positions.size(); ++index) {
            Vertex vertex;
            vertex.position = positions[index];
            vertex.normal = normals[index];
            vertex.color = glm::vec3(0.92F, 0.44F, 0.28F);
            if (index == 0U) {
                // Deliberately non-unit weights exercise the normalization
                // policy and statistics path.
                vertex.weightCount = 3;
                vertex.boneIndices = {2, 3, 4, 0};
                vertex.weights = {0.30F, 0.30F, 0.20F, 0.0F};
            } else if (index < 4U) {
                vertex.weightCount = 2;
                vertex.boneIndices = {2, 3, 0, 0};
                vertex.weights = {0.45F, 0.55F, 0.0F, 0.0F};
            } else {
                vertex.weightCount = 4;
                vertex.boneIndices = {2, 3, 4, 0};
                vertex.weights = {0.10F, 0.35F, 0.45F, 0.10F};
            }
            arm.vertices.push_back(vertex);
        }
    }

    asset.meshes.push_back(std::move(torso));
    asset.meshes.push_back(std::move(arm));
    return asset;
}

} // namespace

const std::vector<GeneratorPreset>& generatorPresets() {
    static const std::vector<GeneratorPreset> presets{
        {"normal", "valid two-mesh rig with looping sampled TRS tracks"},
        {"cycle", "parent indices form a cycle"},
        {"oob-index", "mesh triangle index is out of range"},
        {"oob-bone", "vertex references an out-of-range bone"},
        {"nonmonotonic", "sample times repeat within a track"},
        {"zero-quat", "animation track contains a zero quaternion"},
        {"bad-weight-count", "vertex declares more than four weights"},
        {"bad-weights", "vertex has a zero weight sum"},
        {"bad-order", "header declares an unknown composition order"},
        {"singular", "zero inverse-bind matrix produces normal NaNs"},
        {"truncated", "valid asset body cut short"},
    };
    return presets;
}

bool generateAsset(const std::string& preset, CompositionOrder order,
                   Asset& asset, Error& error) {
    if (order < CompositionOrder::TRS || order > CompositionOrder::SRT) {
        error = makeError(ErrorCode::InvalidCompositionOrder,
                          "generator composition order must be in [0, 5]");
        asset = Asset{};
        return false;
    }
    asset = makeNormalAsset(order);
    if (asset.nodes.empty()) {
        error = makeError(ErrorCode::InvalidNumericValue,
                          "failed to build generated rig");
        return false;
    }

    if (preset == "normal" || preset == "truncated") {
        error = {};
        return true;
    }
    if (preset == "cycle") {
        asset.nodes[0].parent = 1;
    } else if (preset == "oob-index") {
        asset.meshes[0].indices[0] = 999'999U;
    } else if (preset == "oob-bone") {
        asset.meshes[0].vertices[0].boneIndices[0] =
            static_cast<std::uint32_t>(asset.nodes.size());
    } else if (preset == "nonmonotonic") {
        asset.samples[2].time = asset.samples[1].time;
    } else if (preset == "zero-quat") {
        asset.samples[0].rotation = glm::quat(0.0F, 0.0F, 0.0F, 0.0F);
    } else if (preset == "bad-weight-count") {
        asset.meshes[0].vertices[0].weightCount = 5U;
    } else if (preset == "bad-weights") {
        Vertex& vertex = asset.meshes[0].vertices[0];
        vertex.weightCount = 2;
        vertex.weights = {0.0F, 0.0F, 0.0F, 0.0F};
    } else if (preset == "bad-order") {
        asset.compositionOrder = static_cast<CompositionOrder>(99U);
    } else if (preset == "singular") {
        asset.nodes[2].inverseBind = glm::mat4(0.0F);
    } else {
        error = makeError(ErrorCode::InvalidArgument,
                          "unknown generator preset: " + preset);
        asset = Asset{};
        return false;
    }
    error = {};
    return true;
}

bool writeAsset(const Asset& asset, const std::filesystem::path& path,
                Error& error) {
    ByteWriter writer;
    writer.writeBytes("SKA1", 4);
    writer.writeU32(asset.version);
    writer.writeU32(static_cast<std::uint32_t>(asset.compositionOrder));
    writer.writeF32(asset.framesPerSecond);
    writer.writeU32(asset.frameCount);
    writer.writeU32(static_cast<std::uint32_t>(asset.nodes.size()));
    writer.writeU32(static_cast<std::uint32_t>(asset.meshes.size()));
    writer.writeU32(static_cast<std::uint32_t>(asset.samples.size()));

    for (const Node& node : asset.nodes) {
        writer.writeString32(node.name);
        writer.writeI32(node.parent);
        writer.writeU32(node.firstSample);
        writer.writeU32(node.sampleCount);
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                writer.writeF32(node.inverseBind[column][row]);
            }
        }
    }
    for (const AnimationSample& sample : asset.samples) {
        writer.writeF32(sample.time);
        writer.writeF32(sample.translation.x);
        writer.writeF32(sample.translation.y);
        writer.writeF32(sample.translation.z);
        writer.writeF32(sample.rotation.x);
        writer.writeF32(sample.rotation.y);
        writer.writeF32(sample.rotation.z);
        writer.writeF32(sample.rotation.w);
        writer.writeF32(sample.scale.x);
        writer.writeF32(sample.scale.y);
        writer.writeF32(sample.scale.z);
    }
    for (const Mesh& mesh : asset.meshes) {
        writer.writeString32(mesh.name);
        writer.writeU32(static_cast<std::uint32_t>(mesh.vertices.size()));
        writer.writeU32(static_cast<std::uint32_t>(mesh.indices.size()));
        for (const Vertex& vertex : mesh.vertices) {
            writer.writeF32(vertex.position.x);
            writer.writeF32(vertex.position.y);
            writer.writeF32(vertex.position.z);
            writer.writeF32(vertex.normal.x);
            writer.writeF32(vertex.normal.y);
            writer.writeF32(vertex.normal.z);
            writer.writeF32(vertex.color.x);
            writer.writeF32(vertex.color.y);
            writer.writeF32(vertex.color.z);
            for (std::uint32_t slot = 0; slot < 4U; ++slot) {
                writer.writeU32(vertex.boneIndices[slot]);
            }
            writer.writeU32(vertex.weightCount);
            for (std::uint32_t slot = 0; slot < 4U; ++slot) {
                writer.writeF32(vertex.weights[slot]);
            }
        }
        for (const std::uint32_t index : mesh.indices) {
            writer.writeU32(index);
        }
    }

    std::error_code filesystemError;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(),
                                            filesystemError);
        if (filesystemError) {
            error = makeError(ErrorCode::WriteFailed,
                              "cannot create output directory: " +
                                  filesystemError.message());
            return false;
        }
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        error = makeError(ErrorCode::WriteFailed,
                          "cannot open asset output: " + path.string());
        return false;
    }
    stream.write(reinterpret_cast<const char*>(writer.data().data()),
                 static_cast<std::streamsize>(writer.data().size()));
    stream.flush();
    if (!stream) {
        error = makeError(ErrorCode::WriteFailed,
                          "failed to write asset output: " + path.string());
        return false;
    }
    error = {};
    return true;
}

bool generateAssetFile(const std::string& preset, CompositionOrder order,
                       const std::filesystem::path& path, Error& error) {
    Asset asset;
    if (!generateAsset(preset, order, asset, error) ||
        !writeAsset(asset, path, error)) {
        return false;
    }
    if (preset == "truncated") {
        std::error_code filesystemError;
        const std::uintmax_t size = std::filesystem::file_size(
            path, filesystemError);
        if (filesystemError || size < 40U) {
            error = makeError(ErrorCode::WriteFailed,
                              "cannot truncate generated asset");
            return false;
        }
        std::filesystem::resize_file(path, size - 17U, filesystemError);
        if (filesystemError) {
            error = makeError(ErrorCode::WriteFailed,
                              "cannot truncate generated asset: " +
                                  filesystemError.message());
            return false;
        }
    }
    error = {};
    return true;
}

} // namespace skel
