#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <glm/gtc/constants.hpp>

#include "animation.h"
#include "cascade.h"
#include "error.h"
#include "image_io.h"
#include "scene.h"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const char* expression, const char* file, int line) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::cerr << file << ':' << line << ": CHECK failed: "
                  << expression << '\n';
    }
}

bool nearlyEqual(double a, double b, double tolerance = 1.0e-9) {
    return std::abs(a - b) <= tolerance;
}

#define CHECK(expression) check((expression), #expression, __FILE__, __LINE__)

void runTest(const char* name, const std::function<void()>& test) {
    std::cout << "[ RUN      ] " << name << '\n';
    try {
        test();
        std::cout << "[       OK ] " << name << '\n';
    } catch (const std::exception& error) {
        ++g_failures;
        std::cerr << "[  FAILED  ] " << name << ": " << error.what() << '\n';
    }
}

sr::SkinnedVertex makeVertex(glm::dvec3 position,
                             glm::dvec3 normal,
                             std::vector<std::pair<std::uint32_t, float>> weights) {
    sr::SkinnedVertex vertex;
    vertex.px = static_cast<float>(position.x);
    vertex.py = static_cast<float>(position.y);
    vertex.pz = static_cast<float>(position.z);
    vertex.nx = static_cast<float>(normal.x);
    vertex.ny = static_cast<float>(normal.y);
    vertex.nz = static_cast<float>(normal.z);
    for (int i = 0; i < sr::kMaxBoneInfluences; ++i) {
        vertex.boneIndices[static_cast<std::size_t>(i)] = 0;
        vertex.boneWeights[static_cast<std::size_t>(i)] = 0.0f;
    }
    for (std::size_t i = 0; i < weights.size() && i < 4; ++i) {
        vertex.boneIndices[i] = weights[i].first;
        vertex.boneWeights[i] = weights[i].second;
    }
    return vertex;
}

} // namespace

int main() {
    runTest("quaternion shortest arc and normalization", [] {
        const glm::dquat identity(1.0, 0.0, 0.0, 0.0);
        const glm::dquat almostFullTurn =
            glm::angleAxis(glm::radians(350.0), glm::dvec3(0, 0, 1));
        const glm::dquat midpoint = sr::shortestSlerp(identity, almostFullTurn, 0.5);
        CHECK(nearlyEqual(glm::length(midpoint), 1.0, 1.0e-12));
        const glm::dvec3 rotated = midpoint * glm::dvec3(1, 0, 0);
        const double angle = std::atan2(rotated.y, rotated.x);
        CHECK(nearlyEqual(angle, glm::radians(-5.0), 1.0e-9));

        const glm::dquat a =
            glm::angleAxis(glm::radians(170.0), glm::dvec3(0, 1, 0));
        const glm::dquat b =
            glm::angleAxis(glm::radians(-170.0), glm::dvec3(0, 1, 0));
        const glm::dquat across = sr::shortestSlerp(a, b, 0.5);
        const glm::dvec3 xAxis = across * glm::dvec3(1, 0, 0);
        CHECK(nearlyEqual(xAxis.x, -1.0, 1.0e-9));
        CHECK(nearlyEqual(xAxis.y, 0.0, 1.0e-9));
    });

    runTest("irregular keyframe lookup and endpoint clamping", [] {
        std::vector<sr::Keyframe> keys = {
            {0.00, {0, 0, 0}, glm::dquat(1, 0, 0, 0)},
            {0.25, {1, 2, 3}, glm::dquat(1, 0, 0, 0)},
            {1.10, {5, 6, 7}, glm::dquat(1, 0, 0, 0)},
            {2.70, {9, 10, 11}, glm::dquat(1, 0, 0, 0)},
        };
        const sr::PoseSample before = sr::sampleKeyframes(keys, -10.0);
        CHECK(before.clampedBefore);
        CHECK(nearlyEqual(before.translation.x, 0.0));

        const sr::PoseSample after = sr::sampleKeyframes(keys, 10.0);
        CHECK(after.clampedAfter);
        CHECK(nearlyEqual(after.translation.x, 9.0));

        const sr::PoseSample middle = sr::sampleKeyframes(keys, 0.675);
        CHECK(middle.segmentIndex == 1);
        CHECK(nearlyEqual(middle.segmentTime0, 0.25));
        CHECK(nearlyEqual(middle.segmentTime1, 1.10));
        CHECK(nearlyEqual(middle.fraction, 0.5));
        CHECK(nearlyEqual(middle.translation.x, 3.0));
        CHECK(nearlyEqual(middle.translation.y, 4.0));

        const sr::PoseSample exact = sr::sampleKeyframes(keys, 1.10);
        CHECK(nearlyEqual(exact.translation.x, 5.0));
    });

    runTest("three-bone pose hierarchy and bind identity", [] {
        const sr::Skeleton skeleton = sr::makeSkeleton();
        CHECK(skeleton.bones.size() == 3);
        CHECK(skeleton.bones[0].parent == -1);
        CHECK(skeleton.bones[1].parent == 0);
        CHECK(skeleton.bones[2].parent == 1);

        const sr::PoseResult bindPose = sr::evaluatePose(skeleton, 0.0);
        CHECK(nearlyEqual(bindPose.worldMatrices[2][3].x, 0.0));
        CHECK(nearlyEqual(bindPose.worldMatrices[2][3].y, 2.15));
        for (const glm::dmat4& skin : bindPose.skinMatrices) {
            for (int column = 0; column < 4; ++column) {
                for (int row = 0; row < 4; ++row) {
                    const double expected = column == row ? 1.0 : 0.0;
                    CHECK(nearlyEqual(skin[column][row], expected, 1.0e-12));
                }
            }
        }

        const sr::PoseResult animated = sr::evaluatePose(skeleton, 1.2);
        const glm::dvec3 movedTip(
            animated.worldMatrices[2][3].x,
            animated.worldMatrices[2][3].y,
            animated.worldMatrices[2][3].z);
        const glm::dvec3 bindTip(0.0, 2.15, 0.0);
        CHECK(glm::length(movedTip - bindTip) > 1.0e-3);
        CHECK(animated.samples[0].segmentIndex >= 0);
        CHECK(animated.samples[1].segmentIndex >= 0);
        CHECK(animated.samples[2].segmentIndex >= 0);
    });

    runTest("weight normalization, fallback, and LBS", [] {
        sr::SkinnedVertex normalized = makeVertex(
            {1, 2, 3}, {0, 1, 0}, {{0, 2.0f}, {1, 3.0f}, {2, 5.0f}});
        sr::normalizeVertexWeights(normalized);
        CHECK(nearlyEqual(normalized.boneWeights[0], 0.2f, 1.0e-7));
        CHECK(nearlyEqual(normalized.boneWeights[1], 0.3f, 1.0e-7));
        CHECK(nearlyEqual(normalized.boneWeights[2], 0.5f, 1.0e-7));
        CHECK(nearlyEqual(normalized.boneWeights[3], 0.0f, 1.0e-7));

        sr::SkinnedVertex zero =
            makeVertex({0, 0, 0}, {0, 1, 0}, {{0, 0.0f}, {1, 0.0f}});
        sr::normalizeVertexWeights(zero);
        CHECK(zero.boneIndices[0] == 0);
        CHECK(nearlyEqual(zero.boneWeights[0], 1.0f));

        std::vector<glm::dmat4> skin = {
            glm::translate(glm::dmat4(1.0), glm::dvec3(1, 0, 0)),
            glm::translate(glm::dmat4(1.0), glm::dvec3(0, 2, 0)),
            glm::dmat4(1.0),
        };
        sr::SkinnedVertex vertex = makeVertex(
            {0, 0, 0}, {0, 1, 0}, {{0, 0.25f}, {1, 0.75f}});
        const glm::dvec3 position = sr::skinPosition(vertex, skin);
        CHECK(nearlyEqual(position.x, 0.25, 1.0e-7));
        CHECK(nearlyEqual(position.y, 1.5, 1.0e-7));
        sr::validateVertexWeights(vertex, "test");
    });

    runTest("inverse-transpose normal under non-uniform scale", [] {
        glm::dmat4 scale(1.0);
        scale[0][0] = 2.0;
        scale[1][1] = 1.0;
        scale[2][2] = 0.5;
        std::vector<glm::dmat4> skin = {scale, scale, scale};

        sr::SkinnedVertex vertex = makeVertex(
            {0, 0, 0}, glm::normalize(glm::dvec3(1, 1, 0)), {{0, 1.0f}});
        const glm::dvec3 transformedNormal = sr::skinNormal(vertex, skin);
        const glm::dvec3 expected = glm::normalize(glm::dvec3(0.5, 1.0, 0.0));
        CHECK(glm::length(transformedNormal - expected) < 1.0e-9);

        const glm::dvec3 tangent = glm::normalize(glm::dvec3(1, -1, 0));
        const glm::dvec3 transformedTangent =
            glm::dvec3(scale * glm::dvec4(tangent, 0.0));
        CHECK(std::abs(glm::dot(transformedNormal, transformedTangent)) < 1.0e-9);

        const glm::dvec3 naive =
            glm::normalize(glm::vec3(scale * glm::dvec4(vertex.nx, vertex.ny,
                                                       vertex.nz, 0.0)));
        CHECK(std::abs(glm::dot(naive, transformedTangent)) > 0.5);
    });

    runTest("deterministic mesh generation, validation, and rebuild", [] {
        sr::DeterministicScene first(0);
        sr::DeterministicScene second(0);
        CHECK(first.characterBindMesh().vertices.size() ==
              second.characterBindMesh().vertices.size());
        CHECK(first.characterBindMesh().indices ==
              second.characterBindMesh().indices);
        CHECK(std::memcmp(first.characterBindMesh().vertices.data(),
                          second.characterBindMesh().vertices.data(),
                          first.characterBindMesh().vertices.size() *
                              sizeof(sr::SkinnedVertex)) == 0);

        const sr::MeshRevisionInfo info = first.revisionInfo();
        CHECK(info.vertexCount > 200);
        CHECK(info.triangleCount > 100);
        CHECK(info.maxInfluences <= 4);
        CHECK(info.maxInfluences >= 2);
        CHECK(nearlyEqual(info.minWeightSum, 1.0, 1.0e-5));
        CHECK(nearlyEqual(info.maxWeightSum, 1.0, 1.0e-5));

        const auto oldBounds = sr::meshBounds(first.characterBindMesh());
        first.rebuildCharacter(1);
        const auto newBounds = sr::meshBounds(first.characterBindMesh());
        CHECK(first.revisionInfo().revision == 1);
        CHECK(first.revisionInfo().variant == 1);
        CHECK(glm::length(newBounds.extent() - oldBounds.extent()) > 1.0e-4);

        const sr::PoseResult pose = sr::evaluatePose(first.skeleton(), 0.7);
        const sr::MeshData deformed = first.deformedCharacter(pose);
        CHECK(deformed.vertices.size() == first.characterBindMesh().vertices.size());
        CHECK(first.characterBounds(pose).valid);
        CHECK(first.visibleCasterBounds(pose).valid);
    });

    runTest("cascade splits, slice containment, and determinism", [] {
        sr::DeterministicScene scene(0);
        const sr::PoseResult pose = sr::evaluatePose(scene.skeleton(), 0.8);
        sr::Camera camera = sr::evaluateCamera(0.8);
        const sr::Bounds bounds = scene.visibleCasterBounds(pose);
        const sr::Lights lights = scene.lights();
        const sr::CascadeConfig config{2, 512, 0.65, 1.25};
        const sr::CascadeSet cascades = sr::buildCascades(
            camera, bounds, lights.directional, config);
        const sr::CascadeSet repeated = sr::buildCascades(
            camera, bounds, lights.directional, config);
        CHECK(cascades.cascades.size() == 2);
        CHECK(cascades.cascades[0].splitNear == camera.nearPlane);
        CHECK(cascades.cascades[0].splitFar < cascades.cascades[1].splitFar);
        CHECK(cascades.cascades[1].splitFar <= cascades.sceneFar + 1.0e-9);
        for (std::size_t i = 0; i < cascades.cascades.size(); ++i) {
            CHECK(cascades.cascades[i].radius > 0.0);
            CHECK(cascades.cascades[i].texelWorldSize > 0.0);
            for (int column = 0; column < 4; ++column) {
                for (int row = 0; row < 4; ++row) {
                    CHECK(cascades.cascades[i].lightViewProjection[column][row] ==
                          repeated.cascades[i].lightViewProjection[column][row]);
                }
            }
        }

        const glm::dmat4 inverseViewProjection =
            glm::inverse(camera.viewProjection);
        for (std::size_t cascadeIndex = 0;
             cascadeIndex < cascades.cascades.size(); ++cascadeIndex) {
            const sr::Cascade& cascade = cascades.cascades[cascadeIndex];
            for (double depth : {cascade.splitNear, cascade.splitFar}) {
                const double fraction =
                    (depth - camera.nearPlane) /
                    (camera.farPlane - camera.nearPlane);
                for (int corner = 0; corner < 4; ++corner) {
                    const double ndcX = (corner == 0 || corner == 3) ? -1.0 : 1.0;
                    const double ndcY = corner < 2 ? -1.0 : 1.0;
                    glm::dvec4 nearH = inverseViewProjection *
                        glm::dvec4(ndcX, ndcY, -1.0, 1.0);
                    glm::dvec4 farH = inverseViewProjection *
                        glm::dvec4(ndcX, ndcY, 1.0, 1.0);
                    const glm::dvec3 nearPoint = glm::dvec3(nearH) / nearH.w;
                    const glm::dvec3 farPoint = glm::dvec3(farH) / farH.w;
                    const glm::dvec3 point =
                        nearPoint + (farPoint - nearPoint) * fraction;
                    const glm::dvec4 projected =
                        cascade.lightViewProjection * glm::dvec4(point, 1.0);
                    const glm::dvec3 ndc = glm::dvec3(projected) / projected.w;
                    CHECK(ndc.x >= -1.0 - 1.0e-6 && ndc.x <= 1.0 + 1.0e-6);
                    CHECK(ndc.y >= -1.0 - 1.0e-6 && ndc.y <= 1.0 + 1.0e-6);
                    CHECK(ndc.z >= -1.0 - 1.0e-6 && ndc.z <= 1.0 + 1.0e-6);
                }
            }
        }

        bool rejected = false;
        try {
            sr::validateCascadeConfig(sr::CascadeConfig{1, 512, 0.65, 1.25});
        } catch (const sr::AppError& error) {
            rejected = error.code() == sr::ExitCode::Cascade;
        }
        CHECK(rejected);
    });

    runTest("16-bit depth PNG framing and deterministic hash", [] {
        std::vector<float> depth(16 * 8);
        for (std::size_t i = 0; i < depth.size(); ++i) {
            depth[i] = static_cast<float>(i) /
                       static_cast<float>(depth.size() - 1);
        }
        const std::filesystem::path path =
            std::filesystem::temp_directory_path() / "sr_depth_16_test.png";
        sr::writeDepthPng16(path, 16, 8, depth);
        std::ifstream stream(path, std::ios::binary);
        std::vector<std::uint8_t> bytes(
            (std::istreambuf_iterator<char>(stream)),
            std::istreambuf_iterator<char>());
        CHECK(bytes.size() > 40);
        const std::uint8_t signature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
        CHECK(std::memcmp(bytes.data(), signature, 8) == 0);
        CHECK(bytes[24] == 16);
        CHECK(bytes[25] == 0);
        const std::uint64_t firstHash = sr::fnv1a64(bytes.data(), bytes.size());
        sr::writeDepthPng16(path, 16, 8, depth);
        std::ifstream stream2(path, std::ios::binary);
        std::vector<std::uint8_t> bytes2(
            (std::istreambuf_iterator<char>(stream2)),
            std::istreambuf_iterator<char>());
        CHECK(firstHash == sr::fnv1a64(bytes2.data(), bytes2.size()));
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    });

    if (g_failures != 0) {
        std::cerr << g_failures << " failure(s) across " << g_checks
                  << " checks\n";
        return 1;
    }
    std::cout << "All " << g_checks << " checks passed\n";
    return 0;
}
