#include "mesh_io.hpp"
#include "model.hpp"
#include "render.hpp"
#include "sha256.hpp"
#include "skinning.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(std::string("check failed: ") + message);
    }
}

bool closeEnough(double lhs, double rhs, double epsilon = 1e-9) {
    return std::abs(lhs - rhs) <= epsilon;
}

void testSha256() {
    const std::string input = "abc";
    const std::string digest = rigskin::Sha256::digestHex(
        std::vector<std::uint8_t>(input.begin(), input.end()));
    check(
        digest ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 known vector");
}

void testModelAndPose() {
    const std::string ascii = rigskin::generateAsciiSample("preview");
    const rigskin::Model model = rigskin::parseAsciiModel(ascii);

    check(model.bones.size() == 13, "bone count");
    check(model.vertices.size() > 100, "vertex count");
    check(model.triangles.size() > 100, "triangle count");
    check(model.zeroWeightVertices.size() == 1, "zero-weight diagnostic vertex");
    check(model.evalOrder.size() == model.bones.size(), "all bones visited");
    check(model.evalOrder.front() == 0, "root evaluated first");

    for (std::size_t orderIndex = 0; orderIndex < model.evalOrder.size(); ++orderIndex) {
        const int bone = model.evalOrder[orderIndex];
        const int parent = model.bones[static_cast<std::size_t>(bone)].parent;
        if (parent < 0) {
            continue;
        }
        const auto parentPosition = std::find(
            model.evalOrder.begin(), model.evalOrder.end(), parent);
        check(
            parentPosition < model.evalOrder.begin() + static_cast<std::ptrdiff_t>(orderIndex),
            "parent precedes child");
    }

    for (const rigskin::MeshVertex& vertex : model.vertices) {
        if (vertex.weights.empty()) {
            continue;
        }
        double sum = 0.0;
        for (const rigskin::SkinWeight& weight : vertex.weights) {
            sum += weight.weight;
        }
        check(closeEnough(sum, 1.0), "normalized vertex weights");
    }

    const rigskin::Bone& root = model.bones[0];
    check(root.keyframes.size() >= 2, "root has keyframes");
    const rigskin::Keyframe& first = root.keyframes[0];
    const rigskin::Keyframe& second = root.keyframes[1];
    const double sampleTime = (first.time + second.time) * 0.5;
    std::vector<rigskin::Mat4> locals;
    const auto globals =
        rigskin::evaluatePoseGlobals(model, sampleTime, &locals);
    check(globals.size() == model.bones.size(), "global pose size");

    const rigskin::Vec3 sampled = rigskin::transformPoint(locals[0], {0.0, 0.0, 0.0});
    const rigskin::Vec3 expected =
        rigskin::lerp(first.translation, second.translation, 0.5);
    check(closeEnough(sampled.x, expected.x), "nonuniform key interpolation x");
    check(closeEnough(sampled.y, expected.y, 1e-8), "nonuniform key interpolation y");
    check(closeEnough(sampled.z, expected.z), "nonuniform key interpolation z");

    for (std::size_t orderIndex = 1; orderIndex < model.evalOrder.size(); ++orderIndex) {
        const int bone = model.evalOrder[orderIndex];
        const int parent = model.bones[static_cast<std::size_t>(bone)].parent;
        const rigskin::Mat4 expectedGlobal =
            globals[static_cast<std::size_t>(parent)] *
            locals[static_cast<std::size_t>(bone)];
        const rigskin::Vec3 expectedPoint =
            rigskin::transformPoint(expectedGlobal, {0.0, 0.0, 0.0});
        const rigskin::Vec3 actualPoint =
            rigskin::transformPoint(globals[static_cast<std::size_t>(bone)], {0.0, 0.0, 0.0});
        check(closeEnough(actualPoint.x, expectedPoint.x), "global hierarchy x");
        check(closeEnough(actualPoint.y, expectedPoint.y), "global hierarchy y");
        check(closeEnough(actualPoint.z, expectedPoint.z), "global hierarchy z");
    }

    const rigskin::DeformedMesh bindLike =
        rigskin::skinMesh(model, rigskin::evaluatePoseGlobals(model, 0.0));
    const rigskin::DeformedMesh deformed =
        rigskin::skinMesh(model, rigskin::evaluatePoseGlobals(model, 1.17));
    check(
        bindLike.positions.size() == model.vertices.size(),
        "deformed vertex count");
    check(
        bindLike.faceNormals.size() == model.triangles.size(),
        "face normal count");
    check(
        bindLike.degenerateTriangleCount >= 1,
        "diagnostic degenerate triangle detected");
    check(
        model.zeroWeightVertices.size() == 1,
        "zero-weight vertex count for retention test");
    const std::uint32_t zeroVertex = model.zeroWeightVertices.front();
    const rigskin::Vec3& retained = deformed.positions[zeroVertex];
    const rigskin::Vec3& bind = model.vertices[zeroVertex].bindPosition;
    check(closeEnough(retained.x, bind.x), "zero-weight x retained");
    check(closeEnough(retained.y, bind.y), "zero-weight y retained");
    check(closeEnough(retained.z, bind.z), "zero-weight z retained");
    for (std::size_t triangleIndex = 0; triangleIndex < model.triangles.size();
         ++triangleIndex) {
        const auto& triangle = model.triangles[triangleIndex];
        const auto& normal = bindLike.faceNormals[triangleIndex];
        const rigskin::Vec3& a = bindLike.positions[triangle.a];
        const rigskin::Vec3& b = bindLike.positions[triangle.b];
        const rigskin::Vec3& c = bindLike.positions[triangle.c];
        const rigskin::Vec3 windingNormal = rigskin::normalized(
            rigskin::cross(b - a, c - a));
        if (rigskin::lengthSquared(windingNormal) <= 1e-24) {
            continue;
        }
        check(
            rigskin::dot(normal, windingNormal) > 0.999999,
            "face normal follows triangle winding");
    }

    const rigskin::MeshHeader headerWithoutNormals = {};
    check(headerWithoutNormals.frame == 0, "mesh header default");
    const std::vector<std::uint8_t> meshBytes =
        rigskin::encodeMeshBinary(model, 3, bindLike);
    rigskin::MeshHeader header;
    std::string error;
    check(rigskin::decodeMeshHeader(meshBytes, header, error), "mesh header decode");
    check(header.frame == 3, "mesh frame number");
    check(header.vertexCount == model.vertices.size(), "mesh vertex count");
    check(header.triangleCount == model.triangles.size(), "mesh triangle count");

    const rigskin::Camera camera = rigskin::makeCamera(model, 64, 48);
    const rigskin::RenderedImage image =
        rigskin::renderMesh(model, bindLike, camera, 64, 48);
    check(image.rgb.size() == 64 * 48 * 3, "RGB buffer size");
    const std::vector<std::uint8_t> ppm =
        rigskin::encodePpm(image.width, image.height, image.rgb);
    check(rigskin::validatePpmHeader(ppm, 64, 48, error), "PPM header");
}

}  // namespace

int main() {
    try {
        testSha256();
        testModelAndPose();
        std::cout << "rigskin unit tests: PASS\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "rigskin unit tests: FAIL: " << exception.what() << '\n';
        return 1;
    }
}
