#include "skinning.hpp"

#include <stdexcept>

namespace rigskin {

DeformedMesh skinMesh(
    const Model& model,
    const std::vector<Mat4>& poseGlobals) {
    if (poseGlobals.size() != model.bones.size()) {
        throw std::runtime_error("pose bone count does not match model");
    }

    std::vector<Mat4> skinMatrices(model.bones.size());
    for (std::size_t boneIndex = 0; boneIndex < model.bones.size(); ++boneIndex) {
        skinMatrices[boneIndex] =
            poseGlobals[boneIndex] *
            inverse(model.bindGlobals[boneIndex]);
    }

    DeformedMesh deformed;
    deformed.positions.resize(model.vertices.size());
    deformed.vertexNormals.resize(model.vertices.size());

    for (std::size_t vertexIndex = 0; vertexIndex < model.vertices.size();
         ++vertexIndex) {
        const MeshVertex& vertex = model.vertices[vertexIndex];
        if (vertex.weights.empty()) {
            deformed.positions[vertexIndex] = vertex.bindPosition;
            deformed.vertexNormals[vertexIndex] = vertex.bindNormal;
            continue;
        }

        Vec3 position{0.0, 0.0, 0.0};
        Vec3 normal{0.0, 0.0, 0.0};
        for (const SkinWeight& weight : vertex.weights) {
            const Mat4& matrix = skinMatrices[weight.bone];
            position += transformPoint(matrix, vertex.bindPosition) * weight.weight;
            normal += transformVector(matrix, vertex.bindNormal) * weight.weight;
        }

        deformed.positions[vertexIndex] = position;
        deformed.vertexNormals[vertexIndex] =
            lengthSquared(normal) > 1e-24 ? normalized(normal) : Vec3{0.0, 1.0, 0.0};
    }

    deformed.faceNormals.resize(model.triangles.size());
    for (std::size_t triangleIndex = 0; triangleIndex < model.triangles.size();
         ++triangleIndex) {
        const Triangle& triangle = model.triangles[triangleIndex];
        const Vec3& a = deformed.positions[triangle.a];
        const Vec3& b = deformed.positions[triangle.b];
        const Vec3& c = deformed.positions[triangle.c];
        const Vec3 normal = cross(b - a, c - a);
        if (lengthSquared(normal) <= 1e-24) {
            deformed.faceNormals[triangleIndex] = {0.0, 0.0, 0.0};
            ++deformed.degenerateTriangleCount;
        } else {
            deformed.faceNormals[triangleIndex] = normalized(normal);
        }
    }

    return deformed;
}

}  // namespace rigskin

