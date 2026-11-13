#pragma once

#include "model.hpp"

#include <vector>

namespace rigskin {

struct DeformedMesh {
    std::vector<Vec3> positions;
    std::vector<Vec3> vertexNormals;
    std::vector<Vec3> faceNormals;
    std::uint32_t degenerateTriangleCount = 0;
};

DeformedMesh skinMesh(
    const Model& model,
    const std::vector<Mat4>& poseGlobals);

}  // namespace rigskin

