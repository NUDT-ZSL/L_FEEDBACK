#pragma once

#include "asset.hpp"
#include "error.hpp"

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace skel {

struct FrameStats {
    std::uint32_t frame = 0;
    std::uint64_t nanPositions = 0;
    std::uint64_t nanNormals = 0;
    std::uint32_t normalizedVertices = 0;
};

struct BakedFrame {
    std::uint32_t frame = 0;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> colors;
    std::vector<std::uint32_t> indices;
    std::vector<glm::mat4> skinMatrices;
    FrameStats stats;
};

struct InterleavedVertex {
    float position[3];
    float normal[3];
    float color[3];
};

// Composes one local transform. The quaternion is normalized before use.
bool composeLocalTransform(CompositionOrder order,
                           const glm::vec3& translation,
                           const glm::quat& rotation,
                           const glm::vec3& scale,
                           glm::mat4& result,
                           Error& error);

// Shared scene evaluation used by both the OpenGL and offline paths.
bool evaluateFrame(const Asset& asset, std::uint32_t frame,
                   BakedFrame& baked, Error& error);

// Shared upload/interleave conversion used by the OpenGL path and by
// consistency tests. No topology or weighting happens here.
std::vector<InterleavedVertex> makeInterleavedVertices(
    const BakedFrame& baked);

} // namespace skel

