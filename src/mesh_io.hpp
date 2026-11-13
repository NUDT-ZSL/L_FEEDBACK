#pragma once

#include "model.hpp"
#include "skinning.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace rigskin {

struct MeshHeader {
    std::uint32_t frame = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t triangleCount = 0;
    std::uint32_t degenerateTriangleCount = 0;
    std::uint32_t zeroWeightVertexCount = 0;
};

std::vector<std::uint8_t> encodeMeshBinary(
    const Model& model,
    std::uint32_t frameIndex,
    const DeformedMesh& mesh);

bool decodeMeshHeader(
    const std::vector<std::uint8_t>& bytes,
    MeshHeader& header,
    std::string& error);

std::vector<std::uint8_t> encodePpm(
    int width,
    int height,
    const std::vector<std::uint8_t>& rgb);

bool validatePpmHeader(
    const std::vector<std::uint8_t>& bytes,
    int expectedWidth,
    int expectedHeight,
    std::string& error);

}  // namespace rigskin

