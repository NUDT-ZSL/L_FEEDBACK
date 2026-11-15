#pragma once

#include "math_types.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace subdiv {

struct EdgeData {
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    float creaseWeight = 1.0f;
    int faceA = -1;
    int faceB = -1;

    bool isBoundary() const {
        return faceB < 0;
    }
};

struct PolygonMesh {
    std::vector<Vec3> positions;
    std::vector<std::vector<std::uint32_t>> faces;
    std::vector<EdgeData> edges;
    std::vector<std::vector<std::uint32_t>> sourceOriginalVertices;
    std::vector<std::uint32_t> representativeOriginalVertex;
};

struct TriangleMesh {
    std::vector<Vec3> positions;
    std::vector<std::uint32_t> indices;
    std::vector<Vec3> faceNormals;
    std::vector<EdgeData> edges;
};

struct MeshStats {
    std::size_t vertices = 0;
    std::size_t triangles = 0;
    std::size_t quads = 0;
    std::size_t boundaryEdges = 0;
    std::size_t sharpEdges = 0;
    std::size_t traceableVertices = 0;
};

TriangleMesh buildDeterministicMesh(std::uint64_t seed);
PolygonMesh triangleMeshToPolygonMesh(const TriangleMesh& mesh);
std::vector<EdgeData> buildEdges(const std::vector<std::vector<std::uint32_t>>& faces);
std::vector<Vec3> computePolygonFaceNormals(const PolygonMesh& mesh);
MeshStats computeMeshStats(const PolygonMesh& mesh, std::size_t originalTriangleCount);

void applyCreaseOverrides(TriangleMesh& mesh,
                          const std::vector<std::pair<int, float>>& overrides,
                          const std::string& warningPrefix);

std::uint64_t fingerprint(const TriangleMesh& mesh);
std::uint64_t fingerprint(const PolygonMesh& mesh);

}  // namespace subdiv

