#include "mesh.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

namespace subdiv {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hashBytes(std::uint64_t& hash, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= kFnvPrime;
    }
}

template <typename T>
void hashValue(std::uint64_t& hash, const T& value) {
    hashBytes(hash, &value, sizeof(T));
}

std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

float deterministicNoise(std::uint64_t seed, int column, int row) {
    const std::uint64_t key = seed ^
        (static_cast<std::uint64_t>(column + 64) << 32U) ^
        static_cast<std::uint64_t>(row + 64);
    const std::uint64_t bits = mix64(key);
    const float unit = static_cast<float>(bits >> 40U) / 16777216.0f;
    return unit * 2.0f - 1.0f;
}

}  // namespace

TriangleMesh buildDeterministicMesh(std::uint64_t seed) {
    constexpr int kColumns = 7;
    constexpr int kRows = 5;
    constexpr float kHalfWidth = 1.55f;
    constexpr float kHalfLength = 2.15f;

    TriangleMesh mesh;
    mesh.positions.reserve(kColumns * kRows);

    for (int row = 0; row < kRows; ++row) {
        const float v = static_cast<float>(row) / static_cast<float>(kRows - 1);
        const float z = (v * 2.0f - 1.0f) * kHalfLength;
        for (int column = 0; column < kColumns; ++column) {
            const float u = static_cast<float>(column) / static_cast<float>(kColumns - 1);
            const float x = (u * 2.0f - 1.0f) * kHalfWidth;
            const float centerRidge = 0.42f * (1.0f - std::fabs(v * 2.0f - 1.0f));
            const float sideBowl = 0.58f * (u * 2.0f - 1.0f) * (u * 2.0f - 1.0f);
            const float interior = (row > 0 && row < kRows - 1 &&
                                    column > 0 && column < kColumns - 1)
                ? 0.055f * deterministicNoise(seed, column, row)
                : 0.0f;
            mesh.positions.push_back({x, centerRidge + sideBowl + interior, z});
        }
    }

    const auto index = [](int column, int row) {
        return static_cast<std::uint32_t>(row * kColumns + column);
    };

    for (int row = 0; row < kRows - 1; ++row) {
        for (int column = 0; column < kColumns - 1; ++column) {
            const std::uint32_t a = index(column, row);
            const std::uint32_t b = index(column + 1, row);
            const std::uint32_t c = index(column, row + 1);
            const std::uint32_t d = index(column + 1, row + 1);
            if (((column + row) & 1) == 0) {
                mesh.indices.insert(mesh.indices.end(), {a, d, b, a, c, d});
            } else {
                mesh.indices.insert(mesh.indices.end(), {a, c, b, b, c, d});
            }
        }
    }

    mesh.faceNormals.reserve(mesh.indices.size() / 3);
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        const Vec3& a = mesh.positions[mesh.indices[i]];
        const Vec3& b = mesh.positions[mesh.indices[i + 1]];
        const Vec3& c = mesh.positions[mesh.indices[i + 2]];
        mesh.faceNormals.push_back(normalize(cross(b - a, c - a)));
    }

    std::vector<std::vector<std::uint32_t>> faces;
    faces.reserve(mesh.indices.size() / 3);
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        faces.push_back({mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]});
    }
    mesh.edges = buildEdges(faces);

    for (EdgeData& edge : mesh.edges) {
        const Vec3& a = mesh.positions[edge.a];
        const Vec3& b = mesh.positions[edge.b];
        if (edge.isBoundary()) {
            edge.creaseWeight = 0.0f;
        } else if (std::fabs(a.z) < 1.0e-5f && std::fabs(b.z) < 1.0e-5f) {
            edge.creaseWeight = 0.12f;
        } else if ((mix64(seed ^ (static_cast<std::uint64_t>(edge.a) << 32U) ^ edge.b) & 7ULL) == 0ULL) {
            edge.creaseWeight = 0.42f;
        } else {
            edge.creaseWeight = 1.0f;
        }
    }

    return mesh;
}

PolygonMesh triangleMeshToPolygonMesh(const TriangleMesh& mesh) {
    PolygonMesh result;
    result.positions = mesh.positions;
    result.faces.reserve(mesh.indices.size() / 3);

    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        result.faces.push_back({mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]});
    }

    result.edges = mesh.edges;
    result.sourceOriginalVertices.resize(mesh.positions.size());
    result.representativeOriginalVertex.resize(mesh.positions.size());
    for (std::uint32_t i = 0; i < mesh.positions.size(); ++i) {
        result.sourceOriginalVertices[i] = {i};
        result.representativeOriginalVertex[i] = i;
    }
    return result;
}

std::vector<EdgeData> buildEdges(const std::vector<std::vector<std::uint32_t>>& faces) {
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::size_t> lookup;
    std::vector<EdgeData> edges;

    for (std::size_t faceIndex = 0; faceIndex < faces.size(); ++faceIndex) {
        const auto& face = faces[faceIndex];
        if (face.size() < 3) {
            continue;
        }
        for (std::size_t i = 0; i < face.size(); ++i) {
            const std::uint32_t a = face[i];
            const std::uint32_t b = face[(i + 1) % face.size()];
            const std::pair<std::uint32_t, std::uint32_t> key =
                a < b ? std::make_pair(a, b) : std::make_pair(b, a);
            const auto found = lookup.find(key);
            if (found == lookup.end()) {
                EdgeData edge;
                edge.a = key.first;
                edge.b = key.second;
                edge.faceA = static_cast<int>(faceIndex);
                lookup.emplace(key, edges.size());
                edges.push_back(edge);
            } else {
                EdgeData& edge = edges[found->second];
                if (edge.faceA < 0) {
                    edge.faceA = static_cast<int>(faceIndex);
                } else if (edge.faceB < 0) {
                    edge.faceB = static_cast<int>(faceIndex);
                } else {
                    std::cerr << "[mesh] warning: non-manifold edge ("
                              << edge.a << ", " << edge.b << ") has more than two faces\n";
                }
            }
        }
    }

    std::sort(edges.begin(), edges.end(), [](const EdgeData& lhs, const EdgeData& rhs) {
        if (lhs.a != rhs.a) {
            return lhs.a < rhs.a;
        }
        return lhs.b < rhs.b;
    });
    return edges;
}

std::vector<Vec3> computePolygonFaceNormals(const PolygonMesh& mesh) {
    std::vector<Vec3> normals(mesh.faces.size(), Vec3{});
    for (std::size_t faceIndex = 0; faceIndex < mesh.faces.size(); ++faceIndex) {
        const auto& face = mesh.faces[faceIndex];
        if (face.size() < 3) {
            continue;
        }
        const Vec3 origin = mesh.positions[face[0]];
        Vec3 normal{};
        for (std::size_t i = 1; i + 1 < face.size(); ++i) {
            const Vec3 edgeA = mesh.positions[face[i]] - origin;
            const Vec3 edgeB = mesh.positions[face[i + 1]] - origin;
            normal += cross(edgeA, edgeB);
        }
        normals[faceIndex] = normalize(normal);
    }
    return normals;
}

MeshStats computeMeshStats(const PolygonMesh& mesh, std::size_t originalTriangleCount) {
    MeshStats stats;
    stats.vertices = mesh.positions.size();
    stats.triangles = originalTriangleCount;
    stats.quads = 0;
    stats.boundaryEdges = 0;
    stats.sharpEdges = 0;
    stats.traceableVertices = 0;

    for (const auto& face : mesh.faces) {
        if (face.size() == 4) {
            ++stats.quads;
        }
    }
    for (const EdgeData& edge : mesh.edges) {
        if (edge.isBoundary()) {
            ++stats.boundaryEdges;
        }
        if (edge.creaseWeight < 0.999f) {
            ++stats.sharpEdges;
        }
    }
    for (const auto& sources : mesh.sourceOriginalVertices) {
        if (!sources.empty()) {
            ++stats.traceableVertices;
        }
    }
    return stats;
}

void applyCreaseOverrides(TriangleMesh& mesh,
                          const std::vector<std::pair<int, float>>& overrides,
                          const std::string& warningPrefix) {
    for (const auto& overrideValue : overrides) {
        const int edgeIndex = overrideValue.first;
        const float weight = overrideValue.second;
        if (edgeIndex < 0 || static_cast<std::size_t>(edgeIndex) >= mesh.edges.size()) {
            std::cout << "[scene] warning: " << warningPrefix << " edge index " << edgeIndex
                      << " is out of range; using built-in edge weights\n";
            continue;
        }
        if (!std::isfinite(weight) || weight < 0.0f || weight > 1.0f) {
            std::cout << "[scene] warning: " << warningPrefix << " edge " << edgeIndex
                      << " value is outside [0,1]; using built-in edge weight\n";
            continue;
        }
        mesh.edges[static_cast<std::size_t>(edgeIndex)].creaseWeight = weight;
    }
}

std::uint64_t fingerprint(const TriangleMesh& mesh) {
    std::uint64_t hash = kFnvOffset;
    hashValue(hash, mesh.positions.size());
    hashValue(hash, mesh.indices.size());
    for (const Vec3& position : mesh.positions) {
        hashValue(hash, position.x);
        hashValue(hash, position.y);
        hashValue(hash, position.z);
    }
    for (std::uint32_t index : mesh.indices) {
        hashValue(hash, index);
    }
    for (const Vec3& normal : mesh.faceNormals) {
        hashValue(hash, normal.x);
        hashValue(hash, normal.y);
        hashValue(hash, normal.z);
    }
    for (const EdgeData& edge : mesh.edges) {
        hashValue(hash, edge.a);
        hashValue(hash, edge.b);
        hashValue(hash, edge.creaseWeight);
    }
    return hash;
}

std::uint64_t fingerprint(const PolygonMesh& mesh) {
    std::uint64_t hash = kFnvOffset;
    hashValue(hash, mesh.positions.size());
    hashValue(hash, mesh.faces.size());
    for (const Vec3& position : mesh.positions) {
        hashValue(hash, position.x);
        hashValue(hash, position.y);
        hashValue(hash, position.z);
    }
    for (const auto& face : mesh.faces) {
        const std::uint32_t size = static_cast<std::uint32_t>(face.size());
        hashValue(hash, size);
        for (std::uint32_t index : face) {
            hashValue(hash, index);
        }
    }
    for (const EdgeData& edge : mesh.edges) {
        hashValue(hash, edge.a);
        hashValue(hash, edge.b);
        hashValue(hash, edge.creaseWeight);
    }
    for (const auto& sources : mesh.sourceOriginalVertices) {
        const std::uint32_t size = static_cast<std::uint32_t>(sources.size());
        hashValue(hash, size);
        for (std::uint32_t source : sources) {
            hashValue(hash, source);
        }
    }
    return hash;
}

}  // namespace subdiv

