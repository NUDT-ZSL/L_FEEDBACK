#include "subdivision.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace subdiv {
namespace {

using EdgeKey = std::pair<std::uint32_t, std::uint32_t>;

EdgeKey edgeKey(std::uint32_t a, std::uint32_t b) {
    return a < b ? EdgeKey{a, b} : EdgeKey{b, a};
}

std::vector<std::uint32_t> mergeSources(const std::vector<std::uint32_t>& a,
                                        const std::vector<std::uint32_t>& b) {
    std::vector<std::uint32_t> merged;
    merged.reserve(a.size() + b.size());
    std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(merged));
    return merged;
}

Vec3 averageFacePoint(const PolygonMesh& mesh, std::size_t faceIndex) {
    const auto& face = mesh.faces[faceIndex];
    Vec3 sum{};
    for (std::uint32_t index : face) {
        sum += mesh.positions[index];
    }
    return sum / static_cast<float>(face.size());
}

Vec3 vertexPoint(const PolygonMesh& mesh,
                 std::uint32_t vertex,
                 const std::vector<Vec3>& facePoints,
                 const std::vector<std::vector<std::uint32_t>>& incidentEdges,
                 const std::vector<std::vector<std::uint32_t>>& incidentFaces) {
    const Vec3 point = mesh.positions[vertex];
    std::vector<std::uint32_t> boundaryNeighbors;
    std::vector<std::uint32_t> creaseNeighbors;
    float creaseSharpness = 0.0f;

    for (std::uint32_t edgeIndex : incidentEdges[vertex]) {
        const EdgeData& edge = mesh.edges[edgeIndex];
        const std::uint32_t neighbor = edge.a == vertex ? edge.b : edge.a;
        if (edge.isBoundary()) {
            boundaryNeighbors.push_back(neighbor);
        }
        if (edge.creaseWeight < 0.999f) {
            creaseNeighbors.push_back(neighbor);
            creaseSharpness += 1.0f - edge.creaseWeight;
        }
    }

    if (!boundaryNeighbors.empty()) {
        if (boundaryNeighbors.size() == 1) {
            return point;
        }
        return (point * 6.0f + mesh.positions[boundaryNeighbors[0]] +
                mesh.positions[boundaryNeighbors[1]]) / 8.0f;
    }

    const std::size_t valence = incidentEdges[vertex].size();
    if (valence < 3 || incidentFaces[vertex].empty()) {
        return point;
    }

    Vec3 faceAverage{};
    for (std::uint32_t faceIndex : incidentFaces[vertex]) {
        faceAverage += facePoints[faceIndex];
    }
    faceAverage = faceAverage / static_cast<float>(incidentFaces[vertex].size());

    Vec3 edgeMidpointAverage{};
    for (std::uint32_t edgeIndex : incidentEdges[vertex]) {
        const EdgeData& edge = mesh.edges[edgeIndex];
        edgeMidpointAverage += (mesh.positions[edge.a] + mesh.positions[edge.b]) * 0.5f;
    }
    edgeMidpointAverage = edgeMidpointAverage / static_cast<float>(valence);

    const Vec3 smooth =
        (faceAverage + edgeMidpointAverage * 2.0f +
         point * static_cast<float>(valence - 3)) /
        static_cast<float>(valence);

    if (creaseNeighbors.empty()) {
        return smooth;
    }

    const float meanSharpness =
        clamp(creaseSharpness / static_cast<float>(creaseNeighbors.size()), 0.0f, 1.0f);
    if (creaseNeighbors.size() == 2) {
        const Vec3 crease =
            (point * 6.0f + mesh.positions[creaseNeighbors[0]] +
             mesh.positions[creaseNeighbors[1]]) /
            8.0f;
        return lerp(smooth, crease, meanSharpness);
    }
    if (creaseNeighbors.size() >= 3) {
        return lerp(smooth, point, meanSharpness);
    }
    return smooth;
}

}  // namespace

PolygonMesh subdivideCatmullClarkOnce(const PolygonMesh& input) {
    PolygonMesh output;
    if (input.faces.empty()) {
        return output;
    }

    const std::size_t oldVertexCount = input.positions.size();
    const std::size_t faceCount = input.faces.size();

    std::vector<Vec3> facePoints(faceCount);
    for (std::size_t faceIndex = 0; faceIndex < faceCount; ++faceIndex) {
        facePoints[faceIndex] = averageFacePoint(input, faceIndex);
    }

    std::vector<std::vector<std::uint32_t>> incidentEdges(oldVertexCount);
    std::vector<std::vector<std::uint32_t>> incidentFaces(oldVertexCount);
    for (std::uint32_t edgeIndex = 0;
         edgeIndex < static_cast<std::uint32_t>(input.edges.size()); ++edgeIndex) {
        const EdgeData& edge = input.edges[edgeIndex];
        if (edge.a < oldVertexCount && edge.b < oldVertexCount) {
            incidentEdges[edge.a].push_back(edgeIndex);
            incidentEdges[edge.b].push_back(edgeIndex);
        }
    }
    for (std::uint32_t faceIndex = 0; faceIndex < static_cast<std::uint32_t>(faceCount);
         ++faceIndex) {
        for (std::uint32_t vertex : input.faces[faceIndex]) {
            if (vertex < oldVertexCount) {
                incidentFaces[vertex].push_back(faceIndex);
            }
        }
    }

    std::vector<Vec3> newPositions;
    std::vector<std::vector<std::uint32_t>> newSources;
    std::vector<std::uint32_t> representativeOriginal;
    newPositions.reserve(oldVertexCount + faceCount + input.edges.size());
    newSources.reserve(oldVertexCount + faceCount + input.edges.size());
    representativeOriginal.reserve(oldVertexCount + faceCount + input.edges.size());

    for (std::uint32_t vertex = 0; vertex < static_cast<std::uint32_t>(oldVertexCount); ++vertex) {
        newPositions.push_back(
            vertexPoint(input, vertex, facePoints, incidentEdges, incidentFaces));
        newSources.push_back(input.sourceOriginalVertices[vertex]);
        representativeOriginal.push_back(input.representativeOriginalVertex[vertex]);
    }

    const std::size_t facePointOffset = newPositions.size();
    for (std::size_t faceIndex = 0; faceIndex < faceCount; ++faceIndex) {
        newPositions.push_back(facePoints[faceIndex]);
        std::vector<std::uint32_t> sources;
        for (std::uint32_t vertex : input.faces[faceIndex]) {
            sources = mergeSources(sources, input.sourceOriginalVertices[vertex]);
        }
        newSources.push_back(std::move(sources));
        representativeOriginal.push_back(
            newSources.back().empty() ? 0U : newSources.back().front());
    }

    const std::size_t edgePointOffset = newPositions.size();
    std::vector<Vec3> edgePoints(input.edges.size());
    for (std::size_t edgeIndex = 0; edgeIndex < input.edges.size(); ++edgeIndex) {
        const EdgeData& edge = input.edges[edgeIndex];
        const Vec3 midpoint = (input.positions[edge.a] + input.positions[edge.b]) * 0.5f;
        Vec3 edgePoint = midpoint;
        if (!edge.isBoundary()) {
            const Vec3 faceAverage =
                (facePoints[static_cast<std::size_t>(edge.faceA)] +
                 facePoints[static_cast<std::size_t>(edge.faceB)]) *
                0.5f;
            const Vec3 smooth = (midpoint + faceAverage) * 0.5f;
            edgePoint = lerp(midpoint, smooth, edge.creaseWeight);
        }
        edgePoints[edgeIndex] = edgePoint;
        newPositions.push_back(edgePoint);
        newSources.push_back(mergeSources(input.sourceOriginalVertices[edge.a],
                                          input.sourceOriginalVertices[edge.b]));
        representativeOriginal.push_back(std::min(
            input.representativeOriginalVertex[edge.a],
            input.representativeOriginalVertex[edge.b]));
    }

    std::map<EdgeKey, std::uint32_t> inputEdgeLookup;
    for (std::uint32_t edgeIndex = 0;
         edgeIndex < static_cast<std::uint32_t>(input.edges.size()); ++edgeIndex) {
        inputEdgeLookup.emplace(edgeKey(input.edges[edgeIndex].a, input.edges[edgeIndex].b),
                                edgeIndex);
    }

    const auto facePointVertex = [facePointOffset](std::size_t faceIndex) {
        return static_cast<std::uint32_t>(facePointOffset + faceIndex);
    };
    const auto edgePointVertex = [edgePointOffset](std::uint32_t edgeIndex) {
        return static_cast<std::uint32_t>(edgePointOffset + edgeIndex);
    };

    output.faces.reserve(faceCount * 4);
    for (std::size_t faceIndex = 0; faceIndex < faceCount; ++faceIndex) {
        const auto& face = input.faces[faceIndex];
        const std::size_t cornerCount = face.size();
        for (std::size_t corner = 0; corner < cornerCount; ++corner) {
            const std::uint32_t vertex = face[corner];
            const std::uint32_t next = face[(corner + 1) % cornerCount];
            const std::uint32_t previous = face[(corner + cornerCount - 1) % cornerCount];
            const auto nextEdge = inputEdgeLookup.find(edgeKey(vertex, next));
            const auto previousEdge = inputEdgeLookup.find(edgeKey(previous, vertex));
            if (nextEdge == inputEdgeLookup.end() || previousEdge == inputEdgeLookup.end()) {
                continue;
            }
            output.faces.push_back({
                vertex,
                edgePointVertex(nextEdge->second),
                facePointVertex(faceIndex),
                edgePointVertex(previousEdge->second),
            });
        }
    }

    output.positions = std::move(newPositions);
    output.sourceOriginalVertices = std::move(newSources);
    output.representativeOriginalVertex = std::move(representativeOriginal);
    output.edges = buildEdges(output.faces);
    for (EdgeData& edge : output.edges) {
        edge.creaseWeight = 1.0f;
        const bool aIsOldVertex = edge.a < oldVertexCount;
        const bool bIsOldVertex = edge.b < oldVertexCount;
        const bool aIsEdgePoint = edge.a >= edgePointOffset;
        const bool bIsEdgePoint = edge.b >= edgePointOffset;
        if ((aIsOldVertex && bIsEdgePoint) || (bIsOldVertex && aIsEdgePoint)) {
            const std::uint32_t oldVertex = aIsOldVertex ? edge.a : edge.b;
            const std::uint32_t edgePoint = aIsEdgePoint ? edge.a : edge.b;
            const std::uint32_t parentEdge = edgePoint - static_cast<std::uint32_t>(edgePointOffset);
            if (parentEdge < input.edges.size()) {
                const EdgeData& parent = input.edges[parentEdge];
                if (parent.a == oldVertex || parent.b == oldVertex) {
                    edge.creaseWeight = parent.creaseWeight;
                }
            }
        }
    }

    return output;
}

PolygonMesh subdivideCatmullClark(const PolygonMesh& base, int level) {
    PolygonMesh result = base;
    for (int i = 0; i < level; ++i) {
        result = subdivideCatmullClarkOnce(result);
    }
    return result;
}

}  // namespace subdiv

