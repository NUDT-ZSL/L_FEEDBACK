#include "json.h"
#include "mesh.h"
#include "scene.h"
#include "subdivision.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << '\n';
        ++failures;
    }
}

bool near(float a, float b, float epsilon = 1.0e-5f) {
    return std::fabs(a - b) <= epsilon;
}

bool near(const subdiv::Vec3& a, const subdiv::Vec3& b, float epsilon = 1.0e-5f) {
    return near(a.x, b.x, epsilon) && near(a.y, b.y, epsilon) &&
           near(a.z, b.z, epsilon);
}

bool finite(const subdiv::Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

subdiv::Vec3 faceAverage(const subdiv::PolygonMesh& mesh, std::size_t faceIndex) {
    const auto& face = mesh.faces[faceIndex];
    subdiv::Vec3 sum{};
    for (std::uint32_t index : face) {
        sum += mesh.positions[index];
    }
    return sum / static_cast<float>(face.size());
}

void testDeterministicConstruction() {
    const subdiv::TriangleMesh first = subdiv::buildDeterministicMesh(20251029ULL);
    const subdiv::TriangleMesh second = subdiv::buildDeterministicMesh(20251029ULL);
    const subdiv::TriangleMesh other = subdiv::buildDeterministicMesh(77ULL);

    check(subdiv::fingerprint(first) == subdiv::fingerprint(second),
          "same seed must produce an identical mesh fingerprint");
    check(first.positions.size() == second.positions.size(),
          "same seed must produce identical vertex counts");
    check(first.indices == second.indices,
          "same seed must produce identical face indices");
    check(first.faceNormals.size() == first.indices.size() / 3,
          "one face normal per input triangle");
    check(subdiv::fingerprint(first) != subdiv::fingerprint(other),
          "different seeds should alter deterministic sample geometry");
    check(first.indices.size() >= 3 && first.indices.size() % 3 == 0,
          "input sample must be a triangle mesh");

    std::size_t boundaryCount = 0;
    std::size_t creaseCount = 0;
    for (const auto& edge : first.edges) {
        boundaryCount += edge.isBoundary() ? 1U : 0U;
        creaseCount += edge.creaseWeight < 0.999f ? 1U : 0U;
    }
    check(boundaryCount > 0, "sample must contain open boundary edges");
    check(creaseCount > 0, "sample must contain specified sharp/crease edges");

    for (const subdiv::Vec3& position : first.positions) {
        check(finite(position), "constructed positions must be finite");
    }
    for (const subdiv::Vec3& normal : first.faceNormals) {
        check(finite(normal) && near(subdiv::length(normal), 1.0f),
              "constructed face normals must be finite and unit length");
    }
}

void testCreaseOverride() {
    subdiv::TriangleMesh mesh = subdiv::buildDeterministicMesh(20251029ULL);
    const int edgeIndex = static_cast<int>(mesh.edges.size() / 2);
    const float previous = mesh.edges[static_cast<std::size_t>(edgeIndex)].creaseWeight;
    subdiv::applyCreaseOverrides(mesh, {{edgeIndex, 0.0f}}, "test");
    check(mesh.edges[static_cast<std::size_t>(edgeIndex)].creaseWeight == 0.0f,
          "valid scene crease override must replace the built-in weight");

    subdiv::applyCreaseOverrides(mesh, {{edgeIndex, 3.0f}}, "test");
    check(mesh.edges[static_cast<std::size_t>(edgeIndex)].creaseWeight == 0.0f,
          "out-of-range crease override must retain the current weight");

    mesh.edges[static_cast<std::size_t>(edgeIndex)].creaseWeight = previous;
}

void testCatmullClarkRules() {
    const subdiv::TriangleMesh triangles = subdiv::buildDeterministicMesh(20251029ULL);
    const subdiv::PolygonMesh base = subdiv::triangleMeshToPolygonMesh(triangles);
    const subdiv::PolygonMesh levelOne = subdiv::subdivideCatmullClarkOnce(base);

    const std::size_t triangleCount = triangles.indices.size() / 3;
    check(levelOne.faces.size() == triangleCount * 3,
          "one triangle must split into exactly three quads");
    for (const auto& face : levelOne.faces) {
        check(face.size() == 4, "every first-iteration output face must be a quad");
    }
    check(levelOne.positions.size() ==
              base.positions.size() + base.faces.size() + base.edges.size(),
          "first iteration must emit V + F + E vertices");

    const std::size_t facePointOffset = base.positions.size();
    const std::size_t edgePointOffset = base.positions.size() + base.faces.size();
    for (std::size_t faceIndex = 0; faceIndex < base.faces.size(); ++faceIndex) {
        const subdiv::Vec3 expected = faceAverage(base, faceIndex);
        check(near(levelOne.positions[facePointOffset + faceIndex], expected),
              "face point must be the polygon vertex average");
    }

    for (std::size_t edgeIndex = 0; edgeIndex < base.edges.size(); ++edgeIndex) {
        const subdiv::EdgeData& edge = base.edges[edgeIndex];
        const subdiv::Vec3 midpoint =
            (base.positions[edge.a] + base.positions[edge.b]) * 0.5f;
        subdiv::Vec3 expected = midpoint;
        if (!edge.isBoundary()) {
            const subdiv::Vec3 faceAverageValue =
                (faceAverage(base, static_cast<std::size_t>(edge.faceA)) +
                 faceAverage(base, static_cast<std::size_t>(edge.faceB))) *
                0.5f;
            expected = subdiv::lerp(midpoint, (midpoint + faceAverageValue) * 0.5f,
                                    edge.creaseWeight);
        }
        check(near(levelOne.positions[edgePointOffset + edgeIndex], expected),
              "edge point must follow 50/50 smooth rule, crease blend, or boundary midpoint");
    }

    for (std::size_t vertexIndex = 0; vertexIndex < base.positions.size(); ++vertexIndex) {
        check(finite(levelOne.positions[vertexIndex]),
              "vertex point must remain finite");
    }

    std::vector<std::vector<std::uint32_t>> boundaryNeighbors(base.positions.size());
    for (const auto& edge : base.edges) {
        if (edge.isBoundary()) {
            boundaryNeighbors[edge.a].push_back(edge.b);
            boundaryNeighbors[edge.b].push_back(edge.a);
        }
    }
    for (std::size_t vertexIndex = 0; vertexIndex < boundaryNeighbors.size(); ++vertexIndex) {
        if (boundaryNeighbors[vertexIndex].size() == 2) {
            const subdiv::Vec3 expected =
                (base.positions[vertexIndex] * 6.0f +
                 base.positions[boundaryNeighbors[vertexIndex][0]] +
                 base.positions[boundaryNeighbors[vertexIndex][1]]) /
                8.0f;
            check(near(levelOne.positions[vertexIndex], expected),
                  "boundary vertex point must use the maintained boundary rule");
        }
    }

    for (std::size_t i = 0; i < levelOne.positions.size(); ++i) {
        check(i < levelOne.sourceOriginalVertices.size(),
              "every subdivided vertex must have a source mapping record");
        check(!levelOne.sourceOriginalVertices[i].empty(),
              "every subdivided vertex must map to at least one original vertex");
        for (std::uint32_t source : levelOne.sourceOriginalVertices[i]) {
            check(source < base.positions.size(),
                  "source mapping must reference a valid original vertex");
        }
    }

    const subdiv::PolygonMesh levelTwo = subdiv::subdivideCatmullClarkOnce(levelOne);
    check(levelTwo.faces.size() == levelOne.faces.size() * 4,
          "each quad must split into four quads on later iterations");
    for (const auto& face : levelTwo.faces) {
        check(face.size() == 4, "every later-iteration output face must be a quad");
    }

    std::size_t boundaryEdgesLevelOne = 0;
    for (const auto& edge : levelOne.edges) {
        boundaryEdgesLevelOne += edge.isBoundary() ? 1U : 0U;
    }
    check(boundaryEdgesLevelOne > 0, "subdivision must preserve an open boundary");

    const subdiv::PolygonMesh repeat = subdiv::subdivideCatmullClarkOnce(base);
    check(subdiv::fingerprint(levelOne) == subdiv::fingerprint(repeat),
          "subdivision must be deterministic for identical input");
}

void testLevelCountsAndTrace() {
    const subdiv::TriangleMesh triangles = subdiv::buildDeterministicMesh(20251029ULL);
    const subdiv::PolygonMesh base = subdiv::triangleMeshToPolygonMesh(triangles);
    const std::size_t triangleCount = triangles.indices.size() / 3;

    for (int level = 0; level <= 4; ++level) {
        const subdiv::PolygonMesh mesh = subdiv::subdivideCatmullClark(base, level);
        std::size_t quadCount = 0;
        for (const auto& face : mesh.faces) {
            if (face.size() == 4) {
                ++quadCount;
            }
        }
        std::size_t expectedQuads = 0;
        if (level > 0) {
            expectedQuads = triangleCount * 3;
            for (int i = 1; i < level; ++i) {
                expectedQuads *= 4;
            }
        }
        check(quadCount == expectedQuads,
              "subdivision level must produce 3T*4^(L-1) quads");
        for (const auto& sources : mesh.sourceOriginalVertices) {
            check(!sources.empty(), "all levels must retain source mappings");
        }
    }
}

void testJsonAndSceneDefaults() {
    {
        const std::string text = R"({
          "seed": 42,
          "material": {"diffuse": [0.1, 0.2, 0.3], "roughness": 0.7},
          "creaseWeights": [0.0, 0.5, 1.0],
          "subdivision": {"maxLevel": 3, "initialLevel": 1},
          "camera": {
            "position": [1, 2, 3],
            "target": [0, 0, 0],
            "up": [0, 1, 0],
            "fovDegrees": 55
          }
        })";
        const std::string path = "subdiv_test_valid_scene.json";
        {
            std::ofstream file(path);
            file << text;
        }
        const subdiv::SceneLoadResult result = subdiv::loadSceneFile(path);
        std::remove(path.c_str());
        check(result.warnings.empty(), "complete valid scene should not emit warnings");
        check(result.settings.seed == 42, "scene seed must be read");
        check(near(result.settings.diffuse, {0.1f, 0.2f, 0.3f}),
              "scene diffuse color must be read");
        check(near(result.settings.roughness, 0.7f), "scene roughness must be read");
        check(result.settings.creaseOverrides.size() == 3,
              "per-edge crease array must be read");
        check(result.settings.maxSubdivision == 3,
              "subdivision max level must be read");
        check(result.settings.initialSubdivision == 1,
              "subdivision initial level must be read");
        check(near(result.settings.camera.fovDegrees, 55.0f),
              "camera fov must be read");
    }

    {
        const std::string text = R"({
          "seed": -5,
          "material": {"diffuse": [0.2, 2.0, 0.3], "roughness": 2.0},
          "subdivision": {"maxLevel": 99, "initialLevel": 9},
          "creaseWeights": [0.2, 5, -1, "bad"],
          "camera": {
            "position": [0, 1],
            "target": [0, 0, 0],
            "up": [0, 0, 0],
            "fovDegrees": 5
          }
        })";
        const std::string path = "subdiv_test_invalid_scene.json";
        {
            std::ofstream file(path);
            file << text;
        }
        const subdiv::SceneLoadResult result = subdiv::loadSceneFile(path);
        std::remove(path.c_str());
        check(!result.warnings.empty(), "invalid scene must emit warnings");
        std::ostringstream allWarnings;
        for (const std::string& warning : result.warnings) {
            allWarnings << warning << '\n';
        }
        const std::string joined = allWarnings.str();
        check(joined.find("'seed'") != std::string::npos,
              "warning must name seed field");
        check(joined.find("'material.diffuse[1]'") != std::string::npos,
              "warning must name out-of-range diffuse component");
        check(joined.find("'material.roughness'") != std::string::npos,
              "warning must name roughness field");
        check(joined.find("'subdivision.maxLevel'") != std::string::npos,
              "warning must name subdivision max field");
        check(joined.find("'subdivision.initialLevel'") != std::string::npos,
              "warning must name subdivision initial field");
        check(joined.find("'creaseWeights[1]'") != std::string::npos,
              "warning must name invalid crease element");
        check(joined.find("'camera.position'") != std::string::npos,
              "warning must name camera position field");
        check(joined.find("'camera.up'") != std::string::npos,
              "warning must name zero camera up field");
        check(joined.find("'camera.fovDegrees'") != std::string::npos,
              "warning must name camera fov field");
        check(result.settings.seed == 20251029ULL,
              "invalid seed must fall back to default");
        check(near(result.settings.diffuse, {0.20f, 0.52f, 0.30f}),
              "invalid diffuse component must fall back to its component default");
        check(result.settings.maxSubdivision == 4,
              "invalid max subdivision must fall back to default");
    }

    {
        const std::string path = "subdiv_test_malformed.json";
        {
            std::ofstream file(path);
            file << "{\"seed\": 10, \"material\": [this is not json";
        }
        const subdiv::SceneLoadResult result = subdiv::loadSceneFile(path);
        std::remove(path.c_str());
        check(!result.warnings.empty(), "malformed JSON must produce a warning");
        check(result.settings.seed == 20251029ULL,
              "malformed JSON must use all built-in defaults without crashing");
    }

    {
        const subdiv::SceneLoadResult result =
            subdiv::loadSceneFile("subdiv_test_missing_scene.json");
        check(!result.warnings.empty(), "missing scene file must produce a warning");
    }

    {
        subdiv::JsonValue value;
        std::string error;
        check(subdiv::parseJson(R"({"text":"line\n\u4e2d","n":-1.25e2})",
                                value, error),
              "JSON parser must accept strings, escapes and exponents");
        const subdiv::JsonValue* number = value.find("n");
        check(number != nullptr && number->isNumber() && near(
                  static_cast<float>(number->number), -125.0f),
              "JSON parser must decode numeric values");
    }
}

}  // namespace

int main() {
    testDeterministicConstruction();
    testCreaseOverride();
    testCatmullClarkRules();
    testLevelCountsAndTrace();
    testJsonAndSceneDefaults();

    if (failures == 0) {
        std::cout << "[PASS] all deterministic CPU tests passed\n";
        return 0;
    }
    std::cerr << "[FAIL] " << failures << " check(s) failed\n";
    return 1;
}

