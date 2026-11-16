#include "Scene.h"

#include <cfloat>

using namespace DirectX;

namespace {

constexpr float kTerrainHalfExtent = 60.0f;
constexpr uint32_t kTerrainGridSize = 256;
constexpr uint32_t kTerrainSeed = 0x51A7C3E9u;
constexpr uint32_t kCapsuleRadialSegments = 18;
constexpr uint32_t kCapsuleHemisphereRings = 5;

uint32_t Hash2D(int x, int z) {
    uint32_t h = kTerrainSeed;
    h ^= static_cast<uint32_t>(x) * 0x9E3779B1u;
    h ^= static_cast<uint32_t>(z) * 0x85EBCA77u;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

float LatticeValue(int x, int z) {
    const uint32_t bits = Hash2D(x, z) & 0x00FFFFFFu;
    return static_cast<float>(bits) * (2.0f / 16777215.0f) - 1.0f;
}

float SmoothWeight(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float ValueNoise(float x, float z) {
    const float fx = std::floor(x);
    const float fz = std::floor(z);
    const int ix = static_cast<int>(fx);
    const int iz = static_cast<int>(fz);
    const float tx = SmoothWeight(x - fx);
    const float tz = SmoothWeight(z - fz);

    const float a = LatticeValue(ix, iz);
    const float b = LatticeValue(ix + 1, iz);
    const float c = LatticeValue(ix, iz + 1);
    const float d = LatticeValue(ix + 1, iz + 1);
    const float ab = a + (b - a) * tx;
    const float cd = c + (d - c) * tx;
    return ab + (cd - ab) * tz;
}

float FractalNoise(float x, float z) {
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float value = 0.0f;
    float normalization = 0.0f;
    for (int octave = 0; octave < 5; ++octave) {
        value += ValueNoise(x * frequency, z * frequency) * amplitude;
        normalization += amplitude;
        amplitude *= 0.5f;
        frequency *= 2.03f;
    }
    return value / normalization;
}

float TerrainHeight(float x, float z) {
    const float broad = FractalNoise(x * 0.035f + 13.7f, z * 0.035f - 4.1f);
    const float detail = FractalNoise(x * 0.115f - 8.3f, z * 0.115f + 2.9f);
    return broad * 2.2f + detail * 0.32f - 0.15f;
}

void AddVertex(std::vector<Vertex>& vertices, const XMFLOAT3& position,
               const XMFLOAT3& normal, const XMFLOAT3& color,
               const XMFLOAT4& animation) {
    vertices.push_back({position, normal, color, animation});
}

void AddQuadIndices(std::vector<uint32_t>& indices, uint32_t a, uint32_t b,
                    uint32_t c, uint32_t d) {
    indices.push_back(a);
    indices.push_back(c);
    indices.push_back(b);
    indices.push_back(b);
    indices.push_back(c);
    indices.push_back(d);
}

void AppendCapsule(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                   const XMFLOAT3& a, const XMFLOAT3& b, float radius,
                   const XMFLOAT3& color, uint32_t actorIndex,
                   uint32_t capsuleIndex) {
    const XMVECTOR start = XMLoadFloat3(&a);
    const XMVECTOR end = XMLoadFloat3(&b);
    XMVECTOR axis = XMVectorSubtract(end, start);
    axis = XMVector3Normalize(axis);

    XMVECTOR reference = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
    if (std::abs(XMVectorGetY(axis)) > 0.92f) {
        reference = XMVectorSet(1.0f, 0.0f, 0.0f, 0.0f);
    }
    const XMVECTOR u = XMVector3Normalize(XMVector3Cross(reference, axis));
    const XMVECTOR v = XMVector3Normalize(XMVector3Cross(axis, u));

    const uint32_t firstVertex = static_cast<uint32_t>(vertices.size());
    const uint32_t ringsPerHemisphere = kCapsuleHemisphereRings + 1;
    const float pi = XM_PI;
    const float phase =
        static_cast<float>(actorIndex) * 1.73f + static_cast<float>(capsuleIndex) * 0.91f;
    const float amplitude = 1.0f;
    const float frequency = 1.65f + static_cast<float>(actorIndex) * 0.13f;

    auto appendCapsuleRing = [&](const XMVECTOR& center, float axial, float radial,
                                 float normalAxial, float normalRadial) {
        for (uint32_t segment = 0; segment < kCapsuleRadialSegments; ++segment) {
            const float angle =
                (2.0f * pi * static_cast<float>(segment)) /
                static_cast<float>(kCapsuleRadialSegments);
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const XMVECTOR radialDirection =
                XMVectorAdd(XMVectorScale(u, cosine), XMVectorScale(v, sine));
            const XMVECTOR position = XMVectorAdd(
                center, XMVectorAdd(XMVectorScale(axis, axial),
                                    XMVectorScale(radialDirection, radial)));
            XMVECTOR normal = XMVectorAdd(XMVectorScale(axis, normalAxial),
                                         XMVectorScale(radialDirection, normalRadial));
            normal = XMVector3Normalize(normal);

            XMFLOAT3 positionFloat{};
            XMFLOAT3 normalFloat{};
            XMStoreFloat3(&positionFloat, position);
            XMStoreFloat3(&normalFloat, normal);
            const float vertexPhase = phase + positionFloat.y * 0.37f +
                                      positionFloat.x * 0.11f + positionFloat.z * 0.07f;
            AddVertex(vertices, positionFloat, normalFloat, color,
                      {amplitude, frequency, vertexPhase,
                       static_cast<float>(actorIndex + 1)});
        }
    };

    // Lower hemisphere: pole to equator at the start point.
    for (uint32_t ring = 0; ring < ringsPerHemisphere; ++ring) {
        const float phi =
            -0.5f * pi + (0.5f * pi) * static_cast<float>(ring) /
                             static_cast<float>(kCapsuleHemisphereRings);
        appendCapsuleRing(start, radius * std::sin(phi), radius * std::cos(phi),
                          std::sin(phi), std::cos(phi));
    }

    // Upper hemisphere: equator at the end point to the opposite pole.
    for (uint32_t ring = 0; ring < ringsPerHemisphere; ++ring) {
        const float phi =
            (0.5f * pi) * static_cast<float>(ring) /
            static_cast<float>(kCapsuleHemisphereRings);
        appendCapsuleRing(end, radius * std::sin(phi), radius * std::cos(phi),
                          std::sin(phi), std::cos(phi));
    }

    const uint32_t totalRings = ringsPerHemisphere * 2;
    for (uint32_t ring = 0; ring + 1 < totalRings; ++ring) {
        for (uint32_t segment = 0; segment < kCapsuleRadialSegments; ++segment) {
            const uint32_t next = (segment + 1) % kCapsuleRadialSegments;
            const uint32_t ring0 = firstVertex + ring * kCapsuleRadialSegments;
            const uint32_t ring1 = firstVertex + (ring + 1) * kCapsuleRadialSegments;
            AddQuadIndices(indices, ring0 + segment, ring0 + next, ring1 + segment,
                           ring1 + next);
        }
    }
}

XMFLOAT3 Offset(const XMFLOAT3& point, const XMFLOAT3& origin) {
    return {point.x + origin.x, point.y + origin.y, point.z + origin.z};
}

void AppendCharacter(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                     const XMFLOAT3& origin, uint32_t actorIndex,
                     const XMFLOAT3& bodyColor) {
    const XMFLOAT3 dark = {bodyColor.x * 0.55f, bodyColor.y * 0.55f,
                           bodyColor.z * 0.55f};
    const XMFLOAT3 light = {std::min(1.0f, bodyColor.x * 1.18f + 0.08f),
                            std::min(1.0f, bodyColor.y * 1.18f + 0.08f),
                            std::min(1.0f, bodyColor.z * 1.18f + 0.08f)};

    struct CapsuleDefinition {
        XMFLOAT3 a;
        XMFLOAT3 b;
        float radius;
        XMFLOAT3 color;
    };

    const std::array<CapsuleDefinition, 11> capsules = {{
        {{0.0f, 3.0f, 0.0f}, {0.0f, 4.55f, 0.0f}, 0.92f, bodyColor},
        {{0.0f, 5.12f, 0.0f}, {0.0f, 5.82f, 0.0f}, 0.63f, light},
        {{-0.92f, 4.35f, 0.0f}, {-1.55f, 3.55f, 0.06f}, 0.31f, bodyColor},
        {{-1.55f, 3.55f, 0.06f}, {-1.86f, 2.62f, 0.24f}, 0.23f, dark},
        {{0.92f, 4.35f, 0.0f}, {1.55f, 3.55f, 0.06f}, 0.31f, bodyColor},
        {{1.55f, 3.55f, 0.06f}, {1.86f, 2.62f, 0.24f}, 0.23f, dark},
        {{-0.47f, 2.70f, 0.0f}, {-0.62f, 1.45f, 0.02f}, 0.36f, dark},
        {{-0.62f, 1.45f, 0.02f}, {-0.73f, 0.25f, 0.10f}, 0.24f, bodyColor},
        {{0.47f, 2.70f, 0.0f}, {0.62f, 1.45f, 0.02f}, 0.36f, dark},
        {{0.62f, 1.45f, 0.02f}, {0.73f, 0.25f, 0.10f}, 0.24f, bodyColor},
        {{-0.25f, 4.30f, 0.67f}, {0.25f, 4.30f, 0.67f}, 0.16f, light},
    }};

    for (uint32_t i = 0; i < capsules.size(); ++i) {
        const CapsuleDefinition& capsule = capsules[i];
        AppendCapsule(vertices, indices, Offset(capsule.a, origin),
                      Offset(capsule.b, origin), capsule.radius, capsule.color,
                      actorIndex, i);
    }
}

}  // namespace

SceneData GenerateDeterministicScene() {
    SceneData scene;

    const uint32_t terrainVertexCount = kTerrainGridSize * kTerrainGridSize;
    scene.vertices.reserve(terrainVertexCount + 50000);
    scene.indices.reserve((kTerrainGridSize - 1) * (kTerrainGridSize - 1) * 6 + 60000);

    for (uint32_t z = 0; z < kTerrainGridSize; ++z) {
        for (uint32_t x = 0; x < kTerrainGridSize; ++x) {
            const float u = static_cast<float>(x) /
                            static_cast<float>(kTerrainGridSize - 1);
            const float v = static_cast<float>(z) /
                            static_cast<float>(kTerrainGridSize - 1);
            const float worldX = -kTerrainHalfExtent + 2.0f * kTerrainHalfExtent * u;
            const float worldZ = -kTerrainHalfExtent + 2.0f * kTerrainHalfExtent * v;
            const float height = TerrainHeight(worldX, worldZ);
            const float epsilon = 0.25f;
            const float dx =
                (TerrainHeight(worldX + epsilon, worldZ) -
                 TerrainHeight(worldX - epsilon, worldZ)) /
                (2.0f * epsilon);
            const float dz =
                (TerrainHeight(worldX, worldZ + epsilon) -
                 TerrainHeight(worldX, worldZ - epsilon)) /
                (2.0f * epsilon);
            XMFLOAT3 normal{};
            XMStoreFloat3(&normal,
                          XMVector3Normalize(XMVectorSet(-dx, 1.0f, -dz, 0.0f)));

            const float elevation =
                std::clamp((height + 2.3f) / 4.8f, 0.0f, 1.0f);
            const float variation =
                FractalNoise(worldX * 0.19f + 31.0f, worldZ * 0.19f - 17.0f) *
                0.5f + 0.5f;
            XMFLOAT3 color = {
                0.15f + elevation * 0.24f + variation * 0.035f,
                0.28f + elevation * 0.15f + variation * 0.03f,
                0.12f + elevation * 0.12f + variation * 0.025f,
            };
            AddVertex(scene.vertices, {worldX, height, worldZ}, normal, color,
                      {0.0f, 0.0f, 0.0f, 0.0f});
        }
    }

    for (uint32_t z = 0; z + 1 < kTerrainGridSize; ++z) {
        for (uint32_t x = 0; x + 1 < kTerrainGridSize; ++x) {
            const uint32_t v00 = z * kTerrainGridSize + x;
            const uint32_t v10 = v00 + 1;
            const uint32_t v01 = v00 + kTerrainGridSize;
            const uint32_t v11 = v01 + 1;
            scene.indices.push_back(v00);
            scene.indices.push_back(v01);
            scene.indices.push_back(v10);
            scene.indices.push_back(v10);
            scene.indices.push_back(v01);
            scene.indices.push_back(v11);
        }
    }

    const XMFLOAT3 actorA = {-5.7f, TerrainHeight(-5.7f, 0.2f) + 0.22f, 0.2f};
    const XMFLOAT3 actorB = {6.1f, TerrainHeight(6.1f, 2.4f) + 0.22f, 2.4f};
    AppendCharacter(scene.vertices, scene.indices, actorA, 0, {0.82f, 0.28f, 0.20f});
    AppendCharacter(scene.vertices, scene.indices, actorB, 1, {0.16f, 0.52f, 0.66f});

    XMVECTOR light = XMVectorSet(-0.48f, -0.78f, -0.40f, 0.0f);
    XMStoreFloat3(&scene.lightDirection, XMVector3Normalize(light));
    return scene;
}

