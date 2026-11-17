#include "assets.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <utility>

namespace cv {
namespace {

constexpr float kTwoPi = 2.0f * kPi;

Vec3 safeNormalize(Vec3 v, Vec3 fallback = {0.0f, 1.0f, 0.0f}) {
    const float l2 = lengthSquared(v);
    return l2 > 1.0e-12f ? v / std::sqrt(l2) : fallback;
}

uint16_t materialId(const Scene& scene, const char* name) {
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        if (scene.materials[i].name == name) {
            return static_cast<uint16_t>(i);
        }
    }
    return 0;
}

void setWeights(Vertex& v, int bone0, float w0, int bone1 = -1, float w1 = 0.0f,
                int bone2 = -1, float w2 = 0.0f, int bone3 = -1, float w3 = 0.0f) {
    const std::array<int, 4> bones{bone0, bone1, bone2, bone3};
    const std::array<float, 4> weights{w0, w1, w2, w3};
    float sum = 0.0f;
    for (float w : weights) {
        sum += std::max(0.0f, w);
    }
    int out = 0;
    for (int i = 0; i < 4; ++i) {
        if (bones[i] >= 0 && weights[i] > 0.0f) {
            v.boneIndices[static_cast<size_t>(out)] = static_cast<uint16_t>(bones[i]);
            v.boneWeights[static_cast<size_t>(out)] =
                sum > 0.0f ? weights[i] / sum : (out == 0 ? 1.0f : 0.0f);
            ++out;
        }
    }
    if (out == 0) {
        v.boneIndices[0] = 0;
        v.boneWeights[0] = 1.0f;
    }
}

class MeshBuilder {
public:
    explicit MeshBuilder(Mesh& mesh) : mesh_(mesh) {}

    uint32_t addVertex(Vec3 p, Vec3 n, Vec3 t, float tangentW, Vec2 uv, int b0, float w0,
                       int b1 = -1, float w1 = 0.0f, int b2 = -1, float w2 = 0.0f,
                       int b3 = -1, float w3 = 0.0f) {
        Vertex v;
        v.position = p;
        v.normal = safeNormalize(n);
        v.tangent = {t.x, t.y, t.z, tangentW};
        v.uv = uv;
        setWeights(v, b0, w0, b1, w1, b2, w2, b3, w3);
        mesh_.vertices.push_back(v);
        return static_cast<uint32_t>(mesh_.vertices.size() - 1);
    }

    void addTriangle(uint32_t a, uint32_t b, uint32_t c, uint16_t material) {
        mesh_.indices.push_back(a);
        mesh_.indices.push_back(b);
        mesh_.indices.push_back(c);
        mesh_.triangleMaterial.push_back(material);
    }

    void addQuad(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint16_t material) {
        addTriangle(a, b, c, material);
        addTriangle(b, d, c, material);
    }

private:
    Mesh& mesh_;
};

void addTube(Mesh& mesh, uint16_t material, Vec3 p0, Vec3 p1, float radius0, float radius1,
             int segments, int rings, int bone0, int bone1, bool mirrorU) {
    const Vec3 axis = safeNormalize(p1 - p0);
    const Vec3 reference = std::abs(axis.y) < 0.9f ? Vec3{0.0f, 1.0f, 0.0f}
                                                   : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 u = safeNormalize(cross(reference, axis));
    const Vec3 v = normalize(cross(axis, u));
    MeshBuilder builder(mesh);

    std::vector<uint32_t> rowStart;
    rowStart.reserve(static_cast<size_t>(rings + 1));
    for (int i = 0; i <= rings; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(rings);
        const Vec3 center = lerp(p0, p1, t);
        const float radius = lerp(radius0, radius1, t);
        rowStart.push_back(static_cast<uint32_t>(mesh.vertices.size()));
        for (int j = 0; j <= segments; ++j) {
            const float s = static_cast<float>(j) / static_cast<float>(segments);
            const float angle = s * kTwoPi;
            const Vec3 radial = u * std::cos(angle) + v * std::sin(angle);
            const Vec3 position = center + radial * radius;
            const Vec3 tangentDir = (-u * std::sin(angle) + v * std::cos(angle)) *
                                    (mirrorU ? -1.0f : 1.0f);
            // The right arm uses mirrored U coordinates. Its tangent is negated and
            // tangent.w=-1 keeps bitangent = cross(N,T)*w pointing along +V.
            const float uvU = mirrorU ? 1.0f - s : s;
            const float boneMix = t * 0.82f;
            builder.addVertex(position, radial, tangentDir, mirrorU ? -1.0f : 1.0f,
                              {uvU, t}, bone0, 1.0f - boneMix, bone1, boneMix);
        }
    }

    for (int i = 0; i < rings; ++i) {
        for (int j = 0; j < segments; ++j) {
            const uint32_t a = rowStart[static_cast<size_t>(i)] +
                               static_cast<uint32_t>(j);
            const uint32_t b = a + 1;
            const uint32_t c = rowStart[static_cast<size_t>(i + 1)] +
                               static_cast<uint32_t>(j);
            const uint32_t d = c + 1;
            builder.addTriangle(a, b, c, material);
            builder.addTriangle(b, d, c, material);
        }
    }
}

void addSphere(Mesh& mesh, uint16_t material, Vec3 center, Vec3 radii, int lonSegments,
               int latSegments, int bone0, int bone1 = -1, float boneBlend = 0.0f,
               bool mirrorU = false) {
    MeshBuilder builder(mesh);
    std::vector<uint32_t> rows;
    rows.reserve(static_cast<size_t>(latSegments + 1));
    for (int i = 0; i <= latSegments; ++i) {
        const float tv = static_cast<float>(i) / static_cast<float>(latSegments);
        const float theta = tv * kPi;
        const float st = std::sin(theta);
        const float ct = std::cos(theta);
        rows.push_back(static_cast<uint32_t>(mesh.vertices.size()));
        for (int j = 0; j <= lonSegments; ++j) {
            const float s = static_cast<float>(j) / static_cast<float>(lonSegments);
            const float phi = s * kTwoPi;
            const Vec3 unit{st * std::cos(phi), ct, st * std::sin(phi)};
            const Vec3 position = center + unit * radii;
            const Vec3 normal = safeNormalize(
                Vec3{unit.x / std::max(radii.x * radii.x, 1.0e-6f),
                     unit.y / std::max(radii.y * radii.y, 1.0e-6f),
                     unit.z / std::max(radii.z * radii.z, 1.0e-6f)});
            Vec3 tangent = {-st * std::sin(phi), 0.0f, st * std::cos(phi)};
            tangent = safeNormalize(tangent, {1.0f, 0.0f, 0.0f});
            if (mirrorU) {
                tangent = -tangent;
            }
            const float uvU = mirrorU ? 1.0f - s : s;
            builder.addVertex(position, normal, tangent, mirrorU ? -1.0f : 1.0f,
                              {uvU, 1.0f - tv}, bone0, 1.0f - boneBlend, bone1, boneBlend);
        }
    }
    for (int i = 0; i < latSegments; ++i) {
        for (int j = 0; j < lonSegments; ++j) {
            const uint32_t a = rows[static_cast<size_t>(i)] + static_cast<uint32_t>(j);
            const uint32_t b = a + 1;
            const uint32_t c = rows[static_cast<size_t>(i + 1)] + static_cast<uint32_t>(j);
            const uint32_t d = c + 1;
            // d(phi) x d(theta) points outward for this parametrization.
            builder.addTriangle(a, c, b, material);
            builder.addTriangle(b, c, d, material);
        }
    }
}

void addTorso(Mesh& mesh, uint16_t suitMaterial, uint16_t armorMaterial) {
    struct Ring {
        float y;
        float rx;
        float rz;
    };
    const std::array<Ring, 10> rings{{
        {0.96f, 0.205f, 0.145f}, {1.05f, 0.245f, 0.165f}, {1.17f, 0.265f, 0.175f},
        {1.30f, 0.285f, 0.185f}, {1.42f, 0.315f, 0.195f}, {1.53f, 0.350f, 0.205f},
        {1.61f, 0.370f, 0.210f}, {1.68f, 0.360f, 0.200f}, {1.74f, 0.310f, 0.170f},
        {1.78f, 0.205f, 0.125f},
    }};
    constexpr int kSegments = 32;
    MeshBuilder builder(mesh);
    std::vector<uint32_t> rowStart;
    for (const Ring& ring : rings) {
        rowStart.push_back(static_cast<uint32_t>(mesh.vertices.size()));
        for (int j = 0; j <= kSegments; ++j) {
            const float s = static_cast<float>(j) / static_cast<float>(kSegments);
            const float angle = s * kTwoPi;
            const Vec3 radialUnit{std::cos(angle), 0.0f, std::sin(angle)};
            const Vec3 position{radialUnit.x * ring.rx, ring.y, radialUnit.z * ring.rz};
            const Vec3 normal = safeNormalize(
                {radialUnit.x / (ring.rx * ring.rx), 0.0f,
                 radialUnit.z / (ring.rz * ring.rz)});
            const Vec3 tangent =
                safeNormalize({-radialUnit.z * ring.rx, 0.0f, radialUnit.x * ring.rz});

            float spineW = 0.0f;
            float chestW = 0.0f;
            float rootW = 0.0f;
            if (ring.y < 1.18f) {
                const float t = clamp01((ring.y - 0.96f) / 0.22f);
                rootW = 1.0f - t;
                spineW = t;
            } else if (ring.y < 1.48f) {
                const float t = clamp01((ring.y - 1.18f) / 0.30f);
                spineW = 1.0f - t;
                chestW = t;
            } else {
                chestW = 1.0f;
            }
            builder.addVertex(position, normal, tangent, 1.0f, {s, ring.y * 4.0f}, 0,
                              rootW, 1, spineW, 2, chestW);
        }
    }

    for (size_t i = 0; i + 1 < rings.size(); ++i) {
        const uint16_t material = rings[i].y >= 1.30f ? armorMaterial : suitMaterial;
        for (int j = 0; j < kSegments; ++j) {
            const uint32_t a = rowStart[i] + static_cast<uint32_t>(j);
            const uint32_t b = a + 1;
            const uint32_t c = rowStart[i + 1] + static_cast<uint32_t>(j);
            const uint32_t d = c + 1;
            builder.addTriangle(a, b, c, material);
            builder.addTriangle(b, d, c, material);
        }
    }

    // Bottom cap.
    const uint32_t bottomCenter = builder.addVertex(
        {0.0f, 0.96f, 0.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 1.0f, {0.5f, 0.5f},
        0, 1.0f);
    for (int j = 0; j < kSegments; ++j) {
        builder.addTriangle(rowStart[0] + static_cast<uint32_t>(j),
                            rowStart[0] + static_cast<uint32_t>(j) + 1, bottomCenter,
                            suitMaterial);
    }
    // Neck hole cap.
    const uint32_t topCenter = builder.addVertex(
        {0.0f, 1.78f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 1.0f, {0.5f, 0.5f},
        2, 1.0f);
    const size_t lastRow = rings.size() - 1;
    for (int j = 0; j < kSegments; ++j) {
        builder.addTriangle(rowStart[lastRow] + static_cast<uint32_t>(j),
                            topCenter,
                            rowStart[lastRow] + static_cast<uint32_t>(j) + 1,
                            suitMaterial);
    }
}

void addChestStrip(Mesh& mesh, uint16_t material, int bone) {
    MeshBuilder builder(mesh);
    const uint32_t baseVertex = static_cast<uint32_t>(mesh.vertices.size());
    constexpr int kSteps = 10;
    constexpr float kY0 = 1.34f;
    constexpr float kY1 = 1.67f;
    const auto zAt = [](float y) {
        // Just proud of the torso's front surface, following its curvature.
        const float t = clamp01((y - 1.30f) / 0.32f);
        return lerp(0.185f, 0.212f, t) + 0.006f;
    };
    const auto widthAt = [](float y) {
        const float t = clamp01((y - 1.34f) / 0.33f);
        return lerp(0.055f, 0.038f, t);
    };
    for (int i = 0; i <= kSteps; ++i) {
        const float t = static_cast<float>(i) / kSteps;
        const float y = lerp(kY0, kY1, t);
        const float z = zAt(y);
        const float w = widthAt(y);
        const Vec3 n{0.0f, 0.0f, 1.0f};
        const Vec3 tan{1.0f, 0.0f, 0.0f};
        builder.addVertex({-w, y, z}, n, tan, 1.0f, {0.0f, t}, bone, 1.0f);
        builder.addVertex({w, y, z}, n, tan, 1.0f, {1.0f, t}, bone, 1.0f);
    }
    for (int i = 0; i < kSteps; ++i) {
        const uint32_t base = baseVertex + static_cast<uint32_t>(i * 2);
        builder.addTriangle(base, base + 1, base + 2, material);
        builder.addTriangle(base + 1, base + 3, base + 2, material);
    }
}

void addCape(Mesh& mesh, uint16_t material) {
    constexpr int kCols = 14;
    constexpr int kRows = 22;
    MeshBuilder builder(mesh);
    const uint32_t baseVertex = static_cast<uint32_t>(mesh.vertices.size());
    for (int iy = 0; iy <= kRows; ++iy) {
        const float v = static_cast<float>(iy) / kRows;
        const float y = lerp(1.69f, 0.42f, v);
        const float halfWidth = lerp(0.42f, 0.64f, v);
        for (int ix = 0; ix <= kCols; ++ix) {
            const float u = static_cast<float>(ix) / kCols;
            const float x = lerp(-halfWidth, halfWidth, u);
            const float wave = std::sin(v * 7.0f + u * 5.0f) * 0.055f * v;
            const float z = -0.205f - 0.06f * v + wave;
            const Vec3 dx{1.0f, 0.0f, 0.0f};
            const Vec3 n{0.0f, 0.0f, -1.0f};
            const float chestW = (1.0f - v) * 0.7f;
            const float spineW = (1.0f - v) * 0.3f;
            const float rootW = v;
            builder.addVertex({x, y, z}, n, dx, 1.0f, {u, 1.0f - v}, 2, chestW, 1,
                              spineW, 0, rootW, 3, 0.0f);
        }
    }
    const int stride = kCols + 1;
    for (int iy = 0; iy < kRows; ++iy) {
        for (int ix = 0; ix < kCols; ++ix) {
            const uint32_t a =
                baseVertex + static_cast<uint32_t>(iy * stride + ix);
            const uint32_t b = a + 1;
            const uint32_t c = a + static_cast<uint32_t>(stride);
            const uint32_t d = c + 1;
            builder.addQuad(a, b, c, d, material);
        }
    }
}

void addVisor(Mesh& mesh, uint16_t material, int headBone) {
    constexpr int kSteps = 18;
    constexpr float kY0 = 2.075f;
    constexpr float kY1 = 2.235f;
    MeshBuilder builder(mesh);
    const uint32_t baseVertex = static_cast<uint32_t>(mesh.vertices.size());
    for (int iy = 0; iy < 2; ++iy) {
        const float y = iy == 0 ? kY0 : kY1;
        for (int i = 0; i <= kSteps; ++i) {
            const float t = static_cast<float>(i) / kSteps;
            const float phi = lerp(-1.02f, 1.02f, t);
            const Vec3 radial{std::sin(phi), 0.0f, std::cos(phi)};
            const Vec3 p{radial.x * 0.225f, y, 0.045f + radial.z * 0.205f};
            const Vec3 n = safeNormalize({radial.x / 0.225f, 0.0f, radial.z / 0.205f});
            const Vec3 tangent{std::cos(phi), 0.0f, -std::sin(phi)};
            builder.addVertex(p, n, tangent, 1.0f, {t, 1.0f - static_cast<float>(iy)},
                              headBone, 1.0f);
        }
    }
    for (int i = 0; i < kSteps; ++i) {
        const uint32_t a = baseVertex + static_cast<uint32_t>(i);
        const uint32_t b = a + 1;
        const uint32_t c = a + kSteps + 1;
        const uint32_t d = c + 1;
        builder.addQuad(a, b, c, d, material);
    }
}

void addGround(Mesh& mesh, uint16_t material) {
    MeshBuilder builder(mesh);
    constexpr float kExtent = 7.0f;
    constexpr int kTiles = 20;
    const Vec3 n{0.0f, 1.0f, 0.0f};
    const Vec3 t{1.0f, 0.0f, 0.0f};
    const float step = 2.0f * kExtent / static_cast<float>(kTiles);
    for (int zIndex = 0; zIndex < kTiles; ++zIndex) {
        const float z0 = -kExtent + static_cast<float>(zIndex) * step;
        const float z1 = z0 + step;
        for (int xIndex = 0; xIndex < kTiles; ++xIndex) {
            const float x0 = -kExtent + static_cast<float>(xIndex) * step;
            const float x1 = x0 + step;
            const float u0 = (x0 + kExtent) / (2.0f * kExtent) * 4.0f;
            const float u1 = (x1 + kExtent) / (2.0f * kExtent) * 4.0f;
            const float v0 = (z0 + kExtent) / (2.0f * kExtent) * 4.0f;
            const float v1 = (z1 + kExtent) / (2.0f * kExtent) * 4.0f;
            const uint32_t a =
                builder.addVertex({x0, 0.0f, z0}, n, t, 1.0f, {u0, v0}, 0, 0.0f);
            const uint32_t b =
                builder.addVertex({x0, 0.0f, z1}, n, t, 1.0f, {u0, v1}, 0, 0.0f);
            const uint32_t c =
                builder.addVertex({x1, 0.0f, z1}, n, t, 1.0f, {u1, v1}, 0, 0.0f);
            const uint32_t d =
                builder.addVertex({x1, 0.0f, z0}, n, t, 1.0f, {u1, v0}, 0, 0.0f);
            builder.addTriangle(a, b, c, material);
            builder.addTriangle(a, c, d, material);
        }
    }
}

void addEnvironment(Mesh& mesh, uint16_t postMaterial, uint16_t glowMaterial) {
    constexpr int kPosts = 12;
    for (int i = 0; i < kPosts; ++i) {
        const float angle = static_cast<float>(i) * kTwoPi / static_cast<float>(kPosts);
        const float x = std::cos(angle) * 6.2f;
        const float z = std::sin(angle) * 6.2f;
        const float height = 1.4f + static_cast<float>(i % 3) * 0.42f;
        addTube(mesh, postMaterial, {x, 0.0f, z}, {x, height, z}, 0.115f, 0.075f, 8, 3,
                0, 0, false);
        addTube(mesh, glowMaterial, {x, height - 0.05f, z}, {x, height + 0.04f, z},
                0.105f, 0.105f, 8, 1, 0, 0, false);
    }
}

void addLightOrbs(Mesh& mesh, uint16_t material, const Lighting& lighting) {
    for (const Light& light : lighting.pointLights) {
        addSphere(mesh, material, light.position, {light.radius, light.radius, light.radius},
                  12, 8, 0);
    }
}

void makeSkeleton(Skeleton& skeleton) {
    skeleton.bones = {
        {"root", -1, {0.0f, 1.10f, 0.0f}, {}},
        {"spine", 0, {0.0f, 0.34f, 0.0f}, {}},
        {"chest", 1, {0.0f, 0.30f, 0.0f}, {}},
        {"head", 2, {0.0f, 0.30f, 0.0f}, {}},
        {"upperArmL", 2, {-0.39f, 0.18f, 0.0f}, {}},
        {"foreArmL", 4, {-0.33f, -0.17f, 0.0f}, {}},
        {"handL", 5, {-0.27f, -0.13f, 0.0f}, {}},
        {"upperArmR", 2, {0.39f, 0.18f, 0.0f}, {}},
        {"foreArmR", 7, {0.33f, -0.17f, 0.0f}, {}},
        {"handR", 8, {0.27f, -0.13f, 0.0f}, {}},
        {"thighL", 0, {-0.16f, -0.15f, 0.0f}, {}},
        {"shinL", 10, {0.0f, -0.44f, 0.0f}, {}},
        {"footL", 11, {0.0f, -0.39f, 0.075f}, {}},
        {"thighR", 0, {0.16f, -0.15f, 0.0f}, {}},
        {"shinR", 13, {0.0f, -0.44f, 0.0f}, {}},
        {"footR", 14, {0.0f, -0.39f, 0.075f}, {}},
    };

    skeleton.bindGlobal.resize(skeleton.bones.size());
    skeleton.inverseBind.resize(skeleton.bones.size());
    for (size_t i = 0; i < skeleton.bones.size(); ++i) {
        const Bone& bone = skeleton.bones[i];
        const Mat4 local = Mat4::translation(bone.localPosition) *
                           Mat4::rotationZ(bone.bindEuler.z) *
                           Mat4::rotationY(bone.bindEuler.y) *
                           Mat4::rotationX(bone.bindEuler.x);
        skeleton.bindGlobal[i] =
            bone.parent >= 0 ? skeleton.bindGlobal[static_cast<size_t>(bone.parent)] * local
                             : local;
        skeleton.inverseBind[i] = skeleton.bindGlobal[i].inverse();
    }
}

void makeNormalMap(NormalTexture& texture) {
    constexpr int kSize = 256;
    texture.width = kSize;
    texture.height = kSize;
    texture.rgb.resize(static_cast<size_t>(kSize * kSize * 3));

    auto height = [](float u, float v) {
        const float p = std::sin(kTwoPi * (u * 8.0f + v * 2.0f)) * 0.16f;
        const float q = std::sin(kTwoPi * (v * 10.0f - u * 3.0f)) * 0.11f;
        const float fine = std::sin(kTwoPi * (u * 24.0f + v * 19.0f)) * 0.025f;
        const float seamU = std::abs(std::sin(kPi * u * 8.0f));
        const float seamV = std::abs(std::sin(kPi * v * 8.0f));
        const float groove = 0.10f * (1.0f - std::min(seamU, seamV));
        return p + q + fine - groove;
    };
    const float texel = 1.0f / static_cast<float>(kSize);
    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / kSize;
            const float v = (static_cast<float>(y) + 0.5f) / kSize;
            const float hL = height(u - texel, v);
            const float hR = height(u + texel, v);
            const float hD = height(u, v - texel);
            const float hU = height(u, v + texel);
            const Vec3 n = safeNormalize({-(hR - hL) * 3.7f, -(hU - hD) * 3.7f, 1.0f});
            const size_t offset = static_cast<size_t>((y * kSize + x) * 3);
            texture.rgb[offset + 0] =
                static_cast<uint8_t>(std::lround(clamp01(n.x * 0.5f + 0.5f) * 255.0f));
            texture.rgb[offset + 1] =
                static_cast<uint8_t>(std::lround(clamp01(n.y * 0.5f + 0.5f) * 255.0f));
            texture.rgb[offset + 2] =
                static_cast<uint8_t>(std::lround(clamp01(n.z * 0.5f + 0.5f) * 255.0f));
        }
    }
}

Material makeMaterial(const char* name, Vec3 base, float rough, float metallic, Vec3 emissive,
                      float opacity, float normalStrength, bool transparent = false,
                      bool doubleSided = false, bool castsShadow = true) {
    Material m;
    m.name = name;
    m.baseColor = base;
    m.roughness = rough;
    m.metallic = metallic;
    m.emissive = emissive;
    m.opacity = opacity;
    m.normalStrength = normalStrength;
    m.transparent = transparent;
    m.doubleSided = doubleSided;
    m.castsShadow = castsShadow;
    m.stateKey = (doubleSided ? 1ull : 0ull) | (transparent ? 2ull : 0ull);
    return m;
}

}  // namespace

size_t Scene::totalTriangles() const {
    size_t total = 0;
    for (const Mesh& mesh : meshes) {
        total += mesh.indices.size() / 3;
    }
    return total;
}

size_t Scene::totalVertices() const {
    size_t total = 0;
    for (const Mesh& mesh : meshes) {
        total += mesh.vertices.size();
    }
    return total;
}

std::vector<Mat4> buildBoneMatrices(const Skeleton& skeleton, uint32_t tick) {
    // Exact fixed-step animation. No wall clock enters this function.
    constexpr float kFixedDt = 1.0f / 30.0f;
    const float time = static_cast<float>(tick) * kFixedDt;
    const float phase = time * kTwoPi * 0.55f;
    const float stride = std::sin(phase);
    const float strideOpposite = std::sin(phase + kPi);
    const float armSwing = std::sin(phase + 0.4f);

    std::vector<Mat4> global(skeleton.bones.size());
    for (size_t i = 0; i < skeleton.bones.size(); ++i) {
        const Bone& bone = skeleton.bones[i];
        Vec3 offset{};
        float rx = 0.0f;
        float ry = 0.0f;
        float rz = 0.0f;
        if (bone.name == "root") {
            offset.y = std::abs(std::sin(phase * 2.0f)) * 0.016f;
            rz = std::sin(phase) * 0.018f;
        } else if (bone.name == "spine") {
            rx = -0.035f + std::sin(phase) * 0.018f;
            ry = std::sin(phase) * 0.05f;
            rz = std::sin(phase) * 0.018f;
        } else if (bone.name == "chest") {
            ry = std::sin(phase + 0.8f) * 0.08f;
            rz = std::sin(phase) * 0.012f;
        } else if (bone.name == "head") {
            ry = std::sin(phase * 0.5f + 1.0f) * 0.18f;
            rx = std::sin(phase * 0.37f) * 0.035f;
        } else if (bone.name == "upperArmL") {
            rx = -0.16f + armSwing * 0.28f;
            rz = 0.05f + std::sin(phase) * 0.025f;
        } else if (bone.name == "foreArmL") {
            rx = -0.20f - std::max(0.0f, armSwing) * 0.20f;
        } else if (bone.name == "upperArmR") {
            rx = 0.16f - armSwing * 0.28f;
            rz = -0.05f - std::sin(phase) * 0.025f;
        } else if (bone.name == "foreArmR") {
            rx = 0.20f + std::max(0.0f, -armSwing) * 0.20f;
        } else if (bone.name == "thighL") {
            rx = stride * 0.30f;
        } else if (bone.name == "shinL") {
            rx = std::max(0.0f, -stride) * 0.34f;
        } else if (bone.name == "thighR") {
            rx = strideOpposite * 0.30f;
        } else if (bone.name == "shinR") {
            rx = std::max(0.0f, -strideOpposite) * 0.34f;
        }

        const Mat4 local = Mat4::translation(bone.localPosition + offset) *
                           Mat4::rotationZ(bone.bindEuler.z + rz) *
                           Mat4::rotationY(bone.bindEuler.y + ry) *
                           Mat4::rotationX(bone.bindEuler.x + rx);
        global[i] = bone.parent >= 0 ? global[static_cast<size_t>(bone.parent)] * local : local;
    }

    std::vector<Mat4> skin(skeleton.bones.size());
    for (size_t i = 0; i < skeleton.bones.size(); ++i) {
        skin[i] = global[i] * skeleton.inverseBind[i];
    }
    return skin;
}

std::vector<FrameVertex> skinVertices(const Mesh& mesh,
                                      const std::vector<Mat4>& boneMatrices) {
    std::vector<FrameVertex> result;
    result.resize(mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        const Vertex& v = mesh.vertices[i];
        FrameVertex& out = result[i];
        out.uv = v.uv;
        out.worldTangent = v.tangent;
        if (!mesh.skinned) {
            out.worldPosition = v.position;
            out.worldNormal = v.normal;
            continue;
        }

        Vec3 position{};
        Vec3 normal{};
        Vec3 tangent{};
        float weightSum = 0.0f;
        for (int k = 0; k < kMaxBoneInfluences; ++k) {
            const float w = v.boneWeights[static_cast<size_t>(k)];
            if (w <= 0.0f) {
                continue;
            }
            const size_t bone = v.boneIndices[static_cast<size_t>(k)];
            if (bone >= boneMatrices.size()) {
                continue;
            }
            const Mat4& m = boneMatrices[bone];
            position += m.transformPoint(v.position) * w;
            normal += m.transformVector(v.normal) * w;
            tangent += m.transformVector({v.tangent.x, v.tangent.y, v.tangent.z}) * w;
            weightSum += w;
        }
        if (weightSum > 1.0e-6f) {
            position = position / weightSum;
            normal = normal / weightSum;
            tangent = tangent / weightSum;
        } else {
            position = v.position;
            normal = v.normal;
            tangent = {v.tangent.x, v.tangent.y, v.tangent.z};
        }
        out.worldPosition = position;
        out.worldNormal = safeNormalize(normal, v.normal);
        tangent = safeNormalize(tangent, {1.0f, 0.0f, 0.0f});
        out.worldTangent = {tangent.x, tangent.y, tangent.z, v.tangent.w};
    }
    return result;
}

AssetBundle makeProceduralScene() {
    AssetBundle bundle;
    Scene& scene = bundle.scene;

    scene.materials = {
        makeMaterial("Suit", {0.035f, 0.055f, 0.085f}, 0.62f, 0.10f, {}, 1.0f, 0.8f),
        makeMaterial("Armor", {0.09f, 0.38f, 0.58f}, 0.25f, 0.72f, {}, 1.0f, 0.72f),
        makeMaterial("Skin", {0.58f, 0.31f, 0.20f}, 0.66f, 0.02f, {}, 1.0f, 0.7f),
        makeMaterial("Accent", {0.95f, 0.35f, 0.045f}, 0.35f, 0.15f,
                     {2.0f, 0.45f, 0.04f}, 1.0f, 0.5f),
        makeMaterial("Boot", {0.018f, 0.025f, 0.038f}, 0.42f, 0.25f, {}, 1.0f, 0.8f),
        makeMaterial("Cape", {0.015f, 0.32f, 0.56f}, 0.30f, 0.0f, {0.02f, 0.16f, 0.25f},
                     0.32f, 1.0f, true, true, false),
        makeMaterial("Visor", {0.06f, 0.62f, 0.92f}, 0.08f, 0.0f, {0.12f, 0.55f, 0.9f},
                     0.56f, 1.0f, true, true, false),
        makeMaterial("Ground", {0.024f, 0.036f, 0.050f}, 0.82f, 0.10f, {}, 1.0f, 0.55f,
                     false, false, false),
        makeMaterial("Glow", {0.60f, 0.92f, 1.0f}, 0.12f, 0.0f, {4.0f, 5.0f, 7.0f}, 1.0f,
                     0.0f, false, true, false),
    };

    const uint16_t suit = materialId(scene, "Suit");
    const uint16_t armor = materialId(scene, "Armor");
    const uint16_t skin = materialId(scene, "Skin");
    const uint16_t accent = materialId(scene, "Accent");
    const uint16_t boot = materialId(scene, "Boot");
    const uint16_t cape = materialId(scene, "Cape");
    const uint16_t visor = materialId(scene, "Visor");
    const uint16_t ground = materialId(scene, "Ground");
    const uint16_t glow = materialId(scene, "Glow");

    makeSkeleton(scene.skeleton);
    makeNormalMap(scene.normalMap);

    Mesh character;
    character.name = "Aster-9 character";
    character.skinned = true;

    addTorso(character, suit, armor);
    addChestStrip(character, accent, 2);

    // Waist ring and neck.
    addTube(character, accent, {0.0f, 1.015f, 0.0f}, {0.0f, 1.085f, 0.0f}, 0.245f, 0.245f,
            28, 1, 0, 1, false);
    addTube(character, skin, {0.0f, 1.72f, 0.0f}, {0.0f, 1.92f, 0.0f}, 0.105f, 0.095f, 18,
            2, 2, 3, false);

    // Head, jaw, eyes, and mirrored visor.
    addSphere(character, skin, {0.0f, 2.115f, 0.0f}, {0.205f, 0.235f, 0.19f}, 28, 18, 3);
    addSphere(character, skin, {0.0f, 2.035f, 0.055f}, {0.145f, 0.105f, 0.135f}, 20, 12,
              3);
    addSphere(character, glow, {-0.068f, 2.135f, 0.174f}, {0.022f, 0.018f, 0.010f}, 12,
              8, 3);
    addSphere(character, glow, {0.068f, 2.135f, 0.174f}, {0.022f, 0.018f, 0.010f}, 12,
              8, 3);
    addVisor(character, visor, 3);

    // Arms: mirrored U on the right side exercises tangent.w handling.
    const Vec3 shoulderL{-0.39f, 1.93f, 0.0f};
    const Vec3 elbowL{-0.72f, 1.76f, 0.0f};
    const Vec3 wristL{-0.99f, 1.63f, 0.0f};
    const Vec3 shoulderR{0.39f, 1.93f, 0.0f};
    const Vec3 elbowR{0.72f, 1.76f, 0.0f};
    const Vec3 wristR{0.99f, 1.63f, 0.0f};
    addTube(character, suit, shoulderL, elbowL, 0.105f, 0.075f, 18, 5, 4, 5, false);
    addTube(character, suit, shoulderR, elbowR, 0.105f, 0.075f, 18, 5, 7, 8, true);
    addTube(character, armor, elbowL, wristL, 0.080f, 0.060f, 18, 5, 5, 6, false);
    addTube(character, armor, elbowR, wristR, 0.080f, 0.060f, 18, 5, 8, 9, true);
    addSphere(character, armor, shoulderL, {0.145f, 0.135f, 0.145f}, 18, 12, 4, 2, 0.18f);
    addSphere(character, armor, shoulderR, {0.145f, 0.135f, 0.145f}, 18, 12, 7, 2, 0.18f,
              true);
    addSphere(character, boot, wristL, {0.075f, 0.07f, 0.085f}, 14, 10, 6);
    addSphere(character, boot, wristR, {0.075f, 0.07f, 0.085f}, 14, 10, 9, -1, 0.0f, true);

    // Legs and boots.
    const Vec3 hipL{-0.16f, 0.95f, 0.0f};
    const Vec3 kneeL{-0.16f, 0.51f, 0.0f};
    const Vec3 ankleL{-0.16f, 0.115f, 0.0f};
    const Vec3 hipR{0.16f, 0.95f, 0.0f};
    const Vec3 kneeR{0.16f, 0.51f, 0.0f};
    const Vec3 ankleR{0.16f, 0.115f, 0.0f};
    addTube(character, suit, hipL, kneeL, 0.145f, 0.105f, 20, 6, 10, 11, false);
    addTube(character, suit, hipR, kneeR, 0.145f, 0.105f, 20, 6, 13, 14, true);
    addTube(character, boot, kneeL, ankleL, 0.11f, 0.085f, 18, 5, 11, 12, false);
    addTube(character, boot, kneeR, ankleR, 0.11f, 0.085f, 18, 5, 14, 15, true);
    addSphere(character, armor, kneeL, {0.115f, 0.09f, 0.12f}, 16, 10, 11);
    addSphere(character, armor, kneeR, {0.115f, 0.09f, 0.12f}, 16, 10, 14, -1, 0.0f, true);
    addSphere(character, boot, {-0.16f, 0.075f, 0.075f}, {0.115f, 0.075f, 0.205f}, 18, 10,
              12);
    addSphere(character, boot, {0.16f, 0.075f, 0.075f}, {0.115f, 0.075f, 0.205f}, 18, 10,
              15, -1, 0.0f, true);

    addCape(character, cape);
    scene.meshes.push_back(std::move(character));

    Mesh floorMesh;
    floorMesh.name = "Hangar floor";
    floorMesh.skinned = false;
    addGround(floorMesh, ground);
    scene.meshes.push_back(std::move(floorMesh));

    Mesh environment;
    environment.name = "Hangar light pylons";
    environment.skinned = false;
    addEnvironment(environment, suit, glow);
    scene.meshes.push_back(std::move(environment));

    Mesh lightProps;
    lightProps.name = "Light markers";
    lightProps.skinned = false;
    addLightOrbs(lightProps, glow, bundle.lighting);
    scene.meshes.push_back(std::move(lightProps));

    return bundle;
}

}  // namespace cv

