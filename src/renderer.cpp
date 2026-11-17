#include "renderer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace softraster {
namespace {

static_assert(sizeof(float) == 4, "the rasterizer requires 32-bit IEEE-like float storage");

constexpr float kTangentEpsilon = 1.0e-12F;
constexpr float kGamma = 1.0F / 2.2F;

struct GeometryTangents {
    std::vector<Vec3> normal;
    std::vector<Vec3> tangent;
    std::vector<Vec3> bitangent;
};

struct PreparedObject {
    const Mesh* mesh = nullptr;
    const Material* material = nullptr;
    std::vector<Vec3> worldPositions;
    std::vector<Vec3> worldNormals;
    std::vector<Vec3> worldTangents;
    std::vector<Vec3> worldBitangents;
};

struct ClipVertex {
    Vec4 clip;
    Vec3 world;
    Vec3 normal;
    Vec3 tangent;
    Vec3 bitangent;
    Vec2 uv;
};

struct ScreenVertex {
    float x = 0.0F;
    float y = 0.0F;
    float depth = 0.0F;
    float invW = 0.0F;
    Vec3 world;
    Vec3 normal;
    Vec3 tangent;
    Vec3 bitangent;
    Vec2 uv;
};

struct TransparentTriangle {
    const PreparedObject* object = nullptr;
    std::size_t triangle = 0;
    float cameraDepth = 0.0F;
};

float cross2(Vec2 a, Vec2 b, Vec2 c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

Vec3 stableTangent(Vec3 normal) {
    Vec3 axis = std::abs(normal.z) < 0.999F ? Vec3{0.0F, 0.0F, 1.0F}
                                             : Vec3{0.0F, 1.0F, 0.0F};
    return normalize(cross(axis, normal), {1.0F, 0.0F, 0.0F});
}

float planeDistance(Vec4 point, int plane) {
    switch (plane) {
        case 0: return point.x + point.w;
        case 1: return point.w - point.x;
        case 2: return point.y + point.w;
        case 3: return point.w - point.y;
        case 4: return point.z + point.w;
        default: return point.w - point.z;
    }
}

bool needsClipping(const std::array<ClipVertex, 3>& vertices) {
    for (const ClipVertex& vertex : vertices) {
        for (int plane = 0; plane < 6; ++plane) {
            if (planeDistance(vertex.clip, plane) < 0.0F) {
                return true;
            }
        }
    }
    return false;
}

ClipVertex interpolateClip(const ClipVertex& a, const ClipVertex& b, float t) {
    ClipVertex result;
    result.clip = a.clip + (b.clip - a.clip) * t;
    result.world = a.world + (b.world - a.world) * t;
    result.normal = a.normal + (b.normal - a.normal) * t;
    result.tangent = a.tangent + (b.tangent - a.tangent) * t;
    result.bitangent = a.bitangent + (b.bitangent - a.bitangent) * t;
    result.uv = a.uv + (b.uv - a.uv) * t;
    return result;
}

void clipAgainstPlane(const std::vector<ClipVertex>& input, int plane,
                      std::vector<ClipVertex>& output) {
    output.clear();
    if (input.empty()) {
        return;
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
        const ClipVertex& current = input[i];
        const ClipVertex& next = input[(i + 1) % input.size()];
        const float currentDistance = planeDistance(current.clip, plane);
        const float nextDistance = planeDistance(next.clip, plane);
        const bool currentInside = currentDistance >= 0.0F;
        const bool nextInside = nextDistance >= 0.0F;
        if (currentInside) {
            output.push_back(current);
        }
        if (currentInside != nextInside) {
            const float denominator = currentDistance - nextDistance;
            if (std::abs(denominator) > 1.0e-20F) {
                const float t = std::clamp(currentDistance / denominator, 0.0F, 1.0F);
                output.push_back(interpolateClip(current, next, t));
            }
        }
    }
}

void clipToFrustum(const std::array<ClipVertex, 3>& triangle,
                   std::vector<ClipVertex>& output) {
    std::vector<ClipVertex> current(triangle.begin(), triangle.end());
    std::vector<ClipVertex> next;
    for (int plane = 0; plane < 6; ++plane) {
        clipAgainstPlane(current, plane, next);
        current.swap(next);
        if (current.empty()) {
            break;
        }
    }
    output = std::move(current);
}

ScreenVertex toScreen(const ClipVertex& vertex, int width, int height, int viewportX,
                      int viewportY) {
    ScreenVertex result;
    if (std::abs(vertex.clip.w) <= 1.0e-20F) {
        result.invW = 0.0F;
        return result;
    }
    const float invW = 1.0F / vertex.clip.w;
    result.invW = invW;
    const float ndcX = vertex.clip.x * invW;
    const float ndcY = vertex.clip.y * invW;
    const float ndcZ = vertex.clip.z * invW;
    result.x = static_cast<float>(viewportX) +
               (ndcX * 0.5F + 0.5F) * static_cast<float>(width);
    result.y = static_cast<float>(viewportY) +
               (0.5F - ndcY * 0.5F) * static_cast<float>(height);
    result.depth = ndcZ * 0.5F + 0.5F;
    result.world = vertex.world;
    result.normal = vertex.normal;
    result.tangent = vertex.tangent;
    result.bitangent = vertex.bitangent;
    result.uv = vertex.uv;
    return result;
}

GeometryTangents buildGeometryTangents(const Mesh& mesh) {
    GeometryTangents result;
    result.normal.assign(mesh.vertices.size(), Vec3{});
    result.tangent.assign(mesh.vertices.size(), Vec3{});
    result.bitangent.assign(mesh.vertices.size(), Vec3{});

    std::vector<Vec3> fallbackNormals(mesh.vertices.size(), Vec3{});
    std::vector<Vec3> tangents(mesh.vertices.size(), Vec3{});
    std::vector<Vec3> bitangents(mesh.vertices.size(), Vec3{});

    for (std::size_t triangle = 0; triangle < mesh.indices.size(); triangle += 3) {
        const std::uint32_t i0 = mesh.indices[triangle];
        const std::uint32_t i1 = mesh.indices[triangle + 1];
        const std::uint32_t i2 = mesh.indices[triangle + 2];
        const Vertex& v0 = mesh.vertices[i0];
        const Vertex& v1 = mesh.vertices[i1];
        const Vertex& v2 = mesh.vertices[i2];

        const Vec3 edge1 = v1.position - v0.position;
        const Vec3 edge2 = v2.position - v0.position;
        const Vec3 faceNormal = cross(edge1, edge2);
        if (lengthSquared(faceNormal) > kTangentEpsilon) {
            fallbackNormals[i0] = fallbackNormals[i0] + faceNormal;
            fallbackNormals[i1] = fallbackNormals[i1] + faceNormal;
            fallbackNormals[i2] = fallbackNormals[i2] + faceNormal;
        }

        const Vec2 duv1 = v1.uv - v0.uv;
        const Vec2 duv2 = v2.uv - v0.uv;
        const float determinant = duv1.x * duv2.y - duv1.y * duv2.x;
        const float uvScale =
            std::max({std::abs(duv1.x), std::abs(duv1.y), std::abs(duv2.x),
                      std::abs(duv2.y)});
        if (!std::isfinite(determinant) ||
            std::abs(determinant) <= std::max(1.0e-12F, uvScale * uvScale * 1.0e-8F)) {
            continue;
        }

        const Vec3 tangent = (edge1 * duv2.y - edge2 * duv1.y) / determinant;
        const Vec3 bitangent = (edge2 * duv1.x - edge1 * duv2.x) / determinant;
        if (!isFinite(tangent) || !isFinite(bitangent) ||
            lengthSquared(tangent) <= kTangentEpsilon ||
            lengthSquared(bitangent) <= kTangentEpsilon) {
            continue;
        }
        tangents[i0] = tangents[i0] + tangent;
        tangents[i1] = tangents[i1] + tangent;
        tangents[i2] = tangents[i2] + tangent;
        bitangents[i0] = bitangents[i0] + bitangent;
        bitangents[i1] = bitangents[i1] + bitangent;
        bitangents[i2] = bitangents[i2] + bitangent;
    }

    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        Vec3 normal = normalize(mesh.vertices[i].normal, {0.0F, 0.0F, 0.0F});
        if (lengthSquared(normal) <= kTangentEpsilon) {
            normal = normalize(fallbackNormals[i], {0.0F, 1.0F, 0.0F});
        }

        Vec3 tangent = tangents[i] - normal * dot(normal, tangents[i]);
        if (lengthSquared(tangent) <= kTangentEpsilon) {
            tangent = stableTangent(normal);
        } else {
            tangent = normalize(tangent);
        }

        Vec3 bitangent =
            bitangents[i] - normal * dot(normal, bitangents[i]) - tangent * dot(tangent, bitangents[i]);
        if (lengthSquared(bitangent) <= kTangentEpsilon) {
            bitangent = cross(normal, tangent);
        } else {
            bitangent = normalize(bitangent);
            if (dot(bitangent, cross(normal, tangent)) < 0.0F) {
                bitangent = -bitangent;
            }
        }
        result.normal[i] = normal;
        result.tangent[i] = tangent;
        result.bitangent[i] = bitangent;
    }
    return result;
}

std::uint64_t estimatePeakMemory(const Scene& scene, std::uint64_t transparentTriangles,
                                 std::uint64_t ppmBytes) {
    std::uint64_t meshVertices = 0;
    std::uint64_t meshIndices = 0;
    for (const Mesh& mesh : scene.meshes) {
        meshVertices += static_cast<std::uint64_t>(mesh.vertices.size());
        meshIndices += static_cast<std::uint64_t>(mesh.indices.size());
    }
    std::uint64_t instanceVertices = 0;
    for (const SceneObject& object : scene.objects) {
        if (object.meshIndex >= 0 &&
            static_cast<std::size_t>(object.meshIndex) < scene.meshes.size()) {
            instanceVertices +=
                static_cast<std::uint64_t>(scene.meshes[object.meshIndex].vertices.size());
        }
    }
    std::uint64_t texturePixels = 0;
    for (const Material& material : scene.materials) {
        if (material.normalTexture.has_value()) {
            texturePixels += static_cast<std::uint64_t>(material.normalTexture->pixels.size());
        }
    }
    const std::uint64_t pixels =
        static_cast<std::uint64_t>(scene.imageWidth) * static_cast<std::uint64_t>(scene.imageHeight);
    const std::uint64_t shadowPixels =
        scene.directionalLight.shadowEnabled
            ? static_cast<std::uint64_t>(scene.directionalLight.shadowMapSize) *
                  static_cast<std::uint64_t>(scene.directionalLight.shadowMapSize)
            : 0ULL;
    const std::uint64_t sceneDomEstimate = static_cast<std::uint64_t>(scene.sourceBytes) * 8ULL;
    return sceneDomEstimate + meshVertices * 32ULL + meshIndices * 4ULL +
           texturePixels * 12ULL + instanceVertices * 72ULL + pixels * 16ULL +
           shadowPixels * 4ULL + transparentTriangles * 56ULL + ppmBytes + 2ULL * 1024ULL * 1024ULL +
           static_cast<std::uint64_t>(scene.objects.size()) * 192ULL +
           static_cast<std::uint64_t>(scene.meshes.size()) * 128ULL +
           static_cast<std::uint64_t>(scene.materials.size()) * 192ULL;
}

std::uint64_t countTransparentTriangles(const Scene& scene) {
    std::uint64_t count = 0;
    for (const SceneObject& object : scene.objects) {
        const Material& material =
            scene.materials[static_cast<std::size_t>(object.materialIndex)];
        if (!material.isTransparent()) {
            continue;
        }
        count += static_cast<std::uint64_t>(
                     scene.meshes[static_cast<std::size_t>(object.meshIndex)].indices.size()) /
                 3ULL;
    }
    return count;
}

std::uint32_t floatBits(float value) {
    if (value == 0.0F) {
        value = 0.0F;
    }
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::uint64_t countInversions(const std::vector<TransparentTriangle>& triangles,
                              const std::vector<std::size_t>& sortedOrder) {
    const std::size_t count = triangles.size();
    if (count < 2) {
        return 0;
    }
    std::vector<std::size_t> rank(count, 0);
    for (std::size_t position = 0; position < count; ++position) {
        rank[sortedOrder[position]] = position;
    }
    std::vector<std::uint64_t> tree(count + 1, 0);
    const auto add = [&tree, count](std::size_t index) {
        for (std::size_t i = index + 1; i <= count; i += i & (~i + 1)) {
            tree[i] += 1;
        }
    };
    const auto query = [&tree](std::size_t index) {
        std::uint64_t result = 0;
        for (std::size_t i = index; i > 0; i -= i & (~i + 1)) {
            result += tree[i];
        }
        return result;
    };

    std::uint64_t inversions = 0;
    for (std::size_t position = count; position-- > 0;) {
        inversions += query(rank[position]);
        add(rank[position]);
    }
    return inversions;
}

class Renderer {
public:
    explicit Renderer(const Scene& scene) : scene_(scene) {}

    RenderResult run() {
        RenderResult result;
        stats_.width = scene_.imageWidth;
        stats_.height = scene_.imageHeight;
        stats_.viewportX = scene_.viewport.x;
        stats_.viewportY = scene_.viewport.y;
        stats_.viewportWidth = scene_.viewport.width;
        stats_.viewportHeight = scene_.viewport.height;
        stats_.shadowMapSize = scene_.directionalLight.shadowMapSize;
        stats_.shadowEnabled = scene_.directionalLight.shadowEnabled;

        bool validView = false;
        view_ = lookAtRh(scene_.camera.position, scene_.camera.target, scene_.camera.up,
                         validView);
        if (!validView) {
            result.error = makeError(ErrorCode::InvalidValue,
                                     "camera does not define a valid view matrix", "camera");
            return result;
        }
        const float aspect = static_cast<float>(scene_.viewport.width) /
                             static_cast<float>(scene_.viewport.height);
        projection_ = perspectiveRh(scene_.camera.fovYDegrees, aspect, scene_.camera.nearPlane,
                                    scene_.camera.farPlane);
        viewProjection_ = projection_ * view_;

        const std::uint64_t ppmEstimate =
            static_cast<std::uint64_t>(scene_.imageWidth) *
                static_cast<std::uint64_t>(scene_.imageHeight) * 3ULL +
            64ULL;
        const std::uint64_t estimatedPeak =
            estimatePeakMemory(scene_, countTransparentTriangles(scene_), ppmEstimate);
        if (estimatedPeak > scene_.maxMemoryBytes) {
            result.error = makeError(
                ErrorCode::MemoryLimitExceeded,
                "estimated peak memory exceeds limits.maxMemoryBytes",
                "limits.maxMemoryBytes");
            return result;
        }

        prepareObjects();
        if (!collectTransparentTriangles(result)) {
            return result;
        }

        colorBuffer_.assign(static_cast<std::size_t>(scene_.imageWidth) *
                                static_cast<std::size_t>(scene_.imageHeight),
                            scene_.backgroundColor);
        depthBuffer_.assign(static_cast<std::size_t>(scene_.imageWidth) *
                                static_cast<std::size_t>(scene_.imageHeight),
                            std::numeric_limits<float>::infinity());

        buildShadowMap();
        renderOpaque();
        renderTransparent();

        result.frame = encodePpm(estimatedPeak);
        if (!result.frame.has_value()) {
            result.error = makeError(ErrorCode::MemoryLimitExceeded,
                                     "unable to allocate the output PPM buffer", "output");
        }
        return result;
    }

private:
    const Scene& scene_;
    RenderStats stats_;
    std::vector<PreparedObject> objects_;
    std::vector<TransparentTriangle> transparentTriangles_;
    std::vector<Vec3> colorBuffer_;
    std::vector<float> depthBuffer_;
    std::vector<float> shadowDepth_;
    Mat4 view_;
    Mat4 projection_;
    Mat4 viewProjection_;
    Mat4 shadowViewProjection_;
    Vec3 shadowCenter_{};
    float shadowRadius_ = 1.0F;
    int shadowResolution_ = 0;
    bool shadowActive_ = false;

    void prepareObjects() {
        std::vector<GeometryTangents> cache(scene_.meshes.size());
        std::vector<bool> cacheReady(scene_.meshes.size(), false);
        objects_.reserve(scene_.objects.size());
        for (const SceneObject& object : scene_.objects) {
            const Mesh& mesh = scene_.meshes[static_cast<std::size_t>(object.meshIndex)];
            const Material& material =
                scene_.materials[static_cast<std::size_t>(object.materialIndex)];
            const std::size_t meshIndex = static_cast<std::size_t>(object.meshIndex);
            if (!cacheReady[meshIndex]) {
                cache[meshIndex] = buildGeometryTangents(mesh);
                cacheReady[meshIndex] = true;
            }

            bool invertible = false;
            const Mat4 normalMatrix = transpose(inverse(object.model, invertible));
            PreparedObject prepared;
            prepared.mesh = &mesh;
            prepared.material = &material;
            prepared.worldPositions.resize(mesh.vertices.size());
            prepared.worldNormals.resize(mesh.vertices.size());
            prepared.worldTangents.resize(mesh.vertices.size());
            prepared.worldBitangents.resize(mesh.vertices.size());
            for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
                const Vec4 worldPoint =
                    transformPoint(object.model, mesh.vertices[i].position);
                prepared.worldPositions[i] =
                    Vec3{worldPoint.x, worldPoint.y, worldPoint.z};
                Vec3 normal =
                    normalize(transformVector(normalMatrix, cache[meshIndex].normal[i]),
                              {0.0F, 1.0F, 0.0F});
                if (lengthSquared(normal) <= kTangentEpsilon) {
                    normal = Vec3{0.0F, 1.0F, 0.0F};
                }
                Vec3 tangent =
                    transformVector(object.model, cache[meshIndex].tangent[i]);
                tangent = tangent - normal * dot(normal, tangent);
                if (lengthSquared(tangent) <= kTangentEpsilon) {
                    tangent = stableTangent(normal);
                } else {
                    tangent = normalize(tangent);
                }
                Vec3 bitangent =
                    transformVector(object.model, cache[meshIndex].bitangent[i]);
                bitangent = bitangent - normal * dot(normal, bitangent) -
                            tangent * dot(tangent, bitangent);
                if (lengthSquared(bitangent) <= kTangentEpsilon) {
                    bitangent = cross(normal, tangent);
                } else {
                    bitangent = normalize(bitangent);
                    if (dot(bitangent, cross(normal, tangent)) < 0.0F) {
                        bitangent = -bitangent;
                    }
                }
                prepared.worldNormals[i] = normal;
                prepared.worldTangents[i] = tangent;
                prepared.worldBitangents[i] = bitangent;
            }
            objects_.push_back(std::move(prepared));
        }
    }

    bool collectTransparentTriangles(RenderResult& result) {
        std::unordered_set<std::uint32_t> depths;
        for (const PreparedObject& object : objects_) {
            if (!object.material->isTransparent()) {
                continue;
            }
            const std::size_t triangleCount = object.mesh->indices.size() / 3ULL;
            transparentTriangles_.reserve(transparentTriangles_.size() + triangleCount);
            for (std::size_t triangle = 0; triangle < triangleCount; ++triangle) {
                const std::size_t index = triangle * 3ULL;
                const Vec3 p0 = object.worldPositions[object.mesh->indices[index]];
                const Vec3 p1 = object.worldPositions[object.mesh->indices[index + 1]];
                const Vec3 p2 = object.worldPositions[object.mesh->indices[index + 2]];
                const Vec3 centroid = (p0 + p1 + p2) / 3.0F;
                const float cameraDepth = (view_ * Vec4{centroid.x, centroid.y, centroid.z, 1.0F}).z;
                const std::uint32_t bits = floatBits(cameraDepth);
                if (!depths.insert(bits).second) {
                    result.error = makeError(
                        ErrorCode::DuplicateTransparentDepth,
                        "two transparent triangles have the same camera-space centroid depth",
                        "transparency.depthWriteMaxOpacity");
                    return false;
                }
                transparentTriangles_.push_back({&object, triangle, cameraDepth});
            }
        }
        stats_.transparentTriangles = transparentTriangles_.size();
        return true;
    }

    void buildShadowMap() {
        if (!scene_.directionalLight.shadowEnabled) {
            return;
        }
        shadowResolution_ = scene_.directionalLight.shadowMapSize;
        shadowDepth_.assign(static_cast<std::size_t>(shadowResolution_) *
                                static_cast<std::size_t>(shadowResolution_),
                            std::numeric_limits<float>::infinity());

        Vec3 minimum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                     std::numeric_limits<float>::max()};
        Vec3 maximum{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                     -std::numeric_limits<float>::max()};
        for (const PreparedObject& object : objects_) {
            for (const Vec3 position : object.worldPositions) {
                minimum = minComponents(minimum, position);
                maximum = maxComponents(maximum, position);
            }
        }
        if (!isFinite(minimum) || !isFinite(maximum)) {
            return;
        }
        shadowCenter_ = (minimum + maximum) * 0.5F;
        shadowRadius_ = std::max(length(maximum - minimum) * 0.5F, 0.25F);
        const Vec3 lightDirection = normalize(scene_.directionalLight.directionToLight);
        const float distance = shadowRadius_ * 2.0F + 1.0F;
        const Vec3 eye = shadowCenter_ + lightDirection * distance;
        Vec3 up{0.0F, 1.0F, 0.0F};
        if (std::abs(dot(lightDirection, up)) > 0.99F) {
            up = Vec3{1.0F, 0.0F, 0.0F};
        }
        bool valid = false;
        const Mat4 lightView = lookAtRh(eye, shadowCenter_, up, valid);
        if (!valid) {
            return;
        }
        const float extent = shadowRadius_ * 1.1F + 0.1F;
        const Mat4 lightProjection =
            orthographicRh(-extent, extent, -extent, extent, 0.01F, distance + shadowRadius_ * 3.0F + 2.0F);
        shadowViewProjection_ = lightProjection * lightView;
        shadowActive_ = true;

        for (const PreparedObject& object : objects_) {
            for (std::size_t triangle = 0; triangle < object.mesh->indices.size() / 3ULL;
                 ++triangle) {
                rasterizeShadowTriangle(object, triangle);
            }
        }
    }

    void rasterizeShadowTriangle(const PreparedObject& object, std::size_t triangle) {
        const std::size_t base = triangle * 3ULL;
        std::array<ClipVertex, 3> vertices;
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const std::uint32_t vertexIndex = object.mesh->indices[base + corner];
            vertices[corner].world = object.worldPositions[vertexIndex];
            vertices[corner].clip =
                shadowViewProjection_ *
                Vec4{vertices[corner].world.x, vertices[corner].world.y,
                     vertices[corner].world.z, 1.0F};
            vertices[corner].normal = object.worldNormals[vertexIndex];
            vertices[corner].tangent = object.worldTangents[vertexIndex];
            vertices[corner].bitangent = object.worldBitangents[vertexIndex];
            vertices[corner].uv = object.mesh->vertices[vertexIndex].uv;
        }
        std::vector<ClipVertex> polygon;
        clipToFrustum(vertices, polygon);
        if (polygon.size() < 3) {
            return;
        }
        for (std::size_t i = 1; i + 1 < polygon.size(); ++i) {
            const ScreenVertex a =
                toScreen(polygon[0], shadowResolution_, shadowResolution_, 0, 0);
            const ScreenVertex b =
                toScreen(polygon[i], shadowResolution_, shadowResolution_, 0, 0);
            const ScreenVertex c =
                toScreen(polygon[i + 1], shadowResolution_, shadowResolution_, 0, 0);
            if (a.invW == 0.0F || b.invW == 0.0F || c.invW == 0.0F) {
                continue;
            }
            rasterizeShadowFragment(a, b, c);
        }
    }

    void rasterizeShadowFragment(const ScreenVertex& a, const ScreenVertex& b,
                                 const ScreenVertex& c) {
        const float area = cross2({a.x, a.y}, {b.x, b.y}, {c.x, c.y});
        if (std::abs(area) <= 1.0e-12F) {
            return;
        }
        const int minX =
            std::max(0, static_cast<int>(std::floor(std::min({a.x, b.x, c.x}))));
        const int maxX = std::min(
            shadowResolution_ - 1,
            static_cast<int>(std::ceil(std::max({a.x, b.x, c.x}))));
        const int minY =
            std::max(0, static_cast<int>(std::floor(std::min({a.y, b.y, c.y}))));
        const int maxY = std::min(
            shadowResolution_ - 1,
            static_cast<int>(std::ceil(std::max({a.y, b.y, c.y}))));

        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                const Vec2 point{static_cast<float>(x) + 0.5F,
                                 static_cast<float>(y) + 0.5F};
                const float w0 = cross2(point, {b.x, b.y}, {c.x, c.y}) / area;
                const float w1 = cross2({a.x, a.y}, point, {c.x, c.y}) / area;
                const float w2 = 1.0F - w0 - w1;
                if (w0 < 0.0F || w1 < 0.0F || w2 < 0.0F) {
                    continue;
                }
                const float depth = w0 * a.depth + w1 * b.depth + w2 * c.depth;
                if (depth < 0.0F || depth > 1.0F) {
                    continue;
                }
                const std::size_t index =
                    static_cast<std::size_t>(y) * static_cast<std::size_t>(shadowResolution_) +
                    static_cast<std::size_t>(x);
                if (depth < shadowDepth_[index]) {
                    shadowDepth_[index] = depth;
                }
            }
        }
    }

    void renderOpaque() {
        for (const PreparedObject& object : objects_) {
            if (object.material->isTransparent()) {
                continue;
            }
            for (std::size_t triangle = 0; triangle < object.mesh->indices.size() / 3ULL;
                 ++triangle) {
                renderColorTriangle(object, triangle, false);
            }
        }
    }

    void renderTransparent() {
        if (transparentTriangles_.empty()) {
            return;
        }
        std::vector<std::size_t> sortedOrder(transparentTriangles_.size());
        std::iota(sortedOrder.begin(), sortedOrder.end(), 0ULL);
        std::stable_sort(sortedOrder.begin(), sortedOrder.end(),
                         [this](std::size_t left, std::size_t right) {
                             return transparentTriangles_[left].cameraDepth <
                                    transparentTriangles_[right].cameraDepth;
                         });
        stats_.transparentSortSwaps = countInversions(transparentTriangles_, sortedOrder);
        for (const std::size_t index : sortedOrder) {
            const TransparentTriangle& triangle = transparentTriangles_[index];
            renderColorTriangle(*triangle.object, triangle.triangle, true);
        }
    }

    void renderColorTriangle(const PreparedObject& object, std::size_t triangle,
                             bool transparent) {
        const std::size_t base = triangle * 3ULL;
        std::array<ClipVertex, 3> vertices;
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const std::uint32_t vertexIndex = object.mesh->indices[base + corner];
            vertices[corner].world = object.worldPositions[vertexIndex];
            vertices[corner].clip =
                viewProjection_ * Vec4{vertices[corner].world.x, vertices[corner].world.y,
                                       vertices[corner].world.z, 1.0F};
            vertices[corner].normal = object.worldNormals[vertexIndex];
            vertices[corner].tangent = object.worldTangents[vertexIndex];
            vertices[corner].bitangent = object.worldBitangents[vertexIndex];
            vertices[corner].uv = object.mesh->vertices[vertexIndex].uv;
        }
        if (needsClipping(vertices)) {
            ++stats_.clippedTriangles;
        }
        std::vector<ClipVertex> polygon;
        clipToFrustum(vertices, polygon);
        if (polygon.size() < 3) {
            return;
        }

        bool anyCoverage = false;
        for (std::size_t i = 1; i + 1 < polygon.size(); ++i) {
            const ScreenVertex a =
                toScreen(polygon[0], scene_.viewport.width, scene_.viewport.height,
                         scene_.viewport.x, scene_.viewport.y);
            const ScreenVertex b =
                toScreen(polygon[i], scene_.viewport.width, scene_.viewport.height,
                         scene_.viewport.x, scene_.viewport.y);
            const ScreenVertex c =
                toScreen(polygon[i + 1], scene_.viewport.width, scene_.viewport.height,
                         scene_.viewport.x, scene_.viewport.y);
            if (a.invW == 0.0F || b.invW == 0.0F || c.invW == 0.0F) {
                continue;
            }
            anyCoverage = rasterizeColorFragment(a, b, c, object, transparent) || anyCoverage;
        }
        if (anyCoverage) {
            ++stats_.rasterizedTriangles;
        }
    }

    bool rasterizeColorFragment(const ScreenVertex& a, const ScreenVertex& b,
                                const ScreenVertex& c, const PreparedObject& object,
                                bool transparent) {
        const float area = cross2({a.x, a.y}, {b.x, b.y}, {c.x, c.y});
        if (std::abs(area) <= 1.0e-12F) {
            return false;
        }
        // The project defines counter-clockwise world-space triangles as front-facing.
        // Viewport Y points down, so front faces have a negative signed screen area.
        if (area >= 0.0F) {
            return false;
        }

        const int minX = std::max(
            scene_.viewport.x,
            static_cast<int>(std::floor(std::min({a.x, b.x, c.x}))));
        const int maxX = std::min(
            scene_.viewport.x + scene_.viewport.width - 1,
            static_cast<int>(std::ceil(std::max({a.x, b.x, c.x}))));
        const int minY = std::max(
            scene_.viewport.y,
            static_cast<int>(std::floor(std::min({a.y, b.y, c.y}))));
        const int maxY = std::min(
            scene_.viewport.y + scene_.viewport.height - 1,
            static_cast<int>(std::ceil(std::max({a.y, b.y, c.y}))));
        if (minX > maxX || minY > maxY) {
            return false;
        }

        bool covered = false;
        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                const Vec2 point{static_cast<float>(x) + 0.5F,
                                 static_cast<float>(y) + 0.5F};
                const float w0 = cross2(point, {b.x, b.y}, {c.x, c.y}) / area;
                const float w1 = cross2({a.x, a.y}, point, {c.x, c.y}) / area;
                const float w2 = 1.0F - w0 - w1;
                if (w0 < 0.0F || w1 < 0.0F || w2 < 0.0F) {
                    continue;
                }
                covered = true;
                const float depth = w0 * a.depth + w1 * b.depth + w2 * c.depth;
                if (!std::isfinite(depth) || depth < 0.0F || depth > 1.0F) {
                    continue;
                }
                const std::size_t index =
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(scene_.imageWidth) +
                    static_cast<std::size_t>(x);
                if (depth >= depthBuffer_[index]) {
                    continue;
                }

                const float denominator = w0 * a.invW + w1 * b.invW + w2 * c.invW;
                if (!(std::abs(denominator) > 1.0e-20F)) {
                    continue;
                }
                const auto interpolateVec3 = [&](Vec3 va, Vec3 vb, Vec3 vc) {
                    return (va * (w0 * a.invW) + vb * (w1 * b.invW) +
                            vc * (w2 * c.invW)) /
                           denominator;
                };
                const auto interpolateVec2 = [&](Vec2 va, Vec2 vb, Vec2 vc) {
                    return (va * (w0 * a.invW) + vb * (w1 * b.invW) +
                            vc * (w2 * c.invW)) /
                           denominator;
                };
                const Vec3 world = interpolateVec3(a.world, b.world, c.world);
                const Vec3 normal = interpolateVec3(a.normal, b.normal, c.normal);
                const Vec3 tangent = interpolateVec3(a.tangent, b.tangent, c.tangent);
                const Vec3 bitangent =
                    interpolateVec3(a.bitangent, b.bitangent, c.bitangent);
                const Vec2 uv = interpolateVec2(a.uv, b.uv, c.uv);
                const Vec3 shaded =
                    shadeFragment(world, normal, tangent, bitangent, uv, *object.material);

                if (!transparent) {
                    colorBuffer_[index] = shaded;
                    depthBuffer_[index] = depth;
                    ++stats_.opaqueFragments;
                } else {
                    const float alpha = object.material->opacity;
                    colorBuffer_[index] =
                        shaded * alpha + colorBuffer_[index] * (1.0F - alpha);
                    ++stats_.transparentFragments;
                    ++stats_.alphaBlendedFragments;
                    if (alpha < scene_.transparentDepthWriteMaxOpacity) {
                        depthBuffer_[index] = depth;
                        ++stats_.transparentDepthWrites;
                    }
                }
            }
        }
        return covered;
    }

    Vec3 sampleNormalTexture(const Material& material, Vec2 uv, Vec3 normal, Vec3 tangent,
                             Vec3 bitangent) const {
        if (!material.normalTexture.has_value() || material.normalTexture->strength <= 0.0F) {
            return normal;
        }
        const NormalTexture& texture = *material.normalTexture;
        const float u = uv.x - std::floor(uv.x);
        const float v = uv.y - std::floor(uv.y);
        const float sampleX = u * static_cast<float>(texture.width) - 0.5F;
        const float sampleY = v * static_cast<float>(texture.height) - 0.5F;
        const int x0 = static_cast<int>(std::floor(sampleX));
        const int y0 = static_cast<int>(std::floor(sampleY));
        const float fx = sampleX - std::floor(sampleX);
        const float fy = sampleY - std::floor(sampleY);
        const auto wrap = [](int value, int size) {
            int wrapped = value % size;
            if (wrapped < 0) {
                wrapped += size;
            }
            return wrapped;
        };
        const auto fetch = [&](int x, int y) {
            const int wrappedX = wrap(x, texture.width);
            const int wrappedY = wrap(y, texture.height);
            return texture.pixels[static_cast<std::size_t>(wrappedY) *
                                      static_cast<std::size_t>(texture.width) +
                                  static_cast<std::size_t>(wrappedX)];
        };
        const Vec3 s00 = fetch(x0, y0);
        const Vec3 s10 = fetch(x0 + 1, y0);
        const Vec3 s01 = fetch(x0, y0 + 1);
        const Vec3 s11 = fetch(x0 + 1, y0 + 1);
        Vec3 sampled = s00 * ((1.0F - fx) * (1.0F - fy)) +
                       s10 * (fx * (1.0F - fy)) + s01 * ((1.0F - fx) * fy) +
                       s11 * (fx * fy);
        sampled = Vec3{sampled.x * texture.strength, sampled.y * texture.strength,
                       sampled.z};
        if (!isFinite(sampled) || lengthSquared(sampled) <= kTangentEpsilon) {
            return normal;
        }
        sampled = normalize(sampled, normal);
        return normalize(tangent * sampled.x + bitangent * sampled.y + normal * sampled.z,
                         normal);
    }

    Vec3 shadeFragment(Vec3 world, Vec3 interpolatedNormal, Vec3 interpolatedTangent,
                       Vec3 interpolatedBitangent, Vec2 uv,
                       const Material& material) {
        Vec3 normal = normalize(interpolatedNormal, {0.0F, 1.0F, 0.0F});
        Vec3 tangent =
            interpolatedTangent - normal * dot(normal, interpolatedTangent);
        if (lengthSquared(tangent) <= kTangentEpsilon) {
            tangent = stableTangent(normal);
        } else {
            tangent = normalize(tangent);
        }
        Vec3 bitangent = interpolatedBitangent -
                         normal * dot(normal, interpolatedBitangent) -
                         tangent * dot(tangent, interpolatedBitangent);
        const Vec3 geometricBitangent = cross(normal, tangent);
        if (lengthSquared(bitangent) <= kTangentEpsilon) {
            bitangent = geometricBitangent;
        } else {
            bitangent = normalize(bitangent);
            if (dot(bitangent, geometricBitangent) < 0.0F) {
                bitangent = -bitangent;
            }
        }
        normal = sampleNormalTexture(material, uv, normal, tangent, bitangent);

        const Vec3 viewDirection =
            normalize(scene_.camera.position - world, {0.0F, 0.0F, 1.0F});
        Vec3 result = material.baseColor * scene_.ambientColor;

        if (scene_.directionalLight.intensity > 0.0F) {
            const Vec3 lightDirection =
                normalize(scene_.directionalLight.directionToLight);
            const float shadow = sampleShadow(world, normal);
            const float ndotl = std::max(dot(normal, lightDirection), 0.0F);
            if (ndotl > 0.0F) {
                const Vec3 lightColor =
                    scene_.directionalLight.color * scene_.directionalLight.intensity;
                result = result + material.baseColor * lightColor * (ndotl * shadow);
                const Vec3 reflection =
                    normalize(normal * (2.0F * ndotl) - lightDirection, normal);
                const float specular =
                    std::pow(std::max(dot(reflection, viewDirection), 0.0F),
                             material.shininess);
                result = result + material.specularColor * lightColor *
                                      (specular * shadow);
            }
        }

        for (const PointLight& light : scene_.pointLights) {
            if (light.intensity <= 0.0F) {
                continue;
            }
            const Vec3 toLight = light.position - world;
            const float distanceSquared = lengthSquared(toLight);
            if (!(distanceSquared > 1.0e-8F)) {
                continue;
            }
            const float distance = std::sqrt(distanceSquared);
            const Vec3 lightDirection = toLight / distance;
            const float attenuation = 1.0F / std::max(distanceSquared, 1.0e-4F);
            const float ndotl = std::max(dot(normal, lightDirection), 0.0F);
            if (ndotl <= 0.0F) {
                continue;
            }
            const Vec3 lightColor = light.color * (light.intensity * attenuation);
            result = result + material.baseColor * lightColor * ndotl;
            const Vec3 reflection =
                normalize(normal * (2.0F * ndotl) - lightDirection, normal);
            const float specular =
                std::pow(std::max(dot(reflection, viewDirection), 0.0F),
                         material.shininess);
            result = result + material.specularColor * lightColor * specular;
        }
        return result;
    }

    float sampleShadow(Vec3 world, Vec3 normal) {
        if (!scene_.directionalLight.shadowEnabled || !shadowActive_ || shadowDepth_.empty()) {
            return 1.0F;
        }
        const Vec3 biasedWorld =
            world + normal * scene_.directionalLight.normalBias;
        const Vec4 lightClip =
            shadowViewProjection_ *
            Vec4{biasedWorld.x, biasedWorld.y, biasedWorld.z, 1.0F};
        if (!(lightClip.w > 1.0e-8F)) {
            return 1.0F;
        }
        const float invW = 1.0F / lightClip.w;
        const float ndcX = lightClip.x * invW;
        const float ndcY = lightClip.y * invW;
        const float ndcZ = lightClip.z * invW;
        const float u = ndcX * 0.5F + 0.5F;
        const float v = 0.5F - ndcY * 0.5F;
        if (u < 0.0F || u > 1.0F || v < 0.0F || v > 1.0F || ndcZ < -1.0F ||
            ndcZ > 1.0F) {
            return 1.0F;
        }
        const float currentDepth = ndcZ * 0.5F + 0.5F;
        const int centerX = static_cast<int>(std::floor(u * static_cast<float>(shadowResolution_)));
        const int centerY = static_cast<int>(std::floor(v * static_cast<float>(shadowResolution_)));
        const int radius = scene_.directionalLight.pcfRadius;
        int litSamples = 0;
        int totalSamples = 0;
        for (int offsetY = -radius; offsetY <= radius; ++offsetY) {
            for (int offsetX = -radius; offsetX <= radius; ++offsetX) {
                ++totalSamples;
                ++stats_.shadowSampleCount;
                const int x = centerX + offsetX;
                const int y = centerY + offsetY;
                if (x < 0 || y < 0 || x >= shadowResolution_ || y >= shadowResolution_) {
                    ++litSamples;
                    continue;
                }
                const float occluder =
                    shadowDepth_[static_cast<std::size_t>(y) *
                                     static_cast<std::size_t>(shadowResolution_) +
                                 static_cast<std::size_t>(x)];
                if (std::isfinite(occluder) &&
                    currentDepth <=
                        occluder + scene_.directionalLight.depthBias) {
                    ++litSamples;
                }
            }
        }
        return totalSamples == 0
                   ? 1.0F
                   : static_cast<float>(litSamples) / static_cast<float>(totalSamples);
    }

    RenderedFrame encodePpm(std::uint64_t estimatedPeak) {
        RenderedFrame frame;
        frame.width = scene_.imageWidth;
        frame.height = scene_.imageHeight;
        const std::string header = "P6\n" + std::to_string(scene_.imageWidth) + " " +
                                   std::to_string(scene_.imageHeight) + "\n255\n";
        const std::size_t pixelBytes =
            static_cast<std::size_t>(scene_.imageWidth) *
            static_cast<std::size_t>(scene_.imageHeight) * 3ULL;
        try {
            frame.ppm.reserve(header.size() + pixelBytes);
        } catch (const std::bad_alloc&) {
            return frame;
        }
        frame.ppm.insert(frame.ppm.end(), header.begin(), header.end());
        for (const Vec3 linear : colorBuffer_) {
            const Vec3 clamped = clamp01(linear);
            const auto gammaByte = [](float value) {
                const double corrected =
                    std::pow(static_cast<double>(value), static_cast<double>(kGamma));
                return static_cast<std::uint8_t>(
                    std::clamp(std::lround(corrected * 255.0), 0L, 255L));
            };
            frame.ppm.push_back(gammaByte(clamped.x));
            frame.ppm.push_back(gammaByte(clamped.y));
            frame.ppm.push_back(gammaByte(clamped.z));
        }
        frame.stats = stats_;
        frame.stats.ppmBytes = frame.ppm.size();
        frame.stats.peakMemoryEstimatedBytes =
            estimatePeakMemory(scene_, static_cast<std::uint64_t>(transparentTriangles_.size()),
                               static_cast<std::uint64_t>(frame.ppm.size()));
        if (frame.stats.peakMemoryEstimatedBytes < estimatedPeak) {
            frame.stats.peakMemoryEstimatedBytes = estimatedPeak;
        }
        return frame;
    }
};

}  // namespace

RenderResult renderScene(const Scene& scene) {
    Renderer renderer(scene);
    return renderer.run();
}

}  // namespace softraster
