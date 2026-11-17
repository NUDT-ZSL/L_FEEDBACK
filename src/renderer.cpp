#include "raster/renderer.h"

#include "raster/bmp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace raster {
namespace {

struct Plane {
    Vec3 normal;
    double distance = 0.0;
};

struct Aabb {
    Vec3 minimum;
    Vec3 maximum;
};

struct PreparedMaterial {
    const Material* material = nullptr;
    std::optional<Texture> baseTexture;
    std::optional<Texture> detailTexture;
};

struct RawVertex {
    Vec4 clip;
    Vec3 world;
    Vec3 normal;
    Vec2 uv;
    Vec4 color;
};

struct ScreenVertex {
    double x = 0.0;
    double y = 0.0;
    double inverseW = 0.0;
    double depth = 0.0;
    Vec3 world;
    Vec3 normal;
    Vec2 uv;
    Vec4 color;
};

Plane normalizedPlane(const Vec4& row) {
    Vec3 normal{row.x, row.y, row.z};
    const double len = length(normal);
    if (!(len > kEpsilon)) {
        return {};
    }
    normal = normal / len;
    return Plane{normal, row.w / len};
}

std::array<Plane, 6> extractFrustumPlanes(const Mat4& viewProjection) {
    const Vec4 row0{viewProjection.at(0, 0), viewProjection.at(0, 1),
                    viewProjection.at(0, 2), viewProjection.at(0, 3)};
    const Vec4 row1{viewProjection.at(1, 0), viewProjection.at(1, 1),
                    viewProjection.at(1, 2), viewProjection.at(1, 3)};
    const Vec4 row2{viewProjection.at(2, 0), viewProjection.at(2, 1),
                    viewProjection.at(2, 2), viewProjection.at(2, 3)};
    const Vec4 row3{viewProjection.at(3, 0), viewProjection.at(3, 1),
                    viewProjection.at(3, 2), viewProjection.at(3, 3)};

    return {
        normalizedPlane(row2),
        normalizedPlane(Vec4{row3.x - row2.x, row3.y - row2.y,
                             row3.z - row2.z, row3.w - row2.w}),
        normalizedPlane(Vec4{row3.x + row0.x, row3.y + row0.y,
                             row3.z + row0.z, row3.w + row0.w}),
        normalizedPlane(Vec4{row3.x - row0.x, row3.y - row0.y,
                             row3.z - row0.z, row3.w - row0.w}),
        normalizedPlane(Vec4{row3.x + row1.x, row3.y + row1.y,
                             row3.z + row1.z, row3.w + row1.w}),
        normalizedPlane(Vec4{row3.x - row1.x, row3.y - row1.y,
                             row3.z - row1.z, row3.w - row1.w}),
    };
}

bool aabbOutsideFrustum(const Aabb& box, const std::array<Plane, 6>& planes) {
    for (const Plane& plane : planes) {
        const Vec3 positive{
            plane.normal.x >= 0.0 ? box.maximum.x : box.minimum.x,
            plane.normal.y >= 0.0 ? box.maximum.y : box.minimum.y,
            plane.normal.z >= 0.0 ? box.maximum.z : box.minimum.z,
        };
        if (dot(plane.normal, positive) + plane.distance < -1.0e-9) {
            return true;
        }
    }
    return false;
}

double clipDistance(int planeIndex, const Vec4& position) {
    switch (planeIndex) {
        case 0:
            return position.z;
        case 1:
            return position.w - position.z;
        case 2:
            return position.x + position.w;
        case 3:
            return position.w - position.x;
        case 4:
            return position.y + position.w;
        case 5:
            return position.w - position.y;
        default:
            return -1.0;
    }
}

RawVertex interpolate(const RawVertex& a, const RawVertex& b, double t) {
    return RawVertex{
        Vec4{a.clip.x + (b.clip.x - a.clip.x) * t,
             a.clip.y + (b.clip.y - a.clip.y) * t,
             a.clip.z + (b.clip.z - a.clip.z) * t,
             a.clip.w + (b.clip.w - a.clip.w) * t},
        lerp(a.world, b.world, t),
        lerp(a.normal, b.normal, t),
        lerp(a.uv, b.uv, t),
        lerp(a.color, b.color, t),
    };
}

std::vector<RawVertex> clipAgainstPlane(const std::vector<RawVertex>& input,
                                        int planeIndex) {
    if (input.empty()) {
        return {};
    }

    std::vector<RawVertex> output;
    output.reserve(input.size() + 1U);
    RawVertex previous = input.back();
    double previousDistance = clipDistance(planeIndex, previous.clip);
    bool previousInside = previousDistance >= 0.0;

    for (const RawVertex& current : input) {
        const double currentDistance = clipDistance(planeIndex, current.clip);
        const bool currentInside = currentDistance >= 0.0;
        if (currentInside != previousInside) {
            const double denominator = previousDistance - currentDistance;
            double t = 0.0;
            if (std::abs(denominator) > kEpsilon) {
                t = std::clamp(previousDistance / denominator, 0.0, 1.0);
            }
            output.push_back(interpolate(previous, current, t));
        }
        if (currentInside) {
            output.push_back(current);
        }
        previous = current;
        previousDistance = currentDistance;
        previousInside = currentInside;
    }
    return output;
}

std::vector<RawVertex> clipTriangle(const RawVertex triangle[3]) {
    std::vector<RawVertex> polygon{triangle[0], triangle[1], triangle[2]};
    for (int plane = 0; plane < 6 && polygon.size() >= 3U; ++plane) {
        polygon = clipAgainstPlane(polygon, plane);
    }
    if (polygon.size() < 3U) {
        polygon.clear();
    }
    return polygon;
}

bool makeScreenVertex(const RawVertex& input, int width, int height,
                      double inverseNear, double inverseFar,
                      ScreenVertex& output) {
    if (!(input.clip.w > kEpsilon) || !std::isfinite(input.clip.w)) {
        return false;
    }
    const double inverseW = 1.0 / input.clip.w;
    const double ndcX = input.clip.x * inverseW;
    const double ndcY = input.clip.y * inverseW;
    if (!std::isfinite(ndcX) || !std::isfinite(ndcY) ||
        !std::isfinite(inverseW)) {
        return false;
    }

    const double depth =
        (inverseW - inverseFar) / (inverseNear - inverseFar);
    output.x = (ndcX * 0.5 + 0.5) * static_cast<double>(width);
    output.y = (0.5 - ndcY * 0.5) * static_cast<double>(height);
    output.inverseW = inverseW;
    output.depth = std::clamp(depth, 0.0, 1.0);
    output.world = input.world;
    output.normal = input.normal;
    output.uv = input.uv;
    output.color = input.color;
    return std::isfinite(output.x) && std::isfinite(output.y);
}

double edgeFunction(const ScreenVertex& a, const ScreenVertex& b,
                    double pointX, double pointY) {
    return (b.x - a.x) * (pointY - a.y) -
           (b.y - a.y) * (pointX - a.x);
}

bool isTopLeftEdge(const ScreenVertex& a, const ScreenVertex& b) {
    const double deltaX = b.x - a.x;
    const double deltaY = b.y - a.y;
    return (deltaY == 0.0 && deltaX > 0.0) ||
           (deltaX == 0.0 && deltaY < 0.0);
}

Vec3 sampleLayer(const std::optional<Texture>& texture, const Vec3& fallback,
                 double weight, const Vec2& uv) {
    const Vec3 color = texture ? texture->sampleNearestRepeat(uv) : fallback;
    return color * weight;
}

std::uint8_t quantizeGamma(double value) {
    if (!std::isfinite(value) || value <= 0.0) {
        return 0U;
    }
    const double mapped = value / (1.0 + value);
    const double corrected = std::pow(mapped, 1.0 / 2.2);
    const double scaled =
        std::floor(std::clamp(corrected, 0.0, 1.0) * 255.0 + 0.5);
    return static_cast<std::uint8_t>(scaled);
}

std::pair<bool, Texture> loadTextureWithWarning(
    const Scene& scene, const std::string& path, const Material& material,
    const char* layerName, const WarningCallback& warning) {
    std::filesystem::path resolved(path);
    if (resolved.is_relative()) {
        resolved = scene.sourceDirectory / resolved;
    }

    Texture texture;
    std::string error;
    if (!loadBmpTexture(resolved, texture, error)) {
        if (warning) {
            warning("texture '" + resolved.string() + "' for material '" +
                    material.name + "' (" + layerName +
                    " layer) could not be loaded: " + error +
                    "; using material " + layerName + " color");
        }
        return {false, {}};
    }
    return {true, std::move(texture)};
}

}  // namespace

RenderImage renderScene(const Scene& scene, int width, int height,
                        const WarningCallback& warning,
                        RenderStats& stats) {
    stats = {};
    RenderImage image;
    image.width = width;
    image.height = height;
    image.bgrTopDown.assign(static_cast<std::size_t>(width) *
                                static_cast<std::size_t>(height) * 3U,
                            0U);
    std::vector<double> depthBuffer(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
        0.0);

    const double aspect =
        static_cast<double>(width) / static_cast<double>(height);
    const Mat4 view = lookAtLH(scene.camera.position, scene.camera.direction,
                               scene.camera.up);
    const Mat4 projection =
        perspectiveFovLH(scene.camera.fovYRadians, aspect,
                         scene.camera.nearPlane, scene.camera.farPlane);
    const Mat4 viewProjection = projection * view;
    const std::array<Plane, 6> frustumPlanes =
        extractFrustumPlanes(viewProjection);
    const double inverseNear = 1.0 / scene.camera.nearPlane;
    const double inverseFar = 1.0 / scene.camera.farPlane;

    std::vector<PreparedMaterial> preparedMaterials(scene.materials.size());
    for (std::size_t i = 0; i < scene.materials.size(); ++i) {
        const Material& material = scene.materials[i];
        PreparedMaterial prepared;
        prepared.material = &material;
        if (material.baseTexture) {
            auto [loaded, texture] = loadTextureWithWarning(
                scene, *material.baseTexture, material, "base", warning);
            if (loaded) {
                prepared.baseTexture = std::move(texture);
            }
        }
        if (material.detailTexture) {
            auto [loaded, texture] = loadTextureWithWarning(
                scene, *material.detailTexture, material, "detail", warning);
            if (loaded) {
                prepared.detailTexture = std::move(texture);
            }
        }
        preparedMaterials[i] = std::move(prepared);
    }

    const Vec3 ambient{0.1, 0.1, 0.1};

    for (const Mesh& mesh : scene.meshes) {
        stats.totalTriangles += mesh.indices.size() / 3U;

        std::vector<Vec3> worldPositions;
        worldPositions.reserve(mesh.vertices.size());
        Aabb worldBounds{
            {std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()},
            {-std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()},
        };
        bool validPositions = true;
        for (const Vertex& vertex : mesh.vertices) {
            const Vec4 transformed =
                transform(mesh.model,
                          Vec4{vertex.position.x, vertex.position.y,
                               vertex.position.z, 1.0});
            if (!std::isfinite(transformed.w) ||
                std::abs(transformed.w) <= kEpsilon) {
                validPositions = false;
                break;
            }
            const Vec3 world{transformed.x / transformed.w,
                             transformed.y / transformed.w,
                             transformed.z / transformed.w};
            if (!isFinite(world)) {
                validPositions = false;
                break;
            }
            worldPositions.push_back(world);
            worldBounds.minimum = minComponents(worldBounds.minimum, world);
            worldBounds.maximum = maxComponents(worldBounds.maximum, world);
        }

        if (!validPositions ||
            aabbOutsideFrustum(worldBounds, frustumPlanes)) {
            ++stats.culledMeshes;
            continue;
        }

        const PreparedMaterial& material =
            preparedMaterials[mesh.materialIndex];
        std::vector<Vec3> worldNormals;
        worldNormals.reserve(mesh.vertices.size());
        for (const Vertex& vertex : mesh.vertices) {
            worldNormals.push_back(
                normalize(transformVector(mesh.normalMatrix, vertex.normal)));
        }

        for (std::size_t triangleIndex = 0;
             triangleIndex < mesh.indices.size(); triangleIndex += 3U) {
            RawVertex triangle[3];
            bool validTriangle = true;
            for (std::size_t corner = 0; corner < 3U; ++corner) {
                const std::uint32_t vertexIndex =
                    mesh.indices[triangleIndex + corner];
                const Vertex& vertex = mesh.vertices[vertexIndex];
                const Vec3& world = worldPositions[vertexIndex];
                const Vec4 clip =
                    transform(viewProjection, Vec4{world.x, world.y, world.z,
                                                   1.0});
                triangle[corner] = RawVertex{clip,
                                             world,
                                             worldNormals[vertexIndex],
                                             vertex.uv,
                                             vertex.color};
                if (!isFinite(clip) || !isFinite(world)) {
                    validTriangle = false;
                }
            }
            if (!validTriangle) {
                continue;
            }

            const std::vector<RawVertex> clipped = clipTriangle(triangle);
            if (clipped.size() < 3U) {
                continue;
            }

            for (std::size_t fan = 1; fan + 1 < clipped.size(); ++fan) {
                ScreenVertex screen[3];
                if (!makeScreenVertex(clipped[0], width, height, inverseNear,
                                      inverseFar, screen[0]) ||
                    !makeScreenVertex(clipped[fan], width, height, inverseNear,
                                      inverseFar, screen[1]) ||
                    !makeScreenVertex(clipped[fan + 1], width, height,
                                      inverseNear, inverseFar, screen[2])) {
                    continue;
                }
                ++stats.rasterizedTriangles;

                double area = edgeFunction(screen[0], screen[1], screen[2].x,
                                           screen[2].y);
                if (std::abs(area) <= 1.0e-12) {
                    continue;
                }
                if (area < 0.0) {
                    std::swap(screen[1], screen[2]);
                    area = -area;
                }

                const double minimumX =
                    std::min({screen[0].x, screen[1].x, screen[2].x});
                const double maximumX =
                    std::max({screen[0].x, screen[1].x, screen[2].x});
                const double minimumY =
                    std::min({screen[0].y, screen[1].y, screen[2].y});
                const double maximumY =
                    std::max({screen[0].y, screen[1].y, screen[2].y});
                const int startX = std::max(
                    0, static_cast<int>(std::floor(minimumX)));
                const int endX = std::min(
                    width - 1, static_cast<int>(std::ceil(maximumX)));
                const int startY = std::max(
                    0, static_cast<int>(std::floor(minimumY)));
                const int endY = std::min(
                    height - 1, static_cast<int>(std::ceil(maximumY)));

                for (int y = startY; y <= endY; ++y) {
                    for (int x = startX; x <= endX; ++x) {
                        const double sampleX =
                            static_cast<double>(x) + 0.5;
                        const double sampleY =
                            static_cast<double>(y) + 0.5;
                        const double w0 = edgeFunction(
                            screen[1], screen[2], sampleX, sampleY);
                        const double w1 = edgeFunction(
                            screen[2], screen[0], sampleX, sampleY);
                        const double w2 = edgeFunction(
                            screen[0], screen[1], sampleX, sampleY);
                        if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) {
                            continue;
                        }
                        if (w0 == 0.0 && !isTopLeftEdge(screen[1], screen[2])) {
                            continue;
                        }
                        if (w1 == 0.0 && !isTopLeftEdge(screen[2], screen[0])) {
                            continue;
                        }
                        if (w2 == 0.0 && !isTopLeftEdge(screen[0], screen[1])) {
                            continue;
                        }

                        ++stats.totalFragments;
                        const double b0 = w0 / area;
                        const double b1 = w1 / area;
                        const double b2 = w2 / area;
                        const double fragmentDepth =
                            b0 * screen[0].depth + b1 * screen[1].depth +
                            b2 * screen[2].depth;
                        const std::size_t pixelIndex =
                            static_cast<std::size_t>(y) * width +
                            static_cast<std::size_t>(x);
                        if (!(fragmentDepth > depthBuffer[pixelIndex])) {
                            ++stats.depthTestFailures;
                            continue;
                        }

                        const double denominator =
                            b0 * screen[0].inverseW +
                            b1 * screen[1].inverseW +
                            b2 * screen[2].inverseW;
                        if (!(denominator > kEpsilon)) {
                            continue;
                        }
                        const double p0 =
                            b0 * screen[0].inverseW / denominator;
                        const double p1 =
                            b1 * screen[1].inverseW / denominator;
                        const double p2 =
                            b2 * screen[2].inverseW / denominator;

                        const Vec3 world = screen[0].world * p0 +
                                           screen[1].world * p1 +
                                           screen[2].world * p2;
                        if (!isFinite(world)) {
                            continue;
                        }
                        Vec3 normal = screen[0].normal * p0 +
                                      screen[1].normal * p1 +
                                      screen[2].normal * p2;
                        normal = normalize(normal);
                        if (lengthSquared(normal) <= kEpsilon) {
                            normal = normalize(cross(
                                screen[1].world - screen[0].world,
                                screen[2].world - screen[0].world));
                        }
                        const Vec2 uv{screen[0].uv.x * p0 +
                                          screen[1].uv.x * p1 +
                                          screen[2].uv.x * p2,
                                      screen[0].uv.y * p0 +
                                          screen[1].uv.y * p1 +
                                          screen[2].uv.y * p2};
                        const Vec4 vertexColor = clamp01(
                            Vec4{screen[0].color.x * p0 +
                                     screen[1].color.x * p1 +
                                     screen[2].color.x * p2,
                                 screen[0].color.y * p0 +
                                     screen[1].color.y * p1 +
                                     screen[2].color.y * p2,
                                 screen[0].color.z * p0 +
                                     screen[1].color.z * p1 +
                                     screen[2].color.z * p2,
                                 screen[0].color.w * p0 +
                                     screen[1].color.w * p1 +
                                     screen[2].color.w * p2});

                        const Material& materialData = *material.material;
                        const Vec3 base = sampleLayer(
                            material.baseTexture, materialData.baseColor,
                            materialData.baseWeight, uv);
                        const Vec3 detail = sampleLayer(
                            material.detailTexture, materialData.detailColor,
                            materialData.detailWeight, uv);
                        const double layerMix = vertexColor.w;
                        const Vec3 layered =
                            base * (1.0 - layerMix) + detail * layerMix;
                        const Vec3 albedo =
                            materialData.diffuse * layered *
                            Vec3{vertexColor.x, vertexColor.y, vertexColor.z};

                        Vec3 lightSum{};
                        for (const DirectionalLight& light : scene.lights) {
                            const Vec3 toLight = normalize(-light.direction);
                            const double nDotL =
                                std::max(0.0, dot(normal, toLight));
                            lightSum +=
                                light.color * (light.intensity * nDotL);
                        }
                        lightSum = clamp01(lightSum);
                        const Vec3 hdr = albedo * (lightSum + ambient);

                        const std::uint8_t blue = quantizeGamma(hdr.z);
                        const std::uint8_t green = quantizeGamma(hdr.y);
                        const std::uint8_t red = quantizeGamma(hdr.x);
                        const std::size_t outputIndex = pixelIndex * 3U;
                        image.bgrTopDown[outputIndex] = blue;
                        image.bgrTopDown[outputIndex + 1U] = green;
                        image.bgrTopDown[outputIndex + 2U] = red;
                        depthBuffer[pixelIndex] = fragmentDepth;
                    }
                }
            }
        }
    }

    return image;
}

}  // namespace raster
