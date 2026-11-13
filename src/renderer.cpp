#include "renderer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct TileRange {
    bool valid = false;
    int minTx = std::numeric_limits<int>::max();
    int minTy = std::numeric_limits<int>::max();
    int maxTx = -1;
    int maxTy = -1;

    void include(int tx, int ty) {
        valid = true;
        minTx = std::min(minTx, tx);
        minTy = std::min(minTy, ty);
        maxTx = std::max(maxTx, tx);
        maxTy = std::max(maxTy, ty);
    }

    bool intersectsTile(int tx, int ty) const {
        return valid && tx >= minTx && tx <= maxTx && ty >= minTy && ty <= maxTy;
    }

    void merge(const TileRange& other) {
        if (!other.valid) return;
        include(other.minTx, other.minTy);
        include(other.maxTx, other.maxTy);
    }
};

struct VolumeTriangle {
    Vec3 p[3];
};

struct RockCache {
    std::vector<std::vector<VolumeTriangle>> volumes;
    std::array<TileRange, 3> volumeTileRanges;
    TileRange meshTileRange;
    TileRange totalTileRange;
};

struct ClipVertex {
    Vec4 clip;
    Vec3 world;
    Vec3 normal;
};

struct ScreenVertex {
    double x = 0.0;
    double y = 0.0;
    double zNdc = 0.0;
    double invW = 0.0;
    Vec3 world;
    Vec3 normal;
};

struct GBufferTile {
    int x0 = 0;
    int y0 = 0;
    int width = 0;
    int height = 0;
    std::vector<uint32_t> depth;
    std::vector<uint32_t> primitive;
    std::vector<Vec3> world;
    std::vector<Vec3> normal;
    std::vector<int> object;
};

double cross2(double ax, double ay, double bx, double by) {
    return ax * by - ay * bx;
}

uint32_t quantizeDepth(double zNdc, uint32_t maxDepth) {
    const double t = clamp01(zNdc * 0.5 + 0.5);
    const long long value = static_cast<long long>(
        std::llround(t * static_cast<double>(maxDepth)));
    return static_cast<uint32_t>(std::max<long long>(
        0, std::min<long long>(value, static_cast<long long>(maxDepth))));
}

TileRange fullTileRange(const Scene& scene) {
    TileRange range;
    const int tilesX = (scene.width + scene.tileSize - 1) / scene.tileSize;
    const int tilesY = (scene.height + scene.tileSize - 1) / scene.tileSize;
    range.include(0, 0);
    range.include(tilesX - 1, tilesY - 1);
    return range;
}

bool projectToScreen(const Scene& scene, const Mat4& viewProjection, const Vec3& p,
                     double& screenX, double& screenY, bool& behindCamera) {
    const Vec4 clip = viewProjection * Vec4(p.x, p.y, p.z, 1.0);
    if (!std::isfinite(clip.x) || !std::isfinite(clip.y) ||
        !std::isfinite(clip.z) || !std::isfinite(clip.w) || clip.w <= 1e-9) {
        behindCamera = true;
        return false;
    }
    const double ndcX = clip.x / clip.w;
    const double ndcY = clip.y / clip.w;
    screenX = (ndcX * 0.5 + 0.5) * static_cast<double>(scene.width);
    screenY = (0.5 - ndcY * 0.5) * static_cast<double>(scene.height);
    return std::isfinite(screenX) && std::isfinite(screenY);
}

TileRange tileRangeFromScreenBounds(const Scene& scene, double minX, double minY,
                                    double maxX, double maxY) {
    TileRange range;
    if (!std::isfinite(minX) || !std::isfinite(minY) ||
        !std::isfinite(maxX) || !std::isfinite(maxY)) {
        return fullTileRange(scene);
    }
    const double margin = 2.0;
    minX -= margin;
    maxX += margin;
    minY -= margin;
    maxY += margin;
    if (maxX < 0.0 || maxY < 0.0 ||
        minX >= static_cast<double>(scene.width) ||
        minY >= static_cast<double>(scene.height)) {
        return range;
    }
    const int tilesX = (scene.width + scene.tileSize - 1) / scene.tileSize;
    const int tilesY = (scene.height + scene.tileSize - 1) / scene.tileSize;
    const int tx0 = std::max(0, static_cast<int>(std::floor(minX / scene.tileSize)));
    const int ty0 = std::max(0, static_cast<int>(std::floor(minY / scene.tileSize)));
    const int tx1 = std::min(tilesX - 1,
        static_cast<int>(std::floor(maxX / scene.tileSize)));
    const int ty1 = std::min(tilesY - 1,
        static_cast<int>(std::floor(maxY / scene.tileSize)));
    if (tx0 <= tx1 && ty0 <= ty1) {
        range.include(tx0, ty0);
        range.include(tx1, ty1);
    }
    return range;
}

TileRange projectWorldBoundsToTiles(const Scene& scene, const Mat4& viewProjection,
                                    const std::vector<Vec3>& points) {
    if (points.empty()) return {};
    bool behind = false;
    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxX = -std::numeric_limits<double>::max();
    double maxY = -std::numeric_limits<double>::max();
    for (const Vec3& p : points) {
        double sx = 0.0, sy = 0.0;
        if (!projectToScreen(scene, viewProjection, p, sx, sy, behind)) {
            if (behind) return fullTileRange(scene);
            continue;
        }
        minX = std::min(minX, sx);
        minY = std::min(minY, sy);
        maxX = std::max(maxX, sx);
        maxY = std::max(maxY, sy);
    }
    if (minX > maxX) return {};
    return tileRangeFromScreenBounds(scene, minX, minY, maxX, maxY);
}

void addPrism(std::vector<VolumeTriangle>& out,
              const Vec3& p0, const Vec3& p1, const Vec3& p2,
              const Vec3& d0, const Vec3& d1, const Vec3& d2) {
    const Vec3 q0 = p0 + d0;
    const Vec3 q1 = p1 + d1;
    const Vec3 q2 = p2 + d2;
    const Vec3 n = cross(p1 - p0, p2 - p0);
    const double side = dot(n, d0 + d1 + d2);
    if (side >= 0.0) {
        out.push_back({{p0, p2, p1}});
        out.push_back({{q0, q1, q2}});
        out.push_back({{p0, p1, q1}});
        out.push_back({{p0, q1, q0}});
        out.push_back({{p1, p2, q2}});
        out.push_back({{p1, q2, q1}});
        out.push_back({{p2, p0, q0}});
        out.push_back({{p2, q0, q2}});
    } else {
        out.push_back({{p0, p1, p2}});
        out.push_back({{q0, q2, q1}});
        out.push_back({{p0, q0, q1}});
        out.push_back({{p0, q1, p1}});
        out.push_back({{p1, q1, q2}});
        out.push_back({{p1, q2, p2}});
        out.push_back({{p2, q2, q0}});
        out.push_back({{p2, q0, p0}});
    }
}

void collectVolumePoints(const std::vector<VolumeTriangle>& volumes,
                         std::vector<Vec3>& points) {
    points.reserve(points.size() + volumes.size() * 3);
    for (const VolumeTriangle& tri : volumes) {
        points.push_back(tri.p[0]);
        points.push_back(tri.p[1]);
        points.push_back(tri.p[2]);
    }
}

RockCache buildRockCache(const Scene& scene, int rockIndex) {
    const RockNode& rock = scene.rocks[static_cast<size_t>(rockIndex)];
    const Mat4 viewProjection = scene.viewProjectionMatrix();
    RockCache cache;
    cache.volumes.resize(scene.lights.size());
    cache.volumeTileRanges = {};

    std::vector<Vec3> meshPoints;
    meshPoints.reserve(rock.worldTriangles.size() * 3);
    for (const RockTriangle& tri : rock.worldTriangles) {
        meshPoints.push_back(tri.p[0]);
        meshPoints.push_back(tri.p[1]);
        meshPoints.push_back(tri.p[2]);
    }
    cache.meshTileRange = projectWorldBoundsToTiles(scene, viewProjection, meshPoints);

    for (size_t lightIndex = 0; lightIndex < scene.lights.size(); ++lightIndex) {
        const Light& light = scene.lights[lightIndex];
        std::vector<VolumeTriangle>& volumes = cache.volumes[lightIndex];
        volumes.reserve(rock.worldTriangles.size() * 8);
        for (const RockTriangle& tri : rock.worldTriangles) {
            Vec3 d0, d1, d2;
            if (light.type == LightType::Directional) {
                const Vec3 travel = normalized(light.direction) * scene.shadowExtrusion;
                d0 = d1 = d2 = travel;
            } else {
                const Vec3 r0 = tri.p[0] - light.position;
                const Vec3 r1 = tri.p[1] - light.position;
                const Vec3 r2 = tri.p[2] - light.position;
                if (lengthSquared(r0) < 1e-12 || lengthSquared(r1) < 1e-12 ||
                    lengthSquared(r2) < 1e-12) {
                    throw RenderError("E_LIGHT_INVALID",
                        "point light lies exactly on a caster vertex (rock '" + rock.id + "')");
                }
                d0 = normalized(r0) * scene.shadowExtrusion;
                d1 = normalized(r1) * scene.shadowExtrusion;
                d2 = normalized(r2) * scene.shadowExtrusion;
            }
            addPrism(volumes, tri.p[0], tri.p[1], tri.p[2], d0, d1, d2);
        }
        std::vector<Vec3> volumePoints;
        collectVolumePoints(volumes, volumePoints);
        cache.volumeTileRanges[lightIndex] =
            projectWorldBoundsToTiles(scene, viewProjection, volumePoints);
    }

    cache.totalTileRange = cache.meshTileRange;
    for (const TileRange& range : cache.volumeTileRanges) {
        cache.totalTileRange.merge(range);
    }
    return cache;
}

std::vector<RockCache> buildAllCaches(const Scene& scene,
                                      Clock::time_point& buildStart) {
    buildStart = Clock::now();
    std::vector<RockCache> caches(scene.rocks.size());
    for (int i = 0; i < static_cast<int>(scene.rocks.size()); ++i) {
        caches[static_cast<size_t>(i)] = buildRockCache(scene, i);
    }
    return caches;
}

double planeDistance(const ClipVertex& vertex, int plane) {
    const Vec4& c = vertex.clip;
    switch (plane) {
        case 0: return c.z + c.w;     // near
        case 1: return c.w - c.z;     // far
        case 2: return c.x + c.w;     // left
        case 3: return c.w - c.x;     // right
        case 4: return c.y + c.w;     // bottom
        case 5: return c.w - c.y;     // top
        default: return 0.0;
    }
}

ClipVertex lerpClip(const ClipVertex& a, const ClipVertex& b, double t) {
    ClipVertex result;
    result.clip.x = a.clip.x + (b.clip.x - a.clip.x) * t;
    result.clip.y = a.clip.y + (b.clip.y - a.clip.y) * t;
    result.clip.z = a.clip.z + (b.clip.z - a.clip.z) * t;
    result.clip.w = a.clip.w + (b.clip.w - a.clip.w) * t;
    result.world = a.world + (b.world - a.world) * t;
    result.normal = a.normal + (b.normal - a.normal) * t;
    return result;
}

bool clipTriangle(const ClipVertex input[3], std::vector<ClipVertex>& output) {
    std::vector<ClipVertex> polygon(input, input + 3);
    for (int plane = 0; plane < 6; ++plane) {
        if (polygon.empty()) return false;
        std::vector<ClipVertex> next;
        next.reserve(polygon.size() + 1);
        for (size_t i = 0; i < polygon.size(); ++i) {
            const ClipVertex& current = polygon[i];
            const ClipVertex& following = polygon[(i + 1) % polygon.size()];
            const double dc = planeDistance(current, plane);
            const double dn = planeDistance(following, plane);
            const bool currentInside = dc >= 0.0;
            const bool followingInside = dn >= 0.0;
            if (currentInside) next.push_back(current);
            if (currentInside != followingInside) {
                const double denominator = dc - dn;
                const double t = std::abs(denominator) > 1e-15 ? dc / denominator : 0.0;
                next.push_back(lerpClip(current, following,
                                        std::max(0.0, std::min(1.0, t))));
            }
        }
        polygon.swap(next);
    }
    output = std::move(polygon);
    return output.size() >= 3;
}

bool toScreen(const Scene& scene, const ClipVertex& input, ScreenVertex& output) {
    if (input.clip.w <= 1e-12 || !std::isfinite(input.clip.w)) return false;
    const double invW = 1.0 / input.clip.w;
    const double ndcX = input.clip.x * invW;
    const double ndcY = input.clip.y * invW;
    output.x = (ndcX * 0.5 + 0.5) * static_cast<double>(scene.width);
    output.y = (0.5 - ndcY * 0.5) * static_cast<double>(scene.height);
    output.zNdc = clamp01(input.clip.z * invW * 0.5 + 0.5) * 2.0 - 1.0;
    output.invW = invW;
    output.world = input.world;
    output.normal = input.normal;
    return std::isfinite(output.x) && std::isfinite(output.y) &&
           std::isfinite(output.zNdc) && std::isfinite(output.invW);
}

double screenArea(const ScreenVertex& a, const ScreenVertex& b, const ScreenVertex& c) {
    return cross2(b.x - a.x, b.y - a.y, c.x - a.x, c.y - a.y);
}

void drawGeometryTriangle(const Scene& scene, const Mat4& viewProjection,
                          GBufferTile& tile, const Vec3 trianglePositions[3],
                          const Vec3 triangleNormals[3], uint32_t primitiveId,
                          int objectIndex, uint32_t maxDepth,
                          RenderStats& stats, double& workMs) {
    const auto start = Clock::now();
    (void)stats;
    ClipVertex input[3];
    for (int i = 0; i < 3; ++i) {
        input[i].world = trianglePositions[i];
        input[i].normal = triangleNormals[i];
        input[i].clip = viewProjection * Vec4(trianglePositions[i].x,
                                              trianglePositions[i].y,
                                              trianglePositions[i].z, 1.0);
    }
    std::vector<ClipVertex> polygon;
    if (!clipTriangle(input, polygon)) return;
    for (size_t i = 1; i + 1 < polygon.size(); ++i) {
        ScreenVertex s0, s1, s2;
        if (!toScreen(scene, polygon[0], s0) ||
            !toScreen(scene, polygon[i], s1) ||
            !toScreen(scene, polygon[i + 1], s2)) {
            continue;
        }
        const double area = screenArea(s0, s1, s2);
        if (!(area < -1e-10)) continue; // front faces become clockwise after the y flip

        const double minX = std::min({s0.x, s1.x, s2.x});
        const double maxX = std::max({s0.x, s1.x, s2.x});
        const double minY = std::min({s0.y, s1.y, s2.y});
        const double maxY = std::max({s0.y, s1.y, s2.y});
        int xBegin = std::max(tile.x0, static_cast<int>(std::floor(minX)));
        int xEnd = std::min(tile.x0 + tile.width - 1,
                            static_cast<int>(std::ceil(maxX)));
        int yBegin = std::max(tile.y0, static_cast<int>(std::floor(minY)));
        int yEnd = std::min(tile.y0 + tile.height - 1,
                            static_cast<int>(std::ceil(maxY)));
        if (xBegin > xEnd || yBegin > yEnd) continue;
        const double invArea = 1.0 / area;

        for (int y = yBegin; y <= yEnd; ++y) {
            for (int x = xBegin; x <= xEnd; ++x) {
                const double px = static_cast<double>(x) + 0.5;
                const double py = static_cast<double>(y) + 0.5;
                const double w0 = cross2(s2.x - s1.x, s2.y - s1.y,
                                          px - s1.x, py - s1.y) * invArea;
                const double w1 = cross2(s0.x - s2.x, s0.y - s2.y,
                                          px - s2.x, py - s2.y) * invArea;
                const double w2 = 1.0 - w0 - w1;
                if (w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9) continue;

                const double zNdc = w0 * s0.zNdc + w1 * s1.zNdc + w2 * s2.zNdc;
                const uint32_t depth = quantizeDepth(zNdc, maxDepth);
                const int localX = x - tile.x0;
                const int localY = y - tile.y0;
                const size_t index = static_cast<size_t>(localY) * tile.width + localX;
                if (depth > tile.depth[index]) continue;
                if (depth == tile.depth[index] && primitiveId >= tile.primitive[index]) continue;

                const double denominator = w0 * s0.invW + w1 * s1.invW + w2 * s2.invW;
                if (!(std::abs(denominator) > 1e-15)) continue;
                const Vec3 world = (s0.world * (w0 * s0.invW) +
                                    s1.world * (w1 * s1.invW) +
                                    s2.world * (w2 * s2.invW)) / denominator;
                Vec3 normal = normalized(
                    s0.normal * (w0 * s0.invW) +
                    s1.normal * (w1 * s1.invW) +
                    s2.normal * (w2 * s2.invW));
                if (lengthSquared(normal) < 0.5) normal = {0.0, 1.0, 0.0};

                tile.depth[index] = depth;
                tile.primitive[index] = primitiveId;
                tile.world[index] = world;
                tile.normal[index] = normal;
                tile.object[index] = objectIndex;
            }
        }
    }
    workMs += elapsedMs(start, Clock::now());
}

void drawShadowTriangle(const Scene& scene, const Mat4& viewProjection,
                        GBufferTile& tile, const VolumeTriangle& triangle,
                        const std::vector<uint32_t>& sceneDepth, uint32_t maxDepth,
                        uint32_t depthBias, int stencilMax,
                        std::vector<int>& passCount, std::vector<int>& failCount,
                        RenderStats& stats, bool& overflow, double& workMs) {
    const auto start = Clock::now();
    ClipVertex input[3];
    for (int i = 0; i < 3; ++i) {
        input[i].world = triangle.p[i];
        input[i].normal = {};
        input[i].clip = viewProjection * Vec4(triangle.p[i].x,
                                              triangle.p[i].y,
                                              triangle.p[i].z, 1.0);
    }
    std::vector<ClipVertex> polygon;
    if (!clipTriangle(input, polygon)) return;
    for (size_t i = 1; i + 1 < polygon.size(); ++i) {
        ScreenVertex s0, s1, s2;
        if (!toScreen(scene, polygon[0], s0) ||
            !toScreen(scene, polygon[i], s1) ||
            !toScreen(scene, polygon[i + 1], s2)) {
            continue;
        }
        const double area = screenArea(s0, s1, s2);
        if (std::abs(area) <= 1e-10) continue;
        const bool frontFacing = area < 0.0;
        const double invArea = 1.0 / area;

        const double minX = std::min({s0.x, s1.x, s2.x});
        const double maxX = std::max({s0.x, s1.x, s2.x});
        const double minY = std::min({s0.y, s1.y, s2.y});
        const double maxY = std::max({s0.y, s1.y, s2.y});
        int xBegin = std::max(tile.x0, static_cast<int>(std::floor(minX)));
        int xEnd = std::min(tile.x0 + tile.width - 1,
                            static_cast<int>(std::ceil(maxX)));
        int yBegin = std::max(tile.y0, static_cast<int>(std::floor(minY)));
        int yEnd = std::min(tile.y0 + tile.height - 1,
                            static_cast<int>(std::ceil(maxY)));
        if (xBegin > xEnd || yBegin > yEnd) continue;

        for (int y = yBegin; y <= yEnd; ++y) {
            for (int x = xBegin; x <= xEnd; ++x) {
                const double px = static_cast<double>(x) + 0.5;
                const double py = static_cast<double>(y) + 0.5;
                const double w0 = cross2(s2.x - s1.x, s2.y - s1.y,
                                          px - s1.x, py - s1.y) * invArea;
                const double w1 = cross2(s0.x - s2.x, s0.y - s2.y,
                                          px - s2.x, py - s2.y) * invArea;
                const double w2 = 1.0 - w0 - w1;
                if (w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9) continue;
                const int localX = x - tile.x0;
                const int localY = y - tile.y0;
                const size_t index = static_cast<size_t>(localY) * tile.width + localX;
                const uint32_t receiverDepth = sceneDepth[index];
                if (receiverDepth == maxDepth) continue; // volume boundary outside geometry stays lit

                const double zNdc = w0 * s0.zNdc + w1 * s1.zNdc + w2 * s2.zNdc;
                const uint32_t volumeDepth = quantizeDepth(zNdc, maxDepth);
                const bool depthPass =
                    static_cast<uint64_t>(volumeDepth) <=
                    static_cast<uint64_t>(receiverDepth) + static_cast<uint64_t>(depthBias);
                std::vector<int>& counter = depthPass ? passCount : failCount;
                const int delta = depthPass
                    ? (frontFacing ? 1 : -1)
                    : (frontFacing ? -1 : 1);
                const int next = counter[index] + delta;
                if (next > stencilMax || next < -stencilMax) {
                    overflow = true;
                    ++stats.stencilOverflowCount;
                    workMs += elapsedMs(start, Clock::now());
                    return;
                }
                counter[index] = next;
                stats.maxTemplatePassCountAbs = std::max(stats.maxTemplatePassCountAbs,
                                                         std::abs(passCount[index]));
                stats.maxTemplateFailCountAbs = std::max(stats.maxTemplateFailCountAbs,
                                                         std::abs(failCount[index]));
                if (depthPass) ++stats.stencilDepthPassTests;
                else ++stats.stencilDepthFailTests;
            }
        }
    }
    workMs += elapsedMs(start, Clock::now());
}

void shadeTile(const Scene& scene, const GBufferTile& tile,
               const std::vector<std::vector<uint8_t>>& shadowMasks,
               Framebuffer& frame, uint32_t maxDepth, double& workMs) {
    const auto start = Clock::now();
    for (int localY = 0; localY < tile.height; ++localY) {
        for (int localX = 0; localX < tile.width; ++localX) {
            const int x = tile.x0 + localX;
            const int y = tile.y0 + localY;
            const size_t tileIndex = static_cast<size_t>(localY) * tile.width + localX;
            const size_t frameIndex = static_cast<size_t>(y) * frame.width + x;
            Vec3 color;
            if (tile.depth[tileIndex] == maxDepth) {
                const double t = static_cast<double>(y) / static_cast<double>(std::max(1, frame.height - 1));
                const Vec3 horizon{0.60, 0.55, 0.47};
                const Vec3 zenith{0.20, 0.34, 0.53};
                color = horizon * (1.0 - t) + zenith * t;
            } else {
                const Vec3 normal = tile.normal[tileIndex];
                const Vec3 world = tile.world[tileIndex];
                Vec3 surface;
                if (tile.object[tileIndex] == -1) {
                    const MaterialLayer* layer = &scene.layers.front();
                    for (const MaterialLayer& candidate : scene.layers) {
                        if (world.y >= candidate.heightStart && world.y < candidate.heightEnd) {
                            layer = &candidate;
                            break;
                        }
                        if (world.y >= candidate.heightEnd) layer = &candidate;
                    }
                    const double slope = clamp01(1.0 - normal.y);
                    surface = layer->color * (1.0 - 0.48 * slope) +
                              layer->slopeTint * (0.48 * slope);
                    const double band = 0.91 +
                        0.09 * (0.5 + 0.5 * std::sin(world.y * 4.2 + 0.8 * std::sin(world.x * 0.19)));
                    surface = surface * band;
                } else {
                    const RockMaterial& material =
                        scene.rocks[static_cast<size_t>(tile.object[tileIndex])].material;
                    surface = material.baseColor * (0.91 + 0.09 * clamp01(normal.y));
                }

                Vec3 radiance = scene.ambient;
                for (size_t lightIndex = 0; lightIndex < scene.lights.size(); ++lightIndex) {
                    if (shadowMasks[lightIndex][tileIndex]) continue;
                    const Light& light = scene.lights[lightIndex];
                    Vec3 toLight;
                    double attenuation = 1.0;
                    if (light.type == LightType::Directional) {
                        toLight = normalized(light.direction * -1.0);
                    } else {
                        const Vec3 delta = light.position - world;
                        const double distance = length(delta);
                        if (!(distance > 1e-12)) continue;
                        toLight = delta / distance;
                        const double ratio = distance / light.range;
                        attenuation = ratio >= 1.0 ? 0.0 : (1.0 - ratio) * (1.0 - ratio);
                    }
                    const double ndotl = std::max(0.0, dot(normal, toLight));
                    radiance += light.color * (light.intensity * attenuation * ndotl);
                }

                const Vec3 linear{surface.x * radiance.x,
                                  surface.y * radiance.y,
                                  surface.z * radiance.z};
                auto tone = [](double value) {
                    return std::pow(clamp01(1.0 - std::exp(-std::max(0.0, value) * 1.22)), 1.0 / 2.2);
                };
                color = {tone(linear.x), tone(linear.y), tone(linear.z)};
            }

            if (tile.depth[tileIndex] == maxDepth) {
                color = {std::pow(clamp01(color.x), 1.0 / 2.2),
                         std::pow(clamp01(color.y), 1.0 / 2.2),
                         std::pow(clamp01(color.z), 1.0 / 2.2)};
            }
            frame.color[frameIndex * 3 + 0] =
                static_cast<uint8_t>(std::llround(clamp01(color.x) * 255.0));
            frame.color[frameIndex * 3 + 1] =
                static_cast<uint8_t>(std::llround(clamp01(color.y) * 255.0));
            frame.color[frameIndex * 3 + 2] =
                static_cast<uint8_t>(std::llround(clamp01(color.z) * 255.0));
        }
    }
    workMs += elapsedMs(start, Clock::now());
}

bool renderTile(Scene& scene, Framebuffer& frame, const std::vector<RockCache>& caches,
                int tileIndex, RenderStats& stats) {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    frame.tileBounds(tileIndex, x0, y0, x1, y1);
    GBufferTile tile;
    tile.x0 = x0;
    tile.y0 = y0;
    tile.width = x1 - x0;
    tile.height = y1 - y0;
    const size_t pixels = static_cast<size_t>(tile.width) * tile.height;
    const uint32_t maxDepth = scene.depthMax();
    tile.depth.assign(pixels, maxDepth);
    tile.primitive.assign(pixels, std::numeric_limits<uint32_t>::max());
    tile.world.assign(pixels, Vec3{});
    tile.normal.assign(pixels, Vec3{0.0, 1.0, 0.0});
    tile.object.assign(pixels, -2);

    const Mat4 viewProjection = scene.viewProjectionMatrix();
    double rasterMs = 0.0;
    for (const TerrainTriangle& tri : scene.terrainTriangles) {
        drawGeometryTriangle(scene, viewProjection, tile, tri.p, tri.n,
                             tri.primitiveId, -1, maxDepth, stats, rasterMs);
    }
    for (size_t rockIndex = 0; rockIndex < scene.rocks.size(); ++rockIndex) {
        for (const RockTriangle& tri : scene.rocks[rockIndex].worldTriangles) {
            const Vec3 normals[3] = {tri.normal, tri.normal, tri.normal};
            drawGeometryTriangle(scene, viewProjection, tile, tri.p, normals,
                                 tri.primitiveId, static_cast<int>(rockIndex), maxDepth,
                                 stats, rasterMs);
        }
    }
    stats.rasterizationMs += rasterMs;

    const int stencilMax = (1 << scene.stencilBits) - 1;
    std::vector<std::vector<uint8_t>> shadowMasks(
        scene.lights.size(), std::vector<uint8_t>(pixels, 0U));
    std::vector<int> passCount(pixels, 0);
    std::vector<int> failCount(pixels, 0);
    double shadowMs = 0.0;
    bool overflow = false;

    for (size_t lightIndex = 0; lightIndex < scene.lights.size(); ++lightIndex) {
        std::fill(passCount.begin(), passCount.end(), 0);
        std::fill(failCount.begin(), failCount.end(), 0);
        for (size_t rockIndex = 0; rockIndex < scene.rocks.size(); ++rockIndex) {
            const RockCache& cache = caches[rockIndex];
            if (rockIndex >= cache.volumeTileRanges.size()) continue;
            // All three lights are fixed by validation. This keeps cache layout explicit.
            if (!cache.volumeTileRanges[lightIndex].intersectsTile(
                    tileIndex % frame.tilesX, tileIndex / frame.tilesX)) {
                continue;
            }
            for (const VolumeTriangle& volumeTriangle : cache.volumes[lightIndex]) {
                ++stats.shadowVolumeTriangles;
                drawShadowTriangle(scene, viewProjection, tile, volumeTriangle,
                                   tile.depth, maxDepth, scene.shadowDepthBiasUnits,
                                   stencilMax, passCount, failCount, stats, overflow,
                                   shadowMs);
                if (overflow) break;
            }
            if (overflow) break;
        }
        if (overflow) break;
        for (size_t i = 0; i < pixels; ++i) {
            shadowMasks[lightIndex][i] = (passCount[i] != 0 || failCount[i] != 0) ? 1U : 0U;
        }
    }
    stats.shadowMs += shadowMs;
    if (overflow) return false;

    for (size_t i = 0; i < pixels; ++i) {
        if (tile.depth[i] == maxDepth) {
            for (const auto& mask : shadowMasks) {
                if (mask[i] != 0U) {
                    ++stats.shadowMaskedBackgroundPixels;
                    break;
                }
            }
        }
    }
    double shadeMs = 0.0;
    shadeTile(scene, tile, shadowMasks, frame, maxDepth, shadeMs);
    stats.shadingMs += shadeMs;
    for (int localY = 0; localY < tile.height; ++localY) {
        const size_t frameRow = static_cast<size_t>((tile.y0 + localY) * frame.width + tile.x0);
        const size_t tileRow = static_cast<size_t>(localY) * tile.width;
        std::copy(tile.depth.begin() + static_cast<std::ptrdiff_t>(tileRow),
                  tile.depth.begin() + static_cast<std::ptrdiff_t>(tileRow + tile.width),
                  frame.depth.begin() + static_cast<std::ptrdiff_t>(frameRow));
    }
    ++stats.renderedTileCount;
    return true;
}

bool renderTiles(Scene& scene, Framebuffer& frame, const std::vector<RockCache>& caches,
                 const std::vector<int>& tiles, RenderStats& stats) {
    for (const int tileIndex : tiles) {
        if (!renderTile(scene, frame, caches, tileIndex, stats)) return false;
    }
    return true;
}

std::vector<int> allTiles(const Scene& scene) {
    const int tilesX = (scene.width + scene.tileSize - 1) / scene.tileSize;
    const int tilesY = (scene.height + scene.tileSize - 1) / scene.tileSize;
    std::vector<int> tiles;
    tiles.reserve(static_cast<size_t>(tilesX) * tilesY);
    for (int tile = 0; tile < tilesX * tilesY; ++tile) tiles.push_back(tile);
    return tiles;
}

std::vector<std::vector<int>> buildTileDependencies(const Scene& scene,
                                                    const std::vector<RockCache>& caches) {
    const int tilesX = (scene.width + scene.tileSize - 1) / scene.tileSize;
    const int tilesY = (scene.height + scene.tileSize - 1) / scene.tileSize;
    std::vector<std::vector<int>> dependencies(static_cast<size_t>(tilesX) * tilesY);
    for (size_t rockIndex = 0; rockIndex < caches.size(); ++rockIndex) {
        const TileRange& range = caches[rockIndex].totalTileRange;
        if (!range.valid) continue;
        for (int ty = range.minTy; ty <= range.maxTy; ++ty) {
            for (int tx = range.minTx; tx <= range.maxTx; ++tx) {
                dependencies[static_cast<size_t>(ty * tilesX + tx)]
                    .push_back(static_cast<int>(rockIndex));
            }
        }
    }
    return dependencies;
}

void collectFrameStats(const Scene& scene, Framebuffer& frame, RenderStats& stats) {
    stats.frameHash = hex64(frame.colorHash());
    stats.depthHash = hex64(frame.depthHash());
    uint64_t combined = fnv1a64(frame.color.data(), frame.color.size());
    combined = fnv1a64(frame.depth.data(), frame.depth.size() * sizeof(uint32_t), combined);
    stats.combinedHash = hex64(combined);
    stats.tileHashes.clear();
    stats.tileHashes.reserve(static_cast<size_t>(frame.tileCount()));
    for (int tile = 0; tile < frame.tileCount(); ++tile) {
        stats.tileHashes.push_back({tile, hex64(frame.tileColorHash(tile))});
    }

    const uint32_t maxDepth = scene.depthMax();
    std::vector<uint32_t> depths;
    depths.reserve(frame.depth.size());
    for (const uint32_t depth : frame.depth) {
        if (depth != maxDepth) depths.push_back(depth);
    }
    stats.nonBackgroundPixels = depths.size();
    if (!depths.empty()) {
        const auto minmax = std::minmax_element(depths.begin(), depths.end());
        stats.minNonBackgroundDepth = *minmax.first;
        stats.maxNonBackgroundDepth = *minmax.second;
    }
    std::sort(depths.begin(), depths.end());
    depths.erase(std::unique(depths.begin(), depths.end()), depths.end());
    stats.distinctDepthValues = depths.size();
}

void initializeFrameDepth(Scene& scene, Framebuffer& frame) {
    std::fill(frame.depth.begin(), frame.depth.end(), scene.depthMax());
}

void collectRockCenters(const Scene& scene, RenderStats& stats) {
    stats.rockWorldCenters.clear();
    stats.rockWorldCenters.reserve(scene.rocks.size());
    for (const RockNode& rock : scene.rocks) {
        Vec3 sum;
        size_t count = 0;
        for (const RockTriangle& triangle : rock.worldTriangles) {
            sum += triangle.p[0];
            sum += triangle.p[1];
            sum += triangle.p[2];
            count += 3;
        }
        Vec3 center = count == 0 ? Vec3{} : sum / static_cast<double>(count);
        stats.rockWorldCenters.push_back({rock.id, center});
    }
}

bool overflowFailure(RenderResult& result, RenderStats& stats,
                     const std::string& message) {
    result.ok = false;
    result.hasFrame = false;
    result.code = "E_STENCIL_OVERFLOW";
    result.exitCode = 2;
    result.message = message;
    stats.status = "error";
    return false;
}

} // namespace

RenderResult runRender(Scene scene, RenderMode mode) {
    const auto totalStart = Clock::now();
    RenderResult result;
    RenderStats stats;
    stats.mode = mode == RenderMode::Full ? "full" : "incremental";
    stats.width = scene.width;
    stats.height = scene.height;
    stats.tileSize = scene.tileSize;
    stats.tileColumns = (scene.width + scene.tileSize - 1) / scene.tileSize;
    stats.tileRows = (scene.height + scene.tileSize - 1) / scene.tileSize;
    stats.depthBits = scene.depthBits;
    stats.depthMax = scene.depthMax();
    stats.stencilBits = scene.stencilBits;
    stats.initialFullTileCount = stats.tileColumns * stats.tileRows;

    Scene initial = scene;
    Scene current = scene;
    Framebuffer frame(scene.width, scene.height, scene.tileSize);
    initializeFrameDepth(current, frame);
    std::vector<RockCache> currentCaches;
    Clock::time_point cacheStart;
    const auto firstCacheStart = Clock::now();
    (void)firstCacheStart;
    currentCaches = buildAllCaches(current, cacheStart);
    stats.cacheBuildMs += elapsedMs(firstCacheStart, Clock::now());

    const int tileCount = frame.tileCount();
    std::vector<int> cumulativeAffected;
    std::vector<uint8_t> cumulativeMask(static_cast<size_t>(tileCount), 0U);
    std::vector<std::vector<int>> dependencies = buildTileDependencies(current, currentCaches);

    if (mode == RenderMode::Full) {
        Scene finalScene = initial;
        finalScene.changes.clear(); // Drained by applyAllChanges below.
        Scene withChanges = initial;
        applyAllChanges(withChanges);
        finalScene = std::move(withChanges);
        currentCaches = buildAllCaches(finalScene, cacheStart);
        const auto renderStart = Clock::now();
        const bool ok = renderTiles(finalScene, frame, currentCaches,
                                    allTiles(finalScene), stats);
        if (!ok) {
            result.frame = std::move(frame);
            result.stats = stats;
            overflowFailure(result, result.stats,
                "template buffer counter overflowed during full render");
            result.stats.totalMs = elapsedMs(totalStart, Clock::now());
            return result;
        }
        (void)renderStart;
        stats.changesApplied = static_cast<int>(initial.changes.size());
        cumulativeAffected = allTiles(finalScene);
        collectFrameStats(finalScene, frame, stats);
        result.ok = true;
        result.hasFrame = true;
        result.exitCode = 0;
        result.frame = std::move(frame);
        result.stats = stats;
        result.stats.tileDependencies = buildTileDependencies(finalScene, currentCaches);
        collectRockCenters(finalScene, result.stats);
        result.stats.affectedTiles = cumulativeAffected;
        result.stats.affectedTileCount = static_cast<int>(cumulativeAffected.size());
        result.stats.totalMs = elapsedMs(totalStart, Clock::now());
        return result;
    }

    // Incremental mode: render the initial scene completely, then apply mutations
    // one at a time and repaint only tiles touched by changed rocks/shadow volumes.
    {
        const auto initialStart = Clock::now();
        if (!renderTiles(current, frame, currentCaches, allTiles(current), stats)) {
            result.frame = std::move(frame);
            result.stats = stats;
            overflowFailure(result, result.stats,
                "template buffer counter overflowed during initial render");
            result.stats.totalMs = elapsedMs(totalStart, Clock::now());
            return result;
        }
        (void)initialStart;
        dependencies = buildTileDependencies(current, currentCaches);
    }

    Scene finalScene = initial;
    for (size_t changeIndex = 0; changeIndex < initial.changes.size(); ++changeIndex) {
        const auto changeStart = Clock::now();
        const Change& change = initial.changes[changeIndex];
        const std::string target = current.rocks[static_cast<size_t>(change.target)].id;
        std::vector<int> affectedRocks;
        if (change.kind == ChangeKind::Transform) {
            affectedRocks = current.subtree(change.target);
        } else {
            affectedRocks.push_back(change.target);
        }
        std::sort(affectedRocks.begin(), affectedRocks.end());
        affectedRocks.erase(std::unique(affectedRocks.begin(), affectedRocks.end()),
                            affectedRocks.end());

        std::unordered_set<int> affectedSet(affectedRocks.begin(), affectedRocks.end());
        std::vector<uint8_t> dirtyMask(static_cast<size_t>(tileCount), 0U);
        if (change.kind == ChangeKind::Transform) {
            // Transform changes move the rock and all descendants, and therefore also
            // move their shadow volumes. The retained tile dependency map is the source
            // of truth for the old footprint.
            for (int tile = 0; tile < tileCount; ++tile) {
                for (const int rockIndex : dependencies[static_cast<size_t>(tile)]) {
                    if (affectedSet.count(rockIndex) != 0) {
                        dirtyMask[static_cast<size_t>(tile)] = 1U;
                        break;
                    }
                }
            }
        } else {
            // Material parameters affect only visible surface pixels. Shadow volumes
            // are geometry-only and remain valid, so their tiles need no repaint.
            for (const int rockIndex : affectedRocks) {
                const TileRange& meshRange =
                    currentCaches[static_cast<size_t>(rockIndex)].meshTileRange;
                if (!meshRange.valid) continue;
                for (int ty = meshRange.minTy; ty <= meshRange.maxTy; ++ty) {
                    for (int tx = meshRange.minTx; tx <= meshRange.maxTx; ++tx) {
                        dirtyMask[static_cast<size_t>(ty * stats.tileColumns + tx)] = 1U;
                    }
                }
            }
        }

        applyChange(current, change);
        if (change.kind == ChangeKind::Transform) {
            rebuildWorldTransforms(current);
            for (const int rockIndex : affectedRocks) {
                rebuildRockGeometry(current, rockIndex);
            }
            for (const int rockIndex : affectedRocks) {
                const auto cacheBuildStart = Clock::now();
                currentCaches[static_cast<size_t>(rockIndex)] =
                    buildRockCache(current, rockIndex);
                stats.cacheBuildMs += elapsedMs(cacheBuildStart, Clock::now());
            }
        }

        for (const int rockIndex : affectedRocks) {
            const TileRange& range = change.kind == ChangeKind::Transform
                ? currentCaches[static_cast<size_t>(rockIndex)].totalTileRange
                : currentCaches[static_cast<size_t>(rockIndex)].meshTileRange;
            if (!range.valid) continue;
            for (int ty = range.minTy; ty <= range.maxTy; ++ty) {
                for (int tx = range.minTx; tx <= range.maxTx; ++tx) {
                    dirtyMask[static_cast<size_t>(ty * stats.tileColumns + tx)] = 1U;
                }
            }
        }
        std::vector<int> dirtyTiles;
        for (int tile = 0; tile < tileCount; ++tile) {
            if (dirtyMask[static_cast<size_t>(tile)] != 0U) dirtyTiles.push_back(tile);
        }
        if (!renderTiles(current, frame, currentCaches, dirtyTiles, stats)) {
            result.frame = std::move(frame);
            result.stats = stats;
            overflowFailure(result, result.stats,
                "template buffer counter overflowed while incrementally redrawing tiles");
            result.stats.totalMs = elapsedMs(totalStart, Clock::now());
            return result;
        }
        dependencies = buildTileDependencies(current, currentCaches);
        for (const int tile : dirtyTiles) cumulativeMask[static_cast<size_t>(tile)] = 1U;
        for (const int tile : dirtyTiles) {
            if (std::find(cumulativeAffected.begin(), cumulativeAffected.end(), tile) ==
                cumulativeAffected.end()) {
                cumulativeAffected.push_back(tile);
            }
        }
        std::sort(cumulativeAffected.begin(), cumulativeAffected.end());

        ChangeLogEntry log;
        log.index = static_cast<int>(changeIndex);
        log.target = target;
        log.kind = changeKindName(change.kind);
        log.affectedTiles = dirtyTiles;
        log.frameHash = hex64(frame.colorHash());
        stats.changeLog.push_back(std::move(log));
        ++stats.changesApplied;
        stats.incrementalUpdateMs += elapsedMs(changeStart, Clock::now());

        applyChange(finalScene, change);
        if (change.kind == ChangeKind::Transform) {
            rebuildWorldTransforms(finalScene);
            for (const int rockIndex : affectedRocks) rebuildRockGeometry(finalScene, rockIndex);
        }
    }
    rebuildWorldTransforms(finalScene);
    rebuildAllRockGeometry(finalScene);

    // Verify incremental output against an independent full render of the final scene.
    const auto comparisonStart = Clock::now();
    std::vector<RockCache> finalCaches =
        buildAllCaches(finalScene, cacheStart);
    Framebuffer fullFrame(finalScene.width, finalScene.height, finalScene.tileSize);
    initializeFrameDepth(finalScene, fullFrame);
    RenderStats comparisonStats; // Shadow counts for the reference are intentionally diagnostic only.
    comparisonStats.depthBits = finalScene.depthBits;
    comparisonStats.stencilBits = finalScene.stencilBits;
    const bool fullOk = renderTiles(finalScene, fullFrame, finalCaches,
                                    allTiles(finalScene), comparisonStats);
    if (!fullOk) {
        result.frame = std::move(frame);
        result.stats = stats;
        overflowFailure(result, result.stats,
            "template buffer counter overflowed during final full reference render");
        result.stats.totalMs = elapsedMs(totalStart, Clock::now());
        return result;
    }
    stats.fullReferenceFrameHash = hex64(fullFrame.colorHash());
    stats.fullReferenceDepthHash = hex64(fullFrame.depthHash());
    stats.divergenceChecked = true;
    stats.framebufferHashesEqual =
        frame.color == fullFrame.color && frame.depth == fullFrame.depth;
    stats.tileHashesEqual = true;
    for (int tile = 0; tile < tileCount; ++tile) {
        if (frame.tileColorHash(tile) != fullFrame.tileColorHash(tile)) {
            stats.tileHashesEqual = false;
            if (cumulativeMask[static_cast<size_t>(tile)] == 0U) {
                ++stats.unaffectedTileHashMismatches;
            }
        }
    }
    stats.fullComparisonMs = elapsedMs(comparisonStart, Clock::now());
    collectFrameStats(current, frame, stats);
    stats.affectedTiles = cumulativeAffected;
    stats.affectedTileCount = static_cast<int>(cumulativeAffected.size());
    stats.tileDependencies = buildTileDependencies(current, currentCaches);
    collectRockCenters(current, stats);
    stats.totalMs = elapsedMs(totalStart, Clock::now());
    result.stats = stats;
    result.frame = std::move(frame);

    if (!stats.framebufferHashesEqual || !stats.tileHashesEqual) {
        result.ok = false;
        result.hasFrame = false;
        result.code = "E_INCREMENTAL_DIVERGED";
        result.exitCode = 3;
        result.message = "incremental frame differs from final-scene full render";
        result.stats.status = "error";
        return result;
    }

    result.ok = true;
    result.hasFrame = true;
    result.exitCode = 0;
    return result;
}
