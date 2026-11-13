#include "renderer.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>

namespace dungeon {
namespace {

constexpr int kLightTileSize = 16;
constexpr double kRayEpsilon = 1e-8;
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;

struct GBufferPixel {
    Vec3 position;
    Vec3 normal;
    uint32_t materialId = 0;
    double depth = kInfinity;
};

struct ClipVertex {
    Vec3 view;
    Vec3 world;
    Vec3 normal;
};

struct ShadowSample {
    double lit = 1.0;
    uint64_t attempts = 0;
    uint64_t successes = 0;
    uint64_t shadowed = 0;
};

struct AtlasTile {
    bool allocated = false;
    std::vector<double> depth;
    uint64_t lightHash = 0;
    uint64_t sceneHash = 0;
    int geometryVersion = -1;
    std::string action;
};

struct LightStats {
    uint64_t attempts = 0;
    uint64_t successes = 0;
    uint64_t shadowed = 0;
    uint64_t litPixels = 0;
};

double edge2D(double ax, double ay, double bx, double by, double px, double py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

ClipVertex lerpVertex(const ClipVertex& a, const ClipVertex& b, double t) {
    return {
        a.view + (b.view - a.view) * t,
        a.world + (b.world - a.world) * t,
        normalize(a.normal + (b.normal - a.normal) * t),
    };
}

std::vector<ClipVertex> clipNearPlane(const ClipVertex& a, const ClipVertex& b, const ClipVertex& c, double nearZ) {
    const ClipVertex input[3] = {a, b, c};
    std::vector<ClipVertex> output;
    output.reserve(4);
    for (int i = 0; i < 3; ++i) {
        const ClipVertex& current = input[i];
        const ClipVertex& next = input[(i + 1) % 3];
        const bool currentInside = current.view.z >= nearZ;
        const bool nextInside = next.view.z >= nearZ;
        if (currentInside) {
            output.push_back(current);
        }
        if (currentInside != nextInside) {
            const double denominator = next.view.z - current.view.z;
            if (std::abs(denominator) > 1e-30) {
                const double t = (nearZ - current.view.z) / denominator;
                output.push_back(lerpVertex(current, next, t));
            }
        }
    }
    return output;
}

void rasterizeTriangle(
    const Triangle& triangle,
    const Camera& camera,
    int width,
    int height,
    std::vector<GBufferPixel>& gbuffer) {
    const Vec3 faceNormal = normalize(cross(triangle.p1 - triangle.p0, triangle.p2 - triangle.p0));
    const Vec3 centroid = (triangle.p0 + triangle.p1 + triangle.p2) / 3.0;
    if (dot(faceNormal, camera.position - centroid) <= kRayEpsilon) {
        return;
    }

    ClipVertex vertices[3] = {
        {camera.toView(triangle.p0), triangle.p0, triangle.n0},
        {camera.toView(triangle.p1), triangle.p1, triangle.n1},
        {camera.toView(triangle.p2), triangle.p2, triangle.n2},
    };

    std::vector<ClipVertex> clipped = clipNearPlane(vertices[0], vertices[1], vertices[2], camera.nearZ);
    if (clipped.size() < 3) {
        return;
    }

    for (std::size_t fan = 1; fan + 1 < clipped.size(); ++fan) {
        const ClipVertex& v0 = clipped[0];
        const ClipVertex& v1 = clipped[fan];
        const ClipVertex& v2 = clipped[fan + 1];

        double x0 = 0.0;
        double y0 = 0.0;
        double x1 = 0.0;
        double y1 = 0.0;
        double x2 = 0.0;
        double y2 = 0.0;
        camera.project(v0.view, x0, y0);
        camera.project(v1.view, x1, y1);
        camera.project(v2.view, x2, y2);

        const double area = edge2D(x0, y0, x1, y1, x2, y2);
        if (std::abs(area) <= 1e-12) {
            continue;
        }

        const int minX = std::max(0, static_cast<int>(std::floor(std::min({x0, x1, x2}))));
        const int maxX = std::min(width - 1, static_cast<int>(std::ceil(std::max({x0, x1, x2}))));
        const int minY = std::max(0, static_cast<int>(std::floor(std::min({y0, y1, y2}))));
        const int maxY = std::min(height - 1, static_cast<int>(std::ceil(std::max({y0, y1, y2}))));

        for (int py = minY; py <= maxY; ++py) {
            for (int px = minX; px <= maxX; ++px) {
                const double sampleX = static_cast<double>(px) + 0.5;
                const double sampleY = static_cast<double>(py) + 0.5;
                const double w0 = edge2D(x1, y1, x2, y2, sampleX, sampleY) / area;
                const double w1 = edge2D(x2, y2, x0, y0, sampleX, sampleY) / area;
                const double w2 = edge2D(x0, y0, x1, y1, sampleX, sampleY) / area;
                if (w0 < -1e-10 || w1 < -1e-10 || w2 < -1e-10) {
                    continue;
                }

                const double invZ = w0 / v0.view.z + w1 / v1.view.z + w2 / v2.view.z;
                if (!std::isfinite(invZ) || invZ <= 0.0) {
                    continue;
                }
                const double depth = 1.0 / invZ;
                GBufferPixel& target = gbuffer[static_cast<std::size_t>(py) * width + px];
                if (depth >= target.depth - 1e-12) {
                    continue;
                }

                const double p0 = (w0 / v0.view.z) / invZ;
                const double p1 = (w1 / v1.view.z) / invZ;
                const double p2 = (w2 / v2.view.z) / invZ;
                target.position = v0.world * p0 + v1.world * p1 + v2.world * p2;
                target.normal = normalize(v0.normal * p0 + v1.normal * p1 + v2.normal * p2);
                target.materialId = triangle.materialId;
                target.depth = depth;
            }
        }
    }
}

std::vector<GBufferPixel> generateGBuffer(const Scene& scene, const Camera& camera, int width, int height) {
    std::vector<GBufferPixel> gbuffer(static_cast<std::size_t>(width) * height);
    for (const Triangle& triangle : scene.triangles) {
        rasterizeTriangle(triangle, camera, width, height, gbuffer);
    }
    return gbuffer;
}

bool sphereIntersectsTile(
    const Camera& camera,
    int width,
    int height,
    int tileX,
    int tileY,
    int tileWidth,
    int tileHeight,
    const Vec3& center,
    double radius) {
    const double x0 = static_cast<double>(tileX);
    const double y0 = static_cast<double>(tileY);
    const double x1 = static_cast<double>(tileX + tileWidth);
    const double y1 = static_cast<double>(tileY + tileHeight);

    const Vec3 topLeft = camera.rayDirection(x0, y0, width, height);
    const Vec3 topRight = camera.rayDirection(x1, y0, width, height);
    const Vec3 bottomLeft = camera.rayDirection(x0, y1, width, height);
    const Vec3 bottomRight = camera.rayDirection(x1, y1, width, height);
    const Vec3 centerRay = camera.rayDirection(
        0.5 * (x0 + x1),
        0.5 * (y0 + y1),
        width,
        height);

    const double forwardDepth = dot(center - camera.position, camera.basis.forward);
    if (forwardDepth + radius < camera.nearZ || forwardDepth - radius > camera.farZ) {
        return false;
    }

    auto sideRejects = [&](const Vec3& firstCorner, const Vec3& secondCorner) {
        Vec3 planeNormal = normalize(cross(firstCorner, secondCorner));
        if (lengthSquared(planeNormal) <= 1e-30) {
            return false;
        }
        if (dot(planeNormal, centerRay) < 0.0) {
            planeNormal = -planeNormal;
        }
        return dot(planeNormal, center - camera.position) + radius < 0.0;
    };

    if (sideRejects(topLeft, bottomLeft) ||
        sideRejects(topRight, bottomRight) ||
        sideRejects(topLeft, topRight) ||
        sideRejects(bottomLeft, bottomRight)) {
        return false;
    }
    return true;
}

bool rayTriangle(
    const Ray& ray,
    const Triangle& triangle,
    double tMin,
    double tMax,
    double& hitDistance) {
    const Vec3 edge1 = triangle.p1 - triangle.p0;
    const Vec3 edge2 = triangle.p2 - triangle.p0;
    const Vec3 p = cross(ray.direction, edge2);
    const double determinant = dot(edge1, p);
    if (std::abs(determinant) < 1e-12) {
        return false;
    }
    const double invDeterminant = 1.0 / determinant;
    const Vec3 tVector = ray.origin - triangle.p0;
    const double u = dot(tVector, p) * invDeterminant;
    if (u < 0.0 || u > 1.0) {
        return false;
    }
    const Vec3 q = cross(tVector, edge1);
    const double v = dot(ray.direction, q) * invDeterminant;
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }
    const double t = dot(edge2, q) * invDeterminant;
    if (t <= tMin || t >= tMax) {
        return false;
    }
    hitDistance = t;
    return true;
}

double traceShadowRay(const Scene& scene, const Ray& ray, double maxDistance) {
    double nearest = kInfinity;
    for (const Triangle& triangle : scene.triangles) {
        double hit = 0.0;
        if (rayTriangle(ray, triangle, 1e-5, maxDistance, hit) && hit < nearest) {
            nearest = hit;
        }
    }
    return nearest;
}

std::vector<double> buildSpotShadowTile(const Scene& scene, const Light& light, int tileSize) {
    std::vector<double> depth(static_cast<std::size_t>(tileSize) * tileSize, kInfinity);
    const Basis basis = makeBasis(light.direction);
    const double tanOuter = std::tan(std::clamp(light.outerConeDegrees, 1.0, 89.0) * kPi / 180.0);
    for (int y = 0; y < tileSize; ++y) {
        const double v = (static_cast<double>(y) + 0.5) / tileSize;
        const double ndcY = 1.0 - 2.0 * v;
        for (int x = 0; x < tileSize; ++x) {
            const double u = (static_cast<double>(x) + 0.5) / tileSize;
            const double ndcX = 2.0 * u - 1.0;
            const Ray ray{
                light.position,
                normalize(
                    basis.forward +
                    basis.right * (ndcX * tanOuter) +
                    basis.up * (ndcY * tanOuter)),
            };
            depth[static_cast<std::size_t>(y) * tileSize + x] =
                traceShadowRay(scene, ray, light.range);
        }
    }
    return depth;
}

std::vector<double> buildPointShadowTile(const Scene& scene, const Light& light, int tileSize) {
    std::vector<double> depth(static_cast<std::size_t>(tileSize) * tileSize, kInfinity);
    for (int y = 0; y < tileSize; ++y) {
        const double v = (static_cast<double>(y) + 0.5) / tileSize;
        const double theta = v * kPi;
        const double sinTheta = std::sin(theta);
        const double cosTheta = std::cos(theta);
        for (int x = 0; x < tileSize; ++x) {
            const double u = (static_cast<double>(x) + 0.5) / tileSize;
            const double longitude = (u - 0.5) * 2.0 * kPi;
            const Vec3 direction{
                sinTheta * std::sin(longitude),
                cosTheta,
                sinTheta * std::cos(longitude),
            };
            const Ray ray{light.position, direction};
            depth[static_cast<std::size_t>(y) * tileSize + x] =
                traceShadowRay(scene, ray, light.range);
        }
    }
    return depth;
}

std::vector<double> buildShadowTile(const Scene& scene, const Light& light, int tileSize) {
    if (light.type == LightType::Spot) {
        return buildSpotShadowTile(scene, light, tileSize);
    }
    return buildPointShadowTile(scene, light, tileSize);
}

bool lookupDepth(
    const std::vector<double>& depth,
    int tileSize,
    bool wrapU,
    int x,
    int y,
    double& value) {
    if (wrapU) {
        x %= tileSize;
        if (x < 0) {
            x += tileSize;
        }
    } else if (x < 0 || x >= tileSize) {
        return false;
    }
    if (y < 0 || y >= tileSize) {
        return false;
    }
    value = depth[static_cast<std::size_t>(y) * tileSize + x];
    return true;
}

ShadowSample sampleShadow(
    const std::vector<double>& depth,
    int tileSize,
    bool pointLight,
    double u,
    double v,
    double referenceDistance,
    double bias) {
    ShadowSample result;
    const int baseX = static_cast<int>(std::floor(u * tileSize));
    const int baseY = static_cast<int>(std::floor(v * tileSize));
    double litSum = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            double stored = kInfinity;
            ++result.attempts;
            if (!lookupDepth(depth, tileSize, pointLight, baseX + dx, baseY + dy, stored)) {
                continue;
            }
            if (std::isnan(stored)) {
                continue;
            }
            ++result.successes;
            if (referenceDistance - bias > stored) {
                ++result.shadowed;
                litSum += 0.0;
            } else {
                litSum += 1.0;
            }
        }
    }
    if (result.successes != 0) {
        result.lit = litSum / static_cast<double>(result.successes);
    }
    return result;
}

Vec3 materialAlbedo(uint32_t materialId) {
    switch (materialId) {
        case 1:
            return {0.49, 0.45, 0.38};
        case 2:
            return {0.38, 0.41, 0.47};
        case 3:
            return {0.55, 0.51, 0.45};
        case 4:
            return {0.55, 0.32, 0.20};
        default:
            return {0.10, 0.10, 0.10};
    }
}

double pointAttenuation(const Light& light, double distance) {
    if (distance >= light.range) {
        return 0.0;
    }
    const double window = 1.0 - distance / light.range;
    return (light.intensity / (1.0 + 0.20 * distance * distance)) * window * window;
}

double spotAttenuation(const Light& light, const Vec3& lightToSurface) {
    const double cosAngle = dot(normalize(lightToSurface), normalize(light.direction));
    const double cosOuter = std::cos(std::clamp(light.outerConeDegrees, 0.1, 179.0) * kPi / 180.0);
    const double cosInner = std::cos(std::clamp(light.innerConeDegrees, 0.0, 179.0) * kPi / 180.0);
    if (cosAngle <= cosOuter) {
        return 0.0;
    }
    if (cosAngle >= cosInner) {
        return 1.0;
    }
    return smoothStep01((cosAngle - cosOuter) / std::max(1e-12, cosInner - cosOuter));
}

std::pair<double, double> spotShadowUv(const Light& light, const Vec3& directionFromLight) {
    const Basis basis = makeBasis(light.direction);
    const double forward = dot(directionFromLight, basis.forward);
    if (forward <= 1e-9) {
        return {-1.0, -1.0};
    }
    const double tanOuter = std::tan(std::clamp(light.outerConeDegrees, 1.0, 89.0) * kPi / 180.0);
    const double ndcX = (dot(directionFromLight, basis.right) / forward) / tanOuter;
    const double ndcY = (dot(directionFromLight, basis.up) / forward) / tanOuter;
    return {ndcX * 0.5 + 0.5, 0.5 - ndcY * 0.5};
}

std::pair<double, double> pointShadowUv(const Vec3& directionFromLight) {
    const Vec3 dir = normalize(directionFromLight);
    const double u = std::atan2(dir.x, dir.z) / (2.0 * kPi) + 0.5;
    const double v = std::acos(std::clamp(dir.y, -1.0, 1.0)) / kPi;
    return {u, v};
}

std::string lightTypeName(LightType type) {
    return type == LightType::Point ? "point" : "spot";
}

}  // namespace

RenderReport renderDeferred(const Scene& scene, const RenderSettings& settings) {
    RenderReport report;
    report.width = settings.width;
    report.height = settings.height;
    report.tileColumns = (settings.width + kLightTileSize - 1) / kLightTileSize;
    report.tileRows = (settings.height + kLightTileSize - 1) / kLightTileSize;
    report.atlasColumns = settings.atlasColumns;
    report.atlasRows = settings.atlasRows;
    report.shadowTileSize = settings.shadowTileSize;
    report.sceneHash = scene.geometryHash;
    report.cacheRequested = !settings.cachePath.empty();

    const std::vector<GBufferPixel> gbuffer =
        generateGBuffer(scene, settings.camera, settings.width, settings.height);
    report.materialPixelCounts.assign(8, 0);
    for (const GBufferPixel& pixel : gbuffer) {
        if (pixel.materialId != 0) {
            report.gbufferCoveredPixels += 1;
            if (pixel.materialId < report.materialPixelCounts.size()) {
                report.materialPixelCounts[pixel.materialId] += 1;
            }
        }
    }

    std::vector<const Light*> sortedLights;
    sortedLights.reserve(scene.lights.size());
    for (const Light& light : scene.lights) {
        sortedLights.push_back(&light);
    }
    std::sort(sortedLights.begin(), sortedLights.end(), [](const Light* a, const Light* b) {
        return a->id < b->id;
    });

    ShadowCache loadedCache;
    if (report.cacheRequested) {
        report.cacheLoaded = loadShadowCache(
            settings.cachePath,
            settings.atlasColumns,
            settings.atlasRows,
            settings.shadowTileSize,
            loadedCache);
    }

    std::map<uint32_t, const ShadowCacheEntry*> cachedEntries;
    for (const ShadowCacheEntry& entry : loadedCache.entries) {
        cachedEntries[entry.lightId] = &entry;
    }

    const int capacity = settings.atlasColumns * settings.atlasRows;
    std::vector<AtlasTile> atlas(static_cast<std::size_t>(capacity));
    std::vector<const Light*> casters;
    for (const Light* light : sortedLights) {
        if (light->castsShadow) {
            casters.push_back(light);
        }
    }

    std::map<uint32_t, int> slotByLight;
    for (std::size_t index = 0; index < casters.size(); ++index) {
        const Light& light = *casters[index];
        if (static_cast<int>(index) >= capacity) {
            report.businessErrors.push_back({
                "E_ATLAS_FULL",
                "shadow atlas capacity exhausted; deterministic no-shadow fallback applied",
                static_cast<int>(light.id),
            });
            report.degradedLights += 1;
            continue;
        }
        const int slot = static_cast<int>(index);
        slotByLight[light.id] = slot;
        AtlasTile& tile = atlas[static_cast<std::size_t>(slot)];
        tile.allocated = true;
        const uint64_t transformHash = lightTransformHash(light);
        const auto cached = cachedEntries.find(light.id);
        bool reuse = false;
        if (cached != cachedEntries.end()) {
            const ShadowCacheEntry& entry = *cached->second;
            reuse =
                entry.slot == slot &&
                entry.geometryVersion == settings.geometryVersion &&
                entry.lightHash == transformHash &&
                entry.sceneHash == scene.geometryHash &&
                entry.depth.size() ==
                    static_cast<std::size_t>(settings.shadowTileSize) * settings.shadowTileSize;
        }
        if (reuse) {
            tile.depth = cached->second->depth;
            tile.action = "hit";
            report.tileHits += 1;
        } else {
            tile.depth = buildShadowTile(scene, light, settings.shadowTileSize);
            tile.action = "rebuild";
            report.tileRebuilds += 1;
        }
        tile.lightHash = transformHash;
        tile.sceneHash = scene.geometryHash;
        tile.geometryVersion = settings.geometryVersion;
        report.atlasAllocated += 1;
    }

    ShadowCache cacheToSave;
    cacheToSave.columns = settings.atlasColumns;
    cacheToSave.rows = settings.atlasRows;
    cacheToSave.tileSize = settings.shadowTileSize;
    for (const auto& item : slotByLight) {
        const int slot = item.second;
        const AtlasTile& tile = atlas[static_cast<std::size_t>(slot)];
        ShadowCacheEntry entry;
        entry.lightId = item.first;
        entry.slot = slot;
        entry.geometryVersion = tile.geometryVersion;
        entry.lightHash = tile.lightHash;
        entry.sceneHash = tile.sceneHash;
        entry.depth = tile.depth;
        cacheToSave.entries.push_back(std::move(entry));
    }
    if (report.cacheRequested) {
        report.cacheSaved = saveShadowCache(settings.cachePath, cacheToSave);
    }

    std::vector<LightStats> lightStats(sortedLights.size());
    std::map<uint32_t, std::size_t> lightStatsIndex;
    for (std::size_t index = 0; index < sortedLights.size(); ++index) {
        lightStatsIndex[sortedLights[index]->id] = index;
    }

    const Vec3 ambient{0.075, 0.082, 0.105};
    uint64_t pixelHash = kFnvOffset;

    report.tiles.reserve(static_cast<std::size_t>(report.tileColumns) * report.tileRows);
    for (int tileY = 0; tileY < report.tileRows; ++tileY) {
        const int pixelY0 = tileY * kLightTileSize;
        const int tileHeight = std::min(kLightTileSize, settings.height - pixelY0);
        for (int tileX = 0; tileX < report.tileColumns; ++tileX) {
            const int pixelX0 = tileX * kLightTileSize;
            const int tileWidth = std::min(kLightTileSize, settings.width - pixelX0);

            TileReport tile;
            tile.x = pixelX0;
            tile.y = pixelY0;
            tile.width = tileWidth;
            tile.height = tileHeight;
            for (const Light* light : sortedLights) {
                if (sphereIntersectsTile(
                        settings.camera,
                        settings.width,
                        settings.height,
                        pixelX0,
                        pixelY0,
                        tileWidth,
                        tileHeight,
                        light->boundingSphereCenter(),
                        light->boundingSphereRadius())) {
                    tile.lightIds.push_back(light->id);
                }
            }
            std::sort(tile.lightIds.begin(), tile.lightIds.end());
            if (tile.lightIds.empty()) {
                report.ambientOnlyTiles += 1;
            }

            uint64_t tileHash = kFnvOffset;
            uint64_t lumaAccumulator = 0;
            for (int py = pixelY0; py < pixelY0 + tileHeight; ++py) {
                for (int px = pixelX0; px < pixelX0 + tileWidth; ++px) {
                    const std::size_t pixelIndex =
                        static_cast<std::size_t>(py) * settings.width + px;
                    const GBufferPixel& gbufferPixel = gbuffer[pixelIndex];
                    Vec3 color = ambient;

                    if (gbufferPixel.materialId != 0) {
                        const Vec3 albedo = materialAlbedo(gbufferPixel.materialId);
                        for (uint32_t lightId : tile.lightIds) {
                            const auto lightIndexIt = lightStatsIndex.find(lightId);
                            if (lightIndexIt == lightStatsIndex.end()) {
                                continue;
                            }
                            const Light& light = *sortedLights[lightIndexIt->second];
                            const Vec3 toLight = light.position - gbufferPixel.position;
                            const double distance = length(toLight);
                            if (distance <= 1e-9 || distance >= light.range) {
                                continue;
                            }
                            const Vec3 lightDirection = toLight / distance;
                            const double nDotL = std::max(0.0, dot(gbufferPixel.normal, lightDirection));
                            if (nDotL <= 0.0) {
                                continue;
                            }

                            double attenuation = pointAttenuation(light, distance);
                            if (light.type == LightType::Spot) {
                                attenuation *= spotAttenuation(light, -toLight);
                            }
                            if (attenuation <= 0.0) {
                                continue;
                            }

                            double shadowFactor = 1.0;
                            const auto slotIt = slotByLight.find(light.id);
                            if (light.castsShadow && slotIt != slotByLight.end()) {
                                const AtlasTile& tileData = atlas[static_cast<std::size_t>(slotIt->second)];
                                const Vec3 lightToSurface = gbufferPixel.position - light.position;
                                const double referenceDistance = length(lightToSurface);
                                double u = 0.0;
                                double v = 0.0;
                                bool validUv = true;
                                if (light.type == LightType::Spot) {
                                    const auto uv = spotShadowUv(light, lightToSurface);
                                    u = uv.first;
                                    v = uv.second;
                                    validUv = u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0;
                                } else {
                                    const auto uv = pointShadowUv(lightToSurface);
                                    u = uv.first;
                                    v = uv.second;
                                }
                                if (validUv) {
                                    const ShadowSample sample = sampleShadow(
                                        tileData.depth,
                                        settings.shadowTileSize,
                                        light.type == LightType::Point,
                                        u,
                                        v,
                                        referenceDistance,
                                        settings.shadowBias);
                                    shadowFactor = sample.lit;
                                    LightStats& stats = lightStats[lightIndexIt->second];
                                    stats.attempts += sample.attempts;
                                    stats.successes += sample.successes;
                                    stats.shadowed += sample.shadowed;
                                    report.shadowComparisons += sample.attempts;
                                }
                            }

                            const double energy = nDotL * attenuation * shadowFactor;
                            if (energy > 1e-9) {
                                color += albedo * light.color * energy;
                                lightStats[lightIndexIt->second].litPixels += 1;
                            }
                        }
                    }

                    color = clampColor(color, 0.0, 16.0);
                    const uint32_t quantized =
                        quantizeLuminance(color.x, color.y, color.z);
                    tileHash = fnv1aU32(tileHash, quantized);
                    pixelHash = fnv1aU32(pixelHash, quantized);
                    lumaAccumulator += quantized;
                }
            }
            const uint64_t pixelCount =
                static_cast<uint64_t>(tileWidth) * static_cast<uint64_t>(tileHeight);
            tile.luminanceChecksum = tileHash;
            tile.averageLuminance =
                static_cast<double>(lumaAccumulator) /
                (static_cast<double>(pixelCount) * 1000000.0);
            report.tiles.push_back(std::move(tile));
        }
    }

    report.pixelLuminanceChecksum = pixelHash;
    for (const Light* light : sortedLights) {
        LightReport lightReport;
        lightReport.id = light->id;
        lightReport.type = lightTypeName(light->type);
        lightReport.castsShadow = light->castsShadow;
        lightReport.transformHash = lightTransformHash(*light);

        const auto slotIt = slotByLight.find(light->id);
        if (slotIt != slotByLight.end()) {
            lightReport.tileIndex = slotIt->second;
            lightReport.atlasX = slotIt->second % settings.atlasColumns;
            lightReport.atlasY = slotIt->second / settings.atlasColumns;
            lightReport.action = atlas[static_cast<std::size_t>(slotIt->second)].action;
        } else if (light->castsShadow) {
            lightReport.action = "degraded";
        }

        const LightStats& stats = lightStats[lightStatsIndex[light->id]];
        lightReport.shadowTapAttempts = stats.attempts;
        lightReport.shadowTapSuccesses = stats.successes;
        lightReport.shadowTapShadowed = stats.shadowed;
        lightReport.sampleSuccessRate =
            stats.attempts == 0
                ? 0.0
                : static_cast<double>(stats.successes) / static_cast<double>(stats.attempts);
        lightReport.shadowedRate =
            stats.successes == 0
                ? 0.0
                : static_cast<double>(stats.shadowed) / static_cast<double>(stats.successes);
        lightReport.litPixels = stats.litPixels;
        report.lights.push_back(std::move(lightReport));
    }

    return report;
}

}  // namespace dungeon
