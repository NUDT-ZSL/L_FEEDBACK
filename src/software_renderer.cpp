#include "terrain/software_renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace terrain {
namespace {

struct ClipVertex {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct ScreenVertex {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float invW = 1.0f;
};

struct Rgb {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

constexpr std::array<std::array<float, 4>, 6> kClipPlanes{{
    {1.0f, 0.0f, 0.0f, 1.0f},
    {-1.0f, 0.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 0.0f, 1.0f},
    {0.0f, -1.0f, 0.0f, 1.0f},
    {0.0f, 0.0f, 1.0f, 1.0f},
    {0.0f, 0.0f, -1.0f, 1.0f},
}};

float planeDistance(const std::array<float, 4>& plane, const ClipVertex& vertex) {
    return plane[0] * vertex.x + plane[1] * vertex.y +
           plane[2] * vertex.z + plane[3] * vertex.w;
}

ClipVertex interpolate(const ClipVertex& a, const ClipVertex& b, float t) {
    return {
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
        a.w + (b.w - a.w) * t,
    };
}

std::vector<ClipVertex> clipPolygon(std::vector<ClipVertex> polygon) {
    for (const auto& plane : kClipPlanes) {
        if (polygon.empty()) {
            break;
        }
        std::vector<ClipVertex> output;
        output.reserve(polygon.size() + 4);
        ClipVertex previous = polygon.back();
        float previousDistance = planeDistance(plane, previous);
        for (const ClipVertex& current : polygon) {
            const float currentDistance = planeDistance(plane, current);
            const bool previousInside = previousDistance >= 0.0f;
            const bool currentInside = currentDistance >= 0.0f;
            if (previousInside != currentInside) {
                const float denominator = previousDistance - currentDistance;
                const float t = denominator != 0.0f
                                    ? previousDistance / denominator
                                    : 0.0f;
                output.push_back(interpolate(previous, current, t));
            }
            if (currentInside) {
                output.push_back(current);
            }
            previous = current;
            previousDistance = currentDistance;
        }
        polygon = std::move(output);
    }
    return polygon;
}

Rgb paletteColor(int depth, float value) {
    const std::array<Rgb, 6> palette{{
        {0.98f, 0.85f, 0.25f},
        {0.98f, 0.56f, 0.18f},
        {0.90f, 0.24f, 0.22f},
        {0.68f, 0.23f, 0.73f},
        {0.20f, 0.47f, 0.93f},
        {0.15f, 0.85f, 0.84f},
    }};
    const Rgb base = palette[static_cast<std::size_t>(std::clamp(depth, 0, 5))];
    const float scale = 0.62f + 0.38f * clampf(value, 0.0f, 1.0f);
    return {base.r * scale, base.g * scale, base.b * scale};
}

Rgb terrainColor(
    float averageHeight,
    const Vec3& normal,
    float amplitude) {
    const float normalizedHeight =
        clampf(averageHeight / std::max(amplitude, 1.0f) * 0.75f + 0.5f, 0.0f, 1.0f);

    Rgb base{0.16f, 0.34f, 0.18f};
    if (normalizedHeight < 0.52f) {
        base = {0.18f, 0.36f, 0.19f};
    } else if (normalizedHeight < 0.72f) {
        base = {0.44f, 0.40f, 0.27f};
    } else if (normalizedHeight < 0.88f) {
        base = {0.48f, 0.47f, 0.45f};
    } else {
        base = {0.82f, 0.86f, 0.90f};
    }

    const Vec3 lightDirection = normalize({-0.45f, 0.78f, -0.38f});
    const float diffuse = clampf(dot(normal, lightDirection), 0.0f, 1.0f);
    const float slope = clampf(normal.y, 0.0f, 1.0f);
    const float illumination = 0.33f + 0.52f * diffuse + 0.15f * slope;
    return {
        base.r * illumination,
        base.g * illumination,
        base.b * illumination,
    };
}

uint8_t toByte(float value) {
    return static_cast<uint8_t>(
        std::clamp(std::lround(value * 255.0f), 0L, 255L));
}

}  // namespace

SoftwareRenderer::SoftwareRenderer(
    const Terrain& terrain,
    int width,
    int height)
    : terrain_(terrain),
      width_(std::max(width, 1)),
      height_(std::max(height, 1)) {}

ImageRgb SoftwareRenderer::render(
    const SceneFrame& frame,
    DebugView view) const {
    ImageRgb image;
    image.width = width_;
    image.height = height_;
    image.pixels.resize(
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * 3u);

    std::vector<float> depth(
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_),
        std::numeric_limits<float>::infinity());

    const bool wireframe = view == DebugView::Wireframe;
    for (int y = 0; y < height_; ++y) {
        const float v = static_cast<float>(y) / static_cast<float>(std::max(height_ - 1, 1));
        for (int x = 0; x < width_; ++x) {
            const float u = static_cast<float>(x) / static_cast<float>(std::max(width_ - 1, 1));
            Rgb color;
            if (wireframe) {
                const float glow = std::max(0.0f, 1.0f - std::sqrt((u - 0.5f) * (u - 0.5f) +
                                                                   (v - 0.65f) * (v - 0.65f)) *
                                                                   1.4f);
                color = {0.025f + glow * 0.03f, 0.055f + glow * 0.06f, 0.09f + glow * 0.10f};
            } else {
                const Rgb top{0.10f, 0.24f, 0.49f};
                const Rgb horizon{0.60f, 0.73f, 0.84f};
                color = {
                    horizon.r + (top.r - horizon.r) * v,
                    horizon.g + (top.g - horizon.g) * v,
                    horizon.b + (top.b - horizon.b) * v,
                };
            }
            const std::size_t offset =
                (static_cast<std::size_t>(y) * width_ + x) * 3u;
            image.pixels[offset] = toByte(color.r);
            image.pixels[offset + 1] = toByte(color.g);
            image.pixels[offset + 2] = toByte(color.b);
        }
    }

    const float aspect =
        static_cast<float>(width_) / static_cast<float>(height_);
    const Mat4 viewProjection = frame.camera.viewProjectionMatrix(aspect);

    const auto transform = [&viewProjection](const Vec3& position) {
        const Vec3& p = position;
        ClipVertex result;
        result.x = viewProjection(0, 0) * p.x + viewProjection(0, 1) * p.y +
                   viewProjection(0, 2) * p.z + viewProjection(0, 3);
        result.y = viewProjection(1, 0) * p.x + viewProjection(1, 1) * p.y +
                   viewProjection(1, 2) * p.z + viewProjection(1, 3);
        result.z = viewProjection(2, 0) * p.x + viewProjection(2, 1) * p.y +
                   viewProjection(2, 2) * p.z + viewProjection(2, 3);
        result.w = viewProjection(3, 0) * p.x + viewProjection(3, 1) * p.y +
                   viewProjection(3, 2) * p.z + viewProjection(3, 3);
        return result;
    };

    const auto toScreen = [this](const ClipVertex& vertex) {
        const float invW = 1.0f / std::max(vertex.w, 1.0e-7f);
        const float ndcX = vertex.x * invW;
        const float ndcY = vertex.y * invW;
        const float ndcZ = vertex.z * invW;
        ScreenVertex result;
        result.x = (ndcX * 0.5f + 0.5f) * static_cast<float>(width_);
        result.y = (0.5f - ndcY * 0.5f) * static_cast<float>(height_);
        result.z = ndcZ;
        result.invW = invW;
        return result;
    };

    const auto rasterLine = [&](ScreenVertex a, ScreenVertex b, const Rgb& color) {
        auto clipAxis = [](float p, float q, float& u0, float& u1) {
            if (std::abs(p) < 1.0e-8f) {
                return q >= 0.0f;
            }
            const float t = q / p;
            if (p < 0.0f) {
                u0 = std::max(u0, t);
            } else {
                u1 = std::min(u1, t);
            }
            return u0 <= u1;
        };

        float u0 = 0.0f;
        float u1 = 1.0f;
        const float dx = b.x - a.x;
        const float dy = b.y - a.y;
        if (!clipAxis(-dx, a.x, u0, u1) ||
            !clipAxis(dx, static_cast<float>(width_ - 1) - a.x, u0, u1) ||
            !clipAxis(-dy, a.y, u0, u1) ||
            !clipAxis(dy, static_cast<float>(height_ - 1) - a.y, u0, u1)) {
            return;
        }

        const auto sample = [&a, &b](float t) {
            ScreenVertex result;
            result.x = a.x + (b.x - a.x) * t;
            result.y = a.y + (b.y - a.y) * t;
            result.invW = a.invW + (b.invW - a.invW) * t;
            const float zOverW =
                a.z * a.invW + (b.z * b.invW - a.z * a.invW) * t;
            result.z = zOverW / std::max(result.invW, 1.0e-8f);
            return result;
        };
        const ScreenVertex clippedA = sample(u0);
        const ScreenVertex clippedB = sample(u1);
        const float lineDx = clippedB.x - clippedA.x;
        const float lineDy = clippedB.y - clippedA.y;
        const int steps = std::max(
            static_cast<int>(std::ceil(std::max(std::abs(lineDx), std::abs(lineDy)))),
            1);
        for (int step = 0; step <= steps; ++step) {
            const float t =
                static_cast<float>(step) / static_cast<float>(steps);
            const int x = static_cast<int>(std::lround(clippedA.x + lineDx * t));
            const int y = static_cast<int>(std::lround(clippedA.y + lineDy * t));
            if (x < 0 || y < 0 || x >= width_ || y >= height_) {
                continue;
            }
            const float z = clippedA.z + (clippedB.z - clippedA.z) * t;
            const std::size_t pixel =
                static_cast<std::size_t>(y) * width_ + x;
            if (z < -1.0f || z > 1.0f || z >= depth[pixel]) {
                continue;
            }
            depth[pixel] = z;
            const std::size_t offset = pixel * 3u;
            image.pixels[offset] = toByte(color.r);
            image.pixels[offset + 1] = toByte(color.g);
            image.pixels[offset + 2] = toByte(color.b);
        }
    };

    const auto rasterTriangle =
        [&](const std::array<ScreenVertex, 3>& screen, const Rgb& color) {
            const float area =
                (screen[1].x - screen[0].x) * (screen[2].y - screen[0].y) -
                (screen[1].y - screen[0].y) * (screen[2].x - screen[0].x);
            if (std::abs(area) < 1.0e-6f) {
                return;
            }
            const float inverseArea = 1.0f / area;
            const int minX = std::max(
                0,
                static_cast<int>(std::floor(
                    std::min({screen[0].x, screen[1].x, screen[2].x}))));
            const int maxX = std::min(
                width_ - 1,
                static_cast<int>(std::ceil(
                    std::max({screen[0].x, screen[1].x, screen[2].x}))));
            const int minY = std::max(
                0,
                static_cast<int>(std::floor(
                    std::min({screen[0].y, screen[1].y, screen[2].y}))));
            const int maxY = std::min(
                height_ - 1,
                static_cast<int>(std::ceil(
                    std::max({screen[0].y, screen[1].y, screen[2].y}))));
            if (minX > maxX || minY > maxY) {
                return;
            }

            for (int y = minY; y <= maxY; ++y) {
                const float py = static_cast<float>(y) + 0.5f;
                for (int x = minX; x <= maxX; ++x) {
                    const float px = static_cast<float>(x) + 0.5f;
                    const float e0 =
                        (screen[1].x - px) * (screen[2].y - py) -
                        (screen[1].y - py) * (screen[2].x - px);
                    const float e1 =
                        (screen[2].x - px) * (screen[0].y - py) -
                        (screen[2].y - py) * (screen[0].x - px);
                    const float e2 =
                        (screen[0].x - px) * (screen[1].y - py) -
                        (screen[0].y - py) * (screen[1].x - px);
                    const float b0 = e0 * inverseArea;
                    const float b1 = e1 * inverseArea;
                    const float b2 = e2 * inverseArea;
                    if (b0 < -1.0e-5f || b1 < -1.0e-5f || b2 < -1.0e-5f) {
                        continue;
                    }
                    const float invW =
                        b0 * screen[0].invW +
                        b1 * screen[1].invW +
                        b2 * screen[2].invW;
                    if (invW <= 0.0f) {
                        continue;
                    }
                    const float zNdc =
                        (b0 * screen[0].z * screen[0].invW +
                         b1 * screen[1].z * screen[1].invW +
                         b2 * screen[2].z * screen[2].invW) /
                        invW;
                    if (zNdc < -1.0f || zNdc > 1.0f) {
                        continue;
                    }
                    const std::size_t pixel =
                        static_cast<std::size_t>(y) * width_ + x;
                    if (zNdc >= depth[pixel]) {
                        continue;
                    }
                    depth[pixel] = zNdc;
                    const std::size_t offset = pixel * 3u;
                    image.pixels[offset] = toByte(color.r);
                    image.pixels[offset + 1] = toByte(color.g);
                    image.pixels[offset + 2] = toByte(color.b);
                }
            }
        };

    for (std::size_t leafIndex = 0; leafIndex < frame.selection.leaves.size();
         ++leafIndex) {
        const LeafNode& leaf = frame.selection.leaves[leafIndex];
        std::array<Vec3, kPatchVertexCount * kPatchVertexCount> worldVertices{};
        for (int localZ = 0; localZ < kPatchVertexCount; ++localZ) {
            for (int localX = 0; localX < kPatchVertexCount; ++localX) {
                const float worldX =
                    leaf.x0 + (leaf.x1 - leaf.x0) *
                                  (static_cast<float>(localX) /
                                   static_cast<float>(kPatchQuads));
                const float worldZ =
                    leaf.z0 + (leaf.z1 - leaf.z0) *
                                  (static_cast<float>(localZ) /
                                   static_cast<float>(kPatchQuads));
                const std::size_t index =
                    static_cast<std::size_t>(localZ) * kPatchVertexCount +
                    static_cast<std::size_t>(localX);
                worldVertices[index] = {
                    worldX,
                    leaf.vertexY[index],
                    worldZ,
                };
            }
        }

        for (int localZ = 0; localZ < kPatchQuads; ++localZ) {
            for (int localX = 0; localX < kPatchQuads; ++localX) {
                const std::size_t i00 =
                    static_cast<std::size_t>(localZ) * kPatchVertexCount +
                    static_cast<std::size_t>(localX);
                const std::size_t i10 = i00 + 1;
                const std::size_t i01 = i00 + kPatchVertexCount;
                const std::size_t i11 = i01 + 1;
                const std::array<std::array<std::size_t, 3>, 2> triangles{{
                    {{i00, i10, i11}},
                    {{i00, i11, i01}},
                }};
                for (const auto& triangle : triangles) {
                    const Vec3& a = worldVertices[triangle[0]];
                    const Vec3& b = worldVertices[triangle[1]];
                    const Vec3& c = worldVertices[triangle[2]];
                    Vec3 normal = normalize(cross(b - a, c - a));
                    if (normal.y < 0.0f) {
                        normal = -normal;
                    }
                    const float averageHeight = (a.y + b.y + c.y) / 3.0f;
                    Rgb color;
                    if (view == DebugView::Lod) {
                        color = paletteColor(
                            leaf.key.depth,
                            clampf(normal.y * 0.8f, 0.0f, 1.0f));
                    } else {
                        color = terrainColor(
                            averageHeight,
                            normal,
                            terrain_.parameters().amplitude);
                    }

                    std::vector<ClipVertex> polygon{
                        transform(a),
                        transform(b),
                        transform(c),
                    };
                    polygon = clipPolygon(std::move(polygon));
                    if (polygon.size() < 3) {
                        continue;
                    }
                    std::vector<ScreenVertex> screen;
                    screen.reserve(polygon.size());
                    for (const ClipVertex& vertex : polygon) {
                        screen.push_back(toScreen(vertex));
                    }
                    if (wireframe) {
                        const float depthShade =
                            0.58f + 0.08f * static_cast<float>(leaf.key.depth);
                        const Rgb wireColor{0.18f, 0.78f * depthShade, 0.96f};
                        for (std::size_t i = 0; i < screen.size(); ++i) {
                            rasterLine(
                                screen[i],
                                screen[(i + 1) % screen.size()],
                                wireColor);
                        }
                        continue;
                    }
                    for (std::size_t i = 1; i + 1 < screen.size(); ++i) {
                        rasterTriangle(
                            {screen[0], screen[i], screen[i + 1]},
                            color);
                    }
                }
            }
        }
    }

    if (view == DebugView::Seam && !frame.seams.empty()) {
        for (const SeamResult& result : frame.seams) {
            for (std::size_t i = 0; i + 1 < result.seamLines.size(); i += 2) {
                ClipVertex a = transform(result.seamLines[i].position);
                ClipVertex b = transform(result.seamLines[i + 1].position);
                const float da = a.z + a.w;
                const float db = b.z + b.w;
                if (da < 0.0f && db < 0.0f) {
                    continue;
                }
                if (da < 0.0f) {
                    a = interpolate(a, b, da / (da - db));
                } else if (db < 0.0f) {
                    b = interpolate(a, b, db / (db - da));
                }
                if (a.w <= 0.0f || b.w <= 0.0f) {
                    continue;
                }
                ScreenVertex sa = toScreen(a);
                ScreenVertex sb = toScreen(b);
                const float red = 1.0f;
                const float green = 0.09f;
                const float blue = 0.12f;
                const float dx = sb.x - sa.x;
                const float dy = sb.y - sa.y;
                const int steps = std::max(
                    static_cast<int>(std::ceil(std::max(std::abs(dx), std::abs(dy)))),
                    1);
                for (int step = 0; step <= steps; ++step) {
                    const float t =
                        static_cast<float>(step) / static_cast<float>(steps);
                    const int x = static_cast<int>(std::lround(sa.x + dx * t));
                    const int y = static_cast<int>(std::lround(sa.y + dy * t));
                    if (x < 0 || y < 0 || x >= width_ || y >= height_) {
                        continue;
                    }
                    const float z = sa.z + (sb.z - sa.z) * t - 0.0008f;
                    const std::size_t pixel =
                        static_cast<std::size_t>(y) * width_ + x;
                    if (z >= depth[pixel] + 0.0002f) {
                        continue;
                    }
                    depth[pixel] = z;
                    const std::size_t offset = pixel * 3u;
                    image.pixels[offset] = toByte(red);
                    image.pixels[offset + 1] = toByte(green);
                    image.pixels[offset + 2] = toByte(blue);
                }
            }
        }
    }

    return image;
}

}  // namespace terrain

