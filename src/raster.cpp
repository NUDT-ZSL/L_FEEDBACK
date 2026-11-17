#include "raster.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace cv {
namespace {

constexpr int kMaxClipVertices = 12;

float edgeFunction(float ax, float ay, float bx, float by, float px, float py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

float signedArea(const Vec2& a, const Vec2& b, const Vec2& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

Vec4 lerp4(Vec4 a, Vec4 b, float t) { return a + (b - a) * t; }

Vec3 lerp3(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }

Vec2 lerp2(Vec2 a, Vec2 b, float t) { return a + (b - a) * t; }

RasterVertex lerpVertex(const RasterVertex& a, const RasterVertex& b, float t) {
    RasterVertex out;
    out.clip = lerp4(a.clip, b.clip, t);
    out.worldPosition = lerp3(a.worldPosition, b.worldPosition, t);
    out.worldNormal = lerp3(a.worldNormal, b.worldNormal, t);
    out.worldTangent =
        lerp4(a.worldTangent, b.worldTangent, t);
    out.uv = lerp2(a.uv, b.uv, t);
    return out;
}

float planeDistance(const Vec4& p, const Vec4& v) {
    return p.x * v.x + p.y * v.y + p.z * v.z + p.w * v.w;
}

int clipPolygon(const std::array<RasterVertex, 3>& input,
                std::array<RasterVertex, kMaxClipVertices>& output) {
    // D3D clip volume: -w <= x,y <= w and 0 <= z <= w.
    const std::array<Vec4, 6> planes{{
        {1.0f, 0.0f, 0.0f, 1.0f},
        {-1.0f, 0.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 0.0f, 1.0f},
        {0.0f, -1.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, -1.0f, 1.0f},
    }};

    std::array<RasterVertex, kMaxClipVertices> a{};
    std::array<RasterVertex, kMaxClipVertices> b{};
    a[0] = input[0];
    a[1] = input[1];
    a[2] = input[2];
    int count = 3;

    for (const Vec4& plane : planes) {
        int outCount = 0;
        for (int i = 0; i < count; ++i) {
            const RasterVertex& current = a[static_cast<size_t>(i)];
            const RasterVertex& next = a[static_cast<size_t>((i + 1) % count)];
            const float dc = planeDistance(plane, current.clip);
            const float dn = planeDistance(plane, next.clip);
            const bool currentInside = dc >= 0.0f;
            const bool nextInside = dn >= 0.0f;
            if (currentInside) {
                b[static_cast<size_t>(outCount++)] = current;
            }
            if (currentInside != nextInside) {
                const float denom = dc - dn;
                const float t = std::abs(denom) > 1.0e-12f ? dc / denom : 0.0f;
                b[static_cast<size_t>(outCount++)] = lerpVertex(current, next, t);
            }
            if (outCount >= kMaxClipVertices - 1) {
                break;
            }
        }
        count = outCount;
        a = b;
        if (count < 3) {
            return 0;
        }
    }
    output = a;
    return count;
}

struct ScreenVertex {
    RasterVertex attr;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float invW = 1.0f;
};

bool projectToScreen(const RasterVertex& in, int width, int height, ScreenVertex& out) {
    if (in.clip.w <= 1.0e-7f) {
        return false;
    }
    const float invW = 1.0f / in.clip.w;
    const float ndcX = in.clip.x * invW;
    const float ndcY = in.clip.y * invW;
    const float ndcZ = in.clip.z * invW;
    out.attr = in;
    out.x = (ndcX * 0.5f + 0.5f) * static_cast<float>(width);
    out.y = (0.5f - ndcY * 0.5f) * static_cast<float>(height);
    out.z = ndcZ;
    out.invW = invW;
    return true;
}

bool frontFacing(const ScreenVertex& a, const ScreenVertex& b, const ScreenVertex& c) {
    // The scene uses outward counter-clockwise winding as seen from outside.
    // With the left-handed D3D camera basis and screen Y pointing down, that
    // winding maps to positive screen-space signed area.
    return signedArea({a.x, a.y}, {b.x, b.y}, {c.x, c.y}) > 0.0f;
}

bool culled(bool front, CullMode mode) {
    return (mode == CullMode::Back && !front) || (mode == CullMode::Front && front);
}

template <typename Pixel>
void rasterScreenTriangle(const ScreenVertex& v0, const ScreenVertex& v1,
                          const ScreenVertex& v2, int width, int height, Pixel&& pixel) {
    const float area = edgeFunction(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y);
    if (std::abs(area) < 1.0e-12f) {
        return;
    }

    const float minXf = std::min({v0.x, v1.x, v2.x});
    const float maxXf = std::max({v0.x, v1.x, v2.x});
    const float minYf = std::min({v0.y, v1.y, v2.y});
    const float maxYf = std::max({v0.y, v1.y, v2.y});
    const int minX = std::max(0, static_cast<int>(std::floor(minXf - 0.5f)));
    const int maxX = std::min(width - 1, static_cast<int>(std::ceil(maxXf - 0.5f)));
    const int minY = std::max(0, static_cast<int>(std::floor(minYf - 0.5f)));
    const int maxY = std::min(height - 1, static_cast<int>(std::ceil(maxYf - 0.5f)));
    const float invArea = 1.0f / area;

    for (int y = minY; y <= maxY; ++y) {
        const float py = static_cast<float>(y) + 0.5f;
        for (int x = minX; x <= maxX; ++x) {
            const float px = static_cast<float>(x) + 0.5f;
            const float w0 = edgeFunction(v1.x, v1.y, v2.x, v2.y, px, py);
            const float w1 = edgeFunction(v2.x, v2.y, v0.x, v0.y, px, py);
            const float w2 = edgeFunction(v0.x, v0.y, v1.x, v1.y, px, py);
            const bool inside = area > 0.0f ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
                                            : (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
            if (!inside) {
                continue;
            }
            const float l0 = w0 * invArea;
            const float l1 = w1 * invArea;
            const float l2 = w2 * invArea;
            const float z = l0 * v0.z + l1 * v1.z + l2 * v2.z;
            pixel(x, y, z, l0, l1, l2);
        }
    }
}

}  // namespace

void Rasterizer::resize(int width, int height) {
    width_ = std::max(1, width);
    height_ = std::max(1, height);
    color_.assign(static_cast<size_t>(width_) * static_cast<size_t>(height_), Vec3{});
    depth_.assign(static_cast<size_t>(width_) * static_cast<size_t>(height_), 1.0f);
}

void Rasterizer::clear(Vec3 linearColor, float depthValue) {
    std::fill(color_.begin(), color_.end(), linearColor);
    std::fill(depth_.begin(), depth_.end(), depthValue);
}

void Rasterizer::drawTriangle(const RasterVertex& a, const RasterVertex& b,
                              const RasterVertex& c, const RasterState& state,
                              const FragmentShader& shader) {
    std::array<RasterVertex, kMaxClipVertices> clipped{};
    const int count = clipPolygon({a, b, c}, clipped);
    if (count < 3) {
        return;
    }
    std::array<ScreenVertex, kMaxClipVertices> screen{};
    for (int i = 0; i < count; ++i) {
        if (!projectToScreen(clipped[static_cast<size_t>(i)], width_, height_,
                             screen[static_cast<size_t>(i)])) {
            return;
        }
    }

    for (int i = 1; i + 1 < count; ++i) {
        const ScreenVertex& v0 = screen[0];
        const ScreenVertex& v1 = screen[static_cast<size_t>(i)];
        const ScreenVertex& v2 = screen[static_cast<size_t>(i + 1)];
        const bool front = frontFacing(v0, v1, v2);
        if (culled(front, state.cull)) {
            continue;
        }
        rasterScreenTriangle(
            v0, v1, v2, width_, height_,
            [&](int x, int y, float z, float l0, float l1, float l2) {
                const size_t offset =
                    static_cast<size_t>(y) * static_cast<size_t>(width_) +
                    static_cast<size_t>(x);
                if (z < 0.0f || z > 1.0f || z > depth_[offset]) {
                    return;
                }
                const float denom =
                    l0 * v0.invW + l1 * v1.invW + l2 * v2.invW;
                if (std::abs(denom) < 1.0e-12f) {
                    return;
                }
                const float invDenom = 1.0f / denom;
                FragmentInput fragment;
                fragment.x = x;
                fragment.y = y;
                fragment.depth = z;
                fragment.worldPosition =
                    (v0.attr.worldPosition * (l0 * v0.invW) +
                     v1.attr.worldPosition * (l1 * v1.invW) +
                     v2.attr.worldPosition * (l2 * v2.invW)) *
                    invDenom;
                fragment.worldNormal =
                    (v0.attr.worldNormal * (l0 * v0.invW) +
                     v1.attr.worldNormal * (l1 * v1.invW) +
                     v2.attr.worldNormal * (l2 * v2.invW)) *
                    invDenom;
                fragment.worldTangent =
                    (v0.attr.worldTangent * (l0 * v0.invW) +
                     v1.attr.worldTangent * (l1 * v1.invW) +
                     v2.attr.worldTangent * (l2 * v2.invW)) *
                    invDenom;
                fragment.uv =
                    (v0.attr.uv * (l0 * v0.invW) + v1.attr.uv * (l1 * v1.invW) +
                     v2.attr.uv * (l2 * v2.invW)) *
                    invDenom;
                const ShadeResult shaded = shader(fragment);
                if (state.blend) {
                    const float alpha = clamp01(shaded.alpha);
                    color_[offset] =
                        shaded.linearColor * alpha + color_[offset] * (1.0f - alpha);
                } else {
                    color_[offset] = shaded.linearColor;
                }
                if (state.depthWrite) {
                    depth_[offset] = z;
                }
            });
    }
}

void Rasterizer::drawWireTriangle(const RasterVertex& a, const RasterVertex& b,
                                  const RasterVertex& c, CullMode cull,
                                  Vec3 linearColor) {
    std::array<RasterVertex, kMaxClipVertices> clipped{};
    const int count = clipPolygon({a, b, c}, clipped);
    if (count < 3) {
        return;
    }
    std::array<ScreenVertex, kMaxClipVertices> screen{};
    for (int i = 0; i < count; ++i) {
        if (!projectToScreen(clipped[static_cast<size_t>(i)], width_, height_,
                             screen[static_cast<size_t>(i)])) {
            return;
        }
    }
    const bool front = frontFacing(screen[0], screen[1], screen[2]);
    if (culled(front, cull)) {
        return;
    }

    for (int i = 0; i < count; ++i) {
        const ScreenVertex& v0 = screen[static_cast<size_t>(i)];
        const ScreenVertex& v1 = screen[static_cast<size_t>((i + 1) % count)];
        const float dx = v1.x - v0.x;
        const float dy = v1.y - v0.y;
        const float dz = v1.z - v0.z;
        const int steps =
            std::max(1, static_cast<int>(std::ceil(std::max(std::abs(dx), std::abs(dy)))));
        for (int s = 0; s <= steps; ++s) {
            const float t = static_cast<float>(s) / static_cast<float>(steps);
            const int x = static_cast<int>(std::lround(v0.x + dx * t));
            const int y = static_cast<int>(std::lround(v0.y + dy * t));
            const float z = v0.z + dz * t;
            if (x < 0 || x >= width_ || y < 0 || y >= height_ || z < 0.0f || z > 1.0f) {
                continue;
            }
            const size_t offset =
                static_cast<size_t>(y) * static_cast<size_t>(width_) +
                static_cast<size_t>(x);
            if (z <= depth_[offset]) {
                color_[offset] = linearColor;
                depth_[offset] = z;
            }
        }
    }
}

void DepthRasterizer::resize(int width, int height) {
    width_ = std::max(1, width);
    height_ = std::max(1, height);
    depth_.assign(static_cast<size_t>(width_) * static_cast<size_t>(height_), 1.0f);
}

void DepthRasterizer::clear(float value) { std::fill(depth_.begin(), depth_.end(), value); }

void DepthRasterizer::drawTriangle(const Mat4& lightViewProjection, Vec3 a, Vec3 b, Vec3 c,
                                   CullMode cull) {
    RasterVertex vertices[3];
    vertices[0].clip = lightViewProjection.transform({a.x, a.y, a.z, 1.0f});
    vertices[1].clip = lightViewProjection.transform({b.x, b.y, b.z, 1.0f});
    vertices[2].clip = lightViewProjection.transform({c.x, c.y, c.z, 1.0f});
    std::array<RasterVertex, kMaxClipVertices> clipped{};
    const int count = clipPolygon({vertices[0], vertices[1], vertices[2]}, clipped);
    if (count < 3) {
        return;
    }
    std::array<ScreenVertex, kMaxClipVertices> screen{};
    for (int i = 0; i < count; ++i) {
        if (!projectToScreen(clipped[static_cast<size_t>(i)], width_, height_,
                             screen[static_cast<size_t>(i)])) {
            return;
        }
    }
    for (int i = 1; i + 1 < count; ++i) {
        const ScreenVertex& v0 = screen[0];
        const ScreenVertex& v1 = screen[static_cast<size_t>(i)];
        const ScreenVertex& v2 = screen[static_cast<size_t>(i + 1)];
        const bool front = frontFacing(v0, v1, v2);
        if (culled(front, cull)) {
            continue;
        }
        rasterScreenTriangle(v0, v1, v2, width_, height_,
                             [&](int x, int y, float z, float, float, float) {
                                 if (z < 0.0f || z > 1.0f) {
                                     return;
                                 }
                                 const size_t offset =
                                     static_cast<size_t>(y) *
                                         static_cast<size_t>(width_) +
                                     static_cast<size_t>(x);
                                 if (z <= depth_[offset]) {
                                     depth_[offset] = z;
                                 }
                             });
    }
}

}  // namespace cv

