#pragma once

#include "cv_math.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace cv {

enum class CullMode {
    None,
    Back,
    Front,
};

struct RasterVertex {
    Vec4 clip;
    Vec3 worldPosition;
    Vec3 worldNormal;
    Vec4 worldTangent;
    Vec2 uv;
};

struct FragmentInput {
    int x = 0;
    int y = 0;
    float depth = 1.0f;
    Vec3 worldPosition;
    Vec3 worldNormal;
    Vec4 worldTangent;
    Vec2 uv;
};

struct ShadeResult {
    Vec3 linearColor;
    float alpha = 1.0f;
};

using FragmentShader = std::function<ShadeResult(const FragmentInput&)>;

struct RasterState {
    CullMode cull = CullMode::Back;
    bool depthWrite = true;
    bool blend = false;
};

class Rasterizer {
public:
    void resize(int width, int height);
    void clear(Vec3 linearColor, float depthValue = 1.0f);

    // Clip-space triangles. Pixels are sampled at (x+0.5, y+0.5); depth uses
    // LessEqual. Coverage edges are inclusive, which is deterministic.
    void drawTriangle(const RasterVertex& a, const RasterVertex& b, const RasterVertex& c,
                      const RasterState& state, const FragmentShader& shader);
    void drawWireTriangle(const RasterVertex& a, const RasterVertex& b, const RasterVertex& c,
                          CullMode cull, Vec3 linearColor);

    int width() const { return width_; }
    int height() const { return height_; }
    const std::vector<Vec3>& colorBuffer() const { return color_; }
    const std::vector<float>& depthBuffer() const { return depth_; }

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<Vec3> color_;
    std::vector<float> depth_;
};

class DepthRasterizer {
public:
    void resize(int width, int height);
    void clear(float value = 1.0f);
    void drawTriangle(const Mat4& lightViewProjection, Vec3 a, Vec3 b, Vec3 c,
                      CullMode cull);

    int width() const { return width_; }
    int height() const { return height_; }
    const std::vector<float>& depthBuffer() const { return depth_; }

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<float> depth_;
};

}  // namespace cv

