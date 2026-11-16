#pragma once

#include "image.hpp"

#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilegen {

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

struct Affine {
    double a = 1.0;
    double b = 0.0;
    double c = 0.0;
    double d = 1.0;
    double e = 0.0;
    double f = 0.0;

    Vec2 apply(Vec2 point) const;
    double determinant() const noexcept { return a * d - b * c; }
};

Affine operator*(const Affine& left, const Affine& right);

enum class FillRule { NonZero, EvenOdd };
enum class BlendMode { Normal, Add, Multiply };
enum class CapStyle { Butt, Round, Square };
enum class JoinStyle { Miter, Round, Bevel };
enum class PrimitiveKind { Polygon, Polyline, Circle };

struct Color {
    double r = 1.0;
    double g = 1.0;
    double b = 1.0;
    double a = 1.0;
};

struct UvRect {
    double u0 = 0.0;
    double v0 = 0.0;
    double u1 = 1.0;
    double v1 = 1.0;
};

struct FillStyle {
    Color color;
    double opacity = 1.0;
    std::string textureKey;
    UvRect uv;
    bool repeat = false;
};

struct StrokeStyle {
    double width = 1.0;
    Color color;
    double opacity = 1.0;
    CapStyle cap = CapStyle::Butt;
    JoinStyle join = JoinStyle::Miter;
    double miterLimit = 4.0;
};

struct Primitive {
    PrimitiveKind kind = PrimitiveKind::Polygon;
    std::vector<std::vector<Vec2>> rings;
    std::vector<Vec2> points;
    Vec2 center;
    double radius = 0.0;
    FillRule fillRule = FillRule::NonZero;
    std::optional<FillStyle> fill;
    std::optional<StrokeStyle> stroke;
};

struct Layer {
    std::string id;
    bool visible = true;
    double opacity = 1.0;
    BlendMode blend = BlendMode::Normal;
    FillRule fillRule = FillRule::NonZero;
    Affine transform;
    std::vector<Primitive> primitives;
};

struct Scene {
    int width = 0;
    int height = 0;
    Color background{0.0, 0.0, 0.0, 0.0};
    std::vector<Layer> layers;
    std::map<std::string, Texture> textures;
    std::string sourcePath;
    std::string sourceDirectory;
    std::uint32_t sourceCrc32 = 0;
    std::size_t sourceBytes = 0;
};

class SceneError : public std::runtime_error {
public:
    SceneError(std::string path, const std::string& message);

    const std::string& path() const noexcept { return path_; }

private:
    std::string path_;
};

Scene loadScene(const std::string& path);
Color premultiplyColor(const Color& color, double opacity);

}  // namespace tilegen
