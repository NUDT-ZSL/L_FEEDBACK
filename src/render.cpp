#include "render.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tilegen {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

double clamp01(double value) {
    return std::clamp(value, 0.0, 1.0);
}

double maximumTransformScale(const Affine& transform) {
    return std::max(std::hypot(transform.a, transform.b),
                    std::hypot(transform.c, transform.d));
}

PremulPixel scalePixel(const PremulPixel& pixel, double factor) {
    return {static_cast<float>(pixel.r * factor),
            static_cast<float>(pixel.g * factor),
            static_cast<float>(pixel.b * factor),
            static_cast<float>(pixel.a * factor)};
}

PremulPixel addPixels(const PremulPixel& left, const PremulPixel& right) {
    return {left.r + right.r, left.g + right.g, left.b + right.b,
            left.a + right.a};
}

bool hasVisibleAlpha(const Image& image) {
    for (const PremulPixel& pixel : image.pixels) {
        if (pixel.a > 1e-7F) {
            return true;
        }
    }
    return false;
}

struct DevicePolygon {
    std::vector<Vec2> points;
    Bounds bounds;
};

DevicePolygon transformPoints(const std::vector<Vec2>& points,
                              const Affine& transform) {
    DevicePolygon polygon;
    polygon.points.reserve(points.size());
    for (const Vec2& point : points) {
        const Vec2 device = transform.apply(point);
        polygon.points.push_back(device);
        polygon.bounds.include(device);
    }
    return polygon;
}

bool pointInRings(const std::vector<std::vector<Vec2>>& rings, Vec2 point,
                  FillRule rule) {
    int winding = 0;
    int crossingCount = 0;
    for (const auto& ring : rings) {
        if (ring.size() < 3) {
            continue;
        }
        for (std::size_t i = 0; i < ring.size(); ++i) {
            const Vec2 a = ring[i];
            const Vec2 b = ring[(i + 1) % ring.size()];
            if ((a.y > point.y) == (b.y > point.y)) {
                continue;
            }
            const double intersectionX =
                a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if (intersectionX <= point.x) {
                continue;
            }
            ++crossingCount;
            winding += b.y > a.y ? 1 : -1;
        }
    }
    return rule == FillRule::NonZero ? winding != 0 : (crossingCount & 1) != 0;
}

double cross(Vec2 left, Vec2 right) {
    return left.x * right.y - left.y * right.x;
}

Vec2 subtract(Vec2 left, Vec2 right) {
    return {left.x - right.x, left.y - right.y};
}

Vec2 add(Vec2 left, Vec2 right) {
    return {left.x + right.x, left.y + right.y};
}

Vec2 multiply(Vec2 vector, double scalar) {
    return {vector.x * scalar, vector.y * scalar};
}

double dot(Vec2 left, Vec2 right) {
    return left.x * right.x + left.y * right.y;
}

double length(Vec2 vector) {
    return std::hypot(vector.x, vector.y);
}

Vec2 normalized(Vec2 vector) {
    const double magnitude = length(vector);
    if (magnitude == 0.0) {
        return {0.0, 0.0};
    }
    return {vector.x / magnitude, vector.y / magnitude};
}

double distanceToSegment(Vec2 point, Vec2 a, Vec2 b, double& parameter) {
    const Vec2 direction = subtract(b, a);
    const double lengthSquared = dot(direction, direction);
    if (lengthSquared == 0.0) {
        parameter = 0.0;
        return length(subtract(point, a));
    }
    parameter = dot(subtract(point, a), direction) / lengthSquared;
    const double clamped = clamp01(parameter);
    const Vec2 closest = add(a, multiply(direction, clamped));
    return length(subtract(point, closest));
}

bool pointInTriangle(Vec2 point, Vec2 a, Vec2 b, Vec2 c) {
    const double d1 = cross(subtract(b, a), subtract(point, a));
    const double d2 = cross(subtract(c, b), subtract(point, b));
    const double d3 = cross(subtract(a, c), subtract(point, c));
    const bool hasNegative = d1 < -1e-12 || d2 < -1e-12 || d3 < -1e-12;
    const bool hasPositive = d1 > 1e-12 || d2 > 1e-12 || d3 > 1e-12;
    return !(hasNegative && hasPositive);
}

bool pointInSquareCap(Vec2 point, Vec2 endpoint, Vec2 direction, double halfWidth) {
    const Vec2 unit = normalized(direction);
    const Vec2 relative = subtract(point, endpoint);
    const double along = dot(relative, unit);
    const double perpendicular = std::abs(cross(relative, unit));
    return along <= 0.0 && along >= -halfWidth && perpendicular <= halfWidth;
}

bool pointInMiterJoin(Vec2 point, Vec2 vertex, Vec2 incomingDirection,
                      Vec2 outgoingDirection, double halfWidth, double miterLimit) {
    const double turn = cross(incomingDirection, outgoingDirection);
    if (std::abs(turn) < 1e-12) {
        return false;
    }
    Vec2 normal;
    if (turn > 0.0) {
        normal = {incomingDirection.y, -incomingDirection.x};
    } else {
        normal = {-incomingDirection.y, incomingDirection.x};
    }
    const Vec2 outgoingNormal =
        turn > 0.0 ? Vec2{outgoingDirection.y, -outgoingDirection.x}
                   : Vec2{-outgoingDirection.y, outgoingDirection.x};
    const Vec2 firstOuter = add(vertex, multiply(normal, halfWidth));
    const Vec2 secondOuter = add(vertex, multiply(outgoingNormal, halfWidth));

    const Vec2 firstDirection = normalized(incomingDirection);
    const Vec2 secondDirection = normalized(outgoingDirection);
    const double denominator = cross(firstDirection, secondDirection);
    if (std::abs(denominator) < 1e-12) {
        return false;
    }
    const double t =
        cross(subtract(secondOuter, firstOuter), secondDirection) / denominator;
    const Vec2 miter = add(firstOuter, multiply(firstDirection, t));
    if (length(subtract(miter, vertex)) > halfWidth * miterLimit + 1e-9) {
        return pointInTriangle(point, vertex, firstOuter, secondOuter);
    }
    return pointInTriangle(point, vertex, firstOuter, miter) ||
           pointInTriangle(point, vertex, secondOuter, miter);
}

bool pointInStroke(const std::vector<Vec2>& points, double halfWidth, CapStyle cap,
                   JoinStyle join, double miterLimit, bool closed, Vec2 sample) {
    if (points.empty() || halfWidth <= 0.0) {
        return false;
    }
    if (points.size() == 1) {
        return length(subtract(sample, points[0])) <= halfWidth;
    }
    const std::size_t segmentCount = closed ? points.size() : points.size() - 1;
    for (std::size_t i = 0; i < segmentCount; ++i) {
        const Vec2 a = points[i];
        const Vec2 b = points[(i + 1) % points.size()];
        double parameter = 0.0;
        const double distance = distanceToSegment(sample, a, b, parameter);
        if (distance > halfWidth + 1e-12) {
            continue;
        }
        if (closed || (i > 0 && i + 1 < segmentCount)) {
            return true;
        }
        if (i == 0 && parameter < 0.0) {
            continue;
        }
        if (i + 1 == segmentCount && parameter > 1.0) {
            continue;
        }
        return true;
    }

    if (!closed) {
        const Vec2 startDirection = subtract(points[1], points[0]);
        const Vec2 endDirection =
            subtract(points[points.size() - 1], points[points.size() - 2]);
        if (cap == CapStyle::Round) {
            if (length(subtract(sample, points.front())) <= halfWidth ||
                length(subtract(sample, points.back())) <= halfWidth) {
                return true;
            }
        } else if (cap == CapStyle::Square) {
            if (pointInSquareCap(sample, points.front(), startDirection, halfWidth) ||
                pointInSquareCap(sample, points.back(),
                                 multiply(endDirection, -1.0), halfWidth)) {
                return true;
            }
        }
    }

    const std::size_t joinCount =
        closed ? points.size() : (points.size() >= 3 ? points.size() - 2 : 0);
    for (std::size_t i = 0; i < joinCount; ++i) {
        const std::size_t vertexIndex = i + 1;
        const Vec2 vertex = points[vertexIndex % points.size()];
        const Vec2 previous = points[(vertexIndex + points.size() - 1) % points.size()];
        const Vec2 next = points[(vertexIndex + 1) % points.size()];
        const Vec2 incoming = subtract(vertex, previous);
        const Vec2 outgoing = subtract(next, vertex);
        if (length(incoming) == 0.0 || length(outgoing) == 0.0) {
            continue;
        }
        if (join == JoinStyle::Round) {
            if (length(subtract(sample, vertex)) <= halfWidth) {
                return true;
            }
        } else if (join == JoinStyle::Miter) {
            if (pointInMiterJoin(sample, vertex, incoming, outgoing, halfWidth,
                                 miterLimit)) {
                return true;
            }
        } else {
            const double turn = cross(incoming, outgoing);
            if (turn == 0.0) {
                continue;
            }
            Vec2 firstNormal =
                turn > 0.0 ? Vec2{incoming.y, -incoming.x}
                           : Vec2{-incoming.y, incoming.x};
            Vec2 secondNormal =
                turn > 0.0 ? Vec2{outgoing.y, -outgoing.x}
                           : Vec2{-outgoing.y, outgoing.x};
            firstNormal = multiply(normalized(firstNormal), halfWidth);
            secondNormal = multiply(normalized(secondNormal), halfWidth);
            if (pointInTriangle(sample, vertex, add(vertex, firstNormal),
                                add(vertex, secondNormal))) {
                return true;
            }
        }
    }
    return false;
}

std::vector<Vec2> makeCircle(Vec2 center, double radius, int segments = 128) {
    std::vector<Vec2> points;
    points.reserve(static_cast<std::size_t>(segments));
    for (int i = 0; i < segments; ++i) {
        const double angle = 2.0 * kPi * static_cast<double>(i) /
                             static_cast<double>(segments);
        points.push_back({center.x + std::cos(angle) * radius,
                          center.y + std::sin(angle) * radius});
    }
    return points;
}

template <typename Sampler>
void rasterize(Image& destination, Bounds bounds, int samplesPerAxis,
               const Sampler& sampler) {
    if (!bounds.valid || samplesPerAxis <= 0) {
        return;
    }
    const int minX = std::max(0, static_cast<int>(std::floor(bounds.minX)));
    const int minY = std::max(0, static_cast<int>(std::floor(bounds.minY)));
    const int maxX = std::min(destination.width - 1,
                              static_cast<int>(std::ceil(bounds.maxX)));
    const int maxY = std::min(destination.height - 1,
                              static_cast<int>(std::ceil(bounds.maxY)));
    if (minX > maxX || minY > maxY) {
        return;
    }
    const double sampleCount =
        static_cast<double>(samplesPerAxis) * samplesPerAxis;
    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            PremulPixel sum;
            for (int sy = 0; sy < samplesPerAxis; ++sy) {
                for (int sx = 0; sx < samplesPerAxis; ++sx) {
                    const Vec2 sample{
                        static_cast<double>(x) +
                            (static_cast<double>(sx) + 0.5) / samplesPerAxis,
                        static_cast<double>(y) +
                            (static_cast<double>(sy) + 0.5) / samplesPerAxis};
                    const auto source = sampler(sample);
                    if (source) {
                        sum = addPixels(sum, *source);
                    }
                }
            }
            if (sum.a > 0.0F || sum.r > 0.0F || sum.g > 0.0F || sum.b > 0.0F) {
                const PremulPixel source = scalePixel(sum, 1.0 / sampleCount);
                compositePremultiplied(destination, x, y, source);
            }
        }
    }
}

PremulPixel colorForFill(const Scene& scene, const FillStyle& style, Bounds shapeBounds,
                         Vec2 sample) {
    const double alpha = clamp01(style.color.a * style.opacity);
    double red = style.color.r;
    double green = style.color.g;
    double blue = style.color.b;
    if (!style.textureKey.empty()) {
        auto it = scene.textures.find(style.textureKey);
        if (it == scene.textures.end()) {
            throw std::runtime_error("texture was not loaded: " + style.textureKey);
        }
        const double width = std::max(shapeBounds.width(), 1e-12);
        const double height = std::max(shapeBounds.height(), 1e-12);
        const double tx = (sample.x - shapeBounds.minX) / width;
        const double ty = (sample.y - shapeBounds.minY) / height;
        const double u = style.uv.u0 + tx * (style.uv.u1 - style.uv.u0);
        const double v = style.uv.v0 + ty * (style.uv.v1 - style.uv.v0);
        const Rgba8 texel = it->second.sampleBilinear(u, v, style.repeat);
        const double textureAlpha = static_cast<double>(texel.a) / 255.0;
        red *= static_cast<double>(texel.r) / 255.0;
        green *= static_cast<double>(texel.g) / 255.0;
        blue *= static_cast<double>(texel.b) / 255.0;
        const double combinedAlpha = clamp01(alpha * textureAlpha);
        return {static_cast<float>(red * combinedAlpha),
                static_cast<float>(green * combinedAlpha),
                static_cast<float>(blue * combinedAlpha),
                static_cast<float>(combinedAlpha)};
    }
    return {static_cast<float>(red * alpha), static_cast<float>(green * alpha),
            static_cast<float>(blue * alpha), static_cast<float>(alpha)};
}

PremulPixel colorForStroke(const StrokeStyle& style) {
    const double alpha = clamp01(style.color.a * style.opacity);
    return {static_cast<float>(style.color.r * alpha),
            static_cast<float>(style.color.g * alpha),
            static_cast<float>(style.color.b * alpha),
            static_cast<float>(alpha)};
}

void drawPolygonFill(const Scene& scene, Image& image,
                     const std::vector<std::vector<Vec2>>& rings,
                     FillRule fillRule, const FillStyle& style, int samplesPerAxis) {
    Bounds bounds;
    for (const auto& ring : rings) {
        for (Vec2 point : ring) {
            bounds.include(point);
        }
    }
    if (!bounds.valid) {
        return;
    }
    const bool texture = !style.textureKey.empty();
    const PremulPixel constant =
        texture ? PremulPixel{}
                : colorForFill(scene, style, bounds, {bounds.minX, bounds.minY});
    rasterize(image, bounds.expanded(1.0), samplesPerAxis, [&](Vec2 sample) {
        if (!pointInRings(rings, sample, fillRule)) {
            return std::optional<PremulPixel>{};
        }
        if (texture) {
            return std::optional<PremulPixel>{
                colorForFill(scene, style, bounds, sample)};
        }
        return std::optional<PremulPixel>{constant};
    });
}

void drawStroke(Image& image, const std::vector<Vec2>& points, bool closed,
                const StrokeStyle& style, double deviceWidth, int samplesPerAxis) {
    if (style.width <= 0.0 || deviceWidth <= 0.0 || points.empty()) {
        return;
    }
    Bounds bounds;
    for (Vec2 point : points) {
        bounds.include(point);
    }
    const double halfWidth = deviceWidth / 2.0;
    const double extra = halfWidth * std::max(style.miterLimit, 1.0) + 1.0;
    const PremulPixel color = colorForStroke(style);
    rasterize(image, bounds.expanded(extra), samplesPerAxis, [&](Vec2 sample) {
        if (pointInStroke(points, halfWidth, style.cap, style.join,
                          style.miterLimit, closed, sample)) {
            return std::optional<PremulPixel>{color};
        }
        return std::optional<PremulPixel>{};
    });
}

void drawPrimitive(const Scene& scene, Image& image, const Primitive& primitive,
                   const Affine& worldToDevice,
                   int samplesPerAxis) {
    if (primitive.kind == PrimitiveKind::Polygon) {
        std::vector<std::vector<Vec2>> deviceRings;
        deviceRings.reserve(primitive.rings.size());
        for (const auto& ring : primitive.rings) {
            deviceRings.push_back(transformPoints(ring, worldToDevice).points);
        }
        if (primitive.fill) {
            drawPolygonFill(scene, image, deviceRings, primitive.fillRule,
                            *primitive.fill, samplesPerAxis);
        }
        if (primitive.stroke) {
            for (const auto& ring : deviceRings) {
                if (ring.size() < 2) {
                    continue;
                }
                const double deviceWidth =
                    primitive.stroke->width *
                    std::sqrt(std::abs(worldToDevice.determinant()));
                drawStroke(image, ring, true, *primitive.stroke,
                           deviceWidth, samplesPerAxis);
            }
        }
    } else if (primitive.kind == PrimitiveKind::Polyline) {
        const DevicePolygon device =
            transformPoints(primitive.points, worldToDevice);
        if (primitive.stroke) {
            const double deviceWidth =
                primitive.stroke->width *
                std::sqrt(std::abs(worldToDevice.determinant()));
            drawStroke(image, device.points, false, *primitive.stroke,
                       deviceWidth, samplesPerAxis);
        }
        if (primitive.fill) {
            drawPolygonFill(scene, image, {device.points}, primitive.fillRule,
                            *primitive.fill, samplesPerAxis);
        }
    } else {
        const std::vector<Vec2> circle =
            makeCircle(primitive.center, primitive.radius);
        const std::vector<Vec2> deviceCircle =
            transformPoints(circle, worldToDevice).points;
        if (primitive.fill) {
            drawPolygonFill(scene, image, {deviceCircle}, primitive.fillRule,
                            *primitive.fill, samplesPerAxis);
        }
        if (primitive.stroke) {
            const double deviceWidth =
                primitive.stroke->width *
                std::sqrt(std::abs(worldToDevice.determinant()));
            drawStroke(image, deviceCircle, true, *primitive.stroke, deviceWidth,
                       samplesPerAxis);
        }
    }
}

}  // namespace

void Bounds::include(Vec2 point) {
    if (!valid) {
        minX = maxX = point.x;
        minY = maxY = point.y;
        valid = true;
    } else {
        minX = std::min(minX, point.x);
        minY = std::min(minY, point.y);
        maxX = std::max(maxX, point.x);
        maxY = std::max(maxY, point.y);
    }
}

void Bounds::include(const Bounds& other) {
    if (!other.valid) {
        return;
    }
    include(Vec2{other.minX, other.minY});
    include(Vec2{other.maxX, other.maxY});
}

Bounds Bounds::expanded(double amount) const {
    if (!valid) {
        return *this;
    }
    return {minX - amount, minY - amount, maxX + amount, maxY + amount, true};
}

bool Bounds::intersects(const Bounds& other) const {
    return valid && other.valid && minX <= other.maxX && maxX >= other.minX &&
           minY <= other.maxY && maxY >= other.minY;
}

PremulPixel blendPremultiplied(const PremulPixel& destination,
                               const PremulPixel& source, BlendMode mode) {
    const double sa = clamp01(source.a);
    const double da = clamp01(destination.a);
    const double sr = std::max(0.0, static_cast<double>(source.r));
    const double sg = std::max(0.0, static_cast<double>(source.g));
    const double sb = std::max(0.0, static_cast<double>(source.b));
    const double dr = std::max(0.0, static_cast<double>(destination.r));
    const double dg = std::max(0.0, static_cast<double>(destination.g));
    const double db = std::max(0.0, static_cast<double>(destination.b));
    const double outA = clamp01(sa + da - sa * da);

    double outR = 0.0;
    double outG = 0.0;
    double outB = 0.0;
    if (mode == BlendMode::Normal) {
        outR = sr + dr * (1.0 - sa);
        outG = sg + dg * (1.0 - sa);
        outB = sb + db * (1.0 - sa);
    } else {
        const double csR = sa > 0.0 ? sr / sa : 0.0;
        const double csG = sa > 0.0 ? sg / sa : 0.0;
        const double csB = sa > 0.0 ? sb / sa : 0.0;
        const double cbR = da > 0.0 ? dr / da : 0.0;
        const double cbG = da > 0.0 ? dg / da : 0.0;
        const double cbB = da > 0.0 ? db / da : 0.0;
        if (mode == BlendMode::Add) {
            const double br = clamp01(cbR + csR);
            const double bg = clamp01(cbG + csG);
            const double bb = clamp01(cbB + csB);
            outR = (1.0 - sa) * dr + (1.0 - da) * sr + sa * da * br;
            outG = (1.0 - sa) * dg + (1.0 - da) * sg + sa * da * bg;
            outB = (1.0 - sa) * db + (1.0 - da) * sb + sa * da * bb;
        } else {
            const double br = clamp01(cbR * csR);
            const double bg = clamp01(cbG * csG);
            const double bb = clamp01(cbB * csB);
            outR = (1.0 - sa) * dr + (1.0 - da) * sr + sa * da * br;
            outG = (1.0 - sa) * dg + (1.0 - da) * sg + sa * da * bg;
            outB = (1.0 - sa) * db + (1.0 - da) * sb + sa * da * bb;
        }
    }
    return {static_cast<float>(clamp01(outR)), static_cast<float>(clamp01(outG)),
            static_cast<float>(clamp01(outB)), static_cast<float>(outA)};
}

void compositePremultiplied(Image& destination, int x, int y,
                            const PremulPixel& source, BlendMode mode) {
    if (x < 0 || y < 0 || x >= destination.width || y >= destination.height) {
        return;
    }
    destination.at(x, y) =
        blendPremultiplied(destination.at(x, y), source, mode);
}

Bounds primitiveWorldBounds(const Layer& layer, const Primitive& primitive) {
    Bounds bounds;
    auto includePoint = [&bounds, &layer](Vec2 point) {
        bounds.include(layer.transform.apply(point));
    };
    if (primitive.kind == PrimitiveKind::Polygon) {
        for (const auto& ring : primitive.rings) {
            for (Vec2 point : ring) {
                includePoint(point);
            }
        }
    } else if (primitive.kind == PrimitiveKind::Polyline) {
        for (Vec2 point : primitive.points) {
            includePoint(point);
        }
    } else {
        for (Vec2 point : makeCircle(primitive.center, primitive.radius)) {
            includePoint(point);
        }
    }
    if (!bounds.valid) {
        return bounds;
    }
    if (primitive.stroke && primitive.stroke->width > 0.0) {
        const double transformScale = maximumTransformScale(layer.transform);
        const double expansion =
            primitive.stroke->width * std::max(transformScale, 1e-12) *
            std::max(primitive.stroke->miterLimit, 1.0) / 2.0;
        bounds = bounds.expanded(expansion);
    }
    return bounds;
}

Bounds layerWorldBounds(const Layer& layer) {
    Bounds bounds;
    for (const Primitive& primitive : layer.primitives) {
        bounds.include(primitiveWorldBounds(layer, primitive));
    }
    return bounds;
}

RegionRenderResult renderSceneRegion(const Scene& scene, double levelScale,
                                     const Bounds& worldBounds, int outputWidth,
                                     int outputHeight, const RenderSettings& settings,
                                     bool includeBackground) {
    if (!(levelScale > 0.0) || !std::isfinite(levelScale)) {
        throw std::runtime_error("level scale must be finite and positive");
    }
    if (outputWidth <= 0 || outputHeight <= 0 || !worldBounds.valid) {
        throw std::runtime_error("invalid render region");
    }
    if (settings.samplesPerAxis < 1 || settings.samplesPerAxis > 16) {
        throw std::runtime_error("samples per axis must be in [1, 16]");
    }

    RegionRenderResult result;
    result.image = Image(outputWidth, outputHeight);
    if (includeBackground) {
        const PremulPixel background{
            static_cast<float>(scene.background.r * scene.background.a),
            static_cast<float>(scene.background.g * scene.background.a),
            static_cast<float>(scene.background.b * scene.background.a),
            static_cast<float>(scene.background.a)};
        if (background.a > 0.0F) {
            std::fill(result.image.pixels.begin(), result.image.pixels.end(),
                      background);
        }
    }

    if (worldBounds.width() <= 0.0 || worldBounds.height() <= 0.0) {
        throw std::runtime_error("render region has zero area");
    }
    const double scaleX =
        static_cast<double>(outputWidth) / worldBounds.width();
    const double scaleY =
        static_cast<double>(outputHeight) / worldBounds.height();
    const Affine worldToDevice{
        scaleX, 0.0, 0.0, scaleY, -worldBounds.minX * scaleX,
        -worldBounds.minY * scaleY};
    // Tiles are square in world space, so the mapping above is normally
    // isotropic. It remains well-defined for explicitly non-square regions.

    for (const Layer& layer : scene.layers) {
        if (!layer.visible) {
            continue;
        }
        const Bounds bounds = layerWorldBounds(layer);
        if (!bounds.valid || !bounds.intersects(worldBounds)) {
            continue;
        }
        const Affine layerToDevice = worldToDevice * layer.transform;
        Image layerImage(outputWidth, outputHeight);
        for (const Primitive& primitive : layer.primitives) {
            const Bounds primitiveBounds = primitiveWorldBounds(layer, primitive);
            if (!primitiveBounds.valid || !primitiveBounds.intersects(worldBounds)) {
                continue;
            }
            drawPrimitive(scene, layerImage, primitive, layerToDevice,
                          settings.samplesPerAxis);
        }
        if (!hasVisibleAlpha(layerImage)) {
            continue;
        }
        result.contributingLayerIds.push_back(layer.id);
        result.contributingLayerBounds.push_back(bounds);
        for (int y = 0; y < outputHeight; ++y) {
            for (int x = 0; x < outputWidth; ++x) {
                PremulPixel source = scalePixel(layerImage.at(x, y), layer.opacity);
                if (source.a <= 0.0F && source.r <= 0.0F && source.g <= 0.0F &&
                    source.b <= 0.0F) {
                    continue;
                }
                compositePremultiplied(result.image, x, y, source, layer.blend);
            }
        }
    }
    return result;
}

std::vector<Rgba8> imageToStraightRgba8(const Image& image) {
    std::vector<Rgba8> output;
    output.reserve(static_cast<std::size_t>(image.width) *
                   static_cast<std::size_t>(image.height));
    for (const PremulPixel& pixel : image.pixels) {
        const double alpha = clamp01(pixel.a);
        const int alphaByte =
            static_cast<int>(std::lround(alpha * 255.0));
        Rgba8 outputPixel;
        if (alpha > 1e-12) {
            outputPixel.r = static_cast<std::uint8_t>(std::lround(
                clamp01(pixel.r / alpha) * 255.0));
            outputPixel.g = static_cast<std::uint8_t>(std::lround(
                clamp01(pixel.g / alpha) * 255.0));
            outputPixel.b = static_cast<std::uint8_t>(std::lround(
                clamp01(pixel.b / alpha) * 255.0));
        }
        outputPixel.a = static_cast<std::uint8_t>(
            std::clamp(alphaByte, 0, 255));
        output.push_back(outputPixel);
    }
    return output;
}

Image straightRgba8ToImage(const std::vector<Rgba8>& pixels, int width, int height) {
    if (width <= 0 || height <= 0 ||
        pixels.size() != static_cast<std::size_t>(width) *
                              static_cast<std::size_t>(height)) {
        throw std::runtime_error("invalid straight RGBA buffer");
    }
    Image image(width, height);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        const double alpha = static_cast<double>(pixels[i].a) / 255.0;
        image.pixels[i] = {
            static_cast<float>(static_cast<double>(pixels[i].r) / 255.0 * alpha),
            static_cast<float>(static_cast<double>(pixels[i].g) / 255.0 * alpha),
            static_cast<float>(static_cast<double>(pixels[i].b) / 255.0 * alpha),
            static_cast<float>(alpha)};
    }
    return image;
}

}  // namespace tilegen
