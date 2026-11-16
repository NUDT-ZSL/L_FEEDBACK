#pragma once

#include "scene.hpp"

#include <optional>
#include <string>
#include <vector>

namespace tilegen {

struct Bounds {
    double minX = 0.0;
    double minY = 0.0;
    double maxX = 0.0;
    double maxY = 0.0;
    bool valid = false;

    void include(Vec2 point);
    void include(const Bounds& other);
    Bounds expanded(double amount) const;
    bool intersects(const Bounds& other) const;
    double width() const noexcept { return maxX - minX; }
    double height() const noexcept { return maxY - minY; }
};

struct RenderSettings {
    int samplesPerAxis = 4;
};

struct RegionRenderResult {
    Image image;
    std::vector<std::string> contributingLayerIds;
    std::vector<Bounds> contributingLayerBounds;
};

PremulPixel blendPremultiplied(const PremulPixel& destination,
                               const PremulPixel& source, BlendMode mode);
void compositePremultiplied(Image& destination, int x, int y,
                            const PremulPixel& source, BlendMode mode = BlendMode::Normal);

Bounds layerWorldBounds(const Layer& layer);
Bounds primitiveWorldBounds(const Layer& layer, const Primitive& primitive);

RegionRenderResult renderSceneRegion(const Scene& scene, double levelScale,
                                     const Bounds& worldBounds, int outputWidth,
                                     int outputHeight, const RenderSettings& settings,
                                     bool includeBackground);

std::vector<Rgba8> imageToStraightRgba8(const Image& image);
Image straightRgba8ToImage(const std::vector<Rgba8>& pixels, int width, int height);

}  // namespace tilegen
