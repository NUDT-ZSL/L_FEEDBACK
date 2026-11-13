#pragma once

#include "model.hpp"
#include "skinning.hpp"

#include <cstdint>
#include <vector>

namespace rigskin {

struct Camera {
    Vec3 position;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    double centerX = 0.0;
    double centerY = 0.0;
    double viewWidth = 1.0;
    double viewHeight = 1.0;
};

struct RenderedImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;
};

Camera makeCamera(const Model& model, int width, int height);
RenderedImage renderMesh(
    const Model& model,
    const DeformedMesh& mesh,
    const Camera& camera,
    int width,
    int height);

}  // namespace rigskin

