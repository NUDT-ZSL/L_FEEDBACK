#include "render.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace rigskin {
namespace {

struct ProjectedPoint {
    double x = 0.0;
    double y = 0.0;
    double depth = 0.0;
};

ProjectedPoint project(const Camera& camera, const Vec3& point) {
    const Vec3 relative = point - camera.position;
    ProjectedPoint projected;
    projected.x =
        (dot(relative, camera.right) - camera.centerX) / camera.viewWidth + 0.5;
    projected.y =
        0.5 - (dot(relative, camera.up) - camera.centerY) / camera.viewHeight;
    projected.depth = dot(relative, camera.forward);
    return projected;
}

std::uint8_t shadeChannel(double base, double intensity) {
    const double value = base * intensity;
    const double clamped = std::max(0.0, std::min(255.0, value));
    return static_cast<std::uint8_t>(clamped + 0.5);
}

}  // namespace

Camera makeCamera(const Model& model, int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("camera dimensions must be positive");
    }
    if (!model.bindBounds.valid) {
        throw std::runtime_error("cannot build a camera for empty bounds");
    }

    const Vec3 center = model.bindBounds.center();
    const Vec3 extent = model.bindBounds.maximum - model.bindBounds.minimum;
    const double radius = std::max(extent.x, std::max(extent.y, extent.z));
    if (radius <= 1e-12) {
        throw std::runtime_error("model bounds are degenerate");
    }

    Camera camera;
    const Vec3 viewDirection = normalized({0.46, 0.20, 1.0});
    camera.position = center + viewDirection * (radius * 3.0 + 1.0);
    camera.forward = normalized(center - camera.position);
    camera.right = normalized(cross(camera.forward, {0.0, 1.0, 0.0}));
    camera.up = normalized(cross(camera.right, camera.forward));

    double minimumX = std::numeric_limits<double>::infinity();
    double maximumX = -std::numeric_limits<double>::infinity();
    double minimumY = std::numeric_limits<double>::infinity();
    double maximumY = -std::numeric_limits<double>::infinity();
    for (const MeshVertex& vertex : model.vertices) {
        const Vec3 relative = vertex.bindPosition - camera.position;
        const double x = dot(relative, camera.right);
        const double y = dot(relative, camera.up);
        minimumX = std::min(minimumX, x);
        maximumX = std::max(maximumX, x);
        minimumY = std::min(minimumY, y);
        maximumY = std::max(maximumY, y);
    }

    const double aspect =
        static_cast<double>(width) / static_cast<double>(height);
    const double rangeX = std::max(1e-6, maximumX - minimumX);
    const double rangeY = std::max(1e-6, maximumY - minimumY);
    camera.viewWidth = std::max(rangeX * 1.12, rangeY * 1.12 * aspect);
    camera.viewHeight = camera.viewWidth / aspect;
    camera.centerX = (minimumX + maximumX) * 0.5;
    camera.centerY = (minimumY + maximumY) * 0.5;
    return camera;
}

RenderedImage renderMesh(
    const Model& model,
    const DeformedMesh& mesh,
    const Camera& camera,
    int width,
    int height) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("render dimensions must be positive");
    }
    if (mesh.positions.size() != model.vertices.size() ||
        mesh.faceNormals.size() != model.triangles.size()) {
        throw std::runtime_error("render mesh shape does not match model");
    }

    RenderedImage image;
    image.width = width;
    image.height = height;
    image.rgb.resize(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);

    const double widthScale = 6.0 / static_cast<double>(width);
    const double heightScale = 5.0 / static_cast<double>(height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t offset =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(x)) *
                3U;
            image.rgb[offset + 0] =
                static_cast<std::uint8_t>(14.0 + x * widthScale);
            image.rgb[offset + 1] =
                static_cast<std::uint8_t>(18.0 + y * heightScale);
            image.rgb[offset + 2] =
                static_cast<std::uint8_t>(25.0 + (x + y) * heightScale * 0.5);
        }
    }

    std::vector<double> depthBuffer(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
        std::numeric_limits<double>::infinity());

    const Vec3 lightDirection = normalized({-0.42, 0.82, 0.63});
    constexpr double kBarycentricEpsilon = 1e-10;

    for (std::size_t triangleIndex = 0; triangleIndex < model.triangles.size();
         ++triangleIndex) {
        const Triangle& triangle = model.triangles[triangleIndex];
        const Vec3& faceNormal = mesh.faceNormals[triangleIndex];
        if (lengthSquared(faceNormal) <= 1e-24) {
            continue;
        }

        // Orthographic fixed camera: triangles facing away from the camera
        // are culled. Front-facing triangles have normals opposing the view.
        if (dot(faceNormal, camera.forward) >= -1e-12) {
            continue;
        }

        const ProjectedPoint a = project(camera, mesh.positions[triangle.a]);
        const ProjectedPoint b = project(camera, mesh.positions[triangle.b]);
        const ProjectedPoint c = project(camera, mesh.positions[triangle.c]);

        const double pixelA0 = a.x * static_cast<double>(width);
        const double pixelA1 = a.y * static_cast<double>(height);
        const double pixelB0 = b.x * static_cast<double>(width);
        const double pixelB1 = b.y * static_cast<double>(height);
        const double pixelC0 = c.x * static_cast<double>(width);
        const double pixelC1 = c.y * static_cast<double>(height);

        const double area =
            (pixelB0 - pixelA0) * (pixelC1 - pixelA1) -
            (pixelB1 - pixelA1) * (pixelC0 - pixelA0);
        if (std::abs(area) <= kBarycentricEpsilon) {
            continue;
        }

        int minimumX = static_cast<int>(
            std::floor(std::min(pixelA0, std::min(pixelB0, pixelC0))));
        int maximumX = static_cast<int>(
            std::ceil(std::max(pixelA0, std::max(pixelB0, pixelC0))));
        int minimumY = static_cast<int>(
            std::floor(std::min(pixelA1, std::min(pixelB1, pixelC1))));
        int maximumY = static_cast<int>(
            std::ceil(std::max(pixelA1, std::max(pixelB1, pixelC1))));
        minimumX = std::max(0, minimumX);
        maximumX = std::min(width - 1, maximumX);
        minimumY = std::max(0, minimumY);
        maximumY = std::min(height - 1, maximumY);
        if (minimumX > maximumX || minimumY > maximumY) {
            continue;
        }

        const double diffuse =
            std::max(0.0, dot(faceNormal, lightDirection));
        const double intensity = 0.24 + 0.76 * diffuse;
        const std::uint8_t red = shadeChannel(184.0, intensity);
        const std::uint8_t green = shadeChannel(151.0, intensity);
        const std::uint8_t blue = shadeChannel(116.0, intensity);

        for (int y = minimumY; y <= maximumY; ++y) {
            const double sampleY = static_cast<double>(y) + 0.5;
            for (int x = minimumX; x <= maximumX; ++x) {
                const double sampleX = static_cast<double>(x) + 0.5;
                const double weightA =
                    ((pixelB0 - sampleX) * (pixelC1 - sampleY) -
                     (pixelB1 - sampleY) * (pixelC0 - sampleX)) /
                    area;
                const double weightB =
                    ((pixelC0 - sampleX) * (pixelA1 - sampleY) -
                     (pixelC1 - sampleY) * (pixelA0 - sampleX)) /
                    area;
                const double weightC = 1.0 - weightA - weightB;
                if (weightA < -kBarycentricEpsilon ||
                    weightB < -kBarycentricEpsilon ||
                    weightC < -kBarycentricEpsilon) {
                    continue;
                }

                const double depth =
                    weightA * a.depth + weightB * b.depth + weightC * c.depth;
                const std::size_t pixelIndex =
                    static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                    static_cast<std::size_t>(x);
                if (!(depth < depthBuffer[pixelIndex])) {
                    continue;
                }
                depthBuffer[pixelIndex] = depth;
                const std::size_t offset = pixelIndex * 3U;
                image.rgb[offset + 0] = red;
                image.rgb[offset + 1] = green;
                image.rgb[offset + 2] = blue;
            }
        }
    }

    return image;
}

}  // namespace rigskin
