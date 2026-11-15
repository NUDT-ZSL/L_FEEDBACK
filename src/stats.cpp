#include "stats.h"

#include <algorithm>
#include <cmath>

namespace pbr {

RenderStatistics computeStatistics(const Scene& scene, const Image& image) {
    RenderStatistics statistics;
    statistics.materials.resize(scene.materials.size());
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        statistics.materials[i].id = static_cast<int>(i);
        statistics.materials[i].name = scene.materials[i].name;
    }

    std::vector<Vec3> directionSums(scene.materials.size());
    std::vector<double> directionWeights(scene.materials.size(), 0.0);
    for (size_t pixel = 0; pixel < image.pixels.size(); ++pixel) {
        if (pixel >= image.materialIds.size()) {
            break;
        }
        const int materialId = image.materialIds[pixel];
        if (materialId < 0 || materialId >= static_cast<int>(statistics.materials.size())) {
            continue;
        }
        MaterialStats& material = statistics.materials[static_cast<size_t>(materialId)];
        const Vec3& color = image.pixels[pixel];
        ++material.pixelCount;
        material.meanColor += color;

        const float lum = std::max(luminance(color), 0.0f);
        const float logLuminance = std::log2(std::max(lum, 1.0e-6f));
        const int bin = std::clamp(
            static_cast<int>(std::floor(logLuminance + 8.0f)), 0, kHighlightHistogramBins - 1);
        material.highlightHistogram[static_cast<size_t>(bin)] += 1.0f;

        if (pixel < image.reflectionDirections.size()) {
            const double weight = 0.001 + static_cast<double>(lum) * static_cast<double>(lum);
            directionSums[static_cast<size_t>(materialId)] +=
                image.reflectionDirections[pixel] * static_cast<float>(weight);
            directionWeights[static_cast<size_t>(materialId)] += weight;
        }
    }

    for (size_t i = 0; i < statistics.materials.size(); ++i) {
        MaterialStats& material = statistics.materials[i];
        if (material.pixelCount > 0) {
            material.meanColor =
                material.meanColor / static_cast<float>(material.pixelCount);
            for (float& bin : material.highlightHistogram) {
                bin /= static_cast<float>(material.pixelCount);
            }
        }
        if (directionWeights[i] > 1.0e-12) {
            const Vec3 direction = normalize(directionSums[i]);
            if (lengthSquared(direction) > 0.5f) {
                material.specularDirection = direction;
            }
        }
    }
    return statistics;
}

}  // namespace pbr
