#include "offline_renderer.hpp"

#include "image_io.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace skel {
namespace {

struct ClipVertex {
    glm::vec4 clip{0.0F};
    glm::vec3 worldPosition{0.0F};
    glm::vec3 worldNormal{0.0F};
    glm::vec3 color{0.0F};
};

struct ScreenVertex {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    float invW = 0.0F;
    glm::vec3 worldPosition{0.0F};
    glm::vec3 worldNormal{0.0F};
    glm::vec3 color{0.0F};
};

float edgeFunction(float ax, float ay, float bx, float by, float px,
                   float py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

ClipVertex interpolateClip(const ClipVertex& a, const ClipVertex& b,
                           float t) {
    ClipVertex result;
    result.clip = glm::mix(a.clip, b.clip, t);
    result.worldPosition = glm::mix(a.worldPosition, b.worldPosition, t);
    result.worldNormal = glm::mix(a.worldNormal, b.worldNormal, t);
    result.color = glm::mix(a.color, b.color, t);
    return result;
}

std::vector<ClipVertex> clipNearPlane(const std::vector<ClipVertex>& input) {
    std::vector<ClipVertex> output;
    if (input.empty()) {
        return output;
    }
    output.reserve(input.size() + 1U);
    ClipVertex previous = input.back();
    float previousDistance =
        previous.clip.z + previous.clip.w; // >= 0 is inside the near plane.
    bool previousInside = previousDistance >= 0.0F;

    for (const ClipVertex& current : input) {
        const float currentDistance = current.clip.z + current.clip.w;
        const bool currentInside = currentDistance >= 0.0F;
        if (currentInside != previousInside) {
            const float denominator = previousDistance - currentDistance;
            const float t = denominator != 0.0F
                                ? previousDistance / denominator
                                : 0.0F;
            output.push_back(
                interpolateClip(previous, current, glm::clamp(t, 0.0F, 1.0F)));
        }
        if (currentInside) {
            output.push_back(current);
        }
        previous = current;
        previousDistance = currentDistance;
        previousInside = currentInside;
    }
    return output;
}

bool toScreen(const ClipVertex& vertex, const RenderSettings& settings,
              ScreenVertex& output) {
    if (!(vertex.clip.w > 1.0e-7F)) {
        return false;
    }
    const glm::vec3 ndc = glm::vec3(vertex.clip) / vertex.clip.w;
    if (!std::isfinite(ndc.x) || !std::isfinite(ndc.y) ||
        !std::isfinite(ndc.z)) {
        return false;
    }
    output.x = (ndc.x * 0.5F + 0.5F) * static_cast<float>(settings.width);
    output.y = (0.5F - ndc.y * 0.5F) * static_cast<float>(settings.height);
    output.z = ndc.z;
    output.invW = 1.0F / vertex.clip.w;
    output.worldPosition = vertex.worldPosition;
    output.worldNormal = vertex.worldNormal;
    output.color = vertex.color;
    return true;
}

std::uint8_t toByte(float value) {
    if (!std::isfinite(value)) {
        return 0;
    }
    const float clamped = glm::clamp(value, 0.0F, 1.0F);
    return static_cast<std::uint8_t>(clamped * 255.0F + 0.5F);
}

void drawTriangle(const ScreenVertex& v0, const ScreenVertex& v1,
                  const ScreenVertex& v2, const RenderSettings& settings,
                  std::vector<float>& depth,
                  std::vector<std::uint8_t>& color) {
    const float area = edgeFunction(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y);
    if (!std::isfinite(area) || std::abs(area) < 1.0e-9F) {
        return;
    }

    const float minimumX = std::min({v0.x, v1.x, v2.x});
    const float maximumX = std::max({v0.x, v1.x, v2.x});
    const float minimumY = std::min({v0.y, v1.y, v2.y});
    const float maximumY = std::max({v0.y, v1.y, v2.y});
    const float lastX = static_cast<float>(settings.width - 1U);
    const float lastY = static_cast<float>(settings.height - 1U);
    const int startX = static_cast<int>(
        std::floor(std::clamp(minimumX, 0.0F, lastX)));
    const int endX = static_cast<int>(
        std::ceil(std::clamp(maximumX, 0.0F, lastX)));
    const int startY = static_cast<int>(
        std::floor(std::clamp(minimumY, 0.0F, lastY)));
    const int endY = static_cast<int>(
        std::ceil(std::clamp(maximumY, 0.0F, lastY)));
    if (startX > endX || startY > endY) {
        return;
    }

    const bool positiveArea = area > 0.0F;
    const glm::vec3 lightDirection =
        glm::normalize(glm::vec3(-0.35F, 0.75F, 0.56F));
    for (int y = startY; y <= endY; ++y) {
        const float sampleY = static_cast<float>(y) + 0.5F;
        for (int x = startX; x <= endX; ++x) {
            const float sampleX = static_cast<float>(x) + 0.5F;
            float e0 = edgeFunction(v1.x, v1.y, v2.x, v2.y, sampleX, sampleY);
            float e1 = edgeFunction(v2.x, v2.y, v0.x, v0.y, sampleX, sampleY);
            float e2 = edgeFunction(v0.x, v0.y, v1.x, v1.y, sampleX, sampleY);
            if (positiveArea) {
                if (e0 < 0.0F || e1 < 0.0F || e2 < 0.0F) {
                    continue;
                }
            } else if (e0 > 0.0F || e1 > 0.0F || e2 > 0.0F) {
                continue;
            }

            const float b0 = e0 / area;
            const float b1 = e1 / area;
            const float b2 = e2 / area;
            const float ndcDepth = b0 * v0.z + b1 * v1.z + b2 * v2.z;
            const float depthValue =
                glm::clamp((ndcDepth + 1.0F) * 0.5F, 0.0F, 1.0F);
            const std::size_t pixel =
                static_cast<std::size_t>(y) * settings.width +
                static_cast<std::size_t>(x);
            if (!(depthValue < depth[pixel])) {
                continue;
            }

            const float denominator =
                b0 * v0.invW + b1 * v1.invW + b2 * v2.invW;
            if (!std::isfinite(denominator) ||
                std::abs(denominator) < 1.0e-12F) {
                continue;
            }
            glm::vec3 normal =
                (b0 * v0.worldNormal * v0.invW +
                 b1 * v1.worldNormal * v1.invW +
                 b2 * v2.worldNormal * v2.invW) /
                denominator;
            const glm::vec3 vertexColor =
                (b0 * v0.color * v0.invW +
                 b1 * v1.color * v1.invW +
                 b2 * v2.color * v2.invW) /
                denominator;

            const float length = glm::length(normal);
            if (std::isfinite(length) && length > 0.0F) {
                normal /= length;
            }
            const float lambert = std::isfinite(length) && length > 0.0F
                                      ? std::abs(glm::dot(normal,
                                                          lightDirection))
                                      : 0.0F;
            const float intensity = 0.26F + 0.74F * lambert;
            depth[pixel] = depthValue;
            color[pixel * 3U + 0U] = toByte(vertexColor.r * intensity);
            color[pixel * 3U + 1U] = toByte(vertexColor.g * intensity);
            color[pixel * 3U + 2U] = toByte(vertexColor.b * intensity);
        }
    }
}

} // namespace

bool cameraMatrices(const Camera& camera, std::uint32_t width,
                    std::uint32_t height, glm::mat4& view,
                    glm::mat4& projection, Error& error) {
    if (width == 0U || height == 0U) {
        error = makeError(ErrorCode::InvalidArgument,
                          "render dimensions must be positive");
        return false;
    }
    constexpr std::uint32_t kMaxDimension = 16'384U;
    constexpr std::uint64_t kMaxPixels = 100'000'000U;
    if (width > kMaxDimension || height > kMaxDimension ||
        static_cast<std::uint64_t>(width) * height > kMaxPixels) {
        error = makeError(ErrorCode::ResourceTooLarge,
                          "render dimensions exceed the 16384x16384 or "
                          "100M pixel safety limit");
        return false;
    }
    if (!isFinite(camera.eye) || !isFinite(camera.target) ||
        !isFinite(camera.up) || !std::isfinite(camera.fieldOfViewYDegrees) ||
        !std::isfinite(camera.nearPlane) ||
        !std::isfinite(camera.farPlane)) {
        error = makeError(ErrorCode::InvalidArgument,
                          "camera parameters must be finite");
        return false;
    }
    if (glm::distance(camera.eye, camera.target) < 1.0e-6F) {
        error = makeError(ErrorCode::InvalidArgument,
                          "camera eye and target must differ");
        return false;
    }
    if (glm::length(camera.up) < 1.0e-6F) {
        error = makeError(ErrorCode::InvalidArgument,
                          "camera up vector must be non-zero");
        return false;
    }
    if (camera.fieldOfViewYDegrees <= 0.0F ||
        camera.fieldOfViewYDegrees >= 179.0F ||
        camera.nearPlane <= 0.0F ||
        camera.farPlane <= camera.nearPlane) {
        error = makeError(ErrorCode::InvalidArgument,
                          "camera FOV/near/far values are invalid");
        return false;
    }

    view = glm::lookAtRH(camera.eye, camera.target, camera.up);
    const float aspect = static_cast<float>(width) /
                         static_cast<float>(height);
    projection = glm::perspectiveRH_ZO(
        glm::radians(camera.fieldOfViewYDegrees), aspect, camera.nearPlane,
        camera.farPlane);
    error = {};
    return true;
}

bool rasterizeOffline(const BakedFrame& baked, const RenderSettings& settings,
                      OfflineImage& image, Error& error) {
    glm::mat4 view;
    glm::mat4 projection;
    if (!cameraMatrices(settings.camera, settings.width, settings.height, view,
                        projection, error)) {
        return false;
    }
    const glm::mat4 viewProjection = projection * view;
    const std::size_t pixelCount =
        static_cast<std::size_t>(settings.width) * settings.height;
    image.depth.assign(pixelCount, 1.0F);
    image.rgb.assign(pixelCount * 3U, 0U);
    for (std::size_t pixel = 0; pixel < pixelCount; ++pixel) {
        image.rgb[pixel * 3U + 0U] = 14U;
        image.rgb[pixel * 3U + 1U] = 17U;
        image.rgb[pixel * 3U + 2U] = 22U;
    }

    if (baked.positions.size() != baked.normals.size() ||
        baked.positions.size() != baked.colors.size() ||
        baked.indices.size() % 3U != 0U) {
        error = makeError(ErrorCode::InvalidArgument,
                          "baked frame arrays are inconsistent");
        return false;
    }

    for (std::size_t triangle = 0; triangle < baked.indices.size();
         triangle += 3U) {
        std::array<ClipVertex, 3> vertices;
        for (std::size_t corner = 0; corner < 3U; ++corner) {
            const std::uint32_t index = baked.indices[triangle + corner];
            if (index >= baked.positions.size()) {
                error = makeError(ErrorCode::InvalidTriangleIndex,
                                  "baked triangle index is out of range");
                return false;
            }
            const glm::vec3& position = baked.positions[index];
            ClipVertex& vertex = vertices[corner];
            vertex.clip =
                viewProjection * glm::vec4(position, 1.0F);
            vertex.worldPosition = position;
            vertex.worldNormal = baked.normals[index];
            vertex.color = baked.colors[index];
        }

        const std::vector<ClipVertex> clipped =
            clipNearPlane({vertices[0], vertices[1], vertices[2]});
        if (clipped.size() < 3U) {
            continue;
        }
        for (std::size_t fan = 1; fan + 1U < clipped.size(); ++fan) {
            ScreenVertex screen[3];
            if (!toScreen(clipped[0], settings, screen[0]) ||
                !toScreen(clipped[fan], settings, screen[1]) ||
                !toScreen(clipped[fan + 1U], settings, screen[2])) {
                continue;
            }
            drawTriangle(screen[0], screen[1], screen[2], settings,
                         image.depth, image.rgb);
        }
    }
    error = {};
    return true;
}

bool writeOfflineImages(const OfflineImage& image,
                        const RenderSettings& settings,
                        const std::filesystem::path& depthPath,
                        const std::filesystem::path& colorPath,
                        Error& error) {
    if (!writePfm(depthPath, settings.width, settings.height, image.depth,
                  error) ||
        !writePpm(colorPath, settings.width, settings.height, image.rgb,
                  error)) {
        return false;
    }
    error = {};
    return true;
}

} // namespace skel
