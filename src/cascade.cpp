#include "cascade.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>

#include "error.h"

namespace sr {
namespace {

bool isPowerOfTwo(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

bool finiteMatrix(const glm::dmat4& matrix) {
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            if (!std::isfinite(matrix[column][row])) {
                return false;
            }
        }
    }
    return true;
}

std::array<glm::dvec3, 4> frustumPlaneCorners(const glm::dmat4& inverseViewProjection,
                                               double ndcZ) {
    static constexpr double xy[4][2] = {
        {-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0},
    };
    std::array<glm::dvec3, 4> corners{};
    for (int i = 0; i < 4; ++i) {
        const glm::dvec4 homogeneous =
            inverseViewProjection *
            glm::dvec4(xy[i][0], xy[i][1], ndcZ, 1.0);
        corners[static_cast<std::size_t>(i)] =
            glm::dvec3(homogeneous) / homogeneous.w;
    }
    return corners;
}

} // namespace

void validateCascadeConfig(const CascadeConfig& config) {
    if (config.cascadeCount < 2 || config.cascadeCount > 4) {
        fail(ExitCode::Cascade, "E_CASCADE_CONFIG",
             "cascade count must be between 2 and 4");
    }
    if (config.shadowResolution < 256 || config.shadowResolution > 4096 ||
        !isPowerOfTwo(config.shadowResolution)) {
        fail(ExitCode::Cascade, "E_CASCADE_CONFIG",
             "shadow resolution must be a power of two in [256, 4096]");
    }
    if (!std::isfinite(config.splitLambda) ||
        config.splitLambda < 0.0 || config.splitLambda > 1.0) {
        fail(ExitCode::Cascade, "E_CASCADE_CONFIG",
             "cascade split lambda must be in [0, 1]");
    }
    if (!std::isfinite(config.casterPadding) || config.casterPadding < 0.0) {
        fail(ExitCode::Cascade, "E_CASCADE_CONFIG",
             "caster padding must be finite and non-negative");
    }
}

CascadeSet buildCascades(const Camera& camera,
                         const Bounds& visibleCasterBounds,
                         const DirectionalLight& light,
                         const CascadeConfig& config) {
    validateCascadeConfig(config);
    if (!visibleCasterBounds.valid) {
        fail(ExitCode::Cascade, "E_CASCADE_BOUNDS",
             "visible caster bounds are empty");
    }
    if (!(camera.nearPlane > 0.0) || !(camera.farPlane > camera.nearPlane)) {
        fail(ExitCode::Cascade, "E_CASCADE_CONFIG", "camera clipping planes are invalid");
    }

    const glm::dvec3 forward = glm::normalize(camera.target - camera.eye);
    std::array<double, 8> cornerDepths{};
    const auto sceneCorners = visibleCasterBounds.corners();
    double maxSceneDepth = 0.0;
    for (std::size_t i = 0; i < sceneCorners.size(); ++i) {
        const double depth = glm::dot(sceneCorners[i] - camera.eye, forward);
        cornerDepths[i] = depth;
        maxSceneDepth = std::max(maxSceneDepth, depth);
    }
    if (maxSceneDepth <= camera.nearPlane) {
        fail(ExitCode::Cascade, "E_CASCADE_BOUNDS",
             "visible casters are outside the camera forward range");
    }

    CascadeSet result;
    result.cameraNear = camera.nearPlane;
    result.cameraFar = camera.farPlane;
    result.sceneFar = std::min(camera.farPlane,
                               std::max(camera.nearPlane + 0.25,
                                        maxSceneDepth + 0.75));
    if (!(result.sceneFar > result.cameraNear + 0.1)) {
        fail(ExitCode::Cascade, "E_CASCADE_CONFIG",
             "scene depth range is too small for cascaded shadows");
    }

    const int count = config.cascadeCount;
    std::vector<double> splits(static_cast<std::size_t>(count + 1), camera.nearPlane);
    splits.back() = result.sceneFar;
    for (int i = 1; i < count; ++i) {
        const double fraction = static_cast<double>(i) / static_cast<double>(count);
        const double uniform =
            camera.nearPlane + (result.sceneFar - camera.nearPlane) * fraction;
        const double logarithmic =
            camera.nearPlane *
            std::pow(result.sceneFar / camera.nearPlane, fraction);
        splits[static_cast<std::size_t>(i)] =
            config.splitLambda * logarithmic +
            (1.0 - config.splitLambda) * uniform;
    }
    splits[0] = camera.nearPlane;
    for (std::size_t i = 1; i < splits.size(); ++i) {
        if (!(splits[i] > splits[i - 1])) {
            fail(ExitCode::Cascade, "E_CASCADE_CONFIG",
                 "cascade split distances are not strictly increasing");
        }
    }

    const glm::dmat4 inverseViewProjection = glm::inverse(camera.viewProjection);
    const auto cameraNearCorners = frustumPlaneCorners(inverseViewProjection, -1.0);
    const auto cameraFarCorners = frustumPlaneCorners(inverseViewProjection, 1.0);

    const glm::dvec3 lightDirection = glm::normalize(light.directionTowardScene);
    if (!std::isfinite(glm::length(lightDirection)) ||
        glm::length(lightDirection) < 1.0e-9) {
        fail(ExitCode::Cascade, "E_CASCADE_LIGHT", "directional light direction is invalid");
    }
    const glm::dvec3 up =
        std::abs(glm::dot(lightDirection, glm::dvec3(0, 1, 0))) > 0.97
            ? glm::dvec3(0, 0, 1)
            : glm::dvec3(0, 1, 0);
    const glm::dmat4 lightView = glm::lookAt(glm::dvec3(0.0),
                                             lightDirection,
                                             up);

    result.cascades.reserve(static_cast<std::size_t>(count));
    for (int cascadeIndex = 0; cascadeIndex < count; ++cascadeIndex) {
        const double sliceNear = splits[static_cast<std::size_t>(cascadeIndex)];
        const double sliceFar = splits[static_cast<std::size_t>(cascadeIndex + 1)];

        std::vector<glm::dvec3> candidates;
        candidates.reserve(16);
        for (int corner = 0; corner < 4; ++corner) {
            const glm::dvec3& nearCorner =
                cameraNearCorners[static_cast<std::size_t>(corner)];
            const glm::dvec3& farCorner =
                cameraFarCorners[static_cast<std::size_t>(corner)];
            const glm::dvec3 nearPoint =
                nearCorner + (farCorner - nearCorner) *
                                 ((sliceNear - camera.nearPlane) /
                                  (camera.farPlane - camera.nearPlane));
            const glm::dvec3 farPoint =
                nearCorner + (farCorner - nearCorner) *
                                 ((sliceFar - camera.nearPlane) /
                                  (camera.farPlane - camera.nearPlane));
            candidates.push_back(nearPoint);
            candidates.push_back(farPoint);
        }

        double minDepth = std::numeric_limits<double>::infinity();
        double maxDepth = -std::numeric_limits<double>::infinity();
        for (double depth : cornerDepths) {
            minDepth = std::min(minDepth, depth);
            maxDepth = std::max(maxDepth, depth);
        }
        const bool boundsOverlapSlice =
            maxDepth >= sliceNear - config.casterPadding &&
            minDepth <= sliceFar + config.casterPadding;
        if (boundsOverlapSlice) {
            for (const glm::dvec3& corner : sceneCorners) {
                candidates.push_back(corner);
            }
        }

        double minX = std::numeric_limits<double>::infinity();
        double minY = std::numeric_limits<double>::infinity();
        double minZ = std::numeric_limits<double>::infinity();
        double maxX = -std::numeric_limits<double>::infinity();
        double maxY = -std::numeric_limits<double>::infinity();
        double maxZ = -std::numeric_limits<double>::infinity();
        for (const glm::dvec3& candidate : candidates) {
            const glm::dvec3 lightPoint =
                glm::dvec3(lightView * glm::dvec4(candidate, 1.0));
            minX = std::min(minX, lightPoint.x);
            minY = std::min(minY, lightPoint.y);
            minZ = std::min(minZ, lightPoint.z);
            maxX = std::max(maxX, lightPoint.x);
            maxY = std::max(maxY, lightPoint.y);
            maxZ = std::max(maxZ, lightPoint.z);
        }

        const double rawWidth = std::max(maxX - minX, 0.25);
        const double rawHeight = std::max(maxY - minY, 0.25);
        const double rawExtent = std::max(rawWidth, rawHeight);
        const double rawTexel = rawExtent / static_cast<double>(config.shadowResolution);
        double halfExtent = rawExtent * 0.5 * 1.06 + rawTexel * 2.0;
        const double texel = (2.0 * halfExtent) /
                             static_cast<double>(config.shadowResolution);
        const double rawCenterX = (minX + maxX) * 0.5;
        const double rawCenterY = (minY + maxY) * 0.5;
        const double centerX = std::floor(rawCenterX / texel) * texel;
        const double centerY = std::floor(rawCenterY / texel) * texel;
        halfExtent = std::max(halfExtent, rawExtent * 0.5 + texel * 2.0);

        const double depthPadding = 1.25;
        const double depthSpan = std::max(0.1, maxZ - minZ);
        const double depthShift = -maxZ - depthPadding;
        const glm::dmat4 shiftedLightView =
            glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, depthShift)) *
            lightView;
        const double nearZ = depthPadding;
        const double farZ = nearZ + depthSpan + 2.0 * depthPadding;

        Cascade cascade;
        cascade.index = cascadeIndex;
        cascade.splitNear = sliceNear;
        cascade.splitFar = sliceFar;
        cascade.lightView = shiftedLightView;
        cascade.lightProjection = glm::ortho(centerX - halfExtent,
                                             centerX + halfExtent,
                                             centerY - halfExtent,
                                             centerY + halfExtent,
                                             nearZ,
                                             farZ);
        cascade.lightViewProjection = cascade.lightProjection * cascade.lightView;
        cascade.radius = halfExtent;
        cascade.texelWorldSize = 2.0 * halfExtent /
                                 static_cast<double>(config.shadowResolution);
        cascade.lightCenter =
            glm::dvec3(glm::inverse(shiftedLightView) *
                       glm::dvec4(centerX, centerY, -(nearZ + farZ) * 0.5, 1.0));

        if (!finiteMatrix(cascade.lightViewProjection)) {
            fail(ExitCode::Cascade, "E_CASCADE_MATRIX",
                 "cascade light matrix contains non-finite values");
        }
        result.cascades.push_back(cascade);
    }

    if (result.cascades.size() != static_cast<std::size_t>(count)) {
        fail(ExitCode::Cascade, "E_CASCADE_CONFIG", "failed to build all cascades");
    }
    return result;
}

} // namespace sr
