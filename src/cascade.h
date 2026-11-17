#pragma once

#include <vector>

#include "types.h"

namespace sr {

struct CascadeConfig {
    int cascadeCount = 2;
    int shadowResolution = 1024;
    double splitLambda = 0.25;
    double casterPadding = 1.25;
};

struct Cascade {
    int index = 0;
    double splitNear = 0.0;
    double splitFar = 0.0;
    glm::dmat4 lightView{1.0};
    glm::dmat4 lightProjection{1.0};
    glm::dmat4 lightViewProjection{1.0};
    glm::dvec3 lightCenter{0.0};
    double radius = 0.0;
    double texelWorldSize = 0.0;
};

struct CascadeSet {
    std::vector<Cascade> cascades;
    double cameraNear = 0.0;
    double cameraFar = 0.0;
    double sceneFar = 0.0;
};

void validateCascadeConfig(const CascadeConfig& config);
CascadeSet buildCascades(const Camera& camera,
                         const Bounds& visibleCasterBounds,
                         const DirectionalLight& light,
                         const CascadeConfig& config);

} // namespace sr
