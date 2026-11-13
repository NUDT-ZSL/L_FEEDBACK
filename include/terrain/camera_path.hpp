#pragma once

#include "terrain/lod.hpp"

#include <string>
#include <vector>

namespace terrain {

struct CameraKeyframe {
    float time = 0.0f;
    Vec3 position;
    float yawDegrees = 0.0f;
    float pitchDegrees = 0.0f;
};

struct CameraPath {
    std::vector<CameraKeyframe> keyframes;

    Camera sample(float time, float aspect) const;
};

struct PathLoadResult {
    bool ok = false;
    CameraPath path;
    std::string error;
    int line = 0;
};

PathLoadResult loadCameraPath(const std::string& filename);
CameraPath builtInCameraPath();

}  // namespace terrain

