#include "terrain/camera_path.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace terrain {

Camera CameraPath::sample(float time, float aspect) const {
    Camera camera;
    if (keyframes.empty()) {
        return camera;
    }

    if (time <= keyframes.front().time) {
        const CameraKeyframe& key = keyframes.front();
        camera.position = key.position;
        camera.yawDegrees = key.yawDegrees;
        camera.pitchDegrees = key.pitchDegrees;
        camera.nearPlane = 0.5f;
        camera.farPlane = 2400.0f;
        (void)aspect;
        return camera;
    }
    if (time >= keyframes.back().time) {
        const CameraKeyframe& key = keyframes.back();
        camera.position = key.position;
        camera.yawDegrees = key.yawDegrees;
        camera.pitchDegrees = key.pitchDegrees;
        camera.nearPlane = 0.5f;
        camera.farPlane = 2400.0f;
        (void)aspect;
        return camera;
    }

    auto upper = std::upper_bound(
        keyframes.begin(),
        keyframes.end(),
        time,
        [](float value, const CameraKeyframe& keyframe) {
            return value < keyframe.time;
        });
    const CameraKeyframe& b = *upper;
    const CameraKeyframe& a = *(upper - 1);
    const float span = std::max(b.time - a.time, 1.0e-6f);
    const float t = clampf((time - a.time) / span, 0.0f, 1.0f);

    camera.position = a.position + (b.position - a.position) * t;
    const float yawDelta =
        std::remainder(b.yawDegrees - a.yawDegrees, 360.0f);
    camera.yawDegrees = a.yawDegrees + yawDelta * t;
    camera.pitchDegrees =
        a.pitchDegrees + (b.pitchDegrees - a.pitchDegrees) * t;
    camera.nearPlane = 0.5f;
    camera.farPlane = 2400.0f;
    (void)aspect;
    return camera;
}

namespace {

PathLoadResult pathFailure(const std::string& message, int line = 0) {
    PathLoadResult result;
    result.ok = false;
    result.error = message;
    result.line = line;
    return result;
}

bool parseStrictFloat(const std::string& text, float& value) {
    if (text.empty()) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const float parsed = std::strtof(text.c_str(), &end);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
        !std::isfinite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

}  // namespace

PathLoadResult loadCameraPath(const std::string& filename) {
    std::ifstream input(filename);
    if (!input) {
        return pathFailure("unable to open camera path");
    }

    CameraPath path;
    std::string lineText;
    int lineNumber = 0;
    while (std::getline(input, lineText)) {
        ++lineNumber;
        const std::size_t comment = lineText.find('#');
        if (comment != std::string::npos) {
            lineText.erase(comment);
        }

        std::istringstream stream(lineText);
        std::string token;
        if (!(stream >> token)) {
            continue;
        }

        CameraKeyframe keyframe;
        std::string time;
        std::string px;
        std::string py;
        std::string pz;
        std::string yaw;
        std::string pitch;
        if (!(stream >> px >> py >> pz >> yaw >> pitch)) {
            return pathFailure(
                "expected: time positionX positionY positionZ yaw pitch",
                lineNumber);
        }
        time = token;
        if (!parseStrictFloat(time, keyframe.time) ||
            !parseStrictFloat(px, keyframe.position.x) ||
            !parseStrictFloat(py, keyframe.position.y) ||
            !parseStrictFloat(pz, keyframe.position.z) ||
            !parseStrictFloat(yaw, keyframe.yawDegrees) ||
            !parseStrictFloat(pitch, keyframe.pitchDegrees)) {
            return pathFailure("non-numeric keyframe field", lineNumber);
        }

        std::string trailing;
        if (stream >> trailing) {
            return pathFailure("unexpected trailing field", lineNumber);
        }
        if (!std::isfinite(keyframe.time) ||
            !std::isfinite(keyframe.position.x) ||
            !std::isfinite(keyframe.position.y) ||
            !std::isfinite(keyframe.position.z) ||
            !std::isfinite(keyframe.yawDegrees) ||
            !std::isfinite(keyframe.pitchDegrees)) {
            return pathFailure("keyframe values must be finite", lineNumber);
        }
        if (keyframe.time < 0.0f) {
            return pathFailure("keyframe time must be non-negative", lineNumber);
        }
        if (keyframe.pitchDegrees <= -90.0f || keyframe.pitchDegrees >= 90.0f) {
            return pathFailure("pitch must be inside (-90, 90)", lineNumber);
        }
        if (!path.keyframes.empty() &&
            keyframe.time <= path.keyframes.back().time) {
            return pathFailure("keyframe times must be strictly increasing", lineNumber);
        }
        path.keyframes.push_back(keyframe);
    }

    if (!input.eof()) {
        return pathFailure("failed while reading camera path", lineNumber);
    }
    if (path.keyframes.empty()) {
        return pathFailure("camera path contains no keyframes", lineNumber);
    }

    PathLoadResult result;
    result.ok = true;
    result.path = std::move(path);
    return result;
}

CameraPath builtInCameraPath() {
    CameraPath path;
    path.keyframes = {
        {0.0f, {256.0f, 190.0f, -90.0f}, -8.0f, -18.0f},
        {3.0f, {210.0f, 130.0f, 80.0f}, -25.0f, -22.0f},
        {6.0f, {120.0f, 85.0f, 200.0f}, -48.0f, -18.0f},
        {9.0f, {180.0f, 70.0f, 320.0f}, 115.0f, -16.0f},
        {10.0f, {300.0f, 85.0f, 340.0f}, 170.0f, -14.0f},
        {13.0f, {390.0f, 150.0f, 200.0f}, 210.0f, -20.0f},
        {16.0f, {256.0f, 210.0f, -100.0f}, 360.0f, -20.0f},
    };
    return path;
}

}  // namespace terrain

