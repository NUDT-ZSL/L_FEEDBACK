#pragma once

#include <array>

#include <glm/glm.hpp>

#include "config.h"

class OrbitCamera {
public:
    OrbitCamera();

    void setAspect(float aspect);
    void update(float deltaSeconds);
    void orbit(float deltaYawDegrees, float deltaPitchDegrees);
    void zoom(float wheelDelta);
    void pan(float rightAmount, float upAmount);
    void setAutoOrbit(bool enabled);
    void reset();

    bool autoOrbit() const { return autoOrbit_; }
    const glm::vec3& position() const { return position_; }
    const glm::vec3& target() const { return currentTarget_; }
    const glm::vec3& forward() const { return forward_; }
    const glm::mat4& view() const { return view_; }
    const glm::mat4& projection() const { return projection_; }
    const glm::mat4& viewProjection() const { return viewProjection_; }
    float aspect() const { return aspect_; }
    float nearPlane() const { return nearPlane_; }
    float farPlane() const { return farPlane_; }
    float verticalFovRadians() const { return verticalFovRadians_; }
    float yawDegrees() const { return currentYawDegrees_; }
    float pitchDegrees() const { return currentPitchDegrees_; }
    float distance() const { return currentDistance_; }

    float centerViewDepth() const;
    int centerCascade(const std::array<float, kMaxCascades>& splits, int cascadeCount) const;

private:
    void recompute();

    glm::vec3 currentTarget_{0.0f, 1.7f, 0.0f};
    glm::vec3 desiredTarget_{0.0f, 1.7f, 0.0f};
    glm::vec3 position_{0.0f};
    glm::vec3 forward_{0.0f, 0.0f, -1.0f};
    glm::vec3 right_{1.0f, 0.0f, 0.0f};
    glm::vec3 up_{0.0f, 1.0f, 0.0f};

    float currentYawDegrees_ = 38.0f;
    float desiredYawDegrees_ = 38.0f;
    float currentPitchDegrees_ = 21.0f;
    float desiredPitchDegrees_ = 21.0f;
    float currentDistance_ = 17.0f;
    float desiredDistance_ = 17.0f;

    float aspect_ = 16.0f / 9.0f;
    float nearPlane_ = 0.1f;
    float farPlane_ = 60.0f;
    float verticalFovRadians_ = glm::radians(50.0f);

    bool autoOrbit_ = false;
    glm::mat4 view_{1.0f};
    glm::mat4 projection_{1.0f};
    glm::mat4 viewProjection_{1.0f};
};

std::array<float, kMaxCascades> computePracticalSplits(float nearPlane,
                                                       float farPlane,
                                                       int cascadeCount,
                                                       float lambda);

