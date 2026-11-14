#include "camera.h"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

OrbitCamera::OrbitCamera() {
    recompute();
}

void OrbitCamera::setAspect(float aspect) {
    aspect_ = std::max(aspect, 0.1f);
}

void OrbitCamera::update(float deltaSeconds) {
    if (autoOrbit_) {
        desiredYawDegrees_ += 5.0f * deltaSeconds;
    }
    const float blend = 1.0f - std::exp(-8.5f * std::max(deltaSeconds, 0.0f));
    currentYawDegrees_ += (desiredYawDegrees_ - currentYawDegrees_) * blend;
    currentPitchDegrees_ += (desiredPitchDegrees_ - currentPitchDegrees_) * blend;
    currentDistance_ += (desiredDistance_ - currentDistance_) * blend;
    currentTarget_ += (desiredTarget_ - currentTarget_) * blend;
    recompute();
}

void OrbitCamera::orbit(float deltaYawDegrees, float deltaPitchDegrees) {
    desiredYawDegrees_ += deltaYawDegrees;
    desiredPitchDegrees_ = std::clamp(desiredPitchDegrees_ + deltaPitchDegrees, -8.0f, 78.0f);
}

void OrbitCamera::zoom(float wheelDelta) {
    const float factor = std::pow(0.88f, wheelDelta);
    desiredDistance_ = std::clamp(desiredDistance_ * factor, 5.0f, 36.0f);
}

void OrbitCamera::pan(float rightAmount, float upAmount) {
    desiredTarget_ += right_ * rightAmount + up_ * upAmount;
    desiredTarget_.y = std::clamp(desiredTarget_.y, 0.3f, 8.0f);
}

void OrbitCamera::setAutoOrbit(bool enabled) {
    autoOrbit_ = enabled;
}

void OrbitCamera::reset() {
    currentTarget_ = desiredTarget_ = {0.0f, 1.7f, 0.0f};
    currentYawDegrees_ = desiredYawDegrees_ = 38.0f;
    currentPitchDegrees_ = desiredPitchDegrees_ = 21.0f;
    currentDistance_ = desiredDistance_ = 17.0f;
    autoOrbit_ = false;
    recompute();
}

void OrbitCamera::recompute() {
    const float yaw = glm::radians(currentYawDegrees_);
    const float pitch = glm::radians(currentPitchDegrees_);
    const glm::vec3 offset = {
        std::sin(yaw) * std::cos(pitch),
        std::sin(pitch),
        std::cos(yaw) * std::cos(pitch)
    };
    position_ = currentTarget_ + offset * currentDistance_;
    forward_ = glm::normalize(currentTarget_ - position_);
    right_ = glm::normalize(glm::cross(forward_, glm::vec3(0, 1, 0)));
    up_ = glm::normalize(glm::cross(right_, forward_));
    view_ = glm::lookAt(position_, currentTarget_, glm::vec3(0, 1, 0));
    projection_ = glm::perspective(verticalFovRadians_, aspect_, nearPlane_, farPlane_);
    viewProjection_ = projection_ * view_;
}

float OrbitCamera::centerViewDepth() const {
    if (forward_.y >= -0.0001f) {
        return farPlane_;
    }
    const float groundT = -position_.y / forward_.y;
    return std::clamp(groundT, nearPlane_, farPlane_);
}

int OrbitCamera::centerCascade(const std::array<float, kMaxCascades>& splits,
                               int cascadeCount) const {
    const float depth = centerViewDepth();
    for (int i = 0; i < cascadeCount; ++i) {
        if (depth <= splits[static_cast<size_t>(i)]) {
            return i;
        }
    }
    return std::max(cascadeCount - 1, 0);
}

std::array<float, kMaxCascades> computePracticalSplits(float nearPlane,
                                                       float farPlane,
                                                       int cascadeCount,
                                                       float lambda) {
    std::array<float, kMaxCascades> splits{};
    cascadeCount = std::clamp(cascadeCount, 1, kMaxCascades);
    lambda = std::clamp(lambda, 0.0f, 1.0f);
    const float ratio = farPlane / nearPlane;
    for (int i = 1; i <= cascadeCount; ++i) {
        const float fraction = static_cast<float>(i) / static_cast<float>(cascadeCount);
        const float uniform = nearPlane + (farPlane - nearPlane) * fraction;
        const float logarithmic = nearPlane * std::pow(ratio, fraction);
        splits[static_cast<size_t>(i - 1)] =
            lambda * logarithmic + (1.0f - lambda) * uniform;
    }
    splits[static_cast<size_t>(cascadeCount - 1)] = farPlane;
    return splits;
}

