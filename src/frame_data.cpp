#include "frame_data.hpp"

namespace demo {

FrameUniforms makeFrameUniforms(
    const glm::mat4 &viewProjection,
    const glm::mat4 &model,
    const glm::vec3 &diffuseColor,
    float roughness,
    float emissiveStrength,
    const std::array<glm::mat4, kMaxJoints> &jointWorld,
    const glm::vec3 &lightDirection,
    const glm::vec3 &viewPosition,
    double animationTimeSeconds) {
    FrameUniforms frame{};
    frame.viewProjection = viewProjection;
    frame.model = model;
    frame.diffuseColor = glm::vec4(diffuseColor, 1.0f);
    frame.materialParams = glm::vec4(roughness, emissiveStrength,
                                    static_cast<float>(animationTimeSeconds), 0.0f);
    frame.lightDirection = glm::vec4(lightDirection, 0.0f);
    frame.viewPosition = glm::vec4(viewPosition, 1.0f);
    frame.jointWorld = jointWorld;
    return frame;
}

} // namespace demo
