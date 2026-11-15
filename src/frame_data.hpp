#pragma once

#include <glm/glm.hpp>

#include <array>
#include <cstddef>

namespace demo {

constexpr std::size_t kMaxJoints = 8;

// std140 contract shared with the GLSL FrameData block:
//   mat4 viewProjection       offset   0
//   mat4 model                offset  64
//   vec4 diffuseColor         offset 128
//   vec4 materialParams       offset 144  (x=roughness, y=emissive, z=time)
//   vec4 lightDirection       offset 160
//   vec4 viewPosition         offset 176
//   mat4 jointWorld[8]        offset 192
// Total size: 704 bytes.
struct alignas(16) FrameUniforms {
    glm::mat4 viewProjection;
    glm::mat4 model;
    glm::vec4 diffuseColor;
    glm::vec4 materialParams;
    glm::vec4 lightDirection;
    glm::vec4 viewPosition;
    std::array<glm::mat4, kMaxJoints> jointWorld;
};

static_assert(alignof(FrameUniforms) == 16);
static_assert(offsetof(FrameUniforms, viewProjection) == 0);
static_assert(offsetof(FrameUniforms, model) == 64);
static_assert(offsetof(FrameUniforms, diffuseColor) == 128);
static_assert(offsetof(FrameUniforms, materialParams) == 144);
static_assert(offsetof(FrameUniforms, lightDirection) == 160);
static_assert(offsetof(FrameUniforms, viewPosition) == 176);
static_assert(offsetof(FrameUniforms, jointWorld) == 192);
static_assert(sizeof(FrameUniforms) == 704);

FrameUniforms makeFrameUniforms(
    const glm::mat4 &viewProjection,
    const glm::mat4 &model,
    const glm::vec3 &diffuseColor,
    float roughness,
    float emissiveStrength,
    const std::array<glm::mat4, kMaxJoints> &jointWorld,
    const glm::vec3 &lightDirection,
    const glm::vec3 &viewPosition,
    double animationTimeSeconds);

} // namespace demo
