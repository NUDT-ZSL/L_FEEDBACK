#include "shadow_renderer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace {

constexpr const char* kDepthVertexShader = R"GLSL(
#version 450 core
layout(location = 0) in vec3 aPosition;

uniform mat4 uModel;
uniform mat4 uLightViewProjection;

out vec3 vWorldPosition;

void main() {
    vec4 world = uModel * vec4(aPosition, 1.0);
    vWorldPosition = world.xyz;
    gl_Position = uLightViewProjection * world;
}
)GLSL";

constexpr const char* kDepthFragmentShader = R"GLSL(
#version 450 core
in vec3 vWorldPosition;

uniform mat4 uDepthView;
uniform vec3 uLightPosition;
uniform float uDepthFar;
uniform int uPointMode;

void main() {
    float linearDepth;
    if (uPointMode != 0) {
        linearDepth = length(vWorldPosition - uLightPosition);
    } else {
        linearDepth = -(uDepthView * vec4(vWorldPosition, 1.0)).z;
    }
    gl_FragDepth = clamp(linearDepth / uDepthFar, 0.0, 1.0);
}
)GLSL";

constexpr std::array<glm::vec3, 6> kCubeDirections = {
    glm::vec3(1, 0, 0), glm::vec3(-1, 0, 0),
    glm::vec3(0, 1, 0), glm::vec3(0, -1, 0),
    glm::vec3(0, 0, 1), glm::vec3(0, 0, -1)
};

constexpr std::array<glm::vec3, 6> kCubeUps = {
    glm::vec3(0, -1, 0), glm::vec3(0, -1, 0),
    glm::vec3(0, 0, 1), glm::vec3(0, 0, -1),
    glm::vec3(0, -1, 0), glm::vec3(0, -1, 0)
};

std::array<glm::vec3, 8> frustumSliceCorners(const OrbitCamera& camera,
                                             float sliceNear,
                                             float sliceFar) {
    const glm::vec3 forward = camera.forward();
    const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
    const glm::vec3 up = glm::normalize(glm::cross(right, forward));
    const float tanHalfFov = std::tan(camera.verticalFovRadians() * 0.5f);
    const float aspect = camera.aspect();

    std::array<glm::vec3, 8> corners{};
    for (int i = 0; i < 8; ++i) {
        const bool farSide = (i & 4) != 0;
        const float distance = farSide ? sliceFar : sliceNear;
        const float halfHeight = tanHalfFov * distance;
        const float halfWidth = halfHeight * aspect;
        const float sx = (i & 1) != 0 ? 1.0f : -1.0f;
        const float sy = (i & 2) != 0 ? 1.0f : -1.0f;
        corners[static_cast<size_t>(i)] =
            camera.position() + forward * distance + right * (sx * halfWidth) +
            up * (sy * halfHeight);
    }
    return corners;
}

}  // namespace

ShadowRenderer::~ShadowRenderer() {
    shutdown();
}

void ShadowRenderer::initialize(int resolution) {
    depthShader_.build(kDepthVertexShader, kDepthFragmentShader, "shadow_depth");
    createResources(resolution);
    updateCascades(OrbitCamera{}, DirectionalLight{}, 0.72f, 0.12f);
}

void ShadowRenderer::shutdown() {
    destroyResources();
    depthShader_.destroy();
}

void ShadowRenderer::resize(int resolution) {
    resolution = std::clamp(resolution, 256, 8192);
    if (resolution == resolution_ && directionalTexture_ != 0) {
        return;
    }
    createResources(resolution);
}

GLuint ShadowRenderer::pointTexture(int index) const {
    if (index < 0 || index >= kMaxPointLights) {
        return 0;
    }
    return pointTextures_[static_cast<size_t>(index)];
}

void ShadowRenderer::createResources(int resolution) {
    destroyResources();
    resolution_ = resolution;

    glGenTextures(1, &directionalTexture_);
    glBindTexture(GL_TEXTURE_2D_ARRAY, directionalTexture_);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_DEPTH_COMPONENT24,
                   resolution_, resolution_, kMaxCascades);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_NONE);

    glGenFramebuffers(1, &directionalFbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, directionalFbo_);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              directionalTexture_, 0, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Directional cascade framebuffer is incomplete");
    }

    for (int i = 0; i < kMaxPointLights; ++i) {
        glGenTextures(1, &pointTextures_[static_cast<size_t>(i)]);
        glBindTexture(GL_TEXTURE_CUBE_MAP, pointTextures_[static_cast<size_t>(i)]);
        glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_DEPTH_COMPONENT24,
                       resolution_, resolution_);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_COMPARE_MODE, GL_NONE);

        glGenFramebuffers(1, &pointFbos_[static_cast<size_t>(i)]);
        glBindFramebuffer(GL_FRAMEBUFFER, pointFbos_[static_cast<size_t>(i)]);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X,
                               pointTextures_[static_cast<size_t>(i)], 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Point shadow framebuffer is incomplete");
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    const std::size_t oneMap = static_cast<std::size_t>(resolution_) *
                               static_cast<std::size_t>(resolution_) * 4u;
    estimatedBytes_ = oneMap * (static_cast<std::size_t>(kMaxCascades) +
                                6u * static_cast<std::size_t>(kMaxPointLights));
    std::cout << "[SHADOW] allocated resolution=" << resolution_
              << " directional_layers=" << kMaxCascades
              << " point_cubes=" << kMaxPointLights
              << " estimated_vram_mib="
              << (static_cast<double>(estimatedBytes_) / (1024.0 * 1024.0))
              << "\n";
    checkGLError("ShadowRenderer::createResources");
}

void ShadowRenderer::destroyResources() {
    for (int i = 0; i < kMaxPointLights; ++i) {
        if (pointFbos_[static_cast<size_t>(i)] != 0) {
            glDeleteFramebuffers(1, &pointFbos_[static_cast<size_t>(i)]);
            pointFbos_[static_cast<size_t>(i)] = 0;
        }
        if (pointTextures_[static_cast<size_t>(i)] != 0) {
            glDeleteTextures(1, &pointTextures_[static_cast<size_t>(i)]);
            pointTextures_[static_cast<size_t>(i)] = 0;
        }
    }
    if (directionalFbo_ != 0) {
        glDeleteFramebuffers(1, &directionalFbo_);
        directionalFbo_ = 0;
    }
    if (directionalTexture_ != 0) {
        glDeleteTextures(1, &directionalTexture_);
        directionalTexture_ = 0;
    }
    estimatedBytes_ = 0;
}

void ShadowRenderer::updateCascades(const OrbitCamera& camera,
                                    const DirectionalLight& light,
                                    float splitLambda,
                                    float overlapFraction) {
    splitDepths_ = computePracticalSplits(camera.nearPlane(), camera.farPlane(),
                                          cascadeCount_, splitLambda);
    const glm::vec3 toLight = glm::normalize(-light.direction);
    const glm::vec3 stableUp =
        std::abs(toLight.y) > 0.96f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
    const glm::vec3 right = glm::normalize(glm::cross(stableUp, toLight));
    const glm::vec3 up = glm::normalize(glm::cross(toLight, right));
    overlapFraction = std::clamp(overlapFraction, 0.0f, 0.4f);

    for (int i = 0; i < cascadeCount_; ++i) {
        CascadeData& cascade = cascades_[static_cast<size_t>(i)];
        cascade.splitNear = (i == 0)
                                ? camera.nearPlane()
                                : splitDepths_[static_cast<size_t>(i - 1)] *
                                      (1.0f - overlapFraction);
        cascade.splitFar = (i == cascadeCount_ - 1)
                               ? camera.farPlane()
                               : splitDepths_[static_cast<size_t>(i)] *
                                     (1.0f + overlapFraction);
        if (cascade.splitFar <= cascade.splitNear + 0.01f) {
            cascade.splitFar = cascade.splitNear + 0.01f;
        }

        const std::array<glm::vec3, 8> corners =
            frustumSliceCorners(camera, cascade.splitNear, cascade.splitFar);
        glm::vec3 center(0.0f);
        for (const glm::vec3& corner : corners) {
            center += corner;
        }
        center /= 8.0f;
        float radius = 0.0f;
        for (const glm::vec3& corner : corners) {
            radius = std::max(radius, glm::length(corner - center));
        }
        radius = std::max(radius * 1.04f, 0.25f);

        cascade.radius = radius;
        cascade.texelWorld = (2.0f * radius) / static_cast<float>(resolution_);

        const float lightX = glm::dot(center, right);
        const float lightY = glm::dot(center, up);
        const float lightZ = glm::dot(center, toLight);
        const glm::vec3 snappedCenter =
            right * (std::round(lightX / cascade.texelWorld) * cascade.texelWorld) +
            up * (std::round(lightY / cascade.texelWorld) * cascade.texelWorld) +
            toLight * lightZ;

        cascade.view = glm::lookAt(snappedCenter + toLight * (2.0f * radius),
                                   snappedCenter, up);
        cascade.farPlane = 4.0f * radius + 1.0f;
        cascade.projection = glm::ortho(-radius, radius, -radius, radius,
                                        0.05f, cascade.farPlane);
        cascade.viewProjection = cascade.projection * cascade.view;
    }

    for (int i = 0; i < kMaxPointLights; ++i) {
        pointFarPlanes_[static_cast<size_t>(i)] = 1.0f;
    }
}

void ShadowRenderer::setDepthPassUniforms() {
    depthShader_.use();
}

void ShadowRenderer::renderShadowMaps(const Scene& scene,
                                      const LightManager& lights,
                                      const OrbitCamera& camera) {
    if (directionalTexture_ == 0) {
        return;
    }
    updateCascades(camera, lights.directional, 0.72f, 0.12f);

    GLint previousFbo = 0;
    GLint previousViewport[4] = {0, 0, 0, 0};
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previousFbo);
    glGetIntegerv(GL_VIEWPORT, previousViewport);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glDisable(GL_SCISSOR_TEST);

    depthShader_.use();
    depthShader_.setInt("uPointMode", 0);
    if (lights.directional.enabled && lights.directional.castsShadow) {
        renderDirectional(scene, lights);
    }

    for (int i = 0; i < kMaxPointLights; ++i) {
        const PointLight& light = lights.points[static_cast<size_t>(i)];
        if (light.enabled && light.castsShadow) {
            glCullFace(GL_FRONT);
            renderPointLight(scene, light, i);
        }
    }
    glCullFace(GL_BACK);

    glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(previousFbo));
    glViewport(previousViewport[0], previousViewport[1],
               previousViewport[2], previousViewport[3]);
    checkGLError("ShadowRenderer::renderShadowMaps", false);
}

void ShadowRenderer::renderDirectional(const Scene& scene, const LightManager& lights) {
    glBindFramebuffer(GL_FRAMEBUFFER, directionalFbo_);
    glViewport(0, 0, resolution_, resolution_);
    depthShader_.use();
    depthShader_.setInt("uPointMode", 0);
    depthShader_.setVec3("uLightPosition", glm::vec3(0.0f));

    for (int cascadeIndex = 0; cascadeIndex < cascadeCount_; ++cascadeIndex) {
        const CascadeData& cascade = cascades_[static_cast<size_t>(cascadeIndex)];
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  directionalTexture_, 0, cascadeIndex);
        glClearDepth(1.0);
        glClear(GL_DEPTH_BUFFER_BIT);
        depthShader_.setMat4("uLightViewProjection", cascade.viewProjection);
        depthShader_.setMat4("uDepthView", cascade.view);
        depthShader_.setFloat("uDepthFar", cascade.farPlane);

        for (const SceneObject& object : scene.objects()) {
            if (!object.castsShadow) {
                continue;
            }
            depthShader_.setMat4("uModel", object.transform);
            scene.mesh(object.mesh).draw();
        }
    }
    (void)lights;
}

void ShadowRenderer::renderPointLight(const Scene& scene,
                                      const PointLight& light,
                                      int index) {
    const float farPlane = std::max(light.range, 1.0f);
    pointFarPlanes_[static_cast<size_t>(index)] = farPlane;
    const glm::mat4 projection =
        glm::perspective(glm::radians(90.0f), 1.0f, 0.05f, farPlane);

    glBindFramebuffer(GL_FRAMEBUFFER, pointFbos_[static_cast<size_t>(index)]);
    glViewport(0, 0, resolution_, resolution_);
    depthShader_.use();
    depthShader_.setInt("uPointMode", 1);
    depthShader_.setVec3("uLightPosition", light.position);
    depthShader_.setFloat("uDepthFar", farPlane);

    for (int face = 0; face < 6; ++face) {
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
            static_cast<GLenum>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face),
            pointTextures_[static_cast<size_t>(index)], 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Point shadow framebuffer face is incomplete");
        }

        const glm::vec3 target = light.position + kCubeDirections[static_cast<size_t>(face)];
        const glm::mat4 view =
            glm::lookAt(light.position, target, kCubeUps[static_cast<size_t>(face)]);
        depthShader_.setMat4("uLightViewProjection", projection * view);

        glClearDepth(1.0);
        glClear(GL_DEPTH_BUFFER_BIT);
        for (const SceneObject& object : scene.objects()) {
            if (!object.castsShadow) {
                continue;
            }
            depthShader_.setMat4("uModel", object.transform);
            scene.mesh(object.mesh).draw();
        }
    }
}

