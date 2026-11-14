#include "renderer.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace dr {
namespace {

constexpr GLuint kLightBindingPoint = 1;
constexpr std::size_t kLightDataFloats =
    4U + static_cast<std::size_t>(kMaxPointLights) * 8U +
    static_cast<std::size_t>(kMaxDirectionalLights) * 8U;

double milliseconds(std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

} // namespace

bool Renderer::initializeCommon(std::string& error) {
    if (!gbufferProgram_.compile(shaders::kGBufferVertex, shaders::kGBufferFragment, error)) {
        return false;
    }
    if (!lightingProgram_.compile(shaders::kFullscreenVertex, shaders::kLightingFragment, error)) {
        return false;
    }

    gbufferUniforms_.model = gbufferProgram_.uniform("uModel");
    gbufferUniforms_.viewProjection = gbufferProgram_.uniform("uViewProjection");
    gbufferUniforms_.albedo = gbufferProgram_.uniform("uAlbedo");
    gbufferUniforms_.specular = gbufferProgram_.uniform("uSpecular");
    lightingUniforms_.worldTexture = lightingProgram_.uniform("uWorldTexture");
    lightingUniforms_.normalTexture = lightingProgram_.uniform("uNormalTexture");
    lightingUniforms_.albedoTexture = lightingProgram_.uniform("uAlbedoTexture");
    lightingUniforms_.cameraPosition = lightingProgram_.uniform("uCameraPosition");

    if (gbufferUniforms_.model < 0 || gbufferUniforms_.viewProjection < 0 ||
        gbufferUniforms_.albedo < 0 || gbufferUniforms_.specular < 0 ||
        lightingUniforms_.worldTexture < 0 || lightingUniforms_.normalTexture < 0 ||
        lightingUniforms_.albedoTexture < 0 || lightingUniforms_.cameraPosition < 0) {
        error = "one or more shader uniforms were optimized out";
        return false;
    }

    resultTexture_.create2D(kRenderWidth, kRenderHeight, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    resultFramebuffer_.create();
    resultFramebuffer_.attachColor(resultTexture_.id(), GL_COLOR_ATTACHMENT0);
    const GLuint resultAttachment = GL_COLOR_ATTACHMENT0;
    resultFramebuffer_.setDrawBuffers(&resultAttachment, 1);
    if (!resultFramebuffer_.complete(error)) {
        error = "result framebuffer incomplete: " + error;
        return false;
    }

    emptyVertexArray_.create();
    frameQuery_.create(GL_TIME_ELAPSED);

    return true;
}

bool Renderer::initializeGBufferTextures(const GBufferCapture* uploadedData, std::string& error) {
    const void* worldPixels = uploadedData ? uploadedData->world.data() : nullptr;
    const void* normalPixels = uploadedData ? uploadedData->normal.data() : nullptr;
    const void* albedoPixels = uploadedData ? uploadedData->albedo.data() : nullptr;
    const void* depthPixels = uploadedData ? uploadedData->depth.data() : nullptr;

    worldTexture_.create2D(kRenderWidth, kRenderHeight, GL_RGBA32F, GL_RGBA, GL_FLOAT, worldPixels);
    normalTexture_.create2D(kRenderWidth, kRenderHeight, GL_RGBA16F, GL_RGBA, GL_FLOAT, normalPixels);
    albedoTexture_.create2D(kRenderWidth, kRenderHeight, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
                            albedoPixels);
    depthTexture_.create2D(kRenderWidth, kRenderHeight, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT,
                           GL_FLOAT, depthPixels);

    gbufferFramebuffer_.create();
    gbufferFramebuffer_.attachColor(worldTexture_.id(), GL_COLOR_ATTACHMENT0);
    gbufferFramebuffer_.attachColor(normalTexture_.id(), GL_COLOR_ATTACHMENT1);
    gbufferFramebuffer_.attachColor(albedoTexture_.id(), GL_COLOR_ATTACHMENT2);
    gbufferFramebuffer_.attachDepth(depthTexture_.id());
    const GLuint attachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
                                   GL_COLOR_ATTACHMENT2};
    gbufferFramebuffer_.setDrawBuffers(attachments, 3);
    if (!gbufferFramebuffer_.complete(error)) {
        error = "G-Buffer framebuffer incomplete: " + error;
        return false;
    }
    return true;
}

bool Renderer::uploadMesh(const CpuMesh& mesh, GpuMesh& output, std::string& error) {
    if (mesh.vertices.empty() || mesh.indices.empty()) {
        error = "mesh has no vertices or indices";
        return false;
    }

    output.vertexArray.create();
    output.vertexArray.bind();
    output.vertexBuffer.create(GL_ARRAY_BUFFER);
    output.vertexBuffer.upload(mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex),
                               GL_STATIC_DRAW);
    output.indexBuffer.create(GL_ELEMENT_ARRAY_BUFFER);
    output.indexBuffer.upload(mesh.indices.data(), mesh.indices.size() * sizeof(std::uint32_t),
                              GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<const void*>(offsetof(Vertex, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<const void*>(offsetof(Vertex, normal)));
    glBindVertexArray(0);
    output.indexCount = static_cast<GLsizei>(mesh.indices.size());
    return true;
}

bool Renderer::initializeGeometry(const SceneAssets& assets, std::string& error) {
    gpuMeshes_.resize(assets.meshes.size());
    for (std::size_t index = 0; index < assets.meshes.size(); ++index) {
        if (!uploadMesh(assets.meshes[index], gpuMeshes_[index], error)) {
            return false;
        }
    }
    objects_ = assets.objects;
    viewProjection_ = assets.viewProjection;
    return true;
}

bool Renderer::initializeLightBuffer(std::string& error) {
    lightBuffer_.create(GL_UNIFORM_BUFFER);
    lightData_.resize(kLightDataFloats, 0.0f);
    lightBuffer_.upload(lightData_.data(), lightData_.size() * sizeof(float), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, kLightBindingPoint, lightBuffer_.id());

    const GLuint blockIndex = glGetUniformBlockIndex(lightingProgram_.id(), "LightBlock");
    if (blockIndex == GL_INVALID_INDEX) {
        error = "LightBlock uniform block was not found";
        return false;
    }
    glUniformBlockBinding(lightingProgram_.id(), blockIndex, kLightBindingPoint);
    return true;
}

bool Renderer::initializeLive(const Scene& scene, const SceneAssets& assets, const LightSet& lights,
                              std::string& error) {
    snapshotMode_ = false;
    lights_ = lights;
    cameraPosition_ = scene.camera.position;
    ambient_ = scene.ambient;
    exposure_ = scene.exposure;
    if (!initializeCommon(error)) {
        return false;
    }
    if (!initializeGBufferTextures(nullptr, error)) {
        return false;
    }
    if (!initializeGeometry(assets, error)) {
        return false;
    }
    if (!initializeLightBuffer(error)) {
        return false;
    }
    return glGetError() == GL_NO_ERROR;
}

bool Renderer::initializeSnapshot(const SnapshotData& snapshot, const LightSet& lights,
                                  std::string& error) {
    snapshotMode_ = true;
    lights_ = lights;
    cameraPosition_ = snapshot.cameraPosition;
    ambient_ = snapshot.ambient;
    exposure_ = snapshot.exposure;
    if (!initializeCommon(error)) {
        return false;
    }
    if (!initializeGBufferTextures(&snapshot.gbuffer, error)) {
        return false;
    }
    if (!initializeLightBuffer(error)) {
        return false;
    }
    return glGetError() == GL_NO_ERROR;
}

void Renderer::setLights(const LightSet& lights) {
    lights_ = lights;
}

void Renderer::uploadLights() {
    std::fill(lightData_.begin(), lightData_.end(), 0.0f);
    lightData_[0] = ambient_.x;
    lightData_[1] = ambient_.y;
    lightData_[2] = ambient_.z;
    lightData_[3] = exposure_;
    lightData_[4] = static_cast<float>(lights_.pointCount());
    lightData_[5] = static_cast<float>(lights_.directionalCount());

    const std::size_t pointPositionOffset = 8;
    const std::size_t pointColorOffset = pointPositionOffset + kMaxPointLights * 4U;
    const std::size_t directionOffset = pointColorOffset + kMaxPointLights * 4U;
    const std::size_t directionColorOffset = directionOffset + kMaxDirectionalLights * 4U;

    std::size_t pointIndex = 0;
    std::size_t directionalIndex = 0;
    for (const Light& light : lights_.lights) {
        if (light.type == LightType::Point) {
            const std::size_t offset = pointPositionOffset + pointIndex * 4U;
            lightData_[offset] = light.positionOrDirection.x;
            lightData_[offset + 1] = light.positionOrDirection.y;
            lightData_[offset + 2] = light.positionOrDirection.z;
            lightData_[offset + 3] = light.radius;
            const std::size_t colorOffset = pointColorOffset + pointIndex * 4U;
            lightData_[colorOffset] = light.color.x;
            lightData_[colorOffset + 1] = light.color.y;
            lightData_[colorOffset + 2] = light.color.z;
            lightData_[colorOffset + 3] = light.intensity;
            ++pointIndex;
        } else {
            const Vec3 direction = normalize(light.positionOrDirection);
            const std::size_t offset = directionOffset + directionalIndex * 4U;
            lightData_[offset] = direction.x;
            lightData_[offset + 1] = direction.y;
            lightData_[offset + 2] = direction.z;
            lightData_[offset + 3] = 0.0f;
            const std::size_t colorOffset = directionColorOffset + directionalIndex * 4U;
            lightData_[colorOffset] = light.color.x;
            lightData_[colorOffset + 1] = light.color.y;
            lightData_[colorOffset + 2] = light.color.z;
            lightData_[colorOffset + 3] = light.intensity;
            ++directionalIndex;
        }
    }

    lightBuffer_.bind();
    glBufferSubData(GL_UNIFORM_BUFFER, 0, static_cast<GLsizeiptr>(lightData_.size() * sizeof(float)),
                    lightData_.data());
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Renderer::renderGBuffer() {
    glBindFramebuffer(GL_FRAMEBUFFER, gbufferFramebuffer_.id());
    glViewport(0, 0, kRenderWidth, kRenderHeight);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glDisable(GL_CULL_FACE);

    const GLfloat clearWorld[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const GLfloat clearNormal[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const GLfloat clearAlbedo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    glClearBufferfv(GL_COLOR, 0, clearWorld);
    glClearBufferfv(GL_COLOR, 1, clearNormal);
    glClearBufferfv(GL_COLOR, 2, clearAlbedo);
    const GLfloat clearDepth = 1.0f;
    glClearBufferfv(GL_DEPTH, 0, &clearDepth);

    gbufferProgram_.use();
    glUniformMatrix4fv(gbufferUniforms_.viewProjection, 1, GL_FALSE, viewProjection_.data());
    for (const RenderObject& object : objects_) {
        GpuMesh& mesh = gpuMeshes_[object.meshIndex];
        glUniformMatrix4fv(gbufferUniforms_.model, 1, GL_FALSE, object.model.data());
        glUniform3f(gbufferUniforms_.albedo, object.material.albedo.x, object.material.albedo.y,
                    object.material.albedo.z);
        glUniform1f(gbufferUniforms_.specular, object.material.specular);
        mesh.vertexArray.bind();
        glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, nullptr);
    }
    glBindVertexArray(0);
}

void Renderer::renderLighting() {
    glBindFramebuffer(GL_FRAMEBUFFER, resultFramebuffer_.id());
    glViewport(0, 0, kRenderWidth, kRenderHeight);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);

    lightingProgram_.use();
    glBindBufferBase(GL_UNIFORM_BUFFER, kLightBindingPoint, lightBuffer_.id());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, worldTexture_.id());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, normalTexture_.id());
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, albedoTexture_.id());
    glUniform1i(lightingUniforms_.worldTexture, 0);
    glUniform1i(lightingUniforms_.normalTexture, 1);
    glUniform1i(lightingUniforms_.albedoTexture, 2);
    glUniform3f(lightingUniforms_.cameraPosition, cameraPosition_.x, cameraPosition_.y,
                cameraPosition_.z);
    emptyVertexArray_.bind();
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glActiveTexture(GL_TEXTURE0);
}

FrameTiming Renderer::renderFrame(int frameNumber, const std::string& mode) {
    const auto begin = std::chrono::steady_clock::now();
    frameQuery_.begin();
    uploadLights();
    if (snapshotMode_) {
        renderLighting();
    } else {
        renderGBuffer();
        renderLighting();
    }
    frameQuery_.end();
    const auto end = std::chrono::steady_clock::now();

    FrameTiming timing;
    timing.frame = frameNumber;
    timing.mode = mode;
    timing.cpuMs = milliseconds(begin, end);
    timing.gpuMs = static_cast<double>(frameQuery_.result64()) / 1'000'000.0;
    timing.pointLights = pointLightCount();
    timing.directionalLights = directionalLightCount();
    timing.lightVersion = lightVersion();
    return timing;
}

void Renderer::copyResultTopDown(std::vector<std::uint8_t>& rgbaTopDown) {
    glBindFramebuffer(GL_FRAMEBUFFER, resultFramebuffer_.id());
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    std::vector<std::uint8_t> bottomUp(
        static_cast<std::size_t>(kRenderWidth) * static_cast<std::size_t>(kRenderHeight) * 4U);
    glReadPixels(0, 0, kRenderWidth, kRenderHeight, GL_RGBA, GL_UNSIGNED_BYTE, bottomUp.data());

    rgbaTopDown.resize(bottomUp.size());
    const std::size_t rowBytes = static_cast<std::size_t>(kRenderWidth) * 4U;
    for (int row = 0; row < kRenderHeight; ++row) {
        const std::size_t sourceRow = static_cast<std::size_t>(kRenderHeight - 1 - row);
        std::memcpy(rgbaTopDown.data() + static_cast<std::size_t>(row) * rowBytes,
                    bottomUp.data() + sourceRow * rowBytes, rowBytes);
    }
}

bool Renderer::readResultImage(std::vector<std::uint8_t>& rgbaTopDown, std::string& error) {
    copyResultTopDown(rgbaTopDown);
    const GLenum glError = glGetError();
    if (glError != GL_NO_ERROR) {
        error = "OpenGL error while reading result: " + std::to_string(glError);
        return false;
    }
    return true;
}

bool Renderer::captureSnapshot(SnapshotData& snapshot, std::string& error) {
    snapshot.width = kRenderWidth;
    snapshot.height = kRenderHeight;
    snapshot.formatVersion = kSnapshotFormatVersion;
    snapshot.lightVersion = lights_.version;
    snapshot.lights = lights_.lights;
    snapshot.cameraPosition = cameraPosition_;
    snapshot.ambient = ambient_;
    snapshot.exposure = exposure_;
    snapshot.gbuffer.width = kRenderWidth;
    snapshot.gbuffer.height = kRenderHeight;

    const std::size_t floatPixels =
        static_cast<std::size_t>(kRenderWidth) * static_cast<std::size_t>(kRenderHeight) * 4U;
    const std::size_t bytePixels =
        static_cast<std::size_t>(kRenderWidth) * static_cast<std::size_t>(kRenderHeight) * 4U;
    snapshot.gbuffer.world.resize(floatPixels);
    snapshot.gbuffer.normal.resize(floatPixels);
    snapshot.gbuffer.albedo.resize(bytePixels);
    snapshot.gbuffer.depth.resize(floatPixels / 4U);

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, worldTexture_.id());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, snapshot.gbuffer.world.data());
    glBindTexture(GL_TEXTURE_2D, normalTexture_.id());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, snapshot.gbuffer.normal.data());
    glBindTexture(GL_TEXTURE_2D, albedoTexture_.id());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, snapshot.gbuffer.albedo.data());
    glBindTexture(GL_TEXTURE_2D, depthTexture_.id());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, snapshot.gbuffer.depth.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    copyResultTopDown(snapshot.gbuffer.resultRgbaTopDown);

    const GLenum glError = glGetError();
    if (glError != GL_NO_ERROR) {
        error = "OpenGL error while capturing snapshot: " + std::to_string(glError);
        return false;
    }
    return true;
}

std::size_t Renderer::pointLightCount() const {
    return lights_.pointCount();
}

std::size_t Renderer::directionalLightCount() const {
    return lights_.directionalCount();
}

std::uint64_t Renderer::lightVersion() const {
    return lights_.version;
}

std::string Renderer::glErrorText(const char* stage) {
    return std::string(stage) + ": GL error " + std::to_string(glGetError());
}

} // namespace dr
