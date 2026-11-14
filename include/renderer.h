#pragma once

#include "gl_resources.h"
#include "scene.h"
#include "shader_sources.h"
#include "types.h"

#include <string>
#include <vector>

namespace dr {

class Renderer {
public:
    Renderer() = default;
    ~Renderer() = default;
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool initializeLive(const Scene& scene, const SceneAssets& assets, const LightSet& lights,
                        std::string& error);
    bool initializeSnapshot(const SnapshotData& snapshot, const LightSet& lights, std::string& error);
    void setLights(const LightSet& lights);
    FrameTiming renderFrame(int frameNumber, const std::string& mode);
    bool captureSnapshot(SnapshotData& snapshot, std::string& error);
    bool readResultImage(std::vector<std::uint8_t>& rgbaTopDown, std::string& error);

    std::size_t pointLightCount() const;
    std::size_t directionalLightCount() const;
    std::uint64_t lightVersion() const;

private:
    struct GpuMesh {
        GlVertexArray vertexArray;
        GlBuffer vertexBuffer;
        GlBuffer indexBuffer;
        GLsizei indexCount = 0;
    };

    struct GBufferUniforms {
        GLint model = -1;
        GLint viewProjection = -1;
        GLint albedo = -1;
        GLint specular = -1;
    };

    struct LightingUniforms {
        GLint worldTexture = -1;
        GLint normalTexture = -1;
        GLint albedoTexture = -1;
        GLint cameraPosition = -1;
    };

    bool initializeCommon(std::string& error);
    bool initializeGeometry(const SceneAssets& assets, std::string& error);
    bool initializeGBufferTextures(const GBufferCapture* uploadedData, std::string& error);
    bool initializeLightBuffer(std::string& error);
    bool uploadMesh(const CpuMesh& mesh, GpuMesh& output, std::string& error);
    void uploadLights();
    void renderGBuffer();
    void renderLighting();
    void copyResultTopDown(std::vector<std::uint8_t>& rgbaTopDown);
    std::string glErrorText(const char* stage);

    GlProgram gbufferProgram_;
    GlProgram lightingProgram_;
    GBufferUniforms gbufferUniforms_;
    LightingUniforms lightingUniforms_;

    GlTexture worldTexture_;
    GlTexture normalTexture_;
    GlTexture albedoTexture_;
    GlTexture depthTexture_;
    GlFramebuffer gbufferFramebuffer_;

    GlTexture resultTexture_;
    GlFramebuffer resultFramebuffer_;

    GlBuffer lightBuffer_;
    GlVertexArray emptyVertexArray_;
    std::vector<GpuMesh> gpuMeshes_;
    std::vector<RenderObject> objects_;
    Mat4 viewProjection_ = Mat4::identity();
    std::vector<float> lightData_;

    LightSet lights_;
    Vec3 cameraPosition_{};
    Vec3 ambient_{};
    float exposure_ = 1.0f;
    bool snapshotMode_ = false;
    GlQuery frameQuery_;
};

} // namespace dr
