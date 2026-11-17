#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "cascade.h"
#include "cli.h"
#include "scene.h"

namespace sr {

class GLContext;

struct RendererSettings {
    int width = 960;
    int height = 540;
    int shadowSize = 1024;
    int cascadeCount = 2;
    bool forceShaderError = false;
    bool forceFramebufferError = false;
};

struct FrameRenderResult {
    std::vector<std::uint8_t> colorRgbaTopDown;
    std::vector<float> depthTopDown;
    std::vector<glm::dmat4> shadowMatrices;
    std::vector<double> shadowCoverage;
    double shadowPassMs = 0.0;
    double depthPrepassMs = 0.0;
    double colorPassMs = 0.0;
    double readbackMs = 0.0;
    double totalMs = 0.0;
};

class Renderer {
public:
    Renderer(GLContext& context, RendererSettings settings);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void uploadCharacter(const MeshData& mesh);
    void uploadStaticMeshes(const MeshData& occluders, const MeshData& ground);

    FrameRenderResult renderFrame(const PoseResult& pose,
                                  const Camera& camera,
                                  const CascadeSet& cascades,
                                  const Lights& lights,
                                  const Material& characterMaterial,
                                  const Material& occluderMaterial,
                                  const Material& groundMaterial);

    const std::string& glVersion() const noexcept { return glVersion_; }
    const std::string& glRenderer() const noexcept { return glRenderer_; }

private:
    struct GpuMesh {
        unsigned int vao = 0;
        unsigned int vbo = 0;
        unsigned int ebo = 0;
        int indexCount = 0;
    };

    void initialize();
    void shutdown();
    void createPrograms();
    void createTargets();
    void createUniformBuffers();
    void createGpuMesh(GpuMesh& out, const MeshData& mesh, bool dynamic);
    void updateGpuMesh(GpuMesh& out, const MeshData& mesh);
    void destroyGpuMesh(GpuMesh& mesh);
    unsigned int compileProgram(const char* vertexSource, const char* fragmentSource,
                                const char* debugName);
    unsigned int compileShader(unsigned int type, const char* source, const char* debugName);
    void drawMesh(const GpuMesh& mesh) const;
    void drawShadowPass(const CascadeSet& cascades,
                        std::vector<double>& shadowCoverage);
    void drawMainPass(const PoseResult& pose,
                      const Camera& camera,
                      const CascadeSet& cascades,
                      const Lights& lights,
                      const Material& characterMaterial,
                      const Material& occluderMaterial,
                      const Material& groundMaterial,
                      double& depthPrepassMs,
                      double& colorPassMs);
    void uploadBoneBlock(const PoseResult& pose);
    void uploadLightBlock(const Lights& lights);
    void uploadShadowBlock(const CascadeSet& cascades);
    void checkGl(const char* operation);

    GLContext& context_;
    RendererSettings settings_;
    std::string glVersion_;
    std::string glRenderer_;
    bool initialized_ = false;

    GpuMesh character_{};
    GpuMesh occluders_{};
    GpuMesh ground_{};

    unsigned int mainFbo_ = 0;
    unsigned int colorTexture_ = 0;
    unsigned int depthTexture_ = 0;
    std::array<unsigned int, 4> shadowFbos_{};
    std::array<unsigned int, 4> shadowTextures_{};

    unsigned int boneUbo_ = 0;
    unsigned int lightUbo_ = 0;
    unsigned int shadowUbo_ = 0;

    unsigned int skinnedForward_ = 0;
    unsigned int skinnedDepth_ = 0;
    unsigned int staticForward_ = 0;
    unsigned int staticDepth_ = 0;
    unsigned int skinnedShadow_ = 0;
    unsigned int staticShadow_ = 0;
};

} // namespace sr
