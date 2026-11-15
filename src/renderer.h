#pragma once

#include "gl_api.h"
#include "mesh.h"

#include <cstddef>
#include <string>
#include <vector>

struct GLFWwindow;

namespace subdiv {

class Renderer {
public:
    Renderer() = default;
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool initialize(std::string& error);
    void shutdown();

    bool rebuildGpuMesh(const PolygonMesh& mesh, int level, std::string& error);
    void render(const CameraSettings& camera,
                int framebufferWidth,
                int framebufferHeight,
                const Vec3& diffuse,
                float roughness,
                bool normalMode,
                bool useFaceNormal,
                bool wireframe);

    bool checkGlError(const std::string& stage, std::string& error);
    bool saveFramebufferPpm(int width, int height, const std::string& path,
                            std::string& error);

    bool shadersReady() const { return program_ != 0; }
    std::string shaderStatus() const { return shaderStatus_; }
    std::size_t gpuVertexCount() const { return gpuVertexCount_; }
    std::size_t gpuTriangleCount() const { return gpuTriangleCount_; }
    int level() const { return level_; }

private:
    void releaseGpuMesh();

    gl::GLuint program_ = 0;
    gl::GLuint vao_ = 0;
    gl::GLuint vertexBuffer_ = 0;
    gl::GLuint indexBuffer_ = 0;
    gl::GLsizei indexCount_ = 0;
    std::size_t gpuVertexCount_ = 0;
    std::size_t gpuTriangleCount_ = 0;
    int level_ = 0;
    std::string shaderStatus_ = "not compiled";

    gl::GLint viewProjectionLocation_ = -1;
    gl::GLint cameraPositionLocation_ = -1;
    gl::GLint diffuseLocation_ = -1;
    gl::GLint roughnessLocation_ = -1;
    gl::GLint normalModeLocation_ = -1;
    gl::GLint faceNormalLocation_ = -1;
};

}  // namespace subdiv

