#include "renderer.h"

#include "shaders.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace subdiv {
namespace {

struct GpuVertex {
    float position[3];
    float vertexNormal[3];
    float faceNormal[3];
    float faceId;
};

std::string shaderLog(gl::GLuint shader) {
    gl::GLint length = 0;
    gl::GetShaderiv(shader, gl::GL_INFO_LOG_LENGTH, &length);
    if (length <= 1) {
        return "no shader log";
    }
    std::string log(static_cast<std::size_t>(length), '\0');
    gl::GLsizei written = 0;
    gl::GetShaderInfoLog(shader, length, &written, log.data());
    log.resize(static_cast<std::size_t>(written));
    return log;
}

std::string programLog(gl::GLuint program) {
    gl::GLint length = 0;
    gl::GetProgramiv(program, gl::GL_INFO_LOG_LENGTH, &length);
    if (length <= 1) {
        return "no program log";
    }
    std::string log(static_cast<std::size_t>(length), '\0');
    gl::GLsizei written = 0;
    gl::GetProgramInfoLog(program, length, &written, log.data());
    log.resize(static_cast<std::size_t>(written));
    return log;
}

bool compileShader(gl::GLenum type, const char* source, gl::GLuint& shader,
                   std::string& error) {
    shader = gl::CreateShader(type);
    if (shader == 0) {
        error = "glCreateShader returned 0";
        return false;
    }
    gl::ShaderSource(shader, 1, &source, nullptr);
    gl::CompileShader(shader);
    gl::GLint compiled = 0;
    gl::GetShaderiv(shader, gl::GL_COMPILE_STATUS, &compiled);
    if (compiled == gl::GL_FALSE) {
        error = std::string(type == gl::GL_VERTEX_SHADER ? "vertex" : "fragment") +
                " shader compilation failed:\n" + shaderLog(shader);
        gl::DeleteShader(shader);
        shader = 0;
        return false;
    }
    return true;
}

bool buildGpuVertices(const PolygonMesh& mesh,
                      std::vector<GpuVertex>& vertices,
                      std::vector<std::uint32_t>& indices) {
    if (mesh.positions.empty() || mesh.faces.empty()) {
        return false;
    }

    const std::vector<Vec3> faceNormals = computePolygonFaceNormals(mesh);
    std::vector<Vec3> accumulatedNormals(mesh.positions.size(), Vec3{});

    for (std::size_t faceIndex = 0; faceIndex < mesh.faces.size(); ++faceIndex) {
        const auto& face = mesh.faces[faceIndex];
        if (face.size() < 3) {
            continue;
        }
        const Vec3 normal = faceNormals[faceIndex];
        float areaWeight = 1.0f;
        const Vec3 origin = mesh.positions[face[0]];
        Vec3 areaVector{};
        for (std::size_t i = 1; i + 1 < face.size(); ++i) {
            areaVector += cross(mesh.positions[face[i]] - origin,
                                mesh.positions[face[i + 1]] - origin);
        }
        const float area = length(areaVector);
        if (area > 0.0f) {
            areaWeight = area;
        }
        for (std::uint32_t vertex : face) {
            if (vertex < accumulatedNormals.size()) {
                accumulatedNormals[vertex] += normal * areaWeight;
            }
        }
    }

    std::vector<Vec3> vertexNormals(mesh.positions.size());
    for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
        vertexNormals[i] = normalize(accumulatedNormals[i]);
        if (lengthSquared(vertexNormals[i]) < 1.0e-12f) {
            vertexNormals[i] = {0.0f, 1.0f, 0.0f};
        }
    }

    vertices.clear();
    indices.clear();

    for (std::size_t faceIndex = 0; faceIndex < mesh.faces.size(); ++faceIndex) {
        const auto& face = mesh.faces[faceIndex];
        if (face.size() < 3) {
            continue;
        }
        const Vec3 faceNormal = faceNormals[faceIndex];
        for (std::size_t triangle = 1; triangle + 1 < face.size(); ++triangle) {
            const std::uint32_t corners[3] = {
                face[0], face[triangle], face[triangle + 1],
            };
            for (std::uint32_t index : corners) {
                if (index >= mesh.positions.size()) {
                    return false;
                }
                const Vec3& position = mesh.positions[index];
                const Vec3& vertexNormal = vertexNormals[index];
                vertices.push_back({
                    {position.x, position.y, position.z},
                    {vertexNormal.x, vertexNormal.y, vertexNormal.z},
                    {faceNormal.x, faceNormal.y, faceNormal.z},
                    static_cast<float>(faceIndex),
                });
                indices.push_back(static_cast<std::uint32_t>(vertices.size() - 1));
            }
        }
    }
    return !vertices.empty() && !indices.empty();
}

}  // namespace

Renderer::~Renderer() {
    shutdown();
}

bool Renderer::initialize(std::string& error) {
    if (std::getenv("SUBDIV_FORCE_SHADER_ERROR") != nullptr) {
        error = "forced shader failure via SUBDIV_FORCE_SHADER_ERROR";
        return false;
    }

    gl::GLuint vertexShader = 0;
    gl::GLuint fragmentShader = 0;
    std::string compileError;
    if (!compileShader(gl::GL_VERTEX_SHADER, kSurfaceVertexShader, vertexShader,
                       compileError)) {
        error = compileError;
        return false;
    }
    if (!compileShader(gl::GL_FRAGMENT_SHADER, kSurfaceFragmentShader, fragmentShader,
                       compileError)) {
        gl::DeleteShader(vertexShader);
        error = compileError;
        return false;
    }

    program_ = gl::CreateProgram();
    if (program_ == 0) {
        gl::DeleteShader(vertexShader);
        gl::DeleteShader(fragmentShader);
        error = "glCreateProgram returned 0";
        return false;
    }
    gl::AttachShader(program_, vertexShader);
    gl::AttachShader(program_, fragmentShader);
    gl::LinkProgram(program_);
    gl::DeleteShader(vertexShader);
    gl::DeleteShader(fragmentShader);

    gl::GLint linked = 0;
    gl::GetProgramiv(program_, gl::GL_LINK_STATUS, &linked);
    if (linked == gl::GL_FALSE) {
        error = "shader program link failed:\n" + programLog(program_);
        gl::DeleteProgram(program_);
        program_ = 0;
        return false;
    }

    viewProjectionLocation_ = gl::GetUniformLocation(program_, "uViewProjection");
    cameraPositionLocation_ = gl::GetUniformLocation(program_, "uCameraPosition");
    diffuseLocation_ = gl::GetUniformLocation(program_, "uDiffuseColor");
    roughnessLocation_ = gl::GetUniformLocation(program_, "uRoughness");
    normalModeLocation_ = gl::GetUniformLocation(program_, "uNormalMode");
    faceNormalLocation_ = gl::GetUniformLocation(program_, "uUseFaceNormal");
    if (viewProjectionLocation_ < 0 || cameraPositionLocation_ < 0 ||
        diffuseLocation_ < 0 || roughnessLocation_ < 0 ||
        normalModeLocation_ < 0 || faceNormalLocation_ < 0) {
        error = "one or more required shader uniforms were optimized out or not found";
        shutdown();
        return false;
    }

    shaderStatus_ = "compiled+linked";
    return true;
}

void Renderer::shutdown() {
    releaseGpuMesh();
    if (program_ != 0) {
        gl::DeleteProgram(program_);
        program_ = 0;
    }
    shaderStatus_ = "not compiled";
}

void Renderer::releaseGpuMesh() {
    if (indexBuffer_ != 0) {
        gl::DeleteBuffers(1, &indexBuffer_);
        indexBuffer_ = 0;
    }
    if (vertexBuffer_ != 0) {
        gl::DeleteBuffers(1, &vertexBuffer_);
        vertexBuffer_ = 0;
    }
    if (vao_ != 0) {
        gl::DeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    indexCount_ = 0;
    gpuVertexCount_ = 0;
    gpuTriangleCount_ = 0;
    level_ = 0;
}

bool Renderer::rebuildGpuMesh(const PolygonMesh& mesh, int level, std::string& error) {
    std::vector<GpuVertex> vertices;
    std::vector<std::uint32_t> indices;
    if (!buildGpuVertices(mesh, vertices, indices)) {
        error = "subdivided mesh could not be converted to GPU triangles";
        return false;
    }

    releaseGpuMesh();

    gl::GenVertexArrays(1, &vao_);
    if (vao_ == 0) {
        error = "glGenVertexArrays returned 0";
        return false;
    }
    gl::BindVertexArray(vao_);

    gl::GenBuffers(1, &vertexBuffer_);
    gl::GenBuffers(1, &indexBuffer_);
    if (vertexBuffer_ == 0 || indexBuffer_ == 0) {
        error = "glGenBuffers returned 0";
        releaseGpuMesh();
        return false;
    }

    gl::BindBuffer(gl::GL_ARRAY_BUFFER, vertexBuffer_);
    gl::BufferData(gl::GL_ARRAY_BUFFER,
                   static_cast<gl::GLsizeiptr>(vertices.size() * sizeof(GpuVertex)),
                   vertices.data(), gl::GL_STATIC_DRAW);
    gl::BindBuffer(gl::GL_ELEMENT_ARRAY_BUFFER, indexBuffer_);
    gl::BufferData(gl::GL_ELEMENT_ARRAY_BUFFER,
                   static_cast<gl::GLsizeiptr>(indices.size() * sizeof(std::uint32_t)),
                   indices.data(), gl::GL_STATIC_DRAW);

    constexpr gl::GLsizei stride = static_cast<gl::GLsizei>(sizeof(GpuVertex));
    gl::VertexAttribPointer(0, 3, gl::GL_FLOAT, gl::GL_FALSE, stride,
                            reinterpret_cast<const void*>(offsetof(GpuVertex, position)));
    gl::EnableVertexAttribArray(0);
    gl::VertexAttribPointer(1, 3, gl::GL_FLOAT, gl::GL_FALSE, stride,
                            reinterpret_cast<const void*>(offsetof(GpuVertex, vertexNormal)));
    gl::EnableVertexAttribArray(1);
    gl::VertexAttribPointer(2, 3, gl::GL_FLOAT, gl::GL_FALSE, stride,
                            reinterpret_cast<const void*>(offsetof(GpuVertex, faceNormal)));
    gl::EnableVertexAttribArray(2);
    gl::VertexAttribPointer(3, 1, gl::GL_FLOAT, gl::GL_FALSE, stride,
                            reinterpret_cast<const void*>(offsetof(GpuVertex, faceId)));
    gl::EnableVertexAttribArray(3);

    gl::BindVertexArray(0);
    gl::BindBuffer(gl::GL_ARRAY_BUFFER, 0);

    indexCount_ = static_cast<gl::GLsizei>(indices.size());
    gpuVertexCount_ = vertices.size();
    gpuTriangleCount_ = indices.size() / 3;
    level_ = level;

    if (!checkGlError("GPU mesh upload", error)) {
        releaseGpuMesh();
        return false;
    }
    return true;
}

void Renderer::render(const CameraSettings& camera,
                      int framebufferWidth,
                      int framebufferHeight,
                      const Vec3& diffuse,
                      float roughness,
                      bool normalMode,
                      bool useFaceNormal,
                      bool wireframe) {
    gl::Viewport(0, 0, framebufferWidth, framebufferHeight);
    gl::ClearColor(0.025f, 0.032f, 0.045f, 1.0f);
    gl::Clear(gl::GL_COLOR_BUFFER_BIT | gl::GL_DEPTH_BUFFER_BIT);

    if (program_ == 0 || vao_ == 0 || indexCount_ == 0) {
        return;
    }

    gl::Enable(gl::GL_DEPTH_TEST);
    gl::DepthFunc(gl::GL_LEQUAL);
    gl::Enable(gl::GL_CULL_FACE);
    gl::CullFace(gl::GL_BACK);
    gl::FrontFace(gl::GL_CCW);

    const float aspect = framebufferHeight > 0
        ? static_cast<float>(framebufferWidth) / static_cast<float>(framebufferHeight)
        : 1.0f;
    const Mat4 projection =
        Mat4::perspective(camera.fovDegrees * kPi / 180.0f, aspect, 0.05f, 100.0f);
    const Mat4 view = Mat4::lookAt(camera.position, camera.target, camera.up);
    const Mat4 viewProjection = Mat4::multiply(projection, view);

    gl::UseProgram(program_);
    gl::UniformMatrix4fv(viewProjectionLocation_, 1, gl::GL_FALSE, viewProjection.values);
    gl::Uniform3f(cameraPositionLocation_, camera.position.x, camera.position.y,
                  camera.position.z);
    gl::Uniform3f(diffuseLocation_, diffuse.x, diffuse.y, diffuse.z);
    gl::Uniform1f(roughnessLocation_, roughness);
    gl::Uniform1i(normalModeLocation_, normalMode ? 1 : 0);
    gl::Uniform1i(faceNormalLocation_, useFaceNormal ? 1 : 0);

    if (wireframe) {
        gl::PolygonMode(gl::GL_FRONT_AND_BACK, gl::GL_LINE);
    } else {
        gl::PolygonMode(gl::GL_FRONT_AND_BACK, gl::GL_FILL);
    }

    gl::BindVertexArray(vao_);
    gl::DrawElements(gl::GL_TRIANGLES, indexCount_, gl::GL_UNSIGNED_INT, nullptr);
    gl::BindVertexArray(0);
    gl::PolygonMode(gl::GL_FRONT_AND_BACK, gl::GL_FILL);
    gl::UseProgram(0);
}

bool Renderer::checkGlError(const std::string& stage, std::string& error) {
    bool ok = true;
    std::ostringstream stream;
    for (int i = 0; i < 32; ++i) {
        const gl::GLenum code = gl::GetError();
        if (code == gl::GL_NO_ERROR) {
            break;
        }
        ok = false;
        if (stream.tellp() > 0) {
            stream << "; ";
        }
        std::ostringstream hex;
        hex << "0x" << std::hex << code;
        stream << stage << ": " << gl::errorName(code) << " (" << hex.str() << ')';
    }
    if (!ok) {
        error = stream.str();
    }
    return ok;
}

bool Renderer::saveFramebufferPpm(int width, int height, const std::string& path,
                                  std::string& error) {
    if (width <= 0 || height <= 0) {
        error = "invalid framebuffer size for screenshot";
        return false;
    }
    std::vector<unsigned char> pixels(static_cast<std::size_t>(width) *
                                      static_cast<std::size_t>(height) * 4U);
    gl::PixelStorei(gl::GL_PACK_ALIGNMENT, 1);
    gl::ReadPixels(0, 0, width, height, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, pixels.data());
    if (!checkGlError("glReadPixels", error)) {
        return false;
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        error = "unable to open screenshot path '" + path + "'";
        return false;
    }
    file << "P6\n" << width << ' ' << height << "\n255\n";
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t index =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(x)) *
                4U;
            file.put(static_cast<char>(pixels[index]));
            file.put(static_cast<char>(pixels[index + 1]));
            file.put(static_cast<char>(pixels[index + 2]));
        }
    }
    if (!file) {
        error = "failed while writing screenshot '" + path + "'";
        return false;
    }
    return true;
}

}  // namespace subdiv

