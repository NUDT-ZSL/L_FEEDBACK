#include "preview.hpp"

#include "gl_api.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace skel {
namespace {

std::string glfwError;

void glfwErrorCallback(int code, const char* description) {
    std::ostringstream stream;
    stream << "GLFW error " << code << ": "
           << (description != nullptr ? description : "unknown");
    glfwError = stream.str();
}

struct PreviewState {
    GLFWwindow* window = nullptr;
    Camera initialCamera;
    glm::vec3 target{0.0F};
    float distance = 5.0F;
    float yaw = 0.0F;
    float pitch = 0.0F;
    bool wireframe = false;
    bool paused = false;
    bool dragging = false;
    double lastCursorX = 0.0;
    double lastCursorY = 0.0;
};

PreviewState* stateFrom(GLFWwindow* window) {
    return static_cast<PreviewState*>(glfwGetWindowUserPointer(window));
}

void mouseButtonCallback(GLFWwindow* window, int button, int action,
                         int /*mods*/) {
    PreviewState* state = stateFrom(window);
    if (button != GLFW_MOUSE_BUTTON_LEFT) {
        return;
    }
    if (action == GLFW_PRESS) {
        state->dragging = true;
        glfwGetCursorPos(window, &state->lastCursorX, &state->lastCursorY);
    } else if (action == GLFW_RELEASE) {
        state->dragging = false;
    }
}

void cursorCallback(GLFWwindow* window, double x, double y) {
    PreviewState* state = stateFrom(window);
    if (!state->dragging) {
        return;
    }
    const double deltaX = x - state->lastCursorX;
    const double deltaY = y - state->lastCursorY;
    state->lastCursorX = x;
    state->lastCursorY = y;
    state->yaw -= static_cast<float>(deltaX) * 0.008F;
    state->pitch = glm::clamp(state->pitch -
                                  static_cast<float>(deltaY) * 0.008F,
                              -1.45F, 1.45F);
}

void scrollCallback(GLFWwindow* window, double /*xOffset*/, double yOffset) {
    PreviewState* state = stateFrom(window);
    state->distance = glm::clamp(
        state->distance * std::exp(-static_cast<float>(yOffset) * 0.12F),
        0.25F, 80.0F);
}

void keyCallback(GLFWwindow* window, int key, int /*scancode*/, int action,
                 int /*mods*/) {
    if (action != GLFW_PRESS) {
        return;
    }
    PreviewState* state = stateFrom(window);
    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    } else if (key == GLFW_KEY_W) {
        state->wireframe = !state->wireframe;
    } else if (key == GLFW_KEY_SPACE) {
        state->paused = !state->paused;
    } else if (key == GLFW_KEY_R) {
        const glm::vec3 offset = state->initialCamera.eye -
                                 state->initialCamera.target;
        state->target = state->initialCamera.target;
        state->distance = glm::length(offset);
        state->yaw = std::atan2(offset.x, offset.z);
        state->pitch = std::asin(glm::clamp(
            offset.y / std::max(state->distance, 1.0e-6F), -1.0F, 1.0F));
    }
}

GLuint compileShader(GLenum type, const char* source,
                     std::string& diagnostic) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)),
                        '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        diagnostic = "OpenGL shader compilation failed: " + log;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

GLuint buildProgram(std::string& diagnostic) {
    constexpr const char* vertexSource = R"GLSL(
#version 410 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
uniform mat4 uViewProjection;
out vec3 vNormal;
out vec3 vColor;
void main() {
    vNormal = aNormal;
    vColor = aColor;
    gl_Position = uViewProjection * vec4(aPosition, 1.0);
}
)GLSL";
    constexpr const char* fragmentSource = R"GLSL(
#version 410 core
in vec3 vNormal;
in vec3 vColor;
out vec4 fragColor;
void main() {
    vec3 normal = normalize(vNormal);
    vec3 light = normalize(vec3(-0.35, 0.75, 0.56));
    float lambert = abs(dot(normal, light));
    fragColor = vec4(vColor * (0.26 + 0.74 * lambert), 1.0);
}
)GLSL";

    const GLuint vertex = compileShader(GL_VERTEX_SHADER, vertexSource,
                                        diagnostic);
    if (vertex == 0) {
        return 0;
    }
    const GLuint fragment = compileShader(GL_FRAGMENT_SHADER, fragmentSource,
                                          diagnostic);
    if (fragment == 0) {
        glDeleteShader(vertex);
        return 0;
    }
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    GLint status = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)),
                        '\0');
        glGetProgramInfoLog(program, length, nullptr, log.data());
        diagnostic = "OpenGL program link failed: " + log;
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

Camera orbitCamera(const PreviewState& state) {
    Camera camera = state.initialCamera;
    camera.target = state.target;
    const float cosPitch = std::cos(state.pitch);
    const glm::vec3 offset{
        state.distance * cosPitch * std::sin(state.yaw),
        state.distance * std::sin(state.pitch),
        state.distance * cosPitch * std::cos(state.yaw),
    };
    camera.eye = state.target + offset;
    camera.up = glm::vec3(0.0F, 1.0F, 0.0F);
    return camera;
}

} // namespace

PreviewResult runPreview(const Asset& asset, const RenderSettings& settings,
                         std::uint32_t initialFrame, bool forceOffline) {
    PreviewResult result;
    if (forceOffline) {
        result.diagnostic =
            "OpenGL preview disabled by --force-offline";
        result.error = makeError(ErrorCode::OpenGLUnavailable,
                                 result.diagnostic);
        return result;
    }

    glfwError.clear();
    glfwSetErrorCallback(glfwErrorCallback);
    if (glfwInit() == GLFW_FALSE) {
        result.diagnostic =
            "GLFW initialization failed: " +
            (glfwError.empty() ? std::string("no display available")
                               : glfwError);
        result.error = makeError(ErrorCode::OpenGLUnavailable,
                                 result.diagnostic);
        return result;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    GLFWwindow* window =
        glfwCreateWindow(1024, 768, "SkelPreview - OpenGL 4.1",
                         nullptr, nullptr);
    if (window == nullptr) {
        result.diagnostic =
            "OpenGL 4.1 window creation failed: " +
            (glfwError.empty() ? std::string("unknown GLFW error")
                               : glfwError);
        result.error = makeError(ErrorCode::OpenGLUnavailable,
                                 result.diagnostic);
        glfwTerminate();
        return result;
    }
    result.windowOpened = true;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    const int major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    const int minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    if (major < 4 || (major == 4 && minor < 1)) {
        result.diagnostic =
            "context is below OpenGL 4.1 (reported " +
            std::to_string(major) + "." + std::to_string(minor) + ")";
        result.error = makeError(ErrorCode::OpenGLUnavailable,
                                 result.diagnostic);
        glfwDestroyWindow(window);
        glfwTerminate();
        return result;
    }

    std::string loaderDiagnostic;
    if (!gl4::load(loaderDiagnostic)) {
        result.diagnostic = loaderDiagnostic;
        result.error = makeError(ErrorCode::OpenGLUnavailable,
                                 loaderDiagnostic);
        glfwDestroyWindow(window);
        glfwTerminate();
        return result;
    }

    std::string shaderDiagnostic;
    const GLuint program = buildProgram(shaderDiagnostic);
    if (program == 0) {
        result.diagnostic = shaderDiagnostic;
        result.error = makeError(ErrorCode::OpenGLUnavailable,
                                 shaderDiagnostic);
        glfwDestroyWindow(window);
        glfwTerminate();
        return result;
    }

    BakedFrame baked;
    Error bakeError;
    const std::uint32_t startFrame =
        std::min(initialFrame, asset.frameCount - 1U);
    if (!evaluateFrame(asset, startFrame, baked, bakeError)) {
        result.diagnostic = "frame evaluation failed: " + bakeError.message;
        result.error = bakeError;
        glDeleteProgram(program);
        glfwDestroyWindow(window);
        glfwTerminate();
        return result;
    }

    GLuint vertexArray = 0;
    GLuint vertexBuffer = 0;
    GLuint indexBuffer = 0;
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(
                     baked.positions.size() * sizeof(InterleavedVertex)),
                 nullptr, GL_DYNAMIC_DRAW);
    glGenBuffers(1, &indexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(baked.indices.size() *
                                         sizeof(std::uint32_t)),
                 baked.indices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          sizeof(InterleavedVertex),
                          reinterpret_cast<const void*>(
                              offsetof(InterleavedVertex, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 3, GL_FLOAT, GL_FALSE, sizeof(InterleavedVertex),
        reinterpret_cast<const void*>(offsetof(InterleavedVertex, normal)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(
        2, 3, GL_FLOAT, GL_FALSE, sizeof(InterleavedVertex),
        reinterpret_cast<const void*>(offsetof(InterleavedVertex, color)));

    PreviewState state;
    state.window = window;
    state.initialCamera = settings.camera;
    state.target = settings.camera.target;
    const glm::vec3 initialOffset =
        settings.camera.eye - settings.camera.target;
    state.distance = std::max(glm::length(initialOffset), 0.25F);
    state.yaw = std::atan2(initialOffset.x, initialOffset.z);
    state.pitch = std::asin(glm::clamp(initialOffset.y / state.distance,
                                       -1.0F, 1.0F));
    glfwSetWindowUserPointer(window, &state);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);

    GLint viewProjectionLocation =
        glGetUniformLocation(program, "uViewProjection");
    glUseProgram(program);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    std::uint32_t currentFrame = startFrame;
    auto upload = [&]() {
        const std::vector<InterleavedVertex> vertices =
            makeInterleavedVertices(baked);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(
                         vertices.size() * sizeof(InterleavedVertex)),
                     vertices.data(), GL_DYNAMIC_DRAW);
    };
    upload();

    double animationDeadline =
        glfwGetTime() + 1.0 / asset.framesPerSecond;
    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        glfwPollEvents();
        if (!state.paused && glfwGetTime() >= animationDeadline) {
            currentFrame = (currentFrame + 1U) % asset.frameCount;
            Error frameError;
            if (!evaluateFrame(asset, currentFrame, baked, frameError)) {
                result.diagnostic =
                    "animation evaluation failed: " + frameError.message;
                result.error = frameError;
                break;
            }
            upload();
            animationDeadline =
                glfwGetTime() + 1.0 / asset.framesPerSecond;
        }

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
        if (framebufferWidth <= 0 || framebufferHeight <= 0) {
            glfwWaitEvents();
            continue;
        }
        glViewport(0, 0, framebufferWidth, framebufferHeight);
        glClearColor(14.0F / 255.0F, 17.0F / 255.0F, 22.0F / 255.0F,
                     1.0F);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const Camera camera = orbitCamera(state);
        glm::mat4 view;
        glm::mat4 projection;
        Error cameraError;
        if (!cameraMatrices(camera,
                            static_cast<std::uint32_t>(framebufferWidth),
                            static_cast<std::uint32_t>(framebufferHeight),
                            view, projection, cameraError)) {
            result.diagnostic =
                "camera update failed: " + cameraError.message;
            result.error = cameraError;
            break;
        }
        const glm::mat4 viewProjection = projection * view;
        glUniformMatrix4fv(viewProjectionLocation, 1, GL_FALSE,
                           glm::value_ptr(viewProjection));
        glPolygonMode(GL_FRONT_AND_BACK,
                      state.wireframe ? GL_LINE : GL_FILL);
        glDrawElements(GL_TRIANGLES,
                       static_cast<GLsizei>(baked.indices.size()),
                       GL_UNSIGNED_INT, nullptr);
        glfwSwapBuffers(window);
    }

    if (result.error.code == ErrorCode::Ok) {
        result.completed = true;
    }
    glDeleteBuffers(1, &indexBuffer);
    glDeleteBuffers(1, &vertexBuffer);
    glDeleteVertexArrays(1, &vertexArray);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}

} // namespace skel
