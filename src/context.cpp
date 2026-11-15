#include "context.hpp"

#include "error.hpp"

#include <sstream>

namespace scene {
namespace {

std::string g_lastGlfwError;

void glfwErrorCallback(int code, const char* description) {
    std::ostringstream message;
    message << "GLFW error " << code;
    if (description != nullptr) {
        message << ": " << description;
    }
    g_lastGlfwError = message.str();
}

int majorVersion(const char* version) {
    if (version == nullptr) {
        return 0;
    }
    int major = 0;
    int minor = 0;
    std::istringstream stream(version);
    stream >> major;
    stream.get();
    stream >> minor;
    return major * 100 + minor;
}

} // namespace

OffscreenContext::OffscreenContext() {
    glfwSetErrorCallback(glfwErrorCallback);
    g_lastGlfwError.clear();
    if (glfwInit() != GLFW_TRUE) {
        throw AppError(ErrorType::Context,
                       "failed to initialize GLFW: " +
                       (g_lastGlfwError.empty() ? "unknown GLFW error" : g_lastGlfwError));
    }
    glfwInitialized_ = true;

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(SCENE_OSMESA_CONTEXT)
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_OSMESA_CONTEXT_API);
#endif

    window_ = glfwCreateWindow(1, 1, "offscreen-renderer-context",
                               nullptr, nullptr);
    if (window_ == nullptr) {
        glfwTerminate();
        glfwInitialized_ = false;
        throw AppError(ErrorType::Context,
                       "failed to create a hidden OpenGL 3.3 context: " +
                       (g_lastGlfwError.empty() ? "GLFW returned null" : g_lastGlfwError));
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(0);

    glewExperimental = GL_TRUE;
    const GLenum glewStatus = glewInit();
    if (glewStatus != GLEW_OK) {
        const std::string message =
            reinterpret_cast<const char*>(glewGetErrorString(glewStatus));
        glfwDestroyWindow(window_);
        window_ = nullptr;
        glfwTerminate();
        glfwInitialized_ = false;
        throw AppError(ErrorType::Context, "GLEW initialization failed: " + message);
    }

    const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    versionString_ = version != nullptr ? version : "unknown";
    rendererName_ = renderer != nullptr ? renderer : "unknown";
    if (majorVersion(version) < 303) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
        glfwTerminate();
        glfwInitialized_ = false;
        throw AppError(ErrorType::Context,
                       "OpenGL 3.3 or newer is required; context reports " +
                       versionString_);
    }

    while (glGetError() != GL_NO_ERROR) {
    }
}

OffscreenContext::~OffscreenContext() {
    if (window_ != nullptr) {
        glfwMakeContextCurrent(nullptr);
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    if (glfwInitialized_) {
        glfwTerminate();
        glfwInitialized_ = false;
    }
}

} // namespace scene

