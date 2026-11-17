#include "gl_context.h"

#include <array>
#include <cstdio>
#include <sstream>
#include <string>

#include <glad/gl.h>

#include "error.h"

#if defined(SR_PLATFORM_WINDOWS)
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#elif defined(SR_PLATFORM_LINUX_OSMESA)
#ifndef APIENTRY
#define APIENTRY
#endif
#include <GL/osmesa.h>
#endif

namespace sr {
namespace {

struct ContextInfo {
    std::string description;
    std::string version;
    std::string renderer;
};

ContextInfo loadGlAndDescribe(const std::string& backend,
                              GLADloadfunc loader) {
    const int loaded = gladLoadGL(loader);
    if (!loaded) {
        fail(ExitCode::Context, "E_GL_LOADER",
             "failed to load OpenGL function pointers for " + backend);
    }
    if (!GLAD_GL_VERSION_4_3) {
        const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
        fail(ExitCode::Context, "E_GL_VERSION",
             "OpenGL 4.3 is required, context reports: " +
                 std::string(version != nullptr ? version : "unknown"));
    }

    const auto* version = reinterpret_cast<const GLubyte*>(glGetString(GL_VERSION));
    const auto* renderer = reinterpret_cast<const GLubyte*>(glGetString(GL_RENDERER));
    const auto* vendor = reinterpret_cast<const GLubyte*>(glGetString(GL_VENDOR));
    ContextInfo info;
    info.version = version != nullptr ? reinterpret_cast<const char*>(version) : "unknown";
    info.renderer = renderer != nullptr ? reinterpret_cast<const char*>(renderer) : "unknown";
    const std::string vendorText =
        vendor != nullptr ? reinterpret_cast<const char*>(vendor) : "unknown";
    info.description = backend + " / " + vendorText + " / " + info.renderer +
                       " / GL " + info.version;
    return info;
}

#if defined(SR_PLATFORM_WINDOWS)

std::string g_lastGlfwError;

void glfwErrorCallback(int code, const char* description) {
    std::ostringstream message;
    message << "GLFW error " << code << ": "
            << (description != nullptr ? description : "unknown");
    g_lastGlfwError = message.str();
}

GLADapiproc glfwGladLoader(const char* name) {
    return reinterpret_cast<GLADapiproc>(glfwGetProcAddress(name));
}

class GlfwContext final : public GLContext {
public:
    GlfwContext() {
        glfwSetErrorCallback(glfwErrorCallback);
        if (!glfwInit()) {
            fail(ExitCode::Context, "E_CONTEXT_GLFW",
                 g_lastGlfwError.empty() ? "glfwInit failed" : g_lastGlfwError);
        }
        initialized_ = true;

        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
        glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
        glfwWindowHint(GLFW_SAMPLES, 0);

        window_ = glfwCreateWindow(1, 1, "skinned-csm-offscreen", nullptr, nullptr);
        if (window_ == nullptr) {
            const std::string message =
                g_lastGlfwError.empty() ? "glfwCreateWindow failed" : g_lastGlfwError;
            glfwTerminate();
            initialized_ = false;
            fail(ExitCode::Context, "E_CONTEXT_GLFW", message);
        }
        makeCurrent();
        info_ = loadGlAndDescribe("GLFW hidden window", glfwGladLoader);
    }

    ~GlfwContext() override {
        if (window_ != nullptr) {
            glfwDestroyWindow(window_);
            window_ = nullptr;
        }
        if (initialized_) {
            glfwTerminate();
        }
    }

    void makeCurrent() override {
        glfwMakeContextCurrent(window_);
    }

    std::string description() const override {
        return info_.description;
    }

    const std::string& version() const { return info_.version; }
    const std::string& renderer() const { return info_.renderer; }

private:
    GLFWwindow* window_ = nullptr;
    bool initialized_ = false;
    ContextInfo info_;
};

#elif defined(SR_PLATFORM_LINUX_OSMESA)

GLADapiproc osmesaGladLoader(const char* name) {
    return reinterpret_cast<GLADapiproc>(OSMesaGetProcAddress(name));
}

class OSMesaGLContext final : public GLContext {
public:
    OSMesaGLContext() {
        const int attributes[] = {
            OSMESA_FORMAT, OSMESA_RGBA,
            OSMESA_PROFILE, OSMESA_CORE_PROFILE,
            OSMESA_CONTEXT_MAJOR_VERSION, 4,
            OSMESA_CONTEXT_MINOR_VERSION, 3,
            0,
        };
        context_ = OSMesaCreateContextAttribs(attributes, nullptr);
        if (context_ == nullptr) {
            fail(ExitCode::Context, "E_CONTEXT_OSMESA",
                 "OSMesaCreateContextAttribs failed for a 4.3 core context");
        }
        makeCurrent();
        info_ = loadGlAndDescribe("OSMesa", osmesaGladLoader);
    }

    ~OSMesaGLContext() override {
        if (context_ != nullptr) {
            OSMesaDestroyContext(context_);
        }
    }

    void makeCurrent() override {
        if (!OSMesaMakeCurrent(context_, buffer_.data(), GL_UNSIGNED_BYTE, 1, 1)) {
            fail(ExitCode::Context, "E_CONTEXT_OSMESA", "OSMesaMakeCurrent failed");
        }
        OSMesaPixelStore(OSMESA_Y_UP, 1);
    }

    std::string description() const override {
        return info_.description;
    }

private:
    OSMesaContext context_ = nullptr;
    std::array<unsigned char, 4> buffer_{};
    ContextInfo info_;
};

#endif

} // namespace

std::unique_ptr<GLContext> createGLContext(ContextKind kind) {
    ContextKind selected = kind;
    if (selected == ContextKind::Auto) {
#if defined(SR_PLATFORM_WINDOWS)
        selected = ContextKind::Glfw;
#elif defined(SR_PLATFORM_LINUX_OSMESA)
        selected = ContextKind::OSMesa;
#else
        fail(ExitCode::Context, "E_CONTEXT_UNAVAILABLE",
             "no OpenGL context backend is compiled in");
#endif
    }

#if defined(SR_PLATFORM_WINDOWS)
    if (selected == ContextKind::OSMesa) {
        fail(ExitCode::Context, "E_CONTEXT_UNAVAILABLE",
             "OSMesa is not available in the Windows build; use --context glfw or auto");
    }
    return std::make_unique<GlfwContext>();
#elif defined(SR_PLATFORM_LINUX_OSMESA)
    if (selected == ContextKind::Glfw) {
        fail(ExitCode::Context, "E_CONTEXT_UNAVAILABLE",
             "the Linux validation build uses OSMesa; use --context osmesa or auto");
    }
    return std::make_unique<OSMesaGLContext>();
#else
    (void)selected;
    fail(ExitCode::Context, "E_CONTEXT_UNAVAILABLE",
         "no OpenGL context backend is available");
#endif
}

} // namespace sr
