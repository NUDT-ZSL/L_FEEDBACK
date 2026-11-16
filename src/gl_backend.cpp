#include "gl_backend.hpp"

#include <glad/gl.h>

#include <sstream>

#if defined(SKINRENDER_HAVE_EGL)
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <dlfcn.h>
#endif

#if defined(SKINRENDER_HAVE_GLFW)
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#endif

namespace skin {
namespace {

std::string hexValue(unsigned int value) {
    std::ostringstream output;
    output << "0x" << std::hex << value;
    return output.str();
}

#if defined(SKINRENDER_HAVE_EGL)

OpenGlProc eglGlProc(const char* name) {
    OpenGlProc function = reinterpret_cast<OpenGlProc>(
        eglGetProcAddress(name)
    );
    if (function == nullptr) {
        function = reinterpret_cast<OpenGlProc>(dlsym(RTLD_DEFAULT, name));
    }
    return function;
}

class EglContext final : public GlContext {
public:
    ~EglContext() override { shutdown(); }

    bool create(std::string& error) {
        PFNEGLGETPLATFORMDISPLAYEXTPROC getPlatformDisplay =
            reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
                eglGetProcAddress("eglGetPlatformDisplayEXT")
            );

        if (getPlatformDisplay != nullptr) {
            display_ = getPlatformDisplay(
                EGL_PLATFORM_SURFACELESS_MESA,
                EGL_DEFAULT_DISPLAY,
                nullptr
            );
        }
        if (display_ == EGL_NO_DISPLAY) {
            display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        }
        if (display_ == EGL_NO_DISPLAY) {
            error = "EGL could not obtain a display (surfaceless and default failed)";
            return false;
        }

        if (eglInitialize(display_, nullptr, nullptr) != EGL_TRUE) {
            error = "eglInitialize failed: " + hexValue(eglGetError());
            display_ = EGL_NO_DISPLAY;
            return false;
        }
        initialized_ = true;

        if (eglBindAPI(EGL_OPENGL_API) != EGL_TRUE) {
            error = "eglBindAPI(EGL_OPENGL_API) failed: "
                + hexValue(eglGetError());
            return false;
        }

        const EGLint configAttributes[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_DEPTH_SIZE, 24,
            EGL_NONE,
        };
        EGLConfig config = nullptr;
        EGLint configCount = 0;
        if (eglChooseConfig(
                display_,
                configAttributes,
                &config,
                1,
                &configCount
            ) != EGL_TRUE
            || configCount < 1) {
            error = "eglChooseConfig found no OpenGL 4.x configuration: "
                + hexValue(eglGetError());
            return false;
        }

        const EGLint contextAttributes[] = {
            EGL_CONTEXT_MAJOR_VERSION_KHR, 4,
            EGL_CONTEXT_MINOR_VERSION_KHR, 3,
            EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR,
            EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR,
            EGL_NONE,
        };
        context_ = eglCreateContext(
            display_,
            config,
            EGL_NO_CONTEXT,
            contextAttributes
        );
        if (context_ == EGL_NO_CONTEXT) {
            error = "eglCreateContext(OpenGL 4.3 core) failed: "
                + hexValue(eglGetError());
            return false;
        }

        const EGLint pbufferAttributes[] = {
            EGL_WIDTH, 1,
            EGL_HEIGHT, 1,
            EGL_NONE,
        };
        surface_ = eglCreatePbufferSurface(display_, config, pbufferAttributes);
        if (surface_ != EGL_NO_SURFACE) {
            if (eglMakeCurrent(display_, surface_, surface_, context_)
                == EGL_TRUE) {
                makeCurrentSucceeded_ = true;
            } else {
                eglDestroySurface(display_, surface_);
                surface_ = EGL_NO_SURFACE;
            }
        }
        if (!makeCurrentSucceeded_) {
            if (eglMakeCurrent(
                    display_,
                    EGL_NO_SURFACE,
                    EGL_NO_SURFACE,
                    context_
                ) != EGL_TRUE) {
                error = "eglMakeCurrent failed with and without a pbuffer: "
                    + hexValue(eglGetError());
                return false;
            }
            makeCurrentSucceeded_ = true;
        }

        if (!loadOpenGl(&eglGlProc, error)) {
            return false;
        }

        info_.backend = "egl-surfaceless";
        return true;
    }

    void makeCurrent() override {
        if (context_ != EGL_NO_CONTEXT && initialized_) {
            eglMakeCurrent(
                display_,
                surface_,
                surface_,
                context_
            );
        }
    }

    void shutdown() override {
        if (shutdownComplete_) {
            return;
        }
        shutdownComplete_ = true;
        if (display_ != EGL_NO_DISPLAY && context_ != EGL_NO_CONTEXT) {
            eglMakeCurrent(
                display_,
                EGL_NO_SURFACE,
                EGL_NO_SURFACE,
                EGL_NO_CONTEXT
            );
            if (surface_ != EGL_NO_SURFACE) {
                eglDestroySurface(display_, surface_);
                surface_ = EGL_NO_SURFACE;
            }
            eglDestroyContext(display_, context_);
            context_ = EGL_NO_CONTEXT;
            eglReleaseThread();
        }
        if (display_ != EGL_NO_DISPLAY && initialized_) {
            eglTerminate(display_);
        }
        display_ = EGL_NO_DISPLAY;
        initialized_ = false;
    }

private:
    EGLDisplay display_ = EGL_NO_DISPLAY;
    EGLSurface surface_ = EGL_NO_SURFACE;
    EGLContext context_ = EGL_NO_CONTEXT;
    bool initialized_ = false;
    bool makeCurrentSucceeded_ = false;
    bool shutdownComplete_ = false;
};

#endif  // SKINRENDER_HAVE_EGL

#if defined(SKINRENDER_HAVE_GLFW)

OpenGlProc glfwGlProc(const char* name) {
    return reinterpret_cast<OpenGlProc>(glfwGetProcAddress(name));
}

class GlfwContext final : public GlContext {
public:
    ~GlfwContext() override { shutdown(); }

    bool create(std::string& error) {
        if (glfwInit() != GLFW_TRUE) {
            const char* description = nullptr;
            glfwGetError(&description);
            error = std::string("glfwInit failed")
                + (description != nullptr
                    ? std::string(": ") + description
                    : std::string());
            return false;
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

        window_ = glfwCreateWindow(
            1,
            1,
            "skin_batch_renderer_offscreen",
            nullptr,
            nullptr
        );
        if (window_ == nullptr) {
            const char* description = nullptr;
            glfwGetError(&description);
            error = "glfwCreateWindow(OpenGL 4.3 core) failed"
                + (description != nullptr
                    ? std::string(": ") + description
                    : std::string());
            return false;
        }

        glfwMakeContextCurrent(window_);
        glfwSwapInterval(0);
        if (!loadOpenGl(&glfwGlProc, error)) {
            return false;
        }
        info_.backend = "glfw-hidden-window";
        return true;
    }

    void makeCurrent() override {
        if (window_ != nullptr) {
            glfwMakeContextCurrent(window_);
        }
    }

    void shutdown() override {
        if (shutdownComplete_) {
            return;
        }
        shutdownComplete_ = true;
        if (window_ != nullptr) {
            glfwDestroyWindow(window_);
            window_ = nullptr;
        }
        if (initialized_) {
            glfwTerminate();
            initialized_ = false;
        }
    }

private:
    GLFWwindow* window_ = nullptr;
    bool initialized_ = false;
    bool shutdownComplete_ = false;
};

#endif  // SKINRENDER_HAVE_GLFW

}  // namespace

bool GlContext::loadOpenGl(
    OpenGlProcLoader loader,
    std::string& error
) {
    const int loadedVersion = gladLoadGL(loader);
    if (loadedVersion == 0) {
        error = "GLAD failed to load OpenGL function pointers";
        return false;
    }

    GLint major = 0;
    GLint minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    if (major < 4 || (major == 4 && minor < 3)) {
        std::ostringstream message;
        message << "OpenGL 4.3 or newer is required, got " << major << '.' << minor;
        error = message.str();
        return false;
    }

    const GLubyte* vendor = glGetString(GL_VENDOR);
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* version = glGetString(GL_VERSION);
    const GLubyte* glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);
    info_.vendor = vendor != nullptr
        ? reinterpret_cast<const char*>(vendor)
        : "unknown";
    info_.renderer = renderer != nullptr
        ? reinterpret_cast<const char*>(renderer)
        : "unknown";
    info_.version = version != nullptr
        ? reinterpret_cast<const char*>(version)
        : "unknown";
    info_.glslVersion = glslVersion != nullptr
        ? reinterpret_cast<const char*>(glslVersion)
        : "unknown";
    info_.major = static_cast<int>(major);
    info_.minor = static_cast<int>(minor);
    return true;
}

std::unique_ptr<GlContext> GlContext::create(
    const std::string& requestedBackend,
    std::string& error
) {
    const bool automatic = requestedBackend.empty()
        || requestedBackend == "auto";
    std::string combinedErrors;

#if defined(SKINRENDER_HAVE_EGL)
    if (automatic || requestedBackend == "egl") {
        auto context = std::make_unique<EglContext>();
        std::string eglError;
        if (context->create(eglError)) {
            return context;
        }
        combinedErrors += "EGL: " + eglError + "; ";
        if (requestedBackend == "egl") {
            error = combinedErrors;
            return nullptr;
        }
    }
#else
    if (requestedBackend == "egl") {
        error = "EGL backend is not compiled into this build";
        return nullptr;
    }
#endif

#if defined(SKINRENDER_HAVE_GLFW)
    if (automatic || requestedBackend == "glfw") {
        auto context = std::make_unique<GlfwContext>();
        std::string glfwError;
        if (context->create(glfwError)) {
            return context;
        }
        combinedErrors += "GLFW: " + glfwError;
        error = combinedErrors;
        return nullptr;
    }
#else
    if (requestedBackend == "glfw") {
        error = "GLFW backend is not compiled into this build";
        return nullptr;
    }
#endif

    if (automatic) {
        error = combinedErrors.empty()
            ? "no OpenGL backend is available in this build"
            : combinedErrors;
    } else {
        error = "unknown backend: " + requestedBackend;
    }
    return nullptr;
}

}  // namespace skin

