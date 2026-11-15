#include "gl_backend.h"

#include <glad/gl.h>

#include <sstream>
#include <string>

#ifdef _WIN32
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#else
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <dlfcn.h>
#endif

namespace pbr {
namespace {

#ifdef _WIN32
std::string g_glfwError;

void glfwErrorCallback(int code, const char* description) {
    std::ostringstream output;
    output << "GLFW error " << code << ": " << (description != nullptr ? description : "unknown");
    g_glfwError = output.str();
}
#else
GLADapiproc loadGlProc(const char* name) {
    auto proc = reinterpret_cast<GLADapiproc>(eglGetProcAddress(name));
    if (proc == nullptr) {
        proc = reinterpret_cast<GLADapiproc>(dlsym(RTLD_DEFAULT, name));
    }
    return proc;
}
#endif

}  // namespace

struct GLContext::Impl {
#ifdef _WIN32
    GLFWwindow* window = nullptr;
#else
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLSurface surface = EGL_NO_SURFACE;
    EGLContext context = EGL_NO_CONTEXT;
#endif
};

GLContext::GLContext() : impl_(std::make_unique<Impl>()) {}

GLContext::~GLContext() {
#ifdef _WIN32
    if (impl_->window != nullptr) {
        glfwMakeContextCurrent(nullptr);
        glfwDestroyWindow(impl_->window);
    }
    glfwTerminate();
#else
    if (impl_->display != EGL_NO_DISPLAY) {
        eglMakeCurrent(impl_->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (impl_->surface != EGL_NO_SURFACE) {
            eglDestroySurface(impl_->display, impl_->surface);
        }
        if (impl_->context != EGL_NO_CONTEXT) {
            eglDestroyContext(impl_->display, impl_->context);
        }
        eglTerminate(impl_->display);
    }
#endif
}

bool GLContext::initialize(int width, int height, std::string& error) {
#ifdef _WIN32
    glfwSetErrorCallback(glfwErrorCallback);
    if (glfwInit() == GLFW_FALSE) {
        error = "cannot initialize GLFW: " + g_glfwError;
        return false;
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    impl_->window = glfwCreateWindow(width, height, "PBR Offline Renderer", nullptr, nullptr);
    if (impl_->window == nullptr) {
        error = "cannot create an OpenGL 4.5 core context: " + g_glfwError;
        return false;
    }
    glfwMakeContextCurrent(impl_->window);
    if (gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress)) == 0) {
        error = "GLAD failed to load OpenGL function pointers";
        return false;
    }
#else
    impl_->display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    auto getPlatformDisplay =
        reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (getPlatformDisplay != nullptr) {
        EGLDisplay surfaceless =
            getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
        if (surfaceless != EGL_NO_DISPLAY) {
            impl_->display = surfaceless;
        }
    }
    if (impl_->display == EGL_NO_DISPLAY) {
        error = "cannot obtain an EGL display";
        return false;
    }
    EGLint major = 0;
    EGLint minor = 0;
    if (eglInitialize(impl_->display, &major, &minor) == EGL_FALSE) {
        error = "cannot initialize EGL";
        return false;
    }
    if (eglBindAPI(EGL_OPENGL_API) == EGL_FALSE) {
        error = "cannot bind the EGL desktop OpenGL API";
        return false;
    }

    const EGLint configAttributes[] = {
        EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_BIT,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_ALPHA_SIZE,
        8,
        EGL_NONE,
    };
    EGLConfig config = nullptr;
    EGLint configCount = 0;
    if (eglChooseConfig(impl_->display, configAttributes, &config, 1, &configCount) == EGL_FALSE ||
        configCount < 1) {
        error = "cannot choose an EGL OpenGL pbuffer configuration";
        return false;
    }
    const EGLint surfaceAttributes[] = {
        EGL_WIDTH,
        width,
        EGL_HEIGHT,
        height,
        EGL_NONE,
    };
    impl_->surface = eglCreatePbufferSurface(impl_->display, config, surfaceAttributes);
    if (impl_->surface == EGL_NO_SURFACE) {
        error = "cannot create an EGL pbuffer surface";
        return false;
    }
    const EGLint contextAttributes[] = {
        EGL_CONTEXT_MAJOR_VERSION,
        4,
        EGL_CONTEXT_MINOR_VERSION,
        5,
        EGL_CONTEXT_OPENGL_PROFILE_MASK,
        EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE,
    };
    impl_->context = eglCreateContext(impl_->display, config, EGL_NO_CONTEXT, contextAttributes);
    if (impl_->context == EGL_NO_CONTEXT) {
        error = "cannot create an EGL OpenGL 4.5 core context";
        return false;
    }
    if (eglMakeCurrent(impl_->display, impl_->surface, impl_->surface, impl_->context) == EGL_FALSE) {
        error = "cannot make the EGL OpenGL context current";
        return false;
    }
    if (gladLoadGL(loadGlProc) == 0) {
        error = "GLAD failed to load OpenGL function pointers";
        return false;
    }
#endif
    if (!GLAD_GL_VERSION_4_5) {
        const unsigned char* version = glGetString(GL_VERSION);
        error = "an OpenGL 4.5 core context is required; driver reports " +
                std::string(version != nullptr ? reinterpret_cast<const char*>(version) : "unknown");
        return false;
    }
    return true;
}

std::string GLContext::rendererSummary() const {
    const unsigned char* version = glGetString(GL_VERSION);
    const unsigned char* renderer = glGetString(GL_RENDERER);
    std::ostringstream output;
    output << (version != nullptr ? reinterpret_cast<const char*>(version) : "unknown OpenGL")
           << " / " << (renderer != nullptr ? reinterpret_cast<const char*>(renderer) : "unknown renderer");
    return output.str();
}

}  // namespace pbr
