#include "context.hpp"

#include "util.hpp"

#include <glad/gl.h>

#ifdef MATEVAL_HAVE_GLFW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#endif

#include <cstring>
#include <stdexcept>
#include <string>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace mv {
namespace {

std::string glString(GLenum name) {
    const auto* value = glGetString(name);
    return value == nullptr ? std::string{} : std::string(reinterpret_cast<const char*>(value));
}

void requireGl45() {
    GLint major = 0;
    GLint minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    if (major < 4 || (major == 4 && minor < 5)) {
        throw makeError(ExitCode::Runtime, "OPENGL_VERSION_UNSUPPORTED", "context",
                        "OpenGL 4.5 or newer is required; got " + std::to_string(major) + "." +
                            std::to_string(minor));
    }
}

ContextInfo queryInfo(const std::string& api) {
    ContextInfo info;
    info.api = api;
    info.version = glString(GL_VERSION);
    info.vendor = glString(GL_VENDOR);
    info.renderer = glString(GL_RENDERER);
    info.glslVersion = glString(GL_SHADING_LANGUAGE_VERSION);
    return info;
}

#ifdef MATEVAL_HAVE_GLFW
std::string g_glfwError;

void glfwErrorCallback(int code, const char* description) {
    g_glfwError = "GLFW error " + std::to_string(code) + ": " +
                  (description == nullptr ? "unknown" : description);
}
#endif

}  // namespace

struct GlContext::Impl {
    std::string apiName;
    ContextInfo contextInfo;
    std::string contextSignature;

#ifdef MATEVAL_HAVE_GLFW
    GLFWwindow* window = nullptr;
    bool glfwInitialized = false;
#endif

#ifndef _WIN32
    void* eglLibrary = nullptr;
    void* eglDisplay = nullptr;
    void* eglSurface = nullptr;
    void* eglContext = nullptr;

    using EglBoolean = unsigned int;
    using EglEnum = unsigned int;
    using EglInt = int;
    using EglDisplay = void*;
    using EglConfig = void*;
    using EglSurface = void*;
    using EglContext = void*;
    using EglGetProcAddress = void* (*)(const char*);
    using EglGetDisplay = EglDisplay (*)(void*);
    using EglGetPlatformDisplay = EglDisplay (*)(EglEnum, void*, const EglInt*);
    using EglInitialize = EglBoolean (*)(EglDisplay, EglInt*, EglInt*);
    using EglTerminate = EglBoolean (*)(EglDisplay);
    using EglBindAPI = EglBoolean (*)(EglEnum);
    using EglChooseConfig = EglBoolean (*)(EglDisplay, const EglInt*, EglConfig*, EglInt,
                                           EglInt*);
    using EglCreatePbufferSurface = EglSurface (*)(EglDisplay, EglConfig, const EglInt*);
    using EglCreateContext = EglContext (*)(EglDisplay, EglConfig, EglContext, const EglInt*);
    using EglMakeCurrent = EglBoolean (*)(EglDisplay, EglSurface, EglSurface, EglContext);
    using EglDestroySurface = EglBoolean (*)(EglDisplay, EglSurface);
    using EglDestroyContext = EglBoolean (*)(EglDisplay, EglContext);
    using EglReleaseThread = EglBoolean (*)(void);
    using EglGetError = EglInt (*)(void);

    EglGetProcAddress getProcAddress = nullptr;
    EglGetDisplay getDisplay = nullptr;
    EglGetPlatformDisplay getPlatformDisplay = nullptr;
    EglInitialize initialize = nullptr;
    EglTerminate terminate = nullptr;
    EglBindAPI bindApi = nullptr;
    EglChooseConfig chooseConfig = nullptr;
    EglCreatePbufferSurface createPbufferSurface = nullptr;
    EglCreateContext createContext = nullptr;
    EglMakeCurrent makeCurrent = nullptr;
    EglDestroySurface destroySurface = nullptr;
    EglDestroyContext destroyContext = nullptr;
    EglReleaseThread releaseThread = nullptr;
    EglGetError getError = nullptr;

    void* function(const char* name) {
        void* result = dlsym(eglLibrary, name);
        if (result == nullptr && getProcAddress != nullptr) {
            result = reinterpret_cast<void*>(getProcAddress(name));
        }
        return result;
    }

    void loadEglSymbols() {
        eglLibrary = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
        if (eglLibrary == nullptr) {
            throw makeError(ExitCode::Runtime, "EGL_LOAD_FAILED", "context",
                            "cannot load libEGL.so.1");
        }
        getProcAddress =
            reinterpret_cast<EglGetProcAddress>(dlsym(eglLibrary, "eglGetProcAddress"));
        getDisplay = reinterpret_cast<EglGetDisplay>(dlsym(eglLibrary, "eglGetDisplay"));
        getPlatformDisplay = reinterpret_cast<EglGetPlatformDisplay>(
            dlsym(eglLibrary, "eglGetPlatformDisplay"));
        if (getPlatformDisplay == nullptr && getProcAddress != nullptr) {
            getPlatformDisplay = reinterpret_cast<EglGetPlatformDisplay>(
                getProcAddress("eglGetPlatformDisplayEXT"));
        }
        initialize = reinterpret_cast<EglInitialize>(dlsym(eglLibrary, "eglInitialize"));
        terminate = reinterpret_cast<EglTerminate>(dlsym(eglLibrary, "eglTerminate"));
        bindApi = reinterpret_cast<EglBindAPI>(dlsym(eglLibrary, "eglBindAPI"));
        chooseConfig = reinterpret_cast<EglChooseConfig>(dlsym(eglLibrary, "eglChooseConfig"));
        createPbufferSurface = reinterpret_cast<EglCreatePbufferSurface>(
            dlsym(eglLibrary, "eglCreatePbufferSurface"));
        createContext =
            reinterpret_cast<EglCreateContext>(dlsym(eglLibrary, "eglCreateContext"));
        makeCurrent = reinterpret_cast<EglMakeCurrent>(dlsym(eglLibrary, "eglMakeCurrent"));
        destroySurface =
            reinterpret_cast<EglDestroySurface>(dlsym(eglLibrary, "eglDestroySurface"));
        destroyContext =
            reinterpret_cast<EglDestroyContext>(dlsym(eglLibrary, "eglDestroyContext"));
        releaseThread =
            reinterpret_cast<EglReleaseThread>(dlsym(eglLibrary, "eglReleaseThread"));
        getError = reinterpret_cast<EglGetError>(dlsym(eglLibrary, "eglGetError"));
        if (getProcAddress == nullptr || getDisplay == nullptr || initialize == nullptr ||
            terminate == nullptr || bindApi == nullptr || chooseConfig == nullptr ||
            createPbufferSurface == nullptr || createContext == nullptr ||
            makeCurrent == nullptr || destroySurface == nullptr || destroyContext == nullptr) {
            throw makeError(ExitCode::Runtime, "EGL_SYMBOL_FAILED", "context",
                            "libEGL.so.1 is missing required EGL entry points");
        }
    }

    std::string eglError() {
        if (getError == nullptr) {
            return "unknown EGL error";
        }
        auto code = getError();
        return "EGL error 0x" + [&] {
            constexpr char digits[] = "0123456789abcdef";
            std::string result(4, '0');
            for (int i = 3; i >= 0; --i) {
                result[static_cast<std::size_t>(i)] = digits[code & 0xf];
                code >>= 4;
            }
            return result;
        }();
    }

    void createEgl() {
        constexpr EglEnum kOpenGlApi = 0x30a2;
        constexpr EglEnum kPlatformSurfacelessMesa = 0x31dd;
        constexpr EglInt kSurfaceType = 0x3033;
        constexpr EglInt kPbufferBit = 0x0001;
        constexpr EglInt kRenderableType = 0x3040;
        constexpr EglInt kOpenGlBit = 0x0008;
        constexpr EglInt kRedSize = 0x3024;
        constexpr EglInt kGreenSize = 0x3025;
        constexpr EglInt kBlueSize = 0x3026;
        constexpr EglInt kAlphaSize = 0x3021;
        constexpr EglInt kNone = 0x3038;
        constexpr EglInt kContextMajor = 0x3098;
        constexpr EglInt kContextMinor = 0x30fb;
        constexpr EglInt kContextProfileMask = 0x30fd;
        constexpr EglInt kContextCoreProfileBit = 0x00000001;

        loadEglSymbols();
        if (getPlatformDisplay != nullptr) {
            eglDisplay =
                getPlatformDisplay(kPlatformSurfacelessMesa, nullptr, nullptr);
        }
        if (eglDisplay == nullptr) {
            eglDisplay = getDisplay(nullptr);
        }
        if (eglDisplay == nullptr) {
            throw makeError(ExitCode::Runtime, "EGL_DISPLAY_FAILED", "context",
                            "no EGL display is available");
        }
        EglInt major = 0;
        EglInt minor = 0;
        if (!initialize(eglDisplay, &major, &minor)) {
            throw makeError(ExitCode::Runtime, "EGL_INITIALIZE_FAILED", "context",
                            "eglInitialize failed: " + eglError());
        }
        if (!bindApi(kOpenGlApi)) {
            throw makeError(ExitCode::Runtime, "EGL_BIND_FAILED", "context",
                            "eglBindAPI(EGL_OPENGL_API) failed: " + eglError());
        }

        const EglInt configAttributes[] = {
            kSurfaceType, kPbufferBit, kRenderableType, kOpenGlBit,
            kRedSize,     8,           kGreenSize,       8,
            kBlueSize,    8,           kAlphaSize,       8,
            kNone,
        };
        EglConfig config = nullptr;
        EglInt configCount = 0;
        if (!chooseConfig(eglDisplay, configAttributes, &config, 1, &configCount) ||
            configCount == 0) {
            throw makeError(ExitCode::Runtime, "EGL_CONFIG_FAILED", "context",
                            "no EGL OpenGL pbuffer configuration: " + eglError());
        }
        const EglInt surfaceAttributes[] = {0x3057, 1, 0x3056, 1, kNone};
        eglSurface = createPbufferSurface(eglDisplay, config, surfaceAttributes);
        if (eglSurface == nullptr) {
            throw makeError(ExitCode::Runtime, "EGL_SURFACE_FAILED", "context",
                            "eglCreatePbufferSurface failed: " + eglError());
        }
        const EglInt contextAttributes[] = {kContextMajor, 4, kContextMinor, 5,
                                            kContextProfileMask,
                                            kContextCoreProfileBit, kNone};
        eglContext = createContext(eglDisplay, config, nullptr, contextAttributes);
        if (eglContext == nullptr) {
            throw makeError(ExitCode::Runtime, "EGL_CONTEXT_FAILED", "context",
                            "OpenGL 4.5 EGL context creation failed: " + eglError());
        }
        if (!makeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            throw makeError(ExitCode::Runtime, "EGL_CURRENT_FAILED", "context",
                            "eglMakeCurrent failed: " + eglError());
        }
        if (gladLoadGL(reinterpret_cast<GLADloadfunc>(getProcAddress)) == 0) {
            throw makeError(ExitCode::Runtime, "GLAD_LOAD_FAILED", "context",
                            "GLAD could not load the OpenGL entry points");
        }
        requireGl45();
        apiName = "egl-surfaceless";
    }
#endif

    ~Impl() {
#ifdef MATEVAL_HAVE_GLFW
        if (window != nullptr) {
            glfwDestroyWindow(window);
            window = nullptr;
        }
        if (glfwInitialized) {
            glfwTerminate();
            glfwInitialized = false;
        }
#endif
#ifndef _WIN32
        if (eglDisplay != nullptr && makeCurrent != nullptr) {
            makeCurrent(eglDisplay, nullptr, nullptr, nullptr);
            if (eglContext != nullptr && destroyContext != nullptr) {
                destroyContext(eglDisplay, eglContext);
            }
            if (eglSurface != nullptr && destroySurface != nullptr) {
                destroySurface(eglDisplay, eglSurface);
            }
            if (releaseThread != nullptr) {
                releaseThread();
            }
            if (terminate != nullptr) {
                terminate(eglDisplay);
            }
        }
        if (eglLibrary != nullptr) {
            dlclose(eglLibrary);
        }
#endif
    }
};

GlContext::GlContext(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

GlContext::~GlContext() = default;

const ContextInfo& GlContext::info() const {
    return impl_->contextInfo;
}

const std::string& GlContext::signature() const {
    return impl_->contextSignature;
}

std::unique_ptr<GlContext> GlContext::create() {
    auto impl = std::make_unique<Impl>();
    std::string errors;

#ifdef MATEVAL_HAVE_GLFW
    try {
        glfwSetErrorCallback(glfwErrorCallback);
        if (!glfwInit()) {
            throw makeError(ExitCode::Runtime, "GLFW_INIT_FAILED", "context",
                            g_glfwError.empty() ? "glfwInit failed" : g_glfwError);
        }
        impl->glfwInitialized = true;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        impl->window = glfwCreateWindow(1, 1, "material-verify", nullptr, nullptr);
        if (impl->window == nullptr) {
            throw makeError(ExitCode::Runtime, "GLFW_WINDOW_FAILED", "context",
                            g_glfwError.empty() ? "glfwCreateWindow failed" : g_glfwError);
        }
        glfwMakeContextCurrent(impl->window);
        if (gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress)) == 0) {
            throw makeError(ExitCode::Runtime, "GLAD_LOAD_FAILED", "context",
                            "GLAD could not load the OpenGL entry points");
        }
        requireGl45();
        impl->apiName = "glfw-hidden-window";
        impl->contextInfo = queryInfo(impl->apiName);
    } catch (const MaterialError& error) {
        errors = error.message;
        impl->~Impl();
        new (impl.get()) Impl();
    }
#endif

#ifndef _WIN32
    if (impl->contextInfo.version.empty()) {
        try {
            impl->createEgl();
            impl->contextInfo = queryInfo(impl->apiName);
        } catch (const MaterialError& error) {
            if (!errors.empty()) {
                errors += "; ";
            }
            errors += error.message;
        }
    }
#endif

    if (impl->contextInfo.version.empty()) {
        throw makeError(ExitCode::Runtime, "GL_CONTEXT_FAILED", "context",
                        errors.empty() ? "no usable OpenGL 4.5 context" : errors);
    }
    impl->contextSignature =
        sha256OfStrings({impl->contextInfo.vendor, impl->contextInfo.renderer,
                         impl->contextInfo.version, impl->contextInfo.glslVersion});
    return std::unique_ptr<GlContext>(new GlContext(std::move(impl)));
}

}  // namespace mv
