#include "common.hpp"
#include "material.hpp"
#include "opengl_renderer.hpp"
#include "sample_generator.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <string>

namespace {

using EGLBoolean = unsigned int;
using EGLenum = unsigned int;
using EGLint = int;
using EGLDisplay = void*;
using EGLConfig = void*;
using EGLContext = void*;
using EGLSurface = void*;
using EGLProc = void (*)();

constexpr EGLint EGL_NONE = 0x3038;
constexpr EGLint EGL_SURFACE_TYPE = 0x3033;
constexpr EGLint EGL_PBUFFER_BIT = 0x0001;
constexpr EGLint EGL_RENDERABLE_TYPE = 0x3040;
constexpr EGLint EGL_OPENGL_BIT = 0x0008;
constexpr EGLint EGL_OPENGL_API = 0x30A2;
constexpr EGLint EGL_CONTEXT_MAJOR_VERSION = 0x3098;
constexpr EGLint EGL_CONTEXT_MINOR_VERSION = 0x30FB;
constexpr EGLint EGL_CONTEXT_OPENGL_PROFILE_MASK = 0x30FD;
constexpr EGLint EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = 0x00000001;
constexpr EGLint EGL_VERSION = 0x3054;
constexpr EGLint EGL_PLATFORM_SURFACELESS_MESA = 0x31DD;
constexpr EGLDisplay EGL_NO_DISPLAY = nullptr;
constexpr EGLContext EGL_NO_CONTEXT = nullptr;
constexpr EGLSurface EGL_NO_SURFACE = nullptr;

using PFNEGLGETPROCADDRESSPROC = EGLProc (*)(const char*);
using PFNEGLGETPLATFORMDISPLAYPROC = EGLDisplay (*)(EGLenum, void*, const EGLint*);
using PFNEGLGETDISPLAYPROC = EGLDisplay (*)(void*);
using PFNEGLINITIALIZEPROC = EGLBoolean (*)(EGLDisplay, EGLint*, EGLint*);
using PFNEGLTERMINATEPROC = EGLBoolean (*)(EGLDisplay);
using PFNEGLQUERYSTRINGPROC = const char* (*)(EGLDisplay, EGLint);
using PFNEGLGETERRORPROC = EGLint (*)();
using PFNEGLBINDAPIPROC = EGLBoolean (*)(EGLenum);
using PFNEGLCHOOSECONFIGPROC = EGLBoolean (*)(EGLDisplay, const EGLint*, EGLConfig*,
                                               EGLint, EGLint*);
using PFNEGLCREATECONTEXTPROC = EGLContext (*)(EGLDisplay, EGLConfig, EGLContext,
                                               const EGLint*);
using PFNEGLMAKECURRENTPROC = EGLBoolean (*)(EGLDisplay, EGLSurface, EGLSurface,
                                              EGLContext);
using PFNEGLDESTROYCONTEXTPROC = EGLBoolean (*)(EGLDisplay, EGLContext);
using PFNEGLRELEASETHREADPROC = EGLBoolean (*)();

struct EglApi {
    void* library = nullptr;
    PFNEGLGETPROCADDRESSPROC eglGetProcAddress = nullptr;
    PFNEGLGETPLATFORMDISPLAYPROC eglGetPlatformDisplay = nullptr;
    PFNEGLGETDISPLAYPROC eglGetDisplay = nullptr;
    PFNEGLINITIALIZEPROC eglInitialize = nullptr;
    PFNEGLTERMINATEPROC eglTerminate = nullptr;
    PFNEGLQUERYSTRINGPROC eglQueryString = nullptr;
    PFNEGLGETERRORPROC eglGetError = nullptr;
    PFNEGLBINDAPIPROC eglBindAPI = nullptr;
    PFNEGLCHOOSECONFIGPROC eglChooseConfig = nullptr;
    PFNEGLCREATECONTEXTPROC eglCreateContext = nullptr;
    PFNEGLMAKECURRENTPROC eglMakeCurrent = nullptr;
    PFNEGLDESTROYCONTEXTPROC eglDestroyContext = nullptr;
    PFNEGLRELEASETHREADPROC eglReleaseThread = nullptr;

    explicit EglApi(const char* libraryPath) {
        library = dlopen(libraryPath, RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr) {
            throw cr::AppError(cr::E_OPENGL_UNAVAILABLE, dlerror());
        }
        eglGetProcAddress = load<PFNEGLGETPROCADDRESSPROC>("eglGetProcAddress");
        eglGetPlatformDisplay = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYPROC>(
            dlsym(library, "eglGetPlatformDisplay"));
        if (eglGetPlatformDisplay == nullptr) {
            eglGetPlatformDisplay = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYPROC>(
                dlsym(library, "eglGetPlatformDisplayEXT"));
        }
        eglGetDisplay = load<PFNEGLGETDISPLAYPROC>("eglGetDisplay");
        eglInitialize = load<PFNEGLINITIALIZEPROC>("eglInitialize");
        eglTerminate = load<PFNEGLTERMINATEPROC>("eglTerminate");
        eglQueryString = load<PFNEGLQUERYSTRINGPROC>("eglQueryString");
        eglGetError = load<PFNEGLGETERRORPROC>("eglGetError");
        eglBindAPI = load<PFNEGLBINDAPIPROC>("eglBindAPI");
        eglChooseConfig = load<PFNEGLCHOOSECONFIGPROC>("eglChooseConfig");
        eglCreateContext = load<PFNEGLCREATECONTEXTPROC>("eglCreateContext");
        eglMakeCurrent = load<PFNEGLMAKECURRENTPROC>("eglMakeCurrent");
        eglDestroyContext = load<PFNEGLDESTROYCONTEXTPROC>("eglDestroyContext");
        eglReleaseThread = load<PFNEGLRELEASETHREADPROC>("eglReleaseThread");
    }

    ~EglApi() {
        if (library != nullptr) {
            dlclose(library);
        }
    }

    template <typename Function>
    Function load(const char* name) {
        void* symbol = dlsym(library, name);
        if (symbol == nullptr) {
            throw cr::AppError(cr::E_OPENGL_UNAVAILABLE,
                               std::string("missing EGL entry point ") + name);
        }
        return reinterpret_cast<Function>(symbol);
    }
};

struct EglContextGuard {
    EglApi& api;
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    bool initialized = false;

    ~EglContextGuard() {
        if (display != EGL_NO_DISPLAY) {
            api.eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        }
        if (display != EGL_NO_DISPLAY && context != EGL_NO_CONTEXT) {
            api.eglDestroyContext(display, context);
        }
        if (display != EGL_NO_DISPLAY && initialized) {
            api.eglTerminate(display);
        }
        if (api.eglReleaseThread != nullptr) {
            api.eglReleaseThread();
        }
    }
};

void createContext(EglApi& api, EglContextGuard& guard) {
    if (api.eglGetPlatformDisplay != nullptr) {
        guard.display = api.eglGetPlatformDisplay(
            EGL_PLATFORM_SURFACELESS_MESA, EGL_NO_DISPLAY, nullptr);
    }
    if (guard.display == EGL_NO_DISPLAY) {
        guard.display = api.eglGetDisplay(EGL_NO_DISPLAY);
    }
    if (guard.display == EGL_NO_DISPLAY) {
        throw cr::AppError(cr::E_OPENGL_UNAVAILABLE, "no EGL display is available");
    }
    EGLint major = 0;
    EGLint minor = 0;
    if (api.eglInitialize(guard.display, &major, &minor) == 0) {
        throw cr::AppError(cr::E_OPENGL_UNAVAILABLE,
                           "eglInitialize failed: " + std::to_string(api.eglGetError()));
    }
    guard.initialized = true;
    const char* version = api.eglQueryString(guard.display, EGL_VERSION);
    std::cerr << "EGL " << (version != nullptr ? version : "unknown") << ", " << major
              << "." << minor << '\n';
    if (api.eglBindAPI(static_cast<EGLenum>(EGL_OPENGL_API)) == 0) {
        throw cr::AppError(cr::E_OPENGL_UNAVAILABLE, "eglBindAPI(OpenGL) failed");
    }
    const EGLint configAttributes[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
    EGLConfig config = nullptr;
    EGLint configCount = 0;
    if (api.eglChooseConfig(guard.display, configAttributes, &config, 1, &configCount) == 0 ||
        configCount < 1) {
        throw cr::AppError(cr::E_OPENGL_UNAVAILABLE, "no EGL OpenGL config is available");
    }
    const EGLint contextAttributes[] = {
        EGL_CONTEXT_MAJOR_VERSION,
        4,
        EGL_CONTEXT_MINOR_VERSION,
        3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK,
        EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE};
    guard.context =
        api.eglCreateContext(guard.display, config, EGL_NO_CONTEXT, contextAttributes);
    if (guard.context == EGL_NO_CONTEXT) {
        throw cr::AppError(cr::E_OPENGL_UNAVAILABLE,
                           "eglCreateContext failed: " +
                               std::to_string(api.eglGetError()));
    }
    if (api.eglMakeCurrent(guard.display, EGL_NO_SURFACE, EGL_NO_SURFACE, guard.context) ==
        0) {
        throw cr::AppError(cr::E_OPENGL_UNAVAILABLE,
                           "eglMakeCurrent failed: " +
                               std::to_string(api.eglGetError()));
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const cr::fs::path dataDirectory =
            argc > 1 ? cr::fs::path(argv[1]) : cr::fs::path("build/test_run2/data");
        const cr::fs::path outputPath =
            argc > 2 ? cr::fs::path(argv[2])
                     : cr::fs::path("build/test_run2/output/gl_render.png");
        EglApi api("libEGL.so.1");
        EglContextGuard guard{api};
        createContext(api, guard);

        const std::vector<cr::MaterialSpec> specs =
            cr::parseMaterialDirectory(dataDirectory / "materials");
        const cr::MaterialLibrary materials = cr::loadMaterialLibrary(specs);
        const cr::Camera camera = cr::loadCamera(dataDirectory);
        const std::vector<cr::Light> lights = cr::loadDefaultLights(dataDirectory);
        const cr::gl::GlProcLoader loader =
            [&](const char* name) -> cr::gl::GlProc {
            return api.eglGetProcAddress(name);
        };
        const cr::Image image = cr::renderOpenGLDeferredWithCurrentContext(
            materials, camera, lights, loader);
        cr::savePngAtomic(outputPath, image);
        std::cout << "wrote " << outputPath.string() << '\n';

        const cr::fs::path cpuPath = outputPath.parent_path() / "render.png";
        if (cr::pathExists(cpuPath)) {
            const cr::Image cpu = cr::loadPng(cpuPath);
            int maximumDifference = 0;
            std::size_t differingChannels = 0;
            if (cpu.width == image.width && cpu.height == image.height) {
                for (std::size_t index = 0; index < cpu.pixels.size(); ++index) {
                    const int difference =
                        std::abs(static_cast<int>(cpu.pixels[index]) -
                                 static_cast<int>(image.pixels[index]));
                    maximumDifference = std::max(maximumDifference, difference);
                    if (difference > 1) {
                        ++differingChannels;
                    }
                }
            }
            std::cout << "cpu_vs_gl_max_channel_diff=" << maximumDifference
                      << " channels_over_1=" << differingChannels << '\n';
        }
        return 0;
    } catch (const cr::AppError& error) {
        std::cerr << error.code() << ": " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << cr::E_IO << ": " << error.what() << '\n';
        return 1;
    }
}

