#include "gl/gl_api.hpp"
#include "gl/gl_renderer.hpp"
#include "terrain/cache.hpp"
#include "terrain/lod.hpp"
#include "terrain/render_data.hpp"
#include "terrain/seams.hpp"
#include "terrain/terrain.hpp"

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

using EGLDisplay = void*;
using EGLConfig = void*;
using EGLContext = void*;
using EGLSurface = void*;
using EGLint = std::int32_t;
using EGLenum = std::uint32_t;
using EGLBoolean = std::uint32_t;
using EGLAttrib = std::intptr_t;

constexpr EGLDisplay kEglNoDisplay = nullptr;
constexpr EGLContext kEglNoContext = nullptr;
constexpr EGLSurface kEglNoSurface = nullptr;
constexpr EGLenum kEglPlatformSurfacelessMesa = 0x31DD;
constexpr EGLenum kEglPlatformX11 = 0x31D5;
constexpr EGLenum kEglOpenGlApi = 0x30A2;
constexpr EGLint kEglContextMajorVersion = 0x3098;
constexpr EGLint kEglContextMinorVersion = 0x30FB;
constexpr EGLint kEglContextOpenGlProfileMask = 0x30FD;
constexpr EGLint kEglContextOpenGlCoreProfileBit = 0x00000001;
constexpr EGLint kEglNone = 0x3038;
constexpr EGLint kEglSurfaceType = 0x3033;
constexpr EGLint kEglPbufferBit = 0x0001;
constexpr EGLint kEglRenderableType = 0x3040;
constexpr EGLint kEglOpenGlBit = 0x0008;
constexpr EGLint kEglRedSize = 0x3024;
constexpr EGLint kEglGreenSize = 0x3023;
constexpr EGLint kEglBlueSize = 0x3022;
constexpr EGLint kEglAlphaSize = 0x3021;
constexpr EGLint kEglDepthSize = 0x3025;
constexpr EGLint kEglWidth = 0x3057;
constexpr EGLint kEglHeight = 0x3056;

template <typename Function>
Function loadSymbol(void* library, const char* name) {
    return reinterpret_cast<Function>(dlsym(library, name));
}

}  // namespace

int main(int argc, char** argv) {
    const std::string output =
        argc > 1 ? argv[1] : "egl_smoke.ppm";
    void* library = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        std::cerr << "libEGL.so.1 is unavailable: " << dlerror() << "\n";
        return 77;
    }

    using GetPlatformDisplay = EGLDisplay (*)(EGLenum, void*, const EGLint*);
    using Initialize = EGLBoolean (*)(EGLDisplay, EGLint*, EGLint*);
    using BindApi = EGLBoolean (*)(EGLenum);
    using ChooseConfig = EGLBoolean (*)(
        EGLDisplay,
        const EGLint*,
        EGLConfig*,
        EGLint,
        EGLint*);
    using CreatePbufferSurface = EGLSurface (*)(
        EGLDisplay,
        EGLConfig,
        const EGLint*);
    using DestroySurface = EGLBoolean (*)(EGLDisplay, EGLSurface);
    using CreateContext = EGLContext (*)(
        EGLDisplay,
        EGLConfig,
        EGLContext,
        const EGLint*);
    using MakeCurrent = EGLBoolean (*)(
        EGLDisplay,
        EGLSurface,
        EGLSurface,
        EGLContext);
    using GetProcAddress = terrain::gl::ProcAddress (*)(const char*);
    using GetError = EGLint (*)();
    using DestroyContext = EGLBoolean (*)(EGLDisplay, EGLContext);
    using Terminate = EGLBoolean (*)(EGLDisplay);

    const auto initialize = loadSymbol<Initialize>(library, "eglInitialize");
    const auto bindApi = loadSymbol<BindApi>(library, "eglBindAPI");
    const auto chooseConfig =
        loadSymbol<ChooseConfig>(library, "eglChooseConfig");
    const auto createPbufferSurface =
        loadSymbol<CreatePbufferSurface>(library, "eglCreatePbufferSurface");
    const auto destroySurface =
        loadSymbol<DestroySurface>(library, "eglDestroySurface");
    const auto createContext = loadSymbol<CreateContext>(library, "eglCreateContext");
    const auto makeCurrent = loadSymbol<MakeCurrent>(library, "eglMakeCurrent");
    const auto getProcAddress = loadSymbol<GetProcAddress>(library, "eglGetProcAddress");
    const auto getError = loadSymbol<GetError>(library, "eglGetError");
    const auto destroyContext =
        loadSymbol<DestroyContext>(library, "eglDestroyContext");
    const auto terminate = loadSymbol<Terminate>(library, "eglTerminate");
    GetPlatformDisplay getPlatformDisplay = nullptr;
    if (getProcAddress != nullptr) {
        getPlatformDisplay = reinterpret_cast<GetPlatformDisplay>(
            getProcAddress("eglGetPlatformDisplayEXT"));
    }
    if (getPlatformDisplay == nullptr) {
        getPlatformDisplay =
            loadSymbol<GetPlatformDisplay>(library, "eglGetPlatformDisplay");
    }
    if (getPlatformDisplay == nullptr || initialize == nullptr ||
        bindApi == nullptr || chooseConfig == nullptr ||
        createPbufferSurface == nullptr || createContext == nullptr ||
        makeCurrent == nullptr || getProcAddress == nullptr ||
        getError == nullptr ||
        destroySurface == nullptr ||
        destroyContext == nullptr || terminate == nullptr) {
        std::cerr << "libEGL.so.1 is missing required entry points\n";
        dlclose(library);
        return 77;
    }

    void* x11Library = dlopen("libX11.so.6", RTLD_NOW | RTLD_LOCAL);
    using OpenXDisplay = void* (*)(const char*);
    OpenXDisplay openXDisplay = nullptr;
    if (x11Library != nullptr) {
        openXDisplay = loadSymbol<OpenXDisplay>(x11Library, "XOpenDisplay");
    }
    void* x11Display = openXDisplay != nullptr
                           ? openXDisplay(std::getenv("DISPLAY"))
                           : nullptr;
    EGLDisplay display = x11Display != nullptr
                             ? getPlatformDisplay(kEglPlatformX11, x11Display, nullptr)
                             : getPlatformDisplay(kEglPlatformSurfacelessMesa, nullptr, nullptr);
    if (display == kEglNoDisplay) {
        std::cerr << "EGL surfaceless platform is unavailable\n";
        if (x11Library != nullptr) {
            dlclose(x11Library);
        }
        dlclose(library);
        return 77;
    }
    EGLint major = 0;
    EGLint minor = 0;
    if (initialize(display, &major, &minor) == 0) {
        std::cerr << "eglInitialize failed\n";
        if (x11Library != nullptr) {
            dlclose(x11Library);
        }
        dlclose(library);
        return 77;
    }
    if (bindApi(kEglOpenGlApi) == 0) {
        std::cerr << "EGL OpenGL API is unavailable\n";
        terminate(display);
        if (x11Library != nullptr) {
            dlclose(x11Library);
        }
        dlclose(library);
        return 77;
    }

    const EGLint configAttributes[] = {
        kEglSurfaceType,
        kEglPbufferBit,
        kEglRenderableType,
        kEglOpenGlBit,
        kEglRedSize,
        8,
        kEglGreenSize,
        8,
        kEglBlueSize,
        8,
        kEglAlphaSize,
        8,
        kEglDepthSize,
        24,
        kEglNone,
    };
    EGLConfig config = nullptr;
    EGLint configCount = 0;
    chooseConfig(display, configAttributes, &config, 1, &configCount);
    if (configCount < 1) {
        std::cerr << "no EGL OpenGL PBuffer configuration\n";
        terminate(display);
        if (x11Library != nullptr) {
            dlclose(x11Library);
        }
        dlclose(library);
        return 77;
    }

    const EGLint contextAttributes[] = {
        kEglContextMajorVersion,
        4,
        kEglContextMinorVersion,
        6,
        kEglContextOpenGlProfileMask,
        kEglContextOpenGlCoreProfileBit,
        kEglNone,
    };
    EGLContext context =
        createContext(display, config, kEglNoContext, contextAttributes);
    if (context == kEglNoContext) {
        std::cerr << "unable to create an OpenGL 4.6 EGL context, eglGetError=0x"
                  << std::hex << getError() << std::dec << "\n";
        terminate(display);
        if (x11Library != nullptr) {
            dlclose(x11Library);
        }
        dlclose(library);
        return 77;
    }
    const EGLint surfaceAttributes[] = {
        kEglWidth,
        16,
        kEglHeight,
        16,
        kEglNone,
    };
    EGLSurface surface =
        createPbufferSurface(display, config, surfaceAttributes);
    if (surface == kEglNoSurface) {
        std::cerr << "unable to create an EGL PBuffer surface, eglGetError=0x"
                  << std::hex << getError() << std::dec << "\n";
        destroyContext(display, context);
        terminate(display);
        if (x11Library != nullptr) {
            dlclose(x11Library);
        }
        dlclose(library);
        return 77;
    }
    if (makeCurrent(
            display,
            surface,
            surface,
            context) == 0) {
        std::cerr << "EGL eglMakeCurrent failed, eglGetError=0x"
                  << std::hex << getError() << std::dec << "\n";
        destroySurface(display, surface);
        destroyContext(display, context);
        terminate(display);
        if (x11Library != nullptr) {
            dlclose(x11Library);
        }
        dlclose(library);
        return 77;
    }

    terrain::gl::setProcResolver(
        reinterpret_cast<terrain::gl::ProcResolver>(getProcAddress));

    terrain::Terrain terrain{terrain::TerrainParameters{}};
    terrain::GlRenderer renderer(terrain, 480, 270);
    std::string error;
    if (!renderer.initialize(error)) {
        std::cerr << "GL renderer initialization failed: " << error << "\n";
        destroySurface(display, surface);
        destroyContext(display, context);
        terminate(display);
        if (x11Library != nullptr) {
            dlclose(x11Library);
        }
        dlclose(library);
        return 1;
    }

    terrain::Camera camera;
    camera.position = {256.0f, 190.0f, -90.0f};
    camera.yawDegrees = -8.0f;
    camera.pitchDegrees = -18.0f;
    terrain::LodSelector selector(terrain, 16.0f);
    terrain::SeamSolver seams(terrain);
    terrain::NodeStateCache cache(2048);
    const terrain::SceneFrame frame = terrain::buildSceneFrame(
        terrain,
        selector,
        seams,
        cache,
        camera,
        480,
        270,
        terrain::DebugView::Seam,
        0,
        0.0f);
    if (!renderer.render(frame, terrain::DebugView::Seam, error) ||
        !renderer.capturePpm(output, error)) {
        std::cerr << "GL render failed: " << error << "\n";
        renderer.shutdown();
        destroySurface(display, surface);
        destroyContext(display, context);
        terminate(display);
        if (x11Library != nullptr) {
            dlclose(x11Library);
        }
        dlclose(library);
        return 1;
    }

    std::cout << "EGL OpenGL " << major << "." << minor
              << " render/FBO/PPM smoke test passed\n";
    renderer.shutdown();
    destroySurface(display, surface);
    destroyContext(display, context);
    terminate(display);
    if (x11Library != nullptr) {
        dlclose(x11Library);
    }
    dlclose(library);
    return 0;
}

