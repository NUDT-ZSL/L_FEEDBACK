#include "context.hpp"

#include <sstream>

#include "gl_api.hpp"

#ifdef _WIN32
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <windows.h>
#else
#include <cstdint>
#include <dlfcn.h>
#endif

namespace orv {
namespace {

#ifndef _WIN32
using EGLint = std::int32_t;
using EGLBoolean = std::uint32_t;
using EGLenum = std::uint32_t;
using EGLDisplay = void*;
using EGLConfig = void*;
using EGLSurface = void*;
using EGLContext = void*;

constexpr EGLint EGL_DEFAULT_DISPLAY_VALUE = 0;
constexpr EGLint EGL_NONE_VALUE = 0x3038;
constexpr EGLint EGL_SUCCESS_VALUE = 0x3000;
constexpr EGLint EGL_SURFACE_TYPE_VALUE = 0x3033;
constexpr EGLint EGL_PBUFFER_BIT_VALUE = 0x0001;
constexpr EGLint EGL_RENDERABLE_TYPE_VALUE = 0x3040;
constexpr EGLint EGL_OPENGL_BIT_VALUE = 0x0008;
constexpr EGLint EGL_RED_SIZE_VALUE = 0x3024;
constexpr EGLint EGL_GREEN_SIZE_VALUE = 0x3023;
constexpr EGLint EGL_BLUE_SIZE_VALUE = 0x3022;
constexpr EGLint EGL_DEPTH_SIZE_VALUE = 0x3025;
constexpr EGLint EGL_OPENGL_API_VALUE = 0x30A2;
constexpr EGLint EGL_CONTEXT_MAJOR_VERSION_VALUE = 0x3098;
constexpr EGLint EGL_CONTEXT_MINOR_VERSION_VALUE = 0x30FB;
constexpr EGLint EGL_CONTEXT_OPENGL_PROFILE_MASK_VALUE = 0x30FD;
constexpr EGLint EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_VALUE = 0x0001;
constexpr EGLint EGL_PLATFORM_SURFACELESS_MESA_VALUE = 0x31DD;
constexpr EGLint EGL_WIDTH_VALUE = 0x3057;
constexpr EGLint EGL_HEIGHT_VALUE = 0x3056;

struct EglApi {
  void* library = nullptr;
  EGLDisplay (*getDisplay)(EGLint) = nullptr;
  EGLDisplay (*getPlatformDisplay)(EGLenum, void*, const EGLint*) = nullptr;
  EGLBoolean (*initialize)(EGLDisplay, EGLint*, EGLint*) = nullptr;
  EGLBoolean (*terminate)(EGLDisplay) = nullptr;
  EGLBoolean (*bindApi)(EGLenum) = nullptr;
  EGLBoolean (*chooseConfig)(EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*) = nullptr;
  EGLSurface (*createPbufferSurface)(EGLDisplay, EGLConfig, const EGLint*) = nullptr;
  EGLContext (*createContext)(EGLDisplay, EGLConfig, EGLContext, const EGLint*) = nullptr;
  EGLBoolean (*makeCurrent)(EGLDisplay, EGLSurface, EGLSurface, EGLContext) = nullptr;
  EGLBoolean (*destroyContext)(EGLDisplay, EGLContext) = nullptr;
  EGLBoolean (*destroySurface)(EGLDisplay, EGLSurface) = nullptr;
  EGLBoolean (*releaseThread)() = nullptr;
  EGLint (*getError)() = nullptr;
  void* (*getProcAddress)(const char*) = nullptr;
};

template <typename Function>
bool loadEglSymbol(void* library, const char* name, Function& function) {
  function = reinterpret_cast<Function>(dlsym(library, name));
  return function != nullptr;
}

bool loadEgl(EglApi& api, std::string& error) {
  api.library = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
  if (api.library == nullptr) {
    api.library = dlopen("libEGL.so", RTLD_NOW | RTLD_LOCAL);
  }
  if (api.library == nullptr) {
    error = "unable to load libEGL";
    return false;
  }
  bool complete = true;
  complete &= loadEglSymbol(api.library, "eglGetDisplay", api.getDisplay);
  complete &= loadEglSymbol(api.library, "eglInitialize", api.initialize);
  complete &= loadEglSymbol(api.library, "eglTerminate", api.terminate);
  complete &= loadEglSymbol(api.library, "eglBindAPI", api.bindApi);
  complete &= loadEglSymbol(api.library, "eglChooseConfig", api.chooseConfig);
  complete &= loadEglSymbol(api.library, "eglCreatePbufferSurface",
                            api.createPbufferSurface);
  complete &= loadEglSymbol(api.library, "eglCreateContext", api.createContext);
  complete &= loadEglSymbol(api.library, "eglMakeCurrent", api.makeCurrent);
  complete &= loadEglSymbol(api.library, "eglDestroyContext", api.destroyContext);
  complete &= loadEglSymbol(api.library, "eglDestroySurface", api.destroySurface);
  complete &= loadEglSymbol(api.library, "eglReleaseThread", api.releaseThread);
  complete &= loadEglSymbol(api.library, "eglGetError", api.getError);
  complete &= loadEglSymbol(api.library, "eglGetProcAddress", api.getProcAddress);
  loadEglSymbol(api.library, "eglGetPlatformDisplay", api.getPlatformDisplay);
  loadEglSymbol(api.library, "eglGetPlatformDisplayEXT", api.getPlatformDisplay);
  if (!complete) {
    error = "libEGL is missing required entry points";
    return false;
  }
  return true;
}

std::string eglErrorText(EglApi& api, EGLint code) {
  std::ostringstream stream;
  stream << "EGL error 0x" << std::hex << code;
  if (api.getError != nullptr) {
    stream << " (last 0x" << std::hex << api.getError() << ")";
  }
  return stream.str();
}
#endif

}  // namespace

GlContext::~GlContext() { shutdown(); }

bool GlContext::initialize(AppError& error) {
#ifdef _WIN32
  glfwSetErrorCallback(nullptr);
  if (glfwInit() != GLFW_TRUE) {
    error = renderError("RENDER_ERR_CONTEXT_INIT", 3,
                        "GLFW initialization failed");
    return false;
  }
  glfwInitialized_ = true;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  GLFWwindow* window =
      glfwCreateWindow(64, 64, "offline_render_verify", nullptr, nullptr);
  if (window == nullptr) {
    const char* description = nullptr;
    glfwGetError(&description);
    shutdown();
    error = renderError("RENDER_ERR_CONTEXT_INIT", 3,
                        std::string("hidden GLFW OpenGL 3.3 context failed") +
                            (description != nullptr ? ": " + std::string(description) : ""));
    return false;
  }
  window_ = window;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(0);
  const GlProcLoader loader = [](const char* name) -> void* {
    return reinterpret_cast<void*>(glfwGetProcAddress(name));
  };
  std::string missing;
  if (!loadOpenGlFunctions(loader, missing)) {
    shutdown();
    error = renderError("RENDER_ERR_GL_LOAD", 3,
                        "failed to load required OpenGL functions: " + missing);
    return false;
  }
#else
  auto* api = new EglApi();
  eglApi_ = api;
  std::string loadError;
  if (!loadEgl(*api, loadError)) {
    error = renderError("RENDER_ERR_CONTEXT_INIT", 3, loadError);
    shutdown();
    return false;
  }
  libEgl_ = api->library;

  EGLDisplay display = nullptr;
  if (api->getPlatformDisplay != nullptr) {
    display = api->getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA_VALUE, nullptr, nullptr);
  }
  if (display == nullptr) {
    display = api->getDisplay(EGL_DEFAULT_DISPLAY_VALUE);
  }
  if (display == nullptr) {
    error = renderError("RENDER_ERR_CONTEXT_INIT", 3,
                        "unable to obtain an EGL display");
    shutdown();
    return false;
  }
  display_ = display;

  EGLint major = 0;
  EGLint minor = 0;
  if (!api->initialize(display, &major, &minor)) {
    error = renderError("RENDER_ERR_CONTEXT_INIT", 3,
                        "eglInitialize failed: " + eglErrorText(*api, api->getError()));
    shutdown();
    return false;
  }
  if (!api->bindApi(EGL_OPENGL_API_VALUE)) {
    error = renderError("RENDER_ERR_CONTEXT_INIT", 3,
                        "eglBindAPI(OpenGL) failed: " + eglErrorText(*api, api->getError()));
    shutdown();
    return false;
  }

  const EGLint configAttributes[] = {
      EGL_SURFACE_TYPE_VALUE, EGL_PBUFFER_BIT_VALUE,
      EGL_RENDERABLE_TYPE_VALUE, EGL_OPENGL_BIT_VALUE,
      EGL_RED_SIZE_VALUE, 8,
      EGL_GREEN_SIZE_VALUE, 8,
      EGL_BLUE_SIZE_VALUE, 8,
      EGL_DEPTH_SIZE_VALUE, 24,
      EGL_NONE_VALUE};
  EGLConfig config = nullptr;
  EGLint configCount = 0;
  if (!api->chooseConfig(display, configAttributes, &config, 1, &configCount) ||
      configCount == 0 || config == nullptr) {
    error = renderError("RENDER_ERR_CONTEXT_INIT", 3,
                        "unable to choose an EGL OpenGL config");
    shutdown();
    return false;
  }

  const EGLint surfaceAttributes[] = {
      EGL_WIDTH_VALUE, 1,
      EGL_HEIGHT_VALUE, 1,
      EGL_NONE_VALUE};
  EGLSurface surface = api->createPbufferSurface(display, config, surfaceAttributes);
  if (surface == nullptr) {
    error = renderError("RENDER_ERR_CONTEXT_INIT", 3,
                        "eglCreatePbufferSurface failed: " +
                            eglErrorText(*api, api->getError()));
    shutdown();
    return false;
  }
  surface_ = surface;

  const EGLint contextAttributes[] = {
      EGL_CONTEXT_MAJOR_VERSION_VALUE, 3,
      EGL_CONTEXT_MINOR_VERSION_VALUE, 3,
      EGL_CONTEXT_OPENGL_PROFILE_MASK_VALUE, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_VALUE,
      EGL_NONE_VALUE};
  EGLContext context = api->createContext(display, config, nullptr, contextAttributes);
  if (context == nullptr) {
    error = renderError("RENDER_ERR_CONTEXT_INIT", 3,
                        "eglCreateContext failed: " +
                            eglErrorText(*api, api->getError()));
    shutdown();
    return false;
  }
  context_ = context;
  if (!api->makeCurrent(display, surface, surface, context)) {
    error = renderError("RENDER_ERR_CONTEXT_INIT", 3,
                        "eglMakeCurrent failed: " +
                            eglErrorText(*api, api->getError()));
    shutdown();
    return false;
  }

  libGl_ = dlopen("libGL.so.1", RTLD_NOW | RTLD_LOCAL);
  if (libGl_ == nullptr) {
    libGl_ = dlopen("libGL.so", RTLD_NOW | RTLD_LOCAL);
  }
  const GlProcLoader loader = [api, this](const char* name) -> void* {
    void* function = reinterpret_cast<void*>(api->getProcAddress(name));
    if (function == nullptr && libGl_ != nullptr) {
      function = dlsym(libGl_, name);
    }
    return function;
  };
  std::string missing;
  if (!loadOpenGlFunctions(loader, missing)) {
    error = renderError("RENDER_ERR_GL_LOAD", 3,
                        "failed to load required OpenGL functions: " + missing);
    shutdown();
    return false;
  }
#endif

  const auto* vendor = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
  const auto* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
  const auto* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
  std::ostringstream description;
  description << (vendor != nullptr ? vendor : "unknown") << " | "
              << (renderer != nullptr ? renderer : "unknown") << " | "
              << (version != nullptr ? version : "unknown");
  description_ = description.str();
  return true;
}

void GlContext::shutdown() {
#ifdef _WIN32
  if (window_ != nullptr) {
    glfwDestroyWindow(static_cast<GLFWwindow*>(window_));
    window_ = nullptr;
  }
  if (glfwInitialized_) {
    glfwTerminate();
    glfwInitialized_ = false;
  }
#else
  if (eglApi_ != nullptr) {
    auto* api = static_cast<EglApi*>(eglApi_);
    if (context_ != nullptr && api->destroyContext != nullptr) {
      api->destroyContext(display_, context_);
      context_ = nullptr;
    }
    if (surface_ != nullptr && api->destroySurface != nullptr) {
      api->destroySurface(display_, surface_);
      surface_ = nullptr;
    }
    if (api->releaseThread != nullptr) {
      api->releaseThread();
    }
    if (display_ != nullptr && api->terminate != nullptr) {
      api->terminate(display_);
      display_ = nullptr;
    }
    if (api->library != nullptr) {
      dlclose(api->library);
    }
    delete api;
    eglApi_ = nullptr;
    libEgl_ = nullptr;
  }
  if (libGl_ != nullptr) {
    dlclose(libGl_);
    libGl_ = nullptr;
  }
#endif
}

}  // namespace orv
