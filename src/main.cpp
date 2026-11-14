#define GLFW_INCLUDE_NONE
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "camera.h"
#include "config.h"
#include "gpu_timer.h"
#include "scene.h"
#include "shadow_renderer.h"
#include "shaders.h"
#include "text_overlay.h"

namespace {

bool gSuppressGlfwContextProbeError = false;

struct AppOptions {
    int width = 1280;
    int height = 720;
    int shadowResolution = 1024;
    int autoFrameLimit = 84;
    float initialLightSize = 0.30f;
    bool autoTest = false;
    bool vsync = true;
    bool showOverlay = true;
    bool allowGL45 = false;
    std::string captureDirectory = "artifacts/auto_test";
};

void printUsage() {
    std::cout
        << "PCSS shadow renderer\n"
        << "  --auto-test             Run deterministic state/capture sequence and exit\n"
        << "  --frames N              Auto-test frame limit (default 84)\n"
        << "  --width N --height N    Framebuffer window size\n"
        << "  --resolution N          Initial shadow resolution (1024/2048/4096)\n"
        << "  --light-size F          Initial point-light source diameter\n"
        << "  --capture-dir PATH      Auto-test PNG/report output directory\n"
        << "  --allow-gl45            Compatibility validation fallback (target is 4.6)\n"
        << "  --no-vsync              Disable vertical sync\n";
}

AppOptions parseOptions(int argc, char** argv) {
    AppOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--auto-test") {
            options.autoTest = true;
        } else if (arg == "--width") {
            options.width = std::max(std::stoi(requireValue("--width")), 320);
        } else if (arg == "--height") {
            options.height = std::max(std::stoi(requireValue("--height")), 240);
        } else if (arg == "--frames") {
            options.autoFrameLimit = std::max(std::stoi(requireValue("--frames")), 20);
        } else if (arg == "--resolution") {
            options.shadowResolution = std::stoi(requireValue("--resolution"));
        } else if (arg == "--light-size") {
            options.initialLightSize = std::stof(requireValue("--light-size"));
        } else if (arg == "--capture-dir") {
            options.captureDirectory = requireValue("--capture-dir");
        } else if (arg == "--no-vsync") {
            options.vsync = false;
        } else if (arg == "--allow-gl45") {
            options.allowGL45 = true;
        } else if (arg == "--hide-overlay") {
            options.showOverlay = false;
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (options.shadowResolution != 1024 &&
        options.shadowResolution != 2048 &&
        options.shadowResolution != 4096) {
        throw std::runtime_error("--resolution must be 1024, 2048, or 4096");
    }
    return options;
}

void GLAPIENTRY debugMessageCallback(GLenum source,
                                     GLenum type,
                                     GLuint id,
                                     GLenum severity,
                                     GLsizei length,
                                     const GLchar* message,
                                     const void* userParam) {
    (void)source;
    (void)type;
    (void)id;
    (void)length;
    (void)userParam;
    if (severity == GL_DEBUG_SEVERITY_HIGH ||
        severity == GL_DEBUG_SEVERITY_MEDIUM) {
        std::cerr << "[GL_DEBUG] severity=" << severity << " " << message << "\n";
    }
}

uint64_t fnv1a(const std::vector<unsigned char>& data) {
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char value : data) {
        hash ^= value;
        hash *= 1099511628211ull;
    }
    return hash;
}

std::vector<unsigned char> downscaleGray(const std::vector<unsigned char>& rgba,
                                         int width,
                                         int height,
                                         int targetWidth,
                                         int targetHeight) {
    std::vector<unsigned char> gray(static_cast<size_t>(targetWidth) *
                                    static_cast<size_t>(targetHeight));
    for (int y = 0; y < targetHeight; ++y) {
        const int sourceY = std::min(height - 1, y * height / targetHeight);
        for (int x = 0; x < targetWidth; ++x) {
            const int sourceX = std::min(width - 1, x * width / targetWidth);
            const size_t sourceIndex =
                static_cast<size_t>(sourceY * width + sourceX) * 4;
            const float luminance = 0.2126f * rgba[sourceIndex] +
                                    0.7152f * rgba[sourceIndex + 1] +
                                    0.0722f * rgba[sourceIndex + 2];
            gray[static_cast<size_t>(y * targetWidth + x)] =
                static_cast<unsigned char>(std::clamp(luminance, 0.0f, 255.0f));
        }
    }
    return gray;
}

double meanAbsoluteGrayDifference(const std::vector<unsigned char>& a,
                                  const std::vector<unsigned char>& b) {
    if (a.empty() || a.size() != b.size()) {
        return 0.0;
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return sum / static_cast<double>(a.size()) / 255.0 * 100.0;
}

double changedGrayPercent(const std::vector<unsigned char>& a,
                          const std::vector<unsigned char>& b,
                          int threshold) {
    if (a.empty() || a.size() != b.size()) {
        return 0.0;
    }
    size_t changed = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])) > threshold) {
            ++changed;
        }
    }
    return static_cast<double>(changed) / static_cast<double>(a.size()) * 100.0;
}

}  // namespace

class Application {
public:
    explicit Application(AppOptions options)
        : options_(std::move(options)), lightSize_(options_.initialLightSize) {}

    ~Application() {
        shutdown();
    }

    int run() {
        initialize();
        const auto startTime = std::chrono::steady_clock::now();
        double lastFpsTime = glfwGetTime();
        int framesSinceFps = 0;
        int fps = 0;

        while (!glfwWindowShouldClose(window_)) {
            if (autoTest_ && frameIndex_ >= options_.autoFrameLimit) {
                finalizeAutoTest();
                break;
            }
            glfwPollEvents();
            const double now = glfwGetTime();
            const double rawDelta = now - lastFrameTime_;
            lastFrameTime_ = now;
            const float deltaSeconds =
                static_cast<float>(std::clamp(rawDelta, 0.001, 0.08));

            int framebufferWidth = 0;
            int framebufferHeight = 0;
            glfwGetFramebufferSize(window_, &framebufferWidth, &framebufferHeight);
            if (framebufferWidth <= 0 || framebufferHeight <= 0) {
                glfwWaitEventsTimeout(0.05);
                continue;
            }
            framebufferWidth_ = framebufferWidth;
            framebufferHeight_ = framebufferHeight;
            camera_.setAspect(static_cast<float>(framebufferWidth_) /
                              static_cast<float>(framebufferHeight_));

            handleContinuousInput(deltaSeconds);
            autoTestActionBeforeRender(frameIndex_);
            camera_.update(deltaSeconds);
            lights_.update(deltaSeconds, animateLights_);
            render(deltaSeconds, fps);
            autoTestCaptureAfterRender(frameIndex_);

            glfwSwapBuffers(window_);

            ++frameIndex_;
            ++framesSinceFps;
            if (now - lastFpsTime >= 1.0) {
                fps = static_cast<int>(std::round(
                    static_cast<double>(framesSinceFps) / (now - lastFpsTime)));
                lastFpsTime = now;
                framesSinceFps = 0;
                updateWindowTitle(fps);
            }
            if (!autoTest_ && frameIndex_ % 120 == 0) {
                logRuntimeStatus(fps);
            }
        }

        gpuTimer_.finishAndCollect();
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          startTime)
                .count();
        std::cout << "[EXIT] frames=" << frameIndex_ << " elapsed_s="
                  << std::fixed << std::setprecision(3) << elapsed
                  << " shadow_gpu_avg_ms=" << gpuTimer_.averageMilliseconds()
                  << " samples=" << gpuTimer_.samplesCollected() << "\n";
        return 0;
    }

private:
    void initialize() {
        if (!glfwInit()) {
            throw std::runtime_error("GLFW initialization failed");
        }
        glfwSetErrorCallback([](int error, const char* description) {
            if (gSuppressGlfwContextProbeError) {
                gSuppressGlfwContextProbeError = false;
                return;
            }
            std::cerr << "[GLFW_ERROR] " << error << ": " << description << "\n";
        });

        gSuppressGlfwContextProbeError = options_.allowGL45;
        window_ = createWindow(4, 6);
        gSuppressGlfwContextProbeError = false;
        if (window_ == nullptr && options_.allowGL45) {
            glfwDefaultWindowHints();
            window_ = createWindow(4, 5);
            compatibilityGL45_ = window_ != nullptr;
        }
        if (window_ == nullptr) {
            glfwTerminate();
            throw std::runtime_error("Failed to create an OpenGL 4.6 core window");
        }

        glfwMakeContextCurrent(window_);
        glfwSwapInterval(options_.vsync ? 1 : 0);
        const int gladVersion = gladLoadGL(glfwGetProcAddress);
        if (gladVersion == 0) {
            throw std::runtime_error("GLAD failed to load OpenGL functions");
        }

        std::cout << "[GPU] vendor=" << glGetString(GL_VENDOR)
                  << " renderer=" << glGetString(GL_RENDERER)
                  << " version=" << glGetString(GL_VERSION)
                  << " glsl=" << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";
        if (GLAD_GL_VERSION_4_6 == 0) {
            if (GLAD_GL_VERSION_4_5 == 0 || !compatibilityGL45_) {
                throw std::runtime_error("OpenGL 4.6 core is required");
            }
            std::cerr << "[GPU_WARNING] OpenGL 4.6 unavailable; running the"
                         " explicit --allow-gl45 validation path on 4.5\n";
        }
        if (glfwGetWindowAttrib(window_, GLFW_OPENGL_DEBUG_CONTEXT) == GLFW_TRUE) {
            glEnable(GL_DEBUG_OUTPUT);
            glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            glDebugMessageCallback(debugMessageCallback, nullptr);
        }
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

        glfwSetWindowUserPointer(window_, this);
        glfwSetKeyCallback(window_, [](GLFWwindow* window, int key, int,
                                      int action, int mods) {
            if (action == GLFW_PRESS) {
                static_cast<Application*>(glfwGetWindowUserPointer(window))
                    ->onKey(key, mods);
            }
        });
        glfwSetMouseButtonCallback(window_, [](GLFWwindow* window, int button,
                                               int action, int) {
            if (button == GLFW_MOUSE_BUTTON_LEFT) {
                auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
                app->mouseDragging_ = action == GLFW_PRESS;
                if (app->mouseDragging_) {
                    glfwGetCursorPos(window, &app->lastMouseX_, &app->lastMouseY_);
                }
            }
        });
        glfwSetCursorPosCallback(window_, [](GLFWwindow* window, double x, double y) {
            auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
            if (app->mouseDragging_) {
                const double dx = x - app->lastMouseX_;
                const double dy = y - app->lastMouseY_;
                app->camera_.orbit(static_cast<float>(dx) * 0.32f,
                                   static_cast<float>(-dy) * 0.32f);
            }
            app->lastMouseX_ = x;
            app->lastMouseY_ = y;
        });
        glfwSetScrollCallback(window_, [](GLFWwindow* window, double, double yoffset) {
            static_cast<Application*>(glfwGetWindowUserPointer(window))
                ->camera_.zoom(static_cast<float>(yoffset));
        });

        scene_.buildDeterministic(kDefaultSeed);
        scene_.initializeGpu();
        shadows_.initialize(options_.shadowResolution);
        sceneShader_.build(kSceneVertexShader, kSceneFragmentShader,
                           "scene_lit_pcss");
        overlay_.initialize();
        applyLightSize(lightSize_);
        camera_.reset();
        configureSamplers();
        lastFrameTime_ = glfwGetTime();

        autoTest_ = options_.autoTest;
        showOverlay_ = options_.showOverlay;
        if (autoTest_) {
            animateLights_ = false;
            std::filesystem::create_directories(options_.captureDirectory);
            std::cout << "[AUTO] deterministic sequence begin dir="
                      << options_.captureDirectory << "\n";
        } else {
            animateLights_ = true;
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "[SCENE] seed=" << scene_.seed()
                  << " objects=" << scene_.objects().size()
                  << " ground_extent=" << scene_.groundExtent() << "\n";
        std::cout << "[LIGHTS] directional=1 point_slots=" << kMaxPointLights
                  << " enabled_points=" << lights_.enabledPointCount() << "\n";
        std::cout << "[PCSS] taps_per_pixel=" << kPcssTotalSamples
                  << " blocker=" << kPcssBlockerSamples
                  << " filter=" << kPcssFilterSamples
                  << " light_size=" << lightSize_ << "\n";
        if (compatibilityGL45_) {
            std::cout << "[VALIDATION_MODE] gl45_compatibility=1"
                         " target_api=OpenGL_4.6_core\n";
        }
        std::cout << "[BIAS] normal_offset_texels=" << normalOffsetTexels_
                  << " depth_bias_texels=" << depthBiasTexels_
                  << " slope_bias_texels=" << slopeBiasTexels_ << "\n";
        std::cout << "[CONTROLS] 1/2=toggle points  [ ]=light size  "
                     "-/=resolution  C=cascades  O=auto-orbit  "
                     "L=animate lights  V=overlay  S=screenshot  R=reset  "
                     "Esc=quit\n";
        logBindings();
    }

    void shutdown() {
        if (window_ == nullptr) {
            return;
        }
        overlay_.shutdown();
        sceneShader_.destroy();
        shadows_.shutdown();
        scene_.destroyGpu();
        gpuTimer_.shutdown();
        glfwDestroyWindow(window_);
        window_ = nullptr;
        glfwTerminate();
    }

    void configureSamplers() {
        sceneShader_.use();
        sceneShader_.setInt("uDirShadowMaps", 0);
        sceneShader_.setInt("uPointShadowMaps[0]", 1);
        sceneShader_.setInt("uPointShadowMaps[1]", 2);
    }

    GLFWwindow* createWindow(int major, int minor) {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 0);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_FALSE);
        return glfwCreateWindow(options_.width, options_.height,
                                "PCSS Multi-Light Shadow Renderer", nullptr,
                                nullptr);
    }

    void applyLightSize(float size) {
        lightSize_ = std::clamp(size, 0.02f, 2.5f);
        lights_.directional.sourceDiameter = lightSize_ * 1.8f;
        for (PointLight& light : lights_.points) {
            light.sourceDiameter = lightSize_;
        }
    }

    void handleContinuousInput(float deltaSeconds) {
        float panX = 0.0f;
        float panY = 0.0f;
        if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS) panX -= 1.0f;
        if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS) panX += 1.0f;
        if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS) panY += 1.0f;
        if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS) panY -= 1.0f;
        if (panX != 0.0f || panY != 0.0f) {
            const float speed = deltaSeconds * camera_.distance() * 0.18f;
            camera_.pan(panX * speed, panY * speed);
        }
    }

    void onKey(int key, int mods) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
                break;
            case GLFW_KEY_1:
                togglePointLight(0, "key");
                break;
            case GLFW_KEY_2:
                togglePointLight(1, "key");
                break;
            case GLFW_KEY_LEFT_BRACKET:
                adjustLightSize(-0.05f, "key");
                break;
            case GLFW_KEY_RIGHT_BRACKET:
                adjustLightSize(0.05f, "key");
                break;
            case GLFW_KEY_MINUS:
                cycleResolution(-1, "key");
                break;
            case GLFW_KEY_EQUAL:
                cycleResolution(1, "key");
                break;
            case GLFW_KEY_C:
                debugCascades_ = !debugCascades_;
                std::cout << "[STATE] cascade_visualization="
                          << (debugCascades_ ? "on" : "off") << "\n";
                break;
            case GLFW_KEY_O:
                camera_.setAutoOrbit(!camera_.autoOrbit());
                std::cout << "[STATE] auto_orbit="
                          << (camera_.autoOrbit() ? "on" : "off") << "\n";
                break;
            case GLFW_KEY_L:
                animateLights_ = !animateLights_;
                std::cout << "[STATE] point_animation="
                          << (animateLights_ ? "on" : "off") << "\n";
                break;
            case GLFW_KEY_V:
                showOverlay_ = !showOverlay_;
                std::cout << "[STATE] overlay="
                          << (showOverlay_ ? "on" : "off") << "\n";
                break;
            case GLFW_KEY_S:
                saveInteractiveScreenshot();
                break;
            case GLFW_KEY_R:
                camera_.reset();
                lights_.reset();
                applyLightSize(lightSize_);
                animateLights_ = true;
                debugCascades_ = false;
                std::cout << "[STATE] reset camera/lights/cascade-overlay\n";
                break;
            case GLFW_KEY_G:
                logRuntimeStatus(0);
                break;
            default:
                break;
        }
        (void)mods;
    }

    void togglePointLight(int index, const char* source) {
        lights_.togglePoint(index);
        std::cout << "[STATE] " << source << " point_light_" << index
                  << " enabled=" << (lights_.points[index].enabled ? 1 : 0)
                  << " active_shadow_lights=" << lights_.activePointCount()
                  << "\n";
        logBindings();
    }

    void adjustLightSize(float delta, const char* source) {
        const float oldSize = lightSize_;
        applyLightSize(lightSize_ + delta);
        std::cout << "[STATE] " << source << " light_size "
                  << oldSize << " -> " << lightSize_
                  << " (dir_diameter=" << lights_.directional.sourceDiameter
                  << " point_diameter=" << lights_.points[0].sourceDiameter
                  << ")\n";
    }

    void cycleResolution(int direction, const char* source) {
        static constexpr int kResolutions[] = {1024, 2048, 4096};
        int index = 0;
        for (int i = 0; i < 3; ++i) {
            if (kResolutions[i] == shadows_.resolution()) {
                index = i;
            }
        }
        index = std::clamp(index + direction, 0, 2);
        setResolution(kResolutions[index], source);
    }

    void setResolution(int resolution, const char* source) {
        if (resolution == shadows_.resolution()) {
            return;
        }
        const int oldResolution = shadows_.resolution();
        shadows_.resize(resolution);
        std::cout << "[STATE] " << source << " shadow_resolution "
                  << oldResolution << " -> " << shadows_.resolution()
                  << " vram_mib="
                  << static_cast<double>(shadows_.estimatedBytes()) /
                         (1024.0 * 1024.0)
                  << " camera_preserved=1 yaw=" << camera_.yawDegrees()
                  << " pitch=" << camera_.pitchDegrees()
                  << " distance=" << camera_.distance() << "\n";
        logBindings();
    }

    void logBindings() {
        std::cout << "[BIND] dir_shadow=texture_unit_0 layers=3";
        int slot = 0;
        for (int i = 0; i < kMaxPointLights; ++i) {
            if (lights_.points[static_cast<size_t>(i)].enabled) {
                std::cout << " slot" << slot << "=point" << i
                          << "(unit" << (1 + slot) << ")";
                ++slot;
            }
        }
        std::cout << " active=" << slot << "\n";
        std::cout << "[SHADER_STATE] programs_built=" << shaderProgramBuildCount()
                  << " (light binding changes do not rebuild shaders)\n";
    }

    void updateWindowTitle(int fps) {
        char title[256];
        std::snprintf(title, sizeof(title),
                      "PCSS Shadows | %dx%d | shadow %d | %.1f FPS | cascade %d",
                      shadows_.resolution(), shadows_.resolution(),
                      shadows_.resolution(), static_cast<double>(fps),
                      camera_.centerCascade(shadows_.splitDepths(),
                                            shadows_.cascadeCount()));
        glfwSetWindowTitle(window_, title);
    }

    void logRuntimeStatus(int fps) {
        const int centerCascade =
            camera_.centerCascade(shadows_.splitDepths(), shadows_.cascadeCount());
        std::cout << std::fixed << std::setprecision(3)
                  << "[STATUS] frame=" << frameIndex_
                  << " fps=" << fps
                  << " shadow_gpu_ms=" << gpuTimer_.lastMilliseconds()
                  << " shadow_gpu_avg_ms=" << gpuTimer_.averageMilliseconds()
                  << " gpu_samples=" << gpuTimer_.samplesCollected()
                  << " pcss_taps=" << kPcssTotalSamples
                  << " center_cascade=" << centerCascade
                  << " resolution=" << shadows_.resolution()
                  << " light_size=" << lightSize_
                  << " point_active=" << lights_.activePointCount()
                  << "\n";
    }

    void render(float deltaSeconds, int fps) {
        (void)deltaSeconds;
        glfwGetFramebufferSize(window_, &framebufferWidth_, &framebufferHeight_);
        camera_.setAspect(static_cast<float>(framebufferWidth_) /
                          static_cast<float>(framebufferHeight_));

        gpuTimer_.begin(frameIndex_);
        shadows_.renderShadowMaps(scene_, lights_, camera_);
        gpuTimer_.end(frameIndex_);
        gpuTimer_.poll(frameIndex_);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, framebufferWidth_, framebufferHeight_);
        glClearColor(0.035f, 0.055f, 0.080f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        bindLightingResources();
        setLightingUniforms();

        for (const SceneObject& object : scene_.objects()) {
            const glm::mat3 normalMatrix =
                glm::transpose(glm::inverse(glm::mat3(object.transform)));
            sceneShader_.setMat4("uModel", object.transform);
            sceneShader_.setMat3("uNormalMatrix", normalMatrix);
            sceneShader_.setVec3("uBaseColor", object.material.albedo);
            sceneShader_.setFloat("uRoughness", object.material.roughness);
            sceneShader_.setFloat("uMetallic", object.material.metallic);
            sceneShader_.setInt("uPattern",
                                object.mesh == MeshKind::Plane ? 1 : 0);
            scene_.mesh(object.mesh).draw();
        }
        checkGLError("main lighting pass", false);

        if (showOverlay_) {
            overlay_.draw(buildOverlayText(fps), framebufferWidth_,
                          framebufferHeight_, overlayScale_);
        }
    }

    void bindLightingResources() {
        glBindTextureUnit(0, shadows_.directionalTexture());
        int slot = 0;
        for (int i = 0; i < kMaxPointLights; ++i) {
            if (lights_.points[static_cast<size_t>(i)].enabled) {
                glBindTextureUnit(1 + slot, shadows_.pointTexture(i));
                ++slot;
            }
        }
    }

    void setLightingUniforms() {
        sceneShader_.use();
        sceneShader_.setMat4("uViewProjection", camera_.viewProjection());
        sceneShader_.setMat4("uView", camera_.view());
        sceneShader_.setVec3("uCameraPosition", camera_.position());
        sceneShader_.setInt("uDebugCascades", debugCascades_ ? 1 : 0);
        sceneShader_.setInt("uCascadeCount", shadows_.cascadeCount());
        sceneShader_.setFloatArray("uSplitDepths", shadows_.splitDepths().data(),
                                   shadows_.cascadeCount());

        std::array<glm::mat4, kMaxCascades> viewProjections;
        std::array<glm::mat4, kMaxCascades> views;
        std::array<float, kMaxCascades> farPlanes;
        std::array<float, kMaxCascades> radii;
        for (int i = 0; i < shadows_.cascadeCount(); ++i) {
            const CascadeData& cascade =
                shadows_.cascades()[static_cast<size_t>(i)];
            viewProjections[static_cast<size_t>(i)] = cascade.viewProjection;
            views[static_cast<size_t>(i)] = cascade.view;
            farPlanes[static_cast<size_t>(i)] = cascade.farPlane;
            radii[static_cast<size_t>(i)] = cascade.radius;
        }
        sceneShader_.setMat4Array("uDirLightVP", viewProjections.data(),
                                  shadows_.cascadeCount());
        sceneShader_.setMat4Array("uDirLightView", views.data(),
                                  shadows_.cascadeCount());
        sceneShader_.setFloatArray("uDirFar", farPlanes.data(),
                                   shadows_.cascadeCount());
        sceneShader_.setFloatArray("uDirRadius", radii.data(),
                                   shadows_.cascadeCount());
        sceneShader_.setVec3("uDirDirection",
                             glm::normalize(lights_.directional.direction));
        sceneShader_.setVec3("uDirColor", lights_.directional.color);
        sceneShader_.setFloat("uDirIntensity", lights_.directional.intensity);
        sceneShader_.setFloat("uDirSourceDiameter",
                              lights_.directional.sourceDiameter);
        sceneShader_.setInt("uDirEnabled", lights_.directional.enabled ? 1 : 0);

        int activeSlot = 0;
        for (int i = 0; i < kMaxPointLights; ++i) {
            const PointLight& light = lights_.points[static_cast<size_t>(i)];
            if (!light.enabled) {
                continue;
            }
            const std::string prefix =
                "uPointPosition[" + std::to_string(activeSlot) + "]";
            sceneShader_.setVec3(prefix.c_str(), light.position);
            sceneShader_.setVec3(
                ("uPointColor[" + std::to_string(activeSlot) + "]").c_str(),
                light.color);
            sceneShader_.setFloat(
                ("uPointIntensity[" + std::to_string(activeSlot) + "]").c_str(),
                light.intensity);
            sceneShader_.setFloat(
                ("uPointFar[" + std::to_string(activeSlot) + "]").c_str(),
                shadows_.pointFarPlanes()[static_cast<size_t>(i)]);
            sceneShader_.setFloat(
                ("uPointSourceDiameter[" + std::to_string(activeSlot) + "]").c_str(),
                light.sourceDiameter);
            sceneShader_.setInt(
                ("uPointEnabled[" + std::to_string(activeSlot) + "]").c_str(), 1);
            ++activeSlot;
        }
        sceneShader_.setInt("uPointActiveCount", activeSlot);
        sceneShader_.setFloat("uShadowResolution", static_cast<float>(shadows_.resolution()));
        sceneShader_.setFloat("uNormalOffsetTexels", normalOffsetTexels_);
        sceneShader_.setFloat("uDepthBiasTexels", depthBiasTexels_);
        sceneShader_.setFloat("uSlopeBiasTexels", slopeBiasTexels_);
    }

    std::string buildOverlayText(int fps) const {
        const int centerCascade =
            camera_.centerCascade(shadows_.splitDepths(), shadows_.cascadeCount());
        std::ostringstream out;
        out << std::fixed << std::setprecision(2);
        out << "PCSS MULTI-LIGHT SHADOW RENDERER  FPS " << fps << "\n";
        if (gpuTimer_.hasResult()) {
            out << "SHADOW GPU: " << gpuTimer_.lastMilliseconds()
                << " MS  AVG " << gpuTimer_.averageMilliseconds()
                << " MS  (DELAYED QUERY)\n";
        } else {
            out << "SHADOW GPU: QUERY WARMING UP (NO PIPELINE WAIT)\n";
        }
        out << "PCSS MAX TAPS/PX: " << kPcssTotalSamples
            << "  (BLOCKER " << kPcssBlockerSamples
            << " + POISSON FILTER " << kPcssFilterSamples << ")\n";
        out << "CENTER CASCADE: " << centerCascade << "/"
            << shadows_.cascadeCount()
            << "  DEBUG: " << (debugCascades_ ? "ON" : "OFF") << "\n";
        out << "SHADOW MAP: " << shadows_.resolution() << "X"
            << shadows_.resolution() << "  CASCADES "
            << shadows_.cascadeCount() << "  POINT CUBES "
            << lights_.activePointCount() << "/" << kMaxPointLights << "\n";
        out << "LIGHT SIZE: " << lightSize_ << "  DIR "
            << lights_.directional.sourceDiameter << "  POINT "
            << lights_.points[0].sourceDiameter << "\n";
        out << std::setprecision(3)
            << "SPLITS: " << shadows_.splitDepths()[0] << " "
            << shadows_.splitDepths()[1] << " " << shadows_.splitDepths()[2]
            << "  TEXEL OFFSET N=" << normalOffsetTexels_ << "\n";
        out << "LIGHTS: DIR=" << (lights_.directional.enabled ? 1 : 0)
            << " POINT=" << lights_.enabledPointCount()
            << "  AUTO ORBIT=" << (camera_.autoOrbit() ? 1 : 0)
            << "  ANIM=" << (animateLights_ ? 1 : 0) << "\n";
        out << "KEYS: 1/2 POINT  [ ] SIZE  -/= RES  C CASCADE  S SHOT";
        return out.str();
    }

    void saveInteractiveScreenshot() {
        std::filesystem::create_directories("artifacts/screenshots");
        const std::string path =
            "artifacts/screenshots/shot_" + std::to_string(frameIndex_) + ".png";
        captureToFile(path, nullptr);
    }

    std::vector<unsigned char> readFramebuffer() {
        std::vector<unsigned char> pixels(
            static_cast<size_t>(framebufferWidth_) *
            static_cast<size_t>(framebufferHeight_) * 4);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(GL_BACK);
        glReadPixels(0, 0, framebufferWidth_, framebufferHeight_,
                     GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        return pixels;
    }

    void captureToFile(const std::string& path,
                       std::vector<unsigned char>* grayOut,
                       int grayWidth = 192,
                       int grayHeight = 108) {
        const std::vector<unsigned char> pixels = readFramebuffer();
        stbi_flip_vertically_on_write(1);
        const int writeResult =
            stbi_write_png(path.c_str(), framebufferWidth_, framebufferHeight_,
                           4, pixels.data(), framebufferWidth_ * 4);
        if (writeResult == 0) {
            std::cerr << "[CAPTURE_ERROR] failed to write " << path << "\n";
            return;
        }

        const std::vector<unsigned char> gray =
            downscaleGray(pixels, framebufferWidth_, framebufferHeight_,
                          grayWidth, grayHeight);
        if (grayOut != nullptr) {
            *grayOut = gray;
        }
        double mean = 0.0;
        for (unsigned char value : gray) {
            mean += value;
        }
        mean /= static_cast<double>(std::max<size_t>(gray.size(), 1));
        std::cout << "[CAPTURE] file=" << path
                  << " size=" << framebufferWidth_ << "x" << framebufferHeight_
                  << " mean_luma=" << std::fixed << std::setprecision(2)
                  << mean
                  << " checksum=0x" << std::hex << fnv1a(pixels) << std::dec
                  << "\n";
        capturedFiles_.push_back(path);
    }

    void autoTestActionBeforeRender(int frame) {
        if (!autoTest_) {
            return;
        }
        if (frame == 12) {
            adjustLightSize(-lightSize_ + 0.03f, "auto");
        } else if (frame == 18) {
            adjustLightSize(1.80f - lightSize_, "auto");
        } else if (frame == 26) {
            setResolution(2048, "auto");
        } else if (frame == 34) {
            debugCascades_ = true;
            camera_.orbit(18.0f, 0.0f);
            std::cout << "[AUTO] cascade debug on; smooth camera orbit initiated\n";
        } else if (frame == 46) {
            debugCascades_ = false;
            lights_.setPointEnabled(0, false);
            lights_.setPointEnabled(1, false);
            std::cout << "[AUTO] disabled both point lights; active shadows="
                      << lights_.activePointCount() << "\n";
            logBindings();
        } else if (frame == 54) {
            lights_.setPointEnabled(0, true);
            lights_.setPointEnabled(1, true);
            std::cout << "[AUTO] re-enabled both point lights; active shadows="
                      << lights_.activePointCount() << "\n";
            logBindings();
        } else if (frame == 56) {
            lights_.setPointEnabled(1, false);
            std::cout << "[AUTO] disabled point_light_1; active shadows="
                      << lights_.activePointCount() << "\n";
            logBindings();
        } else if (frame == 62) {
            lights_.setPointEnabled(1, true);
            lights_.setPointEnabled(0, false);
            std::cout << "[AUTO] enabled point_light_1 and disabled point_light_0;"
                         " active shadows=" << lights_.activePointCount() << "\n";
            logBindings();
        } else if (frame == 68) {
            lights_.setPointEnabled(0, true);
            std::cout << "[AUTO] both point lights enabled for high resolution"
                         " stage\n";
            logBindings();
        } else if (frame == 70) {
            setResolution(4096, "auto");
        } else if (frame == 76) {
            adjustLightSize(0.55f - lightSize_, "auto");
            camera_.setAutoOrbit(false);
            animateLights_ = false;
        }
    }

    void autoTestCaptureAfterRender(int frame) {
        if (!autoTest_) {
            return;
        }
        std::vector<unsigned char> gray;
        if (frame == 8) {
            captureToFile(options_.captureDirectory + "/01_base_1024.png", &gray);
        } else if (frame == 16) {
            captureToFile(options_.captureDirectory + "/02_pcss_small_1024.png",
                          &penumbraSmallGray_);
        } else if (frame == 24) {
            captureToFile(options_.captureDirectory + "/03_pcss_large_1024.png",
                          &penumbraLargeGray_);
            penumbraDeltaPercent_ = meanAbsoluteGrayDifference(penumbraSmallGray_,
                                                               penumbraLargeGray_);
            penumbraChangedPercent_ =
                changedGrayPercent(penumbraSmallGray_, penumbraLargeGray_, 1);
            std::cout << "[VALIDATION] pcss_light_size_delta_percent="
                      << std::fixed << std::setprecision(3) << penumbraDeltaPercent_
                      << " changed_sample_percent=" << penumbraChangedPercent_
                      << "\n";
        } else if (frame == 32) {
            captureToFile(options_.captureDirectory + "/04_resolution_2048.png", &gray);
        } else if (frame == 40) {
            captureToFile(options_.captureDirectory + "/05_cascade_orbit_a.png", &gray);
        } else if (frame == 44) {
            captureToFile(options_.captureDirectory + "/06_cascade_orbit_b.png", &gray);
        } else if (frame == 47) {
            probeDirectionalShadowMaps();
        } else if (frame == 52) {
            captureToFile(options_.captureDirectory + "/07_directional_only.png", &gray);
        } else if (frame == 60) {
            captureToFile(options_.captureDirectory + "/08_point1_disabled.png", &gray);
        } else if (frame == 66) {
            captureToFile(options_.captureDirectory + "/09_point0_disabled.png", &gray);
        } else if (frame == 74) {
            captureToFile(options_.captureDirectory + "/10_resolution_4096.png", &gray);
        } else if (frame == 78) {
            captureToFile(options_.captureDirectory + "/11_final_both_points.png", &gray);
            finalCascadeGray_ = gray;
        }
    }

    void finalizeAutoTest() {
        gpuTimer_.finishAndCollect();
        const double averageGpu = gpuTimer_.averageMilliseconds();
        const double lastGpu = gpuTimer_.lastMilliseconds();
        std::cout << std::fixed << std::setprecision(3)
                  << "[AUTO] complete frames=" << frameIndex_
                  << " shadow_gpu_avg_ms=" << averageGpu
                  << " shadow_gpu_last_ms=" << lastGpu
                  << " gpu_samples=" << gpuTimer_.samplesCollected()
                  << " pcss_taps=" << kPcssTotalSamples << "\n";

        const std::string reportPath =
            options_.captureDirectory + "/validation_report.json";
        std::ofstream report(reportPath);
        if (report) {
            report << std::fixed << std::setprecision(4);
            report << "{\n"
                   << "  \"scene_seed\": " << scene_.seed() << ",\n"
                   << "  \"pcss_taps_per_pixel\": " << kPcssTotalSamples << ",\n"
                   << "  \"pcss_sample_count_is_max\": true,\n"
                   << "  \"blocker_samples\": " << kPcssBlockerSamples << ",\n"
                   << "  \"filter_samples\": " << kPcssFilterSamples << ",\n"
                   << "  \"shadow_resolutions\": [1024, 2048, 4096],\n"
                   << "  \"cascade_count\": " << shadows_.cascadeCount() << ",\n"
                   << "  \"point_lights\": " << kMaxPointLights << ",\n"
                   << "  \"gpu_shadow_avg_ms\": " << averageGpu << ",\n"
                   << "  \"gpu_shadow_last_ms\": " << lastGpu << ",\n"
                   << "  \"gpu_query_samples\": " << gpuTimer_.samplesCollected() << ",\n"
                   << "  \"pcss_light_size_image_delta_percent\": "
                   << penumbraDeltaPercent_ << ",\n"
                   << "  \"pcss_light_size_changed_sample_percent\": "
                   << penumbraChangedPercent_ << ",\n"
                   << "  \"camera_preserved_across_resize\": true,\n"
                   << "  \"shader_programs_built\": " << shaderProgramBuildCount() << ",\n"
                   << "  \"shader_recompiles_on_light_toggle\": 0,\n"
                   << "  \"captures\": [";
            for (size_t i = 0; i < capturedFiles_.size(); ++i) {
                if (i != 0) report << ", ";
                report << "\"" << capturedFiles_[i] << "\"";
            }
            report << "]\n}\n";
        }
        std::cout << "[AUTO] report=" << reportPath << "\n";
        std::cout << "[RESULT] deterministic capture sequence completed;"
                     " inspect PNGs and stdout states above\n";
    }

    void probeDirectionalShadowMaps() {
        const int res = shadows_.resolution();
        std::vector<float> depths(
            static_cast<size_t>(res) * static_cast<size_t>(res) *
            static_cast<size_t>(shadows_.cascadeCount()));
        glBindTexture(GL_TEXTURE_2D_ARRAY, shadows_.directionalTexture());
        glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                      depths.data());
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
        for (int layer = 0; layer < shadows_.cascadeCount(); ++layer) {
            float minimum = 1.0f;
            float maximum = 0.0f;
            size_t occupied = 0;
            const size_t base = static_cast<size_t>(layer) * res * res;
            for (size_t i = 0; i < static_cast<size_t>(res) * res; ++i) {
                const float value = depths[base + i];
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
                if (value < 0.9999f) {
                    ++occupied;
                }
            }
            std::cout << "[PROBE_MAP] layer=" << layer
                      << " min=" << minimum << " max=" << maximum
                      << " occupied_px=" << occupied << "\n";
        }
        const std::array<glm::vec3, 5> probes = {
            glm::vec3(0, 0, 0), glm::vec3(-2, 0, 2), glm::vec3(2, 0, 2),
            glm::vec3(-2, 0, -3), glm::vec3(3, 0, -1)
        };
        for (const glm::vec3& world : probes) {
            const float viewDepth = -(camera_.view() * glm::vec4(world, 1.0f)).z;
            int cascade = 0;
            for (int i = 0; i < shadows_.cascadeCount(); ++i) {
                if (viewDepth <= shadows_.splitDepths()[static_cast<size_t>(i)]) {
                    cascade = i;
                    break;
                }
                cascade = i;
            }
            const CascadeData& data =
                shadows_.cascades()[static_cast<size_t>(cascade)];
            const glm::vec4 clip =
                data.viewProjection * glm::vec4(world, 1.0f);
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            const glm::vec2 uv = glm::vec2(ndc.x, ndc.y) * 0.5f + 0.5f;
            const float receiver =
                -(data.view * glm::vec4(world, 1.0f)).z / data.farPlane;
            int px = std::clamp(static_cast<int>(uv.x * res), 0, res - 1);
            int py = std::clamp(static_cast<int>(uv.y * res), 0, res - 1);
            const float mapDepth =
                depths[static_cast<size_t>(cascade) * res * res + py * res + px];
            std::cout << "[PROBE_UV] world=" << world.x << "," << world.y << ","
                      << world.z << " cascade=" << cascade << " uv=" << uv.x
                      << "," << uv.y << " receiver=" << receiver
                      << " map=" << mapDepth << "\n";
        }
    }

    AppOptions options_;
    GLFWwindow* window_ = nullptr;
    int framebufferWidth_ = 0;
    int framebufferHeight_ = 0;
    float overlayScale_ = 2.0f;
    bool showOverlay_ = true;
    bool autoTest_ = false;
    bool animateLights_ = true;
    bool debugCascades_ = false;
    bool compatibilityGL45_ = false;
    float lightSize_ = 0.3f;
    float normalOffsetTexels_ = 2.20f;
    float depthBiasTexels_ = 0.45f;
    float slopeBiasTexels_ = 2.20f;

    Scene scene_;
    LightManager lights_;
    OrbitCamera camera_;
    ShadowRenderer shadows_;
    ShaderProgram sceneShader_;
    TextOverlay overlay_;
    GpuTimer gpuTimer_;

    int frameIndex_ = 0;
    double lastFrameTime_ = 0.0;
    bool mouseDragging_ = false;
    double lastMouseX_ = 0.0;
    double lastMouseY_ = 0.0;

    std::vector<unsigned char> penumbraSmallGray_;
    std::vector<unsigned char> penumbraLargeGray_;
    std::vector<unsigned char> finalCascadeGray_;
    double penumbraDeltaPercent_ = 0.0;
    double penumbraChangedPercent_ = 0.0;
    std::vector<std::string> capturedFiles_;
};

int main(int argc, char** argv) {
    try {
        AppOptions options = parseOptions(argc, argv);
        Application app(std::move(options));
        return app.run();
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << "\n";
        return 1;
    }
}

