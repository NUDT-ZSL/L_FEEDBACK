#include "Application.hpp"

#define GLFW_INCLUDE_NONE
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace indoor {
namespace {

int chooseShadowSize(int width, int height) {
    const int smallest = std::max(1, std::min(width, height));
    const int desired = std::clamp(smallest / 8, 32, 192);
    return std::max(32, (desired / 8) * 8);
}

void glfwErrorCallback(int code, const char* description) {
    std::cerr << "GLFW error " << code << ": "
              << (description != nullptr ? description : "") << '\n';
}

} // namespace

Application::Application(LaunchOptions options) : options_(options) {}

Application::~Application() {
    if (glfwInitialized_ && renderer_.hasGpuResources() &&
        glfwGetCurrentContext() != nullptr) {
        renderer_.shutdown();
    }
    destroyWindow();
    if (glfwInitialized_) {
        glfwTerminate();
    }
}

bool Application::createWindow() {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 0);
    window_ = glfwCreateWindow(std::max(options_.width, 320),
                               std::max(options_.height, 200),
                               "Indoor Deferred Renderer", nullptr, nullptr);
    if (window_ == nullptr) {
        return false;
    }
    glfwMakeContextCurrent(window_);
    if (gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress)) == 0) {
        return false;
    }
    glfwSwapInterval(1);
    installCallbacks();
    return true;
}

void Application::destroyWindow() {
    if (window_ != nullptr) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
        glfwMakeContextCurrent(nullptr);
    }
}

void Application::installCallbacks() {
    glfwSetWindowUserPointer(window_, this);
    glfwSetKeyCallback(window_, &Application::keyCallback);
    glfwSetFramebufferSizeCallback(window_, &Application::framebufferSizeCallback);
    glfwSetMouseButtonCallback(window_, &Application::mouseButtonCallback);
    glfwSetCursorPosCallback(window_, &Application::cursorPositionCallback);
    glfwSetScrollCallback(window_, &Application::scrollCallback);
    glfwSetWindowRefreshCallback(window_, &Application::windowRefreshCallback);
}

int Application::run() {
    glfwSetErrorCallback(glfwErrorCallback);
    if (glfwInit() != GLFW_TRUE) {
        throw std::runtime_error("Unable to initialize GLFW");
    }
    glfwInitialized_ = true;
    if (!createWindow()) {
        throw std::runtime_error("Unable to create an OpenGL 4.5 core window");
    }

    std::cout << "OpenGL vendor:   " << glGetString(GL_VENDOR) << '\n'
              << "OpenGL renderer: " << glGetString(GL_RENDERER) << '\n'
              << "OpenGL version:  " << glGetString(GL_VERSION) << '\n';

    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window_, &framebufferWidth, &framebufferHeight);
    settings_.width = std::max(framebufferWidth, 1);
    settings_.height = std::max(framebufferHeight, 1);
    settings_.shadowSize = options_.shadowSize > 0
        ? std::max(options_.shadowSize, 32)
        : chooseShadowSize(settings_.width, settings_.height);
    settings_.shadowCacheCapacity = options_.shadowCacheCapacity > 0
        ? options_.shadowCacheCapacity
        : options_.lightCount;
    if (settings_.shadowCacheCapacity == 0) {
        settings_.shadowCacheCapacity = options_.lightCount;
    }

    scene_ = buildDeterministicScene(options_.seed, options_.lightCount);
    renderer_.initialize(scene_, settings_, INDOOR_SHADER_DIR);
    printStatus("Initial");
    lastFrameSeconds_ = glfwGetTime();

    while (glfwWindowShouldClose(window_) == GLFW_FALSE) {
        glfwPollEvents();

        if (options_.testContextLoss && frameIndex_ >= 1 &&
            !autoContextLossDone_) {
            autoContextLossDone_ = true;
            contextLossRequested_ = true;
        }
        if (contextLossRequested_) {
            contextLossRequested_ = false;
            if (!simulateContextLoss()) {
                std::cerr << "Context recovery failed; closing cleanly\n";
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
                continue;
            }
            updateTitle(true);
            glfwSwapBuffers(window_);
            if (options_.testContextLoss) {
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            }
            lastFrameSeconds_ = glfwGetTime();
            continue;
        }

        handleResizeIfNeeded();
        ++frameIndex_;
        applyResourceCommands(frameIndex_);

        const double now = glfwGetTime();
        const double deltaSeconds = std::min(now - lastFrameSeconds_, 0.05);
        lastFrameSeconds_ = now;

        std::vector<MovedObject> moved;
        if (!paused_) {
            moved = advanceAnimation(scene_, deltaSeconds);
        }
        lastDirtyCount_ = static_cast<std::uint32_t>(moved.empty()
            ? 0
            : findAffectedLights(scene_.lights, moved).size());

        const std::vector<std::uint32_t> dirtyLights =
            moved.empty() ? std::vector<std::uint32_t>{}
                          : findAffectedLights(scene_.lights, moved);
        const FrameSnapshot snapshot = renderer_.prepareFrame(
            scene_, camera_, preset_, dirtyLights, frameIndex_);
        renderer_.render(scene_, snapshot, debugView_);
        lastRenderedDebugView_ = debugView_;
        renderer_.blitFinalToDefaultFramebuffer();
        updateTitle(false);
        glfwSwapBuffers(window_);
    }

    renderer_.shutdown();
    destroyWindow();
    glfwTerminate();
    glfwInitialized_ = false;
    return options_.testContextLoss && recoveryDifference_ != 0 ? 1 : 0;
}

void Application::handleResizeIfNeeded() {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    if (width <= 0 || height <= 0) {
        return;
    }
    if (resizeRequested_ || width != renderer_.settings().width ||
        height != renderer_.settings().height) {
        renderer_.resize(scene_, width, height);
        settings_ = renderer_.settings();
        resizeRequested_ = false;
        updateTitle(true);
    }
}

void Application::applyResourceCommands(std::uint64_t frame) {
    resourceManager_.setCurrentFrame(frame);
    const std::vector<ResourceCommand> commands =
        resourceManager_.pollAtFrameBoundary(frame);
    for (const ResourceCommand& command : commands) {
        if (command.kind == ResourceRequestKind::DeleteLight) {
            if (findLight(scene_, command.lightId) != nullptr) {
                renderer_.releaseLightShadow(command.lightId);
                removeLight(scene_, command.lightId);
                std::cout << "Deleted light " << command.lightId
                          << " at frame boundary " << frame << '\n';
            }
        } else if (command.kind == ResourceRequestKind::SwitchScene) {
            const std::uint64_t nextGeneration = scene_.generation + 1;
            scene_ = buildDeterministicScene(options_.seed, options_.lightCount,
                                             nextGeneration);
            renderer_.switchScene(scene_);
            lastDirtyCount_ = 0;
            std::cout << "Switched deterministic scene generation "
                      << nextGeneration << " at frame boundary " << frame
                      << "; digest=" << sceneDigestHex(scene_) << '\n';
        }
    }
}

bool Application::simulateContextLoss() {
    std::cout << "Simulating OpenGL context loss...\n";
    const std::vector<std::uint8_t> before = renderer_.readFinalPixels();
    const FrameSnapshot snapshot = renderer_.lastSnapshot();
    const RenderSettings settings = renderer_.settings();

    renderer_.invalidateContextWithoutDelete();
    destroyWindow();
    options_.width = settings.width;
    options_.height = settings.height;
    if (!createWindow()) {
        return false;
    }

    renderer_.recoverFromSnapshot(scene_, snapshot, settings, INDOOR_SHADER_DIR);
    renderer_.render(scene_, snapshot, lastRenderedDebugView_);
    const std::vector<std::uint8_t> after = renderer_.readFinalPixels();
    recoveryDifference_ = countPixelDifferences(before, after);
    recoveryMessage_ = recoveryDifference_ == 0
        ? "OK diff=0" : "FAIL diff=" + std::to_string(recoveryDifference_);
    std::cout << "Context recovery: " << recoveryMessage_
              << ", digest_before=0x" << std::hex << pixelDigest(before)
              << ", digest_after=0x" << pixelDigest(after) << std::dec << '\n';
    renderer_.blitFinalToDefaultFramebuffer();
    return true;
}

void Application::updateTitle(bool force) {
    const double now = glfwGetTime();
    if (!force && now - lastTitleSeconds_ < 0.20) {
        return;
    }
    lastTitleSeconds_ = now;
    const ShadowFrameStats& stats = renderer_.stats();
    const float occupancy = stats.cacheCapacity == 0
        ? 0.0f
        : static_cast<float>(stats.residentShadowMaps) /
          static_cast<float>(stats.cacheCapacity) * 100.0f;
    char title[512];
    std::snprintf(
        title, sizeof(title),
        "IndoorDeferred | seed=%llu lights=%u | validShadows=%u | "
        "cache=%u/%u %.1f%% | migrations=%llu (+%llu) | preset=%s | dbg=%s | "
        "dirty=%u rerenders=%u | recovery=%s | frame=%llu%s",
        static_cast<unsigned long long>(scene_.seed), stats.totalLights,
        stats.validShadowMaps, stats.residentShadowMaps, stats.cacheCapacity,
        static_cast<double>(occupancy),
        static_cast<unsigned long long>(stats.versionMigrations),
        static_cast<unsigned long long>(stats.versionMigrationsThisFrame),
        materialPresetName(preset_), debugViewName(debugView_),
        static_cast<unsigned int>(stats.dirtyLightIds.size()),
        stats.shadowMapsRenderedThisFrame,
        recoveryMessage_.c_str(), static_cast<unsigned long long>(frameIndex_),
        paused_ ? " [PAUSED]" : "");
    glfwSetWindowTitle(window_, title);
}

void Application::printStatus(const char* prefix) const {
    std::cout << prefix << ": seed=" << scene_.seed
              << " lights=" << scene_.lights.size()
              << " objects=" << scene_.objects.size()
              << " digest=" << sceneDigestHex(scene_)
              << " shadowSize=" << settings_.shadowSize
              << " cacheCapacity=" << settings_.shadowCacheCapacity << '\n';
}

void Application::keyCallback(GLFWwindow* window, int key, int, int action,
                              int) {
    Application* app =
        static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app == nullptr) {
        return;
    }
    if (action == GLFW_PRESS && key >= GLFW_KEY_1 && key <= GLFW_KEY_5) {
        app->preset_ = static_cast<MaterialPresetId>(key - GLFW_KEY_1);
        std::cout << "Material preset: " << materialPresetName(app->preset_)
                  << '\n';
        app->updateTitle(true);
        return;
    }
    if (action == GLFW_PRESS && key >= GLFW_KEY_F1 && key <= GLFW_KEY_F7) {
        app->debugView_ = static_cast<DebugView>(key - GLFW_KEY_F1);
        app->updateTitle(true);
        return;
    }
    if (action != GLFW_PRESS) {
        return;
    }
    switch (key) {
    case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        break;
    case GLFW_KEY_TAB: {
        const int next = (static_cast<int>(app->debugView_) + 1) %
                         static_cast<int>(DebugView::Count);
        app->debugView_ = static_cast<DebugView>(next);
        app->updateTitle(true);
        break;
    }
    case GLFW_KEY_T:
        app->preset_ = nextMaterialPreset(app->preset_);
        std::cout << "Material preset: " << materialPresetName(app->preset_)
                  << '\n';
        app->updateTitle(true);
        break;
    case GLFW_KEY_P:
        app->paused_ = !app->paused_;
        app->updateTitle(true);
        break;
    case GLFW_KEY_M: {
        const std::vector<MovedObject> moved =
            advanceAnimation(app->scene_, 1.0 / 30.0);
        app->lastDirtyCount_ = static_cast<std::uint32_t>(
            findAffectedLights(app->scene_.lights, moved).size());
        break;
    }
    case GLFW_KEY_X:
        if (!app->scene_.lights.empty()) {
            ResourceRequest request;
            request.kind = ResourceRequestKind::DeleteLight;
            request.lightId = app->scene_.lights.back().id;
            app->resourceManager_.submit(request);
        }
        break;
    case GLFW_KEY_R: {
        ResourceRequest request;
        request.kind = ResourceRequestKind::SwitchScene;
        app->resourceManager_.submit(request);
        break;
    }
    case GLFW_KEY_L:
        app->contextLossRequested_ = true;
        break;
    default:
        break;
    }
}

void Application::framebufferSizeCallback(GLFWwindow* window, int width,
                                          int height) {
    Application* app =
        static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app != nullptr && width > 0 && height > 0) {
        app->resizeRequested_ = true;
    }
}

void Application::mouseButtonCallback(GLFWwindow* window, int button, int action,
                                      int) {
    Application* app =
        static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app == nullptr || button != GLFW_MOUSE_BUTTON_LEFT) {
        return;
    }
    if (action == GLFW_PRESS) {
        app->mouseDragging_ = true;
        glfwGetCursorPos(window, &app->lastMouseX_, &app->lastMouseY_);
    } else if (action == GLFW_RELEASE) {
        app->mouseDragging_ = false;
    }
}

void Application::cursorPositionCallback(GLFWwindow* window, double x,
                                         double y) {
    Application* app =
        static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app == nullptr || !app->mouseDragging_) {
        return;
    }
    const double dx = x - app->lastMouseX_;
    const double dy = y - app->lastMouseY_;
    app->lastMouseX_ = x;
    app->lastMouseY_ = y;
    app->camera_.yaw -= static_cast<float>(dx) * 0.005f;
    app->camera_.pitch = std::clamp(
        app->camera_.pitch - static_cast<float>(dy) * 0.004f, -0.25f, 1.15f);
}

void Application::scrollCallback(GLFWwindow* window, double, double yOffset) {
    Application* app =
        static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app == nullptr) {
        return;
    }
    app->camera_.distance = std::clamp(
        app->camera_.distance - static_cast<float>(yOffset) * 1.2f, 4.0f, 48.0f);
}

void Application::windowRefreshCallback(GLFWwindow* window) {
    Application* app =
        static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app == nullptr || !app->renderer_.hasGpuResources()) {
        return;
    }
    app->renderer_.render(app->scene_, app->renderer_.lastSnapshot(),
                          app->debugView_);
    app->lastRenderedDebugView_ = app->debugView_;
    app->renderer_.blitFinalToDefaultFramebuffer();
    glfwSwapBuffers(window);
}

} // namespace indoor
