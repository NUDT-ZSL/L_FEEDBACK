#pragma once

#include "indoor/DeferredRenderer.hpp"
#include "indoor/ResourceManager.hpp"
#include "indoor/Scene.hpp"

#include <cstdint>
#include <string>

struct GLFWwindow;

namespace indoor {

struct LaunchOptions {
    std::uint64_t seed = 1337;
    std::uint32_t lightCount = 256;
    int width = 1280;
    int height = 720;
    std::uint32_t shadowCacheCapacity = 0;
    int shadowSize = 0;
    bool testContextLoss = false;
};

class Application {
public:
    explicit Application(LaunchOptions options);
    ~Application();

    int run();

private:
    bool createWindow();
    void destroyWindow();
    void installCallbacks();
    void handleResizeIfNeeded();
    void applyResourceCommands(std::uint64_t frame);
    bool simulateContextLoss();
    void updateTitle(bool force);
    void printStatus(const char* prefix) const;

    static void keyCallback(GLFWwindow* window, int key, int scancode, int action,
                            int mods);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action,
                                    int mods);
    static void cursorPositionCallback(GLFWwindow* window, double x, double y);
    static void scrollCallback(GLFWwindow* window, double xOffset, double yOffset);
    static void windowRefreshCallback(GLFWwindow* window);

    LaunchOptions options_;
    GLFWwindow* window_ = nullptr;
    Scene scene_;
    Camera camera_;
    DeferredRenderer renderer_;
    ResourceManager resourceManager_;
    RenderSettings settings_;
    MaterialPresetId preset_ = MaterialPresetId::FullLayered;
    DebugView debugView_ = DebugView::Final;
    DebugView lastRenderedDebugView_ = DebugView::Final;
    std::uint64_t frameIndex_ = 0;
    bool paused_ = false;
    bool glfwInitialized_ = false;
    bool autoContextLossDone_ = false;
    bool contextLossRequested_ = false;
    bool resizeRequested_ = false;
    bool mouseDragging_ = false;
    double lastMouseX_ = 0.0;
    double lastMouseY_ = 0.0;
    double lastFrameSeconds_ = 0.0;
    double lastTitleSeconds_ = -1.0;
    int recoveryDifference_ = -1;
    std::string recoveryMessage_ = "none";
    std::uint32_t lastDirtyCount_ = 0;
};

} // namespace indoor
