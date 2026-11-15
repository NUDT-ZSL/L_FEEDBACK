#include "application.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "error.hpp"
#include "frame_data.hpp"
#include "gl_loader.hpp"
#include "renderer.hpp"
#include "scene.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace demo {

namespace {

constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;
constexpr double kTitleIntervalSeconds = 0.5;
constexpr double kMaxFrameRate = 200.0;
constexpr auto kVsyncOffFrameBudget =
    std::chrono::duration<double>(1.0 / kMaxFrameRate);

std::string gGlfwError;

void glfwErrorCallback(int errorCode, const char *description) {
    std::ostringstream message;
    message << "GLFW error " << errorCode << ": "
            << (description != nullptr ? description : "unknown error");
    gGlfwError = message.str();
}

[[noreturn]] void selfTestFail(const std::string &message) {
    throw AppError(ErrorCode::SelfTestFailed, message);
}

void require(bool condition, const std::string &message) {
    if (!condition) {
        selfTestFail(message);
    }
}

bool mat4Finite(const glm::mat4 &matrix) {
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            if (!std::isfinite(matrix[column][row])) {
                return false;
            }
        }
    }
    return true;
}

class Application {
public:
    explicit Application(const Options &options) : options_(options) {}
    ~Application() {
        cleanup();
    }

    Application(const Application &) = delete;
    Application &operator=(const Application &) = delete;

    int run() {
        initialize();
        if (options_.selfTest) {
            return runSelfTest();
        }
        return runInteractive();
    }

private:
    void initialize() {
        glfwSetErrorCallback(glfwErrorCallback);
        if (glfwInit() != GLFW_TRUE) {
            throw AppError(ErrorCode::GlfwInitializationFailed,
                           "glfwInit failed: " + gGlfwError);
        }
        glfwInitialized_ = true;

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,
                       options_.simulateContextFailure ? 9 : 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,
                       options_.simulateContextFailure ? 9 : 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_FALSE);
        glfwWindowHint(GLFW_SAMPLES, options_.selfTest ? 0 : 4);
        glfwWindowHint(GLFW_VISIBLE,
                       (options_.selfTest || options_.hidden) ? GLFW_FALSE
                                                              : GLFW_TRUE);

        window_ = glfwCreateWindow(kWindowWidth, kWindowHeight,
                                   "Skeletal Material Sync", nullptr, nullptr);
        if (window_ == nullptr) {
            throw AppError(
                ErrorCode::OpenGl45Unavailable,
                "Window/OpenGL 4.5 core context creation failed: " + gGlfwError);
        }
        glfwSetWindowUserPointer(window_, this);
        glfwSetKeyCallback(window_, &Application::keyCallback);

        glfwMakeContextCurrent(window_);
        if (glfwGetCurrentContext() != window_) {
            throw AppError(ErrorCode::OpenGl45Unavailable,
                           "The OpenGL context could not be made current");
        }

        std::string missingFunctions;
        if (!loadOpenGlFunctions(missingFunctions)) {
            throw AppError(ErrorCode::OpenGlLoaderFailed,
                           "Missing OpenGL function pointers: " + missingFunctions);
        }

        const char *version =
            reinterpret_cast<const char *>(glGetString(GL_VERSION));
        const char *renderer =
            reinterpret_cast<const char *>(glGetString(GL_RENDERER));
        const char *glsl =
            reinterpret_cast<const char *>(glGetString(GL_SHADING_LANGUAGE_VERSION));
        if (version == nullptr) {
            throw AppError(ErrorCode::OpenGl45Unavailable,
                           "glGetString(GL_VERSION) returned null");
        }

        int major = 0;
        int minor = 0;
        if (std::sscanf(version, "%d.%d", &major, &minor) != 2 ||
            major < 4 || (major == 4 && minor < 5)) {
            std::ostringstream message;
            message << "OpenGL 4.5 core is required, received: " << version;
            throw AppError(ErrorCode::OpenGl45Unavailable, message.str());
        }

        std::printf("[INFO] OpenGL: %s\n", version);
        std::printf("[INFO] Renderer: %s\n",
                    renderer != nullptr ? renderer : "<unknown>");
        std::printf("[INFO] GLSL: %s\n", glsl != nullptr ? glsl : "<unknown>");

        vsyncEnabled_ = !options_.startVsyncOff;
        glfwSwapInterval(vsyncEnabled_ ? 1 : 0);

        mesh_ = buildCharacterMesh(options_.simulateResourceFailure);
        renderer_.initialize(
            mesh_, RendererOptions{options_.injectShaderCompileFailure,
                                   options_.injectShaderLinkFailure});

        std::printf(
            "[INFO] Scene: %zu joints, %zu joint mesh sections, %zu vertices, "
            "%zu indices\n",
            kJointCount, mesh_.sections.size(), mesh_.vertices.size(),
            mesh_.indices.size());
        std::printf("[INFO] Controls: 1/2/3 material, SPACE animation, V VSync, "
                    "ESC exit\n");

        updateTitleNow();
    }

    int runInteractive() {
        double previousTime = glfwGetTime();
        int renderedFrames = 0;
        const auto benchmarkStart = std::chrono::steady_clock::now();
        while (glfwWindowShouldClose(window_) == GLFW_FALSE) {
            const auto frameStart = std::chrono::steady_clock::now();
            glfwPollEvents();
            if (glfwWindowShouldClose(window_) != GLFW_FALSE) {
                break;
            }

            const double now = glfwGetTime();
            double deltaSeconds = now - previousTime;
            previousTime = now;
            if (!(deltaSeconds >= 0.0)) {
                deltaSeconds = 0.0;
            }

            step(deltaSeconds, nullptr, false);
            updateFrameRate(deltaSeconds);

            if (!vsyncEnabled_) {
                const auto deadline = frameStart + kVsyncOffFrameBudget;
                std::this_thread::sleep_until(deadline);
            }

            ++renderedFrames;
            if (options_.benchmarkFrames > 0 &&
                renderedFrames >= options_.benchmarkFrames) {
                const std::chrono::duration<double> elapsed =
                    std::chrono::steady_clock::now() - benchmarkStart;
                const double measuredFps =
                    static_cast<double>(renderedFrames) / elapsed.count();
                std::printf(
                    "[BENCHMARK] frames=%d elapsed=%.3f fps=%.2f "
                    "vsync=%s\n",
                    renderedFrames, elapsed.count(), measuredFps,
                    vsyncEnabled_ ? "ON" : "OFF");
                break;
            }
        }
        return 0;
    }

    int runSelfTest() {
        std::printf("[SELFTEST] Validating deterministic scene contract...\n");

        require(kJointDefinitions.size() == kJointCount,
                "Joint definition count mismatch");
        for (std::size_t joint = 0; joint < kJointDefinitions.size(); ++joint) {
            const int parent = kJointDefinitions[joint].parent;
            require(parent < static_cast<int>(joint),
                    "Joint hierarchy is not topologically ordered");
        }
        require(kMaterialPresets.size() == 3, "Expected exactly three materials");
        for (const MaterialPreset &material : kMaterialPresets) {
            require(!material.name.empty(), "Material name is empty");
            require(material.roughness >= 0.0f && material.roughness <= 1.0f,
                    "Material roughness is outside [0, 1]");
            require(material.emissiveStrength >= 0.0f,
                    "Material emissive strength is negative");
        }

        const auto &clips = animationClips();
        require(clips.size() == kClipCount, "Expected exactly three clips");
        for (const AnimationClip &clip : clips) {
            require(!clip.name.empty(), "Animation clip name is empty");
            require(clip.duration > 0.0f, "Animation clip duration must be positive");
            require(clip.keyframes.size() >= 2,
                    "Animation clip must have at least two keyframes");
            require(std::abs(clip.keyframes.front().time) < 1.0e-6f,
                    "Animation clip must start at t=0");
            require(std::abs(clip.keyframes.back().time - clip.duration) < 1.0e-6f,
                    "Animation clip must end at its duration");
            for (std::size_t key = 1; key < clip.keyframes.size(); ++key) {
                require(clip.keyframes[key - 1].time < clip.keyframes[key].time,
                        "Animation keyframe times are not strictly increasing");
            }
        }
        require(animation_.clip() == ClipId::Idle, "Initial clip is not Idle");
        require(materialIndex_ == 0, "Initial material is not preset 1");

        require(mesh_.sections.size() == kJointCount,
                "Each joint must have exactly one mesh section");
        std::array<bool, kJointCount> covered{};
        for (const MeshSection &section : mesh_.sections) {
            require(section.jointIndex < kJointCount,
                    "Mesh section has an invalid joint index");
            require(section.indexCount > 0, "Mesh section is empty");
            covered[section.jointIndex] = true;
        }
        for (bool hasGeometry : covered) {
            require(hasGeometry, "A joint is missing associated geometry");
        }

        std::printf(
            "[SELFTEST] UBO size=%zu, offsets: VP=0 model=64 diffuse=128 "
            "params=144 light=160 eye=176 joints=192\n",
            sizeof(FrameUniforms));
        std::printf(
            "[SELFTEST] Geometry: %zu vertices, %zu indices, %zu sections\n",
            mesh_.vertices.size(), mesh_.indices.size(), mesh_.sections.size());
        std::printf("[SELFTEST] Shader mode: %s\n",
                    renderer_.shaderMode() == ShaderMode::CustomLighting
                        ? "CustomLighting"
                        : "SafeFallback");

        constexpr int frames = 150;
        constexpr double fixedDelta = 1.0 / 60.0;
        bool uboReadBackChecked = false;
        bool screenshotWritten = false;

        for (int frame = 0; frame < frames; ++frame) {
            if (frame == 20) {
                renderer_.updateMeshVertices(mesh_.vertices);
            }
            if (frame == 30) {
                handleKey(GLFW_KEY_V, GLFW_PRESS);
                require(!vsyncEnabled_, "V key did not disable VSync");
            }
            if (frame == 31) {
                handleKey(GLFW_KEY_V, GLFW_PRESS);
                require(vsyncEnabled_, "V key did not re-enable VSync");
            }
            if (frame == 40) {
                handleKey(GLFW_KEY_2, GLFW_PRESS);
                handleKey(GLFW_KEY_SPACE, GLFW_PRESS);
                require(materialIndex_ == 1, "Key 2 did not select material 2");
                require(animation_.clip() == ClipId::Walk,
                        "SPACE did not advance Idle to Walk");
            }
            if (frame == 80) {
                handleKey(GLFW_KEY_3, GLFW_PRESS);
                handleKey(GLFW_KEY_SPACE, GLFW_PRESS);
                require(materialIndex_ == 2, "Key 3 did not select material 3");
                require(animation_.clip() == ClipId::Wave,
                        "SPACE did not advance Walk to Wave");
            }
            if (frame == 130) {
                handleKey(GLFW_KEY_1, GLFW_PRESS);
                handleKey(GLFW_KEY_SPACE, GLFW_PRESS);
                require(materialIndex_ == 0, "Key 1 did not select material 1");
                require(animation_.clip() == ClipId::Idle,
                        "SPACE did not wrap Wave back to Idle");
            }

            const char *screenshot = nullptr;
            if (frame == 100 && !options_.screenshotPath.empty()) {
                screenshot = options_.screenshotPath.c_str();
            }

            step(fixedDelta, screenshot, frame == 100);
            if (screenshot != nullptr) {
                screenshotWritten = true;
            }

            for (std::size_t joint = 0; joint < kJointCount; ++joint) {
                require(mat4Finite(lastFrame_.jointWorld[joint]),
                        "Joint world matrix became non-finite");
            }

            if (frame == 100) {
                const FrameUniforms uploaded = renderer_.readBackFrame();
                require(std::memcmp(&uploaded, &lastFrame_, sizeof(FrameUniforms)) == 0,
                        "UBO readback differs from the synchronized CPU frame packet");
                require(std::abs(uploaded.diffuseColor.r -
                                 kMaterialPresets[materialIndex_].diffuseColor.r) <
                            1.0e-6f,
                        "UBO diffuse color does not match the active material");
                require(std::abs(uploaded.materialParams.y -
                                 kMaterialPresets[materialIndex_].emissiveStrength) <
                            1.0e-6f,
                        "UBO emissive strength does not match the active material");
                uboReadBackChecked = true;
            }
        }

        require(uboReadBackChecked, "UBO readback was not exercised");
        require(renderer_.frameUploadCount() == static_cast<std::size_t>(frames),
                "UBO was not uploaded exactly once per rendered frame");
        require(renderer_.headerUploadCount() >= 1,
                "UBO matrix header was never uploaded");
        require(renderer_.meshUpdateCount() == 1,
                "Dynamic VBO update was not exercised exactly once");
        require(animation_.clip() == ClipId::Idle,
                "Final animation state is not Idle");
        require(materialIndex_ == 0, "Final material state is not preset 1");
        if (!options_.screenshotPath.empty()) {
            require(screenshotWritten, "Screenshot capture was not exercised");
        }

        if (options_.injectShaderCompileFailure ||
            options_.injectShaderLinkFailure) {
            require(renderer_.shaderMode() == ShaderMode::SafeFallback,
                    "Shader failure injection did not activate the safe fallback");
        } else {
            require(renderer_.shaderMode() == ShaderMode::CustomLighting,
                    "Custom shader did not compile and link");
        }

        std::printf(
            "[SELFTEST] Frames=%d, dynamic UBO uploads=%zu, UBO header uploads=%zu, "
            "dynamic VBO updates=%zu\n",
            frames, renderer_.frameUploadCount(), renderer_.headerUploadCount(),
            renderer_.meshUpdateCount());
        std::printf("[SELFTEST] Material and animation keyboard transitions: PASS\n");
        std::printf("[SELFTEST] PASS\n");
        return 0;
    }

    void step(double deltaSeconds, const char *screenshotPath,
              bool verifyVisiblePixels) {
        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(window_, &framebufferWidth, &framebufferHeight);
        if (framebufferWidth <= 0 || framebufferHeight <= 0) {
            if (options_.selfTest) {
                selfTestFail("Hidden self-test window has a zero-sized framebuffer");
            }
            glfwWaitEventsTimeout(0.01);
            return;
        }

        animation_.update(deltaSeconds);
        const Pose pose = sampleClip(animation_.currentClip(),
                                     animation_.timeSeconds());
        const auto jointWorld = computeJointWorldMatrices(pose);

        std::array<glm::mat4, kMaxJoints> gpuJoints{};
        gpuJoints.fill(glm::mat4(1.0f));
        std::copy(jointWorld.begin(), jointWorld.end(), gpuJoints.begin());

        const glm::vec3 cameraPosition(2.4f, 1.35f, 4.0f);
        const glm::mat4 view =
            glm::lookAt(cameraPosition, glm::vec3(0.0f, 0.82f, 0.0f),
                        glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::mat4 projection = glm::perspective(
            glm::radians(45.0f),
            static_cast<float>(framebufferWidth) /
                static_cast<float>(framebufferHeight),
            0.1f, 100.0f);
        const glm::mat4 model =
            glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.10f, 0.0f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(1.45f));

        const MaterialPreset &material = kMaterialPresets[materialIndex_];
        lastFrame_ = makeFrameUniforms(
            projection * view, model, material.diffuseColor, material.roughness,
            material.emissiveStrength, gpuJoints,
            glm::normalize(glm::vec3(0.45f, 0.80f, 0.55f)), cameraPosition,
            animation_.timeSeconds());

        glViewport(0, 0, framebufferWidth, framebufferHeight);
        renderer_.updateFrame(lastFrame_);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        renderer_.draw();

        if (screenshotPath != nullptr) {
            writeScreenshot(screenshotPath, framebufferWidth, framebufferHeight);
        }
        if (verifyVisiblePixels) {
            visiblePixelCount_ =
                countVisiblePixels(framebufferWidth, framebufferHeight);
            require(visiblePixelCount_ > 1000,
                    "Rendered frame contains almost no character pixels");
            std::printf("[SELFTEST] Visible character pixels: %zu\n",
                        visiblePixelCount_);
        }

        if (options_.injectGlError) {
            glEnable(static_cast<GLenum>(0xFFFFFFFFu));
        }
        glfwSwapBuffers(window_);
        pollGlErrors();
    }

    void writeScreenshot(const std::string &path, int width, int height) {
        std::vector<unsigned char> pixels(static_cast<std::size_t>(width) *
                                          static_cast<std::size_t>(height) * 3u);
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        const GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            throw AppError(ErrorCode::ScreenshotFailed,
                           std::string("glReadPixels failed with ") +
                               glErrorName(error));
        }

        std::ofstream output(path, std::ios::binary);
        if (!output) {
            throw AppError(ErrorCode::ScreenshotFailed,
                           "Unable to open screenshot path: " + path);
        }
        output << "P6\n" << width << ' ' << height << "\n255\n";
        const std::size_t rowBytes = static_cast<std::size_t>(width) * 3u;
        for (int row = height - 1; row >= 0; --row) {
            const auto *rowStart =
                pixels.data() + static_cast<std::size_t>(row) * rowBytes;
            output.write(reinterpret_cast<const char *>(rowStart),
                         static_cast<std::streamsize>(rowBytes));
        }
        if (!output.good()) {
            throw AppError(ErrorCode::ScreenshotFailed,
                           "Failed while writing screenshot: " + path);
        }
        std::printf("[INFO] Screenshot: %s\n", path.c_str());
    }

    std::size_t countVisiblePixels(int width, int height) {
        std::vector<unsigned char> pixels(static_cast<std::size_t>(width) *
                                          static_cast<std::size_t>(height) * 3u);
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        const GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            throw AppError(ErrorCode::ScreenshotFailed,
                           std::string("glReadPixels pixel check failed with ") +
                               glErrorName(error));
        }

        // Clear color is (0.035, 0.045, 0.070); anything materially brighter or
        // differently colored is considered rendered character geometry.
        constexpr int clearRed = 9;
        constexpr int clearGreen = 11;
        constexpr int clearBlue = 18;
        std::size_t visible = 0;
        for (std::size_t offset = 0; offset + 2 < pixels.size(); offset += 3) {
            const int red = pixels[offset];
            const int green = pixels[offset + 1];
            const int blue = pixels[offset + 2];
            int delta = std::abs(red - clearRed) + std::abs(green - clearGreen) +
                        std::abs(blue - clearBlue);
            if (delta > 24) {
                ++visible;
            }
        }
        return visible;
    }

    void pollGlErrors() {
        bool errorThisFrame = false;
        GLenum error = GL_NO_ERROR;
        while ((error = glGetError()) != GL_NO_ERROR) {
            errorThisFrame = true;
            std::fprintf(stderr, "[GL] %s\n", glErrorName(error));
        }

        if (!errorThisFrame) {
            consecutiveGlErrorFrames_ = 0;
            return;
        }

        ++consecutiveGlErrorFrames_;
        std::fprintf(stderr,
                     "[WARN] OpenGL error detected on consecutive frame %d/3\n",
                     consecutiveGlErrorFrames_);
        if (consecutiveGlErrorFrames_ >= 3) {
            throw AppError(ErrorCode::PersistentGlError,
                           "glGetError reported an error on three consecutive frames");
        }
    }

    void handleKey(int key, int action) {
        if (action != GLFW_PRESS) {
            return;
        }

        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
                break;
            case GLFW_KEY_SPACE:
                animation_.nextClip();
                std::printf("[STATE] Animation -> %s\n",
                            std::string(animation_.currentClip().name).c_str());
                updateTitleNow();
                break;
            case GLFW_KEY_1:
            case GLFW_KEY_2:
            case GLFW_KEY_3:
                materialIndex_ = static_cast<std::size_t>(key - GLFW_KEY_1);
                std::printf("[STATE] Material -> %s\n",
                            std::string(kMaterialPresets[materialIndex_].name).c_str());
                updateTitleNow();
                break;
            case GLFW_KEY_V:
                vsyncEnabled_ = !vsyncEnabled_;
                glfwSwapInterval(vsyncEnabled_ ? 1 : 0);
                std::printf("[STATE] VSync -> %s\n",
                            vsyncEnabled_ ? "ON" : "OFF (200 FPS cap)");
                updateTitleNow();
                break;
            default:
                break;
        }
    }

    void updateFrameRate(double deltaSeconds) {
        fpsElapsedSeconds_ += std::max(deltaSeconds, 0.0);
        ++fpsFrameCount_;
        if (fpsElapsedSeconds_ >= kTitleIntervalSeconds) {
            fps_ = static_cast<double>(fpsFrameCount_) / fpsElapsedSeconds_;
            fpsElapsedSeconds_ = 0.0;
            fpsFrameCount_ = 0;
            updateTitleNow();
        }
    }

    void updateTitleNow() {
        char title[512];
        std::snprintf(
            title, sizeof(title),
            "Skeletal Material Sync | Material: %s | Animation: %s | FPS: %.1f | "
            "VSync: %s",
            std::string(kMaterialPresets[materialIndex_].name).c_str(),
            std::string(animation_.currentClip().name).c_str(), fps_,
            vsyncEnabled_ ? "ON" : "OFF");
        glfwSetWindowTitle(window_, title);
    }

    static void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                            int mods) {
        (void)scancode;
        (void)mods;
        auto *application =
            static_cast<Application *>(glfwGetWindowUserPointer(window));
        if (application != nullptr) {
            application->handleKey(key, action);
        }
    }

    void cleanup() {
        if (window_ != nullptr && glDeleteProgram != nullptr) {
            renderer_.destroy();
        }
        if (window_ != nullptr) {
            glfwSetWindowUserPointer(window_, nullptr);
            glfwDestroyWindow(window_);
            window_ = nullptr;
        }
        if (glfwInitialized_) {
            glfwSetErrorCallback(nullptr);
            glfwTerminate();
            glfwInitialized_ = false;
        }
    }

    Options options_;
    GLFWwindow *window_ = nullptr;
    bool glfwInitialized_ = false;
    Renderer renderer_;
    MeshData mesh_;
    AnimationState animation_;
    std::size_t materialIndex_ = 0;
    bool vsyncEnabled_ = true;
    FrameUniforms lastFrame_{};
    int consecutiveGlErrorFrames_ = 0;
    double fps_ = 0.0;
    double fpsElapsedSeconds_ = 0.0;
    std::size_t fpsFrameCount_ = 0;
    std::size_t visiblePixelCount_ = 0;
};

} // namespace

Options parseCommandLine(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            options.showHelp = true;
        } else if (argument == "--self-test" || argument == "--selftest") {
            options.selfTest = true;
            options.hidden = true;
        } else if (argument == "--hidden") {
            options.hidden = true;
        } else if (argument == "--inject-shader-compile-failure") {
            options.injectShaderCompileFailure = true;
        } else if (argument == "--inject-shader-link-failure") {
            options.injectShaderLinkFailure = true;
        } else if (argument == "--simulate-context-failure") {
            options.simulateContextFailure = true;
        } else if (argument == "--simulate-resource-failure") {
            options.simulateResourceFailure = true;
        } else if (argument == "--inject-gl-error") {
            options.injectGlError = true;
        } else if (argument == "--vsync-off") {
            options.startVsyncOff = true;
        } else if (argument == "--benchmark-frames") {
            if (index + 1 >= argc) {
                throw AppError(ErrorCode::CommandLineError,
                               "--benchmark-frames requires a positive frame count");
            }
            const char *value = argv[++index];
            char *end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (end == value || *end != '\0' || parsed <= 0 || parsed > 100000) {
                throw AppError(ErrorCode::CommandLineError,
                               "Invalid --benchmark-frames value: " +
                                   std::string(value));
            }
            options.benchmarkFrames = static_cast<int>(parsed);
            options.hidden = true;
        } else if (argument == "--screenshot") {
            if (index + 1 >= argc) {
                throw AppError(ErrorCode::CommandLineError,
                               "--screenshot requires a file path");
            }
            options.screenshotPath = argv[++index];
        } else {
            throw AppError(ErrorCode::CommandLineError,
                           "Unknown command-line argument: " + argument);
        }
    }

    if (options.injectShaderCompileFailure &&
        options.injectShaderLinkFailure) {
        throw AppError(ErrorCode::CommandLineError,
                       "Compile-failure and link-failure injection are mutually "
                       "exclusive");
    }
    if (options.selfTest && options.benchmarkFrames > 0) {
        throw AppError(ErrorCode::CommandLineError,
                       "--benchmark-frames cannot be combined with --self-test");
    }
    return options;
}

void printUsage(const char *executableName) {
    std::printf(
        "Usage: %s [options]\n"
        "  --self-test                         Run deterministic GPU integration test\n"
        "  --hidden                            Create a hidden window\n"
        "  --screenshot <path.ppm>             Capture a screenshot during self-test\n"
        "  --inject-shader-compile-failure     Force compile failure and safe fallback\n"
        "  --inject-shader-link-failure        Force link failure and safe fallback\n"
        "  --simulate-context-failure          Exercise the context failure exit path\n"
        "  --simulate-resource-failure         Corrupt procedural data and exit E3001\n"
        "  --inject-gl-error                   Exit E1010 after 3 consecutive GL errors\n"
        "  --benchmark-frames <N>              Hidden normal-loop FPS measurement\n"
        "  --vsync-off                         Start with VSync disabled\n"
        "  --help                              Show this help\n",
        executableName);
}

int runApplication(const Options &options) {
    Application application(options);
    return application.run();
}

} // namespace demo
