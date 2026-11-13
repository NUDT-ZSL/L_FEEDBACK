#include "app.hpp"

#include "gl/gl_renderer.hpp"
#include "terrain/cache.hpp"
#include "terrain/camera_path.hpp"
#include "terrain/lod.hpp"
#include "terrain/seams.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace terrain {
namespace {

PathLoadResult resolvePath(const Options& options) {
    if (options.cameraPath.empty()) {
        PathLoadResult result;
        result.ok = true;
        result.path = builtInCameraPath();
        return result;
    }
    return loadCameraPath(options.cameraPath);
}

std::string captureFilename(const std::string& directory, uint64_t frame) {
    std::ostringstream name;
    name << "frame_" << std::setw(6) << std::setfill('0') << frame << ".ppm";
    return (std::filesystem::path(directory) / name.str()).string();
}

int failGl(const std::string& message) {
    std::cerr << "E_GL_INIT: " << message << "\n";
    return 4;
}

}  // namespace

int runGlApp(const Options& options) {
    const bool wantsCapture =
        !options.screenshot.empty() || !options.captureDirectory.empty();
    const bool interactive =
        options.interactive ||
        (!options.replay && !wantsCapture && !options.headless);

    PathLoadResult pathResult;
    if (!interactive) {
        pathResult = resolvePath(options);
        if (!pathResult.ok) {
            std::cerr << "E_BAD_PATH: " << pathResult.error;
            if (pathResult.line > 0) {
                std::cerr << " at line " << pathResult.line;
            }
            std::cerr << "\n";
            return 3;
        }
    } else {
        pathResult.ok = true;
        pathResult.path = builtInCameraPath();
    }

    if (glfwInit() == GLFW_FALSE) {
        if (options.backend == Backend::Auto && !interactive) {
            std::cerr << "warning: GLFW initialization failed; using software backend\n";
            return runSoftwareApp(options);
        }
        return failGl("GLFW initialization failed");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, interactive ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(
        options.width,
        options.height,
        "Terrain LOD Stitching",
        nullptr,
        nullptr);
    if (window == nullptr) {
        glfwTerminate();
        if (options.backend == Backend::Auto && !interactive) {
            std::cerr << "warning: OpenGL 4.6 context unavailable; using software backend\n";
            return runSoftwareApp(options);
        }
        return failGl("unable to create an OpenGL 4.6 window/context");
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(interactive ? 1 : 0);

    Terrain terrain(options.terrain);
    GlRenderer renderer(terrain, options.width, options.height);
    std::string error;
    if (!renderer.initialize(error)) {
        glfwDestroyWindow(window);
        glfwTerminate();
        if (options.backend == Backend::Auto && !interactive) {
            std::cerr << "warning: OpenGL backend initialization failed; using software backend\n";
            return runSoftwareApp(options);
        }
        return failGl(error);
    }

    LodSelector selector(terrain, options.sseThreshold);
    SeamSolver seams(terrain);
    NodeStateCache cache(static_cast<std::size_t>(options.cacheCapacity));

    int exitCode = 0;
    if (interactive) {
        Camera camera = pathResult.path.sample(
            0.0f,
            static_cast<float>(options.width) /
                static_cast<float>(options.height));
        DebugView view = options.debugView;
        double previousTime = glfwGetTime();
        uint64_t frameNumber = 0;
        while (glfwWindowShouldClose(window) == GLFW_FALSE) {
            glfwPollEvents();
            const double now = glfwGetTime();
            const float delta = static_cast<float>(
                std::clamp(now - previousTime, 0.0, 0.1));
            previousTime = now;

            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
                view = DebugView::Normal;
            }
            if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
                view = DebugView::Wireframe;
            }
            if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
                view = DebugView::Lod;
            }
            if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
                view = DebugView::Seam;
            }
            if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
                const std::string filename =
                    options.screenshot.empty()
                        ? "terrain_lod.ppm"
                        : options.screenshot;
                const SceneFrame frame = buildSceneFrame(
                    terrain,
                    selector,
                    seams,
                    cache,
                    camera,
                    options.width,
                    options.height,
                    view,
                    frameNumber,
                    static_cast<float>(now));
                if (!renderer.render(frame, view, error)) {
                    std::cerr << "E_RENDER_FAIL: " << error << "\n";
                    exitCode = 4;
                    break;
                }
                if (!renderer.capturePpm(filename, error)) {
                    std::cerr << "E_RENDER_FAIL: " << error << "\n";
                    exitCode = 4;
                    break;
                }
            }

            const float turnSpeed = 72.0f * delta;
            if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
                camera.yawDegrees -= turnSpeed;
            }
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
                camera.yawDegrees += turnSpeed;
            }
            if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
                camera.pitchDegrees =
                    clampf(camera.pitchDegrees + turnSpeed, -89.0f, 89.0f);
            }
            if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
                camera.pitchDegrees =
                    clampf(camera.pitchDegrees - turnSpeed, -89.0f, 89.0f);
            }

            const float moveSpeed = 180.0f * delta;
            const glm::vec3 forward{
                camera.forward().x,
                camera.forward().y,
                camera.forward().z,
            };
            const glm::vec3 right =
                glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
            glm::vec3 movement{0.0f};
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                movement += forward;
            }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                movement -= forward;
            }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                movement += right;
            }
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                movement -= right;
            }
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
                movement.y += 1.0f;
            }
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
                movement.y -= 1.0f;
            }
            if (glm::length(movement) > 0.0f) {
                movement = glm::normalize(movement) * moveSpeed;
                camera.position.x += movement.x;
                camera.position.y += movement.y;
                camera.position.z += movement.z;
            }

            const SceneFrame frame = buildSceneFrame(
                terrain,
                selector,
                seams,
                cache,
                camera,
                options.width,
                options.height,
                view,
                frameNumber,
                static_cast<float>(now));
            if (!renderer.render(frame, view, error)) {
                std::cerr << "E_RENDER_FAIL: " << error << "\n";
                exitCode = 4;
                break;
            }
            renderer.presentToDefaultFramebuffer();
            glfwSwapBuffers(window);
            ++frameNumber;
        }
    } else {
        const bool captureEveryFrame = !options.captureDirectory.empty();
        for (int frameIndex = 0; frameIndex < options.frames; ++frameIndex) {
            const float time =
                static_cast<float>(frameIndex) * options.timeStep;
            const Camera camera = pathResult.path.sample(
                time,
                static_cast<float>(options.width) /
                    static_cast<float>(options.height));
            const SceneFrame frame = buildSceneFrame(
                terrain,
                selector,
                seams,
                cache,
                camera,
                options.width,
                options.height,
                options.debugView,
                static_cast<uint64_t>(frameIndex),
                time);
            printStatsJson(std::cout, frame, terrain, options.debugView);
            std::cout.flush();
            if (!renderer.render(frame, options.debugView, error)) {
                std::cerr << "E_RENDER_FAIL: " << error << "\n";
                exitCode = 4;
                break;
            }

            const bool finalFrame = frameIndex + 1 == options.frames;
            if (captureEveryFrame || (finalFrame && !options.screenshot.empty())) {
                const std::string filename =
                    captureEveryFrame
                        ? captureFilename(
                              options.captureDirectory,
                              static_cast<uint64_t>(frameIndex))
                        : options.screenshot;
                if (!renderer.capturePpm(filename, error)) {
                    std::cerr << "E_RENDER_FAIL: " << error << "\n";
                    exitCode = 4;
                    break;
                }
            }
        }
    }

    renderer.shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return exitCode;
}

}  // namespace terrain

