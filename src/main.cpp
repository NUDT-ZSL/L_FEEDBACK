#include "gl_api.h"
#include "mesh.h"
#include "renderer.h"
#include "scene.h"
#include "subdivision.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

using subdiv::CameraSettings;
using subdiv::PolygonMesh;
using subdiv::Renderer;
using subdiv::SceneSettings;
using subdiv::TriangleMesh;
using subdiv::Vec3;

struct Options {
    std::string scenePath = "scene.json";
    int frames = -1;
    int stressSwitches = 0;
    int levelOverride = -1;
    bool normalMode = false;
    bool useFaceNormal = false;
    bool wireframe = false;
    bool listEdges = false;
    std::string screenshotPath;
    std::string traceMapPath;
};

struct AppState {
    SceneSettings scene;
    PolygonMesh baseMesh;
    TriangleMesh triangleMesh;
    Renderer renderer;
    int level = 0;
    int requestedLevel = 0;
    bool normalMode = false;
    bool useFaceNormal = false;
    bool wireframe = false;
    CameraSettings camera;
    bool determinismOnly = false;
};

void glfwErrorCallback(int code, const char* description) {
    std::cerr << "[glfw] error " << code << ": "
              << (description != nullptr ? description : "unknown error") << '\n';
}

void printUsage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "  --scene PATH          JSON scene file (default: scene.json)\n"
        << "  --level N             Override initial subdivision level\n"
        << "  --normal-mode         Start with normal visualization enabled\n"
        << "  --face-normal         Use face normals in normal visualization\n"
        << "  --wireframe           Start in wireframe mode\n"
        << "  --frames N            Render N frames, then exit (automation)\n"
        << "  --stress-switches N   Toggle subdivision level N times, then exit\n"
        << "  --screenshot PATH     Save the current framebuffer as binary PPM\n"
        << "  --trace-map PATH      Write new-vertex -> original-vertex map on switches\n"
        << "  --list-edges          Print deterministic edge indices and weights\n"
        << "  --help                Show this help\n\n"
        << "Keys: Left/Down = coarser, Right/Up = finer, 0-6 = direct level,\n"
        << "      N = normal mode, F = face/vertex normals, W = wireframe,\n"
        << "      R = reset camera, Esc = close window.\n";
}

bool validateGraphicalDisplay(std::string& error) {
#if defined(__linux__)
    const char* displayValue = std::getenv("DISPLAY");
    const char* waylandValue = std::getenv("WAYLAND_DISPLAY");
    const char* runtimeValue = std::getenv("XDG_RUNTIME_DIR");
    const std::string display = displayValue != nullptr ? displayValue : "";
    const std::string wayland = waylandValue != nullptr ? waylandValue : "";
    const std::string runtime = runtimeValue != nullptr ? runtimeValue : "";

    bool x11Reachable = false;
    bool waylandReachable = false;
    if (!display.empty()) {
        if (display.front() == ':') {
            std::size_t end = 1;
            while (end < display.size() &&
                   std::isdigit(static_cast<unsigned char>(display[end])) != 0) {
                ++end;
            }
            if (end > 1) {
                const std::string socket =
                    "/tmp/.X11-unix/X" + display.substr(1, end - 1);
                x11Reachable = ::access(socket.c_str(), F_OK) == 0;
            }
        } else {
            x11Reachable = true;
        }
    }
    if (!wayland.empty()) {
        if (wayland.front() == '/') {
            waylandReachable = ::access(wayland.c_str(), F_OK) == 0;
        } else if (!runtime.empty()) {
            const std::string socket = runtime + "/" + wayland;
            waylandReachable = ::access(socket.c_str(), F_OK) == 0;
        }
    }
    if (!x11Reachable && !waylandReachable) {
        error = "no reachable X11 or Wayland display (DISPLAY='" + display +
                "', WAYLAND_DISPLAY='" + wayland + "')";
        return false;
    }
#else
    (void)error;
#endif
    return true;
}

bool parseInteger(const std::string& text, int& value) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == nullptr || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool parseOptions(int argc, char** argv, Options& options, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto requireValue = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                error = std::string("missing value after ") + name;
                return nullptr;
            }
            return argv[++i];
        };

        if (argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (argument == "--scene") {
            const char* value = requireValue("--scene");
            if (value == nullptr) {
                return false;
            }
            options.scenePath = value;
        } else if (argument == "--frames") {
            const char* value = requireValue("--frames");
            if (value == nullptr || !parseInteger(value, options.frames)) {
                error = "--frames requires a non-negative integer";
                return false;
            }
        } else if (argument == "--stress-switches") {
            const char* value = requireValue("--stress-switches");
            if (value == nullptr || !parseInteger(value, options.stressSwitches)) {
                error = "--stress-switches requires a non-negative integer";
                return false;
            }
        } else if (argument == "--level") {
            const char* value = requireValue("--level");
            if (value == nullptr || !parseInteger(value, options.levelOverride)) {
                error = "--level requires a non-negative integer";
                return false;
            }
        } else if (argument == "--normal-mode") {
            options.normalMode = true;
        } else if (argument == "--face-normal") {
            options.useFaceNormal = true;
        } else if (argument == "--wireframe") {
            options.wireframe = true;
        } else if (argument == "--list-edges") {
            options.listEdges = true;
        } else if (argument == "--screenshot") {
            const char* value = requireValue("--screenshot");
            if (value == nullptr) {
                return false;
            }
            options.screenshotPath = value;
        } else if (argument == "--trace-map") {
            const char* value = requireValue("--trace-map");
            if (value == nullptr) {
                return false;
            }
            options.traceMapPath = value;
        } else {
            error = "unrecognized argument: " + argument;
            return false;
        }
    }
    return true;
}

double processMemoryMiB() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) != 0) {
        return static_cast<double>(counters.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return -1.0;
#else
    std::ifstream file("/proc/self/statm");
    long long totalPages = 0;
    long long residentPages = 0;
    if (file >> totalPages >> residentPages) {
        (void)totalPages;
        const long pageSize = sysconf(_SC_PAGESIZE);
        if (pageSize > 0) {
            return static_cast<double>(residentPages) *
                   static_cast<double>(pageSize) / (1024.0 * 1024.0);
        }
    }
    return -1.0;
#endif
}

PolygonMesh subdivideToLevel(const PolygonMesh& base, int level, double& milliseconds) {
    const auto start = std::chrono::steady_clock::now();
    PolygonMesh result = base;
    for (int i = 0; i < level; ++i) {
        result = subdiv::subdivideCatmullClarkOnce(result);
    }
    const auto finish = std::chrono::steady_clock::now();
    milliseconds =
        std::chrono::duration<double, std::milli>(finish - start).count();
    return result;
}

bool writeTraceMap(const PolygonMesh& mesh, const std::string& path,
                   std::string& error) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        error = "unable to open trace map path '" + path + "'";
        return false;
    }
    file << "new_vertex\trepresentative_original\tsource_original_vertices\n";
    for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
        file << i << '\t'
             << (i < mesh.representativeOriginalVertex.size()
                     ? mesh.representativeOriginalVertex[i]
                     : 0U)
             << '\t';
        if (i < mesh.sourceOriginalVertices.size()) {
            const auto& sources = mesh.sourceOriginalVertices[i];
            for (std::size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
                if (sourceIndex != 0) {
                    file << ',';
                }
                file << sources[sourceIndex];
            }
        }
        file << '\n';
    }
    if (!file) {
        error = "failed while writing trace map '" + path + "'";
        return false;
    }
    return true;
}

bool switchLevel(AppState& app, int newLevel, const Options& options) {
    newLevel = std::max(0, std::min(newLevel, app.scene.maxSubdivision));
    if (newLevel == app.level) {
        return true;
    }

    double subdivisionMilliseconds = 0.0;
    PolygonMesh mesh = subdivideToLevel(app.baseMesh, newLevel, subdivisionMilliseconds);
    std::string error;
    if (!app.renderer.rebuildGpuMesh(mesh, newLevel, error)) {
        std::cerr << "[fatal] GPU mesh rebuild failed at level " << newLevel << ": "
                  << error << '\n';
        return false;
    }

    const subdiv::MeshStats stats =
        subdiv::computeMeshStats(mesh, app.triangleMesh.indices.size() / 3);
    const std::string shaderStatus = app.renderer.shaderStatus();
    std::cout << "[stats] level=" << newLevel
              << " original_triangles=" << stats.triangles
              << " subdivided_quads=" << stats.quads
              << " subdivision_ms=" << std::fixed << std::setprecision(3)
              << subdivisionMilliseconds
              << " shader=" << shaderStatus
              << " vertices=" << stats.vertices
              << " gpu_triangles=" << app.renderer.gpuTriangleCount()
              << " boundary_edges=" << stats.boundaryEdges
              << " crease_edges=" << stats.sharpEdges
              << " traceable_vertices=" << stats.traceableVertices
              << " rss_mib=" << std::fixed << std::setprecision(1)
              << processMemoryMiB()
              << std::endl;

    if (!options.traceMapPath.empty() &&
        !writeTraceMap(mesh, options.traceMapPath, error)) {
        std::cerr << "[fatal] " << error << '\n';
        return false;
    }

    if (!app.renderer.checkGlError("after level switch", error)) {
        std::cerr << "[fatal] OpenGL error after switching to level " << newLevel
                  << ": " << error << '\n';
        return false;
    }

    app.level = newLevel;
    app.requestedLevel = newLevel;
    return true;
}

void updateWindowTitle(GLFWwindow* window, const AppState& app) {
    std::ostringstream title;
    title << "Catmull-Clark Offline Preview | level " << app.level << '/'
          << app.scene.maxSubdivision;
    if (app.normalMode) {
        title << " | normals:" << (app.useFaceNormal ? "face" : "vertex");
    } else {
        title << " | shaded";
    }
    if (app.wireframe) {
        title << " | wireframe";
    }
    glfwSetWindowTitle(window, title.str().c_str());
}

void keyCallback(GLFWwindow* window, int key, int, int action, int) {
    auto* app = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (app == nullptr) {
        return;
    }
    if (action != GLFW_PRESS && action != GLFW_REPEAT) {
        return;
    }

    switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        case GLFW_KEY_UP:
        case GLFW_KEY_RIGHT:
            app->requestedLevel =
                std::min(app->scene.maxSubdivision, app->requestedLevel + 1);
            break;
        case GLFW_KEY_DOWN:
        case GLFW_KEY_LEFT:
            app->requestedLevel = std::max(0, app->requestedLevel - 1);
            break;
        case GLFW_KEY_0:
        case GLFW_KEY_1:
        case GLFW_KEY_2:
        case GLFW_KEY_3:
        case GLFW_KEY_4:
        case GLFW_KEY_5:
        case GLFW_KEY_6:
            app->requestedLevel =
                std::min(app->scene.maxSubdivision, key - GLFW_KEY_0);
            break;
        case GLFW_KEY_N:
            app->normalMode = !app->normalMode;
            updateWindowTitle(window, *app);
            break;
        case GLFW_KEY_F:
            app->useFaceNormal = !app->useFaceNormal;
            updateWindowTitle(window, *app);
            break;
        case GLFW_KEY_W:
            app->wireframe = !app->wireframe;
            updateWindowTitle(window, *app);
            break;
        case GLFW_KEY_R:
            app->camera = app->scene.camera;
            break;
        default:
            break;
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    std::string optionError;
    if (!parseOptions(argc, argv, options, optionError)) {
        std::cerr << "[fatal] " << optionError << "\n\n";
        printUsage(argv[0]);
        return 1;
    }

    subdiv::SceneLoadResult sceneResult = subdiv::loadSceneFile(options.scenePath);
    for (const std::string& warning : sceneResult.warnings) {
        std::cout << "[scene] warning: " << warning << '\n';
    }

    SceneSettings scene = sceneResult.settings;
    if (options.levelOverride >= 0) {
        if (options.levelOverride > scene.maxSubdivision) {
            std::cout << "[scene] warning: CLI --level " << options.levelOverride
                      << " exceeds maxSubdivision " << scene.maxSubdivision
                      << "; clamping to " << scene.maxSubdivision << '\n';
            scene.initialSubdivision = scene.maxSubdivision;
        } else {
            scene.initialSubdivision = options.levelOverride;
        }
    }

    TriangleMesh triangleMesh = subdiv::buildDeterministicMesh(scene.seed);
    subdiv::applyCreaseOverrides(triangleMesh, scene.creaseOverrides, "creaseWeights");
    PolygonMesh baseMesh = subdiv::triangleMeshToPolygonMesh(triangleMesh);

    std::cout << "[mesh] seed=" << scene.seed
              << " vertices=" << triangleMesh.positions.size()
              << " triangles=" << triangleMesh.indices.size() / 3
              << " edges=" << triangleMesh.edges.size()
              << " fingerprint=0x" << std::hex
              << subdiv::fingerprint(triangleMesh) << std::dec << '\n';

    if (options.listEdges) {
        for (std::size_t i = 0; i < triangleMesh.edges.size(); ++i) {
            const auto& edge = triangleMesh.edges[i];
            std::cout << "[edge] index=" << i << " vertices=" << edge.a << '-' << edge.b
                      << " creaseWeight=" << std::fixed << std::setprecision(3)
                      << edge.creaseWeight
                      << (edge.isBoundary() ? " boundary" : " interior") << '\n';
        }
    }

    std::string displayError;
    if (!validateGraphicalDisplay(displayError)) {
        std::cerr << "[fatal] graphical display preflight failed: " << displayError << '\n';
        return 2;
    }

    glfwSetErrorCallback(glfwErrorCallback);
    if (glfwInit() == GLFW_FALSE) {
        std::cerr << "[fatal] GLFW initialization failed\n";
        return 2;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 760,
                                          "Catmull-Clark Offline Preview",
                                          nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "[fatal] GLFW window or OpenGL context creation failed\n";
        glfwTerminate();
        return 3;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(options.frames >= 0 || options.stressSwitches > 0 ? 0 : 1);

    std::vector<std::string> missingFunctions;
    if (!subdiv::gl::loadFunctions(missingFunctions)) {
        std::cerr << "[fatal] failed to load required OpenGL functions:";
        for (const std::string& name : missingFunctions) {
            std::cerr << ' ' << name;
        }
        std::cerr << '\n';
        glfwDestroyWindow(window);
        glfwTerminate();
        return 4;
    }

    const int contextMajor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    const int contextMinor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    if (contextMajor < 4 || (contextMajor == 4 && contextMinor < 1)) {
        std::cerr << "[fatal] OpenGL 4.1+ is required, got " << contextMajor << '.'
                  << contextMinor << '\n';
        glfwDestroyWindow(window);
        glfwTerminate();
        return 4;
    }

    const auto* rendererName = reinterpret_cast<const char*>(subdiv::gl::GetString(
        subdiv::gl::GL_RENDERER));
    const auto* versionName = reinterpret_cast<const char*>(subdiv::gl::GetString(
        subdiv::gl::GL_VERSION));
    std::cout << "[gl] context=" << contextMajor << '.' << contextMinor
              << " renderer=" << (rendererName != nullptr ? rendererName : "unknown")
              << " version=" << (versionName != nullptr ? versionName : "unknown") << '\n';

    AppState app;
    app.scene = scene;
    app.baseMesh = std::move(baseMesh);
    app.triangleMesh = std::move(triangleMesh);
    app.level = -1;
    app.requestedLevel = scene.initialSubdivision;
    app.normalMode = options.normalMode;
    app.useFaceNormal = options.useFaceNormal;
    app.wireframe = options.wireframe;
    app.camera = scene.camera;

    std::string error;
    if (!app.renderer.initialize(error)) {
        std::cerr << "[fatal] shader compilation/link failed: " << error << '\n';
        app.renderer.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 5;
    }

    if (!switchLevel(app, scene.initialSubdivision, options)) {
        app.renderer.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 6;
    }

    glfwSetWindowUserPointer(window, &app);
    glfwSetKeyCallback(window, keyCallback);
    updateWindowTitle(window, app);

    int renderedFrames = 0;
    int stressRemaining = options.stressSwitches;
    int stressDirection = 1;
    bool wroteScreenshot = false;
    bool runtimeError = false;

    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        glfwPollEvents();

        if (app.requestedLevel != app.level) {
            if (!switchLevel(app, app.requestedLevel, options)) {
                runtimeError = true;
                break;
            }
            updateWindowTitle(window, app);
        }

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if (width > 0 && height > 0) {
            app.renderer.render(app.camera, width, height, app.scene.diffuse,
                                app.scene.roughness, app.normalMode,
                                app.useFaceNormal, app.wireframe);

            if (!options.screenshotPath.empty() && !wroteScreenshot) {
                if (!app.renderer.saveFramebufferPpm(width, height,
                                                      options.screenshotPath, error)) {
                    std::cerr << "[fatal] screenshot failed: " << error << '\n';
                    runtimeError = true;
                    break;
                }
                std::cout << "[capture] wrote " << options.screenshotPath << " ("
                          << width << 'x' << height << ")\n";
                wroteScreenshot = true;
            }

            if (!app.renderer.checkGlError("render frame", error)) {
                std::cerr << "[fatal] OpenGL error during render: " << error << '\n';
                runtimeError = true;
                break;
            }
        }

        glfwSwapBuffers(window);
        ++renderedFrames;

        if (stressRemaining > 0) {
            int nextLevel = app.level + stressDirection;
            if (nextLevel > app.scene.maxSubdivision) {
                stressDirection = -1;
                nextLevel = app.level + stressDirection;
            } else if (nextLevel < 0) {
                stressDirection = 1;
                nextLevel = app.level + stressDirection;
            }
            if (nextLevel == app.level) {
                stressDirection = -stressDirection;
                nextLevel = app.level + stressDirection;
            }
            if (!switchLevel(app, nextLevel, options)) {
                runtimeError = true;
                break;
            }
            updateWindowTitle(window, app);
            --stressRemaining;
        }

        const int totalFrames =
            options.stressSwitches > 0
                ? options.stressSwitches + 2
                : options.frames;
        if (totalFrames >= 0 && renderedFrames >= totalFrames) {
            break;
        }
    }

    std::cout << "[exit] frames=" << renderedFrames
              << " final_level=" << app.level
              << " rss_mib=" << std::fixed << std::setprecision(1)
              << processMemoryMiB() << '\n';

    app.renderer.shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return runtimeError ? 7 : 0;
}

