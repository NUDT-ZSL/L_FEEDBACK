#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/gl.h>

#include "renderer/error.hpp"
#include "renderer/logger.hpp"
#include "renderer/renderer.hpp"
#include "renderer/scene.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace renderer {
namespace {

enum class Action {
    Render,
    Validate,
    GenerateSamples,
    Help,
};

struct Options {
    Action action = Action::Render;
    std::string scenePath;
    std::string mode;
    std::string logPath;
    std::string outputPath;
    bool validateOnly = false;
    bool selfTestResources = false;
};

std::string lastGlfwError;

void glfw_error_callback(int errorCode, const char* description) {
    lastGlfwError = "GLFW error " + std::to_string(errorCode) + ": "
                  + (description == nullptr ? std::string("unknown") : std::string(description));
}

std::string require_value(const std::vector<std::string>& arguments,
                          std::size_t& index, const std::string& option) {
    if (index + 1 >= arguments.size()) {
        fail("E_USAGE", option + " requires a value");
    }
    ++index;
    return arguments[index];
}

Options parse_options(const std::vector<std::string>& arguments) {
    Options options;
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        if (argument == "--help" || argument == "-h") {
            options.action = Action::Help;
        } else if (argument == "--validate") {
            options.validateOnly = true;
        } else if (argument == "--self-test-resources") {
            options.selfTestResources = true;
        } else if (argument == "--generate-samples") {
            options.action = Action::GenerateSamples;
            options.scenePath = require_value(arguments, index, argument);
        } else if (argument == "--scene") {
            options.scenePath = require_value(arguments, index, argument);
        } else if (argument == "--mode") {
            options.mode = require_value(arguments, index, argument);
        } else if (argument == "--log") {
            options.logPath = require_value(arguments, index, argument);
        } else if (argument == "--output") {
            options.outputPath = require_value(arguments, index, argument);
        } else {
            fail("E_USAGE", "unknown command-line option: " + argument);
        }
    }

    if (options.action == Action::Help) {
        return options;
    }
    if (options.action == Action::GenerateSamples) {
        if (options.scenePath.empty()) {
            fail("E_USAGE", "--generate-samples requires an output directory");
        }
        return options;
    }
    if (options.scenePath.empty()) {
        fail("E_USAGE", "--scene is required");
    }
    if (options.validateOnly) {
        options.action = Action::Validate;
        return options;
    }
    if (options.mode.empty()) {
        fail("E_USAGE", "--mode fast|high is required");
    }
    if (!is_valid_mode(options.mode)) {
        fail("E_USAGE", "--mode must be fast or high");
    }
    if (options.logPath.empty()) {
        fail("E_USAGE", "--log is required");
    }
    return options;
}

void print_help() {
    std::cout
        << "Offline multi-light HDR preview renderer\n\n"
        << "Usage:\n"
        << "  render --scene <scene.json> --mode <fast|high> --log <output.log>\n"
        << "         [--output <frame.png>] [--self-test-resources]\n"
        << "  render --scene <scene.json> --validate [--log <output.log>]\n"
        << "  render --generate-samples <directory>\n"
        << "  render --help\n";
}

std::string resolve_output_path(const Scene& scene, const Options& options) {
    if (!options.outputPath.empty()) {
        return options.outputPath;
    }
    std::filesystem::path output(scene.settings.output);
    if (output.is_relative()) {
        const std::filesystem::path scenePath(scene.sourcePath);
        if (scenePath.has_parent_path()) {
            output = scenePath.parent_path() / output;
        }
    }
    return output.string();
}

class GlfwRuntime {
public:
    GlfwRuntime() {
        lastGlfwError.clear();
        glfwSetErrorCallback(glfw_error_callback);
        if (glfwInit() != GLFW_TRUE) {
            fail("E_GL_CONTEXT", "glfwInit failed: " + lastGlfwError);
        }
        initialized_ = true;

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
        window_ = glfwCreateWindow(64, 64, "Offline HDR Preview Renderer",
                                   nullptr, nullptr);
        if (window_ == nullptr) {
            glfwTerminate();
            initialized_ = false;
            fail("E_GL_CONTEXT", "failed to create an OpenGL 4.5 window: "
                                     + lastGlfwError);
        }
        glfwMakeContextCurrent(window_);
        glfwSwapInterval(0);
    }

    ~GlfwRuntime() {
        if (window_ != nullptr) {
            glfwDestroyWindow(window_);
            window_ = nullptr;
        }
        if (initialized_) {
            glfwTerminate();
            initialized_ = false;
        }
    }

    GlfwRuntime(const GlfwRuntime&) = delete;
    GlfwRuntime& operator=(const GlfwRuntime&) = delete;

    GLFWwindow* window() const {
        return window_;
    }

private:
    GLFWwindow* window_ = nullptr;
    bool initialized_ = false;
};

void load_gl() {
    const int version = gladLoadGL(
        reinterpret_cast<GLADloadfunc>(glfwGetProcAddress));
    if (version == 0 || !GLAD_GL_VERSION_4_5) {
        fail("E_GL_CONTEXT", "failed to load OpenGL 4.5 core functions");
    }
}

const char* gl_string(GLenum name) {
    const GLubyte* value = glGetString(name);
    return value == nullptr ? "unknown" : reinterpret_cast<const char*>(value);
}

void log_context(Logger& logger) {
    logger.info("gl_context", {
        log_field("vendor", gl_string(GL_VENDOR)),
        log_field("renderer", gl_string(GL_RENDERER)),
        log_field("version", gl_string(GL_VERSION)),
        log_field("glsl", gl_string(GL_SHADING_LANGUAGE_VERSION)),
    });
}

void ensure_resources_released(Renderer& renderer, Logger& logger,
                               const std::string& stage) {
    const ResourceCounts counts = renderer.resource_counts();
    logger.info("resource_balance", {
        log_field("stage", stage),
        log_field("shader_programs", counts.shaderPrograms),
        log_field("textures", counts.textures),
        log_field("framebuffers", counts.framebuffers),
        log_field("renderbuffers", counts.renderbuffers),
        log_field("buffers", counts.buffers),
        log_field("vertex_arrays", counts.vertexArrays),
        log_field("queries", counts.queries),
        log_field("total", counts.total()),
    });
    if (counts.total() != 0) {
        fail("E_RESOURCE_LEAK", "OpenGL resource counters are not zero after shutdown");
    }
}

void run_self_test(const Scene& scene, const Options& options, Logger& logger) {
    Renderer renderer;
    logger.info("resource_self_test_start", {log_field("mode", options.mode)});
    renderer.initialize(scene, options.mode, logger);
    renderer.initialize(scene, options.mode, logger);
    const int firstCount = renderer.resource_counts().total();
    renderer.shutdown();
    ensure_resources_released(renderer, logger, "first_shutdown");

    renderer.initialize(scene, options.mode, logger);
    const int secondCount = renderer.resource_counts().total();
    renderer.shutdown();
    ensure_resources_released(renderer, logger, "second_shutdown");

    logger.info("resource_self_test", {
        log_field("first_initialization_objects", firstCount),
        log_field("second_initialization_objects", secondCount),
        log_field("resource_balance", 0),
    });
}

void run_render(const Scene& scene, const Options& options, Logger& logger) {
    const std::string outputPath = resolve_output_path(scene, options);
    Renderer renderer;
    renderer.initialize(scene, options.mode, logger);
    const RenderStats stats = renderer.render(scene, options.mode, outputPath, logger);
    renderer.shutdown();
    ensure_resources_released(renderer, logger, "render_shutdown");

    std::cout << "output=" << outputPath
              << " mode=" << options.mode
              << " non_background_percent=" << stats.nonBackgroundPercent
              << " geometry_gpu_ms=" << stats.geometryGpuMs
              << " lighting_gpu_ms=" << stats.lightingGpuMs << '\n';
}

}  // namespace
}  // namespace renderer

int main(int argc, char** argv) {
    using namespace renderer;

    Logger logger;
    try {
        std::vector<std::string> arguments;
        arguments.reserve(static_cast<std::size_t>(argc));
        for (int index = 0; index < argc; ++index) {
            arguments.emplace_back(argv[index]);
        }

        const Options options = parse_options(arguments);
        if (options.action == Action::Help) {
            print_help();
            return 0;
        }
        if (options.action == Action::GenerateSamples) {
            generate_samples(options.scenePath);
            std::cout << "samples=" << options.scenePath << '\n';
            return 0;
        }

        if (!options.logPath.empty()) {
            logger.open(options.logPath);
            logger.info("startup", {
                log_field("scene", options.scenePath),
                log_field("mode", options.mode.empty() ? "validate" : options.mode),
            });
        }

        Scene scene = load_scene(options.scenePath);
        if (options.action == Action::Validate) {
            logger.info("scene_validation", {
                log_field("result", "valid"),
                log_field("materials", static_cast<int>(scene.materials.size())),
                log_field("meshes", static_cast<int>(scene.meshes.size())),
                log_field("lights", static_cast<int>(scene.lights.size())),
                log_field("shadow_tiles_required", shadow_tile_demand(scene)),
                log_field("shadow_tiles_available", shadow_tile_capacity(scene)),
            });
            std::cout << "scene=valid materials=" << scene.materials.size()
                      << " meshes=" << scene.meshes.size()
                      << " lights=" << scene.lights.size()
                      << " shadow_tiles=" << shadow_tile_demand(scene)
                      << "/" << shadow_tile_capacity(scene) << '\n';
            return 0;
        }

        {
            GlfwRuntime runtime;
            (void)runtime.window();
            load_gl();
            log_context(logger);
            if (options.selfTestResources) {
                run_self_test(scene, options, logger);
            } else {
                run_render(scene, options, logger);
            }
        }
        logger.flush();
        return 0;
    } catch (const AppError& error) {
        if (logger.isOpen()) {
            logger.error(error.code(), "fatal", {
                log_field("message", error.what()),
            });
        } else {
            write_stderr_error(error.code(), error.what());
        }
        return 1;
    } catch (const std::exception& error) {
        if (logger.isOpen()) {
            logger.error("E_INTERNAL", "fatal", {
                log_field("message", error.what()),
            });
        } else {
            write_stderr_error("E_INTERNAL", error.what());
        }
        return 1;
    }
}
