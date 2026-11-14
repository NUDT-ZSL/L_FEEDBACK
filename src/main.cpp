#include "config.h"
#include "error.h"
#include "offline.h"
#include "renderer_cpu.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#ifdef DEFERRED_HAS_GPU
#include "gpu_renderer.h"
#endif

#ifndef DF_DEFAULT_CONFIG
#define DF_DEFAULT_CONFIG "config/render.json"
#endif

namespace {

struct Options {
    std::string configPath = DF_DEFAULT_CONFIG;
    std::string outputDirectory = "artifacts/offline";
    std::string backend = "auto";
    bool offline = false;
    bool injectGbufferFailure = false;
    bool injectShaderFailure = false;
    bool help = false;
};

void printUsage() {
    std::cout
        << "deferred_fog --config <json> --offline [--output <dir>] [--backend auto|cpu|opengl]\n"
        << "deferred_fog --config <json> --interactive [--backend opengl]\n"
        << "\n"
        << "Offline mode fixes volumetric integration to 32 steps and writes color.png,\n"
        << "depth.pfm, transparent_objects.json and frame_metadata.json.\n";
}

Options parseArgs(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw df::AppError(df::ErrorCode::E_ARGS,
                                   std::string("missing value after ") + name);
            }
            return argv[++i];
        };
        if (arg == "--config") {
            options.configPath = next("--config");
        } else if (arg == "--output") {
            options.outputDirectory = next("--output");
        } else if (arg == "--backend") {
            options.backend = next("--backend");
            if (options.backend != "auto" && options.backend != "cpu" &&
                options.backend != "opengl") {
                throw df::AppError(df::ErrorCode::E_ARGS,
                                   "backend must be auto, cpu or opengl");
            }
        } else if (arg == "--offline") {
            options.offline = true;
        } else if (arg == "--interactive") {
            options.offline = false;
        } else if (arg == "--inject-gbuffer-failure") {
            options.injectGbufferFailure = true;
        } else if (arg == "--inject-shader-failure") {
            options.injectShaderFailure = true;
        } else if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else {
            throw df::AppError(df::ErrorCode::E_ARGS, "unknown argument: " + arg);
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseArgs(argc, argv);
        if (options.help) {
            printUsage();
            return 0;
        }

        const df::RenderConfig config = df::loadConfig(options.configPath);

        if (options.injectShaderFailure) {
            throw df::AppError(df::ErrorCode::E_SHADER_COMPILE,
                               "injected shader compilation failure");
        }
        if (options.injectGbufferFailure) {
            throw df::AppError(df::ErrorCode::E_GBUFFER_FAIL,
                               "injected G-Buffer framebuffer failure");
        }

        if (options.offline) {
            bool useGpu = false;
#ifdef DEFERRED_HAS_GPU
            useGpu = options.backend == "opengl" || options.backend == "auto";
#endif
            if (options.backend == "opengl" && !useGpu) {
                throw df::AppError(df::ErrorCode::E_GL_CONTEXT,
                                   "OpenGL backend was not built (GLFW/GLAD unavailable)");
            }
            if (useGpu) {
#ifdef DEFERRED_HAS_GPU
                const df::OfflineArtifacts artifacts =
                    df::renderOfflineGpu(config, options.outputDirectory);
                std::cout << "backend=" << artifacts.frame.stats.backend
                          << " color=" << artifacts.colorPng
                          << " depth=" << artifacts.depthPfm
                          << " transparent=" << artifacts.transparentJson
                          << " lights_version=" << artifacts.frame.stats.lightVersion
                          << " fog_steps=" << artifacts.frame.stats.fogSteps
                          << " gpu_renderer=\"" << artifacts.frame.stats.gpuRenderer << "\"\n";
#endif
            } else {
                const df::OfflineArtifacts artifacts =
                    df::renderOffline(config, options.outputDirectory);
                std::cout << "backend=" << artifacts.frame.stats.backend
                          << " color=" << artifacts.colorPng
                          << " depth=" << artifacts.depthPfm
                          << " transparent=" << artifacts.transparentJson
                          << " lights_version=" << artifacts.frame.stats.lightVersion
                          << " fog_steps=" << artifacts.frame.stats.fogSteps
                          << " color_hash=" << artifacts.colorHash
                          << " depth_hash=" << artifacts.depthHash << '\n';
            }
            return 0;
        }

        if (options.backend == "cpu") {
            throw df::AppError(df::ErrorCode::E_GL_CONTEXT,
                               "interactive mode requires the OpenGL backend");
        }
#ifdef DEFERRED_HAS_GPU
        return df::runInteractiveGpu(config, options.configPath);
#else
        throw df::AppError(df::ErrorCode::E_GL_CONTEXT,
                           "OpenGL backend was not built; run with --offline");
#endif
    } catch (const df::AppError& error) {
        std::cerr << df::errorCodeName(error.code()) << ": " << error.what() << '\n';
        return df::exitCode(error.code());
    } catch (const std::exception& error) {
        std::cerr << "E_RENDER_FAIL: " << error.what() << '\n';
        return df::exitCode(df::ErrorCode::E_RENDER_FAIL);
    }
}
