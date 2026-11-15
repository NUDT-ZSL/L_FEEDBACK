#include "context.hpp"
#include "error.hpp"
#include "pipeline.hpp"
#include "renderer.hpp"
#include "scene.hpp"

#include <nlohmann/json.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

struct Options {
    uint64_t seed = 0;
    int width = 960;
    int height = 540;
    int warmupFrames = 2;
    std::string pipelinePath;
    std::string outputPath;
    std::string shaderDirectory;
    std::string saveStatePath;
    std::string stateRoundtripPath;
    std::string reconfigurePipelinePath;
    std::string reconfigureOutputPath;
    bool noLights = false;
    bool help = false;
};

[[noreturn]] void cliError(const std::string& message) {
    throw scene::AppError(scene::ErrorType::Cli, message);
}

uint64_t parseSeed(const std::string& value) {
    try {
        size_t consumed = 0;
        const uint64_t result = std::stoull(value, &consumed, 10);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    } catch (const std::exception&) {
        cliError("--seed must be an unsigned 64-bit integer, got '" + value + "'");
    }
}

int parseInt(const std::string& option, const std::string& value,
             int minimum, int maximum) {
    try {
        size_t consumed = 0;
        const int result = std::stoi(value, &consumed, 10);
        if (consumed != value.size() || result < minimum || result > maximum) {
            throw std::invalid_argument("out of range");
        }
        return result;
    } catch (const std::exception&) {
        cliError(option + " must be an integer in [" + std::to_string(minimum) +
                 ", " + std::to_string(maximum) + "], got '" + value + "'");
    }
}

Options parseOptions(int argc, char** argv) {
    Options options;
    bool seedSeen = false;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto nextValue = [&](const std::string& option) -> std::string {
            if (i + 1 >= argc) {
                cliError(option + " requires a value");
            }
            return argv[++i];
        };

        if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else if (argument == "--seed") {
            options.seed = parseSeed(nextValue(argument));
            seedSeen = true;
        } else if (argument == "--pipeline") {
            options.pipelinePath = nextValue(argument);
        } else if (argument == "--output") {
            options.outputPath = nextValue(argument);
        } else if (argument == "--width") {
            options.width = parseInt(argument, nextValue(argument), 16, 8192);
        } else if (argument == "--height") {
            options.height = parseInt(argument, nextValue(argument), 16, 8192);
        } else if (argument == "--warmup") {
            options.warmupFrames = parseInt(argument, nextValue(argument), 0, 64);
        } else if (argument == "--shaders") {
            options.shaderDirectory = nextValue(argument);
        } else if (argument == "--save-state") {
            options.saveStatePath = nextValue(argument);
        } else if (argument == "--state-roundtrip") {
            options.stateRoundtripPath = nextValue(argument);
        } else if (argument == "--reconfigure") {
            options.reconfigurePipelinePath = nextValue(argument);
        } else if (argument == "--reconfigure-output") {
            options.reconfigureOutputPath = nextValue(argument);
        } else if (argument == "--no-lights") {
            options.noLights = true;
        } else {
            cliError("unknown command-line option '" + argument + "'");
        }
    }

    if (options.help) {
        return options;
    }
    if (!seedSeen) {
        cliError("--seed is required");
    }
    if (options.pipelinePath.empty()) {
        cliError("--pipeline is required");
    }
    if (options.outputPath.empty()) {
        cliError("--output is required");
    }
    if (!options.reconfigurePipelinePath.empty() &&
        options.reconfigureOutputPath.empty()) {
        cliError("--reconfigure requires --reconfigure-output");
    }
    if (options.reconfigureOutputPath.empty() !=
        options.reconfigurePipelinePath.empty()) {
        cliError("--reconfigure-output may only be used with --reconfigure");
    }
    return options;
}

void printUsage() {
    std::cout
        << "Usage: offscreen_renderer --seed N --pipeline FILE --output FILE [options]\n"
        << "\n"
        << "Required:\n"
        << "  --seed N                 Unsigned 64-bit deterministic scene seed\n"
        << "  --pipeline FILE          Pipeline JSON configuration\n"
        << "  --output FILE            Final PNG path\n"
        << "\n"
        << "Options:\n"
        << "  --width N                Output width (default 960)\n"
        << "  --height N               Output height (default 540)\n"
        << "  --warmup N               Warmup frames before measurement (default 2)\n"
        << "  --shaders DIR            Directory containing GLSL shader files\n"
        << "  --save-state FILE        Serialize the active pipeline after rendering\n"
        << "  --state-roundtrip FILE   Save, reload, re-render, and require identical pixels\n"
        << "  --reconfigure FILE       Apply another pipeline in the same process\n"
        << "  --reconfigure-output F   PNG path for the re-rendered pipeline\n"
        << "  --no-lights              Generate the scene without lights (ambient test)\n"
        << "  --help                   Show this help\n";
}

json timingsJson(const std::vector<scene::TimingEntry>& timings) {
    json result = json::array();
    for (const auto& timing : timings) {
        result.push_back({
            {"name", timing.name},
            {"enabled", timing.enabled},
            {"ms", timing.milliseconds}
        });
    }
    return result;
}

json renderJson(const scene::RenderResult& result) {
    return {
        {"width", result.width},
        {"height", result.height},
        {"total_render_ms", result.totalMilliseconds},
        {"prepare_ms", result.prepareMilliseconds},
        {"present_ms", result.presentMilliseconds},
        {"readback_ms", result.readbackMilliseconds},
        {"stages_ms", timingsJson(result.stageTimings)},
        {"post_effects_ms", timingsJson(result.effectTimings)}
    };
}

void writePng(const std::string& path, int width, int height,
              const std::vector<uint8_t>& pixels) {
    const std::filesystem::path target(path);
    std::error_code ec;
    if (!target.parent_path().empty()) {
        std::filesystem::create_directories(target.parent_path(), ec);
        if (ec) {
            throw scene::AppError(scene::ErrorType::FileIo,
                                  "cannot create PNG directory '" +
                                  target.parent_path().string() + "': " +
                                  ec.message());
        }
    }

    stbi_flip_vertically_on_write(1);
    const int written = stbi_write_png(path.c_str(), width, height, 4,
                                       pixels.data(), width * 4);
    stbi_flip_vertically_on_write(0);
    if (written == 0) {
        throw scene::AppError(scene::ErrorType::FileIo,
                              "failed to write PNG '" + path + "'");
    }
}

void requireSamePixels(const std::vector<uint8_t>& expected,
                       const std::vector<uint8_t>& actual) {
    if (expected.size() != actual.size()) {
        throw scene::AppError(scene::ErrorType::StateMismatch,
                              "state roundtrip image size changed");
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            throw scene::AppError(
                scene::ErrorType::StateMismatch,
                "state roundtrip pixels differ at byte " + std::to_string(i) +
                " (expected " + std::to_string(expected[i]) +
                ", got " + std::to_string(actual[i]) + ")");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.help) {
            printUsage();
            return 0;
        }

        scene::PipelineConfig initialConfig =
            scene::PipelineConfig::load(options.pipelinePath);
        std::optional<scene::PipelineConfig> reconfigureConfig;
        if (!options.reconfigurePipelinePath.empty()) {
            reconfigureConfig =
                scene::PipelineConfig::load(options.reconfigurePipelinePath);
        }

        std::vector<uint8_t> primaryPixels;
        std::vector<uint8_t> reconfiguredPixels;
        std::vector<uint8_t> restoredPixels;
        scene::RenderResult primaryResult;
        std::optional<scene::RenderResult> reconfiguredResult;
        std::optional<scene::RenderResult> restoredResult;
        std::string glVersion;
        std::string glRenderer;

        {
            scene::OffscreenContext context;
            scene::Scene generatedScene(options.seed, !options.noLights);
            scene::Renderer renderer(generatedScene, options.width, options.height,
                                     options.shaderDirectory);

            renderer.applyPipeline(initialConfig);
            for (int i = 0; i < options.warmupFrames; ++i) {
                renderer.renderFrame(false);
            }
            primaryResult = renderer.renderFrame(true);
            primaryPixels = primaryResult.pixels;

            if (!options.saveStatePath.empty()) {
                renderer.pipeline().save(options.saveStatePath);
            }

            if (reconfigureConfig.has_value()) {
                renderer.applyPipeline(*reconfigureConfig);
                for (int i = 0; i < options.warmupFrames; ++i) {
                    renderer.renderFrame(false);
                }
                reconfiguredResult = renderer.renderFrame(true);
                reconfiguredPixels = reconfiguredResult->pixels;
            }

            if (!options.stateRoundtripPath.empty()) {
                renderer.pipeline().save(options.stateRoundtripPath);
                const scene::PipelineConfig restored =
                    scene::PipelineConfig::load(options.stateRoundtripPath);
                renderer.applyPipeline(restored);
                for (int i = 0; i < options.warmupFrames; ++i) {
                    renderer.renderFrame(false);
                }
                restoredResult = renderer.renderFrame(true);
                restoredPixels = restoredResult->pixels;
                requireSamePixels(reconfiguredPixels.empty()
                                      ? primaryPixels : reconfiguredPixels,
                                  restoredPixels);
            }

            glVersion = context.versionString();
            glRenderer = context.rendererName();
        }

        writePng(options.outputPath, options.width, options.height, primaryPixels);
        if (!options.reconfigureOutputPath.empty()) {
            writePng(options.reconfigureOutputPath, options.width, options.height,
                     reconfiguredPixels);
        }
        const std::string restoredOutput =
            (!options.reconfigureOutputPath.empty()
                 ? options.reconfigureOutputPath
                 : options.outputPath) + ".restored.png";
        if (!options.stateRoundtripPath.empty()) {
            writePng(restoredOutput, options.width, options.height, restoredPixels);
        }

        json summary = {
            {"status", "ok"},
            {"seed", options.seed},
            {"pipeline", options.pipelinePath},
            {"output", options.outputPath},
            {"width", options.width},
            {"height", options.height},
            {"warmup_frames", options.warmupFrames},
            {"lights_enabled", !options.noLights},
            {"context", {{"version", glVersion}, {"renderer", glRenderer}}},
            {"primary", renderJson(primaryResult)}
        };
        if (reconfiguredResult.has_value()) {
            summary["reconfigure"] = {
                {"pipeline", options.reconfigurePipelinePath},
                {"output", options.reconfigureOutputPath},
                {"render", renderJson(*reconfiguredResult)}
            };
        }
        if (restoredResult.has_value()) {
            summary["state_roundtrip"] = {
                {"path", options.stateRoundtripPath},
                {"restored_output", restoredOutput},
                {"identical", true},
                {"render", renderJson(*restoredResult)}
            };
        }
        std::cout << summary.dump(2) << '\n';
        return 0;
    } catch (const scene::AppError& error) {
        const json errorJson = {
            {"type", scene::errorTypeName(error.type())},
            {"message", error.what()},
            {"exit_code", scene::errorExitCode(error.type())}
        };
        std::cerr << errorJson.dump() << '\n';
        return scene::errorExitCode(error.type());
    } catch (const std::exception& error) {
        const json errorJson = {
            {"type", "internal_error"},
            {"message", error.what()},
            {"exit_code", scene::errorExitCode(scene::ErrorType::Internal)}
        };
        std::cerr << errorJson.dump() << '\n';
        return scene::errorExitCode(scene::ErrorType::Internal);
    }
}

