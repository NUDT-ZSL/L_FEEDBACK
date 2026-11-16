#include "json.hpp"
#include "pipeline.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using tilegen::JsonError;
using tilegen::JsonParser;
using tilegen::JsonValue;
using tilegen::PipelineOptions;

struct Settings {
    std::optional<std::string> scene;
    std::optional<std::string> output;
    std::optional<int> tileSize;
    std::optional<int> levels;
    std::optional<int> previewLevel;
    std::optional<int> previewWidth;
    std::optional<int> previewHeight;
    std::optional<int> samples;
    std::optional<bool> resume;
    std::optional<int> simulateDiskFullLevel;
};

int parseInteger(const std::string& text, const std::string& option) {
    int value = 0;
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc() || result.ptr != end) {
        throw std::runtime_error(option + " requires an integer, got '" + text + "'");
    }
    return value;
}

bool parseBoolean(const std::string& text, const std::string& option) {
    if (text == "true" || text == "1" || text == "yes" || text == "on") {
        return true;
    }
    if (text == "false" || text == "0" || text == "no" || text == "off") {
        return false;
    }
    throw std::runtime_error(option + " requires true or false, got '" + text + "'");
}

std::string requireString(const JsonValue& value, const std::string& path) {
    if (!value.isString()) {
        throw std::runtime_error(path + " must be a string");
    }
    return value.asString();
}

int requireInteger(const JsonValue& value, const std::string& path) {
    if (!value.isNumber() || !std::isfinite(value.asNumber()) ||
        value.asNumber() < static_cast<double>(std::numeric_limits<int>::min()) ||
        value.asNumber() > static_cast<double>(std::numeric_limits<int>::max()) ||
        std::floor(value.asNumber()) != value.asNumber()) {
        throw std::runtime_error(path + " must be an integer");
    }
    return static_cast<int>(value.asNumber());
}

std::string jsonPath(const std::string& base, const std::string& key) {
    return base + "." + key;
}

void applyOption(Settings& settings, const std::string& key,
                 const JsonValue& value, const std::string& path) {
    if (key == "scene" || key == "scene_path" || key == "scene_file") {
        if (value.isObject()) {
            const JsonValue* nested = value.find("path");
            if (nested == nullptr) {
                throw std::runtime_error(jsonPath(path, "scene") +
                                         ".path is required");
            }
            settings.scene = requireString(*nested, jsonPath(path, "scene.path"));
        } else {
            settings.scene = requireString(value, path);
        }
    } else if (key == "output" || key == "output_dir" || key == "out") {
        if (value.isObject()) {
            const JsonValue* nested = value.find("path");
            if (nested == nullptr) {
                throw std::runtime_error(jsonPath(path, "output") +
                                         ".path is required");
            }
            settings.output =
                requireString(*nested, jsonPath(path, "output.path"));
        } else {
            settings.output = requireString(value, path);
        }
    } else if (key == "tile_size" || key == "tileSize") {
        settings.tileSize = requireInteger(value, path);
    } else if (key == "levels" || key == "level_count") {
        settings.levels = requireInteger(value, path);
    } else if (key == "preview_level" || key == "previewLevel") {
        settings.previewLevel = requireInteger(value, path);
    } else if (key == "preview_width" || key == "previewWidth") {
        settings.previewWidth = requireInteger(value, path);
    } else if (key == "preview_height" || key == "previewHeight") {
        settings.previewHeight = requireInteger(value, path);
    } else if (key == "samples" || key == "samples_per_axis" ||
               key == "supersampling") {
        settings.samples = requireInteger(value, path);
    } else if (key == "resume") {
        if (!value.isBool()) {
            throw std::runtime_error(path + " must be a boolean");
        }
        settings.resume = value.asBool();
    } else if (key == "simulate_disk_full_level" ||
               key == "fail_at_level") {
        settings.simulateDiskFullLevel = requireInteger(value, path);
    }
}

Settings loadConfig(const std::string& path) {
    JsonParser parser;
    JsonValue root = parser.parseFile(path);
    if (!root.isObject()) {
        throw std::runtime_error("config root must be an object");
    }
    Settings settings;
    const JsonValue* object = &root;
    if (const JsonValue* pipeline = root.find("pipeline");
        pipeline != nullptr && pipeline->isObject()) {
        object = pipeline;
    }
    for (const auto& entry : object->asObject()) {
        applyOption(settings, entry.first, entry.second, "$." + entry.first);
    }
    return settings;
}

bool isValueOption(const std::string& option) {
    return option == "--scene" || option == "--scene-file" ||
           option == "--output" || option == "--out" ||
           option == "--tile-size" || option == "--levels" ||
           option == "--preview-level" || option == "--preview-width" ||
           option == "--preview-height" || option == "--samples" ||
           option == "--config" || option == "--simulate-disk-full-level";
}

bool isBooleanOption(const std::string& option) {
    return option == "--resume";
}

struct CliArgument {
    std::string name;
    std::optional<std::string> value;
};

std::vector<CliArgument> tokenizeArguments(int argc, char** argv,
                                           std::vector<std::string>& positional) {
    std::vector<CliArgument> arguments;
    for (int i = 1; i < argc; ++i) {
        std::string token = argv[i];
        if (token == "-h") {
            arguments.push_back(CliArgument{"--help", std::nullopt});
            continue;
        }
        if (token == "--") {
            for (++i; i < argc; ++i) {
                positional.push_back(argv[i]);
            }
            break;
        }
        if (token.size() > 2 && token[0] == '-' && token[1] == '-') {
            CliArgument argument;
            const std::size_t equals = token.find('=');
            if (equals != std::string::npos) {
                argument.name = token.substr(0, equals);
                argument.value = token.substr(equals + 1);
            } else {
                argument.name = token;
            }
            if (isValueOption(argument.name) && !argument.value) {
                if (i + 1 >= argc) {
                    throw std::runtime_error(argument.name + " requires a value");
                }
                argument.value = argv[++i];
            }
            if (isBooleanOption(argument.name) && !argument.value) {
                argument.value = "true";
            }
            arguments.push_back(std::move(argument));
        } else if (!token.empty() && token[0] == '-') {
            throw std::runtime_error("unknown short option: " + token);
        } else {
            positional.push_back(token);
        }
    }
    return arguments;
}

void applyCli(Settings& settings, const CliArgument& argument) {
    if (argument.name == "--scene" || argument.name == "--scene-file") {
        settings.scene = *argument.value;
    } else if (argument.name == "--output" || argument.name == "--out") {
        settings.output = *argument.value;
    } else if (argument.name == "--tile-size") {
        settings.tileSize = parseInteger(*argument.value, argument.name);
    } else if (argument.name == "--levels") {
        settings.levels = parseInteger(*argument.value, argument.name);
    } else if (argument.name == "--preview-level") {
        settings.previewLevel = parseInteger(*argument.value, argument.name);
    } else if (argument.name == "--preview-width") {
        settings.previewWidth = parseInteger(*argument.value, argument.name);
    } else if (argument.name == "--preview-height") {
        settings.previewHeight = parseInteger(*argument.value, argument.name);
    } else if (argument.name == "--samples") {
        settings.samples = parseInteger(*argument.value, argument.name);
    } else if (argument.name == "--simulate-disk-full-level") {
        settings.simulateDiskFullLevel =
            parseInteger(*argument.value, argument.name);
    } else if (argument.name == "--resume") {
        settings.resume = parseBoolean(*argument.value, argument.name);
    }
}

void printHelp() {
    std::cout
        << "tilegen - deterministic offline vector tile pyramid generator\n\n"
        << "Usage:\n"
        << "  tilegen --scene scene.json --output out [options]\n"
        << "  tilegen --config pipeline.json [options]\n\n"
        << "Options:\n"
        << "  --scene PATH                  Scene JSON (required unless configured)\n"
        << "  --output DIR                  Output directory\n"
        << "  --tile-size N                 Tile edge in pixels (default 256)\n"
        << "  --levels N                    Pyramid level count (default 3)\n"
        << "  --preview-level N             Preview source level (default middle)\n"
        << "  --preview-width N             Preview PNG width (default 512)\n"
        << "  --preview-height N            Preview PNG height (default 256)\n"
        << "  --samples N                   Supersampling grid per axis (default 4)\n"
        << "  --config PATH                 Load defaults from a JSON config\n"
        << "  --resume                      Reuse manifest-valid tiles and continue\n"
        << "  --simulate-disk-full-level N  Return ENOSPC from level N onward\n"
        << "  --help                        Show this help\n"
        << "  --version                     Print version\n\n"
        << "Precedence: built-in defaults < config file < explicit CLI options.\n"
        << "Level 0 is the full scene resolution; each next level halves scale.\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> positional;
        const std::vector<CliArgument> arguments =
            tokenizeArguments(argc, argv, positional);

        for (const CliArgument& argument : arguments) {
            if (argument.name == "--help") {
                printHelp();
                return 0;
            }
            if (argument.name == "--version") {
                std::cout << "tilegen 1.0.0\n";
                return 0;
            }
        }

        Settings settings;
        for (const CliArgument& argument : arguments) {
            if (argument.name == "--config") {
                Settings configured = loadConfig(*argument.value);
                auto merge = [](auto& target, const auto& source) {
                    if (source) {
                        target = source;
                    }
                };
                merge(settings.scene, configured.scene);
                merge(settings.output, configured.output);
                merge(settings.tileSize, configured.tileSize);
                merge(settings.levels, configured.levels);
                merge(settings.previewLevel, configured.previewLevel);
                merge(settings.previewWidth, configured.previewWidth);
                merge(settings.previewHeight, configured.previewHeight);
                merge(settings.samples, configured.samples);
                merge(settings.resume, configured.resume);
                merge(settings.simulateDiskFullLevel,
                      configured.simulateDiskFullLevel);
            }
        }
        for (const CliArgument& argument : arguments) {
            if (argument.name != "--config") {
                applyCli(settings, argument);
            }
        }

        std::size_t positionalIndex = 0;
        if (!settings.scene && positionalIndex < positional.size()) {
            settings.scene = positional[positionalIndex++];
        }
        if (!settings.output && positionalIndex < positional.size()) {
            settings.output = positional[positionalIndex++];
        }
        if (positionalIndex < positional.size()) {
            throw std::runtime_error("unexpected positional argument: " +
                                     positional[positionalIndex]);
        }

        PipelineOptions options;
        options.scenePath = settings.scene.value_or("");
        options.outputDirectory = settings.output.value_or("");
        if (settings.tileSize) {
            options.tileSize = *settings.tileSize;
        }
        if (settings.levels) {
            options.levels = *settings.levels;
        }
        options.previewLevel = settings.previewLevel.value_or(
            (options.levels - 1) / 2);
        if (settings.previewWidth) {
            options.previewWidth = *settings.previewWidth;
        }
        if (settings.previewHeight) {
            options.previewHeight = *settings.previewHeight;
        }
        if (settings.samples) {
            options.samplesPerAxis = *settings.samples;
        }
        if (settings.resume) {
            options.resume = *settings.resume;
        }
        if (settings.simulateDiskFullLevel) {
            options.simulateDiskFullLevel = *settings.simulateDiskFullLevel;
        } else if (const char* environment =
                       std::getenv("TILEGEN_SIMULATE_DISK_FULL_LEVEL")) {
            options.simulateDiskFullLevel =
                parseInteger(environment, "TILEGEN_SIMULATE_DISK_FULL_LEVEL");
        }

        if (options.scenePath.empty() || options.outputDirectory.empty()) {
            printHelp();
            std::cerr << "\n[ERROR] --scene and --output are required\n";
            return 1;
        }
        if (options.levels <= 0) {
            throw std::runtime_error("levels must be greater than zero");
        }
        if (options.previewLevel < 0 || options.previewLevel >= options.levels) {
            throw std::runtime_error("preview level must be in [0, levels-1]");
        }
        if (options.tileSize <= 0 || options.previewWidth <= 0 ||
            options.previewHeight <= 0 || options.samplesPerAxis < 1 ||
            options.samplesPerAxis > 16) {
            throw std::runtime_error("tile size, preview dimensions, and samples "
                                     "must be positive; samples must be <= 16");
        }
        return tilegen::runPipeline(options);
    } catch (const JsonError& error) {
        std::cerr << "[ERROR] [config] " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] [config] " << error.what() << '\n';
        return 1;
    }
}
