#include "cli.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <sstream>
#include <vector>

namespace dr {
namespace {

std::string trim(const std::string& value) {
    const auto begin = std::find_if_not(value.begin(), value.end(),
                                        [](unsigned char character) { return std::isspace(character); });
    const auto end = std::find_if_not(value.rbegin(), value.rend(),
                                      [](unsigned char character) { return std::isspace(character); })
                         .base();
    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
}

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> parts;
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = value.find(delimiter, begin);
        parts.push_back(trim(value.substr(begin, end == std::string::npos ? end : end - begin)));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return parts;
}

bool parseFloat(const std::string& text, float& value) {
    try {
        std::size_t consumed = 0;
        value = std::stof(text, &consumed);
        return consumed == text.size();
    } catch (const std::exception&) {
        return false;
    }
}

bool parseInt(const std::string& text, int& value) {
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    return result.ec == std::errc{} && result.ptr == end;
}

bool parseVec3(const std::string& text, Vec3& value) {
    const std::vector<std::string> parts = split(text, ',');
    return parts.size() == 3 && parseFloat(parts[0], value.x) && parseFloat(parts[1], value.y) &&
           parseFloat(parts[2], value.z);
}

bool consumeValue(int argc, char** argv, int& index, const char* flag, std::string& value,
                  std::string& error) {
    if (index + 1 >= argc) {
        error = std::string("missing value for ") + flag;
        return false;
    }
    value = argv[++index];
    return true;
}

} // namespace

bool parseAddLight(const std::string& specification, Light& light, std::string& error) {
    const std::vector<std::string> fields = split(specification, ':');
    if (fields.size() != 5 && fields.size() != 6) {
        error = "add-light must be id:type:position:color:intensity[:radius]";
        return false;
    }

    light = Light{};
    light.id = fields[0];
    if (light.id.empty()) {
        error = "add-light id must not be empty";
        return false;
    }
    if (fields[1] == "point") {
        light.type = LightType::Point;
        light.radius = 5.0f;
    } else if (fields[1] == "directional") {
        light.type = LightType::Directional;
        light.radius = 0.0f;
    } else {
        error = "add-light type must be point or directional";
        return false;
    }
    if (!parseVec3(fields[2], light.positionOrDirection)) {
        error = "add-light position/direction must be x,y,z";
        return false;
    }
    if (!parseVec3(fields[3], light.color)) {
        error = "add-light color must be r,g,b";
        return false;
    }
    if (!parseFloat(fields[4], light.intensity)) {
        error = "add-light intensity must be a number";
        return false;
    }
    if (fields.size() == 6 && !parseFloat(fields[5], light.radius)) {
        error = "add-light radius must be a number";
        return false;
    }
    if (light.type == LightType::Directional) {
        light.radius = 0.0f;
    }
    if (light.intensity < 0.0f || light.color.x < 0.0f || light.color.y < 0.0f ||
        light.color.z < 0.0f) {
        error = "add-light color and intensity must be non-negative";
        return false;
    }
    if (light.type == LightType::Point && light.radius <= 0.0f) {
        error = "add-light point radius must be positive";
        return false;
    }
    return true;
}

bool parseArguments(int argc, char** argv, Options& options, std::string& error, bool& showHelp) {
    options = Options{};
    showHelp = false;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            showHelp = true;
            return true;
        }
        if (argument == "--render" || argument == "--snapshot-save" ||
            argument == "--snapshot-load") {
            if (options.modeSpecified) {
                error = "only one of --render, --snapshot-save and --snapshot-load may be used";
                return false;
            }
            options.modeSpecified = true;
            if (argument == "--render") {
                options.mode = RunMode::Live;
            } else if (argument == "--snapshot-save") {
                options.mode = RunMode::SnapshotSave;
            } else {
                options.mode = RunMode::SnapshotLoad;
            }
            continue;
        }

        std::string value;
        if (argument == "--scene") {
            if (!consumeValue(argc, argv, index, "--scene", value, error)) {
                return false;
            }
            options.scenePath = value;
        } else if (argument == "--snapshot") {
            if (!consumeValue(argc, argv, index, "--snapshot", value, error)) {
                return false;
            }
            options.snapshotPath = value;
        } else if (argument == "--output") {
            if (!consumeValue(argc, argv, index, "--output", value, error)) {
                return false;
            }
            options.outputPath = value;
        } else if (argument == "--compare") {
            if (!consumeValue(argc, argv, index, "--compare", value, error)) {
                return false;
            }
            options.comparePath = value;
        } else if (argument == "--max-channel-diff") {
            if (!consumeValue(argc, argv, index, "--max-channel-diff", value, error)) {
                return false;
            }
            if (!parseInt(value, options.maxChannelDifference) || options.maxChannelDifference < 0 ||
                options.maxChannelDifference > 255) {
                error = "--max-channel-diff must be an integer in [0,255]";
                return false;
            }
        } else if (argument == "--frames") {
            if (!consumeValue(argc, argv, index, "--frames", value, error)) {
                return false;
            }
            if (!parseInt(value, options.frames) || options.frames < 1 || options.frames > 10000) {
                error = "--frames must be an integer in [1,10000]";
                return false;
            }
        } else if (argument == "--warmup") {
            if (!consumeValue(argc, argv, index, "--warmup", value, error)) {
                return false;
            }
            if (!parseInt(value, options.warmupFrames) || options.warmupFrames < 0 ||
                options.warmupFrames > 10000) {
                error = "--warmup must be an integer in [0,10000]";
                return false;
            }
        } else if (argument == "--add-light") {
            if (!consumeValue(argc, argv, index, "--add-light", value, error)) {
                return false;
            }
            options.addLightSpecs.push_back(value);
        } else if (argument == "--remove-light") {
            if (!consumeValue(argc, argv, index, "--remove-light", value, error)) {
                return false;
            }
            options.removeLightIds.push_back(value);
        } else {
            error = "unknown argument: " + argument;
            return false;
        }
    }

    if (!options.modeSpecified) {
        error = "one render mode is required";
        return false;
    }
    if ((!options.addLightSpecs.empty() || !options.removeLightIds.empty()) &&
        options.mode != RunMode::SnapshotLoad) {
        error = "--add-light and --remove-light require --snapshot-load";
        return false;
    }
    return true;
}

void printUsage(std::ostream& output) {
    output
        << "Deferred snapshot renderer (800x600)\n"
        << "Usage:\n"
        << "  deferred_renderer --render [options]\n"
        << "  deferred_renderer --snapshot-save [options]\n"
        << "  deferred_renderer --snapshot-load [options]\n"
        << "\n"
        << "Modes:\n"
        << "  --render           Rebuild the G-Buffer and run deferred lighting.\n"
        << "  --snapshot-save    Render live, then write a versioned snapshot.\n"
        << "  --snapshot-load    Restore G-Buffer/lights and run lighting without geometry.\n"
        << "\n"
        << "Options:\n"
        << "  --scene <json>     Scene description (default: sample_scene.json).\n"
        << "  --snapshot <path>  Snapshot input or output base.\n"
        << "  --output <png>     PNG output path (default: artifacts/render.png).\n"
        << "  --add-light <spec> Add after snapshot load:\n"
        << "                     id:point:x,y,z:r,g,b:intensity[:radius]\n"
        << "                     id:directional:dx,dy,dz:r,g,b:intensity\n"
        << "  --remove-light <id> Remove a light after snapshot load.\n"
        << "  --frames <n>       Measured frames (default: 1).\n"
        << "  --warmup <n>       Unmeasured warmup frames (default: 0).\n"
        << "  --compare <png>    Compare output with a reference image.\n"
        << "  --max-channel-diff <n> Allowed per-channel difference (default: 1).\n"
        << "  --help             Show this help.\n"
        << "\n"
        << "Exit codes: 0 success, 1 runtime/validation failure, 2 invalid arguments.\n"
        << "Timing: one JSON object per measured frame and a final summary on stdout.\n"
        << "Errors: JSON diagnostics on stderr; snapshot failures fall back to live rendering.\n";
}

} // namespace dr
