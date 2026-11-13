#include "terrain/cli.hpp"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

namespace terrain {
namespace {

bool parseUint32(const std::string& text, uint32_t& value) {
    if (text.empty() || text.front() == '-' || text.front() == '+') {
        return false;
    }
    uint64_t wide = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), wide, 10);
    if (error != std::errc{} || end != text.data() + text.size() ||
        wide > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    value = static_cast<uint32_t>(wide);
    return true;
}

bool parseInteger(const std::string& text, int& value) {
    if (text.empty()) {
        return false;
    }
    int parsed = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), parsed, 10);
    if (error != std::errc{} || end != text.data() + text.size()) {
        return false;
    }
    value = parsed;
    return true;
}

bool parseFloat(const std::string& text, float& value) {
    if (text.empty()) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const float parsed = std::strtof(text.c_str(), &end);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
        !std::isfinite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

ParseResult invalidParam(const std::string& message) {
    ParseResult result;
    result.ok = false;
    result.isPathError = false;
    result.error = "E_INVALID_PARAM: " + message;
    return result;
}

ParseResult badPath(const std::string& message) {
    ParseResult result;
    result.ok = false;
    result.isPathError = true;
    result.error = "E_BAD_PATH: " + message;
    return result;
}

std::string nextValue(
    int argc,
    char** argv,
    int& index,
    const std::string& option,
    ParseResult& error) {
    if (index + 1 >= argc) {
        error = option == "--camera-path"
                    ? badPath("missing value for --camera-path")
                    : invalidParam("missing value for " + option);
        return {};
    }
    ++index;
    return argv[index];
}

}  // namespace

ParseResult parseCommandLine(int argc, char** argv) {
    ParseResult result;
    result.options = Options{};
    if (argc <= 1) {
        result.ok = true;
        return result;
    }

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            result.options.help = true;
            result.ok = true;
            return result;
        }
        if (argument == "--replay") {
            result.options.replay = true;
            continue;
        }
        if (argument == "--headless") {
            result.options.headless = true;
            continue;
        }
        if (argument == "--interactive") {
            result.options.interactive = true;
            continue;
        }
        if (argument == "--seed") {
            const std::string value =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            uint32_t seed = 0;
            if (!parseUint32(value, seed)) {
                return invalidParam("--seed must be an unsigned 32-bit integer");
            }
            result.options.terrain.seed = seed;
            continue;
        }
        if (argument == "--amplitude") {
            const std::string value =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            float amplitude = 0.0f;
            if (!parseFloat(value, amplitude) || amplitude <= 0.0f ||
                amplitude > 4096.0f) {
                return invalidParam("--amplitude must be in (0, 4096]");
            }
            result.options.terrain.amplitude = amplitude;
            continue;
        }
        if (argument == "--frequency") {
            const std::string value =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            float frequency = 0.0f;
            if (!parseFloat(value, frequency) || frequency <= 0.0f ||
                frequency > 64.0f) {
                return invalidParam("--frequency must be in (0, 64]");
            }
            result.options.terrain.frequency = frequency;
            continue;
        }
        if (argument == "--octaves") {
            const std::string value =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            int octaves = 0;
            if (!parseInteger(value, octaves) || octaves < 1 || octaves > 12) {
                return invalidParam("--octaves must be in [1, 12]");
            }
            result.options.terrain.octaves = octaves;
            continue;
        }
        if (argument == "--camera-path") {
            const std::string value =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            result.options.cameraPath = value;
            result.options.replay = true;
            continue;
        }
        if (argument == "--screenshot") {
            result.options.screenshot =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            continue;
        }
        if (argument == "--capture-dir") {
            result.options.captureDirectory =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            continue;
        }
        if (argument == "--time-step") {
            const std::string value =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            if (!parseFloat(value, result.options.timeStep) ||
                result.options.timeStep <= 0.0f ||
                result.options.timeStep > 60.0f) {
                return invalidParam("--time-step must be in (0, 60]");
            }
            continue;
        }
        if (argument == "--frames") {
            const std::string value =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            if (!parseInteger(value, result.options.frames) ||
                result.options.frames < 1 ||
                result.options.frames > 100000) {
                return invalidParam("--frames must be in [1, 100000]");
            }
            continue;
        }
        if (argument == "--width") {
            const std::string value =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            if (!parseInteger(value, result.options.width) ||
                result.options.width < 64 ||
                result.options.width > 4096) {
                return invalidParam("--width must be in [64, 4096]");
            }
            continue;
        }
        if (argument == "--height") {
            const std::string value =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            if (!parseInteger(value, result.options.height) ||
                result.options.height < 64 ||
                result.options.height > 4096) {
                return invalidParam("--height must be in [64, 4096]");
            }
            continue;
        }
        if (argument == "--sse") {
            const std::string value =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            if (!parseFloat(value, result.options.sseThreshold) ||
                result.options.sseThreshold <= 0.0f ||
                result.options.sseThreshold > 1000.0f) {
                return invalidParam("--sse must be in (0, 1000]");
            }
            continue;
        }
        if (argument == "--cache-capacity") {
            const std::string value =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            if (!parseInteger(value, result.options.cacheCapacity) ||
                result.options.cacheCapacity < 16 ||
                result.options.cacheCapacity > 65536) {
                return invalidParam("--cache-capacity must be in [16, 65536]");
            }
            continue;
        }
        if (argument == "--debug-view") {
            const std::string value =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            if (value == "normal") {
                result.options.debugView = DebugView::Normal;
            } else if (value == "wireframe") {
                result.options.debugView = DebugView::Wireframe;
            } else if (value == "lod") {
                result.options.debugView = DebugView::Lod;
            } else if (value == "seam") {
                result.options.debugView = DebugView::Seam;
            } else {
                return invalidParam(
                    "--debug-view must be normal, wireframe, lod, or seam");
            }
            continue;
        }
        if (argument == "--backend") {
            const std::string value =
                nextValue(argc, argv, index, argument, result);
            if (!result.error.empty()) {
                return result;
            }
            if (value == "auto") {
                result.options.backend = Backend::Auto;
            } else if (value == "gl") {
                result.options.backend = Backend::Gl;
            } else if (value == "software") {
                result.options.backend = Backend::Software;
            } else {
                return invalidParam("--backend must be auto, gl, or software");
            }
            continue;
        }
        return invalidParam("unknown argument: " + argument);
    }

    if (result.options.interactive && result.options.replay) {
        return invalidParam("--interactive cannot be combined with --replay");
    }
    if (result.options.replay && result.options.cameraPath.empty()) {
        return badPath("--replay requires --camera-path");
    }
    if (!result.options.captureDirectory.empty()) {
        result.options.headless = true;
    }
    result.ok = true;
    return result;
}

std::string debugViewName(DebugView view) {
    switch (view) {
        case DebugView::Normal:
            return "normal";
        case DebugView::Wireframe:
            return "wireframe";
        case DebugView::Lod:
            return "lod";
        case DebugView::Seam:
            return "seam";
    }
    return "normal";
}

std::string usageText() {
    return R"(terrain_lod - deterministic quadtree terrain LOD

Usage:
  terrain_lod --camera-path FILE [options]
  terrain_lod [--interactive] [options]
  terrain_lod --replay --camera-path FILE [options]

Terrain:
  --seed N              unsigned 32-bit seed (default 1337)
  --amplitude F         height amplitude in (0, 4096] (default 48)
  --frequency F         base noise frequency in (0, 64] (default 2.4)
  --octaves N           fBm octaves in [1, 12] (default 5)
  --sse F               screen-space error threshold in (0, 1000] (default 16)

Replay:
  --camera-path FILE    text keyframes: time px py pz yaw pitch
  --time-step F         replay time step in (0, 60] seconds (default 0.1)
  --frames N            number of frames in [1, 100000] (default 1)
  --screenshot FILE     write the final frame as binary PPM (P6)
  --capture-dir DIR     write every frame as DIR/frame_NNNNNN.ppm

View and backend:
  --debug-view V        normal | wireframe | lod | seam
  --backend V           auto | gl | software
  --width N             framebuffer width in [64, 4096] (default 960)
  --height N            framebuffer height in [64, 4096] (default 540)
  --headless            keep the GLFW window hidden
  --interactive         open the interactive desktop window
  --cache-capacity N    bounded node cache capacity in [16, 65536]

Exit codes:
  0 success
  2 E_INVALID_PARAM: invalid numerical or command-line parameter
  3 E_BAD_PATH: camera path missing, unreadable, or malformed
  4 rendering/backend initialization failure
)";
}

}  // namespace terrain

