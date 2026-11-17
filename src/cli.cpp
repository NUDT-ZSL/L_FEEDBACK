#include "cli.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string_view>

#include "error.h"

namespace sr {
namespace {

int parseInteger(const std::string& key, const std::string& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' ||
        parsed < -2147483647L || parsed > 2147483647L) {
        fail(ExitCode::Usage, "E_USAGE", "--" + key + " expects an integer");
    }
    return static_cast<int>(parsed);
}

double parseDouble(const std::string& key, const std::string& value) {
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (errno != 0 || end == value.c_str() || *end != '\0' ||
        !std::isfinite(parsed)) {
        fail(ExitCode::Usage, "E_USAGE", "--" + key + " expects a finite number");
    }
    return parsed;
}

std::pair<std::string, std::string> splitOption(const std::string& argument) {
    const std::size_t equals = argument.find('=');
    if (equals == std::string::npos) {
        return {argument, {}};
    }
    return {argument.substr(0, equals), argument.substr(equals + 1)};
}

} // namespace

const char* contextKindName(ContextKind kind) {
    switch (kind) {
    case ContextKind::Auto:
        return "auto";
    case ContextKind::Glfw:
        return "glfw";
    case ContextKind::OSMesa:
        return "osmesa";
    }
    return "unknown";
}

Options parseCommandLine(int argc, char** argv) {
    Options options;
    bool outputSpecified = false;

    for (int i = 1; i < argc; ++i) {
        std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            options.helpRequested = true;
            continue;
        }
        if (argument.rfind("--", 0) != 0) {
            fail(ExitCode::Usage, "E_USAGE",
                 "unexpected positional argument '" + argument + "'");
        }

        auto [keyWithDashes, inlineValue] = splitOption(argument);
        const std::string key = keyWithDashes.substr(2);
        auto readValue = [&]() -> std::string {
            if (!inlineValue.empty()) {
                return inlineValue;
            }
            if (i + 1 >= argc) {
                fail(ExitCode::Usage, "E_USAGE", "--" + key + " requires a value");
            }
            return argv[++i];
        };

        if (key == "start-frame") {
            options.startFrame = parseInteger(key, readValue());
        } else if (key == "frames") {
            options.frameCount = parseInteger(key, readValue());
        } else if (key == "width") {
            options.width = parseInteger(key, readValue());
        } else if (key == "height") {
            options.height = parseInteger(key, readValue());
        } else if (key == "shadow-size") {
            options.shadowSize = parseInteger(key, readValue());
        } else if (key == "cascades" || key == "cascade-count") {
            options.cascadeCount = parseInteger(key, readValue());
        } else if (key == "split-lambda") {
            options.splitLambda = parseDouble(key, readValue());
        } else if (key == "fps") {
            options.fps = parseDouble(key, readValue());
        } else if (key == "mesh-variant") {
            options.meshVariant = parseInteger(key, readValue());
        } else if (key == "regenerate-frame") {
            options.regenerateFrame = parseInteger(key, readValue());
        } else if (key == "regenerate-variant") {
            options.regenerateVariant = parseInteger(key, readValue());
        } else if (key == "output") {
            options.outputDirectory = readValue();
            outputSpecified = true;
        } else if (key == "context") {
            const std::string value = readValue();
            if (value == "auto") {
                options.context = ContextKind::Auto;
            } else if (value == "glfw") {
                options.context = ContextKind::Glfw;
            } else if (value == "osmesa") {
                options.context = ContextKind::OSMesa;
            } else {
                fail(ExitCode::Usage, "E_USAGE",
                     "--context must be auto, glfw, or osmesa");
            }
        } else if (key == "test-error") {
            options.testError = readValue();
            if (options.testError != "none" &&
                options.testError != "shader" &&
                options.testError != "framebuffer") {
                fail(ExitCode::Usage, "E_USAGE",
                     "--test-error must be none, shader, or framebuffer");
            }
        } else {
            fail(ExitCode::Usage, "E_USAGE", "unknown option --" + key);
        }
    }

    if (options.helpRequested) {
        return options;
    }
    if (!outputSpecified || options.outputDirectory.empty()) {
        fail(ExitCode::Usage, "E_USAGE", "--output is required");
    }
    if (options.startFrame < 0) {
        fail(ExitCode::Usage, "E_USAGE", "--start-frame must be non-negative");
    }
    if (options.frameCount < 1 || options.frameCount > 100000) {
        fail(ExitCode::Usage, "E_USAGE", "--frames must be in [1, 100000]");
    }
    if (options.width < 64 || options.width > 4096 ||
        options.height < 64 || options.height > 4096) {
        fail(ExitCode::Usage, "E_USAGE",
             "--width and --height must be in [64, 4096]");
    }
    if (!(options.fps > 0.0) || options.fps > 1000.0) {
        fail(ExitCode::Usage, "E_USAGE", "--fps must be in (0, 1000]");
    }
    if (options.meshVariant < 0 || options.meshVariant > 1000) {
        fail(ExitCode::Usage, "E_USAGE", "--mesh-variant must be in [0, 1000]");
    }
    if (options.regenerateFrame < -1) {
        fail(ExitCode::Usage, "E_USAGE", "--regenerate-frame must be -1 or non-negative");
    }
    if (options.regenerateVariant < 0 || options.regenerateVariant > 1000) {
        fail(ExitCode::Usage, "E_USAGE",
             "--regenerate-variant must be in [0, 1000]");
    }
    return options;
}

std::string usageText() {
    return R"(skinned_csm_renderer - deterministic offline skinned-character renderer

Usage:
  skinned_csm_renderer --output DIR [options]

Required:
  --output DIR              Independent output directory.

Frames and camera:
  --start-frame N           First frame index (default 0).
  --frames N                Number of frames to render (default 1).
  --fps F                   Animation/camera frames per second (default 24).

Image and shadow settings:
  --width N                 Color/depth width, 64..4096 (default 960).
  --height N                Color/depth height, 64..4096 (default 540).
  --cascades N              Directional shadow cascades, 2..4 (default 2).
  --shadow-size N           Per-cascade texture size, power of two,
                            256..4096 (default 1024).
  --split-lambda F          Practical/logarithmic split blend, 0..1 (default 0.25).

Deterministic mesh regeneration:
  --mesh-variant N          Initial procedural character variant (default 0).
  --regenerate-frame N      Rebuild the character mesh before this frame and
                            continue in the same process (default disabled).
  --regenerate-variant N    Variant used by the in-process rebuild.

Context and diagnostics:
  --context auto|glfw|osmesa
                            Windows uses GLFW. Linux validation uses OSMesa.
  --test-error none|shader|framebuffer
                            Test-only failure injection for exit-code checks.
  -h, --help                Print this text.

Exit codes:
  0 success, 2 usage, 3 output I/O, 4 GL context, 5 shader compile/link,
  6 incomplete framebuffer, 7 invalid cascade configuration, 8 scene/math.
)";
}

} // namespace sr
