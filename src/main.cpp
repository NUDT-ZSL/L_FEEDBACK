#include "raster/bmp.h"
#include "raster/dxmath_compat.h"
#include "raster/json.h"
#include "raster/renderer.h"
#include "raster/scene.h"

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

constexpr int kMaximumResolution = 16384;

void printUsage(std::ostream& output) {
    output << "Usage: soft_raster --input <scene.json> --output <image.bmp>\n"
              "                   [--width <pixels>] [--height <pixels>]\n"
              "\n"
              "Optional resolution overrides replace the scene resolution.\n";
}

bool parsePositiveInt(const std::string& text, int& value) {
    int parsed = 0;
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        parsed <= 0 || parsed > kMaximumResolution) {
        return false;
    }
    value = parsed;
    return true;
}

bool readFile(const std::filesystem::path& path, std::string& content,
              std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open scene file";
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size < 0 || size > 64LL * 1024LL * 1024LL) {
        error = "scene file is too large";
        return false;
    }
    content.resize(static_cast<std::size_t>(size));
    if (size > 0 &&
        !input.read(content.data(), static_cast<std::streamsize>(size))) {
        error = "failed to read scene file";
        return false;
    }
    return true;
}

struct CommandLine {
    std::string input;
    std::string output;
    std::optional<int> width;
    std::optional<int> height;
    bool help = false;
};

bool parseCommandLine(int argc, char** argv, CommandLine& options,
                      std::string& error) {
    auto takeValue = [&](int& index, const std::string& flag,
                         std::string& value) -> bool {
        if (index + 1 >= argc) {
            error = "missing value after " + flag;
            return false;
        }
        value = argv[++index];
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else if (argument == "--input" || argument == "-i") {
            if (!takeValue(i, argument, options.input)) {
                return false;
            }
        } else if (argument.rfind("--input=", 0) == 0) {
            options.input = argument.substr(8);
        } else if (argument == "--output" || argument == "-o") {
            if (!takeValue(i, argument, options.output)) {
                return false;
            }
        } else if (argument.rfind("--output=", 0) == 0) {
            options.output = argument.substr(9);
        } else if (argument == "--width") {
            std::string value;
            if (!takeValue(i, argument, value)) {
                return false;
            }
            int parsed = 0;
            if (!parsePositiveInt(value, parsed)) {
                error = "invalid --width value";
                return false;
            }
            options.width = parsed;
        } else if (argument.rfind("--width=", 0) == 0) {
            int parsed = 0;
            if (!parsePositiveInt(argument.substr(8), parsed)) {
                error = "invalid --width value";
                return false;
            }
            options.width = parsed;
        } else if (argument == "--height") {
            std::string value;
            if (!takeValue(i, argument, value)) {
                return false;
            }
            int parsed = 0;
            if (!parsePositiveInt(value, parsed)) {
                error = "invalid --height value";
                return false;
            }
            options.height = parsed;
        } else if (argument.rfind("--height=", 0) == 0) {
            int parsed = 0;
            if (!parsePositiveInt(argument.substr(9), parsed)) {
                error = "invalid --height value";
                return false;
            }
            options.height = parsed;
        } else {
            error = "unknown argument: " + argument;
            return false;
        }
    }

    if (!options.help && (options.input.empty() || options.output.empty())) {
        error = "--input and --output are required";
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::cout.imbue(std::locale::classic());
    std::cerr.imbue(std::locale::classic());

    CommandLine options;
    std::string error;
    if (!parseCommandLine(argc, argv, options, error)) {
        std::cerr << "error: " << error << '\n';
        printUsage(std::cerr);
        return 1;
    }
    if (options.help) {
        printUsage(std::cout);
        return 0;
    }

    if (!raster::platform::verifyDirectXMathHeader()) {
        std::cerr << "error: DirectXMath platform sanity check failed\n";
        return 1;
    }

    std::string source;
    if (!readFile(options.input, source, error)) {
        std::cerr << "error: " << error << ": " << options.input << '\n';
        return 1;
    }

    try {
        const raster::JsonValue root = raster::parseJson(source);
        const raster::Scene scene =
            raster::parseScene(root, options.input);
        const int width = options.width.value_or(scene.width);
        const int height = options.height.value_or(scene.height);

        raster::RenderStats stats;
        const raster::RenderImage image = raster::renderScene(
            scene, width, height,
            [](const std::string& warning) {
                std::cerr << "warning: " << warning << '\n';
            },
            stats);

        std::string writeError;
        if (!raster::writeBmp24(options.output, image.width, image.height,
                                image.bgrTopDown, writeError)) {
            std::cerr << "error: cannot write output '" << options.output
                      << "': " << writeError << '\n';
            return 2;
        }

        std::cout << "{\"totalTriangles\":" << stats.totalTriangles
                  << ",\"culledMeshes\":" << stats.culledMeshes
                  << ",\"rasterizedTriangles\":" << stats.rasterizedTriangles
                  << ",\"totalFragments\":" << stats.totalFragments
                  << ",\"depthTestFailures\":" << stats.depthTestFailures
                  << "}\n";
    } catch (const raster::JsonError& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    } catch (const raster::SceneError& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    } catch (const std::bad_alloc&) {
        std::cerr << "error: insufficient memory while rendering\n";
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }

    return 0;
}

