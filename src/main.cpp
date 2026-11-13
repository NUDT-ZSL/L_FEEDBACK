#include "atomic_io.hpp"
#include "cache.hpp"
#include "errors.hpp"
#include "image.hpp"
#include "render.hpp"
#include "scene.hpp"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <locale>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace {

volatile std::sig_atomic_t g_interrupted = 0;

void handleInterrupt(int) {
    g_interrupted = 1;
}

void printUsage(std::ostream& output) {
    output << "Usage: lf_baker --scene <scene.json> --output <dir> --cache <dir>\n";
}

std::filesystem::path pathFromUtf8(const std::string& text) {
#ifdef _WIN32
    return std::filesystem::u8path(text);
#else
    return std::filesystem::path(text);
#endif
}

struct Options {
    std::filesystem::path scenePath;
    std::filesystem::path outputDirectory;
    std::filesystem::path cacheDirectory;
};

Options parseArguments(const std::vector<std::string>& arguments) {
    Options options;
    bool haveScene = false;
    bool haveOutput = false;
    bool haveCache = false;

    for (std::size_t i = 1; i < arguments.size(); ++i) {
        const std::string& argument = arguments[i];
        if (argument == "--help" || argument == "-h") {
            printUsage(std::cout);
            std::exit(0);
        }

        std::string key = argument;
        std::optional<std::string> inlineValue;
        const std::size_t equals = argument.find('=');
        if (equals != std::string::npos) {
            key = argument.substr(0, equals);
            inlineValue = argument.substr(equals + 1);
        }

        if (key != "--scene" && key != "--output" && key != "--cache") {
            throw lfb::BusinessError(lfb::ExitCode::E_USAGE,
                                     "unknown argument '" + argument + "'");
        }
        if (i + 1 >= arguments.size() && !inlineValue.has_value()) {
            throw lfb::BusinessError(lfb::ExitCode::E_USAGE,
                                     "missing value for " + key);
        }
        const std::string value = inlineValue.has_value() ? *inlineValue : arguments[++i];
        if (value.empty()) {
            throw lfb::BusinessError(lfb::ExitCode::E_USAGE, "empty value for " + key);
        }

        if (key == "--scene") {
            if (haveScene) {
                throw lfb::BusinessError(lfb::ExitCode::E_USAGE, "--scene specified twice");
            }
            options.scenePath = pathFromUtf8(value);
            haveScene = true;
        } else if (key == "--output") {
            if (haveOutput) {
                throw lfb::BusinessError(lfb::ExitCode::E_USAGE, "--output specified twice");
            }
            options.outputDirectory = pathFromUtf8(value);
            haveOutput = true;
        } else {
            if (haveCache) {
                throw lfb::BusinessError(lfb::ExitCode::E_USAGE, "--cache specified twice");
            }
            options.cacheDirectory = pathFromUtf8(value);
            haveCache = true;
        }
    }

    if (!haveScene || !haveOutput || !haveCache) {
        throw lfb::BusinessError(
            lfb::ExitCode::E_USAGE,
            "--scene, --output and --cache are all required");
    }
    return options;
}

std::string frameName(std::uint32_t index, const char* extension) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "frame_%06u%s", index, extension);
    return buffer;
}

std::string averageText(double average) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.2f", average);
    return buffer;
}

void writeManifest(const std::filesystem::path& outputDirectory,
                   const lfb::Scene& scene,
                   const std::vector<double>& averages) {
    std::ostringstream manifest;
    manifest << "{\n";
    manifest << "  \"frames\": [\n";
    for (std::size_t i = 0; i < scene.frames.size(); ++i) {
        const auto& frame = scene.frames[i];
        manifest << "    {\n";
        manifest << "      \"index\": " << frame.index << ",\n";
        manifest << "      \"path\": \"" << frameName(frame.index, ".png") << "\",\n";
        manifest << "      \"average_luminance\": " << averageText(averages[i]) << "\n";
        manifest << "    }";
        if (i + 1 != scene.frames.size()) {
            manifest << ",";
        }
        manifest << "\n";
    }
    manifest << "  ]\n";
    manifest << "}\n";

    const std::string text = manifest.str();
    const std::vector<std::uint8_t> bytes(text.begin(), text.end());
    std::string error;
    if (!lfb::io::atomicWriteBytes(outputDirectory / "manifest.json", bytes, error)) {
        throw lfb::BusinessError(lfb::ExitCode::E_OUTPUT_IO,
                                 "cannot write manifest.json: " + error);
    }
}

void removeStaleFrames(const std::filesystem::path& outputDirectory,
                       const lfb::Scene& scene) {
    std::set<std::string> expected;
    for (const auto& frame : scene.frames) {
        expected.insert(frameName(frame.index, ".png"));
    }
    std::error_code ec;
    std::filesystem::directory_iterator iterator(outputDirectory, ec);
    const std::filesystem::directory_iterator end;
    while (!ec && iterator != end) {
        const auto& entry = *iterator;
        const std::string name = entry.path().filename().string();
        const bool looksLikeFrame =
            name.rfind("frame_", 0) == 0 &&
            name.size() >= 4 &&
            name.compare(name.size() - 4, 4, ".png") == 0;
        if (looksLikeFrame && expected.find(name) == expected.end()) {
            std::filesystem::remove(entry.path(), ec);
            if (ec) {
                throw lfb::BusinessError(
                    lfb::ExitCode::E_OUTPUT_IO,
                    "cannot remove stale output '" + name + "': " + ec.message());
            }
        }
        iterator.increment(ec);
    }
    if (ec) {
        throw lfb::BusinessError(lfb::ExitCode::E_OUTPUT_IO,
                                 "cannot scan output directory: " + ec.message());
    }
}

int run(const std::vector<std::string>& arguments) {
    std::cout.imbue(std::locale::classic());
    std::cerr.imbue(std::locale::classic());

    std::signal(SIGINT, handleInterrupt);
    std::signal(SIGTERM, handleInterrupt);

    const Options options = parseArguments(arguments);
    const lfb::Scene scene = lfb::loadScene(options.scenePath);

    std::string ioError;
    if (!lfb::io::ensureWritableDirectory(options.outputDirectory, ioError)) {
        throw lfb::BusinessError(lfb::ExitCode::E_OUTPUT_IO, ioError);
    }
    if (!lfb::io::ensureWritableDirectory(options.cacheDirectory, ioError)) {
        throw lfb::BusinessError(lfb::ExitCode::E_CACHE_IO, ioError);
    }

    lfb::io::cleanupTemporaryFiles(options.outputDirectory);
    {
        std::error_code removeError;
        std::filesystem::remove(options.outputDirectory / "manifest.json", removeError);
        if (removeError) {
            throw lfb::BusinessError(
                lfb::ExitCode::E_OUTPUT_IO,
                "cannot remove stale manifest.json: " + removeError.message());
        }
    }

    const std::string globalKey = lfb::globalCacheKey(scene);
    lfb::FrameCache cache(options.cacheDirectory, globalKey);
    cache.cleanHalfProducts();

    std::vector<double> averages(scene.frames.size(), 0.0);
    std::size_t baked = 0;
    std::size_t cached = 0;
    std::size_t completed = 0;
    const std::size_t total = scene.frames.size();

    for (std::size_t i = 0; i < scene.frames.size(); ++i) {
        if (g_interrupted != 0) {
            break;
        }
        const lfb::FrameDesc& frame = scene.frames[i];
        const std::string subKey = lfb::frameCacheKey(scene, frame);

        std::optional<lfb::Image> image =
            cache.load(frame.index, subKey, scene.width, scene.height);
        if (image.has_value()) {
            ++cached;
        } else {
            image = lfb::renderFrame(scene, frame);
            cache.commit(frame.index, subKey, *image);
            ++baked;
        }

        const std::vector<std::uint8_t> png = lfb::encodePng(*image);
        const std::filesystem::path pngPath =
            options.outputDirectory / frameName(frame.index, ".png");
        if (!lfb::io::atomicWriteBytes(pngPath, png, ioError)) {
            throw lfb::BusinessError(
                lfb::ExitCode::E_OUTPUT_IO,
                "cannot write output '" + pngPath.string() + "': " + ioError);
        }

        averages[i] = lfb::averageRgb(*image);
        ++completed;
        std::cout << "FRAME " << (i + 1) << "/" << total << '\n' << std::flush;
    }

    if (completed == total) {
        removeStaleFrames(options.outputDirectory, scene);
        writeManifest(options.outputDirectory, scene, averages);
    }

    std::cout << "SUMMARY " << baked << " " << cached << " " << total << '\n'
              << std::flush;
    return 0;
}

}  // namespace

#ifdef _WIN32
std::vector<std::string> utf8Arguments(int argc, wchar_t** argv) {
    std::vector<std::string> result;
    result.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        const int bytes = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
        if (bytes <= 1) {
            result.emplace_back();
            continue;
        }
        std::vector<char> buffer(static_cast<std::size_t>(bytes), '\0');
        WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, buffer.data(), bytes, nullptr, nullptr);
        result.emplace_back(buffer.data(), static_cast<std::size_t>(bytes - 1));
    }
    return result;
}

int wmain(int argc, wchar_t** argv) {
    try {
        return run(utf8Arguments(argc, argv));
    } catch (const lfb::BusinessError& error) {
        std::cerr << lfb::exitCodeName(error.code()) << ": " << error.what() << '\n';
        return static_cast<int>(error.code());
    } catch (const std::exception& error) {
        std::cerr << "E_RUNTIME: " << error.what() << '\n';
        return static_cast<int>(lfb::ExitCode::E_RUNTIME);
    }
}
#else
int main(int argc, char** argv) {
    try {
        std::vector<std::string> arguments;
        arguments.reserve(static_cast<std::size_t>(argc));
        for (int i = 0; i < argc; ++i) {
            arguments.emplace_back(argv[i]);
        }
        return run(arguments);
    } catch (const lfb::BusinessError& error) {
        std::cerr << lfb::exitCodeName(error.code()) << ": " << error.what() << '\n';
        return static_cast<int>(error.code());
    } catch (const std::exception& error) {
        std::cerr << "E_RUNTIME: " << error.what() << '\n';
        return static_cast<int>(lfb::ExitCode::E_RUNTIME);
    }
}
#endif
