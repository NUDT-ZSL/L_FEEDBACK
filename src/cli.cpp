#include "cli.hpp"

#include "asset.hpp"
#include "asset_reader.hpp"
#include "error.hpp"
#include "offline_renderer.hpp"
#include "preview.hpp"
#include "report.hpp"
#include "scene.hpp"

#ifdef SKELPREVIEW_HAS_TESTS
#include "test_runner.hpp"
#endif

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace skel {
namespace {

enum class Command {
    None,
    Preview,
    Offline,
    SelfTest,
    Help,
};

struct Options {
    Command command = Command::None;
    std::filesystem::path assetPath;
    std::filesystem::path selfTestDirectory;
    std::filesystem::path outputDirectory = "offline_out";
    std::optional<std::filesystem::path> reportPath;
    std::uint32_t startFrame = 0;
    std::uint32_t endFrame = 0;
    bool frameRangeProvided = false;
    bool outputDirectoryProvided = false;
    std::uint32_t width = 640;
    std::uint32_t height = 480;
    Camera camera;
    bool forceOffline = false;
};

struct RunResult {
    int exitCode = 0;
    Error error;
};

void printUsage(std::ostream& stream) {
    stream
        << "SkelPreview 1.0 - deterministic skeletal animation preview\n\n"
        << "Usage:\n"
        << "  skelpreview preview <asset.ska> [options]\n"
        << "  skelpreview offline <asset.ska> [options]\n"
        << "  skelpreview selftest [work-directory]\n\n"
        << "Common options:\n"
        << "  --frame N             Render one frame (default 0)\n"
        << "  --start-frame N       First frame of an inclusive range\n"
        << "  --end-frame N         Last frame of an inclusive range\n"
        << "  --width N             Image width (default 640)\n"
        << "  --height N            Image height (default 480)\n"
        << "  --output-dir DIR      Output directory (default offline_out)\n"
        << "  --report FILE         JSON report path\n"
        << "  --eye X,Y,Z           Camera eye position\n"
        << "  --target X,Y,Z        Camera target position\n"
        << "  --up X,Y,Z            Camera up vector\n"
        << "  --fov DEGREES         Vertical field of view\n"
        << "  --near VALUE          Near clipping plane\n"
        << "  --far VALUE           Far clipping plane\n"
        << "  --force-offline       Skip window creation and use fallback\n\n"
        << "Preview keys:\n"
        << "  Left drag rotate, wheel zoom, W wireframe, Space pause,\n"
        << "  R reset camera, Esc quit.\n\n"
        << "Outputs: depth_<frame:04>.pfm and color_<frame:04>.ppm.\n";
}

bool parseUnsigned(std::string_view text, std::uint32_t& value) {
    if (text.empty()) {
        return false;
    }
    std::uint64_t parsed = 0;
    for (const char character : text) {
        if (character < '0' || character > '9') {
            return false;
        }
        parsed = parsed * 10U +
                 static_cast<std::uint64_t>(character - '0');
        if (parsed > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
    }
    value = static_cast<std::uint32_t>(parsed);
    return true;
}

bool parseFloat(std::string_view text, float& value) {
    if (text.empty()) {
        return false;
    }
    const std::string owned(text);
    char* end = nullptr;
    const float parsed = std::strtof(owned.c_str(), &end);
    if (end == owned.c_str() || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

bool parseVec3(std::string_view text, glm::vec3& value) {
    std::vector<std::string_view> parts;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::size_t end =
            comma == std::string_view::npos ? text.size() : comma;
        parts.push_back(text.substr(begin, end - begin));
        if (comma == std::string_view::npos) {
            break;
        }
        begin = comma + 1U;
    }
    if (parts.size() != 3U) {
        return false;
    }
    return parseFloat(parts[0], value.x) && parseFloat(parts[1], value.y) &&
           parseFloat(parts[2], value.z);
}

bool nextValue(int argc, char** argv, int& index, std::string_view option,
               std::string& value, Error& error) {
    if (index + 1 >= argc) {
        error = makeError(ErrorCode::InvalidArgument,
                          "missing value for " + std::string(option));
        return false;
    }
    value = argv[++index];
    return true;
}

bool parseOptions(int argc, char** argv, Options& options, Error& error) {
    if (argc < 2) {
        error = makeError(ErrorCode::InvalidArgument, "missing command");
        return false;
    }
    const std::string command = argv[1];
    if (command == "help" || command == "--help" || command == "-h") {
        options.command = Command::Help;
        return true;
    }
    if (command == "preview") {
        options.command = Command::Preview;
    } else if (command == "offline") {
        options.command = Command::Offline;
    } else if (command == "selftest") {
        options.command = Command::SelfTest;
        if (argc >= 3 && argv[2][0] != '-') {
            options.selfTestDirectory = argv[2];
        } else {
            options.selfTestDirectory = "selftest_out";
        }
        return true;
    } else {
        error = makeError(ErrorCode::InvalidArgument,
                          "unknown command: " + command);
        return false;
    }

    if (argc < 3) {
        error = makeError(ErrorCode::InvalidArgument,
                          "missing asset path after command");
        return false;
    }
    options.assetPath = argv[2];
    bool explicitStart = false;
    bool explicitEnd = false;

    for (int index = 3; index < argc; ++index) {
        const std::string_view option = argv[index];
        std::string value;
        if (option == "--frame") {
            if (!nextValue(argc, argv, index, option, value, error) ||
                !parseUnsigned(value, options.startFrame)) {
                if (!error) {
                    error = makeError(ErrorCode::InvalidArgument,
                                      "invalid --frame value");
                }
                return false;
            }
            options.endFrame = options.startFrame;
            options.frameRangeProvided = true;
            explicitStart = true;
            explicitEnd = true;
        } else if (option == "--start-frame") {
            if (!nextValue(argc, argv, index, option, value, error) ||
                !parseUnsigned(value, options.startFrame)) {
                if (!error) {
                    error = makeError(ErrorCode::InvalidArgument,
                                      "invalid --start-frame value");
                }
                return false;
            }
            options.frameRangeProvided = true;
            explicitStart = true;
        } else if (option == "--end-frame") {
            if (!nextValue(argc, argv, index, option, value, error) ||
                !parseUnsigned(value, options.endFrame)) {
                if (!error) {
                    error = makeError(ErrorCode::InvalidArgument,
                                      "invalid --end-frame value");
                }
                return false;
            }
            options.frameRangeProvided = true;
            explicitEnd = true;
        } else if (option == "--width") {
            if (!nextValue(argc, argv, index, option, value, error) ||
                !parseUnsigned(value, options.width)) {
                if (!error) {
                    error = makeError(ErrorCode::InvalidArgument,
                                      "invalid --width value");
                }
                return false;
            }
        } else if (option == "--height") {
            if (!nextValue(argc, argv, index, option, value, error) ||
                !parseUnsigned(value, options.height)) {
                if (!error) {
                    error = makeError(ErrorCode::InvalidArgument,
                                      "invalid --height value");
                }
                return false;
            }
        } else if (option == "--output-dir") {
            if (!nextValue(argc, argv, index, option, value, error)) {
                return false;
            }
            options.outputDirectory = value;
            options.outputDirectoryProvided = true;
        } else if (option == "--report") {
            if (!nextValue(argc, argv, index, option, value, error)) {
                return false;
            }
            options.reportPath = std::filesystem::path(value);
        } else if (option == "--eye") {
            if (!nextValue(argc, argv, index, option, value, error) ||
                !parseVec3(value, options.camera.eye)) {
                if (!error) {
                    error = makeError(ErrorCode::InvalidArgument,
                                      "invalid --eye value; expected X,Y,Z");
                }
                return false;
            }
        } else if (option == "--target") {
            if (!nextValue(argc, argv, index, option, value, error) ||
                !parseVec3(value, options.camera.target)) {
                if (!error) {
                    error = makeError(ErrorCode::InvalidArgument,
                                      "invalid --target value; expected X,Y,Z");
                }
                return false;
            }
        } else if (option == "--up") {
            if (!nextValue(argc, argv, index, option, value, error) ||
                !parseVec3(value, options.camera.up)) {
                if (!error) {
                    error = makeError(ErrorCode::InvalidArgument,
                                      "invalid --up value; expected X,Y,Z");
                }
                return false;
            }
        } else if (option == "--fov") {
            if (!nextValue(argc, argv, index, option, value, error) ||
                !parseFloat(value, options.camera.fieldOfViewYDegrees)) {
                if (!error) {
                    error = makeError(ErrorCode::InvalidArgument,
                                      "invalid --fov value");
                }
                return false;
            }
        } else if (option == "--near") {
            if (!nextValue(argc, argv, index, option, value, error) ||
                !parseFloat(value, options.camera.nearPlane)) {
                if (!error) {
                    error = makeError(ErrorCode::InvalidArgument,
                                      "invalid --near value");
                }
                return false;
            }
        } else if (option == "--far") {
            if (!nextValue(argc, argv, index, option, value, error) ||
                !parseFloat(value, options.camera.farPlane)) {
                if (!error) {
                    error = makeError(ErrorCode::InvalidArgument,
                                      "invalid --far value");
                }
                return false;
            }
        } else if (option == "--force-offline") {
            options.forceOffline = true;
        } else if (option == "--help" || option == "-h") {
            options.command = Command::Help;
            return true;
        } else {
            error = makeError(ErrorCode::InvalidArgument,
                              "unknown option: " + std::string(option));
            return false;
        }
    }

    if (explicitStart != explicitEnd) {
        error = makeError(
            ErrorCode::InvalidArgument,
            "--start-frame and --end-frame must be provided together");
        return false;
    }
    if (options.width == 0U || options.height == 0U) {
        error = makeError(ErrorCode::InvalidArgument,
                          "image dimensions must be positive");
        return false;
    }
    if (options.startFrame > options.endFrame) {
        error = makeError(ErrorCode::InvalidFrameRange,
                          "start frame is greater than end frame");
        return false;
    }
    return true;
}

std::string frameFileName(const char* prefix, const char* extension,
                          std::uint32_t frame) {
    std::ostringstream stream;
    stream << prefix << '_' << std::setw(4) << std::setfill('0') << frame
           << extension;
    return stream.str();
}

RunResult runOffline(const Asset& asset, const Options& options,
                     std::string mode) {
    RunResult result;
    const auto totalStart = std::chrono::steady_clock::now();
    RenderSettings settings;
    settings.width = options.width;
    settings.height = options.height;
    settings.camera = options.camera;

    RenderReport report;
    report.assetFile = options.assetPath.filename().string();
    report.mode = std::move(mode);
    report.assetFrameCount = asset.frameCount;
    report.framesPerSecond = asset.framesPerSecond;
    report.compositionOrder = asset.compositionOrder;
    report.width = options.width;
    report.height = options.height;
    report.camera = options.camera;
    report.vertexCount = asset.vertexCount();
    report.triangleCount = asset.triangleCount();

    const std::filesystem::path outputDirectory = options.outputDirectory;
    const std::filesystem::path reportPath =
        options.reportPath.value_or(outputDirectory / "report.json");
    std::error_code directoryError;
    std::filesystem::create_directories(outputDirectory, directoryError);
    if (directoryError) {
        result.error = makeError(ErrorCode::WriteFailed,
                                 "cannot create output directory: " +
                                     directoryError.message());
        result.exitCode = processExitCode(result.error.code);
        return result;
    }

    for (std::uint64_t frameValue = options.startFrame;
         frameValue <= options.endFrame; ++frameValue) {
        const std::uint32_t frame = static_cast<std::uint32_t>(frameValue);
        if (frame >= asset.frameCount) {
            result.error = makeError(
                ErrorCode::InvalidFrameRange,
                "frame " + std::to_string(frame) +
                    " exceeds asset frame range [0, " +
                    std::to_string(asset.frameCount - 1U) + "]");
            result.exitCode = processExitCode(result.error.code);
            return result;
        }

        const auto frameStart = std::chrono::steady_clock::now();
        BakedFrame baked;
        if (!evaluateFrame(asset, frame, baked, result.error)) {
            result.exitCode = processExitCode(result.error.code);
            return result;
        }
        report.normalizedVertexCount =
            std::max(report.normalizedVertexCount,
                     baked.stats.normalizedVertices);

        OfflineImage image;
        if (!rasterizeOffline(baked, settings, image, result.error)) {
            result.exitCode = processExitCode(result.error.code);
            return result;
        }

        const std::string depthName =
            frameFileName("depth", ".pfm", frame);
        const std::string colorName =
            frameFileName("color", ".ppm", frame);
        if (!writeOfflineImages(image, settings,
                                outputDirectory / depthName,
                                outputDirectory / colorName,
                                result.error)) {
            result.exitCode = processExitCode(result.error.code);
            return result;
        }

        FrameRecord record;
        record.frame = frame;
        record.stats = baked.stats;
        record.depthFile = depthName;
        record.colorFile = colorName;
        record.elapsedMs =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - frameStart)
                .count();
        report.frames.push_back(std::move(record));

        if (frameValue == std::numeric_limits<std::uint32_t>::max()) {
            break;
        }
    }

    report.elapsedMs =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - totalStart)
            .count();
    if (!writeStatisticsJson(reportPath, report, result.error)) {
        result.exitCode = processExitCode(result.error.code);
        return result;
    }

    std::cout << "rendered " << report.frames.size() << " frame(s), "
              << report.vertexCount << " vertices, "
              << report.triangleCount << " triangles\n"
              << "report: " << reportPath.string() << "\n";
    if (report.totalNanPositions() != 0U ||
        report.totalNanNormals() != 0U) {
        std::cerr << "NaN detected: positions="
                  << report.totalNanPositions()
                  << " normals=" << report.totalNanNormals() << "\n";
        result.error = makeError(
            ErrorCode::NaNProduced,
            "one or more rendered frames contain NaN position/normal data");
        result.exitCode = processExitCode(ErrorCode::NaNProduced);
        return result;
    }
    result.exitCode = 0;
    result.error = {};
    return result;
}

RunResult executeOffline(const Options& options, std::string mode) {
    RunResult result;
    Asset asset;
    if (!AssetReader::load(options.assetPath, asset, result.error)) {
        result.exitCode = processExitCode(result.error.code);
        return result;
    }
    return runOffline(asset, options, std::move(mode));
}

RunResult executePreview(const Options& options) {
    RunResult result;
    Asset asset;
    if (!AssetReader::load(options.assetPath, asset, result.error)) {
        result.exitCode = processExitCode(result.error.code);
        return result;
    }

    RenderSettings settings;
    settings.width = options.width;
    settings.height = options.height;
    settings.camera = options.camera;
    PreviewResult preview = runPreview(
        asset, settings,
        options.frameRangeProvided ? options.startFrame : 0U,
        options.forceOffline);
    if (preview.completed) {
        result.exitCode = 0;
        return result;
    }

    if (preview.error.code == ErrorCode::OpenGLUnavailable) {
        std::cerr << "[preview] " << preview.diagnostic
                  << "\n[preview] falling back to deterministic offline "
                     "rendering\n";
        Options fallback = options;
        if (!fallback.outputDirectoryProvided) {
            fallback.outputDirectory = "preview_fallback";
        }
        return runOffline(asset, fallback, "offline_fallback");
    }

    result.error = preview.error;
    result.exitCode = processExitCode(preview.error.code);
    if (!preview.diagnostic.empty()) {
        std::cerr << "[preview] " << preview.diagnostic << "\n";
    }
    return result;
}

void printError(const Error& error) {
    std::cerr << errorCodeName(error.code) << " ("
              << static_cast<int>(error.code) << "): " << error.message
              << "\n";
}

} // namespace

int runCli(int argc, char** argv) {
    Options options;
    Error error;
    if (!parseOptions(argc, argv, options, error)) {
        printError(error);
        printUsage(std::cerr);
        return processExitCode(error.code);
    }

    if (options.command == Command::Help) {
        printUsage(std::cout);
        return 0;
    }
    if (options.command == Command::SelfTest) {
#ifdef SKELPREVIEW_HAS_TESTS
        return runSelfTests(options.selfTestDirectory);
#else
        std::cerr << "selftest support was disabled at configure time\n";
        return processExitCode(ErrorCode::InvalidArgument);
#endif
    }
    if (options.command == Command::Preview) {
        RunResult result = executePreview(options);
        if (result.exitCode != 0 && result.error) {
            printError(result.error);
        }
        return result.exitCode;
    }
    if (options.command == Command::Offline) {
        RunResult result = executeOffline(options, "offline");
        if (result.exitCode != 0 && result.error) {
            printError(result.error);
        }
        return result.exitCode;
    }
    printUsage(std::cerr);
    return processExitCode(ErrorCode::InvalidArgument);
}

} // namespace skel

