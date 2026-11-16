#include "common.hpp"
#include "config.hpp"
#include "gl_backend.hpp"
#include "logger.hpp"
#include "ppm.hpp"
#include "renderer.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace skin {
namespace {

struct CommandLine {
    std::filesystem::path configPath;
    std::optional<std::filesystem::path> outputDirectory;
    std::filesystem::path logPath = "render.log";
    std::string backend = "auto";
    bool help = false;
    std::string error;
};

struct RunSummary {
    int successCount = 0;
    int failureCount = 0;
    int exitCode = 2;
};

void printUsage(std::ostream& output) {
    output
        << "Usage: skin_renderer --config <config.json>"
        << " [--output-dir <dir>] [--log <render.log>]"
        << " [--backend auto|egl|glfw]\n"
        << "Offscreen OpenGL skin material batch renderer.\n";
}

CommandLine parseCommandLine(int argc, char** argv) {
    CommandLine command;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto requireValue = [&](const char* option) -> std::string {
            if (i + 1 >= argc) {
                command.error = std::string("missing value for ") + option;
                return {};
            }
            return argv[++i];
        };

        if (argument == "--config") {
            const std::string value = requireValue("--config");
            if (!value.empty()) {
                command.configPath = value;
            }
        } else if (argument == "--output-dir") {
            const std::string value = requireValue("--output-dir");
            if (!value.empty()) {
                command.outputDirectory = std::filesystem::path(value);
            }
        } else if (argument == "--log") {
            const std::string value = requireValue("--log");
            if (!value.empty()) {
                command.logPath = value;
            }
        } else if (argument == "--backend") {
            const std::string value = requireValue("--backend");
            if (!value.empty()) {
                command.backend = value;
            }
        } else if (argument == "--help" || argument == "-h") {
            command.help = true;
        } else {
            command.error = "unknown argument: " + argument;
            break;
        }
    }
    if (!command.help && command.error.empty()
        && command.configPath.empty()) {
        command.error = "--config is required";
    }
    return command;
}

std::string formatDouble(double value) {
    std::ostringstream output;
    output << std::setprecision(12) << value;
    return output.str();
}

struct BaselineResult {
    std::string status = "none";
    std::optional<PixelDifference> difference;
    std::string baselineFingerprint;
    std::string warning;
};

BaselineResult evaluateBaseline(
    const GroupSpec& group,
    const RgbImage& rendered,
    Logger& logger
) {
    BaselineResult result;
    if (group.baselineConfigured.empty()) {
        return result;
    }

    std::error_code existsError;
    const bool exists = std::filesystem::exists(group.baselinePath, existsError);
    if (existsError || !exists) {
        result.status = "missing";
        result.warning =
            "baseline file does not exist; comparison skipped: "
            + group.baselinePath.string();
        logger.warning(group.name, kErrorNone, result.warning);
        return result;
    }

    std::string loadError;
    const RgbImage baseline = loadImageForComparison(
        group.baselinePath,
        loadError
    );
    if (!baseline.valid()) {
        result.status = "invalid";
        result.warning =
            "baseline could not be decoded; comparison skipped: " + loadError;
        logger.warning(group.name, kErrorNone, result.warning);
        return result;
    }
    result.baselineFingerprint = formatHex64(
        fnv1a64(baseline.pixels.data(), baseline.pixels.size())
    );

    PixelDifference difference;
    std::string compareError;
    if (!compareImages(rendered, baseline, difference, compareError)) {
        result.status = "dimension_mismatch";
        result.warning =
            "baseline comparison skipped: " + compareError;
        logger.warning(group.name, kErrorNone, result.warning);
        return result;
    }

    result.status = "compared";
    result.difference = difference;
    std::ostringstream statistics;
    statistics
        << "schema=" << kStatsSchemaVersion
        << " baseline=\"" << group.baselinePath.string() << "\""
        << " baseline_version=\"" << group.baselineVersion << "\""
        << " pixels=" << difference.pixelCount
        << " mse=" << formatDouble(difference.meanSquaredError)
        << " mean_abs=" << formatDouble(difference.meanAbsoluteError)
        << " max_abs=" << difference.maximumAbsoluteError;
    logger.stat(group.name, statistics.str());
    return result;
}

bool writeDiffStats(
    const std::filesystem::path& path,
    const nlohmann::json& stats,
    std::string& error
) {
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        error = "cannot open stats file: " + path.string();
        return false;
    }
    output << stats.dump(2) << '\n';
    output.flush();
    if (!output.good()) {
        error = "failed writing stats file: " + path.string();
        return false;
    }
    return true;
}

}  // namespace

RunSummary runApplication(const CommandLine& command, Logger& logger) {
    RunSummary summary;
    if (!command.error.empty()) {
        logger.error("-", kErrorGroupJson, command.error);
        return summary;
    }
    if (command.backend != "auto"
        && command.backend != "egl"
        && command.backend != "glfw") {
        logger.error("-", kErrorGroupJson, "invalid backend selection");
        return summary;
    }

    ConfigLoadResult loaded = loadConfig(command.configPath);
    if (!loaded.ok) {
        logger.error("-", kErrorGroupJson, "config initialization failed: " + loaded.error);
        return summary;
    }
    RenderConfig& config = loaded.config;
    if (command.outputDirectory.has_value()) {
        std::error_code absoluteError;
        config.outputDirectory = std::filesystem::absolute(
            *command.outputDirectory,
            absoluteError
        );
        if (absoluteError) {
            config.outputDirectory = *command.outputDirectory;
        }
        config.outputDirectoryConfigured =
            config.outputDirectory.string();
    }

    std::error_code directoryError;
    std::filesystem::create_directories(
        config.outputDirectory,
        directoryError
    );
    if (directoryError
        || !std::filesystem::is_directory(config.outputDirectory)) {
        logger.error(
            "-",
            kErrorOutputIo,
            "cannot create output directory: " + config.outputDirectory.string()
        );
        return summary;
    }

    std::string contextError;
    std::unique_ptr<GlContext> context = GlContext::create(
        command.backend,
        contextError
    );
    if (!context) {
        logger.error(
            "-",
            kErrorOpenGl,
            "OpenGL context initialization failed: " + contextError
        );
        return summary;
    }

    logger.info(
        "-",
        "config=\"" + config.sourcePath.string() + "\""
            + " fingerprint=" + config.fingerprint
            + " width=" + std::to_string(config.width)
            + " height=" + std::to_string(config.height)
            + " groups=" + std::to_string(config.groups.size())
            + " seed=" + std::to_string(config.seed)
    );

    int exitCode = 0;
    std::vector<nlohmann::json> groupStats;
    groupStats.reserve(config.groups.size());
    OutputPathAllocator outputNames(config.outputDirectory);

    try {
        Renderer renderer(*context, config, logger);
        for (const GroupSpec& group : config.groups) {
            try {
                if (!group.valid()) {
                    throw GroupFailure(
                        group.errorCode,
                        group.errorMessage.empty()
                            ? "material parameter validation failed"
                            : group.errorMessage
                    );
                }

                const RgbImage image = renderer.renderGroup(
                    group.name,
                    group.primary,
                    group.reference
                );
                const BaselineResult baseline = evaluateBaseline(
                    group,
                    image,
                    logger
                );
                const std::filesystem::path outputPath =
                    outputNames.allocate(group.name);

                std::string writeError;
                if (!writePpmAtomic(outputPath, image, writeError)) {
                    outputNames.release(outputPath);
                    throw GroupFailure(kErrorOutputIo, writeError);
                }

                const std::string renderFingerprint = formatHex64(
                    fnv1a64(image.pixels.data(), image.pixels.size())
                );

                nlohmann::json record;
                record["index"] = group.index;
                record["name"] = group.name;
                record["status"] = "success";
                record["output"] = outputPath.string();
                record["render_fingerprint"] = renderFingerprint;
                record["baseline_path"] = group.baselineConfigured;
                record["baseline_version"] = group.baselineVersion;
                record["baseline_status"] = baseline.status;
                if (!baseline.baselineFingerprint.empty()) {
                    record["baseline_fingerprint"] =
                        baseline.baselineFingerprint;
                }
                if (!baseline.warning.empty()) {
                    record["baseline_warning"] = baseline.warning;
                }
                if (baseline.difference.has_value()) {
                    record["comparison"] = {
                        {"schema_version", kStatsSchemaVersion},
                        {"pixel_count", baseline.difference->pixelCount},
                        {"mse", baseline.difference->meanSquaredError},
                        {"mean_absolute_error",
                            baseline.difference->meanAbsoluteError},
                        {"max_absolute_error",
                            baseline.difference->maximumAbsoluteError},
                    };
                }
                groupStats.push_back(std::move(record));
                ++summary.successCount;
                logger.info(
                    group.name,
                    "rendered output=\"" + outputPath.string() + "\""
                );
            } catch (const GroupFailure& failure) {
                ++summary.failureCount;
                logger.error(group.name, failure.code(), failure.what());
                groupStats.push_back({
                    {"index", group.index},
                    {"name", group.name},
                    {"status", "failed"},
                    {"error_code", failure.code()},
                    {"error", failure.what()},
                });
            } catch (const std::exception& exception) {
                ++summary.failureCount;
                logger.error(group.name, kErrorUnexpected, exception.what());
                groupStats.push_back({
                    {"index", group.index},
                    {"name", group.name},
                    {"status", "failed"},
                    {"error_code", kErrorUnexpected},
                    {"error", exception.what()},
                });
            }
        }
        exitCode = summary.failureCount == 0 ? 0 : 1;
        renderer.shutdown();
    } catch (const std::exception& exception) {
        logger.error(
            "-",
            kErrorOpenGl,
            std::string("renderer initialization failed: ") + exception.what()
        );
        exitCode = 2;
    }

    nlohmann::json stats;
    stats["stats_schema_version"] = kStatsSchemaVersion;
    stats["renderer_version"] = kRendererVersion;
    stats["generated_at"] = Logger::timestamp();
    stats["config_path"] = config.sourcePath.string();
    stats["config_fingerprint"] = config.fingerprint;
    stats["backend"] = context->info().backend;
    stats["run"] = {
        {"width", config.width},
        {"height", config.height},
        {"seed", config.seed},
        {"success_count", summary.successCount},
        {"failure_count", summary.failureCount},
        {"exit_code", exitCode},
    };
    stats["groups"] = groupStats;

    std::string statsError;
    if (!writeDiffStats("diff_stats.json", stats, statsError)) {
        logger.error("-", kErrorOutputIo, statsError);
    }

    logger.info(
        "-",
        "run_complete success=" + std::to_string(summary.successCount)
            + " failed=" + std::to_string(summary.failureCount)
            + " exit=" + std::to_string(exitCode)
    );
    context->shutdown();
    summary.exitCode = exitCode;
    return summary;
}

}  // namespace skin

int main(int argc, char** argv) {
    skin::CommandLine command = skin::parseCommandLine(argc, argv);
    if (command.help) {
        skin::printUsage(std::cout);
        return 0;
    }

    skin::Logger logger;
    if (!logger.open(command.logPath)) {
        std::cout << "success=0 failed=0 exit=2\n";
        return 2;
    }

    skin::RunSummary summary;
    try {
        summary = skin::runApplication(command, logger);
    } catch (const std::exception& exception) {
        logger.error(
            "-",
            skin::kErrorUnexpected,
            std::string("unhandled initialization error: ") + exception.what()
        );
        summary.exitCode = 2;
    }
    std::cout << "success=" << summary.successCount
              << " failed=" << summary.failureCount
              << " exit=" << summary.exitCode << '\n';
    return summary.exitCode;
}

