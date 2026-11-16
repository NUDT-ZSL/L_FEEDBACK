#include "cli.hpp"

#include "util.hpp"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <sstream>
#include <string>

namespace mv {
namespace {

std::optional<std::pair<int, int>> parseSize(const std::string& value) {
    const auto parts = split(value, 'x');
    if (parts.size() != 2) {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    const long width = std::strtol(parts[0].c_str(), &end, 10);
    if (errno != 0 || end == parts[0].c_str() || *end != '\0' || width <= 0) {
        return std::nullopt;
    }
    errno = 0;
    const long height = std::strtol(parts[1].c_str(), &end, 10);
    if (errno != 0 || end == parts[1].c_str() || *end != '\0' || height <= 0) {
        return std::nullopt;
    }
    return std::make_pair(static_cast<int>(width), static_cast<int>(height));
}

std::optional<std::array<float, 4>> parseConstant(const std::string& value) {
    const auto parts = split(value, '=');
    if (parts.size() != 2 || trim(parts[0]).empty()) {
        return std::nullopt;
    }
    const auto numberParts = split(parts[1], ',');
    if (numberParts.empty() || numberParts.size() > 4) {
        return std::nullopt;
    }
    std::array<float, 4> result{{0.0f, 0.0f, 0.0f, 0.0f}};
    for (std::size_t i = 0; i < numberParts.size(); ++i) {
        const auto item = trim(numberParts[i]);
        char* end = nullptr;
        errno = 0;
        const float number = std::strtof(item.c_str(), &end);
        if (errno != 0 || end == item.c_str() || *end != '\0' || item.empty()) {
            return std::nullopt;
        }
        result[i] = number;
    }
    return result;
}

std::optional<int> parseInt(const std::string& value) {
    char* end = nullptr;
    errno = 0;
    const long result = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || result < 1 ||
        result > 1000000) {
        return std::nullopt;
    }
    return static_cast<int>(result);
}

}  // namespace

std::string helpText() {
    return R"(material-verify - deterministic offline GLSL material validator

Usage:
  material-verify --material <material.json> [options]

Required:
  --material <path>          Material metadata JSON file.

Options:
  --variants <list>          Comma-separated metadata variant names, "all",
                             or an ad-hoc macro expression such as FOO+BAR.
                             Defaults to metadata default_variants.
  --cache-dir <dir>          Reuse linked GL program binaries. The cache key
                             includes shader hashes, macros and dependency data.
  --output <path>            JSON report path. Defaults to <material>.report.json.
  --texture-size <WxH>       Override every procedural dependency texture size.
  --const <name=v0[,v1..]>   Override a UBO vec4 (up to four components).
                             May be repeated. Omitted components are zero.
  --dump-pixels <dir>        Write raw attachment readbacks next to the report.
  --max-diffs <n>            Maximum difference records per rule (default 4096).
  --deterministic            Set duration_ms to zero and omit unstable timing.
  --help                     Show this help.

Exit codes:
  0 success, 2 input, 3 shader compile, 4 shader link, 5 framebuffer,
  6 attachment contract, 7 validation mismatch, 8 GL/runtime, 9 cache,
  10 report output.
)";
}

CliResult parseCommandLine(int argc, char** argv) {
    CommandLine command;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string inlineValue;
        const auto equals = arg.find('=');
        if (equals != std::string::npos) {
            inlineValue = arg.substr(equals + 1);
            arg = arg.substr(0, equals);
        }

        const auto value = [&](const char* option) -> std::optional<std::string> {
            (void)option;
            if (!inlineValue.empty()) {
                return inlineValue;
            }
            if (i + 1 >= argc) {
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };

        if (arg == "--help" || arg == "-h") {
            command.showHelp = true;
            continue;
        }
        if (arg == "--material") {
            const auto v = value("--material");
            if (!v) {
                return {std::nullopt,
                        makeError(ExitCode::Input, "CLI_ARGUMENT_MISSING", "input",
                                  "--material requires a value")};
            }
            command.materialPath = *v;
        } else if (arg == "--variants") {
            const auto v = value("--variants");
            if (!v) {
                return {std::nullopt,
                        makeError(ExitCode::Input, "CLI_ARGUMENT_MISSING", "input",
                                  "--variants requires a value")};
            }
            command.variants = *v;
        } else if (arg == "--cache-dir") {
            const auto v = value("--cache-dir");
            if (!v) {
                return {std::nullopt,
                        makeError(ExitCode::Input, "CLI_ARGUMENT_MISSING", "input",
                                  "--cache-dir requires a value")};
            }
            command.cacheDir = *v;
        } else if (arg == "--output") {
            const auto v = value("--output");
            if (!v) {
                return {std::nullopt,
                        makeError(ExitCode::Input, "CLI_ARGUMENT_MISSING", "input",
                                  "--output requires a value")};
            }
            command.outputPath = *v;
        } else if (arg == "--dump-pixels") {
            const auto v = value("--dump-pixels");
            if (!v) {
                return {std::nullopt,
                        makeError(ExitCode::Input, "CLI_ARGUMENT_MISSING", "input",
                                  "--dump-pixels requires a value")};
            }
            command.pixelDir = *v;
        } else if (arg == "--texture-size") {
            const auto v = value("--texture-size");
            if (!v) {
                return {std::nullopt,
                        makeError(ExitCode::Input, "CLI_ARGUMENT_MISSING", "input",
                                  "--texture-size requires a value")};
            }
            const auto size = parseSize(*v);
            if (!size) {
                return {std::nullopt,
                        makeError(ExitCode::Input, "CLI_ARGUMENT_INVALID", "input",
                                  "invalid --texture-size, expected WxH with positive values: " +
                                      *v)};
            }
            command.textureSize = size;
        } else if (arg == "--const") {
            const auto v = value("--const");
            if (!v) {
                return {std::nullopt,
                        makeError(ExitCode::Input, "CLI_ARGUMENT_MISSING", "input",
                                  "--const requires name=value[,value...]")};
            }
            const auto nameEnd = v->find('=');
            const auto number = parseConstant(*v);
            if (!number || nameEnd == std::string::npos) {
                return {std::nullopt,
                        makeError(ExitCode::Input, "CLI_ARGUMENT_INVALID", "input",
                                  "invalid --const expression: " + *v)};
            }
            command.constantOverrides[trim(v->substr(0, nameEnd))] = *number;
        } else if (arg == "--max-diffs") {
            const auto v = value("--max-diffs");
            if (!v) {
                return {std::nullopt,
                        makeError(ExitCode::Input, "CLI_ARGUMENT_MISSING", "input",
                                  "--max-diffs requires a value")};
            }
            const auto count = parseInt(*v);
            if (!count) {
                return {std::nullopt,
                        makeError(ExitCode::Input, "CLI_ARGUMENT_INVALID", "input",
                                  "invalid --max-diffs value: " + *v)};
            }
            command.maxDiffs = *count;
        } else if (arg == "--deterministic") {
            command.deterministic = true;
        } else {
            return {std::nullopt,
                    makeError(ExitCode::Input, "CLI_ARGUMENT_UNKNOWN", "input",
                              "unknown argument: " + arg)};
        }
    }

    if (!command.showHelp && command.materialPath.empty()) {
        return {std::nullopt,
                makeError(ExitCode::Input, "CLI_MATERIAL_REQUIRED", "input",
                          "--material is required")};
    }
    return {command, std::nullopt};
}

}  // namespace mv
