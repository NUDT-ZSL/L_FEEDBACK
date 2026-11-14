#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <locale>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "error.h"
#include "exporter.h"
#include "sample_gen.h"

namespace {

constexpr const char* kUsage =
    "voice-replay 1.0.0\n"
    "Usage:\n"
    "  voice-replay export --input <dir> --output <dir> [--resume]\n"
    "  voice-replay --input <dir> --output <dir> [--resume]\n"
    "  voice-replay generate --output <dir>\n"
    "\n"
    "Options:\n"
    "  --input <dir>                 Input directory with events.json, metadata and raw files\n"
    "  --output <dir>                Output directory\n"
    "  --resume                      Continue from .voice-replay-state.json\n"
    "  --abort-after-blocks <n>      Test hook: exit 75 after n completed blocks\n"
    "  -h, --help                    Show this help\n"
    "\n"
    "Environment test hook:\n"
    "  VOICE_REPLAY_ABORT_AFTER_BLOCKS=<n>\n";

std::filesystem::path path_from_arg(const std::string& arg) {
    return std::filesystem::u8path(arg);
}

int parse_nonnegative(const std::string& text, const std::string& option) {
    if (text.empty()) {
        throw vr::AppError(vr::ErrorCode::Usage,
                           "missing value for " + option);
    }
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (end == nullptr || *end != '\0' || value < 0 ||
        value > 1000000000L) {
        throw vr::AppError(vr::ErrorCode::Usage,
                           "invalid non-negative integer for " + option + ": " + text);
    }
    return static_cast<int>(value);
}

}  // namespace

int main(int argc, char** argv) {
    std::locale::global(std::locale::classic());
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    try {
        if (argc <= 1) {
            throw vr::AppError(vr::ErrorCode::Usage,
                               "missing --input and --output arguments");
        }
        std::string command;
        int index = 1;
        if (std::string(argv[1]) == "export" ||
            std::string(argv[1]) == "generate") {
            command = argv[1];
            index = 2;
        }

        std::filesystem::path input;
        std::filesystem::path output;
        bool has_input = false;
        bool has_output = false;
        bool resume = false;
        int abort_after_blocks = -1;

        for (; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "-h" || arg == "--help") {
                std::cerr << kUsage;
                return 0;
            }
            if (arg == "--input") {
                if (++index >= argc) {
                    throw vr::AppError(vr::ErrorCode::Usage,
                                       "missing value for --input");
                }
                input = path_from_arg(argv[index]);
                has_input = true;
            } else if (arg == "--output") {
                if (++index >= argc) {
                    throw vr::AppError(vr::ErrorCode::Usage,
                                       "missing value for --output");
                }
                output = path_from_arg(argv[index]);
                has_output = true;
            } else if (arg == "--resume") {
                resume = true;
            } else if (arg == "--abort-after-blocks") {
                if (++index >= argc) {
                    throw vr::AppError(vr::ErrorCode::Usage,
                                       "missing value for --abort-after-blocks");
                }
                abort_after_blocks = parse_nonnegative(
                    argv[index], "--abort-after-blocks");
            } else if (!arg.empty() && arg[0] == '-') {
                throw vr::AppError(vr::ErrorCode::Usage,
                                   "unknown option: " + arg);
            } else {
                throw vr::AppError(vr::ErrorCode::Usage,
                                   "unexpected positional argument: " + arg);
            }
        }

        if (command == "generate") {
            if (!has_output) {
                throw vr::AppError(vr::ErrorCode::Usage,
                                   "generate requires --output <dir>");
            }
            vr::generate_samples(output);
            std::cerr << "voice-replay: generated deterministic samples in "
                      << output.string() << "\n";
            return 0;
        }

        if (!has_input || !has_output) {
            throw vr::AppError(vr::ErrorCode::Usage,
                               "export requires --input <dir> and --output <dir>");
        }
        vr::run_export(input, output, resume, abort_after_blocks);
        return 0;
    } catch (const vr::AppError& error) {
        std::cerr << "voice-replay: error=" << vr::error_code_name(error.code())
                  << " exit=" << vr::error_exit_code(error.code())
                  << " message=" << error.what() << "\n";
        return vr::error_exit_code(error.code());
    } catch (const std::exception& error) {
        std::cerr << "voice-replay: error="
                  << vr::error_code_name(vr::ErrorCode::Internal)
                  << " exit=" << vr::error_exit_code(vr::ErrorCode::Internal)
                  << " message=" << error.what() << "\n";
        return vr::error_exit_code(vr::ErrorCode::Internal);
    }
}

