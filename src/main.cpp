#include "pipeline.hpp"

#include <csignal>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {

void signalHandler(int) {
    rigskin::requestInterrupt();
}

#ifdef _WIN32
BOOL WINAPI consoleHandler(DWORD controlType) {
    if (controlType == CTRL_C_EVENT || controlType == CTRL_BREAK_EVENT ||
        controlType == CTRL_CLOSE_EVENT || controlType == CTRL_LOGOFF_EVENT ||
        controlType == CTRL_SHUTDOWN_EVENT) {
        rigskin::requestInterrupt();
        return TRUE;
    }
    return FALSE;
}
#endif

void installInterruptHandlers() {
    std::signal(SIGINT, signalHandler);
#ifdef SIGTERM
    std::signal(SIGTERM, signalHandler);
#endif
#ifdef SIGBREAK
    std::signal(SIGBREAK, signalHandler);
#endif
#ifdef _WIN32
    SetConsoleCtrlHandler(consoleHandler, TRUE);
#endif
}

int parsePositiveInt(const std::string& text, const std::string& option) {
    std::size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(text, &consumed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + option + ": " + text);
    }
    if (consumed != text.size() || value <= 0) {
        throw std::runtime_error("invalid value for " + option + ": " + text);
    }
    return value;
}

int parseNonNegativeInt(const std::string& text, const std::string& option) {
    std::size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(text, &consumed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + option + ": " + text);
    }
    if (consumed != text.size() || value < 0) {
        throw std::runtime_error("invalid value for " + option + ": " + text);
    }
    return value;
}

void printUsage() {
    std::cout
        << "rigskin - deterministic offline skeletal animation preview\n\n"
        << "Usage:\n"
        << "  rigskin --threads N --snapshot-dir DIR [options]\n\n"
        << "Required:\n"
        << "  --threads N            worker count (1..4096)\n"
        << "  --snapshot-dir DIR     snapshot/output directory\n\n"
        << "Options:\n"
        << "  --frames N             number of timeline frames (default 24)\n"
        << "  --width N              PPM width (default 640)\n"
        << "  --height N             PPM height (default 480)\n"
        << "  --sample NAME          built-in sample; only 'preview' (default)\n"
        << "  --commit-delay-ms N    optional post-commit fault-injection delay\n"
        << "  --help                  show this help\n";
}

}  // namespace

int main(int argc, char** argv) {
    rigskin::RunOptions options;
    options.config.frameCount = 24;
    options.config.width = 640;
    options.config.height = 480;
    options.threads = 1;
    bool sawThreads = false;
    bool sawSnapshotDirectory = false;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            auto requireValue = [&](const char* option) -> std::string {
                if (index + 1 >= argc) {
                    throw std::runtime_error(
                        std::string("missing value for ") + option);
                }
                return argv[++index];
            };

            if (argument == "--help" || argument == "-h") {
                printUsage();
                return 0;
            }
            if (argument == "--threads") {
                options.threads =
                    parsePositiveInt(requireValue("--threads"), "--threads");
                sawThreads = true;
            } else if (argument == "--snapshot-dir") {
                options.snapshotDirectory = requireValue("--snapshot-dir");
                sawSnapshotDirectory = true;
            } else if (argument == "--frames") {
                options.config.frameCount =
                    parsePositiveInt(requireValue("--frames"), "--frames");
            } else if (argument == "--width") {
                options.config.width =
                    parsePositiveInt(requireValue("--width"), "--width");
            } else if (argument == "--height") {
                options.config.height =
                    parsePositiveInt(requireValue("--height"), "--height");
            } else if (argument == "--sample") {
                const std::string sample =
                    requireValue("--sample");
                if (sample != "preview") {
                    throw std::runtime_error(
                        "unknown sample '" + sample + "'; expected 'preview'");
                }
            } else if (argument == "--commit-delay-ms") {
                options.commitDelayMilliseconds = parseNonNegativeInt(
                    requireValue("--commit-delay-ms"), "--commit-delay-ms");
            } else {
                throw std::runtime_error("unknown argument: " + argument);
            }
        }

        if (!sawThreads || !sawSnapshotDirectory) {
            throw std::runtime_error(
                "--threads and --snapshot-dir are required");
        }
    } catch (const std::exception& exception) {
        std::cerr << "rigskin: " << exception.what() << "\n\n";
        printUsage();
        return 2;
    }

    installInterruptHandlers();
    const rigskin::RunResult result = rigskin::runPipeline(options);

    std::cout << "snapshot=" << options.snapshotDirectory.string() << '\n';
    std::cout << "status=";
    switch (result.status) {
        case rigskin::RunStatus::Complete:
            std::cout << "complete\n";
            break;
        case rigskin::RunStatus::Interrupted:
            std::cout << "interrupted\n";
            break;
        case rigskin::RunStatus::Failed:
            std::cout << "failed\n";
            break;
    }
    std::cout << "frames_success=" << result.framesSuccess << '\n';
    std::cout << "frames_reused=" << result.framesReused << '\n';
    std::cout << "frames_recomputed=" << result.framesRecomputed << '\n';
    std::cout << "error_codes=";
    if (result.errorCodes.empty()) {
        std::cout << "none\n";
    } else {
        for (std::size_t index = 0; index < result.errorCodes.size(); ++index) {
            if (index != 0) {
                std::cout << ',';
            }
            std::cout << result.errorCodes[index];
        }
        std::cout << '\n';
    }
    if (!result.message.empty()) {
        std::cerr << "rigskin: " << result.message << '\n';
    }
    return result.exitCode;
}
