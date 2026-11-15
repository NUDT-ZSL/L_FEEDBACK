#include "automix/config.h"
#include "automix/engine.h"
#include "automix/error.h"
#include "automix/logger.h"
#include "automix/stereo.h"
#include "automix/wav_io.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <cwchar>
#endif

namespace {

using automix::AppError;
using automix::ErrorCode;

struct CommandLine {
    std::filesystem::path configPath;
    std::optional<std::filesystem::path> logDir;
    std::optional<std::string> logFile;
    bool help = false;
    bool version = false;
};

std::string usage() {
    return "Usage: automix --config <config.json> [--log-dir <dir>] "
           "[--log-file <name>]\n"
           "       automix --help | --version\n";
}

CommandLine parseCommandLine(int argc, char** argv) {
    CommandLine command;
    for (int index = 1; index < argc; ++index) {
        std::string argument = argv[index];
        auto requireValue = [&](const std::string& option) -> std::string {
            if (index + 1 >= argc) {
                throw AppError(ErrorCode::Usage, "missing value after " + option);
            }
            return argv[++index];
        };
        if (argument == "--help" || argument == "-h") {
            command.help = true;
        } else if (argument == "--version") {
            command.version = true;
        } else if (argument == "--config") {
            command.configPath = std::filesystem::u8path(requireValue(argument));
        } else if (argument.rfind("--config=", 0) == 0) {
            command.configPath =
                std::filesystem::u8path(argument.substr(std::string("--config=").size()));
        } else if (argument == "--log-dir") {
            command.logDir = std::filesystem::u8path(requireValue(argument));
        } else if (argument.rfind("--log-dir=", 0) == 0) {
            command.logDir =
                std::filesystem::u8path(argument.substr(std::string("--log-dir=").size()));
        } else if (argument == "--log-file") {
            command.logFile = requireValue(argument);
        } else if (argument.rfind("--log-file=", 0) == 0) {
            command.logFile = argument.substr(std::string("--log-file=").size());
        } else if (!argument.empty() && argument[0] != '-' && command.configPath.empty()) {
            command.configPath = std::filesystem::u8path(argument);
        } else {
            throw AppError(ErrorCode::Usage, "unknown argument: " + argument);
        }
    }
    if (command.help || command.version) {
        return command;
    }
    if (command.configPath.empty()) {
        throw AppError(ErrorCode::Usage, "a JSON configuration file is required");
    }
    command.configPath =
        std::filesystem::absolute(command.configPath).lexically_normal();
    return command;
}

void writeTextFile(const std::filesystem::path& path, const std::string& text) {
    std::error_code ec;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) {
            throw AppError(ErrorCode::OutputIo,
                           "cannot create report directory: " + ec.message());
        }
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw AppError(ErrorCode::OutputIo,
                       "cannot create report file: " + automix::pathToUtf8(path));
    }
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.flush();
    if (!output) {
        throw AppError(ErrorCode::OutputIo,
                       "failed while writing report file: " + automix::pathToUtf8(path));
    }
}

void replaceFile(const std::filesystem::path& temporary,
                 const std::filesystem::path& destination) {
    std::error_code ec;
    std::filesystem::remove(destination, ec);
    if (ec) {
        throw AppError(ErrorCode::OutputIo,
                       "cannot replace output file '" + automix::pathToUtf8(destination) +
                           "': " + ec.message());
    }
    ec.clear();
    std::filesystem::rename(temporary, destination, ec);
    if (ec) {
        throw AppError(ErrorCode::OutputIo,
                       "cannot commit output file '" + automix::pathToUtf8(destination) +
                           "': " + ec.message());
    }
}

void removeIfPresent(const std::filesystem::path& path) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

}  // namespace

int runAutomix(int argc, char** argv) {
    std::unique_ptr<automix::Logger> logger;
    std::filesystem::path temporaryWav;
    std::filesystem::path temporaryReport;

    try {
        const CommandLine command = parseCommandLine(argc, argv);
        if (command.help) {
            std::cout << usage() << "\nConfiguration schema:\n"
                      << automix::configSchemaSummary() << '\n';
            return 0;
        }
        if (command.version) {
            std::cout << "automix 1.0.0\n";
            return 0;
        }

        std::filesystem::path bootstrapLogDir;
        if (command.logDir.has_value()) {
            bootstrapLogDir =
                std::filesystem::absolute(*command.logDir).lexically_normal();
        } else {
            bootstrapLogDir = command.configPath.parent_path() / "logs";
        }
        const std::string bootstrapLogFile =
            command.logFile.value_or("automix.log");
        logger = std::make_unique<automix::Logger>(
            bootstrapLogDir, bootstrapLogFile);

        std::ostringstream startup;
        startup << "startup";
        for (int index = 0; index < argc; ++index) {
            startup << " arg" << index << "=" << argv[index];
        }
        logger->info(startup.str());

        automix::ConfigOverrides overrides;
        overrides.logDir = command.logDir;
        overrides.logFile = command.logFile;
        const automix::AppConfig config =
            automix::parseConfigFile(command.configPath, overrides);

        if (config.logDir != logger->directory() || config.logFile != bootstrapLogFile) {
            logger = std::make_unique<automix::Logger>(config.logDir, config.logFile);
            logger->info("logging switched to configured destination");
        }
        logger->info("configuration validated: " + automix::pathToUtf8(config.configPath));

        automix::RenderResult renderResult =
            automix::renderMix(config, *logger);

        temporaryWav = config.output;
        temporaryWav += ".automix.part";
        temporaryReport = config.report;
        temporaryReport += ".automix.part";
        removeIfPresent(temporaryWav);
        removeIfPresent(temporaryReport);

        const auto writeStart = std::chrono::steady_clock::now();
        automix::writeFloatWav(temporaryWav, renderResult.audio);
        const auto writeEnd = std::chrono::steady_clock::now();
        const double writeMs =
            std::chrono::duration<double, std::milli>(writeEnd - writeStart).count();

        const std::string report =
            automix::buildReportJson(config, renderResult, writeMs);
        writeTextFile(temporaryReport, report);

        replaceFile(temporaryWav, config.output);
        try {
            replaceFile(temporaryReport, config.report);
        } catch (...) {
            removeIfPresent(config.output);
            throw;
        }

        std::ostringstream completion;
        completion << "output committed output=" << automix::pathToUtf8(config.output)
                   << " report=" << automix::pathToUtf8(config.report)
                   << " write_ms=" << writeMs;
        logger->info(completion.str());

        std::cout << "wrote " << automix::pathToUtf8(config.output) << '\n'
                  << "wrote " << automix::pathToUtf8(config.report) << '\n';
        return 0;
    } catch (const AppError& error) {
        removeIfPresent(temporaryWav);
        removeIfPresent(temporaryReport);
        if (logger) {
            logger->error(std::string(automix::errorCodeName(error.code())) + ": " +
                          error.what());
        }
        std::cerr << automix::errorCodeName(error.code()) << ": " << error.what()
                  << '\n';
        return automix::exitCodeFor(error.code());
    } catch (const std::exception& error) {
        removeIfPresent(temporaryWav);
        removeIfPresent(temporaryReport);
        if (logger) {
            logger->error(std::string("E_INTERNAL: ") + error.what());
        }
        std::cerr << "E_INTERNAL: " << error.what() << '\n';
        return automix::exitCodeFor(ErrorCode::Internal);
    }
}

#ifdef _WIN32
int wmain(int argc, wchar_t** wideArgv) {
    std::vector<std::string> arguments;
    std::vector<char*> narrowArgv;
    arguments.reserve(static_cast<std::size_t>(argc));
    narrowArgv.reserve(static_cast<std::size_t>(argc) + 1);
    for (int index = 0; index < argc; ++index) {
        const wchar_t* wide = wideArgv[index];
        const int length = static_cast<int>(std::wcslen(wide));
        const int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide,
                                              length, nullptr, 0, nullptr, nullptr);
        std::string utf8;
        if (bytes > 0) {
            utf8.resize(static_cast<std::size_t>(bytes));
            WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide, length,
                                utf8.data(), bytes, nullptr, nullptr);
        }
        arguments.push_back(std::move(utf8));
    }
    for (std::string& argument : arguments) {
        narrowArgv.push_back(argument.data());
    }
    narrowArgv.push_back(nullptr);
    return runAutomix(argc, narrowArgv.data());
}
#else
int main(int argc, char** argv) {
    return runAutomix(argc, argv);
}
#endif
