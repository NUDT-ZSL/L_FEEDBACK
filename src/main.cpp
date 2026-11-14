#include "app/Application.hpp"

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::uint64_t parseUnsigned(const char* text, const char* option) {
    std::uint64_t value = 0;
    const std::string input(text);
    const auto result = std::from_chars(input.data(), input.data() + input.size(),
                                        value);
    if (result.ec != std::errc{} || result.ptr != input.data() + input.size()) {
        throw std::runtime_error(std::string("Invalid value for ") + option + ": " +
                                 text);
    }
    return value;
}

void printUsage() {
    std::cout
        << "IndoorDeferredRenderer\n"
        << "  --seed N          deterministic scene seed (default 1337)\n"
        << "  --lights N        point-light count (default 256)\n"
        << "  --width N         initial window width (default 1280)\n"
        << "  --height N        initial window height (default 720)\n"
        << "  --cache N         shadow cache capacity (default = light count)\n"
        << "  --shadow-size N   optional fixed cube shadow-map size\n"
        << "  --test-context-loss  render one frame, lose/recreate the real\n"
        << "                       GLFW context, compare pixels and exit\n"
        << "  --help\n\n"
        << "Keys: 1..5 material presets, T cycle preset, F1..F7 debug views,\n"
        << "      Tab cycle view,\n"
        << "      P pause animation, M step animation, X delete a light,\n"
        << "      R deterministic scene switch, L simulate context loss.\n";
}

} // namespace

int main(int argc, char** argv) {
    indoor::LaunchOptions options;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help" || argument == "-h") {
                printUsage();
                return 0;
            }
            if (argument == "--test-context-loss") {
                options.testContextLoss = true;
                continue;
            }
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + argument);
            }
            const char* value = argv[++i];
            if (argument == "--seed") {
                options.seed = parseUnsigned(value, "--seed");
            } else if (argument == "--lights") {
                options.lightCount = static_cast<std::uint32_t>(
                    parseUnsigned(value, "--lights"));
                if (options.lightCount == 0) {
                    throw std::runtime_error("--lights must be at least 1");
                }
            } else if (argument == "--width") {
                options.width = static_cast<int>(parseUnsigned(value, "--width"));
            } else if (argument == "--height") {
                options.height = static_cast<int>(parseUnsigned(value, "--height"));
            } else if (argument == "--cache") {
                options.shadowCacheCapacity = static_cast<std::uint32_t>(
                    parseUnsigned(value, "--cache"));
            } else if (argument == "--shadow-size") {
                options.shadowSize = static_cast<int>(
                    parseUnsigned(value, "--shadow-size"));
            } else {
                throw std::runtime_error("Unknown option: " + argument);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n\n";
        printUsage();
        return 2;
    }

    try {
        indoor::Application application(options);
        return application.run();
    } catch (const std::exception& error) {
        std::cerr << "Fatal: " << error.what() << '\n';
        return 1;
    }
}
