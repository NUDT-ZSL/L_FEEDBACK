#include "errors.hpp"
#include "renderer.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

void printUsage() {
    std::cout
        << "volume_renderer --scene <scene.json> [options]\n"
        << "\n"
        << "Options:\n"
        << "  --out <png>                 Override main PNG output path\n"
        << "  --stats <json>              Override statistics JSON path\n"
        << "  --reference-out <png>       Override reference PNG path\n"
        << "  --shader-dir <dir>          Directory containing compiled .spv files\n"
        << "  --recovery-dir <dir>        Directory for fixed-name recovery files\n"
        << "  --inject shader|alloc|desc|device-lost\n"
        << "  --volume-step <meters>      Override JSON step (0.01..1.0)\n"
        << "  --volume-max-steps <n>      Override JSON max samples (8..2048)\n"
        << "  --frames <n>                Number of identical main frames (default 2)\n"
        << "  --no-reference              Skip the fixed reference frame\n"
        << "  --clear-recovery            Remove recovery files after success\n"
        << "  --validation on|off         Enable Vulkan validation layers\n"
        << "  --help                      Show this help\n";
}

std::string nextValue(int argc, char** argv, int& index, const std::string& option) {
    if (index + 1 >= argc) {
        vlr::fail(vlr::ErrorCode::Usage, option + " requires a value");
    }
    return argv[++index];
}

int parseInteger(const std::string& text, const std::string& option) {
    try {
        std::size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("trailing");
        }
        return value;
    } catch (const std::exception&) {
        vlr::fail(vlr::ErrorCode::Usage, option + " requires an integer");
    }
}

float parseFloat(const std::string& text, const std::string& option) {
    try {
        std::size_t consumed = 0;
        const float value = std::stof(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("trailing");
        }
        return value;
    } catch (const std::exception&) {
        vlr::fail(vlr::ErrorCode::Usage, option + " requires a number");
    }
}

vlr::Injection parseInjection(const std::string& value) {
    if (value == "shader") {
        return vlr::Injection::Shader;
    }
    if (value == "alloc") {
        return vlr::Injection::Allocation;
    }
    if (value == "desc") {
        return vlr::Injection::Descriptor;
    }
    if (value == "device-lost") {
        return vlr::Injection::DeviceLost;
    }
    vlr::fail(vlr::ErrorCode::Usage,
              "--inject must be one of shader, alloc, desc, device-lost");
}

vlr::RendererOptions parseOptions(int argc, char** argv) {
    vlr::RendererOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            printUsage();
            std::exit(0);
        } else if (argument == "--scene") {
            options.scenePath = nextValue(argc, argv, i, argument);
        } else if (argument == "--out") {
            options.outputPng = nextValue(argc, argv, i, argument);
        } else if (argument == "--stats") {
            options.statsJson = nextValue(argc, argv, i, argument);
        } else if (argument == "--reference-out") {
            options.referencePng = nextValue(argc, argv, i, argument);
        } else if (argument == "--shader-dir") {
            options.shaderDirectory = nextValue(argc, argv, i, argument);
        } else if (argument == "--recovery-dir") {
            options.recoveryDirectory = nextValue(argc, argv, i, argument);
        } else if (argument == "--inject") {
            options.injection = parseInjection(nextValue(argc, argv, i, argument));
        } else if (argument == "--volume-step") {
            options.volumeStepOverride =
                parseFloat(nextValue(argc, argv, i, argument), argument);
        } else if (argument == "--volume-max-steps") {
            options.volumeMaxStepsOverride =
                parseInteger(nextValue(argc, argv, i, argument), argument);
        } else if (argument == "--frames") {
            options.frameCount =
                parseInteger(nextValue(argc, argv, i, argument), argument);
        } else if (argument == "--no-reference") {
            options.noReference = true;
        } else if (argument == "--clear-recovery") {
            options.clearRecovery = true;
        } else if (argument == "--validation") {
            const std::string value = nextValue(argc, argv, i, argument);
            if (value == "on") {
                options.enableValidation = true;
            } else if (value == "off") {
                options.enableValidation = false;
            } else {
                vlr::fail(vlr::ErrorCode::Usage,
                          "--validation must be 'on' or 'off'");
            }
        } else {
            vlr::fail(vlr::ErrorCode::Usage, "unknown argument: " + argument);
        }
    }
    if (options.scenePath.empty()) {
        vlr::fail(vlr::ErrorCode::Usage, "--scene is required");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    vlr::RendererOptions options;
    try {
        options = parseOptions(argc, argv);
    } catch (const vlr::AppError& error) {
        std::cerr << vlr::errorCodeName(error.code()) << ": " << error.what() << "\n";
        return error.exitCode();
    }

    int exitCode = 0;
    std::string businessCode;
    std::string message;
    int resources = 0;
    long long allocations = 0;
    bool deviceLost = false;
    try {
        {
            vlr::Renderer renderer(options);
            try {
                renderer.run();
            } catch (const vlr::AppError& error) {
                exitCode = error.exitCode();
                businessCode = vlr::errorCodeName(error.code());
                message = error.what();
            } catch (const std::exception& error) {
                exitCode = 1;
                businessCode = "E_INTERNAL";
                message = error.what();
            }
            renderer.cleanup();
            resources = renderer.liveResourceCount();
            allocations = renderer.liveAllocatorCount();
            deviceLost = renderer.deviceWasLost();
        }
    } catch (const std::exception& error) {
        exitCode = 1;
        businessCode = "E_INTERNAL";
        message = error.what();
    }

    std::cerr << "cleanup: resources_live=" << resources
              << " allocator_live=" << allocations
              << " device_lost=" << (deviceLost ? "true" : "false") << "\n";
    if (resources != 0) {
        std::cerr << "E_RECOVERY: custom resource counter is not zero after cleanup\n";
        if (exitCode == 0) {
            exitCode = 12;
        }
    }
    if (exitCode != 0) {
        std::cerr << businessCode << ": " << message << "\n";
    }
    return exitCode;
}
