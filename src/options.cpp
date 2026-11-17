#include "options.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace cv {
namespace {

bool parseInteger(const std::string& text, long& value) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    value = parsed;
    return true;
}

bool parseFloat(const std::string& text, float& value) {
    if (text.empty()) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const float parsed = std::strtof(text.c_str(), &end);
    if (errno != 0 || end == text.c_str() || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

std::string normalizeKey(const std::string& raw) {
    if (raw.empty()) {
        return {};
    }
    std::string key = raw;
    if (key.size() >= 2 && key[0] == '\'' && key.back() == '\'') {
        key = key.substr(1, key.size() - 2);
    }
    return key;
}

}  // namespace

AppOptions parseOptions(int argc, char** argv) {
    AppOptions options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string value;
        const size_t equals = arg.find('=');
        if (equals != std::string::npos) {
            value = arg.substr(equals + 1);
            arg = arg.substr(0, equals);
        }
        auto requireValue = [&]() {
            if (!value.empty()) {
                return true;
            }
            if (i + 1 < argc) {
                value = argv[++i];
                return true;
            }
            return false;
        };
        auto readInt = [&](int& target, int minValue, int maxValue,
                           const char* optionName) {
            if (!requireValue()) {
                options.valid = false;
                options.error = std::string("missing value for ") + optionName;
                return false;
            }
            long parsed = 0;
            if (!parseInteger(value, parsed) || parsed < minValue || parsed > maxValue) {
                options.valid = false;
                options.error = std::string("invalid value for ") + optionName + ": " + value;
                return false;
            }
            target = static_cast<int>(parsed);
            return true;
        };
        auto readU32 = [&](uint32_t& target, uint32_t maxValue, const char* optionName) {
            if (!requireValue()) {
                options.valid = false;
                options.error = std::string("missing value for ") + optionName;
                return false;
            }
            long parsed = 0;
            if (!parseInteger(value, parsed) || parsed < 0 ||
                static_cast<unsigned long>(parsed) > maxValue) {
                options.valid = false;
                options.error = std::string("invalid value for ") + optionName + ": " + value;
                return false;
            }
            target = static_cast<uint32_t>(parsed);
            return true;
        };
        auto readFloat = [&](float& target, float minValue, float maxValue,
                             bool& wasSet, const char* optionName) {
            if (!requireValue()) {
                options.valid = false;
                options.error = std::string("missing value for ") + optionName;
                return false;
            }
            float parsed = 0.0f;
            if (!parseFloat(value, parsed) || parsed < minValue || parsed > maxValue) {
                options.valid = false;
                options.error = std::string("invalid value for ") + optionName + ": " + value;
                return false;
            }
            target = parsed;
            wasSet = true;
            return true;
        };
        auto readString = [&](std::string& target, const char* optionName) {
            if (!requireValue()) {
                options.valid = false;
                options.error = std::string("missing value for ") + optionName;
                return false;
            }
            target = value;
            return true;
        };
        auto readKey = [&](char& target, const char* optionName) {
            if (!requireValue()) {
                options.valid = false;
                options.error = std::string("missing value for ") + optionName;
                return false;
            }
            const std::string key = normalizeKey(value);
            if (key.size() != 1 || static_cast<unsigned char>(key[0]) < 0x20) {
                options.valid = false;
                options.error = std::string("hotkey must be one printable character: ") +
                                optionName;
                return false;
            }
            target = static_cast<char>(std::toupper(static_cast<unsigned char>(key[0])));
            return true;
        };

        if (arg == "--help" || arg == "-h") {
            options.helpRequested = true;
        } else if (arg == "--width") {
            if (!readInt(options.width, 64, 4096, "--width")) {
                return options;
            }
        } else if (arg == "--height") {
            if (!readInt(options.height, 64, 4096, "--height")) {
                return options;
            }
        } else if (arg == "--tick") {
            if (!readU32(options.startTick, 100000000u, "--tick")) {
                return options;
            }
        } else if (arg == "--cascade-res") {
            if (!readInt(options.cascadeResolution, 64, 2048, "--cascade-res")) {
                return options;
            }
        } else if (arg == "--cube-res") {
            if (!readInt(options.pointShadowResolution, 32, 1024, "--cube-res")) {
                return options;
            }
        } else if (arg == "--yaw") {
            if (!readFloat(options.yawDegrees, -360.0f, 360.0f, options.hasYaw, "--yaw")) {
                return options;
            }
        } else if (arg == "--pitch") {
            if (!readFloat(options.pitchDegrees, -65.0f, 65.0f, options.hasPitch,
                           "--pitch")) {
                return options;
            }
        } else if (arg == "--distance") {
            if (!readFloat(options.cameraDistance, 2.5f, 18.0f,
                           options.hasCameraDistance, "--distance")) {
                return options;
            }
        } else if (arg == "--ppm" || arg == "--ppm-path") {
            if (!readString(options.ppmPath, "--ppm")) {
                return options;
            }
        } else if (arg == "--bmp" || arg == "--bmp-path") {
            if (!readString(options.bmpPath, "--bmp")) {
                return options;
            }
        } else if (arg == "--ppm-key") {
            if (!readKey(options.ppmKey, "--ppm-key")) {
                return options;
            }
        } else if (arg == "--bmp-key") {
            if (!readKey(options.bmpKey, "--bmp-key")) {
                return options;
            }
        } else if (arg == "--no-shadows") {
            options.shadows = false;
        } else if (arg == "--paused") {
            options.startPaused = true;
        } else if (arg == "--wireframe") {
            options.wireframe = true;
        } else {
            options.valid = false;
            options.error = "unknown option: " + arg;
            return options;
        }
    }
    return options;
}

void printUsage(const char* executableName) {
    std::cout
        << "Usage: " << executableName << " [options]\n"
        << "  --width N             Render width (64..4096, default 960)\n"
        << "  --height N            Render height (64..4096, default 640)\n"
        << "  --tick N              Initial fixed 30 Hz animation tick\n"
        << "  --cascade-res N       Directional cascade map size (default 1024)\n"
        << "  --cube-res N          Point cube-face map size (default 256)\n"
        << "  --yaw DEG             Camera orbit yaw in degrees\n"
        << "  --pitch DEG           Camera pitch in degrees (-65..65)\n"
        << "  --distance N          Camera orbit distance (2.5..18)\n"
        << "  --no-shadows          Disable shadow-map generation and sampling\n"
        << "  --ppm PATH            PPM(P6) export path (default frame.ppm)\n"
        << "  --bmp PATH            BMP(24-bit) export path (default frame.bmp)\n"
        << "  --ppm-key C           Configurable PPM hotkey (default P)\n"
        << "  --bmp-key C           Configurable BMP hotkey (default B)\n"
        << "  --paused              Start paused\n"
        << "  --wireframe           Start in wireframe mode\n"
        << "  --help                Show this help\n\n"
        << "Interactive keys: Space=pause/resume, N=step one 1/30 s tick,\n"
        << "F=wireframe/solid, P/B=export, mouse drag=orbit, wheel=zoom, Esc=quit.\n";
}

}  // namespace cv

