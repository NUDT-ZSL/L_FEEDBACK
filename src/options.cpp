#include "options.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace pbr {
namespace {

bool parseFloat(const std::string& text, float& value) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const float parsed = std::strtof(text.c_str(), &end);
    if (errno != 0 || end == text.c_str() || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

bool parseInt(const std::string& text, int& value) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' || parsed < -1000000 || parsed > 1000000) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

}  // namespace

bool parseTonemap(const std::string& text, Tonemap& tonemap) {
    if (text == "reinhard") {
        tonemap = Tonemap::Reinhard;
        return true;
    }
    if (text == "aces") {
        tonemap = Tonemap::ACES;
        return true;
    }
    if (text == "uncharted2") {
        tonemap = Tonemap::Uncharted2;
        return true;
    }
    return false;
}

const char* tonemapName(Tonemap tonemap) {
    switch (tonemap) {
        case Tonemap::Reinhard:
            return "reinhard";
        case Tonemap::ACES:
            return "aces";
        case Tonemap::Uncharted2:
            return "uncharted2";
    }
    return "unknown";
}

std::string exposureLabel(float exposure) {
    std::ostringstream output;
    if (std::fabs(exposure - std::round(exposure)) < 1.0e-4f) {
        const int integerExposure = static_cast<int>(std::round(exposure));
        if (integerExposure > 0) {
            output << "ev+" << integerExposure;
        } else {
            output << "ev" << integerExposure;
        }
    } else {
        output << "ev" << std::showpos << std::fixed << std::setprecision(2) << exposure;
    }
    return output.str();
}

void printUsage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "  --scene <file>          Scene JSON (default: assets/scene.json)\n"
        << "  --output <prefix>       Output prefix (default: output/render)\n"
        << "  --mode <render|validate> Rendering or validation mode (default: render)\n"
        << "  --samples <1..4096>     Direct-light samples per pixel (default: 256)\n"
        << "  --exposure <EV>         Preview exposure in [-10, 10] (default: 0)\n"
        << "  --tonemap <curve>       reinhard | aces | uncharted2 (default: aces)\n"
        << "  --subframes <1..128>    Motion-blur accumulation subframes (default: 8)\n"
        << "  --reference <file>      Reference stats JSON (default: assets/reference_stats.json)\n"
        << "  --report <file>         Validation report path (default: output/validation_report.txt)\n"
        << "  --write-reference <file> Write stats of this render as a new reference\n"
        << "  --help                  Show this help\n"
        << "\nOutputs are fixed at 512x512 and named <prefix>_<exposure>_<curve>.hdr/.png.\n"
        << "Exit codes: 0 success, 1 invalid arguments, 2 scene load failure, 3 render/IO failure.\n";
}

bool parseOptions(int argc, char** argv, Options& options, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            options.showHelp = true;
            continue;
        }
        if (i + 1 >= argc) {
            error = "missing value after " + arg;
            return false;
        }
        const std::string value = argv[++i];
        if (arg == "--scene") {
            options.scenePath = value;
        } else if (arg == "--output") {
            options.outputPrefix = value;
        } else if (arg == "--mode") {
            if (value == "render") {
                options.mode = Mode::Render;
            } else if (value == "validate") {
                options.mode = Mode::Validate;
            } else {
                error = "--mode must be render or validate, got '" + value + "'";
                return false;
            }
        } else if (arg == "--samples") {
            if (!parseInt(value, options.samples) || options.samples < 1 || options.samples > 4096) {
                error = "--samples must be an integer in [1, 4096]";
                return false;
            }
        } else if (arg == "--subframes") {
            if (!parseInt(value, options.subframes) || options.subframes < 1 || options.subframes > 128) {
                error = "--subframes must be an integer in [1, 128]";
                return false;
            }
        } else if (arg == "--exposure") {
            if (!parseFloat(value, options.exposure) || options.exposure < -10.0f || options.exposure > 10.0f) {
                error = "--exposure must be a finite number in [-10, 10]";
                return false;
            }
        } else if (arg == "--tonemap") {
            if (!parseTonemap(value, options.tonemap)) {
                error = "--tonemap must be reinhard, aces, or uncharted2";
                return false;
            }
        } else if (arg == "--reference") {
            options.referencePath = value;
        } else if (arg == "--report") {
            options.reportPath = value;
        } else if (arg == "--write-reference") {
            options.writeReferencePath = value;
        } else {
            error = "unknown option '" + arg + "'";
            return false;
        }
    }

    if (options.scenePath.empty()) {
        error = "--scene must not be empty";
        return false;
    }
    if (options.outputPrefix.empty()) {
        error = "--output must not be empty";
        return false;
    }
    if (options.referencePath.empty() || options.reportPath.empty()) {
        error = "reference and report paths must not be empty";
        return false;
    }
    if (options.writeReferencePath.empty() && !options.referencePath.empty() &&
        options.mode == Mode::Validate) {
        // The default is intentionally allowed; scene/reference existence is checked later.
    }
    return true;
}

}  // namespace pbr
