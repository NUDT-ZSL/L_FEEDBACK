#pragma once

#include <string>

namespace pbr {

enum class Mode {
    Render,
    Validate,
};

enum class Tonemap {
    Reinhard,
    ACES,
    Uncharted2,
};

struct Options {
    std::string scenePath = "assets/scene.json";
    std::string outputPrefix = "output/render";
    std::string referencePath = "assets/reference_stats.json";
    std::string reportPath = "output/validation_report.txt";
    std::string writeReferencePath;
    Mode mode = Mode::Render;
    Tonemap tonemap = Tonemap::ACES;
    int samples = 256;
    int subframes = 8;
    float exposure = 0.0f;
    bool showHelp = false;
};

enum class ExitCode {
    Ok = 0,
    InvalidArguments = 1,
    SceneLoadFailed = 2,
    RenderOrIoFailed = 3,
};

bool parseOptions(int argc, char** argv, Options& options, std::string& error);
void printUsage(const char* executable);
const char* tonemapName(Tonemap tonemap);
bool parseTonemap(const std::string& text, Tonemap& tonemap);
std::string exposureLabel(float exposure);

}  // namespace pbr
