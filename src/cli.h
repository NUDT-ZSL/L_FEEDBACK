#pragma once

#include <filesystem>
#include <string>

namespace sr {

enum class ContextKind {
    Auto,
    Glfw,
    OSMesa,
};

struct Options {
    int startFrame = 0;
    int frameCount = 1;
    int width = 960;
    int height = 540;
    int shadowSize = 1024;
    int cascadeCount = 2;
    double splitLambda = 0.25;
    double fps = 24.0;
    int meshVariant = 0;
    int regenerateFrame = -1;
    int regenerateVariant = 0;
    std::filesystem::path outputDirectory;
    ContextKind context = ContextKind::Auto;
    std::string testError;
    bool helpRequested = false;
};

Options parseCommandLine(int argc, char** argv);
std::string usageText();
const char* contextKindName(ContextKind kind);

} // namespace sr
