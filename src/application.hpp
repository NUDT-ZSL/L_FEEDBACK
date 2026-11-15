#pragma once

#include <string>

namespace demo {

struct Options {
    bool selfTest = false;
    bool hidden = false;
    bool injectShaderCompileFailure = false;
    bool injectShaderLinkFailure = false;
    bool simulateContextFailure = false;
    bool simulateResourceFailure = false;
    bool injectGlError = false;
    bool startVsyncOff = false;
    bool showHelp = false;
    int benchmarkFrames = 0;
    std::string screenshotPath;
};

Options parseCommandLine(int argc, char **argv);
void printUsage(const char *executableName);
int runApplication(const Options &options);

} // namespace demo
