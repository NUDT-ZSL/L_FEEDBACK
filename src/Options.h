#pragma once

#include "Common.h"

struct Options {
    uint32_t frames = 60;
    std::filesystem::path outputDirectory = L"output";
    std::array<double, 3> cascadeSplits = {0.1, 0.3, 0.6};
    double slopeOffset = 2.0;
    double constOffset = 0.0015;
    bool hasDeviceLossInjection = false;
    uint32_t deviceLossInjectionFrame = 0;
};

struct ParseResult {
    Options options;
    bool showHelp = false;
};

ParseResult ParseCommandLine(int argc, wchar_t** argv);
void PrintUsage(std::wostream& output);

