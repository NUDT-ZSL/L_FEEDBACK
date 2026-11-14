#pragma once

#include "types.h"

#include <ostream>
#include <string>
#include <vector>

namespace dr {

enum class RunMode {
    Live,
    SnapshotSave,
    SnapshotLoad,
};

struct Options {
    RunMode mode = RunMode::Live;
    bool modeSpecified = false;
    std::string scenePath = "sample_scene.json";
    std::string snapshotPath = "snapshots/lighting.v1.dgsnap";
    std::string outputPath = "artifacts/render.png";
    std::string comparePath;
    int maxChannelDifference = 1;
    int frames = 1;
    int warmupFrames = 0;
    std::vector<std::string> addLightSpecs;
    std::vector<std::string> removeLightIds;
};

bool parseAddLight(const std::string& specification, Light& light, std::string& error);
bool parseArguments(int argc, char** argv, Options& options, std::string& error, bool& showHelp);
void printUsage(std::ostream& output);

} // namespace dr
