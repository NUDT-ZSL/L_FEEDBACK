#pragma once

#include <string>

namespace tilegen {

struct PipelineOptions {
    std::string scenePath;
    std::string outputDirectory;
    int tileSize = 256;
    int levels = 3;
    int previewLevel = -1;
    int previewWidth = 512;
    int previewHeight = 256;
    int samplesPerAxis = 4;
    bool resume = false;
    int simulateDiskFullLevel = -1;
};

int runPipeline(const PipelineOptions& options);

}  // namespace tilegen
