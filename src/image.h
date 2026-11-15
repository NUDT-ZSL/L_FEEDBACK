#pragma once

#include "math.h"
#include "options.h"

#include <string>
#include <vector>

namespace pbr {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<Vec3> pixels;
    std::vector<int> materialIds;
    std::vector<Vec3> reflectionDirections;
};

struct OutputPaths {
    std::string hdr;
    std::string png;
};

struct ColorPipelineInfo {
    float exposureEv = 0.0f;
    Tonemap tonemap = Tonemap::ACES;
    int samples = 256;
    int subframes = 8;
    unsigned int seed = 0x105u;
    std::string scenePath;
};

bool buildOutputPaths(const std::string& prefix,
                      float exposure,
                      Tonemap tonemap,
                      OutputPaths& paths,
                      std::string& error);
bool writeHdr(const std::string& path, const Image& image, const ColorPipelineInfo& info, std::string& error);
bool writePng(const std::string& path, const Image& image, const ColorPipelineInfo& info, std::string& error);
bool ensureParentDirectory(const std::string& path, std::string& error);

}  // namespace pbr
