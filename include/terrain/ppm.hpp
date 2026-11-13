#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace terrain {

struct ImageRgb {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

bool writePpm(const std::string& filename, const ImageRgb& image, std::string& error);
void ensureParentDirectory(const std::string& filename);

}  // namespace terrain

