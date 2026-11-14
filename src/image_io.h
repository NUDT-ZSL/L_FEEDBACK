#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace df {

struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;
};

void writePngRgb(const std::string& path, int width, int height,
                 const std::vector<std::uint8_t>& rgb);
void writePfm(const std::string& path, int width, int height,
              const std::vector<float>& values);
std::vector<float> readPfm(const std::string& path, int& width, int& height);
RgbImage readPngRgb(const std::string& path);

} // namespace df

