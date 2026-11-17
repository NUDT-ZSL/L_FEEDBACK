#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace terrain {

bool savePngAtomic(const std::filesystem::path& path, int width, int height,
                   const std::vector<uint8_t>& rgb);

struct LoadedImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgb;
};

bool loadPngRgb(const std::filesystem::path& path, LoadedImage& image);

}  // namespace terrain

