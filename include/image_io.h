#pragma once

#include "types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dr {

bool writePng(const std::string& path, int width, int height,
              const std::vector<std::uint8_t>& rgbaTopDown, std::string& error);
bool readPng(const std::string& path, int& width, int& height,
             std::vector<std::uint8_t>& rgbaTopDown, std::string& error);
bool compareImages(const std::vector<std::uint8_t>& actual, int actualWidth, int actualHeight,
                   const std::vector<std::uint8_t>& reference, int referenceWidth, int referenceHeight,
                   ImageDiff& diff, std::string& error);

} // namespace dr
