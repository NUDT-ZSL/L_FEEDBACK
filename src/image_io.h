#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace sr {

void writeColorPngRgba8(const std::filesystem::path& path,
                        int width,
                        int height,
                        const std::vector<std::uint8_t>& rgbaTopDown);

void writeDepthPng16(const std::filesystem::path& path,
                     int width,
                     int height,
                     const std::vector<float>& depthTopDown);

std::uint64_t fnv1a64(const void* data, std::size_t size);

} // namespace sr
