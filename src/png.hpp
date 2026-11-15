#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace vlr {

std::vector<std::uint8_t> encodePngRgba8(int width, int height,
                                         const std::vector<std::uint8_t>& rgba);
void writeBytesAtomic(const std::filesystem::path& path,
                      const std::vector<std::uint8_t>& bytes);
void writeTextAtomic(const std::filesystem::path& path, const std::string& text);

} // namespace vlr
