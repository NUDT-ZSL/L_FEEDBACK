#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cloud {

void writeBytes(const std::string& path, const std::vector<std::uint8_t>& bytes);
void writeRGBA8(const std::string& path, const std::vector<std::uint8_t>& rgba);
void writePPM(const std::string& path, int width, int height, const std::vector<std::uint8_t>& rgba);
void writeU32(const std::string& path, const std::vector<std::uint32_t>& values);
void writeF32(const std::string& path, const std::vector<float>& values);

}  // namespace cloud

