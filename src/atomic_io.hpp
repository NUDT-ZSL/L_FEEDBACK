#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace lfb::io {

bool ensureWritableDirectory(const std::filesystem::path& directory, std::string& error);
bool atomicWriteBytes(const std::filesystem::path& path,
                      const std::vector<std::uint8_t>& bytes,
                      std::string& error);
bool readBytes(const std::filesystem::path& path,
               std::vector<std::uint8_t>& bytes,
               std::string& error);
bool readText(const std::filesystem::path& path, std::string& text, std::string& error);
void cleanupTemporaryFiles(const std::filesystem::path& directory);

}  // namespace lfb::io
