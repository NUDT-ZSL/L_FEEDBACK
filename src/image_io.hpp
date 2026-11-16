#pragma once

#include "error.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace skel {

bool writePfm(const std::filesystem::path& path, std::uint32_t width,
              std::uint32_t height, const std::vector<float>& depth,
              Error& error);

bool writePpm(const std::filesystem::path& path, std::uint32_t width,
              std::uint32_t height,
              const std::vector<std::uint8_t>& rgb, Error& error);

} // namespace skel

