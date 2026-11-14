#pragma once

#include <filesystem>
#include <span>
#include <vector>

#include "baker/model.h"

namespace baker {

std::vector<std::uint8_t> read_file(const std::filesystem::path& path,
                                    std::uint64_t maximum_size,
                                    const char* description);
Scene parse_scene(std::span<const std::uint8_t> bytes);
Scene load_scene(const std::filesystem::path& path);

}  // namespace baker
