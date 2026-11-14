#pragma once

#include <cstdint>
#include <vector>

namespace baker {

struct GeneratorOptions {
    std::uint32_t width = 160;
    std::uint32_t height = 120;
    std::uint32_t layers = 6;
    std::uint32_t tex_size = 64;
    std::uint64_t seed = 22;
};

std::vector<std::uint8_t> generate_scene_bytes(const GeneratorOptions& options);

}  // namespace baker
