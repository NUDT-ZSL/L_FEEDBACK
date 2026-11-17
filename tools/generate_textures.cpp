#include "raster/bmp.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> solidBgr(
    std::uint32_t width, std::uint32_t height, std::uint8_t blue,
    std::uint8_t green, std::uint8_t red) {
    std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(width) * height * 3U);
    for (std::size_t i = 0; i < pixels.size(); i += 3U) {
        pixels[i] = blue;
        pixels[i + 1U] = green;
        pixels[i + 2U] = red;
    }
    return pixels;
}

bool writeTexture(const std::filesystem::path& path, std::uint32_t width,
                  std::uint32_t height, std::vector<std::uint8_t> pixels) {
    std::string error;
    if (!raster::writeBmp24(path, static_cast<int>(width),
                            static_cast<int>(height), pixels, error)) {
        std::cerr << "error: cannot write '" << path.string()
                  << "': " << error << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: generate_textures <output-directory>\n";
        return 1;
    }

    const std::filesystem::path outputDirectory = argv[1];
    std::error_code filesystemError;
    std::filesystem::create_directories(outputDirectory, filesystemError);
    if (filesystemError) {
        std::cerr << "error: cannot create texture directory: "
                  << filesystemError.message() << '\n';
        return 1;
    }

    constexpr std::uint32_t size = 4;
    std::vector<std::uint8_t> base =
        solidBgr(size, size, 0U, 0U, 255U);
    base[0] = 0U;
    base[1] = 255U;
    base[2] = 0U;

    std::vector<std::uint8_t> detail =
        solidBgr(size, size, 255U, 0U, 0U);
    detail[(size - 1U) * 3U] = 0U;
    detail[(size - 1U) * 3U + 1U] = 255U;
    detail[(size - 1U) * 3U + 2U] = 0U;

    if (!writeTexture(outputDirectory / "base.bmp", size, size,
                      std::move(base)) ||
        !writeTexture(outputDirectory / "detail.bmp", size, size,
                      std::move(detail))) {
        return 1;
    }
    return 0;
}

