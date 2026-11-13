#include "terrain/ppm.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace terrain {

void ensureParentDirectory(const std::string& filename) {
    const std::filesystem::path path(filename);
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code error;
        std::filesystem::create_directories(parent, error);
    }
}

bool writePpm(
    const std::string& filename,
    const ImageRgb& image,
    std::string& error) {
    if (image.width <= 0 || image.height <= 0) {
        error = "invalid PPM dimensions";
        return false;
    }
    const std::size_t required =
        static_cast<std::size_t>(image.width) *
        static_cast<std::size_t>(image.height) * 3u;
    if (image.pixels.size() < required) {
        error = "PPM pixel buffer is too small";
        return false;
    }

    ensureParentDirectory(filename);
    std::ofstream output(filename, std::ios::binary);
    if (!output) {
        error = "unable to open PPM output";
        return false;
    }
    output << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    output.write(
        reinterpret_cast<const char*>(image.pixels.data()),
        static_cast<std::streamsize>(required));
    if (!output) {
        error = "failed while writing PPM output";
        return false;
    }
    return true;
}

}  // namespace terrain

