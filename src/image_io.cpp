#include "image_io.h"
#include "error.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <cstring>
#include <fstream>
#include <sstream>

namespace df {

void writePngRgb(const std::string& path, int width, int height,
                 const std::vector<std::uint8_t>& rgb) {
    if (width <= 0 || height <= 0 ||
        rgb.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u) {
        throw AppError(ErrorCode::E_IO, "invalid RGB image dimensions for " + path);
    }
    const int stride = width * 3;
    if (stbi_write_png(path.c_str(), width, height, 3, rgb.data(), stride) == 0) {
        throw AppError(ErrorCode::E_IO, "failed to write PNG: " + path);
    }
}

void writePfm(const std::string& path, int width, int height,
              const std::vector<float>& values) {
    if (width <= 0 || height <= 0 ||
        values.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {
        throw AppError(ErrorCode::E_IO, "invalid PFM image dimensions for " + path);
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw AppError(ErrorCode::E_IO, "cannot open PFM for writing: " + path);
    }
    out << "Pf\n" << width << " " << height << "\n-1.0\n";
    for (int y = height - 1; y >= 0; --y) {
        out.write(reinterpret_cast<const char*>(values.data() +
                                               static_cast<std::size_t>(y) * width),
                  static_cast<std::streamsize>(width * sizeof(float)));
    }
    if (!out) {
        throw AppError(ErrorCode::E_IO, "failed while writing PFM: " + path);
    }
}

std::vector<float> readPfm(const std::string& path, int& width, int& height) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw AppError(ErrorCode::E_IO, "cannot open PFM for reading: " + path);
    }
    std::string magic;
    float scale = 0.0f;
    in >> magic >> width >> height >> scale;
    if (magic != "Pf" || width <= 0 || height <= 0 || scale == 0.0f) {
        throw AppError(ErrorCode::E_IO, "invalid PFM header: " + path);
    }
    in.get();
    std::vector<float> values(static_cast<std::size_t>(width) * height);
    for (int y = height - 1; y >= 0; --y) {
        in.read(reinterpret_cast<char*>(values.data() + static_cast<std::size_t>(y) * width),
                static_cast<std::streamsize>(width * sizeof(float)));
    }
    if (!in) {
        throw AppError(ErrorCode::E_IO, "truncated PFM: " + path);
    }
    if (scale > 0.0f) {
        for (float& value : values) value *= scale;
    }
    return values;
}

RgbImage readPngRgb(const std::string& path) {
    int channels = 0;
    int width = 0;
    int height = 0;
    stbi_uc* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!data) {
        throw AppError(ErrorCode::E_IO, "failed to read PNG: " + path);
    }
    RgbImage image;
    image.width = width;
    image.height = height;
    image.pixels.assign(data, data + static_cast<std::size_t>(width) * height * 3u);
    stbi_image_free(data);
    return image;
}

} // namespace df

