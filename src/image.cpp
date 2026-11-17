#include "image.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <sstream>

namespace ct {
namespace {

std::uint8_t toByte(double linear) {
    const double gamma = std::pow(std::clamp(linear, 0.0, 1.0), 1.0 / 2.2);
    return static_cast<std::uint8_t>(std::clamp(std::floor(gamma * 255.0 + 0.5), 0.0, 255.0));
}

bool ensureParent(const std::string& path, std::string& error) {
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (parent.empty()) {
        return true;
    }
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) {
        error = "cannot create output directory '" + parent.string() + "': " + ec.message();
        return false;
    }
    return true;
}

void putLe16(std::ofstream& out, std::uint16_t value) {
    const char bytes[2] = {
        static_cast<char>(value & 0xff),
        static_cast<char>((value >> 8) & 0xff),
    };
    out.write(bytes, 2);
}

void putLe32(std::ofstream& out, std::uint32_t value) {
    const char bytes[4] = {
        static_cast<char>(value & 0xff),
        static_cast<char>((value >> 8) & 0xff),
        static_cast<char>((value >> 16) & 0xff),
        static_cast<char>((value >> 24) & 0xff),
    };
    out.write(bytes, 4);
}

}  // namespace

bool writeBmp(const std::string& path, const ColorBuffer& image, std::string& error) {
    if (image.width <= 0 || image.height <= 0 ||
        image.pixels.size() != static_cast<std::size_t>(image.width) * image.height) {
        error = "invalid color buffer";
        return false;
    }
    const int row_bytes = image.width * 3;
    const int padding = (4 - (row_bytes % 4)) % 4;
    const std::uint32_t pixel_bytes =
        static_cast<std::uint32_t>((row_bytes + padding) * image.height);
    const std::uint32_t header_bytes = 14 + 40;
    if (pixel_bytes > 0xffffffffu - header_bytes) {
        error = "image is too large for BMP";
        return false;
    }
    if (!ensureParent(path, error)) {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        error = "cannot open image file '" + path + "'";
        return false;
    }
    out.put('B');
    out.put('M');
    putLe32(out, header_bytes + pixel_bytes);
    putLe16(out, 0);
    putLe16(out, 0);
    putLe32(out, header_bytes);
    putLe32(out, 40);
    putLe32(out, static_cast<std::uint32_t>(image.width));
    putLe32(out, static_cast<std::uint32_t>(image.height));
    putLe16(out, 1);
    putLe16(out, 24);
    putLe32(out, 0);
    putLe32(out, pixel_bytes);
    putLe32(out, 2835);
    putLe32(out, 2835);
    putLe32(out, 0);
    putLe32(out, 0);

    std::vector<char> row(static_cast<std::size_t>(row_bytes + padding), 0);
    for (int y = image.height - 1; y >= 0; --y) {
        for (int x = 0; x < image.width; ++x) {
            const Vec3 color = image.at(x, y);
            row[static_cast<std::size_t>(x) * 3 + 0] =
                static_cast<char>(toByte(color.z));
            row[static_cast<std::size_t>(x) * 3 + 1] =
                static_cast<char>(toByte(color.y));
            row[static_cast<std::size_t>(x) * 3 + 2] =
                static_cast<char>(toByte(color.x));
        }
        out.write(row.data(), static_cast<std::streamsize>(row.size()));
    }
    if (!out) {
        error = "failed while writing BMP file '" + path + "'";
        return false;
    }
    return true;
}

bool writePpm(const std::string& path, const ColorBuffer& image, std::string& error) {
    if (image.width <= 0 || image.height <= 0) {
        error = "invalid color buffer";
        return false;
    }
    if (!ensureParent(path, error)) {
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        error = "cannot open image file '" + path + "'";
        return false;
    }
    out << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    std::vector<char> row(static_cast<std::size_t>(image.width) * 3);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const Vec3 color = image.at(x, y);
            row[static_cast<std::size_t>(x) * 3 + 0] = static_cast<char>(toByte(color.x));
            row[static_cast<std::size_t>(x) * 3 + 1] = static_cast<char>(toByte(color.y));
            row[static_cast<std::size_t>(x) * 3 + 2] = static_cast<char>(toByte(color.z));
        }
        out.write(row.data(), static_cast<std::streamsize>(row.size()));
    }
    if (!out) {
        error = "failed while writing PPM file '" + path + "'";
        return false;
    }
    return true;
}

bool writeJsonText(const std::string& path, const std::string& text, std::string& error) {
    if (!ensureParent(path, error)) {
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        error = "cannot open report file '" + path + "'";
        return false;
    }
    out << text;
    if (!out) {
        error = "failed while writing report file '" + path + "'";
        return false;
    }
    return true;
}

}  // namespace ct
