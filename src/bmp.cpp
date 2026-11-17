#include "raster/bmp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace raster {
namespace {

std::uint16_t readU16(const std::vector<std::uint8_t>& data,
                      std::size_t offset) {
    return static_cast<std::uint16_t>(data[offset]) |
           static_cast<std::uint16_t>(data[offset + 1] << 8U);
}

std::uint32_t readU32(const std::vector<std::uint8_t>& data,
                      std::size_t offset) {
    return static_cast<std::uint32_t>(data[offset]) |
           (static_cast<std::uint32_t>(data[offset + 1]) << 8U) |
           (static_cast<std::uint32_t>(data[offset + 2]) << 16U) |
           (static_cast<std::uint32_t>(data[offset + 3]) << 24U);
}

std::int32_t readI32(const std::vector<std::uint8_t>& data,
                    std::size_t offset) {
    return static_cast<std::int32_t>(readU32(data, offset));
}

void appendU16(std::vector<std::uint8_t>& output, std::uint16_t value) {
    output.push_back(static_cast<std::uint8_t>(value & 0xffU));
    output.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
}

void appendU32(std::vector<std::uint8_t>& output, std::uint32_t value) {
    output.push_back(static_cast<std::uint8_t>(value & 0xffU));
    output.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
    output.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xffU));
    output.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xffU));
}

}  // namespace

bool loadBmpTexture(const std::filesystem::path& path, Texture& texture,
                    std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open file";
        return false;
    }

    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size < 54 || size > 128LL * 1024LL * 1024LL) {
        error = "invalid or unsupported BMP file size";
        return false;
    }

    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    if (!input.read(reinterpret_cast<char*>(data.data()), size)) {
        error = "failed to read file";
        return false;
    }

    if (data[0] != 'B' || data[1] != 'M') {
        error = "missing BMP signature";
        return false;
    }

    const std::uint32_t pixelOffset = readU32(data, 10);
    const std::uint32_t dibSize = readU32(data, 14);
    if (dibSize < 40U || data.size() < 14U + static_cast<std::size_t>(dibSize)) {
        error = "unsupported BMP header";
        return false;
    }

    const std::int32_t width = readI32(data, 18);
    const std::int32_t signedHeight = readI32(data, 22);
    const std::uint16_t planes = readU16(data, 26);
    const std::uint16_t bitsPerPixel = readU16(data, 28);
    const std::uint32_t compression = readU32(data, 30);
    if (width <= 0 || signedHeight == 0 || planes != 1U ||
        (bitsPerPixel != 24U && bitsPerPixel != 32U) || compression != 0U) {
        error = "only uncompressed 24/32-bit BMP textures are supported";
        return false;
    }

    const std::uint32_t height =
        static_cast<std::uint32_t>(signedHeight < 0 ? -signedHeight
                                                    : signedHeight);
    const std::uint64_t rowStride =
        ((static_cast<std::uint64_t>(width) * bitsPerPixel + 31U) / 32U) * 4U;
    const std::uint64_t pixelBytes = rowStride * height;
    if (pixelOffset > data.size() ||
        pixelBytes > data.size() - pixelOffset) {
        error = "truncated BMP pixel data";
        return false;
    }

    Texture loaded;
    loaded.width = static_cast<std::uint32_t>(width);
    loaded.height = height;
    loaded.texels.resize(static_cast<std::size_t>(width) * height);
    const std::size_t bytesPerPixel = bitsPerPixel / 8U;
    const bool bottomUp = signedHeight > 0;

    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint32_t sourceY = bottomUp ? (height - 1U - y) : y;
        const std::size_t rowStart =
            static_cast<std::size_t>(pixelOffset + rowStride * sourceY);
        for (std::uint32_t x = 0; x < loaded.width; ++x) {
            const std::size_t source =
                rowStart + static_cast<std::size_t>(x) * bytesPerPixel;
            const double blue = data[source] / 255.0;
            const double green = data[source + 1] / 255.0;
            const double red = data[source + 2] / 255.0;
            loaded.texels[static_cast<std::size_t>(y) * loaded.width + x] =
                Vec3{red, green, blue};
        }
    }

    texture = std::move(loaded);
    return true;
}

bool writeBmp24(const std::filesystem::path& path, int width, int height,
                const std::vector<std::uint8_t>& bgrTopDown,
                std::string& error) {
    if (width <= 0 || height <= 0) {
        error = "image dimensions must be positive";
        return false;
    }
    const std::uint64_t expectedSize =
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) *
        3U;
    if (bgrTopDown.size() != expectedSize) {
        error = "pixel buffer size does not match dimensions";
        return false;
    }

    const std::uint64_t rowStride =
        ((static_cast<std::uint64_t>(width) * 3U + 3U) / 4U) * 4U;
    const std::uint64_t imageSize = rowStride * static_cast<std::uint64_t>(height);
    const std::uint64_t fileSize = 54U + imageSize;
    if (fileSize > std::numeric_limits<std::uint32_t>::max()) {
        error = "BMP output exceeds the 32-bit BMP size limit";
        return false;
    }

    std::vector<std::uint8_t> header;
    header.reserve(54);
    header.push_back('B');
    header.push_back('M');
    appendU32(header, static_cast<std::uint32_t>(fileSize));
    appendU16(header, 0);
    appendU16(header, 0);
    appendU32(header, 54);

    appendU32(header, 40);
    appendU32(header, static_cast<std::uint32_t>(width));
    appendU32(header, static_cast<std::uint32_t>(height));
    appendU16(header, 1);
    appendU16(header, 24);
    appendU32(header, 0);
    appendU32(header, static_cast<std::uint32_t>(imageSize));
    appendU32(header, 2835);
    appendU32(header, 2835);
    appendU32(header, 0);
    appendU32(header, 0);

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "cannot open output file for writing";
        return false;
    }
    output.write(reinterpret_cast<const char*>(header.data()),
                 static_cast<std::streamsize>(header.size()));

    std::vector<std::uint8_t> row(static_cast<std::size_t>(rowStride), 0U);
    for (int y = height - 1; y >= 0; --y) {
        const std::size_t sourceRow =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(width) * 3U;
        std::copy_n(bgrTopDown.begin() + static_cast<std::ptrdiff_t>(sourceRow),
                    static_cast<std::size_t>(width) * 3U, row.begin());
        output.write(reinterpret_cast<const char*>(row.data()),
                     static_cast<std::streamsize>(row.size()));
        if (!output) {
            error = "failed while writing BMP data";
            return false;
        }
    }

    output.close();
    if (!output) {
        error = "failed to close output file";
        return false;
    }
    return true;
}

}  // namespace raster

