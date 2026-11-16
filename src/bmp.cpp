#include "bmp.hpp"

#include "error.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>

namespace preview {
namespace {

using Bytes = std::vector<std::uint8_t>;

void appendU16(Bytes& bytes, std::uint16_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
}

void appendU32(Bytes& bytes, std::uint32_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
}

std::uint8_t toByte(float value) {
    if (!(value > 0.0f)) {
        return 0;
    }
    if (value >= 1.0f) {
        return 255;
    }
    return static_cast<std::uint8_t>(value * 255.0f + 0.5f);
}

std::string unicodePath(const std::filesystem::path& path) {
    return path.u8string();
}

} // namespace

void writeBmp24(const std::filesystem::path& path, const Image& image) {
    if (image.width <= 0 || image.height <= 0 ||
        image.pixels.size() !=
            static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height)) {
        throw AppError(ErrorCode::OutputWrite, "invalid image buffer passed to BMP writer");
    }

    const std::uint64_t rowBytes64 =
        (static_cast<std::uint64_t>(image.width) * 3ULL + 3ULL) & ~3ULL;
    const std::uint64_t pixelBytes64 = rowBytes64 * static_cast<std::uint64_t>(image.height);
    const std::uint64_t totalBytes64 = 54ULL + pixelBytes64;
    if (rowBytes64 > std::numeric_limits<std::uint32_t>::max() ||
        pixelBytes64 > std::numeric_limits<std::uint32_t>::max() ||
        totalBytes64 > std::numeric_limits<std::uint32_t>::max()) {
        throw AppError(ErrorCode::OutputWrite, "BMP output exceeds the 32-bit BMP size limit");
    }

    const std::uint32_t rowBytes = static_cast<std::uint32_t>(rowBytes64);
    const std::uint32_t pixelBytes = static_cast<std::uint32_t>(pixelBytes64);
    const std::uint32_t totalBytes = static_cast<std::uint32_t>(totalBytes64);

    Bytes header;
    header.reserve(54U);
    appendU16(header, 0x4D42U);
    appendU32(header, totalBytes);
    appendU16(header, 0U);
    appendU16(header, 0U);
    appendU32(header, 54U);

    appendU32(header, 40U);
    appendU32(header, static_cast<std::uint32_t>(image.width));
    appendU32(header, static_cast<std::uint32_t>(image.height));
    appendU16(header, 1U);
    appendU16(header, 24U);
    appendU32(header, 0U);
    appendU32(header, pixelBytes);
    appendU32(header, 2835U);
    appendU32(header, 2835U);
    appendU32(header, 0U);
    appendU32(header, 0U);

    std::error_code directoryError;
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, directoryError);
        if (directoryError) {
            throw AppError(
                ErrorCode::OutputWrite,
                "cannot create output directory '" + unicodePath(parent) + "': " +
                    directoryError.message());
        }
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw AppError(
            ErrorCode::OutputWrite,
            "cannot open output file for writing: " + unicodePath(path));
    }
    output.write(
        reinterpret_cast<const char*>(header.data()),
        static_cast<std::streamsize>(header.size()));

    Bytes row(static_cast<std::size_t>(rowBytes), 0U);
    for (int outputY = 0; outputY < image.height; ++outputY) {
        const int sourceY = image.height - 1 - outputY;
        for (int x = 0; x < image.width; ++x) {
            const glm::vec3& pixel = image.pixels[
                static_cast<std::size_t>(sourceY) * static_cast<std::size_t>(image.width) +
                static_cast<std::size_t>(x)];
            const std::size_t offset = static_cast<std::size_t>(x) * 3U;
            row[offset + 0U] = toByte(pixel.b);
            row[offset + 1U] = toByte(pixel.g);
            row[offset + 2U] = toByte(pixel.r);
        }
        output.write(
            reinterpret_cast<const char*>(row.data()),
            static_cast<std::streamsize>(row.size()));
        if (!output) {
            throw AppError(
                ErrorCode::OutputWrite,
                "write failed for output file: " + unicodePath(path));
        }
    }

    output.flush();
    if (!output) {
        throw AppError(
            ErrorCode::OutputWrite,
            "flush failed for output file: " + unicodePath(path));
    }
}

} // namespace preview
