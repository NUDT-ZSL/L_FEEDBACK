#include "image_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#define STBIW_WINDOWS_UTF8
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "error.h"

namespace sr {
namespace {

std::uint32_t crc32(const std::uint8_t* data, std::size_t size) {
    static std::array<std::uint32_t, 256> table{};
    static bool initialized = false;
    if (!initialized) {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t value = i;
            for (int bit = 0; bit < 8; ++bit) {
                value = (value & 1U) ? (0xedb88320U ^ (value >> 1U)) : (value >> 1U);
            }
            table[i] = value;
        }
        initialized = true;
    }

    std::uint32_t crc = 0xffffffffU;
    for (std::size_t i = 0; i < size; ++i) {
        crc = table[(crc ^ data[i]) & 0xffU] ^ (crc >> 8U);
    }
    return crc ^ 0xffffffffU;
}

void appendU32BigEndian(std::vector<std::uint8_t>& output, std::uint32_t value) {
    output.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xffU));
    output.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xffU));
    output.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
    output.push_back(static_cast<std::uint8_t>(value & 0xffU));
}

void appendChunk(std::vector<std::uint8_t>& png,
                 const char type[5],
                 const std::vector<std::uint8_t>& payload) {
    appendU32BigEndian(png, static_cast<std::uint32_t>(payload.size()));
    const std::size_t typeOffset = png.size();
    png.insert(png.end(), type, type + 4);
    png.insert(png.end(), payload.begin(), payload.end());
    const std::uint32_t crc =
        crc32(png.data() + typeOffset, 4 + payload.size());
    appendU32BigEndian(png, crc);
}

void appendStoredZlib(std::vector<std::uint8_t>& output,
                      const std::vector<std::uint8_t>& raw) {
    output.push_back(0x78);
    output.push_back(0x01);
    std::size_t offset = 0;
    do {
        const std::size_t remaining = raw.size() - offset;
        const std::size_t blockSize = std::min<std::size_t>(remaining, 65535);
        const bool finalBlock = offset + blockSize == raw.size();
        output.push_back(finalBlock ? 0x01 : 0x00);
        const std::uint16_t length = static_cast<std::uint16_t>(blockSize);
        const std::uint16_t inverse = static_cast<std::uint16_t>(~length);
        output.push_back(static_cast<std::uint8_t>(length & 0xffU));
        output.push_back(static_cast<std::uint8_t>((length >> 8U) & 0xffU));
        output.push_back(static_cast<std::uint8_t>(inverse & 0xffU));
        output.push_back(static_cast<std::uint8_t>((inverse >> 8U) & 0xffU));
        output.insert(output.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
                      raw.begin() + static_cast<std::ptrdiff_t>(offset + blockSize));
        offset += blockSize;
    } while (offset < raw.size());

    std::uint32_t a = 1;
    std::uint32_t b = 0;
    for (std::uint8_t byte : raw) {
        a = (a + byte) % 65521U;
        b = (b + a) % 65521U;
    }
    appendU32BigEndian(output, (b << 16U) | a);
}

void writeBinaryFile(const std::filesystem::path& path,
                     const std::vector<std::uint8_t>& bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        fail(ExitCode::OutputIo, "E_OUTPUT_IO",
             "cannot open image for writing: " + path.string());
    }
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    stream.flush();
    if (!stream) {
        fail(ExitCode::OutputIo, "E_OUTPUT_IO",
             "failed to write image: " + path.string());
    }
}

} // namespace

void writeColorPngRgba8(const std::filesystem::path& path,
                        int width,
                        int height,
                        const std::vector<std::uint8_t>& rgbaTopDown) {
    const std::size_t expected =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4U;
    if (width <= 0 || height <= 0 || rgbaTopDown.size() != expected) {
        fail(ExitCode::OutputIo, "E_OUTPUT_IO", "invalid RGBA8 image dimensions");
    }
    stbi_write_png_compression_level = 8;
    stbi_write_force_png_filter = -1;
    stbi_flip_vertically_on_write(0);
    const std::string filename = path.u8string();
    const int stride = width * 4;
    if (!stbi_write_png(filename.c_str(), width, height, 4,
                        rgbaTopDown.data(), stride)) {
        fail(ExitCode::OutputIo, "E_OUTPUT_IO",
             "stb_image_write failed for " + filename);
    }
}

void writeDepthPng16(const std::filesystem::path& path,
                     int width,
                     int height,
                     const std::vector<float>& depthTopDown) {
    const std::size_t expected =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (width <= 0 || height <= 0 || depthTopDown.size() != expected) {
        fail(ExitCode::OutputIo, "E_OUTPUT_IO", "invalid depth image dimensions");
    }

    const std::size_t rowBytes = static_cast<std::size_t>(width) * 2U + 1U;
    std::vector<std::uint8_t> raw(rowBytes * static_cast<std::size_t>(height), 0);
    for (int y = 0; y < height; ++y) {
        std::size_t offset = static_cast<std::size_t>(y) * rowBytes;
        raw[offset++] = 0;
        for (int x = 0; x < width; ++x) {
            const float depth = depthTopDown[
                static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(x)];
            const double normalized = std::clamp(
                std::isfinite(depth) ? static_cast<double>(depth) : 1.0,
                0.0, 1.0);
            const std::uint16_t sample = static_cast<std::uint16_t>(
                std::lround(normalized * 65535.0));
            raw[offset++] = static_cast<std::uint8_t>((sample >> 8U) & 0xffU);
            raw[offset++] = static_cast<std::uint8_t>(sample & 0xffU);
        }
    }

    std::vector<std::uint8_t> png;
    png.reserve(raw.size() + raw.size() / 128 + 1024);
    static constexpr std::uint8_t signature[8] =
        {137, 80, 78, 71, 13, 10, 26, 10};
    png.insert(png.end(), std::begin(signature), std::end(signature));

    std::vector<std::uint8_t> ihdr;
    appendU32BigEndian(ihdr, static_cast<std::uint32_t>(width));
    appendU32BigEndian(ihdr, static_cast<std::uint32_t>(height));
    ihdr.push_back(16);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    appendChunk(png, "IHDR", ihdr);

    std::vector<std::uint8_t> compressed;
    compressed.reserve(raw.size() + raw.size() / 65535 * 5 + 16);
    appendStoredZlib(compressed, raw);
    appendChunk(png, "IDAT", compressed);
    appendChunk(png, "IEND", {});
    writeBinaryFile(path, png);
}

std::uint64_t fnv1a64(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace sr
