#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tilegen {

struct Rgba8 {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
    std::uint8_t a = 0;
};

// Premultiplied linear-ish RGBA used by the compositor. The pipeline treats
// channel values as normalized display values so blend-output tests are stable.
struct PremulPixel {
    float r = 0.0F;
    float g = 0.0F;
    float b = 0.0F;
    float a = 0.0F;
};

struct Image {
    int width = 0;
    int height = 0;
    std::vector<PremulPixel> pixels;

    Image() = default;
    Image(int width, int height, PremulPixel fill = {});

    PremulPixel& at(int x, int y);
    const PremulPixel& at(int x, int y) const;
    bool empty() const noexcept { return width <= 0 || height <= 0; }
};

struct Texture {
    int width = 0;
    int height = 0;
    std::vector<Rgba8> pixels;

    bool empty() const noexcept { return width <= 0 || height <= 0; }
    Rgba8 sampleBilinear(double u, double v, bool repeat) const;
};

std::uint32_t crc32(const std::uint8_t* data, std::size_t size,
                    std::uint32_t initial = 0);
std::uint32_t crc32(const std::vector<std::uint8_t>& data);
std::string crc32Hex(std::uint32_t value);

std::vector<std::uint8_t> encodePngRgba(int width, int height,
                                        const std::vector<Rgba8>& pixels);
std::vector<Rgba8> decodePngRgba(const std::vector<std::uint8_t>& bytes,
                                 int& width, int& height);
std::vector<Rgba8> decodePngRgbaFile(const std::string& path, int& width,
                                     int& height);

std::vector<std::uint8_t> readBinaryFile(const std::string& path);
void writeBinaryFile(const std::string& path, const std::vector<std::uint8_t>& bytes);

}  // namespace tilegen
