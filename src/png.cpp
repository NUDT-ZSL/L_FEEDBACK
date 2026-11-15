#include "png.hpp"

#include "errors.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <string>

namespace vlr {
namespace {

std::uint32_t crc32(const std::uint8_t* data, std::size_t size) {
    std::uint32_t crc = 0xffffffffu;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1u) ^ (0xedb88320u & mask);
        }
    }
    return ~crc;
}

std::uint32_t adler32(const std::uint8_t* data, std::size_t size) {
    constexpr std::uint32_t mod = 65521u;
    std::uint32_t a = 1u;
    std::uint32_t b = 0u;
    for (std::size_t i = 0; i < size; ++i) {
        a = (a + data[i]) % mod;
        b = (b + a) % mod;
    }
    return (b << 16u) | a;
}

void appendU32(std::vector<std::uint8_t>& output, std::uint32_t value) {
    output.push_back(static_cast<std::uint8_t>((value >> 24u) & 0xffu));
    output.push_back(static_cast<std::uint8_t>((value >> 16u) & 0xffu));
    output.push_back(static_cast<std::uint8_t>((value >> 8u) & 0xffu));
    output.push_back(static_cast<std::uint8_t>(value & 0xffu));
}

void appendChunk(std::vector<std::uint8_t>& png, const char type[5],
                 const std::vector<std::uint8_t>& data) {
    appendU32(png, static_cast<std::uint32_t>(data.size()));
    const std::size_t start = png.size();
    for (int i = 0; i < 4; ++i) {
        png.push_back(static_cast<std::uint8_t>(type[i]));
    }
    png.insert(png.end(), data.begin(), data.end());
    appendU32(png, crc32(png.data() + start, 4 + data.size()));
}

std::vector<std::uint8_t> zlibStored(const std::vector<std::uint8_t>& input) {
    std::vector<std::uint8_t> output;
    output.reserve(input.size() + input.size() / 65535u * 5u + 16u);
    output.push_back(0x78u);
    output.push_back(0x01u);

    std::size_t offset = 0;
    do {
        const std::size_t blockSize = std::min<std::size_t>(65535u, input.size() - offset);
        const bool final = offset + blockSize == input.size();
        output.push_back(final ? 0x01u : 0x00u);
        const std::uint16_t length = static_cast<std::uint16_t>(blockSize);
        const std::uint16_t inverse = static_cast<std::uint16_t>(~length);
        output.push_back(static_cast<std::uint8_t>(length & 0xffu));
        output.push_back(static_cast<std::uint8_t>(length >> 8u));
        output.push_back(static_cast<std::uint8_t>(inverse & 0xffu));
        output.push_back(static_cast<std::uint8_t>(inverse >> 8u));
        output.insert(output.end(), input.begin() + static_cast<std::ptrdiff_t>(offset),
                      input.begin() + static_cast<std::ptrdiff_t>(offset + blockSize));
        offset += blockSize;
    } while (offset < input.size());

    appendU32(output, adler32(input.data(), input.size()));
    return output;
}

void writeBytes(const std::filesystem::path& path,
                const std::vector<std::uint8_t>& bytes) {
    std::error_code ec;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) {
            fail(ErrorCode::Io, "cannot create output directory: " + path.parent_path().string());
        }
    }
    const std::filesystem::path temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            fail(ErrorCode::Io, "cannot open temporary output file: " + temporary.string());
        }
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        if (!output) {
            fail(ErrorCode::Io, "cannot write temporary output file: " + temporary.string());
        }
    }
    std::filesystem::rename(temporary, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(temporary, path, ec);
        if (ec) {
            fail(ErrorCode::Io, "cannot replace output file: " + path.string());
        }
    }
}

} // namespace

std::vector<std::uint8_t> encodePngRgba8(int width, int height,
                                         const std::vector<std::uint8_t>& rgba) {
    if (width <= 0 || height <= 0 ||
        rgba.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u) {
        fail(ErrorCode::Io, "invalid RGBA image dimensions for PNG encoder");
    }

    std::vector<std::uint8_t> scanlines;
    scanlines.reserve(static_cast<std::size_t>(height) *
                      (static_cast<std::size_t>(width) * 4u + 1u));
    for (int y = 0; y < height; ++y) {
        scanlines.push_back(0u);
        const auto begin = rgba.begin() + static_cast<std::ptrdiff_t>(y) * width * 4;
        scanlines.insert(scanlines.end(), begin, begin + static_cast<std::ptrdiff_t>(width) * 4);
    }

    std::vector<std::uint8_t> png = {0x89u, 0x50u, 0x4eu, 0x47u,
                                     0x0du, 0x0au, 0x1au, 0x0au};
    std::vector<std::uint8_t> ihdr;
    appendU32(ihdr, static_cast<std::uint32_t>(width));
    appendU32(ihdr, static_cast<std::uint32_t>(height));
    ihdr.insert(ihdr.end(), {8u, 6u, 0u, 0u, 0u});
    appendChunk(png, "IHDR", ihdr);
    appendChunk(png, "IDAT", zlibStored(scanlines));
    appendChunk(png, "IEND", {});
    return png;
}

void writeBytesAtomic(const std::filesystem::path& path,
                      const std::vector<std::uint8_t>& bytes) {
    writeBytes(path, bytes);
}

void writeTextAtomic(const std::filesystem::path& path, const std::string& text) {
    writeBytes(path, std::vector<std::uint8_t>(text.begin(), text.end()));
}

} // namespace vlr
