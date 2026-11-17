#include "image_io.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace cv {
namespace {

uint16_t readLe16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t readLe32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

void writeLe16(std::vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xffu));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xffu));
}

void writeLe32(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xffu));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xffu));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xffu));
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xffu));
}

bool writeFile(const std::string& path, const std::vector<uint8_t>& bytes,
               std::string* error) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        if (error) {
            *error = "cannot open output file: " + path;
        }
        return false;
    }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.flush();
    if (!out) {
        if (error) {
            *error = "failed while writing: " + path;
        }
        return false;
    }
    return true;
}

bool readFile(const std::string& path, std::vector<uint8_t>& bytes, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error) {
            *error = "cannot open input file: " + path;
        }
        return false;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0 || static_cast<uint64_t>(size) > (1ull << 31)) {
        if (error) {
            *error = "invalid or excessively large file: " + path;
        }
        return false;
    }
    in.seekg(0, std::ios::beg);
    bytes.resize(static_cast<size_t>(size));
    if (!bytes.empty()) {
        in.read(reinterpret_cast<char*>(bytes.data()), size);
    }
    if (!in) {
        if (error) {
            *error = "failed while reading: " + path;
        }
        return false;
    }
    return true;
}

bool parsePpmHeader(const std::vector<uint8_t>& bytes, int& width, int& height,
                    size_t& dataOffset, std::string* error) {
    size_t i = 0;
    auto skipSpace = [&]() {
        while (i < bytes.size()) {
            if (bytes[i] == '#') {
                while (i < bytes.size() && bytes[i] != '\n') {
                    ++i;
                }
            } else if (bytes[i] == ' ' || bytes[i] == '\t' || bytes[i] == '\r' ||
                       bytes[i] == '\n') {
                ++i;
            } else {
                break;
            }
        }
    };
    auto readToken = [&]() -> std::string {
        skipSpace();
        const size_t start = i;
        while (i < bytes.size() && bytes[i] != ' ' && bytes[i] != '\t' &&
               bytes[i] != '\r' && bytes[i] != '\n' && bytes[i] != '#') {
            ++i;
        }
        return std::string(reinterpret_cast<const char*>(bytes.data() + start), i - start);
    };

    if (bytes.size() < 10 || bytes[0] != 'P' || bytes[1] != '6') {
        if (error) {
            *error = "PPM magic must be P6";
        }
        return false;
    }
    i = 2;
    const std::string widthToken = readToken();
    const std::string heightToken = readToken();
    const std::string maxToken = readToken();
    auto parseToken = [](const std::string& token, long& value) {
        if (token.empty()) {
            return false;
        }
        errno = 0;
        char* end = nullptr;
        const long parsed = std::strtol(token.c_str(), &end, 10);
        if (errno != 0 || end == token.c_str() || *end != '\0') {
            return false;
        }
        value = parsed;
        return true;
    };
    long parsedWidth = 0;
    long parsedHeight = 0;
    long maxValue = 0;
    if (!parseToken(widthToken, parsedWidth) || !parseToken(heightToken, parsedHeight) ||
        !parseToken(maxToken, maxValue)) {
        if (error) {
            *error = "PPM header contains a non-numeric token";
        }
        return false;
    }
    width = static_cast<int>(parsedWidth);
    height = static_cast<int>(parsedHeight);
    if (parsedWidth <= 0 || parsedHeight <= 0 || maxValue != 255 ||
        parsedWidth > 100000 || parsedHeight > 100000) {
        if (error) {
            *error = "PPM dimensions/maxval are invalid";
        }
        return false;
    }
    // Binary PPM has exactly one whitespace delimiter after maxval. Do not
    // consume further bytes: the first pixel channel may itself be whitespace.
    if (i >= bytes.size() || (bytes[i] != ' ' && bytes[i] != '\t' &&
                              bytes[i] != '\r' && bytes[i] != '\n')) {
        if (error) {
            *error = "PPM header is missing its payload delimiter";
        }
        return false;
    }
    if (bytes[i] == '\r' && i + 1 < bytes.size() && bytes[i + 1] == '\n') {
        i += 2;
    } else {
        ++i;
    }
    dataOffset = i;
    const uint64_t expected =
        static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 3ull;
    if (static_cast<uint64_t>(bytes.size() - dataOffset) != expected) {
        if (error) {
            *error = "PPM pixel payload length does not match dimensions";
        }
        return false;
    }
    return true;
}

}  // namespace

ExportReport exportPpmP6(const std::string& path, const Image& image) {
    ExportReport report;
    report.path = path;
    if (!image.valid()) {
        report.error = "framebuffer is invalid";
        return report;
    }
    std::vector<uint8_t> bytes;
    bytes.reserve(image.rgb.size() + 32);
    const std::string header = "P6\n" + std::to_string(image.width) + " " +
                               std::to_string(image.height) + "\n255\n";
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.insert(bytes.end(), image.rgb.begin(), image.rgb.end());
    if (!writeFile(path, bytes, &report.error)) {
        return report;
    }
    if (!validatePpmHeader(path, &report.error)) {
        return report;
    }
    report.ok = true;
    report.pixelHash = hashImage(image);
    return report;
}

ExportReport exportBmp24(const std::string& path, const Image& image) {
    ExportReport report;
    report.path = path;
    if (!image.valid()) {
        report.error = "framebuffer is invalid";
        return report;
    }
    const uint32_t rowBytes =
        (static_cast<uint32_t>(image.width) * 3u + 3u) & ~3u;
    const uint32_t pixelBytes = rowBytes * static_cast<uint32_t>(image.height);
    const uint32_t fileSize = 14u + 40u + pixelBytes;
    std::vector<uint8_t> bytes;
    bytes.reserve(fileSize);
    bytes.push_back('B');
    bytes.push_back('M');
    writeLe32(bytes, fileSize);
    writeLe16(bytes, 0);
    writeLe16(bytes, 0);
    writeLe32(bytes, 54);

    writeLe32(bytes, 40);
    writeLe32(bytes, static_cast<uint32_t>(image.width));
    writeLe32(bytes, static_cast<uint32_t>(image.height));
    writeLe16(bytes, 1);
    writeLe16(bytes, 24);
    writeLe32(bytes, 0);
    writeLe32(bytes, pixelBytes);
    writeLe32(bytes, 2835);
    writeLe32(bytes, 2835);
    writeLe32(bytes, 0);
    writeLe32(bytes, 0);

    for (int y = image.height - 1; y >= 0; --y) {
        const size_t rowStart =
            static_cast<size_t>(y) * static_cast<size_t>(image.width) * 3;
        for (int x = 0; x < image.width; ++x) {
            const size_t p = rowStart + static_cast<size_t>(x) * 3;
            bytes.push_back(image.rgb[p + 2]);
            bytes.push_back(image.rgb[p + 1]);
            bytes.push_back(image.rgb[p + 0]);
        }
        while ((bytes.size() - 54) % 4 != 0) {
            bytes.push_back(0);
        }
    }

    if (!writeFile(path, bytes, &report.error)) {
        return report;
    }
    if (!validateBmpHeader(path, &report.error)) {
        return report;
    }
    report.ok = true;
    report.pixelHash = hashImage(image);
    return report;
}

bool validatePpmHeader(const std::string& path, std::string* error) {
    std::vector<uint8_t> bytes;
    if (!readFile(path, bytes, error)) {
        return false;
    }
    int width = 0;
    int height = 0;
    size_t offset = 0;
    return parsePpmHeader(bytes, width, height, offset, error);
}

bool validateBmpHeader(const std::string& path, std::string* error) {
    std::vector<uint8_t> bytes;
    if (!readFile(path, bytes, error)) {
        return false;
    }
    if (bytes.size() < 54 || bytes[0] != 'B' || bytes[1] != 'M') {
        if (error) {
            *error = "BMP signature/header is missing";
        }
        return false;
    }
    const uint32_t fileSize = readLe32(bytes.data() + 2);
    const uint32_t pixelOffset = readLe32(bytes.data() + 10);
    const uint32_t dibSize = readLe32(bytes.data() + 14);
    const uint32_t width = readLe32(bytes.data() + 18);
    const int32_t height = static_cast<int32_t>(readLe32(bytes.data() + 22));
    const uint16_t planes = readLe16(bytes.data() + 26);
    const uint16_t bpp = readLe16(bytes.data() + 28);
    const uint32_t compression = readLe32(bytes.data() + 30);
    const uint32_t imageSize = readLe32(bytes.data() + 34);
    if (fileSize != bytes.size() || pixelOffset != 54 || dibSize != 40 || width == 0 ||
        height <= 0 || planes != 1 || bpp != 24 || compression != 0) {
        if (error) {
            *error = "BMP header fields are invalid (expect 24-bit BI_RGB, bottom-up)";
        }
        return false;
    }
    const uint64_t expectedSize =
        (static_cast<uint64_t>(width) * 3ull + 3ull) / 4ull * 4ull *
        static_cast<uint64_t>(height);
    if (imageSize != expectedSize ||
        fileSize != 54ull + static_cast<uint64_t>(imageSize)) {
        if (error) {
            *error = "BMP payload size does not match header";
        }
        return false;
    }
    return true;
}

bool loadPpmP6(const std::string& path, Image& image, std::string* error) {
    std::vector<uint8_t> bytes;
    if (!readFile(path, bytes, error)) {
        return false;
    }
    int width = 0;
    int height = 0;
    size_t offset = 0;
    if (!parsePpmHeader(bytes, width, height, offset, error)) {
        return false;
    }
    image.width = width;
    image.height = height;
    image.rgb.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset), bytes.end());
    return true;
}

bool loadBmp24(const std::string& path, Image& image, std::string* error) {
    if (!validateBmpHeader(path, error)) {
        return false;
    }
    std::vector<uint8_t> bytes;
    if (!readFile(path, bytes, error)) {
        return false;
    }
    const int width = static_cast<int>(readLe32(bytes.data() + 18));
    const int height = static_cast<int>(readLe32(bytes.data() + 22));
    const uint32_t rowBytes = (static_cast<uint32_t>(width) * 3u + 3u) & ~3u;
    image.width = width;
    image.height = height;
    image.rgb.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
    for (int y = 0; y < height; ++y) {
        const size_t srcRow = 54 + static_cast<size_t>(height - 1 - y) * rowBytes;
        const size_t dstRow = static_cast<size_t>(y) * static_cast<size_t>(width) * 3;
        for (int x = 0; x < width; ++x) {
            const size_t src = srcRow + static_cast<size_t>(x) * 3;
            const size_t dst = dstRow + static_cast<size_t>(x) * 3;
            image.rgb[dst + 0] = bytes[src + 2];
            image.rgb[dst + 1] = bytes[src + 1];
            image.rgb[dst + 2] = bytes[src + 0];
        }
    }
    return true;
}

uint64_t hashImage(const Image& image) {
    uint64_t hash = 1469598103934665603ull;
    auto mix = [&hash](uint8_t byte) {
        hash ^= byte;
        hash *= 1099511628211ull;
    };
    mix(static_cast<uint8_t>(image.width & 0xff));
    mix(static_cast<uint8_t>((image.width >> 8) & 0xff));
    mix(static_cast<uint8_t>(image.height & 0xff));
    mix(static_cast<uint8_t>((image.height >> 8) & 0xff));
    for (uint8_t byte : image.rgb) {
        mix(byte);
    }
    return hash;
}

}  // namespace cv

