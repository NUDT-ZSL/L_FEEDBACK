#include "image.h"

#include "errors.h"
#include "json.h"

#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace cloud {
namespace {

void ensureParent(const std::string& path) {
    const std::filesystem::path target(path);
    std::error_code error;
    if (target.has_parent_path()) {
        std::filesystem::create_directories(target.parent_path(), error);
        if (error) {
            throw BusinessError("E_OUTPUT", kExitOutput, "cannot create output directory",
                                json::Value::Object{{"path", target.parent_path().string()}});
        }
    }
}

}  // namespace

void writeBytes(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    ensureParent(path);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw BusinessError("E_OUTPUT", kExitOutput, "cannot open output file",
                            json::Value::Object{{"path", path}});
    }
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    output.flush();
    if (!output) {
        throw BusinessError("E_OUTPUT", kExitOutput, "failed while writing output file",
                            json::Value::Object{{"path", path}});
    }
}

void writeRGBA8(const std::string& path, const std::vector<std::uint8_t>& rgba) {
    writeBytes(path, rgba);
}

void writePPM(const std::string& path, int width, int height, const std::vector<std::uint8_t>& rgba) {
    if (rgba.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4) {
        throw BusinessError("E_OUTPUT", kExitOutput, "RGBA buffer size does not match PPM dimensions");
    }
    ensureParent(path);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw BusinessError("E_OUTPUT", kExitOutput, "cannot open PPM output",
                            json::Value::Object{{"path", path}});
    }
    output << "P6\n" << width << ' ' << height << "\n255\n";
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3);
    for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(width) * height; ++pixel) {
        rgb[pixel * 3 + 0] = rgba[pixel * 4 + 0];
        rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1];
        rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2];
    }
    output.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    output.flush();
    if (!output) {
        throw BusinessError("E_OUTPUT", kExitOutput, "failed while writing PPM output");
    }
}

void writeU32(const std::string& path, const std::vector<std::uint32_t>& values) {
    std::vector<std::uint8_t> bytes(values.size() * 4);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::uint32_t value = values[index];
        bytes[index * 4] = static_cast<std::uint8_t>(value & 0xffu);
        bytes[index * 4 + 1] = static_cast<std::uint8_t>((value >> 8u) & 0xffu);
        bytes[index * 4 + 2] = static_cast<std::uint8_t>((value >> 16u) & 0xffu);
        bytes[index * 4 + 3] = static_cast<std::uint8_t>((value >> 24u) & 0xffu);
    }
    writeBytes(path, bytes);
}

void writeF32(const std::string& path, const std::vector<float>& values) {
    std::vector<std::uint8_t> bytes(values.size() * 4);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(values[index]);
        bytes[index * 4] = static_cast<std::uint8_t>(bits & 0xffu);
        bytes[index * 4 + 1] = static_cast<std::uint8_t>((bits >> 8u) & 0xffu);
        bytes[index * 4 + 2] = static_cast<std::uint8_t>((bits >> 16u) & 0xffu);
        bytes[index * 4 + 3] = static_cast<std::uint8_t>((bits >> 24u) & 0xffu);
    }
    writeBytes(path, bytes);
}

}  // namespace cloud

