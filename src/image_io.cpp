#include "image_io.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace skel {
namespace {

bool ensureParentDirectory(const std::filesystem::path& path, Error& error) {
    if (path.parent_path().empty()) {
        return true;
    }
    std::error_code filesystemError;
    std::filesystem::create_directories(path.parent_path(), filesystemError);
    if (filesystemError) {
        error = makeError(ErrorCode::WriteFailed,
                          "cannot create output directory: " +
                              filesystemError.message());
        return false;
    }
    return true;
}

} // namespace

bool writePfm(const std::filesystem::path& path, std::uint32_t width,
              std::uint32_t height, const std::vector<float>& depth,
              Error& error) {
    if (width == 0U || height == 0U ||
        depth.size() != static_cast<std::size_t>(width) * height) {
        error = makeError(ErrorCode::InvalidArgument,
                          "PFM dimensions do not match the depth buffer");
        return false;
    }
    if (!ensureParentDirectory(path, error)) {
        return false;
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        error = makeError(ErrorCode::WriteFailed,
                          "cannot open PFM output: " + path.string());
        return false;
    }
    stream << "PF\n" << width << ' ' << height << "\n-1.0\n";

    // PFM stores rows bottom-to-top. Values are written explicitly as
    // little-endian IEEE-754 floats so output is platform independent.
    for (std::uint32_t y = height; y-- > 0U;) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::uint32_t raw =
                std::bit_cast<std::uint32_t>(depth[static_cast<std::size_t>(y) *
                                                       width +
                                                   x]);
            const unsigned char bytes[4]{
                static_cast<unsigned char>(raw & 0xFFU),
                static_cast<unsigned char>((raw >> 8U) & 0xFFU),
                static_cast<unsigned char>((raw >> 16U) & 0xFFU),
                static_cast<unsigned char>((raw >> 24U) & 0xFFU),
            };
            stream.write(reinterpret_cast<const char*>(bytes), 4);
        }
    }
    stream.flush();
    if (!stream) {
        error = makeError(ErrorCode::WriteFailed,
                          "failed while writing PFM output: " + path.string());
        return false;
    }
    error = {};
    return true;
}

bool writePpm(const std::filesystem::path& path, std::uint32_t width,
              std::uint32_t height,
              const std::vector<std::uint8_t>& rgb, Error& error) {
    if (width == 0U || height == 0U ||
        rgb.size() != static_cast<std::size_t>(width) * height * 3U) {
        error = makeError(ErrorCode::InvalidArgument,
                          "PPM dimensions do not match the color buffer");
        return false;
    }
    if (!ensureParentDirectory(path, error)) {
        return false;
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        error = makeError(ErrorCode::WriteFailed,
                          "cannot open PPM output: " + path.string());
        return false;
    }
    stream << "P6\n" << width << ' ' << height << "\n255\n";
    stream.write(reinterpret_cast<const char*>(rgb.data()),
                 static_cast<std::streamsize>(rgb.size()));
    stream.flush();
    if (!stream) {
        error = makeError(ErrorCode::WriteFailed,
                          "failed while writing PPM output: " + path.string());
        return false;
    }
    error = {};
    return true;
}

} // namespace skel

