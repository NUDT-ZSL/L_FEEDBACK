#include "image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace pbr {
namespace {

std::string sanitizeComment(std::string text) {
    for (char& character : text) {
        if (character == '\n' || character == '\r') {
            character = ' ';
        }
    }
    return text;
}

float tonemapReinhard(float value) {
    return value / (1.0f + value);
}

float tonemapAces(float value) {
    const float numerator = value * (2.51f * value + 0.03f);
    const float denominator = value * (2.43f * value + 0.59f) + 0.14f;
    return numerator / std::max(denominator, 1.0e-8f);
}

float unchartedFilm(float value) {
    constexpr float a = 0.15f;
    constexpr float b = 0.50f;
    constexpr float c = 0.10f;
    constexpr float d = 0.20f;
    constexpr float e = 0.02f;
    constexpr float f = 0.30f;
    return ((value * (a * value + c * b) + d * e) / (value * (a * value + b) + d * f)) - e / f;
}

float tonemapUncharted2(float value) {
    constexpr float whitePoint = 11.2f;
    return unchartedFilm(value) / unchartedFilm(whitePoint);
}

float tonemapValue(float value, Tonemap tonemap) {
    value = std::max(value, 0.0f);
    switch (tonemap) {
        case Tonemap::Reinhard:
            return tonemapReinhard(value);
        case Tonemap::ACES:
            return tonemapAces(value);
        case Tonemap::Uncharted2:
            return tonemapUncharted2(value);
    }
    return tonemapReinhard(value);
}

float linearToSrgb(float value) {
    value = std::max(0.0f, std::min(1.0f, value));
    if (value <= 0.0031308f) {
        return value * 12.92f;
    }
    return 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

std::array<unsigned char, 4> encodeRgbe(const Vec3& color) {
    const float maximum = std::max({color.x, color.y, color.z});
    if (!(maximum > 1.0e-32f) || !std::isfinite(maximum)) {
        return {0, 0, 0, 0};
    }
    int exponent = 0;
    const float mantissa = std::frexp(maximum, &exponent);
    const float scale = mantissa * 256.0f / maximum;
    const int r = std::clamp(static_cast<int>(color.x * scale), 0, 255);
    const int g = std::clamp(static_cast<int>(color.y * scale), 0, 255);
    const int b = std::clamp(static_cast<int>(color.z * scale), 0, 255);
    const int e = std::clamp(exponent + 128, 0, 255);
    return {
        static_cast<unsigned char>(r),
        static_cast<unsigned char>(g),
        static_cast<unsigned char>(b),
        static_cast<unsigned char>(e),
    };
}

void writeRleChannel(std::ofstream& output, const std::vector<unsigned char>& channel) {
    const int width = static_cast<int>(channel.size());
    int cursor = 0;
    while (cursor < width) {
        int runLength = 1;
        while (cursor + runLength < width && runLength < 127 &&
               channel[static_cast<size_t>(cursor + runLength)] ==
                   channel[static_cast<size_t>(cursor)]) {
            ++runLength;
        }
        if (runLength >= 4) {
            const unsigned char count = static_cast<unsigned char>(128 + runLength);
            const unsigned char value = channel[static_cast<size_t>(cursor)];
            output.put(static_cast<char>(count));
            output.put(static_cast<char>(value));
            cursor += runLength;
        } else {
            const int literalStart = cursor;
            int literalCount = 0;
            while (cursor < width && literalCount < 128) {
                int nextRun = 1;
                while (cursor + nextRun < width && nextRun < 127 &&
                       channel[static_cast<size_t>(cursor + nextRun)] ==
                           channel[static_cast<size_t>(cursor)]) {
                    ++nextRun;
                }
                if (nextRun >= 4) {
                    break;
                }
                ++cursor;
                ++literalCount;
            }
            if (literalCount == 0) {
                literalCount = 1;
                ++cursor;
            }
            output.put(static_cast<char>(literalCount));
            for (int i = 0; i < literalCount; ++i) {
                output.put(static_cast<char>(channel[static_cast<size_t>(literalStart + i)]));
            }
        }
    }
}

bool createDirectoriesForFile(const std::string& path, std::string& error) {
    const std::filesystem::path filePath(path);
    const std::filesystem::path parent = filePath.parent_path();
    if (parent.empty()) {
        return true;
    }
    std::error_code code;
    std::filesystem::create_directories(parent, code);
    if (code) {
        error = "cannot create output directory '" + parent.string() + "': " + code.message();
        return false;
    }
    return true;
}

}  // namespace

bool ensureParentDirectory(const std::string& path, std::string& error) {
    return createDirectoriesForFile(path, error);
}

bool buildOutputPaths(const std::string& prefix,
                      float exposure,
                      Tonemap tonemap,
                      OutputPaths& paths,
                      std::string& error) {
    std::filesystem::path base(prefix);
    const bool trailingSeparator =
        !prefix.empty() && (prefix.back() == '/' || prefix.back() == '\\');
    std::error_code directoryError;
    const bool existingDirectory = std::filesystem::is_directory(base, directoryError);
    if (trailingSeparator || existingDirectory) {
        std::filesystem::create_directories(base, directoryError);
        if (directoryError) {
            error = "cannot create output directory '" + base.string() + "': " + directoryError.message();
            return false;
        }
        base /= "render";
    }
    std::string extension = base.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (extension == ".hdr" || extension == ".png") {
        base.replace_extension();
    }
    if (prefix.empty() || base.filename().empty()) {
        error = "--output must name a file prefix, not only a directory";
        return false;
    }
    const std::string suffix =
        "_" + exposureLabel(exposure) + "_" + std::string(tonemapName(tonemap));
    paths.hdr = (base.string() + suffix + ".hdr");
    paths.png = (base.string() + suffix + ".png");
    return ensureParentDirectory(paths.hdr, error);
}

bool writeHdr(const std::string& path, const Image& image, const ColorPipelineInfo& info, std::string& error) {
    if (image.width <= 0 || image.height <= 0 ||
        image.pixels.size() != static_cast<size_t>(image.width) * static_cast<size_t>(image.height)) {
        error = "HDR write failed: image dimensions and pixel data do not match";
        return false;
    }
    if (!ensureParentDirectory(path, error)) {
        return false;
    }

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        error = "HDR write failed: cannot open '" + path + "'";
        return false;
    }
    output << "#?RADIANCE\n";
    output << "# PBR offline renderer linear floating-point output\n";
    output << "# exposure_ev=" << info.exposureEv << "\n";
    output << "# tonemap=" << tonemapName(info.tonemap) << "\n";
    output << "# tonemap_exposure_multiplier=" << std::pow(2.0f, info.exposureEv) << "\n";
    output << "# linear_color_space=scene_linear\n";
    output << "# samples=" << info.samples << "\n";
    output << "# subframes=" << info.subframes << "\n";
    output << "# scene=" << sanitizeComment(info.scenePath) << "\n";
    output << "FORMAT=32-bit_rle_rgbe\n\n";
    output << "-Y " << image.height << " +X " << image.width << "\n";

    std::vector<std::vector<unsigned char>> channels(4);
    for (auto& channel : channels) {
        channel.resize(static_cast<size_t>(image.width));
    }
    for (int y = 0; y < image.height; ++y) {
        output.put(2);
        output.put(2);
        output.put(static_cast<char>((image.width >> 8) & 0xFF));
        output.put(static_cast<char>(image.width & 0xFF));
        for (int x = 0; x < image.width; ++x) {
            const Vec3& color = image.pixels[static_cast<size_t>(y) * static_cast<size_t>(image.width) +
                                            static_cast<size_t>(x)];
            const auto encoded = encodeRgbe(color);
            for (int channel = 0; channel < 4; ++channel) {
                channels[static_cast<size_t>(channel)][static_cast<size_t>(x)] =
                    encoded[static_cast<size_t>(channel)];
            }
        }
        for (const auto& channel : channels) {
            writeRleChannel(output, channel);
        }
    }
    output.flush();
    if (!output) {
        error = "HDR write failed while writing '" + path + "'";
        return false;
    }
    return true;
}

bool writePng(const std::string& path, const Image& image, const ColorPipelineInfo& info, std::string& error) {
    if (image.width <= 0 || image.height <= 0 ||
        image.pixels.size() != static_cast<size_t>(image.width) * static_cast<size_t>(image.height)) {
        error = "PNG write failed: image dimensions and pixel data do not match";
        return false;
    }
    if (!createDirectoriesForFile(path, error)) {
        return false;
    }

    const float exposureMultiplier = std::pow(2.0f, info.exposureEv);
    std::vector<unsigned char> rgba(static_cast<size_t>(image.width) * static_cast<size_t>(image.height) * 4u);
    for (size_t i = 0; i < image.pixels.size(); ++i) {
        const Vec3 exposed = clampColor(image.pixels[i] * exposureMultiplier, 0.0f, 1.0e20f);
        const Vec3 mapped = {
            tonemapValue(exposed.x, info.tonemap),
            tonemapValue(exposed.y, info.tonemap),
            tonemapValue(exposed.z, info.tonemap),
        };
        const Vec3 srgb = {
            linearToSrgb(mapped.x),
            linearToSrgb(mapped.y),
            linearToSrgb(mapped.z),
        };
        rgba[i * 4 + 0] = static_cast<unsigned char>(std::lround(std::clamp(srgb.x, 0.0f, 1.0f) * 255.0f));
        rgba[i * 4 + 1] = static_cast<unsigned char>(std::lround(std::clamp(srgb.y, 0.0f, 1.0f) * 255.0f));
        rgba[i * 4 + 2] = static_cast<unsigned char>(std::lround(std::clamp(srgb.z, 0.0f, 1.0f) * 255.0f));
        rgba[i * 4 + 3] = 255;
    }

    if (stbi_write_png(path.c_str(), image.width, image.height, 4, rgba.data(), image.width * 4) == 0) {
        error = "PNG write failed: cannot write '" + path + "'";
        return false;
    }
    return true;
}

}  // namespace pbr
