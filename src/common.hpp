#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace skin {

inline constexpr const char* kRendererVersion = SKINRENDER_VERSION;
inline constexpr int kStatsSchemaVersion = 1;

enum ErrorCode : int {
    kErrorNone = 0,
    kErrorGroupJson = 100,
    kErrorGroupName = 101,
    kErrorMaterialFields = 102,
    kErrorAlbedo = 110,
    kErrorRoughness = 111,
    kErrorAnisotropy = 112,
    kErrorSss = 113,
    kErrorSpecular = 114,
    kErrorReferenceMaterial = 115,
    kErrorOpenGl = 200,
    kErrorFramebufferReadback = 201,
    kErrorOutputIo = 300,
    kErrorUnexpected = 900,
};

struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;

    [[nodiscard]] bool valid() const {
        return width > 0
            && height > 0
            && pixels.size()
                == static_cast<std::size_t>(width)
                    * static_cast<std::size_t>(height) * 3U;
    }
};

struct PixelDifference {
    double meanSquaredError = 0.0;
    double meanAbsoluteError = 0.0;
    int maximumAbsoluteError = 0;
    std::uint64_t pixelCount = 0;
};

class GroupFailure : public std::runtime_error {
public:
    GroupFailure(int code, std::string message)
        : std::runtime_error(std::move(message)), code_(code) {}

    [[nodiscard]] int code() const noexcept { return code_; }

private:
    int code_;
};

inline std::uint64_t fnv1a64(const std::uint8_t* data, std::size_t size) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline std::uint64_t fnv1a64(std::string_view text) {
    return fnv1a64(
        reinterpret_cast<const std::uint8_t*>(text.data()),
        text.size()
    );
}

inline std::string formatHex64(std::uint64_t value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(16, '0');
    for (int i = 15; i >= 0; --i) {
        result[static_cast<std::size_t>(i)] = digits[value & 0xFULL];
        value >>= 4U;
    }
    return result;
}

inline std::string sanitizeFileStem(std::string name) {
    static constexpr std::string_view invalid = "/\\:*?\"<>|";
    for (char& ch : name) {
        unsigned char byte = static_cast<unsigned char>(ch);
        if (byte < 0x20U || invalid.find(ch) != std::string_view::npos) {
            ch = '_';
        }
    }

    while (!name.empty() && name.front() == '.') {
        name.front() = '_';
    }
    while (!name.empty()
        && (name.back() == ' ' || name.back() == '.')) {
        name.pop_back();
    }

    if (name.empty()) {
        name = "material";
    }
    if (name.size() > 100U) {
        name.resize(100U);
    }
    return name;
}

}  // namespace skin

