#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cv {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;

    bool valid() const {
        return width > 0 && height > 0 &&
               rgb.size() == static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    }
};

struct ExportReport {
    bool ok = false;
    std::string error;
    std::string path;
    uint64_t pixelHash = 0;
};

ExportReport exportPpmP6(const std::string& path, const Image& image);
ExportReport exportBmp24(const std::string& path, const Image& image);

bool validatePpmHeader(const std::string& path, std::string* error = nullptr);
bool validateBmpHeader(const std::string& path, std::string* error = nullptr);
bool loadPpmP6(const std::string& path, Image& image, std::string* error = nullptr);
bool loadBmp24(const std::string& path, Image& image, std::string* error = nullptr);

uint64_t hashImage(const Image& image);

}  // namespace cv

