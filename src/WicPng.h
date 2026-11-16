#pragma once

#include "Common.h"

class WicPngWriter {
public:
    WicPngWriter();

    void Write(const std::filesystem::path& path, uint32_t width, uint32_t height,
               const std::vector<uint8_t>& bgraPixels);

private:
    ComPtr<IWICImagingFactory> factory_;
};

