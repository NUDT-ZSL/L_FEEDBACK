#pragma once

#include "common.hpp"

#include <filesystem>
#include <string>
#include <unordered_set>

namespace skin {

bool writePpmAtomic(
    const std::filesystem::path& path,
    const RgbImage& image,
    std::string& error
);

RgbImage loadImageForComparison(
    const std::filesystem::path& path,
    std::string& error
);

bool compareImages(
    const RgbImage& rendered,
    const RgbImage& baseline,
    PixelDifference& difference,
    std::string& error
);

class OutputPathAllocator {
public:
    explicit OutputPathAllocator(std::filesystem::path directory);

    std::filesystem::path allocate(const std::string& groupName);
    void release(const std::filesystem::path& path);

private:
    std::filesystem::path directory_;
    std::unordered_set<std::string> claimed_;
};

}  // namespace skin

