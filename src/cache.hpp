#pragma once

#include "image.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace lfb {

class FrameCache {
public:
    FrameCache(std::filesystem::path root, std::string globalKey);

    void cleanHalfProducts() const;
    std::optional<Image> load(std::uint32_t frameIndex,
                              const std::string& subKey,
                              int width,
                              int height) const;
    void commit(std::uint32_t frameIndex,
                const std::string& subKey,
                const Image& image) const;

private:
    std::filesystem::path entryDirectory(const std::string& subKey) const;

    std::filesystem::path root_;
    std::string globalKey_;
};

}  // namespace lfb
