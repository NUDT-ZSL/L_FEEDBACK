#pragma once

#include "asset.hpp"
#include "error.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace skel {

class AssetReader {
public:
    // On failure `asset` is reset to a default-constructed value. The parser
    // never publishes a partially populated asset.
    static bool load(const std::filesystem::path& path, Asset& asset,
                     Error& error);
    static bool parse(const std::vector<std::uint8_t>& bytes, Asset& asset,
                      Error& error);
};

} // namespace skel

