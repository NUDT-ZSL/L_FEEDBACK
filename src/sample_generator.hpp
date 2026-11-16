#pragma once

#include "asset.hpp"
#include "error.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace skel {

struct GeneratorPreset {
    const char* name;
    const char* description;
};

const std::vector<GeneratorPreset>& generatorPresets();
bool generateAsset(const std::string& preset, CompositionOrder order,
                   Asset& asset, Error& error);
bool writeAsset(const Asset& asset, const std::filesystem::path& path,
                Error& error);
bool generateAssetFile(const std::string& preset, CompositionOrder order,
                       const std::filesystem::path& path, Error& error);

} // namespace skel

