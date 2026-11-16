#pragma once

#include "types.hpp"

#include <string>
#include <vector>

namespace mv {

MaterialSpec loadMaterial(const std::string& metadataPath);
std::vector<VariantRequest> resolveVariants(const MaterialSpec& material,
                                            const std::string& selection);
void applyDependencyOverrides(MaterialSpec& material, const CommandLine& command);
std::vector<std::string> sortedMacros(const std::vector<std::string>& macros);

}  // namespace mv
