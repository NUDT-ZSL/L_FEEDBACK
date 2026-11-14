#pragma once

#include "config.h"
#include "offline.h"

#include <string>

namespace df {

OfflineArtifacts renderOfflineGpu(const RenderConfig& config,
                                  const std::string& outputDirectory);
int runInteractiveGpu(const RenderConfig& config, const std::string& configPath);

} // namespace df

