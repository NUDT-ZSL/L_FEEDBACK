#pragma once

#include "environment.h"
#include "options.h"
#include "stats.h"

#include <array>
#include <string>
#include <vector>

namespace pbr {

struct ValidationThresholds {
    float meanColorRgbDistance = 0.035f;
    float highlightHistogramL1 = 0.15f;
    float specularDirectionDegrees = 4.0f;
};

struct ReferenceConfiguration {
    std::string scenePath;
    int samples = 256;
    int subframes = 8;
    int width = 512;
    int height = 512;
    unsigned int seed = 0x105u;
};

struct ReferenceStats {
    ReferenceConfiguration config;
    ValidationThresholds thresholds;
    std::vector<MaterialStats> materials;
    EnvironmentSH environmentSh;
};

struct MetricResult {
    std::string materialName;
    int materialId = -1;
    std::string metric;
    float current = 0.0f;
    float reference = 0.0f;
    float threshold = 0.0f;
    bool passes = false;
    std::string detail;
};

struct ValidationResult {
    bool passes = false;
    bool configurationMatches = true;
    std::string configurationMessage;
    std::vector<MetricResult> metrics;
};

bool writeReferenceStats(const std::string& path,
                         const std::string& scenePath,
                         const ColorPipelineInfo& renderInfo,
                         const RenderStatistics& statistics,
                         const EnvironmentSH& environmentSh,
                         std::string& error);
bool loadReferenceStats(const std::string& path, ReferenceStats& reference, std::string& error);
ValidationResult validateStatistics(const RenderStatistics& current,
                                    const ReferenceStats& reference,
                                    const ColorPipelineInfo& renderInfo);
bool writeValidationReport(const std::string& path,
                           const ValidationResult& result,
                           const RenderStatistics& current,
                           const ReferenceStats& reference,
                           const ColorPipelineInfo& renderInfo,
                           std::string& error);

}  // namespace pbr
