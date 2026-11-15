#include "validation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pbr {
namespace {

using json = nlohmann::json;

class ReferenceError : public std::runtime_error {
public:
    explicit ReferenceError(const std::string& message) : std::runtime_error(message) {}
};

[[noreturn]] void fail(const std::string& path, const std::string& message) {
    throw ReferenceError("'" + path + "': " + message);
}

const json& requireMember(const json& object, const char* key, const std::string& path) {
    if (!object.is_object()) {
        fail(path, "expected object");
    }
    const auto found = object.find(key);
    if (found == object.end()) {
        fail(path + "." + key, "missing required field");
    }
    return *found;
}

const json& requireObject(const json& object, const char* key, const std::string& path) {
    const json& value = requireMember(object, key, path);
    if (!value.is_object()) {
        fail(path + "." + key, "expected object");
    }
    return value;
}

const json& requireArray(const json& object, const char* key, const std::string& path) {
    const json& value = requireMember(object, key, path);
    if (!value.is_array()) {
        fail(path + "." + key, "expected array");
    }
    return value;
}

float requireNumber(const json& object, const char* key, const std::string& path) {
    const json& value = requireMember(object, key, path);
    if (!value.is_number()) {
        fail(path + "." + key, "expected number");
    }
    const double parsed = value.get<double>();
    if (!std::isfinite(parsed)) {
        fail(path + "." + key, "expected a finite number");
    }
    return static_cast<float>(parsed);
}

int requireInt(const json& object, const char* key, const std::string& path) {
    const json& value = requireMember(object, key, path);
    if (!value.is_number_integer()) {
        fail(path + "." + key, "expected integer");
    }
    return value.get<int>();
}

std::string requireString(const json& object, const char* key, const std::string& path) {
    const json& value = requireMember(object, key, path);
    if (!value.is_string()) {
        fail(path + "." + key, "expected string");
    }
    return value.get<std::string>();
}

Vec3 parseVec3Value(const json& value, const std::string& path) {
    if (!value.is_array() || value.size() != 3) {
        fail(path, "expected array of three numbers");
    }
    Vec3 result;
    float* components[3] = {&result.x, &result.y, &result.z};
    for (int i = 0; i < 3; ++i) {
        if (!value[static_cast<size_t>(i)].is_number()) {
            fail(path, "expected three numbers");
        }
        *components[i] = value[static_cast<size_t>(i)].get<float>();
    }
    return result;
}

Vec3 requireVec3(const json& object, const char* key, const std::string& path) {
    return parseVec3Value(requireMember(object, key, path), path + "." + key);
}

std::array<float, kHighlightHistogramBins> requireHistogram(const json& object,
                                                            const char* key,
                                                            const std::string& path) {
    const json& array = requireArray(object, key, path);
    if (array.size() != static_cast<size_t>(kHighlightHistogramBins)) {
        fail(path + "." + key, "expected 16 histogram bins");
    }
    std::array<float, kHighlightHistogramBins> result{};
    for (size_t i = 0; i < result.size(); ++i) {
        if (!array[i].is_number()) {
            fail(path + "." + key, "histogram bins must be numbers");
        }
        result[i] = array[i].get<float>();
    }
    return result;
}

json vec3ToJson(const Vec3& value) {
    return json::array({value.x, value.y, value.z});
}

Vec3 vec3FromJson(const json& value, const std::string& path) {
    return parseVec3Value(value, path);
}

std::string formatVec3(const Vec3& value) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6) << "[" << value.x << ", " << value.y << ", " << value.z
           << "]";
    return output.str();
}

std::string formatHistogram(const std::array<float, kHighlightHistogramBins>& histogram) {
    std::ostringstream output;
    output << "[";
    for (size_t i = 0; i < histogram.size(); ++i) {
        if (i != 0) {
            output << ", ";
        }
        output << std::fixed << std::setprecision(5) << histogram[i];
    }
    output << "]";
    return output.str();
}

bool ensureParent(const std::string& path, std::string& error) {
    return ensureParentDirectory(path, error);
}

bool sameScenePath(const std::string& current, const std::string& baseline) {
    std::error_code currentError;
    std::error_code baselineError;
    const std::filesystem::path currentPath = std::filesystem::weakly_canonical(current, currentError);
    const std::filesystem::path baselinePath = std::filesystem::weakly_canonical(baseline, baselineError);
    if (!currentError && !baselineError) {
        return currentPath == baselinePath;
    }
    return std::filesystem::path(current).lexically_normal() ==
           std::filesystem::path(baseline).lexically_normal();
}

}  // namespace

bool writeReferenceStats(const std::string& path,
                         const std::string& scenePath,
                         const ColorPipelineInfo& renderInfo,
                         const RenderStatistics& statistics,
                         const EnvironmentSH& environmentSh,
                         std::string& error) {
    json root;
    root["schema"] = "pbr-reference-stats-v1";
    root["generated_by"] = "pbr_renderer";
    root["config"] = {
        {"scene", scenePath},
        {"width", kResolution},
        {"height", kResolution},
        {"samples", renderInfo.samples},
        {"subframes", renderInfo.subframes},
        {"seed", renderInfo.seed},
        {"exposure_ev", renderInfo.exposureEv},
        {"tonemap", tonemapName(renderInfo.tonemap)},
        {"dof_denoise", renderInfo.samples < 64 ? "bilateral_7x7" : "bilateral_5x5"},
        {"environment_sh_samples", 8192},
    };
    root["thresholds"] = {
        {"mean_color_rgb_distance", 0.035},
        {"highlight_histogram_l1", 0.15},
        {"specular_direction_degrees", 4.0},
    };
    root["environment_sh"] = {
        {"radiance", json::array()},
        {"irradiance", json::array()},
    };
    for (const Vec3& coefficient : environmentSh.radiance) {
        root["environment_sh"]["radiance"].push_back(vec3ToJson(coefficient));
    }
    for (const Vec3& coefficient : environmentSh.irradiance) {
        root["environment_sh"]["irradiance"].push_back(vec3ToJson(coefficient));
    }
    root["materials"] = json::array();
    for (const MaterialStats& material : statistics.materials) {
        json entry;
        entry["id"] = material.id;
        entry["name"] = material.name;
        entry["pixel_count"] = material.pixelCount;
        entry["mean_color"] = vec3ToJson(material.meanColor);
        entry["highlight_histogram"] = json::array();
        for (float bin : material.highlightHistogram) {
            entry["highlight_histogram"].push_back(bin);
        }
        entry["specular_direction"] = vec3ToJson(material.specularDirection);
        root["materials"].push_back(std::move(entry));
    }

    if (!ensureParent(path, error)) {
        return false;
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        error = "Reference write failed: cannot open '" + path + "'";
        return false;
    }
    output << root.dump(2) << "\n";
    if (!output) {
        error = "Reference write failed while writing '" + path + "'";
        return false;
    }
    return true;
}

bool loadReferenceStats(const std::string& path, ReferenceStats& reference, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "Reference load error: cannot open '" + path + "'";
        return false;
    }
    try {
        json root;
        input >> root;
        if (!root.is_object()) {
            fail("reference", "root must be an object");
        }
        const std::string schema = requireString(root, "schema", "reference");
        if (schema != "pbr-reference-stats-v1") {
            fail("reference.schema", "unsupported schema '" + schema + "'");
        }

        const json& config = requireObject(root, "config", "reference");
        reference.config.scenePath = requireString(config, "scene", "reference.config");
        reference.config.width = requireInt(config, "width", "reference.config");
        reference.config.height = requireInt(config, "height", "reference.config");
        reference.config.samples = requireInt(config, "samples", "reference.config");
        reference.config.subframes = requireInt(config, "subframes", "reference.config");
        if (config.contains("seed") && config["seed"].is_number_integer()) {
            reference.config.seed = static_cast<unsigned int>(config["seed"].get<unsigned int>());
        }

        const json& thresholds = requireObject(root, "thresholds", "reference");
        reference.thresholds.meanColorRgbDistance =
            requireNumber(thresholds, "mean_color_rgb_distance", "reference.thresholds");
        reference.thresholds.highlightHistogramL1 =
            requireNumber(thresholds, "highlight_histogram_l1", "reference.thresholds");
        reference.thresholds.specularDirectionDegrees =
            requireNumber(thresholds, "specular_direction_degrees", "reference.thresholds");
        if (reference.thresholds.meanColorRgbDistance <= 0.0f ||
            reference.thresholds.highlightHistogramL1 <= 0.0f ||
            reference.thresholds.specularDirectionDegrees <= 0.0f) {
            fail("reference.thresholds", "all thresholds must be positive");
        }

        if (root.contains("environment_sh")) {
            const json& sh = requireObject(root, "environment_sh", "reference");
            if (sh.contains("radiance")) {
                const json& radiance = sh["radiance"];
                if (!radiance.is_array() || radiance.size() != 9) {
                    fail("reference.environment_sh.radiance", "expected nine coefficients");
                }
                for (size_t i = 0; i < 9; ++i) {
                    reference.environmentSh.radiance[i] =
                        vec3FromJson(radiance[i], "reference.environment_sh.radiance");
                }
            }
            if (sh.contains("irradiance")) {
                const json& irradiance = sh["irradiance"];
                if (!irradiance.is_array() || irradiance.size() != 9) {
                    fail("reference.environment_sh.irradiance", "expected nine coefficients");
                }
                for (size_t i = 0; i < 9; ++i) {
                    reference.environmentSh.irradiance[i] =
                        vec3FromJson(irradiance[i], "reference.environment_sh.irradiance");
                }
            }
        }

        const json& materials = requireArray(root, "materials", "reference");
        for (size_t i = 0; i < materials.size(); ++i) {
            const std::string materialPath = "reference.materials[" + std::to_string(i) + "]";
            const json& entry = materials[i];
            if (!entry.is_object()) {
                fail(materialPath, "expected object");
            }
            MaterialStats material;
            material.id = requireInt(entry, "id", materialPath);
            if (material.id < 0 || material.id >= kMaxMaterials) {
                fail(materialPath + ".id", "material ID is outside [0, 15]");
            }
            material.name = requireString(entry, "name", materialPath);
            const json& pixelCount = requireMember(entry, "pixel_count", materialPath);
            if (!pixelCount.is_number_integer()) {
                fail(materialPath + ".pixel_count", "expected integer");
            }
            material.pixelCount = pixelCount.get<long long>();
            material.meanColor = requireVec3(entry, "mean_color", materialPath);
            material.highlightHistogram = requireHistogram(entry, "highlight_histogram", materialPath);
            material.specularDirection = normalize(requireVec3(entry, "specular_direction", materialPath));
            reference.materials.push_back(material);
        }
    } catch (const json::exception& exception) {
        error = "Reference load error: JSON parse/type failure in '" + path + "': " + exception.what();
        return false;
    } catch (const std::exception& exception) {
        error = "Reference load error: " + std::string(exception.what());
        return false;
    }
    return true;
}

ValidationResult validateStatistics(const RenderStatistics& current,
                                    const ReferenceStats& reference,
                                    const ColorPipelineInfo& renderInfo) {
    ValidationResult result;
    result.passes = true;

    std::ostringstream configuration;
    bool mismatch = false;
    auto addMismatch = [&](const std::string& text) {
        if (mismatch) {
            configuration << "; ";
        }
        configuration << text;
        mismatch = true;
    };
    if (!sameScenePath(renderInfo.scenePath, reference.config.scenePath)) {
        addMismatch("scene '" + renderInfo.scenePath + "' vs baseline '" + reference.config.scenePath + "'");
    }
    if (reference.config.samples != renderInfo.samples) {
        addMismatch("samples " + std::to_string(renderInfo.samples) + " vs baseline " +
                    std::to_string(reference.config.samples));
    }
    if (reference.config.subframes != renderInfo.subframes) {
        addMismatch("subframes " + std::to_string(renderInfo.subframes) + " vs baseline " +
                    std::to_string(reference.config.subframes));
    }
    if (reference.config.seed != renderInfo.seed) {
        addMismatch("seed " + std::to_string(renderInfo.seed) + " vs baseline " +
                    std::to_string(reference.config.seed));
    }
    if (reference.config.width != kResolution || reference.config.height != kResolution) {
        addMismatch("baseline resolution is not 512x512");
    }
    result.configurationMatches = !mismatch;
    result.configurationMessage = mismatch ? configuration.str() : "render settings match baseline";

    for (const MaterialStats& material : current.materials) {
        const auto found = std::find_if(reference.materials.begin(), reference.materials.end(),
                                        [&](const MaterialStats& candidate) {
                                            return candidate.id == material.id;
                                        });
        if (found == reference.materials.end()) {
            MetricResult missing;
            missing.materialId = material.id;
            missing.materialName = material.name;
            missing.metric = "reference_entry";
            missing.passes = false;
            missing.detail = "reference statistics do not contain this material ID";
            result.metrics.push_back(std::move(missing));
            result.passes = false;
            continue;
        }
        const MaterialStats& baseline = *found;

        MetricResult meanColor;
        meanColor.materialId = material.id;
        meanColor.materialName = material.name;
        meanColor.metric = "mean_color_rgb_distance";
        meanColor.current = length(material.meanColor - baseline.meanColor);
        meanColor.reference = 0.0f;
        meanColor.threshold = reference.thresholds.meanColorRgbDistance;
        meanColor.passes = meanColor.current <= meanColor.threshold;
        meanColor.detail = "current=" + formatVec3(material.meanColor) +
                           " reference=" + formatVec3(baseline.meanColor) +
                           " pixels=" + std::to_string(material.pixelCount) + "/" +
                           std::to_string(baseline.pixelCount);
        result.metrics.push_back(std::move(meanColor));

        MetricResult histogram;
        histogram.materialId = material.id;
        histogram.materialName = material.name;
        histogram.metric = "highlight_histogram_l1";
        histogram.current = 0.0f;
        for (size_t i = 0; i < material.highlightHistogram.size(); ++i) {
            histogram.current +=
                std::fabs(material.highlightHistogram[i] - baseline.highlightHistogram[i]);
        }
        histogram.reference = 0.0f;
        histogram.threshold = reference.thresholds.highlightHistogramL1;
        histogram.passes = histogram.current <= histogram.threshold;
        histogram.detail = "current=" + formatHistogram(material.highlightHistogram) +
                           " reference=" + formatHistogram(baseline.highlightHistogram);
        result.metrics.push_back(std::move(histogram));

        MetricResult direction;
        direction.materialId = material.id;
        direction.materialName = material.name;
        direction.metric = "specular_direction_degrees";
        const float cosine = clampf(dot(normalize(material.specularDirection),
                                        normalize(baseline.specularDirection)),
                                    -1.0f, 1.0f);
        direction.current = std::acos(cosine) * 180.0f / static_cast<float>(kPi);
        direction.reference = 0.0f;
        direction.threshold = reference.thresholds.specularDirectionDegrees;
        direction.passes = direction.current <= direction.threshold;
        direction.detail = "current=" + formatVec3(material.specularDirection) +
                           " reference=" + formatVec3(baseline.specularDirection);
        result.metrics.push_back(std::move(direction));
    }

    for (const MetricResult& metric : result.metrics) {
        if (!metric.passes) {
            result.passes = false;
        }
    }
    return result;
}

bool writeValidationReport(const std::string& path,
                           const ValidationResult& result,
                           const RenderStatistics& current,
                           const ReferenceStats& reference,
                           const ColorPipelineInfo& renderInfo,
                           std::string& error) {
    if (!ensureParent(path, error)) {
        return false;
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        error = "Validation report write failed: cannot open '" + path + "'";
        return false;
    }
    output << "PBR OFFLINE VALIDATION REPORT\n";
    output << "=============================\n";
    output << "scene: " << renderInfo.scenePath << "\n";
    output << "resolution: 512x512\n";
    output << "render_samples: " << renderInfo.samples << "\n";
    output << "render_subframes: " << renderInfo.subframes << "\n";
    output << "render_seed: " << renderInfo.seed << "\n";
    output << "baseline_samples: " << reference.config.samples << "\n";
    output << "baseline_subframes: " << reference.config.subframes << "\n";
    output << "configuration: " << (result.configurationMatches ? "MATCH" : "MISMATCH")
           << " (" << result.configurationMessage << ")\n";
    output << "thresholds: mean_color_rgb_distance=" << reference.thresholds.meanColorRgbDistance
           << " highlight_histogram_l1=" << reference.thresholds.highlightHistogramL1
           << " specular_direction_degrees=" << reference.thresholds.specularDirectionDegrees << "\n";
    output << "\n";

    for (const MaterialStats& material : current.materials) {
        output << "Material " << material.id << " (" << material.name << ")\n";
        const auto found = std::find_if(reference.materials.begin(), reference.materials.end(),
                                        [&](const MaterialStats& candidate) {
                                            return candidate.id == material.id;
                                        });
        if (found != reference.materials.end()) {
            output << "  pixel_count: current=" << material.pixelCount
                   << " reference=" << found->pixelCount << "\n";
        }
        for (const MetricResult& metric : result.metrics) {
            if (metric.materialId != material.id) {
                continue;
            }
            output << "  " << (metric.passes ? "PASS" : "FAIL") << " " << metric.metric
                   << ": current=" << std::fixed << std::setprecision(6) << metric.current
                   << " threshold=" << metric.threshold << "\n";
            output << "       " << metric.detail << "\n";
        }
        output << "\n";
    }

    size_t passed = 0;
    for (const MetricResult& metric : result.metrics) {
        if (metric.passes) {
            ++passed;
        }
    }
    output << "SUMMARY: " << (result.passes ? "PASS" : "FAIL") << " (" << passed << "/"
           << result.metrics.size() << " metrics within threshold)\n";
    if (!result.passes) {
        output << "Out-of-threshold metrics are listed above by material and metric; this is a "
                  "validation result, not a renderer failure.\n";
    }
    output.flush();
    if (!output) {
        error = "Validation report write failed while writing '" + path + "'";
        return false;
    }
    return true;
}

}  // namespace pbr
