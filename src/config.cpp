#include "config.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace skin {
namespace {

using Json = nlohmann::json;

bool finite(float value) {
    return std::isfinite(value);
}

bool readNumber(
    const Json& object,
    const char* key,
    float& output,
    std::string& error
) {
    if (!object.is_object() || !object.contains(key)) {
        error = std::string("missing field '") + key + "'";
        return false;
    }
    const Json& value = object.at(key);
    if (!value.is_number()) {
        error = std::string("field '") + key + "' must be a number";
        return false;
    }
    float candidate = 0.0F;
    try {
        candidate = value.get<float>();
    } catch (const std::exception&) {
        error = std::string("field '") + key + "' is outside float range";
        return false;
    }
    if (!finite(candidate)) {
        error = std::string("field '") + key + "' must be finite";
        return false;
    }
    output = candidate;
    return true;
}

bool readOptionalNumber(
    const Json& object,
    const char* key,
    float& output,
    std::string& error
) {
    if (!object.contains(key)) {
        return true;
    }
    return readNumber(object, key, output, error);
}

bool readVec3(
    const Json& object,
    const char* key,
    glm::vec3& output,
    bool required,
    std::string& error
) {
    if (!object.contains(key)) {
        if (required) {
            error = std::string("missing field '") + key + "'";
            return false;
        }
        return true;
    }
    const Json& value = object.at(key);
    if (!value.is_array() || value.size() != 3U) {
        error = std::string("field '") + key + "' must be an array of 3 numbers";
        return false;
    }
    glm::vec3 candidate{};
    for (std::size_t i = 0; i < 3U; ++i) {
        if (!value[i].is_number()) {
            error = std::string("field '") + key + "' contains a non-number";
            return false;
        }
        try {
            candidate[static_cast<glm::length_t>(i)] =
                value[i].get<float>();
        } catch (const std::exception&) {
            error = std::string("field '") + key
                + "' contains a value outside float range";
            return false;
        }
        if (!finite(candidate[static_cast<glm::length_t>(i)])) {
            error = std::string("field '") + key + "' contains a non-finite value";
            return false;
        }
    }
    output = candidate;
    return true;
}

bool readOptionalVec3(
    const Json& object,
    const char* key,
    glm::vec3& output,
    std::string& error
) {
    return readVec3(object, key, output, false, error);
}

bool inUnitRange(const glm::vec3& value) {
    for (glm::length_t i = 0; i < 3; ++i) {
        if (value[i] < 0.0F || value[i] > 1.0F) {
            return false;
        }
    }
    return true;
}

bool parseMaterialCore(
    const Json& object,
    MaterialSpec& material,
    int& code,
    std::string& error
) {
    if (!readVec3(object, "albedo", material.albedo, true, error)) {
        code = kErrorMaterialFields;
        return false;
    }
    if (!inUnitRange(material.albedo)) {
        code = kErrorAlbedo;
        error = "albedo channels must be within [0, 1]";
        return false;
    }

    if (!readNumber(object, "roughness", material.roughness, error)) {
        code = kErrorMaterialFields;
        return false;
    }
    if (material.roughness <= 0.0F || material.roughness > 1.0F) {
        code = kErrorRoughness;
        error = "roughness must be greater than 0 and at most 1";
        return false;
    }

    if (!readNumber(object, "anisotropy", material.anisotropy, error)) {
        code = kErrorMaterialFields;
        return false;
    }
    if (material.anisotropy < 0.0F || material.anisotropy > 1.0F) {
        code = kErrorAnisotropy;
        error = "anisotropy must be within [0, 1]";
        return false;
    }

    if (!readOptionalNumber(
            object,
            "anisotropy_rotation",
            material.anisotropyRotation,
            error
        )) {
        code = kErrorMaterialFields;
        return false;
    }

    if (!readOptionalNumber(
            object,
            "specular_f0",
            material.specularF0,
            error
        )) {
        code = kErrorSpecular;
        return false;
    }
    if (material.specularF0 < 0.0F || material.specularF0 > 1.0F) {
        code = kErrorSpecular;
        error = "specular_f0 must be within [0, 1]";
        return false;
    }

    return true;
}

bool parseMaterialSss(
    const Json& object,
    MaterialSpec& material,
    int& code,
    std::string& error
) {
    if (object.contains("sss")) {
        const Json& sss = object.at("sss");
        if (!sss.is_object()) {
            code = kErrorSss;
            error = "field 'sss' must be an object";
            return false;
        }
        if (!readOptionalNumber(
                sss,
                "strength",
                material.sssStrength,
                error
            )
            || !readOptionalNumber(
                sss,
                "curvature",
                material.sssCurvature,
                error
            )
            || !readOptionalVec3(
                sss,
                "color",
                material.sssColor,
                error
            )) {
            code = kErrorSss;
            return false;
        }
    } else {
        if (!readOptionalNumber(
                object,
                "sss_strength",
                material.sssStrength,
                error
            )
            || !readOptionalNumber(
                object,
                "sss_curvature",
                material.sssCurvature,
                error
            )
            || !readOptionalVec3(
                object,
                "sss_color",
                material.sssColor,
                error
            )) {
            code = kErrorSss;
            return false;
        }
    }

    if (material.sssStrength < 0.0F || material.sssStrength > 10.0F) {
        code = kErrorSss;
        error = "sss strength must be within [0, 10]";
        return false;
    }
    if (material.sssCurvature < 0.0F || material.sssCurvature > 1.0F) {
        code = kErrorSss;
        error = "sss curvature must be within [0, 1]";
        return false;
    }
    if (!inUnitRange(material.sssColor)) {
        code = kErrorSss;
        error = "sss color channels must be within [0, 1]";
        return false;
    }
    return true;
}

bool parseMaterial(
    const Json& object,
    MaterialSpec& material,
    int& code,
    std::string& error
) {
    if (!object.is_object()) {
        code = kErrorMaterialFields;
        error = "material entry must be a JSON object";
        return false;
    }
    if (object.contains("name")) {
        if (!object.at("name").is_string()) {
            code = kErrorGroupName;
            error = "material name must be a string";
            return false;
        }
        material.name = object.at("name").get<std::string>();
    }
    if (!parseMaterialCore(object, material, code, error)) {
        return false;
    }
    if (!parseMaterialSss(object, material, code, error)) {
        return false;
    }
    return true;
}

bool readInt(
    const Json& object,
    const char* key,
    int defaultValue,
    int minimum,
    int maximum,
    int& output,
    std::string& error
) {
    if (!object.contains(key)) {
        output = defaultValue;
        return true;
    }
    const Json& value = object.at(key);
    if (!value.is_number_integer()) {
        error = std::string("field '") + key + "' must be an integer";
        return false;
    }
    long long candidate = 0;
    try {
        if (value.is_number_unsigned()) {
            const unsigned long long unsignedValue =
                value.get<unsigned long long>();
            if (unsignedValue
                > static_cast<unsigned long long>(
                    std::numeric_limits<long long>::max()
                )) {
                error = std::string("field '") + key + "' is too large";
                return false;
            }
            candidate = static_cast<long long>(unsignedValue);
        } else {
            candidate = value.get<long long>();
        }
    } catch (const std::exception&) {
        error = std::string("field '") + key + "' is outside integer range";
        return false;
    }
    if (candidate < minimum || candidate > maximum) {
        std::ostringstream message;
        message << "field '" << key << "' must be within ["
                << minimum << ", " << maximum << ']';
        error = message.str();
        return false;
    }
    output = static_cast<int>(candidate);
    return true;
}

std::string jsonScalarToString(const Json& value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_number_unsigned()) {
        return std::to_string(value.get<unsigned long long>());
    }
    if (value.is_number_integer()) {
        return std::to_string(value.get<long long>());
    }
    if (value.is_number_float()) {
        return std::to_string(value.get<double>());
    }
    return value.dump();
}

void assignPath(
    RenderConfig& config,
    const std::string& configured,
    std::filesystem::path& target
) {
    std::filesystem::path path(configured);
    target = path.is_absolute() ? path : config.baseDirectory / path;
}

}  // namespace

MaterialSpec defaultReferenceMaterial() {
    MaterialSpec material;
    material.name = "reference_skin";
    material.albedo = glm::vec3(0.62F, 0.34F, 0.27F);
    material.roughness = 0.62F;
    material.anisotropy = 0.12F;
    material.anisotropyRotation = 0.45F;
    material.sssColor = glm::vec3(1.0F, 0.73F, 0.62F);
    material.sssStrength = 0.88F;
    material.sssCurvature = 0.060F;
    material.specularF0 = 0.028F;
    return material;
}

ConfigLoadResult loadConfig(const std::filesystem::path& path) {
    ConfigLoadResult result;

    std::ifstream input(path, std::ios::in | std::ios::binary);
    if (!input.is_open()) {
        result.error = "cannot open config file: " + path.string();
        return result;
    }
    std::string text(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>()
    );
    input.close();

    Json root;
    try {
        root = Json::parse(text);
    } catch (const std::exception& exception) {
        result.error = std::string("JSON parse error: ") + exception.what();
        return result;
    }
    if (!root.is_object()) {
        result.error = "config root must be a JSON object";
        return result;
    }

    RenderConfig config;
    config.sourcePath = std::filesystem::absolute(path);
    config.baseDirectory = config.sourcePath.parent_path();
    config.fingerprint = formatHex64(fnv1a64(text));

    std::string error;
    if (!readInt(root, "width", 512, 1, 8192, config.width, error)
        || !readInt(root, "height", 512, 1, 8192, config.height, error)) {
        result.error = error;
        return result;
    }
    if (root.contains("seed")) {
        const Json& seed = root.at("seed");
        if (!seed.is_number_unsigned() && !seed.is_number_integer()) {
            result.error = "field 'seed' must be a non-negative integer";
            return result;
        }
        unsigned long long value = 0;
        try {
            if (seed.is_number_unsigned()) {
                value = seed.get<unsigned long long>();
            } else {
                const long long signedValue = seed.get<long long>();
                if (signedValue < 0) {
                    result.error = "field 'seed' must be non-negative";
                    return result;
                }
                value = static_cast<unsigned long long>(signedValue);
            }
        } catch (const std::exception&) {
            result.error = "field 'seed' is outside integer range";
            return result;
        }
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            result.error = "field 'seed' must fit in uint32";
            return result;
        }
        config.seed = static_cast<std::uint32_t>(value);
    }

    if (root.contains("output_dir")) {
        if (!root.at("output_dir").is_string()) {
            result.error = "field 'output_dir' must be a string";
            return result;
        }
        config.outputDirectoryConfigured =
            root.at("output_dir").get<std::string>();
    }
    assignPath(
        config,
        config.outputDirectoryConfigured,
        config.outputDirectory
    );

    if (root.contains("sss_lut")) {
        const Json& lut = root.at("sss_lut");
        if (!lut.is_object()) {
            result.error = "field 'sss_lut' must be an object";
            return result;
        }
        if (!readInt(
                lut,
                "profile_samples",
                config.lut.profileSamples,
                8,
                4096,
                config.lut.profileSamples,
                error
            )
            || !readInt(
                lut,
                "curvature_samples",
                config.lut.curvatureSamples,
                8,
                4096,
                config.lut.curvatureSamples,
                error
            )
            || !readInt(
                lut,
                "scattering_samples",
                config.lut.scatteringSamples,
                1,
                256,
                config.lut.scatteringSamples,
                error
            )) {
            result.error = error;
            return result;
        }
        float curvatureMin = config.lut.curvatureMin;
        float curvatureMax = config.lut.curvatureMax;
        if (!readOptionalNumber(lut, "curvature_min", curvatureMin, error)
            || !readOptionalNumber(lut, "curvature_max", curvatureMax, error)) {
            result.error = error;
            return result;
        }
        if (!(curvatureMax > curvatureMin)) {
            result.error = "sss_lut curvature_max must exceed curvature_min";
            return result;
        }
        config.lut.curvatureMin = curvatureMin;
        config.lut.curvatureMax = curvatureMax;
        if (!readOptionalVec3(
                lut,
                "scatter_color",
                config.lut.scatterColor,
                error
            )
            || !readOptionalVec3(
                lut,
                "scatter_radius",
                config.lut.scatterRadius,
                error
            )
            || !readOptionalNumber(
                lut,
                "wrap",
                config.lut.wrap,
                error
            )
            || !readOptionalNumber(
                lut,
                "profile_power",
                config.lut.profilePower,
                error
            )) {
            result.error = error;
            return result;
        }
        if (config.lut.wrap < 0.0F || config.lut.wrap > 4.0F) {
            result.error = "sss_lut wrap must be within [0, 4]";
            return result;
        }
        if (config.lut.profilePower <= 0.0F
            || config.lut.profilePower > 16.0F) {
            result.error = "sss_lut profile_power must be within (0, 16]";
            return result;
        }
        if (!inUnitRange(config.lut.scatterColor)) {
            result.error = "sss_lut scatter_color channels must be in [0, 1]";
            return result;
        }
        for (glm::length_t i = 0; i < 3; ++i) {
            if (config.lut.scatterRadius[i] < 0.0F) {
                result.error = "sss_lut scatter_radius cannot be negative";
                return result;
            }
        }
    }

    if (root.contains("tone_mapping")) {
        const Json& tone = root.at("tone_mapping");
        if (!tone.is_object()) {
            result.error = "field 'tone_mapping' must be an object";
            return result;
        }
        if (tone.contains("mode")) {
            if (!tone.at("mode").is_string()) {
                result.error = "tone_mapping mode must be a string";
                return result;
            }
            const std::string mode = tone.at("mode").get<std::string>();
            if (mode == "aces") {
                config.tone.mode = ToneMapMode::Aces;
            } else if (mode == "reinhard") {
                config.tone.mode = ToneMapMode::Reinhard;
            } else if (mode == "none") {
                config.tone.mode = ToneMapMode::None;
            } else {
                result.error = "tone_mapping mode must be aces, reinhard or none";
                return result;
            }
        }
        if (!readOptionalNumber(
                tone,
                "exposure",
                config.tone.exposure,
                error
            )
            || !readOptionalNumber(
                tone,
                "gamma",
                config.tone.gamma,
                error
            )) {
            result.error = error;
            return result;
        }
        if (config.tone.exposure < 0.0F
            || config.tone.exposure > 100.0F
            || config.tone.gamma <= 0.0F
            || config.tone.gamma > 10.0F) {
            result.error = "tone mapping exposure/gamma is out of range";
            return result;
        }
    }

    if (root.contains("camera")) {
        const Json& camera = root.at("camera");
        if (!camera.is_object()) {
            result.error = "field 'camera' must be an object";
            return result;
        }
        if (!readOptionalVec3(camera, "position", config.camera.position, error)
            || !readOptionalVec3(camera, "target", config.camera.target, error)
            || !readOptionalVec3(camera, "up", config.camera.up, error)
            || !readOptionalNumber(
                camera,
                "fov_y_degrees",
                config.camera.fovYDegrees,
                error
            )) {
            result.error = error;
            return result;
        }
    }
    if (glm::distance(config.camera.position, config.camera.target) < 0.0001F
        || glm::length(config.camera.up) < 0.0001F) {
        result.error = "camera position/target/up is degenerate";
        return result;
    }
    if (config.camera.fovYDegrees <= 1.0F
        || config.camera.fovYDegrees >= 179.0F) {
        result.error = "camera fov_y_degrees must be within (1, 179)";
        return result;
    }

    if (root.contains("light")) {
        const Json& light = root.at("light");
        if (!light.is_object()) {
            result.error = "field 'light' must be an object";
            return result;
        }
        if (!readOptionalVec3(light, "position", config.light.position, error)
            || !readOptionalVec3(light, "target", config.light.target, error)
            || !readOptionalVec3(light, "color", config.light.color, error)
            || !readOptionalNumber(light, "width", config.light.width, error)
            || !readOptionalNumber(light, "height", config.light.height, error)
            || !readOptionalNumber(
                light,
                "intensity",
                config.light.intensity,
                error
            )
            || !readInt(
                light,
                "samples_x",
                config.light.samplesX,
                1,
                16,
                config.light.samplesX,
                error
            )
            || !readInt(
                light,
                "samples_y",
                config.light.samplesY,
                1,
                16,
                config.light.samplesY,
                error
            )) {
            result.error = error;
            return result;
        }
    }
    if (glm::distance(config.light.position, config.light.target) < 0.0001F
        || config.light.width <= 0.0F
        || config.light.height <= 0.0F
        || config.light.intensity < 0.0F
        || !inUnitRange(config.light.color)) {
        result.error = "light geometry/color/intensity is invalid";
        return result;
    }

    if (root.contains("environment")) {
        const Json& environment = root.at("environment");
        if (!environment.is_object()) {
            result.error = "field 'environment' must be an object";
            return result;
        }
        if (!readOptionalVec3(
                environment,
                "ambient_sky",
                config.environment.sky,
                error
            )
            || !readOptionalVec3(
                environment,
                "ambient_ground",
                config.environment.ground,
                error
            )
            || !readOptionalNumber(
                environment,
                "ambient_intensity",
                config.environment.intensity,
                error
            )) {
            result.error = error;
            return result;
        }
        if (config.environment.intensity < 0.0F
            || config.environment.intensity > 100.0F) {
            result.error = "environment ambient_intensity is out of range";
            return result;
        }
    }

    if (root.contains("scene")) {
        const Json& scene = root.at("scene");
        if (!scene.is_object()) {
            result.error = "field 'scene' must be an object";
            return result;
        }
        if (!readOptionalNumber(
                scene,
                "sphere_radius",
                config.scene.sphereRadius,
                error
            )
            || !readOptionalNumber(
                scene,
                "sphere_spacing",
                config.scene.sphereSpacing,
                error
            )
            || !readOptionalVec3(
                scene,
                "background",
                config.scene.background,
                error
            )) {
            result.error = error;
            return result;
        }
    }
    if (config.scene.sphereRadius <= 0.0F
        || config.scene.sphereSpacing < config.scene.sphereRadius * 2.0F) {
        result.error = "scene spheres must have positive radius and not overlap";
        return result;
    }
    if (!inUnitRange(config.scene.background)) {
        result.error = "scene background channels must be within [0, 1]";
        return result;
    }

    MaterialSpec globalReference = defaultReferenceMaterial();
    bool hasGlobalReference = false;
    if (root.contains("reference_material")) {
        int code = kErrorReferenceMaterial;
        std::string referenceError;
        if (!parseMaterial(
                root.at("reference_material"),
                globalReference,
                code,
                referenceError
            )) {
            result.error =
                "invalid top-level reference_material: " + referenceError;
            return result;
        }
        if (globalReference.name.empty()) {
            globalReference.name = "reference_skin";
        }
        hasGlobalReference = true;
    }

    if (!root.contains("materials")
        || !root.at("materials").is_array()
        || root.at("materials").empty()) {
        result.error = "materials must be a non-empty array";
        return result;
    }

    const Json& materials = root.at("materials");
    config.groups.reserve(materials.size());
    for (std::size_t index = 0; index < materials.size(); ++index) {
        GroupSpec group;
        group.index = static_cast<int>(index);
        group.reference = globalReference;
        if (!hasGlobalReference) {
            group.reference.name = "reference_skin";
        }

        const Json& entry = materials[index];
        if (!entry.is_object()) {
            group.name = "group_" + std::to_string(index);
            group.errorCode = kErrorGroupJson;
            group.errorMessage = "material group must be a JSON object";
            config.groups.push_back(std::move(group));
            continue;
        }

        if (entry.contains("name") && entry.at("name").is_string()) {
            group.name = entry.at("name").get<std::string>();
        } else {
            group.name = "group_" + std::to_string(index);
        }
        if (group.name.empty()) {
            group.errorCode = kErrorGroupName;
            group.errorMessage = "group name must not be empty";
            group.name = "group_" + std::to_string(index);
        }

        if (entry.contains("baseline")) {
            if (!entry.at("baseline").is_string()) {
                group.errorCode = kErrorGroupJson;
                group.errorMessage = "baseline must be a string path";
            } else {
                group.baselineConfigured =
                    entry.at("baseline").get<std::string>();
                if (!group.baselineConfigured.empty()) {
                    assignPath(
                        config,
                        group.baselineConfigured,
                        group.baselinePath
                    );
                }
            }
        }
        if (entry.contains("baseline_version")) {
            group.baselineVersion =
                jsonScalarToString(entry.at("baseline_version"));
        }

        if (group.errorCode == kErrorNone && entry.contains("spheres")) {
            const Json& spheres = entry.at("spheres");
            if (!spheres.is_array() || spheres.size() != 2U) {
                group.errorCode = kErrorMaterialFields;
                group.errorMessage =
                    "spheres must be an array containing exactly two materials";
            } else {
                int code = kErrorMaterialFields;
                std::string materialError;
                if (!parseMaterial(
                        spheres[0],
                        group.primary,
                        code,
                        materialError
                    )) {
                    group.errorCode = code;
                    group.errorMessage =
                        "primary sphere: " + materialError;
                } else if (!parseMaterial(
                               spheres[1],
                               group.reference,
                               code,
                               materialError
                           )) {
                    group.errorCode = code;
                    group.errorMessage =
                        "reference sphere: " + materialError;
                } else {
                    group.primary.name = group.name;
                }
            }
        } else if (group.errorCode == kErrorNone) {
            int code = kErrorMaterialFields;
            std::string materialError;
            if (!parseMaterial(entry, group.primary, code, materialError)) {
                group.errorCode = code;
                group.errorMessage = materialError;
            } else {
                group.primary.name = group.name;
            }

            if (group.errorCode == kErrorNone
                && entry.contains("reference_material")) {
                if (!parseMaterial(
                        entry.at("reference_material"),
                        group.reference,
                        code,
                        materialError
                    )) {
                    group.errorCode = kErrorReferenceMaterial;
                    group.errorMessage =
                        "reference_material: " + materialError;
                } else if (group.reference.name.empty()) {
                    group.reference.name = "reference_skin";
                }
            }
        }

        config.groups.push_back(std::move(group));
    }

    result.ok = true;
    result.config = std::move(config);
    return result;
}

}  // namespace skin

