#include "material.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>

namespace cr {
namespace {

float luminance(const Vec3& color) {
    return clamp(0.2126F * color.x + 0.7152F * color.y + 0.0722F * color.z, 0.0F, 1.0F);
}

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        if (character >= 'A' && character <= 'Z') {
            return static_cast<char>(character - 'A' + 'a');
        }
        return static_cast<char>(character);
    });
    return value;
}

fs::path resolveTexturePath(const fs::path& materialDirectory, const Json& layer,
                            const char* field, const std::string& context) {
    if (!layer.contains(field) || !layer.at(field).is_string()) {
        throw AppError(E_MAT_PARSE, context + " is missing string field '" + field + "'");
    }
    const fs::path relative = layer.at(field).get<std::string>();
    if (relative.empty()) {
        throw AppError(E_MAT_PARSE, context + " has an empty '" + field + "' path");
    }
    const fs::path resolved = relative.is_absolute() ? relative : materialDirectory / relative;
    if (!pathExists(resolved) || !isRegularFile(resolved)) {
        throw AppError(E_MAT_PARSE,
                       context + " texture '" + field + "' does not exist: " +
                           resolved.string());
    }
    return resolved;
}

MaterialSpec parseMaterialFile(const fs::path& path) {
    Json document;
    try {
        document = Json::parse(readTextFile(path));
    } catch (const AppError&) {
        throw;
    } catch (const std::exception& error) {
        throw AppError(E_MAT_PARSE,
                       "invalid JSON in '" + path.string() + "': " + error.what());
    }

    if (!document.is_object()) {
        throw AppError(E_MAT_PARSE, "material root must be an object: " + path.string());
    }
    if (!document.contains("name") || !document.at("name").is_string()) {
        throw AppError(E_MAT_PARSE, "material is missing string field 'name': " +
                                        path.string());
    }
    if (!document.contains("layers") || !document.at("layers").is_array() ||
        document.at("layers").empty()) {
        throw AppError(E_MAT_PARSE,
                       "material 'layers' must be a non-empty array: " + path.string());
    }

    MaterialSpec material;
    material.name = document.at("name").get<std::string>();
    material.sourceFile = path;
    if (material.name.empty()) {
        throw AppError(E_MAT_PARSE, "material name cannot be empty: " + path.string());
    }

    const fs::path materialDirectory = path.parent_path();
    std::size_t index = 0;
    for (const Json& layerJson : document.at("layers")) {
        const std::string context =
            "material '" + material.name + "' layer " + std::to_string(index);
        if (!layerJson.is_object()) {
            throw AppError(E_MAT_PARSE, context + " must be an object");
        }

        MaterialLayerSpec layer;
        if (!layerJson.contains("name") || !layerJson.at("name").is_string()) {
            throw AppError(E_MAT_PARSE, context + " is missing string field 'name'");
        }
        layer.name = layerJson.at("name").get<std::string>();
        layer.baseColorTexture =
            resolveTexturePath(materialDirectory, layerJson, "baseColor", context);
        layer.metallicTexture =
            resolveTexturePath(materialDirectory, layerJson, "metallic", context);
        layer.roughnessTexture =
            resolveTexturePath(materialDirectory, layerJson, "roughness", context);
        layer.normalTexture =
            resolveTexturePath(materialDirectory, layerJson, "normal", context);

        if (!layerJson.contains("blendMode") || !layerJson.at("blendMode").is_string()) {
            throw AppError(E_MAT_PARSE, context + " is missing string field 'blendMode'");
        }
        const std::string blendMode = lowerAscii(layerJson.at("blendMode").get<std::string>());
        if (blendMode != "override" && blendMode != "linear" && blendMode != "mask") {
            throw AppError(E_MAT_PARSE, context + " has illegal blendMode '" + blendMode + "'");
        }
        layer.blendMode = blendModeFromName(blendMode);

        if (!layerJson.contains("opacity") || !layerJson.at("opacity").is_number()) {
            throw AppError(E_MAT_PARSE, context + " is missing numeric field 'opacity'");
        }
        layer.opacity = layerJson.at("opacity").get<float>();
        if (!std::isfinite(layer.opacity) || layer.opacity < 0.0F || layer.opacity > 1.0F) {
            throw AppError(E_MAT_PARSE, context + " opacity must be in [0,1]");
        }

        material.layers.push_back(std::move(layer));
        ++index;
    }
    return material;
}

}  // namespace

const char* blendModeName(BlendMode mode) {
    switch (mode) {
        case BlendMode::Override:
            return "override";
        case BlendMode::Linear:
            return "linear";
        case BlendMode::Mask:
            return "mask";
    }
    return "override";
}

BlendMode blendModeFromName(const std::string& name) {
    if (name == "override") {
        return BlendMode::Override;
    }
    if (name == "linear") {
        return BlendMode::Linear;
    }
    if (name == "mask") {
        return BlendMode::Mask;
    }
    throw AppError(E_MAT_PARSE, "unsupported blendMode '" + name + "'");
}

Texture2D::Texture2D(Image image) : image_(std::move(image)) {}

TextureSample Texture2D::sample(float u, float v) const {
    if (image_.empty()) {
        return {};
    }
    const float wrappedU = u - std::floor(u);
    const float wrappedV = v - std::floor(v);
    const float x = wrappedU * static_cast<float>(image_.width) - 0.5F;
    const float y = wrappedV * static_cast<float>(image_.height) - 0.5F;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);

    auto fetch = [&](int px, int py) {
        const int wrappedX = ((px % image_.width) + image_.width) % image_.width;
        const int wrappedY = ((py % image_.height) + image_.height) % image_.height;
        const std::uint8_t* pixel =
            image_.pixels.data() +
            (static_cast<std::size_t>(wrappedY) * static_cast<std::size_t>(image_.width) +
             static_cast<std::size_t>(wrappedX)) *
                4U;
        return TextureSample{
            Vec3(static_cast<float>(pixel[0]) / 255.0F,
                 static_cast<float>(pixel[1]) / 255.0F,
                 static_cast<float>(pixel[2]) / 255.0F),
            static_cast<float>(pixel[3]) / 255.0F};
    };

    const TextureSample topLeft = fetch(x0, y0);
    const TextureSample topRight = fetch(x0 + 1, y0);
    const TextureSample bottomLeft = fetch(x0, y0 + 1);
    const TextureSample bottomRight = fetch(x0 + 1, y0 + 1);
    const Vec3 top = mix(topLeft.rgb, topRight.rgb, tx);
    const Vec3 bottom = mix(bottomLeft.rgb, bottomRight.rgb, tx);
    const float alphaTop = mix(topLeft.alpha, topRight.alpha, tx);
    const float alphaBottom = mix(bottomLeft.alpha, bottomRight.alpha, tx);
    return {mix(top, bottom, ty), mix(alphaTop, alphaBottom, ty)};
}

void MaterialLibrary::add(LoadedMaterial material) {
    materials_.push_back(std::move(material));
}

const LoadedMaterial* MaterialLibrary::find(const std::string& name) const {
    for (const LoadedMaterial& material : materials_) {
        if (material.name == name) {
            return &material;
        }
    }
    return nullptr;
}

const LoadedMaterial& MaterialLibrary::require(const std::string& name) const {
    const LoadedMaterial* material = find(name);
    if (material == nullptr) {
        throw AppError(E_MAT_PARSE, "required material is unavailable: " + name);
    }
    return *material;
}

std::vector<MaterialSpec> parseMaterialDirectory(const fs::path& materialDirectory) {
    std::vector<fs::path> files;
    std::error_code error;
    for (fs::directory_iterator iterator(materialDirectory, error), end; iterator != end;
         iterator.increment(error)) {
        if (error) {
            throw AppError(error == std::errc::permission_denied ? E_IO_DENIED : E_IO,
                           "cannot enumerate materials directory '" +
                               materialDirectory.string() + "': " + error.message());
        }
        if (iterator->is_regular_file() && iterator->path().extension() == ".json") {
            files.push_back(iterator->path());
        }
    }
    if (error) {
        throw AppError(error == std::errc::permission_denied ? E_IO_DENIED : E_IO,
                       "cannot enumerate materials directory '" + materialDirectory.string() +
                           "': " + error.message());
    }
    std::sort(files.begin(), files.end());

    std::vector<MaterialSpec> materials;
    std::set<std::string> names;
    for (const fs::path& file : files) {
        try {
            MaterialSpec material = parseMaterialFile(file);
            if (!names.insert(material.name).second) {
                throw AppError(E_MAT_PARSE,
                               "duplicate material name '" + material.name + "'");
            }
            materials.push_back(std::move(material));
        } catch (const AppError& errorValue) {
            if (std::string(errorValue.code()) != E_MAT_PARSE) {
                throw;
            }
            std::cerr << errorValue.code() << ": " << errorValue.what() << '\n';
        }
    }
    return materials;
}

MaterialLibrary loadMaterialLibrary(const std::vector<MaterialSpec>& specs) {
    MaterialLibrary library;
    for (const MaterialSpec& spec : specs) {
        LoadedMaterial material;
        material.name = spec.name;
        material.sourceFile = spec.sourceFile;
        for (const MaterialLayerSpec& layerSpec : spec.layers) {
            LoadedLayer layer;
            layer.spec = layerSpec;
            layer.baseColor = Texture2D(loadPng(layerSpec.baseColorTexture));
            layer.metallic = Texture2D(loadPng(layerSpec.metallicTexture));
            layer.roughness = Texture2D(loadPng(layerSpec.roughnessTexture));
            layer.normal = Texture2D(loadPng(layerSpec.normalTexture));
            material.layers.push_back(std::move(layer));
        }
        library.add(std::move(material));
    }
    return library;
}

MaterialSurface composeMaterial(const LoadedMaterial& material, float u, float v) {
    MaterialSurface surface;
    bool hasBaseValue = false;
    for (const LoadedLayer& layer : material.layers) {
        const TextureSample baseSample = layer.baseColor.sample(u, v);
        const TextureSample metalSample = layer.metallic.sample(u, v);
        const TextureSample roughSample = layer.roughness.sample(u, v);
        const TextureSample normalSample = layer.normal.sample(u, v);

        float weight = layer.spec.opacity;
        if (layer.spec.blendMode == BlendMode::Linear) {
            weight *= baseSample.alpha;
        } else if (layer.spec.blendMode == BlendMode::Mask) {
            weight *= baseSample.alpha * luminance(baseSample.rgb);
        }
        weight = clamp(weight, 0.0F, 1.0F);

        const Vec3 tangentNormal = normalize(
            Vec3(normalSample.rgb.x * 2.0F - 1.0F, normalSample.rgb.y * 2.0F - 1.0F,
                 normalSample.rgb.z * 2.0F - 1.0F));
        if (!hasBaseValue) {
            surface.baseColor = baseSample.rgb;
            surface.metallic = metalSample.rgb.x;
            surface.roughness = roughSample.rgb.x;
            surface.tangentNormal = tangentNormal;
            hasBaseValue = true;
            continue;
        }
        surface.baseColor = mix(surface.baseColor, baseSample.rgb, weight);
        surface.metallic = mix(surface.metallic, metalSample.rgb.x, weight);
        surface.roughness = mix(surface.roughness, roughSample.rgb.x, weight);
        surface.tangentNormal =
            normalize(mix(surface.tangentNormal, tangentNormal, weight));
    }
    surface.metallic = clamp(surface.metallic, 0.0F, 1.0F);
    surface.roughness = clamp(surface.roughness, 0.03F, 1.0F);
    surface.tangentNormal = normalize(surface.tangentNormal);
    return surface;
}

}  // namespace cr

