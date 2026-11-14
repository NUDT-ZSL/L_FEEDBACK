#pragma once

#include "common.hpp"

#include <array>
#include <string>
#include <vector>

namespace cr {

enum class BlendMode : int {
    Override = 0,
    Linear = 1,
    Mask = 2,
};

const char* blendModeName(BlendMode mode);
BlendMode blendModeFromName(const std::string& name);

struct MaterialLayerSpec {
    std::string name;
    fs::path baseColorTexture;
    fs::path metallicTexture;
    fs::path roughnessTexture;
    fs::path normalTexture;
    BlendMode blendMode = BlendMode::Override;
    float opacity = 1.0F;
};

struct MaterialSpec {
    std::string name;
    fs::path sourceFile;
    std::vector<MaterialLayerSpec> layers;
};

struct TextureSample {
    Vec3 rgb;
    float alpha = 1.0F;
};

class Texture2D {
public:
    Texture2D() = default;
    explicit Texture2D(Image image);

    [[nodiscard]] const Image& image() const noexcept { return image_; }
    [[nodiscard]] TextureSample sample(float u, float v) const;

private:
    Image image_;
};

struct LoadedLayer {
    MaterialLayerSpec spec;
    Texture2D baseColor;
    Texture2D metallic;
    Texture2D roughness;
    Texture2D normal;
};

struct LoadedMaterial {
    std::string name;
    fs::path sourceFile;
    std::vector<LoadedLayer> layers;
};

struct MaterialSurface {
    Vec3 baseColor{0.5F, 0.5F, 0.5F};
    float metallic = 0.0F;
    float roughness = 0.5F;
    Vec3 tangentNormal{0.0F, 0.0F, 1.0F};
};

class MaterialLibrary {
public:
    void add(LoadedMaterial material);
    [[nodiscard]] const LoadedMaterial* find(const std::string& name) const;
    [[nodiscard]] const LoadedMaterial& require(const std::string& name) const;
    [[nodiscard]] const std::vector<LoadedMaterial>& materials() const noexcept {
        return materials_;
    }

private:
    std::vector<LoadedMaterial> materials_;
};

std::vector<MaterialSpec> parseMaterialDirectory(const fs::path& materialDirectory);
MaterialLibrary loadMaterialLibrary(const std::vector<MaterialSpec>& specs);
MaterialSurface composeMaterial(const LoadedMaterial& material, float u, float v);

}  // namespace cr

