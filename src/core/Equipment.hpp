#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/Math.hpp"

namespace lcr {

struct Material {
  Vec3 baseColor{0.5F, 0.5F, 0.5F};
  float metallic = 0.0F;
  float roughness = 0.7F;
  Vec3 emissiveColor{0.0F, 0.0F, 0.0F};
  float emissiveStrength = 0.0F;
  float opacity = 1.0F;
};

struct TextureData {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> rgba;
  std::uint64_t hash = 0;
};

struct EquipmentSet {
  std::string id;
  std::string displayName;
  Material bodyMaterial;
  Material weaponMaterial;
  Material capeMaterial;
  TextureData bodyTexture;
  TextureData weaponTexture;
  std::uint64_t hash = 0;
};

struct EquipmentBuildOptions {
  std::uint64_t seed = 1;
  std::uint32_t textureSize = 128;
  bool forceTextureFailure = false;
};

class EquipmentCatalog {
 public:
  EquipmentCatalog() = default;

  void build(const EquipmentBuildOptions& options);

  const std::vector<EquipmentSet>& sets() const { return sets_; }
  const EquipmentSet& active() const;
  const EquipmentSet& setAt(std::size_t index) const;
  std::size_t activeIndex() const { return activeIndex_; }
  bool switchTo(std::size_t index);
  bool switchById(const std::string& id);
  bool toggle();
  std::uint64_t catalogHash() const { return catalogHash_; }

 private:
  std::vector<EquipmentSet> sets_;
  std::size_t activeIndex_ = 0;
  std::uint64_t catalogHash_ = 0;
};

TextureData generateEquipmentTexture(
    std::uint64_t seed,
    std::uint32_t setIndex,
    bool weapon,
    std::uint32_t size,
    bool forceFailure);

std::vector<std::uint8_t> textureCanonicalBytes(const TextureData& texture);

}  // namespace lcr

