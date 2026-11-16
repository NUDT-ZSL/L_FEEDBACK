#include "core/Equipment.hpp"

#include <cmath>
#include <cstring>
#include <limits>

#include "core/Error.hpp"
#include "core/Serializer.hpp"

namespace lcr {
namespace {

std::uint32_t mix32(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7FEB352DU;
  value ^= value >> 15U;
  value *= 0x846CA68BU;
  return value ^ (value >> 16U);
}

float noise01(const std::uint32_t x, const std::uint32_t y, const std::uint32_t salt) {
  return static_cast<float>(mix32(x * 0x9E3779B9U + y * 0x85EBCA6BU + salt) >> 8U) *
         (1.0F / 16777216.0F);
}

std::uint8_t toByte(const float value) {
  return static_cast<std::uint8_t>(
      std::fmax(0.0F, std::fmin(255.0F, value * 255.0F + 0.5F)));
}

TextureData finishTexture(TextureData texture) {
  const std::vector<std::uint8_t> bytes = textureCanonicalBytes(texture);
  texture.hash = hashBytes(bytes);
  return texture;
}

void buildTexturesForSet(
    EquipmentSet& set,
    const std::size_t setIndex,
    const EquipmentBuildOptions& options) {
  set.bodyTexture = generateEquipmentTexture(
      options.seed, static_cast<std::uint32_t>(setIndex), false,
      options.textureSize, options.forceTextureFailure);
  set.weaponTexture = generateEquipmentTexture(
      options.seed, static_cast<std::uint32_t>(setIndex), true,
      options.textureSize, options.forceTextureFailure);
}

}  // namespace

std::vector<std::uint8_t> textureCanonicalBytes(const TextureData& texture) {
  CanonicalWriter writer;
  writer.string("lcr-texture-v1");
  writer.u32(texture.width);
  writer.u32(texture.height);
  writer.u32(static_cast<std::uint32_t>(texture.rgba.size()));
  for (const std::uint8_t value : texture.rgba) {
    writer.u8(value);
  }
  return writer.bytes();
}

TextureData generateEquipmentTexture(
    const std::uint64_t seed,
    const std::uint32_t setIndex,
    const bool weapon,
    const std::uint32_t size,
    const bool forceFailure) {
  if (forceFailure) {
    fail(
        ErrorCode::TextureGeneration,
        weapon ? "equipment.texture.weapon" : "equipment.texture.body",
        "forced texture generation failure");
  }
  if (size < 16U || size > 1024U || (size & (size - 1U)) != 0U) {
    fail(
        ErrorCode::TextureGeneration,
        "equipment.texture.validate",
        "texture size must be a power of two in [16, 1024]");
  }
  const std::size_t pixelCount = static_cast<std::size_t>(size) * size;
  if (pixelCount > std::numeric_limits<std::size_t>::max() / 4U) {
    fail(ErrorCode::TextureGeneration, "equipment.texture.allocate", "texture size overflow");
  }

  TextureData texture;
  texture.width = size;
  texture.height = size;
  texture.rgba.resize(pixelCount * 4U);
  const std::uint32_t salt =
      static_cast<std::uint32_t>(seed) ^
      (static_cast<std::uint32_t>(seed >> 32U) * 0x27D4EB2DU) ^
      (setIndex * 0xC2B2AE35U) ^
      (weapon ? 0xA511E9B3U : 0x63D83595U);

  for (std::uint32_t y = 0; y < size; ++y) {
    const float v = (static_cast<float>(y) + 0.5F) / static_cast<float>(size);
    for (std::uint32_t x = 0; x < size; ++x) {
      const float u = (static_cast<float>(x) + 0.5F) / static_cast<float>(size);
      const float n0 = noise01(x / 2U, y / 2U, salt);
      const float n1 = noise01(x / 8U, y / 8U, salt ^ 0x9E3779B9U);
      const float weave = 0.5F + 0.5F * std::sin((u * 18.0F + v * 5.0F) * kPi * 2.0F);
      const float rune = std::fabs(std::sin((u * 3.0F + v * 2.0F + n1 * 0.3F) * kPi * 4.0F));
      const float border = std::fmin(
          std::fmin(u, 1.0F - u),
          std::fmin(v, 1.0F - v));
      const float edge = std::fmax(0.0F, std::fmin(1.0F, border * 14.0F));
      float r = 0.0F;
      float g = 0.0F;
      float b = 0.0F;
      float glow = 0.0F;

      if (setIndex % 2U == 0U) {
        const float plate = 0.26F + 0.14F * n1 + 0.08F * weave;
        r = plate * 0.78F + 0.03F * n0;
        g = plate * 0.92F + 0.05F * n0;
        b = plate * 1.20F + 0.10F * n0;
        const float line = std::pow(std::fmax(0.0F, rune - 0.93F) / 0.07F, 2.0F);
        glow = line * (weapon ? 1.0F : 0.35F);
        r += glow * 0.10F;
        g += glow * 0.55F;
        b += glow * 1.0F;
      } else {
        const float plate = 0.24F + 0.13F * n1 + 0.06F * weave;
        r = plate * 1.45F + 0.05F * n0;
        g = plate * 0.47F + 0.05F * n0;
        b = plate * 0.20F + 0.02F * n0;
        const float line = std::pow(std::fmax(0.0F, rune - 0.91F) / 0.09F, 2.0F);
        glow = line * (weapon ? 1.0F : 0.45F);
        r += glow * 1.0F;
        g += glow * 0.26F;
        b += glow * 0.02F;
      }
      r *= 0.72F + 0.28F * edge;
      g *= 0.72F + 0.28F * edge;
      b *= 0.72F + 0.28F * edge;
      glow *= 0.55F + 0.45F * edge;

      const std::size_t index =
          (static_cast<std::size_t>(y) * size + x) * 4U;
      texture.rgba[index + 0U] = toByte(r);
      texture.rgba[index + 1U] = toByte(g);
      texture.rgba[index + 2U] = toByte(b);
      texture.rgba[index + 3U] = toByte(glow);
    }
  }
  return finishTexture(std::move(texture));
}

void EquipmentCatalog::build(const EquipmentBuildOptions& options) {
  sets_.clear();
  activeIndex_ = 0;
  sets_.reserve(2);

  EquipmentSet iron;
  iron.id = "iron";
  iron.displayName = "Iron Warden";
  iron.bodyMaterial.baseColor = {0.47F, 0.57F, 0.70F};
  iron.bodyMaterial.metallic = 0.82F;
  iron.bodyMaterial.roughness = 0.30F;
  iron.bodyMaterial.emissiveColor = {0.08F, 0.35F, 1.0F};
  iron.bodyMaterial.emissiveStrength = 0.18F;
  iron.weaponMaterial.baseColor = {0.70F, 0.82F, 0.96F};
  iron.weaponMaterial.metallic = 0.92F;
  iron.weaponMaterial.roughness = 0.20F;
  iron.weaponMaterial.emissiveColor = {0.10F, 0.55F, 1.0F};
  iron.weaponMaterial.emissiveStrength = 1.65F;
  iron.capeMaterial.baseColor = {0.16F, 0.24F, 0.40F};
  iron.capeMaterial.metallic = 0.0F;
  iron.capeMaterial.roughness = 0.86F;
  iron.capeMaterial.emissiveColor = {0.02F, 0.08F, 0.20F};
  iron.capeMaterial.emissiveStrength = 0.08F;
  iron.capeMaterial.opacity = 0.72F;

  EquipmentSet ember;
  ember.id = "ember";
  ember.displayName = "Ember Vanguard";
  ember.bodyMaterial.baseColor = {0.72F, 0.24F, 0.09F};
  ember.bodyMaterial.metallic = 0.58F;
  ember.bodyMaterial.roughness = 0.46F;
  ember.bodyMaterial.emissiveColor = {1.0F, 0.30F, 0.04F};
  ember.bodyMaterial.emissiveStrength = 0.52F;
  ember.weaponMaterial.baseColor = {0.95F, 0.44F, 0.16F};
  ember.weaponMaterial.metallic = 0.70F;
  ember.weaponMaterial.roughness = 0.33F;
  ember.weaponMaterial.emissiveColor = {1.0F, 0.18F, 0.015F};
  ember.weaponMaterial.emissiveStrength = 3.10F;
  ember.capeMaterial.baseColor = {0.42F, 0.095F, 0.04F};
  ember.capeMaterial.metallic = 0.0F;
  ember.capeMaterial.roughness = 0.78F;
  ember.capeMaterial.emissiveColor = {0.80F, 0.12F, 0.01F};
  ember.capeMaterial.emissiveStrength = 0.34F;
  ember.capeMaterial.opacity = 0.68F;

  sets_.push_back(std::move(iron));
  sets_.push_back(std::move(ember));
  for (std::size_t i = 0; i < sets_.size(); ++i) {
    buildTexturesForSet(sets_[i], i, options);
  }
  for (EquipmentSet& set : sets_) {
    CanonicalWriter writer;
    writer.string("lcr-equipment-v1");
    writer.string(set.id);
    writer.string(set.displayName);
    for (const Material* material : {&set.bodyMaterial, &set.weaponMaterial, &set.capeMaterial}) {
      writer.f32(material->baseColor.x);
      writer.f32(material->baseColor.y);
      writer.f32(material->baseColor.z);
      writer.f32(material->metallic);
      writer.f32(material->roughness);
      writer.f32(material->emissiveColor.x);
      writer.f32(material->emissiveColor.y);
      writer.f32(material->emissiveColor.z);
      writer.f32(material->emissiveStrength);
      writer.f32(material->opacity);
    }
    writer.u64(set.bodyTexture.hash);
    writer.u64(set.weaponTexture.hash);
    set.hash = hashBytes(writer.bytes());
  }

  CanonicalWriter catalog;
  catalog.string("lcr-equipment-catalog-v1");
  catalog.u32(static_cast<std::uint32_t>(sets_.size()));
  for (const EquipmentSet& set : sets_) {
    catalog.u64(set.hash);
  }
  catalogHash_ = hashBytes(catalog.bytes());
}

const EquipmentSet& EquipmentCatalog::active() const {
  if (sets_.empty() || activeIndex_ >= sets_.size()) {
    fail(ErrorCode::InternalInvariant, "equipment.active", "equipment catalog is empty");
  }
  return sets_[activeIndex_];
}

const EquipmentSet& EquipmentCatalog::setAt(const std::size_t index) const {
  if (index >= sets_.size()) {
    fail(ErrorCode::InvalidArgument, "equipment.setAt", "equipment index out of range");
  }
  return sets_[index];
}

bool EquipmentCatalog::switchTo(const std::size_t index) {
  if (index >= sets_.size()) {
    return false;
  }
  activeIndex_ = index;
  return true;
}

bool EquipmentCatalog::switchById(const std::string& id) {
  for (std::size_t i = 0; i < sets_.size(); ++i) {
    if (sets_[i].id == id) {
      activeIndex_ = i;
      return true;
    }
  }
  return false;
}

bool EquipmentCatalog::toggle() {
  if (sets_.size() < 2U) {
    return false;
  }
  activeIndex_ = (activeIndex_ + 1U) % sets_.size();
  return true;
}

}  // namespace lcr
