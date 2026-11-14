#include "lf/brdf.h"

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>

namespace lf {

namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kEpsilon = 1.0e-6F;

glm::vec3 safe_normalize(const glm::vec3& value) {
  const float length = glm::length(value);
  return length > kEpsilon ? value / length : glm::vec3(0.0F);
}

}  // namespace

float distribution_ggx(float noh, float roughness) {
  const float alpha = std::clamp(roughness * roughness, 0.0025F, 1.0F);
  const float alpha2 = alpha * alpha;
  const float denominator = noh * noh * (alpha2 - 1.0F) + 1.0F;
  return alpha2 / (kPi * denominator * denominator + kEpsilon);
}

float distribution_charlie(float noh, float roughness) {
  const float alpha = std::clamp(roughness, 0.05F, 1.0F);
  const float inverseAlpha = 1.0F / alpha;
  const float sinTheta2 = std::max(0.0F, 1.0F - noh * noh);
  const float sine = std::sqrt(sinTheta2);
  return (2.0F + inverseAlpha) * std::pow(sine, inverseAlpha) / (2.0F * kPi);
}

float geometry_smith_ggx(float nov, float nol, float roughness) {
  const float alpha = std::clamp(roughness * roughness, 0.0025F, 1.0F);
  const float alpha2 = alpha * alpha;
  const float g1v =
      2.0F * nov / std::max(nov + std::sqrt(alpha2 + (1.0F - alpha2) * nov * nov),
                            kEpsilon);
  const float g1l =
      2.0F * nol / std::max(nol + std::sqrt(alpha2 + (1.0F - alpha2) * nol * nol),
                            kEpsilon);
  return g1v * g1l;
}

glm::vec3 fresnel_schlick(float cosine, const glm::vec3& f0) {
  const float factor = std::pow(std::clamp(1.0F - cosine, 0.0F, 1.0F), 5.0F);
  return f0 + (glm::vec3(1.0F) - f0) * factor;
}

glm::vec3 evaluate_layered_brdf(const SurfaceSample& surface,
                                const glm::vec3& lightDirection,
                                const glm::vec3& radiance) {
  const glm::vec3 normal = safe_normalize(surface.normal);
  const glm::vec3 view = safe_normalize(surface.viewDirection);
  const glm::vec3 light = safe_normalize(lightDirection);
  const float nol = glm::dot(normal, light);
  const float nov = std::max(glm::dot(normal, view), kEpsilon);
  if (nol <= 0.0F || glm::length(normal) < 0.5F || glm::length(light) < 0.5F) {
    return glm::vec3(0.0F);
  }

  const glm::vec3 halfVector = safe_normalize(view + light);
  const float noh = std::max(glm::dot(normal, halfVector), 0.0F);
  const float voh = std::max(glm::dot(view, halfVector), 0.0F);
  const float roughness = std::clamp(surface.roughness, 0.04F, 1.0F);
  const float metallic = std::clamp(surface.metallic, 0.0F, 1.0F);
  glm::vec3 weights = glm::max(surface.layerWeights, glm::vec3(0.0F));
  const float weightSum = weights.x + weights.y + weights.z;
  weights = weightSum > kEpsilon ? weights / weightSum : glm::vec3(1.0F, 0.0F, 0.0F);

  const glm::vec3 diffuseColor = surface.albedo * (1.0F - metallic);
  const glm::vec3 diffuse = diffuseColor / kPi;
  const glm::vec3 f0 = glm::mix(glm::vec3(0.04F), surface.albedo, metallic);
  const float distribution = distribution_ggx(noh, roughness);
  const float geometry = geometry_smith_ggx(nov, nol, roughness);
  const glm::vec3 fresnel = fresnel_schlick(voh, f0);
  const glm::vec3 specular =
      distribution * geometry * fresnel / std::max(4.0F * nov * nol, kEpsilon);

  const float charlie = distribution_charlie(noh, roughness);
  const float sheenVisibility =
      1.0F / std::max(4.0F * (nov + nol - nov * nol), kEpsilon);
  const glm::vec3 sheenColor = glm::mix(glm::vec3(0.35F), surface.albedo, 0.55F);
  const glm::vec3 cloth = charlie * sheenVisibility * sheenColor;

  const glm::vec3 weighted =
      weights.x * diffuse + weights.y * specular + weights.z * cloth;
  return weighted * radiance * nol;
}

glm::vec3 evaluate_light(const Light& light, const glm::vec3& worldPosition,
                         const SurfaceSample& surface) {
  if (!light.enabled) {
    return glm::vec3(0.0F);
  }
  glm::vec3 direction{0.0F};
  glm::vec3 radiance = light.color * light.intensity;

  if (light.type == LightType::Directional) {
    direction = -safe_normalize(light.direction);
  } else {
    const glm::vec3 offset = light.position - worldPosition;
    const float distance = glm::length(offset);
    if (distance < kEpsilon || distance > light.range) {
      return glm::vec3(0.0F);
    }
    direction = offset / distance;
    const float denominator = light.constantAttenuation +
                              light.linearAttenuation * distance +
                              light.quadraticAttenuation * distance * distance;
    const float attenuation = 1.0F / std::max(denominator, kEpsilon);
    const float rangeRatio = distance / light.range;
    const float window =
        std::pow(std::clamp(1.0F - std::pow(rangeRatio, 4.0F), 0.0F, 1.0F), 2.0F);
    radiance *= attenuation * window;

    if (light.type == LightType::Spot) {
      const float cosAngle = glm::dot(-direction, safe_normalize(light.direction));
      const float inner = std::cos(light.innerConeDegrees * kPi / 180.0F);
      const float outer = std::cos(light.outerConeDegrees * kPi / 180.0F);
      const float edge = std::clamp((cosAngle - outer) / std::max(inner - outer, kEpsilon),
                                    0.0F, 1.0F);
      radiance *= edge * edge * (3.0F - 2.0F * edge);
    }
  }

  return evaluate_layered_brdf(surface, direction, radiance);
}

}  // namespace lf

