#pragma once

#include "lf/config.h"

#include <glm/glm.hpp>

namespace lf {

struct BrdfTerms {
  glm::vec3 diffuse{0.0F};
  glm::vec3 specular{0.0F};
  glm::vec3 cloth{0.0F};
  glm::vec3 total{0.0F};
};

struct SurfaceSample {
  glm::vec3 normal{0.0F, 1.0F, 0.0F};
  glm::vec3 viewDirection{0.0F, 0.0F, 1.0F};
  glm::vec3 albedo{0.8F};
  float roughness = 0.5F;
  float metallic = 0.0F;
  glm::vec3 layerWeights{0.55F, 0.28F, 0.17F};
};

float distribution_ggx(float noh, float roughness);
float distribution_charlie(float noh, float roughness);
float geometry_smith_ggx(float nov, float nol, float roughness);
glm::vec3 fresnel_schlick(float cosine, const glm::vec3& f0);
glm::vec3 evaluate_layered_brdf(const SurfaceSample& surface, const glm::vec3& lightDirection,
                                const glm::vec3& radiance);

glm::vec3 evaluate_light(const Light& light, const glm::vec3& worldPosition,
                         const SurfaceSample& surface);

}  // namespace lf

