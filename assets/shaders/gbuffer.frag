#version 430 core

in vec3 vWorldPosition;
in vec3 vWorldNormal;
in vec3 vWorldTangent;
in vec2 vUv;

layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec4 gNormal;
layout(location = 2) out vec4 gAlbedo;
layout(location = 3) out vec4 gLayerWeights;
layout(location = 4) out vec2 gRoughnessMetallic;

uniform sampler2D uBaseColorTextures[3];
uniform sampler2D uNormalTextures[3];
uniform sampler2D uRoughnessTextures[3];
uniform sampler2D uMetallicTextures[3];
uniform vec3 uLayerWeights;

void main() {
  vec3 weights = max(uLayerWeights, vec3(0.0));
  const float weightSum = dot(weights, vec3(1.0));
  weights = weightSum > 0.000001 ? weights / weightSum : vec3(1.0, 0.0, 0.0);

  const vec3 geometricNormal = normalize(vWorldNormal);
  vec3 tangent = vWorldTangent - geometricNormal * dot(geometricNormal, vWorldTangent);
  if (length(tangent) < 0.00001) {
    tangent = abs(geometricNormal.y) < 0.9 ? cross(geometricNormal, vec3(0.0, 1.0, 0.0))
                                         : cross(geometricNormal, vec3(1.0, 0.0, 0.0));
  }
  tangent = normalize(tangent);
  const vec3 bitangent = normalize(cross(geometricNormal, tangent));

  vec3 layeredNormal = vec3(0.0);
  vec3 albedo = vec3(0.0);
  float roughness = 0.0;
  float metallic = 0.0;
  for (int layer = 0; layer < 3; ++layer) {
    const float weight = weights[layer];
    const vec3 tangentNormal = texture(uNormalTextures[layer], vUv).xyz * 2.0 - 1.0;
    const vec3 worldNormal =
        normalize(tangent * tangentNormal.x + bitangent * tangentNormal.y +
                  geometricNormal * max(tangentNormal.z, 0.05));
    layeredNormal += worldNormal * weight;
    albedo += texture(uBaseColorTextures[layer], vUv).rgb * weight;
    roughness += texture(uRoughnessTextures[layer], vUv).r * weight;
    metallic += texture(uMetallicTextures[layer], vUv).r * weight;
  }
  if (length(layeredNormal) < 0.00001) {
    layeredNormal = geometricNormal;
  }

  gPosition = vec4(vWorldPosition, 1.0);
  gNormal = vec4(normalize(layeredNormal), 1.0);
  gAlbedo = vec4(clamp(albedo, 0.0, 1.0), 1.0);
  gLayerWeights = vec4(weights, 1.0);
  gRoughnessMetallic = clamp(vec2(roughness, metallic), 0.0, 1.0);
}

