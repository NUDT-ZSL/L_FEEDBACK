#pragma once

namespace orv::shaders {

inline constexpr const char* kGeometryVertex = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aTangent;
layout(location = 3) in vec2 aUv;

uniform mat4 uModel;
uniform mat4 uViewProjection;
uniform mat4 uView;

out vec3 vWorldPosition;
out vec3 vNormal;
out vec3 vTangent;
out vec2 vUv;
out float vViewDepth;

void main() {
  vec4 world = uModel * vec4(aPosition, 1.0);
  mat3 normalMatrix = transpose(inverse(mat3(uModel)));
  vWorldPosition = world.xyz;
  vNormal = normalMatrix * aNormal;
  vTangent = mat3(uModel) * aTangent;
  vUv = aUv;
  vViewDepth = -(uView * world).z;
  gl_Position = uViewProjection * world;
}
)GLSL";

inline constexpr const char* kGeometryFragment = R"GLSL(
#version 330 core
layout(location = 0) out vec4 oAlbedoMetallic;
layout(location = 1) out vec4 oNormalRoughness;
layout(location = 2) out vec4 oPositionDepth;
layout(location = 3) out vec4 oTangentAnisotropy;
layout(location = 4) out vec4 oAnisotropyRotation;

uniform vec3 uBaseColor;
uniform float uRoughness;
uniform float uMetallic;
uniform float uAnisotropyStrength;
uniform float uAnisotropyRotation;

in vec3 vWorldPosition;
in vec3 vNormal;
in vec3 vTangent;
in vec2 vUv;
in float vViewDepth;

void main() {
  vec3 normal = normalize(vNormal);
  vec3 tangent = normalize(vTangent);
  oAlbedoMetallic = vec4(uBaseColor, uMetallic);
  oNormalRoughness = vec4(normal, uRoughness);
  oPositionDepth = vec4(vWorldPosition, vViewDepth);
  oTangentAnisotropy = vec4(tangent, uAnisotropyStrength);
  oAnisotropyRotation = vec4(uAnisotropyRotation, vUv.x, vUv.y, 0.0);
}
)GLSL";

inline constexpr const char* kShadowVertex = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPosition;

uniform mat4 uModel;
uniform mat4 uLightViewProjection;

out vec3 vWorldPosition;

void main() {
  vec4 world = uModel * vec4(aPosition, 1.0);
  vWorldPosition = world.xyz;
  gl_Position = uLightViewProjection * world;
}
)GLSL";

inline constexpr const char* kShadowFragment = R"GLSL(
#version 330 core
uniform int uRadialDistance;
uniform vec3 uLightPosition;
uniform float uLightFar;

in vec3 vWorldPosition;

void main() {
  float depth = gl_FragCoord.z;
  if (uRadialDistance == 1) {
    depth = clamp(length(vWorldPosition - uLightPosition) / uLightFar, 0.0, 1.0);
  }
  gl_FragDepth = depth;
}
)GLSL";

inline constexpr const char* kFullscreenVertex = R"GLSL(
#version 330 core
out vec2 vUv;

void main() {
  vec2 position = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = position;
  gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

inline constexpr const char* kDeferredFragment = R"GLSL(
#version 330 core
const float PI = 3.14159265359;

in vec2 vUv;
out vec4 oColor;

uniform sampler2D uAlbedoMetallic;
uniform sampler2D uNormalRoughness;
uniform sampler2D uPositionDepth;
uniform sampler2D uTangentAnisotropy;
uniform sampler2D uAnisotropyRotation;

uniform vec3 uCameraPosition;
uniform int uHasDirectional;
uniform vec3 uDirectionalDirection;
uniform vec3 uDirectionalColor;
uniform float uDirectionalIntensity;
uniform mat4 uDirectionalViewProjection;
uniform sampler2D uDirectionalShadow;

uniform int uPointLightCount;
uniform vec3 uPointPositions[4];
uniform vec3 uPointColors[4];
uniform float uPointIntensities[4];
uniform float uPointRanges[4];
uniform float uPointFars[4];
uniform samplerCube uPointShadow0;
uniform samplerCube uPointShadow1;
uniform samplerCube uPointShadow2;
uniform samplerCube uPointShadow3;

uniform int uShadowMapSize;
uniform float uShadowBias;

float directionalVisibility(vec3 position, vec3 normal) {
  if (uHasDirectional == 0) {
    return 1.0;
  }
  vec3 offsetPosition = position + normal * max(uShadowBias * 2.0, 0.0001);
  vec4 clip = uDirectionalViewProjection * vec4(offsetPosition, 1.0);
  if (clip.w <= 0.0) {
    return 1.0;
  }
  vec3 coordinate = clip.xyz / clip.w * 0.5 + 0.5;
  if (coordinate.x <= 0.0 || coordinate.x >= 1.0 ||
      coordinate.y <= 0.0 || coordinate.y >= 1.0 ||
      coordinate.z <= 0.0 || coordinate.z >= 1.0) {
    return 1.0;
  }
  float bias = max(uShadowBias, 0.00001);
  float texel = 1.0 / float(uShadowMapSize);
  float visible = 0.0;
  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      float sampleDepth = texture(
          uDirectionalShadow,
          coordinate.xy + vec2(float(x), float(y)) * texel).r;
      visible += coordinate.z - bias <= sampleDepth ? 1.0 : 0.0;
    }
  }
  return visible / 9.0;
}

void pointBasis(vec3 direction, out vec3 right, out vec3 up) {
  vec3 absoluteDirection = abs(direction);
  if (absoluteDirection.x >= absoluteDirection.y &&
      absoluteDirection.x >= absoluteDirection.z) {
    right = normalize(vec3(0.0, 0.0, direction.x >= 0.0 ? -1.0 : 1.0));
    up = vec3(0.0, 1.0, 0.0);
  } else if (absoluteDirection.y >= absoluteDirection.z) {
    right = vec3(1.0, 0.0, 0.0);
    up = normalize(vec3(0.0, 0.0, direction.y >= 0.0 ? 1.0 : -1.0));
  } else {
    right = normalize(vec3(direction.z >= 0.0 ? 1.0 : -1.0, 0.0, 0.0));
    up = vec3(0.0, 1.0, 0.0);
  }
}

float pointVisibility(vec3 position, vec3 normal, vec3 lightPosition, float lightFar,
                      samplerCube shadowMap) {
  vec3 offsetPosition = position + normal * max(uShadowBias * 4.0, 0.001);
  vec3 lightToPosition = offsetPosition - lightPosition;
  float distance = length(lightToPosition);
  if (distance <= 1.0e-5 || distance >= lightFar) {
    return 1.0;
  }
  vec3 direction = lightToPosition / distance;
  vec3 right;
  vec3 up;
  pointBasis(direction, right, up);
  float spread = 1.6 / float(uShadowMapSize);
  float bias = max(uShadowBias, 0.00002);
  float current = distance / lightFar;
  float visible = 0.0;
  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      vec3 sampleDirection =
          normalize(direction + (right * float(x) + up * float(y)) * spread);
      float sampleDepth = texture(shadowMap, sampleDirection).r;
      visible += current - bias <= sampleDepth ? 1.0 : 0.0;
    }
  }
  return visible / 9.0;
}

float pointVisibilityByIndex(int index, vec3 position, vec3 normal,
                             vec3 lightPosition, float lightFar) {
  if (index == 0) {
    return pointVisibility(position, normal, lightPosition, lightFar, uPointShadow0);
  }
  if (index == 1) {
    return pointVisibility(position, normal, lightPosition, lightFar, uPointShadow1);
  }
  if (index == 2) {
    return pointVisibility(position, normal, lightPosition, lightFar, uPointShadow2);
  }
  return pointVisibility(position, normal, lightPosition, lightFar, uPointShadow3);
}

float distributionGGXAnisotropic(vec3 normal, vec3 halfVector, vec3 tangent,
                                  float roughness, float anisotropy) {
  float alpha = max(roughness * roughness, 0.001);
  float aspect = sqrt(clamp(1.0 - 0.9 * anisotropy, 0.05, 1.0));
  float alphaTangent = max(alpha / aspect, 0.0005);
  float alphaBitangent = max(alpha * aspect, 0.0005);
  vec3 bitangent = cross(normal, tangent);
  float hTangent = dot(halfVector, tangent);
  float hBitangent = dot(halfVector, bitangent);
  float hNormal = dot(halfVector, normal);
  float denominator = (hTangent * hTangent) / (alphaTangent * alphaTangent) +
                      (hBitangent * hBitangent) / (alphaBitangent * alphaBitangent) +
                      hNormal * hNormal;
  return 1.0 / max(PI * alphaTangent * alphaBitangent * denominator * denominator,
                   1.0e-8);
}

float geometrySchlickGGX(float nDotV, float roughness) {
  float r = roughness + 1.0;
  float k = (r * r) / 8.0;
  return nDotV / max(nDotV * (1.0 - k) + k, 1.0e-6);
}

vec3 evaluateLight(vec3 normal, vec3 viewDirection, vec3 tangent, vec3 albedo,
                   float metallic, float roughness, float anisotropy,
                   vec3 lightDirection, vec3 radiance, float visibility) {
  float nDotL = max(dot(normal, lightDirection), 0.0);
  float nDotV = max(dot(normal, viewDirection), 0.0001);
  if (nDotL <= 0.0) {
    return vec3(0.0);
  }
  vec3 halfVector = normalize(lightDirection + viewDirection);
  float distribution =
      distributionGGXAnisotropic(normal, halfVector, tangent, roughness, anisotropy);
  float geometry = geometrySchlickGGX(nDotV, roughness) *
                   geometrySchlickGGX(nDotL, roughness);
  vec3 f0 = mix(vec3(0.04), albedo, metallic);
  vec3 fresnel = f0 + (1.0 - f0) *
                          pow(clamp(1.0 - dot(halfVector, viewDirection), 0.0, 1.0), 5.0);
  vec3 specular = distribution * geometry * fresnel /
                  max(4.0 * nDotV * nDotL, 0.001);
  vec3 diffuse = (1.0 - fresnel) * albedo * (1.0 - metallic) / PI;
  return (diffuse + specular) * radiance * nDotL * visibility;
}

void main() {
  vec4 albedoMetallic = texture(uAlbedoMetallic, vUv);
  vec4 normalRoughness = texture(uNormalRoughness, vUv);
  vec4 positionDepth = texture(uPositionDepth, vUv);

  if (positionDepth.a <= 0.0001) {
    oColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  vec3 position = positionDepth.xyz;
  vec3 normal = normalize(normalRoughness.xyz);
  vec3 viewDirection = normalize(uCameraPosition - position);
  vec3 albedo = albedoMetallic.rgb;
  float metallic = clamp(albedoMetallic.a, 0.0, 1.0);
  float roughness = clamp(normalRoughness.a, 0.04, 1.0);
  vec4 tangentAnisotropy = texture(uTangentAnisotropy, vUv);
  float anisotropy = clamp(tangentAnisotropy.a, 0.0, 1.0);
  float rotation = texture(uAnisotropyRotation, vUv).r;
  vec3 tangent = tangentAnisotropy.xyz;
  tangent = normalize(tangent - normal * dot(normal, tangent));
  vec3 bitangent = cross(normal, tangent);
  tangent = normalize(cos(rotation) * tangent + sin(rotation) * bitangent);

  vec3 color = vec3(0.0);
  float directionalShadow = 1.0;
  float combinedPointShadow = 1.0;

  if (uHasDirectional == 1) {
    vec3 lightDirection = normalize(-uDirectionalDirection);
    directionalShadow = directionalVisibility(position, normal);
    color += evaluateLight(normal, viewDirection, tangent, albedo, metallic, roughness,
                           anisotropy, lightDirection,
                           uDirectionalColor * uDirectionalIntensity,
                           directionalShadow);
  }

  for (int i = 0; i < 4; ++i) {
    if (i >= uPointLightCount) {
      break;
    }
    vec3 delta = uPointPositions[i] - position;
    float distanceSquared = dot(delta, delta);
    float range = uPointRanges[i];
    if (distanceSquared >= range * range) {
      continue;
    }
    float distance = sqrt(max(distanceSquared, 1.0e-6));
    vec3 lightDirection = delta / distance;
    float window = clamp(1.0 - pow(distance / range, 4.0), 0.0, 1.0);
    float attenuation = window * window / max(distanceSquared, 0.01);
    float visibility = pointVisibilityByIndex(i, position, normal, uPointPositions[i],
                                              uPointFars[i]);
    combinedPointShadow = min(combinedPointShadow, visibility);
    vec3 radiance = uPointColors[i] * uPointIntensities[i] * attenuation;
    color += evaluateLight(normal, viewDirection, tangent, albedo, metallic, roughness,
                           anisotropy, lightDirection, radiance, visibility);
  }

  float aggregateShadow = directionalShadow * combinedPointShadow;
  oColor = vec4(max(color, vec3(0.0)), clamp(aggregateShadow, 0.0, 1.0));
}
)GLSL";

}  // namespace orv::shaders
