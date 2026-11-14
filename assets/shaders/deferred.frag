#version 430 core

in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uGPosition;
uniform sampler2D uGNormal;
uniform sampler2D uGAlbedo;
uniform sampler2D uGLayerWeights;
uniform sampler2D uGRoughnessMetallic;
uniform vec3 uCameraPosition;

const int kMaxLights = 32;
uniform int uLightCount;
uniform int uLightType[kMaxLights];
uniform vec3 uLightPosition[kMaxLights];
uniform vec3 uLightDirection[kMaxLights];
uniform vec3 uLightColor[kMaxLights];
uniform float uLightIntensity[kMaxLights];
uniform float uLightRange[kMaxLights];
uniform float uLightInnerCos[kMaxLights];
uniform float uLightOuterCos[kMaxLights];
uniform vec3 uLightAttenuation[kMaxLights];

const float kPi = 3.14159265358979323846;

float distributionGgx(float noh, float roughness) {
  const float alpha = clamp(roughness * roughness, 0.0025, 1.0);
  const float alpha2 = alpha * alpha;
  const float denominator = noh * noh * (alpha2 - 1.0) + 1.0;
  return alpha2 / (kPi * denominator * denominator + 0.000001);
}

float distributionCharlie(float noh, float roughness) {
  const float alpha = clamp(roughness, 0.05, 1.0);
  const float inverseAlpha = 1.0 / alpha;
  const float sinTheta2 = max(0.0, 1.0 - noh * noh);
  return (2.0 + inverseAlpha) * pow(sqrt(sinTheta2), inverseAlpha) / (2.0 * kPi);
}

float geometrySmithGgx(float nov, float nol, float roughness) {
  const float alpha = clamp(roughness * roughness, 0.0025, 1.0);
  const float alpha2 = alpha * alpha;
  const float g1v =
      2.0 * nov / max(nov + sqrt(alpha2 + (1.0 - alpha2) * nov * nov), 0.000001);
  const float g1l =
      2.0 * nol / max(nol + sqrt(alpha2 + (1.0 - alpha2) * nol * nol), 0.000001);
  return g1v * g1l;
}

vec3 fresnelSchlick(float cosine, vec3 f0) {
  return f0 + (vec3(1.0) - f0) * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

vec3 evaluateLayeredBrdf(vec3 normal, vec3 viewDirection, vec3 lightDirection,
                         vec3 albedo, float roughness, float metallic, vec3 weights,
                         vec3 radiance) {
  const float nol = dot(normal, lightDirection);
  const float nov = max(dot(normal, viewDirection), 0.000001);
  if (nol <= 0.0) {
    return vec3(0.0);
  }
  const vec3 halfVector = normalize(viewDirection + lightDirection);
  const float noh = max(dot(normal, halfVector), 0.0);
  const float voh = max(dot(viewDirection, halfVector), 0.0);

  const vec3 diffuse = albedo * (1.0 - metallic) / kPi;
  const vec3 f0 = mix(vec3(0.04), albedo, metallic);
  const vec3 specular = distributionGgx(noh, roughness) *
                        geometrySmithGgx(nov, nol, roughness) *
                        fresnelSchlick(voh, f0) /
                        max(4.0 * nov * nol, 0.000001);

  const float charlie = distributionCharlie(noh, roughness);
  const float sheenVisibility =
      1.0 / max(4.0 * (nov + nol - nov * nol), 0.000001);
  const vec3 sheenColor = mix(vec3(0.35), albedo, 0.55);
  const vec3 cloth = charlie * sheenVisibility * sheenColor;

  return (weights.x * diffuse + weights.y * specular + weights.z * cloth) *
         radiance * nol;
}

void main() {
  const vec4 positionSample = texture(uGPosition, vUv);
  if (positionSample.a < 0.5) {
    fragColor = vec4(0.018, 0.022, 0.030, 1.0);
    return;
  }

  const vec3 worldPosition = positionSample.xyz;
  const vec3 normal = normalize(texture(uGNormal, vUv).xyz);
  const vec3 albedo = texture(uGAlbedo, vUv).rgb;
  const vec3 weights = texture(uGLayerWeights, vUv).xyz;
  const vec2 roughnessMetallic = texture(uGRoughnessMetallic, vUv).xy;
  const vec3 viewDirection = normalize(uCameraPosition - worldPosition);

  vec3 color = albedo * 0.012;
  for (int index = 0; index < kMaxLights; ++index) {
    if (index >= uLightCount) {
      break;
    }
    vec3 lightDirection = vec3(0.0);
    vec3 radiance = uLightColor[index] * uLightIntensity[index];
    if (uLightType[index] == 1) {
      lightDirection = -normalize(uLightDirection[index]);
    } else {
      const vec3 offset = uLightPosition[index] - worldPosition;
      const float distance = length(offset);
      if (distance < 0.000001 || distance > uLightRange[index]) {
        continue;
      }
      lightDirection = offset / distance;
      const vec3 attenuation = uLightAttenuation[index];
      const float denominator =
          attenuation.x + attenuation.y * distance + attenuation.z * distance * distance;
      const float rangeRatio = distance / uLightRange[index];
      const float window = pow(clamp(1.0 - pow(rangeRatio, 4.0), 0.0, 1.0), 2.0);
      radiance *= (1.0 / max(denominator, 0.000001)) * window;

      if (uLightType[index] == 2) {
        const float cosAngle = dot(-lightDirection, normalize(uLightDirection[index]));
        const float edge = clamp(
            (cosAngle - uLightOuterCos[index]) /
                max(uLightInnerCos[index] - uLightOuterCos[index], 0.000001),
            0.0, 1.0);
        radiance *= edge * edge * (3.0 - 2.0 * edge);
      }
    }
    color += evaluateLayeredBrdf(normal, viewDirection, lightDirection, albedo,
                                 roughnessMetallic.x, roughnessMetallic.y, weights,
                                 radiance);
  }

  color = color / (color + vec3(1.0));
  color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
  fragColor = vec4(color, 1.0);
}

