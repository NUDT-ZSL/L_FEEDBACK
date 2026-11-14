#version 430 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aTangent;
layout(location = 3) in vec2 aUv;
layout(location = 4) in ivec4 aJoints;
layout(location = 5) in vec4 aWeights;

const int kBoneCount = 18;

uniform mat4 uJoints[kBoneCount];
uniform mat4 uModel;
uniform mat3 uNormalMatrix;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vWorldPosition;
out vec3 vWorldNormal;
out vec3 vWorldTangent;
out vec2 vUv;

void main() {
  vec4 skinnedPosition = vec4(0.0);
  vec3 skinnedNormal = vec3(0.0);
  vec3 skinnedTangent = vec3(0.0);
  for (int influence = 0; influence < 4; ++influence) {
    const float weight = aWeights[influence];
    if (weight <= 0.0) {
      continue;
    }
    const int jointIndex = clamp(aJoints[influence], 0, kBoneCount - 1);
    const mat4 skin = uJoints[jointIndex];
    skinnedPosition += (skin * vec4(aPosition, 1.0)) * weight;
    const mat3 skinNormal = transpose(inverse(mat3(skin)));
    skinnedNormal += (skinNormal * aNormal) * weight;
    skinnedTangent += (mat3(skin) * aTangent) * weight;
  }

  const vec4 worldPosition = uModel * skinnedPosition;
  vWorldPosition = worldPosition.xyz;
  vWorldNormal = normalize(uNormalMatrix * skinnedNormal);
  vWorldTangent = normalize(mat3(uModel) * skinnedTangent);
  vUv = aUv;
  gl_Position = uProjection * uView * worldPosition;
}

