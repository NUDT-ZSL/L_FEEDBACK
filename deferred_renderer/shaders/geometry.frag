#version 330 core

// MRT output slots. The slot order here is the G-Buffer attachment contract:
//   location 0 -> world position   (RGBA32F)
//   location 1 -> world normal     (RGB16F)
//   location 2 -> albedo + mat id  (RGBA8)
// The lighting pass must sample the same slots in exactly this order.
layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec3 gNormal;
layout(location = 2) out vec4 gAlbedoMaterial;

in vec3 vWorldPos;
in vec3 vNormal;

const int MAX_MATERIALS = 32;

uniform vec3 uAlbedo[MAX_MATERIALS];
uniform int uMaterialId;
uniform vec3 uCamPos;

void main() {
    // xyz: world-space position; w: linear view distance (handy teaching data).
    float linearDepth = length(vWorldPos - uCamPos);
    gPosition = vec4(vWorldPos, linearDepth);

    gNormal = vec3(normalize(vNormal));

    // RGB stores albedo, A stores the material index as an 8-bit unorm.
    // The lighting pass decodes it and looks up roughness / metallic / emissive.
    float idEncoded = float(uMaterialId) / 255.0;
    gAlbedoMaterial = vec4(uAlbedo[uMaterialId], idEncoded);
}
