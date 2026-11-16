#version 450 core

layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oAlphaCopy;
layout(location = 2) out vec4 oNormal;
layout(location = 3) out vec4 oDependency;

layout(std140, binding = 0) uniform DepConstants {
    vec4 tint;
    vec4 depth_normal;
    vec4 packed_data;
    vec4 seed;
} constants;

layout(binding = 0) uniform sampler2D uAlbedo;
layout(binding = 1) uniform sampler2D uDetail;

void main() {
#ifdef BROKEN_COMPILE
    this_variant_is_intentionally_not_valid_glsl = 1.0;
#endif

    ivec2 pixel = ivec2(floor(gl_FragCoord.xy));
    uint hashValue = uint(pixel.x) * 0x9e3779b1u ^ uint(pixel.y) * 0x85ebca77u;
    hashValue ^= hashValue >> 16;
    hashValue *= 0x7feb352du;
    hashValue ^= hashValue >> 15;
    float alpha = float(hashValue & 255u) / 255.0;

    vec4 albedo = texelFetch(uAlbedo, pixel % textureSize(uAlbedo, 0), 0);
    vec4 detail = texelFetch(uDetail, pixel % textureSize(uDetail, 0), 0);

#ifdef TINTED_PIPELINE
    vec3 tintedColor = constants.tint.rgb;
#else
    vec3 tintedColor = constants.tint.rgb;
#endif

    oColor = vec4(tintedColor, alpha);
    oAlphaCopy = vec4(alpha, alpha, alpha, alpha);
    oNormal = vec4(0.5, 0.5, 1.0, 1.0);
    oDependency = vec4(albedo.r, detail.g, constants.tint.b, constants.tint.a);
}
