#version 460

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(set = 0, binding = 5) uniform sampler2D opaqueColorTexture;
layout(set = 0, binding = 6, rgba16f) uniform image2D volumeImage;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 finalColor;

void main() {
    const ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec3 linearColor = textureLod(opaqueColorTexture, uv, 0.0).rgb;
    linearColor += imageLoad(volumeImage, pixel).rgb;
    linearColor *= ubo.toneParams.x * ubo.toneParams.y;
    finalColor = vec4(linearToSrgb(reinhard(linearColor)), 1.0);
}
