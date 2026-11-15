#version 330 core

in vec2 vUV;

uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uIntensity;

out vec4 outColor;

void main() {
    vec3 scene = texture(uScene, vUV).rgb;
    vec3 bloom = texture(uBloom, vUV).rgb;
    outColor = vec4(scene + bloom * uIntensity, 1.0);
}

