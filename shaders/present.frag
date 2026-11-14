#version 450 core

layout(binding = 0) uniform sampler2D uSource;
in vec2 vUv;
out vec4 fragColor;

void main() {
    vec3 hdr = texture(uSource, vUv).rgb;
    vec3 mapped = hdr / (vec3(1.0) + hdr);
    fragColor = vec4(pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}

