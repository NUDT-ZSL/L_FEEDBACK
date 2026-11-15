#version 330 core

in vec2 vUV;

uniform sampler2D uInput;

out vec4 outColor;

vec3 linearToSrgb(vec3 color) {
    color = clamp(color, 0.0, 1.0);
    vec3 low = color * 12.92;
    vec3 high = 1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055;
    return mix(low, high, step(vec3(0.0031308), color));
}

void main() {
    outColor = vec4(linearToSrgb(texture(uInput, vUV).rgb), 1.0);
}

