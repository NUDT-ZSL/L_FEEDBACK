#version 330 core

in vec2 vUV;

uniform sampler2D uInput;
uniform float uContrast;
uniform float uSaturation;
uniform vec3 uTint;
uniform vec3 uLift;

out vec4 outColor;

void main() {
    vec3 color = texture(uInput, vUV).rgb;
    color = (color - 0.5) * uContrast + 0.5;
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = mix(vec3(luminance), color, uSaturation);
    color = color * uTint + uLift;
    outColor = vec4(color, 1.0);
}

