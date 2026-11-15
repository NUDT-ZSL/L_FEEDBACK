#version 330 core

in vec2 vUV;

uniform sampler2D uInput;
uniform float uThreshold;
uniform float uKnee;

out vec4 outColor;

void main() {
    vec3 color = texture(uInput, vUV).rgb;
    float brightness = max(color.r, max(color.g, color.b));
    float knee = max(uKnee, 1.0e-4);
    float soft = clamp(brightness - uThreshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    float contribution = max(soft, brightness - uThreshold) /
                         max(brightness, 1.0e-5);
    outColor = vec4(color * contribution, 1.0);
}

