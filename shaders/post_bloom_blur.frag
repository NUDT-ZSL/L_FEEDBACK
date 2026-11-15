#version 330 core

in vec2 vUV;

uniform sampler2D uInput;
uniform vec2 uDirection;

out vec4 outColor;

void main() {
    const float weights[5] = float[](
        0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162
    );
    vec3 result = texture(uInput, vUV).rgb * weights[0];
    for (int i = 1; i < 5; ++i) {
        vec2 offset = uDirection * float(i);
        result += texture(uInput, vUV + offset).rgb * weights[i];
        result += texture(uInput, vUV - offset).rgb * weights[i];
    }
    outColor = vec4(result, 1.0);
}

