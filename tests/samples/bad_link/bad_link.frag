#version 450 core

in vec4 vValue;
layout(location = 0) out vec4 oColor;

void main() {
    oColor = vec4(vValue.xyz, 1.0);
}
