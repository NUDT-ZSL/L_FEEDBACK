#version 450 core

layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oNormal;

void main() {
    oColor = vec4(0.2, 0.4, 0.6, 1.0);
    oNormal = vec4(0.5, 0.5, 1.0, 1.0);
}
