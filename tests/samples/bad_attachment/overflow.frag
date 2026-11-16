#version 450 core

layout(location = 0) out vec4 oColor;
layout(location = 4) out vec4 oOutOfBounds;

void main() {
    oColor = vec4(1.0);
    oOutOfBounds = vec4(1.0, 0.0, 0.0, 1.0);
}
