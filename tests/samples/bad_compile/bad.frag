#version 450 core

layout(location = 0) out vec4 oColor;

void main() {
    oColor = vec4(1.0);
    this_is_intentionally_broken;
}
