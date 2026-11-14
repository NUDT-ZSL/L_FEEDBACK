#version 450 core

layout(location = 0) in vec3 aPosition;
layout(location = 4) in mat4 aModel;

uniform mat4 uLightView;
uniform mat4 uLightProjection;

void main() {
    gl_Position = uLightProjection * uLightView * aModel * vec4(aPosition, 1.0);
}
