#version 330 core

layout(location = 0) in vec3 aPosition;

uniform mat4 uLightViewProjection;

out vec3 vWorldPosition;

void main()
{
    vWorldPosition = aPosition;
    gl_Position = uLightViewProjection * vec4(aPosition, 1.0);
}

