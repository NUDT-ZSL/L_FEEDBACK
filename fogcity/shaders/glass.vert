#version 410 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_color;

uniform mat4 u_view_proj;

out vec4 v_color;

void main()
{
    v_color = in_color;
    gl_Position = u_view_proj * vec4(in_position, 1.0);
}

