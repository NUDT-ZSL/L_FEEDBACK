#version 410 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;
layout(location = 3) in float in_building_id;

uniform mat4 u_view_proj;

out vec3 v_normal;
out vec3 v_color;

void main()
{
    v_normal = in_normal;
    v_color = in_color;
    gl_Position = u_view_proj * vec4(in_position, 1.0);
}

