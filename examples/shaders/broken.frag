#version 330 core
layout(location = 0) out vec4 frag_world_position;
void main() {
    vec3 bad_value = vec3(1.0, 2.0, );
    frag_world_position = vec4(bad_value, 1.0)
}
