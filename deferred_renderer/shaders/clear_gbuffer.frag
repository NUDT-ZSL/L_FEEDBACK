#version 330 core

layout(location = 0) out vec4 out_albedo_roughness;
layout(location = 1) out vec4 out_normal_side;
layout(location = 2) out float out_linear_depth;
layout(location = 3) out uint out_material_id;

void main() {
    out_albedo_roughness = vec4(0.0);
    out_normal_side = vec4(0.0, 0.0, 0.0, 1.0);
    out_linear_depth = 0.0;
    out_material_id = 0u;
}

