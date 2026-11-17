#version 330 core

uniform sampler2D u_albedo_texture;
uniform sampler2D u_roughness_texture;
uniform sampler2D u_normal_texture;
uniform float u_normal_strength;
uniform int u_material_id;

in VS_OUT {
    vec3 world_position;
    vec3 world_normal;
    vec3 world_tangent;
    float tangent_sign;
    vec2 uv;
    float linear_depth;
} fs_in;

layout(location = 0) out vec4 out_albedo_roughness;
layout(location = 1) out vec4 out_normal_side;
layout(location = 2) out float out_linear_depth;
layout(location = 3) out uint out_material_id;

void main() {
    float side = gl_FrontFacing ? 1.0 : -1.0;
    vec3 geometric_normal = normalize(fs_in.world_normal) * side;
    vec3 tangent = normalize(
        fs_in.world_tangent
        - geometric_normal * dot(geometric_normal, fs_in.world_tangent)
        + vec3(1e-7, 0.0, 0.0)
    );
    vec3 bitangent = cross(geometric_normal, tangent) * fs_in.tangent_sign;

    vec3 tangent_normal = texture(u_normal_texture, fs_in.uv).xyz * 2.0 - 1.0;
    tangent_normal.xy *= u_normal_strength;
    tangent_normal = normalize(tangent_normal);
    vec3 world_normal = normalize(
        tangent * tangent_normal.x
        + bitangent * tangent_normal.y
        + geometric_normal * tangent_normal.z
    );

    vec4 albedo = texture(u_albedo_texture, fs_in.uv);
    float roughness = texture(u_roughness_texture, fs_in.uv).r;

    out_albedo_roughness = vec4(albedo.rgb, roughness);
    out_normal_side = vec4(normalize(world_normal), side);
    out_linear_depth = fs_in.linear_depth;
    out_material_id = uint(u_material_id);
}

