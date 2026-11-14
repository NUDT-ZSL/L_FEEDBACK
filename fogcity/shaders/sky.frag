#version 410 core

in vec2 v_uv;

uniform mat4 u_inv_view_proj;
uniform vec3 u_light_direction;
uniform vec3 u_light_color;

out vec4 out_color;

void main()
{
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec4 near_h = u_inv_view_proj * vec4(ndc, -1.0, 1.0);
    vec4 far_h = u_inv_view_proj * vec4(ndc, 1.0, 1.0);
    vec3 near_world = near_h.xyz / near_h.w;
    vec3 far_world = far_h.xyz / far_h.w;
    vec3 ray_direction = normalize(far_world - near_world);

    float height = clamp(ray_direction.y, 0.0, 1.0);
    vec3 horizon_color = vec3(0.72, 0.77, 0.78);
    vec3 zenith_color = vec3(0.18, 0.34, 0.54);
    vec3 color = mix(horizon_color, zenith_color, pow(height, 0.58));

    vec3 sun_direction = normalize(-u_light_direction);
    float sun_alignment = max(dot(ray_direction, sun_direction), 0.0);
    float sun_disc = pow(sun_alignment, 900.0);
    float sun_halo = pow(sun_alignment, 18.0);
    color += u_light_color * (sun_disc * 3.2 + sun_halo * 0.20);

    out_color = vec4(color, 1.0);
}

