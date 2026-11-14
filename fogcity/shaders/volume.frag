#version 410 core

in vec2 v_uv;

uniform sampler2D u_camera_depth;
uniform sampler2D u_shadow_depth;
uniform usampler2D u_shadow_id;
uniform mat4 u_inv_view_proj;
uniform mat4 u_light_view_proj;
uniform vec3 u_camera_position;
uniform vec3 u_light_direction;
uniform vec3 u_light_color;
uniform vec3 u_fog_color;
uniform float u_light_intensity;
uniform float u_fog_density;
uniform float u_fog_extinction;
uniform float u_fog_scattering;
uniform float u_fog_anisotropy;
uniform float u_fog_ambient;
uniform float u_near;
uniform float u_far;
uniform float u_shadow_bias;
uniform int u_steps;
uniform int u_shadow_resolution;
uniform vec2 u_resolution;

out vec4 out_volume;

float henyey_greenstein(float mu, float g)
{
    float denominator = max(1.0 + g * g - 2.0 * g * mu, 1e-5);
    return (1.0 - g * g) / (12.566370614 * pow(denominator, 1.5));
}

float shadow_visibility(vec3 world_position)
{
    vec4 light_clip = u_light_view_proj * vec4(world_position, 1.0);
    if (light_clip.w <= 0.0) {
        return 1.0;
    }
    vec3 light_ndc = light_clip.xyz / light_clip.w;
    if (any(lessThan(light_ndc, vec3(-1.0))) || any(greaterThan(light_ndc, vec3(1.0)))) {
        return 1.0;
    }
    vec2 light_uv = light_ndc.xy * 0.5 + 0.5;
    ivec2 coord = clamp(
        ivec2(floor(light_uv * float(u_shadow_resolution))),
        ivec2(0),
        ivec2(u_shadow_resolution - 1)
    );
    uint occluder_id = texelFetch(u_shadow_id, coord, 0).r;
    if (occluder_id == 0u) {
        return 1.0;
    }
    float stored_depth = texelFetch(u_shadow_depth, coord, 0).r;
    float sample_depth = light_ndc.z * 0.5 + 0.5;
    return sample_depth <= stored_depth + u_shadow_bias ? 1.0 : 0.0;
}

void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 ndc = uv * 2.0 - 1.0;

    vec4 near_h = u_inv_view_proj * vec4(ndc, -1.0, 1.0);
    vec4 far_h = u_inv_view_proj * vec4(ndc, 1.0, 1.0);
    vec3 near_world = near_h.xyz / near_h.w;
    vec3 far_world = far_h.xyz / far_h.w;
    vec3 ray_direction = normalize(far_world - near_world);

    float depth = texelFetch(u_camera_depth, pixel, 0).r;
    float ray_end;
    if (depth >= 0.999999) {
        ray_end = length(far_world - u_camera_position);
    } else {
        vec4 surface_h = u_inv_view_proj * vec4(ndc, depth * 2.0 - 1.0, 1.0);
        vec3 surface_world = surface_h.xyz / surface_h.w;
        ray_end = length(surface_world - u_camera_position);
    }
    ray_end = clamp(ray_end, u_near, u_far);

    float sigma_t = max(u_fog_density * u_fog_extinction, 0.0);
    float sigma_s = sigma_t * max(u_fog_scattering, 0.0);
    float transmittance_end = exp(-sigma_t * ray_end);
    float phase = henyey_greenstein(
        clamp(-dot(ray_direction, u_light_direction), -1.0, 1.0),
        u_fog_anisotropy
    );
    float step_length = (u_far - u_near) / float(u_steps);
    vec3 inscatter = vec3(0.0);

    for (int index = 0; index < u_steps; ++index) {
        float t0 = u_near + float(index) * step_length;
        if (t0 >= ray_end) {
            break;
        }
        float t1 = min(t0 + step_length, ray_end);
        float distance_step = t1 - t0;
        float sample_distance = 0.5 * (t0 + t1);
        vec3 sample_position = u_camera_position + ray_direction * sample_distance;
        float visibility = shadow_visibility(sample_position);
        float sample_transmittance = exp(-sigma_t * sample_distance);
        vec3 sun_term = u_light_color * (u_light_intensity * phase * visibility);
        inscatter += sample_transmittance * sigma_s * sun_term * distance_step;
    }

    vec3 ambient_term = u_fog_color * (u_fog_ambient * (1.0 - transmittance_end));
    out_volume = vec4(inscatter + ambient_term, transmittance_end);
}

