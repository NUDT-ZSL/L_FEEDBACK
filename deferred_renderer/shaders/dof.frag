#version 330 core

#define DOF_TAPS 32

uniform sampler2D u_hdr_texture;
uniform sampler2D u_linear_depth;
uniform float u_focus_distance;
uniform float u_aperture;
uniform float u_focal_length_m;
uniform float u_sensor_width_m;
uniform float u_max_blur_radius;
uniform float u_resolution_width;

out vec4 out_dof;

const float GOLDEN_ANGLE = 2.39996322972865332;
const float PI = 3.14159265358979323846;

float coc_radius(float linear_depth) {
    if (u_max_blur_radius <= 0.0) {
        return 0.0;
    }
    if (linear_depth <= 0.0) {
        return u_max_blur_radius * 0.75;
    }
    float aperture_diameter = u_focal_length_m / u_aperture;
    float focus_delta = abs(linear_depth - u_focus_distance);
    float denominator = max(
        linear_depth * abs(u_focus_distance - u_focal_length_m),
        1e-6
    );
    float coc_diameter_m = aperture_diameter * u_focal_length_m
        * focus_delta / denominator;
    float pixel_size_m = u_sensor_width_m / u_resolution_width;
    return min(
        u_max_blur_radius,
        max(0.0, 0.5 * coc_diameter_m / pixel_size_m)
    );
}

void main() {
    ivec2 center_pixel = ivec2(gl_FragCoord.xy);
    vec3 center_color = texelFetch(u_hdr_texture, center_pixel, 0).rgb;
    float center_depth = texelFetch(u_linear_depth, center_pixel, 0).r;
    float center_coc = coc_radius(center_depth);

    if (u_max_blur_radius <= 0.0) {
        out_dof = vec4(center_color, 1.0);
        return;
    }

    bool center_foreground = center_depth > 0.0
        && center_depth < u_focus_distance;
    vec3 foreground_color = vec3(0.0);
    vec3 background_color = vec3(0.0);
    float foreground_weight = 0.0;
    float background_weight = 0.0;

    if (center_foreground) {
        foreground_color += center_color;
        foreground_weight += 1.0;
    } else {
        background_color += center_color;
        background_weight += 1.0;
    }

    for (int tap = 0; tap < DOF_TAPS; ++tap) {
        float fraction = (float(tap) + 0.5) / float(DOF_TAPS);
        float radius = sqrt(fraction) * (u_max_blur_radius + 0.75);
        float angle = float(tap) * GOLDEN_ANGLE;
        vec2 offset = vec2(cos(angle), sin(angle)) * radius;
        ivec2 sample_pixel = clamp(
            center_pixel + ivec2(round(offset)),
            ivec2(0),
            ivec2(textureSize(u_hdr_texture, 0) - 1)
        );
        if (sample_pixel == center_pixel) {
            continue;
        }
        float sample_depth = texelFetch(u_linear_depth, sample_pixel, 0).r;
        float sample_coc = coc_radius(sample_depth);
        float distance_pixels = length(vec2(sample_pixel - center_pixel));
        float reach = max(center_coc, sample_coc) + 0.75;
        if (distance_pixels > reach) {
            continue;
        }

        float spatial = 1.0 - smoothstep(
            0.0,
            u_max_blur_radius + 0.75,
            distance_pixels
        );
        float coc_weight = 0.25 + 0.75 * clamp(
            sample_coc / max(u_max_blur_radius, 1e-4),
            0.0,
            1.0
        );
        float weight = max(spatial * coc_weight, 0.0001);
        vec3 sample_color = texelFetch(u_hdr_texture, sample_pixel, 0).rgb;
        bool sample_foreground = sample_depth > 0.0
            && sample_depth < u_focus_distance;
        if (sample_foreground) {
            foreground_color += sample_color * weight;
            foreground_weight += weight;
        } else {
            background_color += sample_color * weight;
            background_weight += weight;
        }
    }

    float total_weight = foreground_weight + background_weight;
    if (total_weight <= 1e-6) {
        out_dof = vec4(center_color, 1.0);
        return;
    }

    vec3 foreground = foreground_weight > 0.0
        ? foreground_color / foreground_weight
        : vec3(0.0);
    vec3 background = background_weight > 0.0
        ? background_color / background_weight
        : vec3(0.0);
    float foreground_alpha = foreground_weight / total_weight;
    out_dof = vec4(
        foreground * foreground_alpha
        + background * (1.0 - foreground_alpha),
        1.0
    );
}

