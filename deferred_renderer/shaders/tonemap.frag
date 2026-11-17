#version 330 core

uniform sampler2D u_hdr_texture;
uniform float u_exposure;
uniform int u_curve;

out vec4 out_color;

vec3 reinhard(vec3 color) {
    return color / (vec3(1.0) + color);
}

vec3 aces_narkowicz(vec3 color) {
    return clamp(
        (color * (2.51 * color + 0.03))
        / (color * (2.43 * color + 0.59) + 0.14),
        0.0,
        1.0
    );
}

vec3 hable_filmic(vec3 color) {
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    return clamp(
        ((color * (A * color + C * B) + D * E)
        / (color * (A * color + B) + D * F))
        - E / F,
        0.0,
        1.0
    );
}

vec3 linear_to_srgb(vec3 color) {
    vec3 low = color * 12.92;
    vec3 high = 1.055 * pow(max(color, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(low, high, step(vec3(0.0031308), color));
}

void main() {
    vec3 color = texelFetch(u_hdr_texture, ivec2(gl_FragCoord.xy), 0).rgb
        * u_exposure;
    if (u_curve == 0) {
        color = clamp(color, 0.0, 1.0);
    } else if (u_curve == 1) {
        color = reinhard(color);
    } else if (u_curve == 2) {
        color = aces_narkowicz(color);
    } else {
        color = hable_filmic(color);
    }
    out_color = vec4(linear_to_srgb(color), 1.0);
}

