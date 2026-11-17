#version 330 core

uniform sampler2D u_frame_texture;
uniform vec2 u_resolution;

out vec4 out_color;

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    ivec2 texture_size = textureSize(u_frame_texture, 0);
    ivec2 texel = clamp(
        ivec2(uv * vec2(texture_size)),
        ivec2(0),
        texture_size - 1
    );
    out_color = texelFetch(u_frame_texture, texel, 0);
}

