#version 410 core

uniform sampler2D u_base_color;
uniform sampler2D u_volume_color;

out vec4 out_color;

void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec3 base = texelFetch(u_base_color, pixel, 0).rgb;
    vec4 volume = texelFetch(u_volume_color, pixel, 0);
    out_color = vec4(base * volume.a + volume.rgb, 1.0);
}

