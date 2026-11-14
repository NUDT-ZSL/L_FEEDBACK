#version 410 core

in vec3 v_normal;
in vec3 v_color;

uniform vec3 u_light_direction;
uniform vec3 u_light_color;
uniform float u_light_intensity;

out vec4 out_color;

void main()
{
    vec3 normal = normalize(v_normal);
    if (!gl_FrontFacing) {
        normal = -normal;
    }
    vec3 to_light = normalize(-u_light_direction);
    float diffuse = max(dot(normal, to_light), 0.0);
    float sky_fill = 0.48 + 0.16 * clamp(normal.y, 0.0, 1.0);
    vec3 lit = v_color * (sky_fill + diffuse * u_light_color * u_light_intensity * 0.72);
    out_color = vec4(lit, 1.0);
}

