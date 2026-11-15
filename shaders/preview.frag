#version 330 core

in vec2 vUV;

uniform sampler2D uPosition;
uniform sampler2D uNormal;
uniform sampler2D uMaterial;
uniform vec2 uResolution;
uniform vec3 uClearColor;

out vec4 outColor;

void main() {
    vec4 position = texture(uPosition, vUV);
    if (position.w < 0.5) {
        float height = gl_FragCoord.y / max(uResolution.y, 1.0);
        outColor = vec4(mix(uClearColor * 0.55, uClearColor * 1.35, height), 1.0);
        return;
    }

    vec3 normal = normalize(texture(uNormal, vUV).xyz);
    vec4 material = texture(uMaterial, vUV);
    float hemisphere = 0.55 + 0.45 * max(normal.y, 0.0);
    vec3 ambient = vec3(0.10, 0.12, 0.16) * hemisphere;
    vec3 f0 = mix(vec3(0.04), material.rgb, material.a);
    outColor = vec4(material.rgb * ambient + f0 * 0.06, 1.0);
}

