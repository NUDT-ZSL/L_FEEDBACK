#version 330 core

in vec2 vUV;

uniform sampler2D uInput;
uniform vec2 uInvResolution;
uniform float uEdgeThreshold;

out vec4 outColor;

float luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

vec3 fetchColor(vec2 offset) {
    return texture(uInput, vUV + offset * uInvResolution).rgb;
}

void main() {
    vec3 center = fetchColor(vec2(0.0));
    vec3 northWest = fetchColor(vec2(-1.0, -1.0));
    vec3 northEast = fetchColor(vec2(1.0, -1.0));
    vec3 southWest = fetchColor(vec2(-1.0, 1.0));
    vec3 southEast = fetchColor(vec2(1.0, 1.0));
    vec3 north = fetchColor(vec2(0.0, -1.0));
    vec3 south = fetchColor(vec2(0.0, 1.0));
    vec3 east = fetchColor(vec2(1.0, 0.0));
    vec3 west = fetchColor(vec2(-1.0, 0.0));

    float lumaCenter = luma(center);
    float lumaNorthWest = luma(northWest);
    float lumaNorthEast = luma(northEast);
    float lumaSouthWest = luma(southWest);
    float lumaSouthEast = luma(southEast);
    float lumaNorth = luma(north);
    float lumaSouth = luma(south);
    float lumaEast = luma(east);
    float lumaWest = luma(west);

    float lumaMinimum = min(lumaCenter,
        min(min(lumaNorth, lumaSouth), min(lumaEast, lumaWest)));
    float lumaMaximum = max(lumaCenter,
        max(max(lumaNorth, lumaSouth), max(lumaEast, lumaWest)));
    float contrast = lumaMaximum - lumaMinimum;
    if (contrast < max(uEdgeThreshold, lumaMaximum * 0.125)) {
        outColor = vec4(center, 1.0);
        return;
    }

    vec2 direction = vec2(
        -((lumaNorthWest + lumaNorthEast) - (lumaSouthWest + lumaSouthEast)),
        (lumaNorthWest + lumaSouthWest) - (lumaNorthEast + lumaSouthEast)
    );
    float directionReduce = max(
        (lumaNorthWest + lumaNorthEast + lumaSouthWest + lumaSouthEast) *
        0.25 * 0.125, 0.0078125);
    float reciprocalMinimum = 1.0 /
        (min(abs(direction.x), abs(direction.y)) + directionReduce);
    direction = clamp(direction * reciprocalMinimum, vec2(-8.0), vec2(8.0)) *
                uInvResolution;

    vec3 resultA = 0.5 * (fetchColor(direction * 0.5) +
                          fetchColor(-direction * 0.5));
    vec3 resultB = resultA * 0.5 + 0.25 *
        (fetchColor(direction) + fetchColor(-direction));
    float lumaB = luma(resultB);
    vec3 result = (lumaB < lumaMinimum || lumaB > lumaMaximum)
                ? resultA : resultB;
    outColor = vec4(result, 1.0);
}

