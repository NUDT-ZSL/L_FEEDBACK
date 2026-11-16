#version 330 core

uniform sampler2D uPositionTexture;
uniform sampler2D uNormalMaterialTexture;
uniform isampler2D uTileEntries;
uniform vec2 uScreenSize;
uniform int uTileSize;
uniform int uTilesX;
uniform int uEntryCount;

uniform vec3 uEntryVplPosition[MAX_DELTA];
uniform vec3 uEntryVplColorPower[MAX_DELTA];
uniform vec3 uEntryVplDirection[MAX_DELTA];
uniform vec4 uEntryVplParams[MAX_DELTA];
uniform int uEntryShadowMapIndex[MAX_DELTA];
uniform samplerCube uShadowMaps[MAX_SHADOWS];

uniform mat4 uView;
uniform vec3 uCameraPosition;
uniform int uMaterialCount;
uniform int uMaterialType[MAX_MATERIALS];
uniform vec3 uMaterialAlbedo[MAX_MATERIALS];
uniform float uMaterialRoughness[MAX_MATERIALS];
uniform float uMaterialSpecular[MAX_MATERIALS];
uniform float uMaterialShininess[MAX_MATERIALS];

layout(location = 0) out vec4 outContribution;

float oren_nayar_factor(vec3 normal, vec3 viewDir, vec3 lightDir, float roughness)
{
    float nDotL = max(0.0, dot(normal, lightDir));
    float nDotV = max(0.0, dot(normal, viewDir));
    if (nDotL <= 0.0 || nDotV <= 0.0) {
        return 0.0;
    }
    float sigma = max(1.0e-4, roughness) * 1.57079632679;
    float sigma2 = sigma * sigma;
    float a = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
    float b = 0.45 * sigma2 / (sigma2 + 0.09);
    vec3 lightTangent = normalize(lightDir - normal * nDotL + vec3(1.0e-5, 0.0, 0.0));
    vec3 viewTangent = normalize(viewDir - normal * nDotV + vec3(1.0e-5, 0.0, 0.0));
    float cosineDelta = max(0.0, dot(lightTangent, viewTangent));
    float sinAlpha = sqrt(max(0.0, 1.0 - nDotL * nDotL));
    float sinBeta = sqrt(max(0.0, 1.0 - nDotV * nDotV));
    return nDotL * (a + b * cosineDelta * max(sinAlpha, sinBeta));
}

vec3 material_response(int materialIndex, vec3 normal, vec3 viewDir, vec3 lightDir)
{
    int safeIndex = (materialIndex >= 0 && materialIndex < uMaterialCount) ? materialIndex : 0;
    vec3 albedo = uMaterialAlbedo[safeIndex];
    float nDotL = max(0.0, dot(normal, lightDir));
    if (nDotL <= 0.0) {
        return vec3(0.0);
    }
    int materialType = uMaterialType[safeIndex];
    if (materialType == 1) {
        float factor = oren_nayar_factor(
            normal,
            viewDir,
            lightDir,
            uMaterialRoughness[safeIndex]
        );
        return albedo * factor / 3.14159265359;
    }
    vec3 diffuse = albedo * nDotL / 3.14159265359;
    if (materialType == 2) {
        vec3 halfVector = normalize(viewDir + lightDir);
        float nDotH = max(0.0, dot(normal, halfVector));
        float nDotV = max(1.0e-4, dot(normal, viewDir));
        float shininess = max(1.0, uMaterialShininess[safeIndex]);
        float specularNorm = (shininess + 8.0) / (8.0 * 3.14159265359);
        vec3 specular = vec3(uMaterialSpecular[safeIndex])
            * specularNorm
            * pow(nDotH, shininess)
            * nDotL / nDotV;
        return diffuse + specular;
    }
    // Unknown material types are normalized on the CPU. This branch is the
    // final Lambertian safety net if an out-of-range index reaches the GPU.
    return diffuse;
}

float sample_shadow_depth(int shadowIndex, vec3 direction)
{
    // GLSL 3.30 forbids dynamic sampler-array indexing, so the eight-entry
    // batch is unrolled explicitly.
    if (shadowIndex == 0) return texture(uShadowMaps[0], direction).r;
    if (shadowIndex == 1) return texture(uShadowMaps[1], direction).r;
    if (shadowIndex == 2) return texture(uShadowMaps[2], direction).r;
    if (shadowIndex == 3) return texture(uShadowMaps[3], direction).r;
    if (shadowIndex == 4) return texture(uShadowMaps[4], direction).r;
    if (shadowIndex == 5) return texture(uShadowMaps[5], direction).r;
    if (shadowIndex == 6) return texture(uShadowMaps[6], direction).r;
    return texture(uShadowMaps[7], direction).r;
}

float shadow_visibility(int shadowIndex, vec3 lightToSurface, float normalizedDistance)
{
    vec3 direction = normalize(lightToSurface);
    float depth = sample_shadow_depth(shadowIndex, direction);
    if (normalizedDistance <= depth + 0.012) {
        return 1.0;
    }
    // Four extra taps soften the cube-map edge without a separate blur pass.
    vec3 tangent = normalize(cross(direction, abs(direction.y) < 0.9
        ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    vec3 bitangent = cross(direction, tangent);
    float visibility = 0.0;
    visibility += normalizedDistance <= sample_shadow_depth(
        shadowIndex, normalize(direction + tangent * 0.010)
    ) + 0.012 ? 1.0 : 0.0;
    visibility += normalizedDistance <= sample_shadow_depth(
        shadowIndex, normalize(direction - tangent * 0.010)
    ) + 0.012 ? 1.0 : 0.0;
    visibility += normalizedDistance <= sample_shadow_depth(
        shadowIndex, normalize(direction + bitangent * 0.010)
    ) + 0.012 ? 1.0 : 0.0;
    visibility += normalizedDistance <= sample_shadow_depth(
        shadowIndex, normalize(direction - bitangent * 0.010)
    ) + 0.012 ? 1.0 : 0.0;
    return visibility * 0.25;
}

void main()
{
    vec4 positionSample = texelFetch(
        uPositionTexture,
        ivec2(gl_FragCoord.xy),
        0
    );
    if (positionSample.a <= 0.0) {
        outContribution = vec4(0.0);
        return;
    }
    vec4 normalMaterial = texelFetch(
        uNormalMaterialTexture,
        ivec2(gl_FragCoord.xy),
        0
    );
    vec3 worldPosition = positionSample.xyz;
    vec3 normal = normalize(normalMaterial.xyz);
    int materialIndex = int(normalMaterial.w + 0.5);
    vec3 viewDir = normalize(uCameraPosition - worldPosition);

    ivec2 tile = ivec2(gl_FragCoord.xy) / uTileSize;
    int tileIndex = tile.y * uTilesX + tile.x;
    vec3 accumulated = vec3(0.0);
    for (int slot = 0; slot < MAX_DELTA; ++slot) {
        int entry = texelFetch(uTileEntries, ivec2(slot, tileIndex), 0).r;
        if (entry < 0 || entry >= uEntryCount) {
            break;
        }
        vec3 delta = uEntryVplPosition[entry] - worldPosition;
        float distanceSquared = dot(delta, delta);
        float lightRange = uEntryVplParams[entry].x;
        float distance = sqrt(distanceSquared);
        if (distance >= lightRange || distance <= 1.0e-5) {
            continue;
        }
        vec3 lightDir = delta / distance;
        float nDotL = dot(normal, lightDir);
        if (nDotL <= 0.0) {
            continue;
        }

        float coneOuter = uEntryVplParams[entry].y;
        float spot = 1.0;
        if (coneOuter > -0.5) {
            float cone = dot(-lightDir, normalize(uEntryVplDirection[entry]));
            if (cone <= coneOuter) {
                continue;
            }
            spot = clamp((cone - coneOuter) / max(1.0e-4, 0.10), 0.0, 1.0);
        }

        float normalizedDistance = distance / lightRange;
        float visibility = 1.0;
        if (uEntryVplParams[entry].w > 0.5) {
            visibility = shadow_visibility(
                uEntryShadowMapIndex[entry],
                -delta,
                normalizedDistance
            );
        }
        if (visibility <= 0.0) {
            continue;
        }

        float normalized = distance / lightRange;
        float rangeWindow = pow(max(0.0, 1.0 - pow(normalized, 4.0)), 2.0);
        float attenuation = rangeWindow / (distanceSquared + 0.03);
        vec3 response = material_response(materialIndex, normal, viewDir, lightDir);
        vec3 radiance = uEntryVplColorPower[entry] * attenuation * spot * visibility;
        accumulated += response * radiance * uEntryVplParams[entry].z;
    }
    outContribution = vec4(accumulated, 0.0);
}

