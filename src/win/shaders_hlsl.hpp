#pragma once

namespace uid166::shaders {

inline constexpr const char* scene = R"HLSL(
static const uint MAX_BONES = 64;

struct InstanceData
{
    row_major float4x4 world;
    uint material;
    uint objectId;
    float2 padding;
};

cbuffer FrameCB : register(b0)
{
    row_major float4x4 gViewProj;
    row_major float4x4 gCascadeMatrices[3];
    row_major float4x4 gSpotMatrices[4];
    float4 gCascadeSplits;
    float4 gCameraPos;
    float4 gCameraForward;
    float4 gDirLightDirection;
    float4 gDirLightColor;
    float4 gLocalPosRange[4];
    float4 gLocalColorIntensity[4];
    float4 gLocalDirType[4];
    float4 gLocalAndSlices[4];
    float4 gMaterialColors[8];
};

cbuffer ObjectCB : register(b1)
{
    row_major float4x4 gWorld;
    float4 gObjectParams;
};

cbuffer BoneCB : register(b2)
{
    row_major float4x4 gBones[MAX_BONES];
};

StructuredBuffer<InstanceData> gInstances : register(t0);
Texture2DArray gCascadeShadow : register(t1);
TextureCubeArray gPointShadow : register(t2);
Texture2DArray gSpotShadow : register(t3);
SamplerComparisonState gShadowSampler : register(s0);

struct StaticVSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

struct SkinnedVSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    uint4 boneIndices : BLENDINDICES;
    float4 boneWeights : BLENDWEIGHT;
};

struct VSOutput
{
    float4 svPosition : SV_POSITION;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    nointerpolation uint material : TEXCOORD3;
};

VSOutput SceneVS(StaticVSInput input, uint instanceId : SV_InstanceID)
{
    InstanceData instance = gInstances[instanceId];
    float4 worldPos = mul(float4(input.position, 1.0f), instance.world);
    float3 normal = normalize(mul(float4(input.normal, 0.0f), instance.world).xyz);
    VSOutput output;
    output.svPosition = mul(worldPos, gViewProj);
    output.worldPos = worldPos.xyz;
    output.normal = normal;
    output.uv = input.uv;
    output.material = instance.material;
    return output;
}

VSOutput SkinnedVS(SkinnedVSInput input)
{
    float4 skinnedPosition = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 skinnedNormal = float3(0.0f, 0.0f, 0.0f);
    [unroll]
    for (uint i = 0; i < 4; ++i)
    {
        uint bone = min(input.boneIndices[i], MAX_BONES - 1);
        float weight = input.boneWeights[i];
        skinnedPosition += mul(float4(input.position, 1.0f), gBones[bone]) * weight;
        skinnedNormal += mul(float4(input.normal, 0.0f), gBones[bone]).xyz * weight;
    }
    float4 worldPos = mul(skinnedPosition, gWorld);
    VSOutput output;
    output.svPosition = mul(worldPos, gViewProj);
    output.worldPos = worldPos.xyz;
    output.normal = normalize(mul(float4(skinnedNormal, 0.0f), gWorld).xyz);
    output.uv = input.uv;
    output.material = (uint)gObjectParams.x;
    return output;
}

float SampleCascade(float3 worldPos, float3 normal, uint cascade)
{
    float4 lightPos = mul(float4(worldPos + normal * 0.025f, 1.0f), gCascadeMatrices[cascade]);
    float2 uv = lightPos.xy * float2(0.5f, -0.5f) + 0.5f;
    if (any(uv < 0.0f) || any(uv > 1.0f) || lightPos.z <= 0.0f || lightPos.z >= 1.0f)
    {
        return 1.0f;
    }
    return gCascadeShadow.SampleCmpLevelZero(
        gShadowSampler, float3(uv, (float)cascade), lightPos.z - 0.0012f).r;
}

float SamplePointShadow(float3 worldPos, float3 normal, uint lightIndex)
{
    float3 lightPos = gLocalPosRange[lightIndex].xyz;
    float range = max(gLocalPosRange[lightIndex].w, 0.01f);
    float3 offsetPos = worldPos + normal * 0.025f;
    float3 delta = offsetPos - lightPos;
    float distanceToLight = length(delta);
    float slice = gLocalAndSlices[lightIndex].z;
    float compareDepth = distanceToLight / range - 0.0015f;
    return gPointShadow.SampleCmpLevelZero(
        gShadowSampler, float4(delta / max(distanceToLight, 0.0001f), slice), compareDepth).r;
}

float SampleSpotShadow(float3 worldPos, float3 normal, uint lightIndex)
{
    float4 lightPos = mul(float4(worldPos + normal * 0.025f, 1.0f), gSpotMatrices[lightIndex]);
    float2 uv = lightPos.xy * float2(0.5f, -0.5f) + 0.5f;
    if (any(uv < 0.0f) || any(uv > 1.0f) || lightPos.z <= 0.0f || lightPos.z >= 1.0f)
    {
        return 1.0f;
    }
    float slice = gLocalAndSlices[lightIndex].w;
    return gSpotShadow.SampleCmpLevelZero(
        gShadowSampler, float3(uv, slice), lightPos.z - 0.0015f).r;
}

float4 ScenePS(VSOutput input) : SV_TARGET
{
    float3 normal = normalize(input.normal);
    float3 viewVector = normalize(gCameraPos.xyz - input.worldPos);
    uint material = min(input.material, 7u);
    float3 albedo = gMaterialColors[material].rgb;
    float3 lighting = albedo * float3(0.075f, 0.085f, 0.11f);

    float viewDepth = dot(input.worldPos - gCameraPos.xyz, gCameraForward.xyz);
    uint cascade = 0;
    if (viewDepth > gCascadeSplits.x) cascade = 1;
    if (viewDepth > gCascadeSplits.y) cascade = 2;
    float directionalShadow = SampleCascade(input.worldPos, normal, cascade);
    float3 toDirectional = -normalize(gDirLightDirection.xyz);
    float directionalNdotL = saturate(dot(normal, toDirectional));
    float3 directionalColor = gDirLightColor.rgb * gDirLightColor.w;
    float3 halfDir = normalize(toDirectional + viewVector);
    float directionalSpec = pow(saturate(dot(normal, halfDir)), 48.0f) * 0.18f;
    lighting += (albedo * directionalNdotL + directionalSpec) * directionalColor *
                directionalShadow;

    [unroll]
    for (uint lightIndex = 0; lightIndex < 4; ++lightIndex)
    {
        uint lightType = (uint)gLocalDirType[lightIndex].w;
        float3 lightPosition = gLocalPosRange[lightIndex].xyz;
        float range = max(gLocalPosRange[lightIndex].w, 0.01f);
        float3 toLightVector = lightPosition - input.worldPos;
        float distanceToLight = length(toLightVector);
        if (distanceToLight >= range || lightType == 0u)
        {
            continue;
        }
        float3 toLight = toLightVector / max(distanceToLight, 0.0001f);
        float ndotl = saturate(dot(normal, toLight));
        float attenuation = saturate(1.0f - distanceToLight / range);
        attenuation *= attenuation;
        float spotFactor = 1.0f;
        if (lightType == 2u)
        {
            float spotCosine = dot(-toLight, normalize(gLocalDirType[lightIndex].xyz));
            spotFactor = smoothstep(gLocalAndSlices[lightIndex].y,
                                    gLocalAndSlices[lightIndex].x, spotCosine);
        }
        if (ndotl <= 0.0f || attenuation <= 0.0f || spotFactor <= 0.0f)
        {
            continue;
        }
        float shadow = lightType == 1u
            ? SamplePointShadow(input.worldPos, normal, lightIndex)
            : SampleSpotShadow(input.worldPos, normal, lightIndex);
        float3 localHalf = normalize(toLight + viewVector);
        float localSpec = pow(saturate(dot(normal, localHalf)), 32.0f) * 0.15f;
        float3 lightColor = gLocalColorIntensity[lightIndex].rgb *
                            gLocalColorIntensity[lightIndex].w;
        lighting += (albedo * ndotl + localSpec) * lightColor * attenuation *
                    spotFactor * shadow;
    }

    float3 mapped = lighting / (lighting + 1.0f);
    mapped = pow(max(mapped, 0.0f), 1.0f / 2.2f);
    return float4(mapped, 1.0f);
}
)HLSL";

inline constexpr const char* shadow = R"HLSL(
struct InstanceData
{
    row_major float4x4 world;
    uint material;
    uint objectId;
    float2 padding;
};

cbuffer ShadowCB : register(b3)
{
    row_major float4x4 gShadowViewProj;
};

cbuffer ObjectCB : register(b1)
{
    row_major float4x4 gWorld;
    float4 gObjectParams;
};

cbuffer BoneCB : register(b2)
{
    row_major float4x4 gBones[64];
};

StructuredBuffer<InstanceData> gInstances : register(t0);

struct StaticVSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

struct SkinnedVSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    uint4 boneIndices : BLENDINDICES;
    float4 boneWeights : BLENDWEIGHT;
};

float4 StaticShadowVS(StaticVSInput input, uint instanceId : SV_InstanceID) : SV_POSITION
{
    InstanceData instance = gInstances[instanceId];
    float4 worldPos = mul(float4(input.position, 1.0f), instance.world);
    return mul(worldPos, gShadowViewProj);
}

float4 SkinnedShadowVS(SkinnedVSInput input) : SV_POSITION
{
    float4 skinnedPosition = float4(0.0f, 0.0f, 0.0f, 0.0f);
    [unroll]
    for (uint i = 0; i < 4; ++i)
    {
        uint bone = min(input.boneIndices[i], 63u);
        skinnedPosition += mul(float4(input.position, 1.0f), gBones[bone]) *
                           input.boneWeights[i];
    }
    return mul(mul(skinnedPosition, gWorld), gShadowViewProj);
}
)HLSL";

inline constexpr const char* fullscreen = R"HLSL(
Texture2D gSceneColor : register(t0);
SamplerState gLinearSampler : register(s0);

struct VSOutput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

VSOutput FullscreenVS(uint vertexId : SV_VertexID)
{
    float2 positions[3] =
    {
        float2(-1.0f, -1.0f),
        float2(3.0f, -1.0f),
        float2(-1.0f, 3.0f)
    };
    VSOutput output;
    float2 p = positions[vertexId];
    output.position = float4(p, 0.0f, 1.0f);
    output.uv = float2((p.x + 1.0f) * 0.5f, 1.0f - (p.y + 1.0f) * 0.5f);
    return output;
}

float4 FullscreenPS(VSOutput input) : SV_TARGET
{
    return gSceneColor.Sample(gLinearSampler, input.uv);
}
)HLSL";

} // namespace uid166::shaders

