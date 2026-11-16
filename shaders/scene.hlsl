cbuffer SceneConstants : register(b0)
{
    row_major float4x4 MainViewProj;
    row_major float4x4 MainView;
    row_major float4x4 LightViewProj[3];
    float4 CameraPositionTime;
    float4 LightDirectionIntensity;
    float4 CascadeDepths;
    float4 ShadowBiasAndMapSize;
    float4 CascadeTransitions;
    uint ActiveCascade;
    float3 ConstantPadding;
};

Texture2DArray<float> ShadowMaps : register(t0);
SamplerState ShadowPointSampler : register(s0);

struct VertexInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float3 color : COLOR;
    float4 animation : TEXCOORD0;
};

struct MainVertexOutput
{
    float4 position : SV_Position;
    float3 worldPosition : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float3 color : TEXCOORD2;
    float viewDepth : TEXCOORD3;
};

struct ShadowVertexOutput
{
    float4 position : SV_Position;
};

float3 AnimateVertex(VertexInput input, float time)
{
    // Terrain uses animation.x == 0. Character vertices share this exact path in
    // both the shadow and main passes.
    const float phase = input.animation.z + time * input.animation.y;
    const float spatial = input.position.x * 0.31 + input.position.z * 0.17;
    const float waveA = sin(phase + spatial);
    const float waveB = sin(phase * 0.73 + input.position.z * 0.47 +
                            input.animation.w * 1.7);
    const float waveC = cos(phase * 0.91 + input.position.x * 0.23 -
                            input.animation.w);
    const float3 localOffset =
        float3(waveA * 0.08 + waveB * 0.025,
               waveC * 0.045,
               waveA * 0.04 - waveB * 0.02) *
        input.animation.x;
    const float groupWave = sin(time * 1.25 + input.animation.w * 1.9);
    const float groupLift = cos(time * 0.97 + input.animation.w * 1.3);
    const float3 groupOffset =
        float3(groupWave * 0.06, groupLift * 0.035, groupWave * 0.035) *
        input.animation.x;
    return input.position + localOffset + groupOffset;
}

MainVertexOutput MainVS(VertexInput input)
{
    MainVertexOutput output;
    const float3 worldPosition = AnimateVertex(input, CameraPositionTime.w);
    output.position = mul(float4(worldPosition, 1.0), MainViewProj);
    output.worldPosition = worldPosition;
    output.normal = input.normal;
    output.color = input.color;
    output.viewDepth = mul(float4(worldPosition, 1.0), MainView).z;
    return output;
}

ShadowVertexOutput ShadowVS(VertexInput input)
{
    ShadowVertexOutput output;
    const float3 worldPosition = AnimateVertex(input, CameraPositionTime.w);
    output.position =
        mul(float4(worldPosition, 1.0), LightViewProj[ActiveCascade]);
    return output;
}

float ShadowPS(ShadowVertexOutput input) : SV_Depth
{
    const float depth = input.position.z;
    const float slope = max(abs(ddx(depth)), abs(ddy(depth)));
    const float bias = ShadowBiasAndMapSize.x * slope + ShadowBiasAndMapSize.y;
    return saturate(depth + bias);
}

float SamplePcf9(uint cascade, float2 uv, float receiverDepth)
{
    const float texel = 1.0 / ShadowBiasAndMapSize.z;
    float visible = 0.0;

    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            const float2 offset = float2(x, y) * texel;
            const float storedDepth =
                ShadowMaps.SampleLevel(ShadowPointSampler,
                                       float3(uv + offset, cascade), 0.0);
            visible += storedDepth + 1e-6 >= receiverDepth ? 1.0 : 0.0;
        }
    }
    return visible / 9.0;
}

float ShadowAtPosition(float3 worldPosition, float viewDepth)
{
    const float split0 = CascadeDepths.y;
    const float split1 = CascadeDepths.z;
    const float transition0 = CascadeTransitions.x;
    const float transition1 = CascadeTransitions.y;

    const float fade0 = smoothstep(split0 - transition0,
                                   split0 + transition0, viewDepth);
    const float fade1 = smoothstep(split1 - transition1,
                                   split1 + transition1, viewDepth);

    const float3 weights = float3(
        1.0 - fade0,
        fade0 * (1.0 - fade1),
        fade1);

    float visibility = 0.0;
    float weightSum = 0.0;

    [unroll]
    for (uint cascade = 0; cascade < 3; ++cascade)
    {
        if (weights[cascade] > 1e-5)
        {
            const float4 lightPosition =
                mul(float4(worldPosition, 1.0), LightViewProj[cascade]);
            const float2 uv =
                float2(lightPosition.x * 0.5 + 0.5,
                       0.5 - lightPosition.y * 0.5);
            visibility +=
                weights[cascade] * SamplePcf9(cascade, uv, lightPosition.z);
            weightSum += weights[cascade];
        }
    }

    return visibility / max(weightSum, 1e-5);
}

float4 MainPS(MainVertexOutput input) : SV_Target
{
    const float3 normal = normalize(input.normal);
    const float3 albedo = saturate(input.color);
    const float3 toLight = -normalize(LightDirectionIntensity.xyz);
    const float diffuseAmount = saturate(dot(normal, toLight));
    const float shadow = ShadowAtPosition(input.worldPosition, input.viewDepth);

    const float3 ambient = albedo * float3(0.19, 0.22, 0.27) *
                           (0.58 + 0.42 * saturate(normal.y));
    const float3 sunColor = float3(1.0, 0.94, 0.82);
    const float3 direct = albedo * sunColor * diffuseAmount *
                          LightDirectionIntensity.w * shadow;

    const float3 linearColor = ambient + direct;
    const float3 displayColor = pow(saturate(linearColor), 1.0 / 2.2);
    return float4(displayColor, 1.0);
}

