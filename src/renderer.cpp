#include "renderer/renderer.hpp"

#include "renderer/error.hpp"
#include "renderer/logger.hpp"

#include <glad/gl.h>

#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace renderer {
namespace {

constexpr int kMaxLights = 32;
constexpr int kMaxSlots = kMaxLights * 6;

const char* kGeometryVertexShader = R"GLSL(#version 450 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uViewProjection;
uniform mat4 uModel;
uniform mat3 uNormalMatrix;

out vec3 vWorldPosition;
out vec3 vWorldNormal;

void main() {
    vec4 world = uModel * vec4(aPosition, 1.0);
    vWorldPosition = world.xyz;
    vWorldNormal = normalize(uNormalMatrix * aNormal);
    gl_Position = uViewProjection * world;
}
)GLSL";

const char* kGeometryFragmentShader = R"GLSL(#version 450 core
layout(location = 0) out vec4 oPosition;
layout(location = 1) out vec4 oNormal;
layout(location = 2) out vec4 oAlbedo;
layout(location = 3) out vec4 oMaterial;

in vec3 vWorldPosition;
in vec3 vWorldNormal;

uniform vec3 uAlbedo;
uniform float uMetallic;
uniform float uRoughness;

void main() {
    oPosition = vec4(vWorldPosition, 1.0);
    oNormal = vec4(normalize(vWorldNormal), 1.0);
    oAlbedo = vec4(uAlbedo, 1.0);
    oMaterial = vec4(uMetallic, uRoughness, 0.0, 1.0);
}
)GLSL";

const char* kShadowVertexShader = R"GLSL(#version 450 core
layout(location = 0) in vec3 aPosition;
uniform mat4 uModel;
uniform mat4 uViewProjection;
void main() {
    gl_Position = uViewProjection * uModel * vec4(aPosition, 1.0);
}
)GLSL";

const char* kShadowFragmentShader = R"GLSL(#version 450 core
void main() {
}
)GLSL";

const char* kFullscreenVertexShader = R"GLSL(#version 450 core
out vec2 vUv;
void main() {
    vec2 positions[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    vec2 position = positions[gl_VertexID];
    vUv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
)GLSL";

const char* kLightingFragmentShader = R"GLSL(#version 450 core
const int MAX_LIGHTS = 32;
const int MAX_SLOTS = 192;
const float PI = 3.14159265358979323846;

layout(location = 0) out vec4 oHdrColor;
layout(location = 1) out float oShadowVisibility;

in vec2 vUv;

uniform sampler2D uGPosition;
uniform sampler2D uGNormal;
uniform sampler2D uGAlbedo;
uniform sampler2D uGMaterial;
uniform sampler2D uShadowAtlas;

uniform vec3 uCameraPosition;
uniform vec3 uBackground;
uniform int uLightCount;
uniform int uLightType[MAX_LIGHTS];
uniform vec3 uLightPosition[MAX_LIGHTS];
uniform vec3 uLightDirection[MAX_LIGHTS];
uniform vec3 uLightColor[MAX_LIGHTS];
uniform float uLightIntensity[MAX_LIGHTS];
uniform float uLightRange[MAX_LIGHTS];
uniform vec2 uLightConeCos[MAX_LIGHTS];
uniform float uLightShadowBias[MAX_LIGHTS];
uniform int uLightCastShadow[MAX_LIGHTS];
uniform int uLightFirstSlot[MAX_LIGHTS];
uniform int uLightFaceCount[MAX_LIGHTS];

uniform int uSlotColumn[MAX_SLOTS];
uniform int uSlotRow[MAX_SLOTS];
uniform mat4 uSlotViewProjection[MAX_SLOTS];

uniform int uAtlasColumns;
uniform int uAtlasRows;
uniform int uAtlasTileResolution;
uniform int uPcfKernel;
uniform int uShadowSamples;
uniform float uPcfStep;

float distributionGGX(float nDotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denominator = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denominator * denominator, 1.0e-7);
}

float geometrySchlickGGX(float nDotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return nDotV / max(nDotV * (1.0 - k) + k, 1.0e-7);
}

float geometrySmith(float nDotV, float nDotL, float roughness) {
    return geometrySchlickGGX(nDotV, roughness) * geometrySchlickGGX(nDotL, roughness);
}

vec3 fresnelSchlick(float cosine, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

float sampleShadowSlot(int slot, vec3 worldPosition, float bias, out bool valid) {
    vec4 clip = uSlotViewProjection[slot] * vec4(worldPosition, 1.0);
    if (clip.w <= 1.0e-6) {
        valid = false;
        return 1.0;
    }
    vec3 ndc = clip.xyz / clip.w;
    vec2 tileUv = ndc.xy * 0.5 + 0.5;
    if (ndc.z < -1.0 || ndc.z > 1.0
        || tileUv.x < 0.0 || tileUv.x > 1.0
        || tileUv.y < 0.0 || tileUv.y > 1.0) {
        valid = false;
        return 1.0;
    }
    valid = true;

    float reference = ndc.z * 0.5 + 0.5 - bias;
    int halfKernel = uPcfKernel / 2;
    float visible = 0.0;
    int taken = 0;
    float texel = 1.0 / float(uAtlasTileResolution);

    for (int ring = 0; ring <= halfKernel; ++ring) {
        for (int y = -ring; y <= ring; ++y) {
            for (int x = -ring; x <= ring; ++x) {
                if (taken >= uShadowSamples) {
                    break;
                }
                if (max(abs(x), abs(y)) != ring) {
                    continue;
                }
                vec2 offset = vec2(float(x), float(y)) * uPcfStep * texel;
                vec2 sampleTileUv = clamp(tileUv + offset,
                                          vec2(0.5 * texel), vec2(1.0 - 0.5 * texel));
                vec2 atlasUv =
                    (vec2(float(uSlotColumn[slot]), float(uSlotRow[slot])) + sampleTileUv)
                    / vec2(float(uAtlasColumns), float(uAtlasRows));
                float storedDepth = texture(uShadowAtlas, atlasUv).r;
                visible += reference <= storedDepth ? 1.0 : 0.0;
                ++taken;
            }
            if (taken >= uShadowSamples) {
                break;
            }
        }
        if (taken >= uShadowSamples) {
            break;
        }
    }
    return visible / float(max(taken, 1));
}

float sampleSingleShadow(int lightIndex, vec3 worldPosition, float bias) {
    bool valid = false;
    float value = sampleShadowSlot(
        uLightFirstSlot[lightIndex], worldPosition, bias, valid);
    return valid ? value : 1.0;
}

float samplePointShadow(int lightIndex, vec3 worldPosition, float bias) {
    float result = -1.0;
    int firstSlot = uLightFirstSlot[lightIndex];
    int faceCount = uLightFaceCount[lightIndex];
    for (int face = 0; face < faceCount; ++face) {
        bool valid = false;
        float value = sampleShadowSlot(firstSlot + face, worldPosition, bias, valid);
        if (valid) {
            result = max(result, value);
        }
    }
    return result < 0.0 ? 1.0 : result;
}

void main() {
    vec4 positionSample = texture(uGPosition, vUv);
    if (positionSample.a < 0.5) {
        oHdrColor = vec4(uBackground, 0.0);
        oShadowVisibility = 1.0;
        return;
    }

    vec3 worldPosition = positionSample.xyz;
    vec3 normal = normalize(texture(uGNormal, vUv).xyz);
    vec3 albedo = clamp(texture(uGAlbedo, vUv).rgb, 0.0, 1.0);
    vec2 material = texture(uGMaterial, vUv).rg;
    float metallic = clamp(material.x, 0.0, 1.0);
    float roughness = clamp(material.y, 0.045, 1.0);

    vec3 viewDirection = normalize(uCameraPosition - worldPosition);
    float nDotV = max(dot(normal, viewDirection), 1.0e-4);
    vec3 f0 = mix(vec3(0.04), albedo, metallic);

    vec3 shadowedRadiance = vec3(0.0);
    vec3 unshadowedRadiance = vec3(0.0);

    for (int lightIndex = 0; lightIndex < MAX_LIGHTS; ++lightIndex) {
        if (lightIndex >= uLightCount) {
            break;
        }

        int type = uLightType[lightIndex];
        vec3 lightDirection;
        vec3 radiance;

        if (type == 0) {
            vec3 delta = uLightPosition[lightIndex] - worldPosition;
            float distanceToLight = length(delta);
            if (distanceToLight <= 1.0e-5) {
                continue;
            }
            lightDirection = delta / distanceToLight;
            float ratio = distanceToLight / max(uLightRange[lightIndex], 1.0e-5);
            float smoothRange = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
            float attenuation =
                (smoothRange * smoothRange) / max(distanceToLight * distanceToLight, 0.001);
            radiance = uLightColor[lightIndex] * uLightIntensity[lightIndex] * attenuation;
        } else if (type == 1) {
            lightDirection = normalize(-uLightDirection[lightIndex]);
            radiance = uLightColor[lightIndex] * uLightIntensity[lightIndex];
        } else {
            vec3 delta = uLightPosition[lightIndex] - worldPosition;
            float distanceToLight = length(delta);
            if (distanceToLight <= 1.0e-5) {
                continue;
            }
            lightDirection = delta / distanceToLight;
            float ratio = distanceToLight / max(uLightRange[lightIndex], 1.0e-5);
            float smoothRange = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
            float attenuation =
                (smoothRange * smoothRange) / max(distanceToLight * distanceToLight, 0.001);
            float cosine = dot(-lightDirection, uLightDirection[lightIndex]);
            float cone = smoothstep(uLightConeCos[lightIndex].y,
                                    uLightConeCos[lightIndex].x, cosine);
            radiance = uLightColor[lightIndex] * uLightIntensity[lightIndex]
                     * attenuation * cone;
        }

        float nDotL = max(dot(normal, lightDirection), 0.0);
        if (nDotL <= 0.0) {
            continue;
        }

        vec3 halfDirection = normalize(viewDirection + lightDirection);
        float nDotH = max(dot(normal, halfDirection), 0.0);
        float vDotH = max(dot(viewDirection, halfDirection), 0.0);
        float distribution = distributionGGX(nDotH, roughness);
        float geometry = geometrySmith(nDotV, nDotL, roughness);
        vec3 fresnel = fresnelSchlick(vDotH, f0);
        vec3 specular = (distribution * geometry * fresnel)
                      / max(4.0 * nDotV * nDotL, 1.0e-6);
        vec3 diffuseWeight = (vec3(1.0) - fresnel) * (1.0 - metallic);
        vec3 diffuse = diffuseWeight * albedo / PI;
        vec3 contribution = (diffuse + specular) * radiance * nDotL;

        float visibility = 1.0;
        if (uLightCastShadow[lightIndex] != 0) {
            float bias = uLightShadowBias[lightIndex];
            visibility = type == 0
                ? samplePointShadow(lightIndex, worldPosition, bias)
                : sampleSingleShadow(lightIndex, worldPosition, bias);
        }
        shadowedRadiance += contribution * visibility;
        unshadowedRadiance += contribution;
    }

    float luminanceDenominator = max(dot(unshadowedRadiance, vec3(0.2126, 0.7152, 0.0722)),
                                     1.0e-8);
    float visibilityRatio =
        clamp(dot(shadowedRadiance, vec3(0.2126, 0.7152, 0.0722)) / luminanceDenominator,
              0.0, 1.0);
    oHdrColor = vec4(shadowedRadiance, 1.0);
    oShadowVisibility = visibilityRatio;
}
)GLSL";

const char* kFilterFragmentShader = R"GLSL(#version 450 core
layout(location = 0) out float oVisibility;
in vec2 vUv;
uniform sampler2D uVisibility;
uniform int uRadius;
uniform float uWeights[17];

void main() {
    ivec2 size = textureSize(uVisibility, 0);
    ivec2 center = ivec2(gl_FragCoord.xy);
    float accumulated = 0.0;
    float weightSum = 0.0;
    for (int y = -8; y <= 8; ++y) {
        if (abs(y) > uRadius) {
            continue;
        }
        for (int x = -8; x <= 8; ++x) {
            if (abs(x) > uRadius) {
                continue;
            }
            float weight = uWeights[x + uRadius] * uWeights[y + uRadius];
            ivec2 coordinate = clamp(center + ivec2(x, y), ivec2(0), size - 1);
            accumulated += texelFetch(uVisibility, coordinate, 0).r * weight;
            weightSum += weight;
        }
    }
    oVisibility = accumulated / max(weightSum, 1.0e-8);
}
)GLSL";

const char* kCompositeFragmentShader = R"GLSL(#version 450 core
layout(location = 0) out vec4 oColor;
in vec2 vUv;
uniform sampler2D uHdr;
uniform sampler2D uVisibility;
uniform sampler2D uFilteredVisibility;
uniform bool uUseFilter;

void main() {
    vec4 color = texture(uHdr, vUv);
    if (color.a < 0.5 || !uUseFilter) {
        oColor = color;
        return;
    }

    ivec2 coordinate = ivec2(gl_FragCoord.xy);
    float before = texelFetch(uVisibility, coordinate, 0).r;
    if (before >= 0.999) {
        oColor = color;
        return;
    }
    float after = texelFetch(uFilteredVisibility, coordinate, 0).r;
    float ratio = clamp(after / max(before, 0.05), 0.0, 4.0);
    color.rgb *= ratio;
    oColor = color;
}
)GLSL";

const char* kTonemapFragmentShader = R"GLSL(#version 450 core
layout(location = 0) out vec4 oColor;
in vec2 vUv;
uniform sampler2D uHdr;
uniform int uToneMapping;
uniform float uExposure;

vec3 acesFilmic(vec3 color) {
    return clamp((color * (2.51 * color + 0.03))
               / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
}

vec3 linear_to_srgb(vec3 color) {
    vec3 low = color * 12.92;
    vec3 high = 1.055 * pow(max(color, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(low, high, step(vec3(0.0031308), color));
}

void main() {
    vec3 hdr = max(texture(uHdr, vUv).rgb, vec3(0.0)) * uExposure;
    vec3 mapped;
    if (uToneMapping == 0) {
        mapped = acesFilmic(hdr);
    } else {
        mapped = hdr / (vec3(1.0) + hdr);
    }
    oColor = vec4(linear_to_srgb(mapped), 1.0);
}
)GLSL";

GLuint get_uniform(GLuint program, const char* name) {
    const GLint location = glGetUniformLocation(program, name);
    if (location < 0) {
        fail("E_GL_RESOURCE", std::string("missing shader uniform: ") + name);
    }
    return static_cast<GLuint>(location);
}

std::string gl_error_hex(GLenum error) {
    std::ostringstream stream;
    stream << "0x" << std::hex << static_cast<unsigned int>(error);
    return stream.str();
}

double query_milliseconds(GLuint query) {
    GLuint64 nanoseconds = 0;
    glGetQueryObjectui64v(query, GL_QUERY_RESULT, &nanoseconds);
    return static_cast<double>(nanoseconds) / 1000000.0;
}

const char* type_name(LightType type) {
    switch (type) {
    case LightType::Point:
        return "point";
    case LightType::Directional:
        return "directional";
    case LightType::Spot:
        return "spot";
    }
    return "unknown";
}

void png_write_callback(void* context, void* data, int size) {
    auto* stream = static_cast<std::ofstream*>(context);
    stream->write(static_cast<const char*>(data), size);
}

}  // namespace

Renderer::~Renderer() {
    shutdown();
}

void Renderer::initialize(const Scene& scene, const std::string& mode, Logger& logger) {
    if (initialized_) {
        logger.info("renderer_initialize", {log_field("reused", true)});
        return;
    }
    try {
        build_pipeline(scene, mode, logger);
        initialized_ = true;
    } catch (...) {
        shutdown();
        throw;
    }
}

RenderStats Renderer::render(const Scene& scene, const std::string& mode,
                             const std::string& outputPath, Logger& logger) {
    if (!initialized_) {
        fail("E_RENDER", "renderer must be initialized before rendering");
    }
    if (!is_valid_mode(mode)) {
        fail("E_USAGE", "mode must be 'fast' or 'high'");
    }
    const ModeSettings& modeConfig = mode_settings(scene, mode);
    if (modeConfig.shadowSamples != shadowSamples_
        || modeConfig.postprocessIterations != postprocessIterations_
        || modeConfig.gaussianFilter != gaussianFilter_
        || effective_tile_resolution(scene, mode) != tileResolution_
        || scene.settings.denoise.filterRadius != filterRadius_
        || scene.settings.denoise.weights != filterWeights_
        || lightPlans_.size() != scene.lights.size()
        || scene.settings.width != width_ || scene.settings.height != height_) {
        fail("E_RENDER", "scene or mode changed after renderer initialization");
    }

    RenderStats stats;
    stats.shadowSamples = shadowSamples_;
    stats.postprocessIterations = postprocessIterations_;
    stats.gaussianFilterApplied = gaussianFilter_;
    stats.tileResolution = tileResolution_;
    stats.atlasWidth = atlasWidth_;
    stats.atlasHeight = atlasHeight_;

    log_light_plan(scene, logger);
    logger.info("render_start", {
        log_field("mode", mode),
        log_field("width", width_),
        log_field("height", height_),
        log_field("shadow_samples", shadowSamples_),
        log_field("postprocess_iterations", postprocessIterations_),
        log_field("gaussian_filter", gaussianFilter_),
        log_field("atlas_tile_resolution", tileResolution_),
        log_field("atlas_width", atlasWidth_),
        log_field("atlas_height", atlasHeight_),
    });

    glBindFramebuffer(GL_FRAMEBUFFER, geometryFbo_);
    glViewport(0, 0, width_, height_);
    glBeginQuery(GL_TIME_ELAPSED, geometryQuery_);
    render_geometry_pass(scene);
    glEndQuery(GL_TIME_ELAPSED);

    glBindFramebuffer(GL_FRAMEBUFFER, shadowFbo_);
    glViewport(0, 0, atlasWidth_, atlasHeight_);
    glBeginQuery(GL_TIME_ELAPSED, shadowQuery_);
    render_shadow_atlas(scene);
    glEndQuery(GL_TIME_ELAPSED);

    glBindFramebuffer(GL_FRAMEBUFFER, lightFbo_);
    glViewport(0, 0, width_, height_);
    glBeginQuery(GL_TIME_ELAPSED, lightingQuery_);
    render_lighting_pass(scene);
    glEndQuery(GL_TIME_ELAPSED);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glFinish();

    stats.geometryGpuMs = query_milliseconds(geometryQuery_);
    stats.shadowGpuMs = query_milliseconds(shadowQuery_);
    stats.lightingGpuMs = query_milliseconds(lightingQuery_);

    glBindTexture(GL_TEXTURE_2D, gPosition_);
    std::vector<float> positionData(
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * 4u);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, positionData.data());

    std::vector<unsigned char> geometryMask(
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_), 0);
    for (std::size_t pixel = 0; pixel < geometryMask.size(); ++pixel) {
        geometryMask[pixel] = positionData[pixel * 4 + 3] >= 0.5f ? 1u : 0u;
        stats.nonBackgroundPixels += geometryMask[pixel] != 0 ? 1u : 0u;
    }
    stats.totalPixels = geometryMask.size();
    stats.nonBackgroundPercent =
        stats.totalPixels == 0
            ? 0.0
            : 100.0 * static_cast<double>(stats.nonBackgroundPixels)
                    / static_cast<double>(stats.totalPixels);

    auto read_visibility = [&](GLuint texture) {
        std::vector<float> values(
            static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_));
        glBindTexture(GL_TEXTURE_2D, texture);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, values.data());
        return values;
    };

    auto compute_stats = [&](const std::vector<float>& values) {
        ShadowIntensityStats result;
        double sum = 0.0;
        result.minimum = std::numeric_limits<double>::max();
        result.maximum = std::numeric_limits<double>::lowest();
        for (std::size_t pixel = 0; pixel < values.size(); ++pixel) {
            if (geometryMask[pixel] == 0) {
                continue;
            }
            const double value = static_cast<double>(values[pixel]);
            result.minimum = std::min(result.minimum, value);
            result.maximum = std::max(result.maximum, value);
            sum += value;
            ++result.geometryPixels;
            if (value < 0.999) {
                ++result.shadowedPixels;
            }
        }
        if (result.geometryPixels == 0) {
            result.minimum = 0.0;
            result.maximum = 0.0;
            result.mean = 0.0;
        } else {
            result.mean = sum / static_cast<double>(result.geometryPixels);
        }
        return result;
    };

    const std::vector<float> visibilityBefore = read_visibility(shadowVisibility_);
    stats.shadowBefore = compute_stats(visibilityBefore);

    unsigned int finalVisibility = 0;
    render_denoise(scene, finalVisibility);
    const std::vector<float> visibilityAfter = read_visibility(finalVisibility);
    stats.shadowAfter = compute_stats(visibilityAfter);

    render_composite_and_tonemap(scene, finalVisibility);
    write_png(outputPath);

    stats.lights.clear();
    for (std::size_t index = 0; index < scene.lights.size(); ++index) {
        LightAtlasInfo info;
        info.lightIndex = static_cast<int>(index);
        info.type = type_name(scene.lights[index].type);
        info.castShadow = scene.lights[index].castShadow;
        info.firstSlot = lightPlans_[index].firstSlot;
        info.faceCount = lightPlans_[index].faceCount;
        info.slots = lightPlans_[index].slots;
        stats.lights.push_back(std::move(info));
    }

    logger.info("gpu_timings", {
        log_field("geometry_gpu_ms", stats.geometryGpuMs, 3),
        log_field("shadow_gpu_ms", stats.shadowGpuMs, 3),
        log_field("lighting_gpu_ms", stats.lightingGpuMs, 3),
    });
    logger.info("shadow_stats", {
        log_field("filter_applied", stats.gaussianFilterApplied),
        log_field("filter_iterations", stats.postprocessIterations),
        log_field("filter_radius", filterRadius_),
        log_field("before_min", stats.shadowBefore.minimum, 6),
        log_field("before_mean", stats.shadowBefore.mean, 6),
        log_field("before_max", stats.shadowBefore.maximum, 6),
        log_field("after_min", stats.shadowAfter.minimum, 6),
        log_field("after_mean", stats.shadowAfter.mean, 6),
        log_field("after_max", stats.shadowAfter.maximum, 6),
        log_field("geometry_pixels", static_cast<unsigned int>(stats.shadowBefore.geometryPixels)),
        log_field("shadowed_pixels", static_cast<unsigned int>(stats.shadowBefore.shadowedPixels)),
    });
    logger.info("image_stats", {
        log_field("non_background_pixels",
                  static_cast<unsigned int>(stats.nonBackgroundPixels)),
        log_field("total_pixels", static_cast<unsigned int>(stats.totalPixels)),
        log_field("non_background_percent", stats.nonBackgroundPercent, 6),
        log_field("output", outputPath),
    });
    logger.info("render_complete", {
        log_field("output", outputPath),
        log_field("mode", mode),
    });
    return stats;
}

void Renderer::shutdown() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    glBindVertexArray(0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    delete_framebuffer(geometryFbo_);
    delete_framebuffer(shadowFbo_);
    delete_framebuffer(lightFbo_);
    delete_framebuffer(denoiseFbo_);
    delete_framebuffer(compositeFbo_);
    delete_framebuffer(tonemapFbo_);

    delete_renderbuffer(geometryDepthRbo_);

    delete_texture(gPosition_);
    delete_texture(gNormal_);
    delete_texture(gAlbedo_);
    delete_texture(gMaterial_);
    delete_texture(shadowAtlas_);
    delete_texture(hdrColor_);
    delete_texture(shadowVisibility_);
    delete_texture(filterScratchA_);
    delete_texture(filterScratchB_);
    delete_texture(hdrFinal_);
    delete_texture(outputTexture_);

    delete_program(geometryProgram_);
    delete_program(shadowProgram_);
    delete_program(lightingProgram_);
    delete_program(filterProgram_);
    delete_program(compositeProgram_);
    delete_program(tonemapProgram_);

    for (MeshGpu& mesh : meshes_) {
        delete_vertex_array(mesh.vao);
        delete_buffer(mesh.vertexBuffer);
        delete_buffer(mesh.indexBuffer);
    }
    meshes_.clear();
    delete_vertex_array(fullscreenVao_);

    delete_query(geometryQuery_);
    delete_query(shadowQuery_);
    delete_query(lightingQuery_);

    slots_.clear();
    lightPlans_.clear();
    filterWeights_.clear();
    initialized_ = false;
}

void Renderer::build_pipeline(const Scene& scene, const std::string& mode, Logger& logger) {
    width_ = scene.settings.width;
    height_ = scene.settings.height;
    atlasColumns_ = scene.settings.shadowAtlas.columns;
    atlasRows_ = scene.settings.shadowAtlas.rows;
    pcfKernel_ = scene.settings.shadowAtlas.pcfKernel;
    pcfStep_ = scene.settings.shadowAtlas.pcfStep;
    tileResolution_ = effective_tile_resolution(scene, mode);
    atlasWidth_ = tileResolution_ * atlasColumns_;
    atlasHeight_ = tileResolution_ * atlasRows_;
    shadowSamples_ = mode_settings(scene, mode).shadowSamples;
    postprocessIterations_ = mode_settings(scene, mode).postprocessIterations;
    gaussianFilter_ = mode_settings(scene, mode).gaussianFilter;
    filterRadius_ = scene.settings.denoise.filterRadius;
    filterWeights_ = scene.settings.denoise.weights;

    GLint maximumTextureSize = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maximumTextureSize);
    if (maximumTextureSize <= 0
        || width_ > maximumTextureSize || height_ > maximumTextureSize
        || atlasWidth_ > maximumTextureSize || atlasHeight_ > maximumTextureSize) {
        fail("E_GL_RESOURCE", "scene textures exceed GL_MAX_TEXTURE_SIZE");
    }

    build_shadow_plan(scene);
    create_programs();
    create_meshes(scene);
    create_targets(scene);
    fullscreenVao_ = create_vertex_array();
    geometryQuery_ = create_query();
    shadowQuery_ = create_query();
    lightingQuery_ = create_query();

    glDisable(GL_BLEND);
    glDisable(GL_DITHER);
    glDisable(GL_FRAMEBUFFER_SRGB);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    check_gl("pipeline initialization");

    logger.info("renderer_initialized", {
        log_field("resource_objects", resourceCounts_.total()),
        log_field("shader_programs", resourceCounts_.shaderPrograms),
        log_field("textures", resourceCounts_.textures),
        log_field("framebuffers", resourceCounts_.framebuffers),
        log_field("buffers", resourceCounts_.buffers),
        log_field("vertex_arrays", resourceCounts_.vertexArrays),
        log_field("queries", resourceCounts_.queries),
    });
}

void Renderer::build_shadow_plan(const Scene& scene) {
    slots_.clear();
    lightPlans_.clear();
    int nextSlot = 0;

    for (std::size_t lightIndex = 0; lightIndex < scene.lights.size(); ++lightIndex) {
        const Light& light = scene.lights[lightIndex];
        LightPlan plan;
        if (light.castShadow) {
            plan.firstSlot = nextSlot;
            plan.faceCount = light.type == LightType::Point ? 6 : 1;
            for (int face = 0; face < plan.faceCount; ++face) {
                if (nextSlot >= atlasColumns_ * atlasRows_) {
                    fail("E_INVALID_SCENE", "shadow atlas capacity was exceeded");
                }
                Mat4 view{};
                Mat4 projection{};
                if (light.type == LightType::Directional) {
                    const Vec3 center = scene.bounds.center();
                    const Vec3 direction = normalize(light.direction);
                    const Vec3 up =
                        std::abs(direction.y) > 0.95f ? Vec3{0.0f, 0.0f, 1.0f}
                                                     : Vec3{0.0f, 1.0f, 0.0f};
                    const Vec3 eye = center - direction * (light.farPlane * 0.5f);
                    view = lookAt(eye, center, up);
                    projection = orthographic(
                        -light.orthoSize, light.orthoSize,
                        -light.orthoSize, light.orthoSize,
                        light.nearPlane, light.farPlane);
                } else if (light.type == LightType::Spot) {
                    const Vec3 up =
                        std::abs(light.direction.y) > 0.95f ? Vec3{0.0f, 0.0f, 1.0f}
                                                           : Vec3{0.0f, 1.0f, 0.0f};
                    view = lookAt(light.position, light.position + light.direction, up);
                    projection = perspective(
                        radians(light.outerConeDegrees * 2.0f), 1.0f,
                        light.nearPlane, light.farPlane);
                } else {
                    static constexpr std::array<Vec3, 6> directions{{
                        {1.0f, 0.0f, 0.0f},
                        {-1.0f, 0.0f, 0.0f},
                        {0.0f, 1.0f, 0.0f},
                        {0.0f, -1.0f, 0.0f},
                        {0.0f, 0.0f, 1.0f},
                        {0.0f, 0.0f, -1.0f},
                    }};
                    static constexpr std::array<Vec3, 6> ups{{
                        {0.0f, -1.0f, 0.0f},
                        {0.0f, -1.0f, 0.0f},
                        {0.0f, 0.0f, 1.0f},
                        {0.0f, 0.0f, -1.0f},
                        {0.0f, -1.0f, 0.0f},
                        {0.0f, -1.0f, 0.0f},
                    }};
                    view = lookAt(light.position,
                                  light.position + directions[static_cast<std::size_t>(face)],
                                  ups[static_cast<std::size_t>(face)]);
                    projection = perspective(kPi * 0.5f, 1.0f,
                                             light.nearPlane, light.farPlane);
                }

                SlotPlan slot;
                slot.lightIndex = static_cast<int>(lightIndex);
                slot.faceIndex = face;
                slot.column = nextSlot % atlasColumns_;
                slot.row = nextSlot / atlasColumns_;
                slot.viewProjection = projection * view;
                slots_.push_back(slot);

                plan.slots.push_back(nextSlot);
                plan.slotColumns.push_back(slot.column);
                plan.slotRows.push_back(slot.row);
                plan.slotMatrices.push_back(slot.viewProjection);
                ++nextSlot;
            }
        }
        lightPlans_.push_back(std::move(plan));
    }
}

void Renderer::create_programs() {
    geometryProgram_ = create_program(
        kGeometryVertexShader, kGeometryFragmentShader, "geometry");
    shadowProgram_ = create_program(
        kShadowVertexShader, kShadowFragmentShader, "shadow");
    lightingProgram_ = create_program(
        kFullscreenVertexShader, kLightingFragmentShader, "lighting");
    filterProgram_ = create_program(
        kFullscreenVertexShader, kFilterFragmentShader, "filter");
    compositeProgram_ = create_program(
        kFullscreenVertexShader, kCompositeFragmentShader, "composite");
    tonemapProgram_ = create_program(
        kFullscreenVertexShader, kTonemapFragmentShader, "tonemap");
}

void Renderer::create_meshes(const Scene& scene) {
    meshes_.clear();
    meshes_.reserve(scene.meshes.size());
    for (const Mesh& mesh : scene.meshes) {
        meshes_.push_back({});
        MeshGpu& gpu = meshes_.back();
        gpu.vao = create_vertex_array();
        gpu.vertexBuffer = create_buffer();
        gpu.indexBuffer = create_buffer();
        gpu.indexCount = static_cast<int>(mesh.indices.size());
        gpu.materialIndex = mesh.material;

        glBindVertexArray(gpu.vao);
        glBindBuffer(GL_ARRAY_BUFFER, gpu.vertexBuffer);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(mesh.vertices.size() * sizeof(Vertex)),
                     mesh.vertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpu.indexBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(mesh.indices.size() * sizeof(std::uint32_t)),
                     mesh.indices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                              reinterpret_cast<const void*>(offsetof(Vertex, position)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                              reinterpret_cast<const void*>(offsetof(Vertex, normal)));
        glBindVertexArray(0);
    }
}

void Renderer::create_targets(const Scene& scene) {
    gPosition_ = create_texture_2d(width_, height_, GL_RGBA32F, GL_RGBA, GL_FLOAT,
                                   GL_NEAREST, GL_NEAREST);
    gNormal_ = create_texture_2d(width_, height_, GL_RGBA16F, GL_RGBA, GL_FLOAT,
                                 GL_NEAREST, GL_NEAREST);
    gAlbedo_ = create_texture_2d(width_, height_, GL_RGBA16F, GL_RGBA, GL_FLOAT,
                                 GL_NEAREST, GL_NEAREST);
    gMaterial_ = create_texture_2d(width_, height_, GL_RG16F, GL_RG, GL_FLOAT,
                                   GL_NEAREST, GL_NEAREST);
    shadowAtlas_ = create_texture_2d(atlasWidth_, atlasHeight_, GL_DEPTH_COMPONENT32F,
                                     GL_DEPTH_COMPONENT, GL_FLOAT,
                                     GL_NEAREST, GL_NEAREST);
    hdrColor_ = create_texture_2d(width_, height_, GL_RGBA16F, GL_RGBA, GL_FLOAT,
                                  GL_LINEAR, GL_LINEAR);
    shadowVisibility_ = create_texture_2d(width_, height_, GL_R16F, GL_RED, GL_FLOAT,
                                          GL_LINEAR, GL_LINEAR);
    filterScratchA_ = create_texture_2d(width_, height_, GL_R16F, GL_RED, GL_FLOAT,
                                        GL_LINEAR, GL_LINEAR);
    filterScratchB_ = create_texture_2d(width_, height_, GL_R16F, GL_RED, GL_FLOAT,
                                        GL_LINEAR, GL_LINEAR);
    hdrFinal_ = create_texture_2d(width_, height_, GL_RGBA16F, GL_RGBA, GL_FLOAT,
                                  GL_LINEAR, GL_LINEAR);
    outputTexture_ = create_texture_2d(width_, height_, GL_RGBA8, GL_RGBA,
                                       GL_UNSIGNED_BYTE, GL_NEAREST, GL_NEAREST);

    geometryDepthRbo_ = create_renderbuffer();
    glBindRenderbuffer(GL_RENDERBUFFER, geometryDepthRbo_);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width_, height_);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    auto check_framebuffer = [](const char* label) {
        const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            fail("E_GL_RESOURCE",
                 std::string("incomplete framebuffer: ") + label + " " + gl_error_hex(status));
        }
    };

    geometryFbo_ = create_framebuffer();
    glBindFramebuffer(GL_FRAMEBUFFER, geometryFbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, gPosition_, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                           GL_TEXTURE_2D, gNormal_, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2,
                           GL_TEXTURE_2D, gAlbedo_, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3,
                           GL_TEXTURE_2D, gMaterial_, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, geometryDepthRbo_);
    const GLenum geometryBuffers[4] = {
        GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
        GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3,
    };
    glDrawBuffers(4, geometryBuffers);
    check_framebuffer("geometry");

    shadowFbo_ = create_framebuffer();
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                           GL_TEXTURE_2D, shadowAtlas_, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    check_framebuffer("shadow atlas");

    lightFbo_ = create_framebuffer();
    glBindFramebuffer(GL_FRAMEBUFFER, lightFbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, hdrColor_, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                           GL_TEXTURE_2D, shadowVisibility_, 0);
    const GLenum lightBuffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, lightBuffers);
    check_framebuffer("lighting");

    denoiseFbo_ = create_framebuffer();
    glBindFramebuffer(GL_FRAMEBUFFER, denoiseFbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, filterScratchA_, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    check_framebuffer("denoise");

    compositeFbo_ = create_framebuffer();
    glBindFramebuffer(GL_FRAMEBUFFER, compositeFbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, hdrFinal_, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    check_framebuffer("composite");

    tonemapFbo_ = create_framebuffer();
    glBindFramebuffer(GL_FRAMEBUFFER, tonemapFbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, outputTexture_, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    check_framebuffer("tonemap");

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    (void)scene;
}

void Renderer::render_geometry_pass(const Scene& scene) {
    glBindFramebuffer(GL_FRAMEBUFFER, geometryFbo_);
    glViewport(0, 0, width_, height_);
    const GLenum buffers[4] = {
        GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
        GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3,
    };
    glDrawBuffers(4, buffers);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glDisable(GL_BLEND);

    const float aspect = static_cast<float>(width_) / static_cast<float>(height_);
    const Mat4 view = lookAt(
        scene.camera.position, scene.camera.target, scene.camera.up);
    const Mat4 projection = perspective(
        radians(scene.camera.fovYDegrees), aspect,
        scene.camera.nearPlane, scene.camera.farPlane);
    const Mat4 viewProjection = projection * view;

    glUseProgram(geometryProgram_);
    glUniformMatrix4fv(get_uniform(geometryProgram_, "uViewProjection"),
                       1, GL_FALSE, viewProjection.m.data());

    for (std::size_t index = 0; index < scene.meshes.size(); ++index) {
        const Mesh& mesh = scene.meshes[index];
        const MeshGpu& gpu = meshes_[index];
        const Mat3 normal = normalMatrix(mesh.transform);
        glUniformMatrix4fv(get_uniform(geometryProgram_, "uModel"),
                           1, GL_FALSE, mesh.transform.m.data());
        glUniformMatrix3fv(get_uniform(geometryProgram_, "uNormalMatrix"),
                           1, GL_FALSE, normal.m.data());
        upload_material(scene, gpu);
        glBindVertexArray(gpu.vao);
        glDrawElements(GL_TRIANGLES, gpu.indexCount, GL_UNSIGNED_INT, nullptr);
    }
    glBindVertexArray(0);
    glUseProgram(0);
}

void Renderer::render_shadow_atlas(const Scene& scene) {
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFbo_);
    glViewport(0, 0, atlasWidth_, atlasHeight_);
    glDisable(GL_SCISSOR_TEST);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearDepth(1.0);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    glDisable(GL_BLEND);
    glUseProgram(shadowProgram_);

    for (const SlotPlan& slot : slots_) {
        glViewport(slot.column * tileResolution_, slot.row * tileResolution_,
                   tileResolution_, tileResolution_);
        glEnable(GL_SCISSOR_TEST);
        glScissor(slot.column * tileResolution_, slot.row * tileResolution_,
                  tileResolution_, tileResolution_);
        glUniformMatrix4fv(get_uniform(shadowProgram_, "uViewProjection"),
                           1, GL_FALSE, slot.viewProjection.m.data());
        for (std::size_t meshIndex = 0; meshIndex < scene.meshes.size(); ++meshIndex) {
            const Mesh& mesh = scene.meshes[meshIndex];
            const MeshGpu& gpu = meshes_[meshIndex];
            glUniformMatrix4fv(get_uniform(shadowProgram_, "uModel"),
                               1, GL_FALSE, mesh.transform.m.data());
            glBindVertexArray(gpu.vao);
            glDrawElements(GL_TRIANGLES, gpu.indexCount, GL_UNSIGNED_INT, nullptr);
        }
    }

    glDisable(GL_SCISSOR_TEST);
    glCullFace(GL_BACK);
    glBindVertexArray(0);
    glUseProgram(0);
}

void Renderer::render_lighting_pass(const Scene& scene) {
    glBindFramebuffer(GL_FRAMEBUFFER, lightFbo_);
    glViewport(0, 0, width_, height_);
    const GLenum buffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, buffers);
    const GLfloat clearColor[4] = {
        scene.settings.background.x, scene.settings.background.y,
        scene.settings.background.z, 0.0f,
    };
    const GLfloat clearVisibility[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    glClearBufferfv(GL_COLOR, 0, clearColor);
    glClearBufferfv(GL_COLOR, 1, clearVisibility);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);

    glUseProgram(lightingProgram_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gPosition_);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, gNormal_);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, gAlbedo_);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, gMaterial_);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, shadowAtlas_);
    glUniform1i(get_uniform(lightingProgram_, "uGPosition"), 0);
    glUniform1i(get_uniform(lightingProgram_, "uGNormal"), 1);
    glUniform1i(get_uniform(lightingProgram_, "uGAlbedo"), 2);
    glUniform1i(get_uniform(lightingProgram_, "uGMaterial"), 3);
    glUniform1i(get_uniform(lightingProgram_, "uShadowAtlas"), 4);
    glUniform3f(get_uniform(lightingProgram_, "uCameraPosition"),
                scene.camera.position.x, scene.camera.position.y,
                scene.camera.position.z);
    glUniform3f(get_uniform(lightingProgram_, "uBackground"),
                scene.settings.background.x, scene.settings.background.y,
                scene.settings.background.z);
    glUniform1i(get_uniform(lightingProgram_, "uAtlasColumns"), atlasColumns_);
    glUniform1i(get_uniform(lightingProgram_, "uAtlasRows"), atlasRows_);
    glUniform1i(get_uniform(lightingProgram_, "uAtlasTileResolution"),
                tileResolution_);
    glUniform1i(get_uniform(lightingProgram_, "uPcfKernel"), pcfKernel_);
    glUniform1i(get_uniform(lightingProgram_, "uShadowSamples"), shadowSamples_);
    glUniform1f(get_uniform(lightingProgram_, "uPcfStep"), pcfStep_);
    upload_lights(scene);
    draw_fullscreen();
    glUseProgram(0);
    glActiveTexture(GL_TEXTURE0);
}

void Renderer::render_denoise(const Scene&, unsigned int& finalVisibility) {
    finalVisibility = shadowVisibility_;
    if (!gaussianFilter_) {
        return;
    }

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glViewport(0, 0, width_, height_);
    glUseProgram(filterProgram_);
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(get_uniform(filterProgram_, "uVisibility"), 0);
    glUniform1i(get_uniform(filterProgram_, "uRadius"), filterRadius_);

    std::array<float, 17> weights{};
    for (std::size_t index = 0; index < filterWeights_.size(); ++index) {
        weights[index] = filterWeights_[index];
    }
    glUniform1fv(get_uniform(filterProgram_, "uWeights[0]"), 17, weights.data());

    unsigned int source = shadowVisibility_;
    for (int iteration = 0; iteration < postprocessIterations_; ++iteration) {
        const unsigned int destination =
            iteration % 2 == 0 ? filterScratchA_ : filterScratchB_;
        glBindFramebuffer(GL_FRAMEBUFFER, denoiseFbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, destination, 0);
        const GLenum buffer = GL_COLOR_ATTACHMENT0;
        glDrawBuffers(1, &buffer);
        glBindTexture(GL_TEXTURE_2D, source);
        draw_fullscreen();
        source = destination;
    }
    finalVisibility = source;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
}

void Renderer::render_composite_and_tonemap(const Scene& scene,
                                            unsigned int finalVisibility) {
    glBindFramebuffer(GL_FRAMEBUFFER, compositeFbo_);
    glViewport(0, 0, width_, height_);
    const GLenum compositeBuffer = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &compositeBuffer);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);

    glUseProgram(compositeProgram_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hdrColor_);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, shadowVisibility_);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, finalVisibility);
    glUniform1i(get_uniform(compositeProgram_, "uHdr"), 0);
    glUniform1i(get_uniform(compositeProgram_, "uVisibility"), 1);
    glUniform1i(get_uniform(compositeProgram_, "uFilteredVisibility"), 2);
    glUniform1i(get_uniform(compositeProgram_, "uUseFilter"), gaussianFilter_ ? 1 : 0);
    draw_fullscreen();

    glBindFramebuffer(GL_FRAMEBUFFER, tonemapFbo_);
    glViewport(0, 0, width_, height_);
    const GLenum tonemapBuffer = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &tonemapBuffer);
    glUseProgram(tonemapProgram_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hdrFinal_);
    glUniform1i(get_uniform(tonemapProgram_, "uHdr"), 0);
    glUniform1i(get_uniform(tonemapProgram_, "uToneMapping"),
                scene.settings.toneMapping == "aces" ? 0 : 1);
    glUniform1f(get_uniform(tonemapProgram_, "uExposure"),
                scene.settings.exposure);
    draw_fullscreen();

    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glActiveTexture(GL_TEXTURE0);
}

void Renderer::write_png(const std::string& outputPath) {
    std::vector<unsigned char> rgba(
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * 4u);
    glBindFramebuffer(GL_FRAMEBUFFER, tonemapFbo_);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    check_gl("image readback");

    std::vector<unsigned char> rgb(
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * 3u);
    for (int y = 0; y < height_; ++y) {
        const int sourceY = height_ - 1 - y;
        for (int x = 0; x < width_; ++x) {
            const std::size_t source =
                (static_cast<std::size_t>(sourceY) * static_cast<std::size_t>(width_)
                 + static_cast<std::size_t>(x)) * 4u;
            const std::size_t destination =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(width_)
                 + static_cast<std::size_t>(x)) * 3u;
            rgb[destination + 0] = rgba[source + 0];
            rgb[destination + 1] = rgba[source + 1];
            rgb[destination + 2] = rgba[source + 2];
        }
    }

    const std::filesystem::path path(outputPath);
    if (path.has_parent_path()) {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            fail("E_OUTPUT_IO", "cannot create output directory: " + error.message());
        }
    }
    std::ofstream stream(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!stream) {
        fail("E_OUTPUT_IO", "failed to open PNG output: " + outputPath);
    }
    const int written = stbi_write_png_to_func(
        png_write_callback, &stream, width_, height_, 3, rgb.data(), width_ * 3);
    stream.flush();
    if (written == 0 || !stream) {
        fail("E_OUTPUT_IO", "failed to write PNG: " + outputPath);
    }
}

void Renderer::log_light_plan(const Scene& scene, Logger& logger) const {
    for (std::size_t index = 0; index < lightPlans_.size(); ++index) {
        const LightPlan& plan = lightPlans_[index];
        std::string slotText;
        if (!plan.slots.empty()) {
            std::ostringstream slots;
            for (std::size_t slotIndex = 0; slotIndex < plan.slots.size(); ++slotIndex) {
                if (slotIndex != 0) {
                    slots << ',';
                }
                slots << plan.slots[slotIndex];
            }
            slotText = slots.str();
        }
        logger.info("light_atlas", {
            log_field("light_index", static_cast<int>(index)),
            log_field("light_type", type_name(scene.lights[index].type)),
            log_field("atlas_slot", plan.firstSlot >= 0
                                      ? std::to_string(plan.firstSlot)
                                      : std::string("none")),
            log_field("face_count", plan.faceCount),
            log_field("face_slots", slotText.empty() ? std::string("none") : slotText),
        });
    }
}

void Renderer::upload_lights(const Scene& scene) {
    const int count = static_cast<int>(lightPlans_.size());
    std::vector<int> types(static_cast<std::size_t>(count));
    std::vector<int> castShadow(static_cast<std::size_t>(count));
    std::vector<int> firstSlot(static_cast<std::size_t>(count));
    std::vector<int> faceCount(static_cast<std::size_t>(count));
    std::vector<float> positions(static_cast<std::size_t>(count) * 3u);
    std::vector<float> directions(static_cast<std::size_t>(count) * 3u);
    std::vector<float> colors(static_cast<std::size_t>(count) * 3u);
    std::vector<float> intensities(static_cast<std::size_t>(count));
    std::vector<float> ranges(static_cast<std::size_t>(count));
    std::vector<float> cones(static_cast<std::size_t>(count) * 2u);
    std::vector<float> biases(static_cast<std::size_t>(count));

    for (int index = 0; index < count; ++index) {
        const Light& light = scene.lights[static_cast<std::size_t>(index)];
        types[static_cast<std::size_t>(index)] = static_cast<int>(light.type);
        castShadow[static_cast<std::size_t>(index)] = light.castShadow ? 1 : 0;
        firstSlot[static_cast<std::size_t>(index)] = lightPlans_[static_cast<std::size_t>(index)].firstSlot;
        faceCount[static_cast<std::size_t>(index)] = lightPlans_[static_cast<std::size_t>(index)].faceCount;
        positions[static_cast<std::size_t>(index) * 3u + 0u] = light.position.x;
        positions[static_cast<std::size_t>(index) * 3u + 1u] = light.position.y;
        positions[static_cast<std::size_t>(index) * 3u + 2u] = light.position.z;
        directions[static_cast<std::size_t>(index) * 3u + 0u] = light.direction.x;
        directions[static_cast<std::size_t>(index) * 3u + 1u] = light.direction.y;
        directions[static_cast<std::size_t>(index) * 3u + 2u] = light.direction.z;
        colors[static_cast<std::size_t>(index) * 3u + 0u] = light.color.x;
        colors[static_cast<std::size_t>(index) * 3u + 1u] = light.color.y;
        colors[static_cast<std::size_t>(index) * 3u + 2u] = light.color.z;
        intensities[static_cast<std::size_t>(index)] = light.intensity;
        ranges[static_cast<std::size_t>(index)] = light.range;
        cones[static_cast<std::size_t>(index) * 2u + 0u] =
            std::cos(radians(light.innerConeDegrees));
        cones[static_cast<std::size_t>(index) * 2u + 1u] =
            std::cos(radians(light.outerConeDegrees));
        biases[static_cast<std::size_t>(index)] = light.shadowBias;
    }

    const int slotCount = static_cast<int>(slots_.size());
    std::vector<int> columns(static_cast<std::size_t>(slotCount));
    std::vector<int> rows(static_cast<std::size_t>(slotCount));
    std::vector<float> matrices(static_cast<std::size_t>(slotCount) * 16u);
    for (int index = 0; index < slotCount; ++index) {
        columns[static_cast<std::size_t>(index)] = slots_[static_cast<std::size_t>(index)].column;
        rows[static_cast<std::size_t>(index)] = slots_[static_cast<std::size_t>(index)].row;
        std::copy(slots_[static_cast<std::size_t>(index)].viewProjection.m.begin(),
                  slots_[static_cast<std::size_t>(index)].viewProjection.m.end(),
                  matrices.begin() + static_cast<std::ptrdiff_t>(index) * 16);
    }

    glUniform1i(get_uniform(lightingProgram_, "uLightCount"), count);
    glUniform1iv(get_uniform(lightingProgram_, "uLightType[0]"), count, types.data());
    glUniform1iv(get_uniform(lightingProgram_, "uLightCastShadow[0]"), count,
                 castShadow.data());
    glUniform1iv(get_uniform(lightingProgram_, "uLightFirstSlot[0]"), count,
                 firstSlot.data());
    glUniform1iv(get_uniform(lightingProgram_, "uLightFaceCount[0]"), count,
                 faceCount.data());
    glUniform3fv(get_uniform(lightingProgram_, "uLightPosition[0]"), count,
                 positions.data());
    glUniform3fv(get_uniform(lightingProgram_, "uLightDirection[0]"), count,
                 directions.data());
    glUniform3fv(get_uniform(lightingProgram_, "uLightColor[0]"), count,
                 colors.data());
    glUniform1fv(get_uniform(lightingProgram_, "uLightIntensity[0]"), count,
                 intensities.data());
    glUniform1fv(get_uniform(lightingProgram_, "uLightRange[0]"), count,
                 ranges.data());
    glUniform2fv(get_uniform(lightingProgram_, "uLightConeCos[0]"), count,
                 cones.data());
    glUniform1fv(get_uniform(lightingProgram_, "uLightShadowBias[0]"), count,
                 biases.data());
    glUniform1iv(get_uniform(lightingProgram_, "uSlotColumn[0]"), slotCount,
                 columns.data());
    glUniform1iv(get_uniform(lightingProgram_, "uSlotRow[0]"), slotCount,
                 rows.data());
    glUniformMatrix4fv(get_uniform(lightingProgram_, "uSlotViewProjection[0]"),
                       slotCount, GL_FALSE, matrices.data());
}

void Renderer::upload_material(const Scene& scene, const MeshGpu& mesh) {
    const Material& material =
        scene.materials[static_cast<std::size_t>(mesh.materialIndex)];
    glUniform3f(get_uniform(geometryProgram_, "uAlbedo"),
                material.albedo.x, material.albedo.y, material.albedo.z);
    glUniform1f(get_uniform(geometryProgram_, "uMetallic"), material.metallic);
    glUniform1f(get_uniform(geometryProgram_, "uRoughness"), material.roughness);
}

void Renderer::draw_fullscreen() const {
    glBindVertexArray(fullscreenVao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}

void Renderer::check_gl(const char* stage) const {
    GLenum error = glGetError();
    if (error == GL_NO_ERROR) {
        return;
    }
    std::string message = std::string("OpenGL error during ") + stage + ": "
                        + gl_error_hex(error);
    while ((error = glGetError()) != GL_NO_ERROR) {
        message += ", " + gl_error_hex(error);
    }
    fail("E_RENDER", message);
}

unsigned int Renderer::create_texture_2d(int width, int height,
                                         unsigned int internalFormat,
                                         unsigned int format, unsigned int type,
                                         unsigned int minFilter,
                                         unsigned int magFilter) {
    GLuint texture = 0;
    glGenTextures(1, &texture);
    if (texture == 0) {
        fail("E_GL_RESOURCE", "glGenTextures failed");
    }
    ++resourceCounts_.textures;
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(internalFormat),
                 width, height, 0, format, type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    static_cast<GLint>(minFilter));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    static_cast<GLint>(magFilter));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    if (format == GL_DEPTH_COMPONENT) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

unsigned int Renderer::create_framebuffer() {
    GLuint framebuffer = 0;
    glGenFramebuffers(1, &framebuffer);
    if (framebuffer == 0) {
        fail("E_GL_RESOURCE", "glGenFramebuffers failed");
    }
    ++resourceCounts_.framebuffers;
    return framebuffer;
}

unsigned int Renderer::create_renderbuffer() {
    GLuint renderbuffer = 0;
    glGenRenderbuffers(1, &renderbuffer);
    if (renderbuffer == 0) {
        fail("E_GL_RESOURCE", "glGenRenderbuffers failed");
    }
    ++resourceCounts_.renderbuffers;
    return renderbuffer;
}

unsigned int Renderer::create_buffer() {
    GLuint buffer = 0;
    glGenBuffers(1, &buffer);
    if (buffer == 0) {
        fail("E_GL_RESOURCE", "glGenBuffers failed");
    }
    ++resourceCounts_.buffers;
    return buffer;
}

unsigned int Renderer::create_vertex_array() {
    GLuint vertexArray = 0;
    glGenVertexArrays(1, &vertexArray);
    if (vertexArray == 0) {
        fail("E_GL_RESOURCE", "glGenVertexArrays failed");
    }
    ++resourceCounts_.vertexArrays;
    return vertexArray;
}

unsigned int Renderer::create_query() {
    GLuint query = 0;
    glGenQueries(1, &query);
    if (query == 0) {
        fail("E_GL_RESOURCE", "glGenQueries failed");
    }
    ++resourceCounts_.queries;
    return query;
}

unsigned int Renderer::create_program(const char* vertexSource,
                                      const char* fragmentSource,
                                      const char* label) {
    auto compile = [label](GLenum type, const char* source) {
        const GLuint shader = glCreateShader(type);
        if (shader == 0) {
            fail("E_GL_RESOURCE",
                 std::string("glCreateShader failed for ") + label);
        }
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);
        GLint success = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (success != GL_TRUE) {
            GLint length = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
            std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
            glGetShaderInfoLog(shader, length, nullptr, log.data());
            glDeleteShader(shader);
            fail("E_GL_RESOURCE",
                 std::string("shader compile failed for ") + label + ": " + log);
        }
        return shader;
    };

    const GLuint vertexShader = compile(GL_VERTEX_SHADER, vertexSource);
    const GLuint fragmentShader = compile(GL_FRAGMENT_SHADER, fragmentSource);
    const GLuint program = glCreateProgram();
    if (program == 0) {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        fail("E_GL_RESOURCE",
             std::string("glCreateProgram failed for ") + label);
    }
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    glDetachShader(program, vertexShader);
    glDetachShader(program, fragmentShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint success = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success != GL_TRUE) {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetProgramInfoLog(program, length, nullptr, log.data());
        glDeleteProgram(program);
        fail("E_GL_RESOURCE",
             std::string("shader link failed for ") + label + ": " + log);
    }
    ++resourceCounts_.shaderPrograms;
    return program;
}

void Renderer::delete_texture(unsigned int& texture) {
    if (texture != 0) {
        glDeleteTextures(1, &texture);
        texture = 0;
        --resourceCounts_.textures;
    }
}

void Renderer::delete_framebuffer(unsigned int& framebuffer) {
    if (framebuffer != 0) {
        glDeleteFramebuffers(1, &framebuffer);
        framebuffer = 0;
        --resourceCounts_.framebuffers;
    }
}

void Renderer::delete_renderbuffer(unsigned int& renderbuffer) {
    if (renderbuffer != 0) {
        glDeleteRenderbuffers(1, &renderbuffer);
        renderbuffer = 0;
        --resourceCounts_.renderbuffers;
    }
}

void Renderer::delete_buffer(unsigned int& buffer) {
    if (buffer != 0) {
        glDeleteBuffers(1, &buffer);
        buffer = 0;
        --resourceCounts_.buffers;
    }
}

void Renderer::delete_vertex_array(unsigned int& vertexArray) {
    if (vertexArray != 0) {
        glDeleteVertexArrays(1, &vertexArray);
        vertexArray = 0;
        --resourceCounts_.vertexArrays;
    }
}

void Renderer::delete_query(unsigned int& query) {
    if (query != 0) {
        glDeleteQueries(1, &query);
        query = 0;
        --resourceCounts_.queries;
    }
}

void Renderer::delete_program(unsigned int& program) {
    if (program != 0) {
        glDeleteProgram(program);
        program = 0;
        --resourceCounts_.shaderPrograms;
    }
}

}  // namespace renderer
