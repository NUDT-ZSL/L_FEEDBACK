#include "renderer_cpu.h"

#include "noise.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>

namespace df {
namespace {

Vec3 lightRadiance(const Light& light, const Vec3& position, const Vec3& normal,
                   const Vec3& viewDir, const Material& material) {
    Vec3 lightDir;
    float attenuation = 1.0f;
    if (light.type == LightType::Directional) {
        lightDir = normalize(-light.direction);
    } else {
        const Vec3 delta = light.position - position;
        const float distance = length(delta);
        if (distance < 1.0e-5f) return {};
        lightDir = delta / distance;
        attenuation = 1.0f / (1.0f + 0.08f * distance + 0.055f * distance * distance);
    }
    const float ndotl = std::max(dot(normal, lightDir), 0.0f);
    if (ndotl <= 0.0f) return {};

    const Vec3 halfDir = normalize(lightDir + viewDir);
    const float ndoth = std::max(dot(normal, halfDir), 0.0f);
    const float shininess = 2.0f + 126.0f * (1.0f - material.roughness) *
                                      (1.0f - material.roughness);
    const float specPower = std::pow(ndoth, shininess);
    const Vec3 f0 = mix(Vec3{0.04f, 0.04f, 0.04f}, material.albedo, material.metallic);
    const float specStrength = (0.15f + 0.85f * (1.0f - material.roughness)) * (1.0f - material.roughness);
    const Vec3 diffuse = material.albedo * ((1.0f - material.metallic) / kPi);
    const Vec3 specular = f0 * (specPower * specStrength * (shininess + 2.0f) / (8.0f * kPi));
    return (diffuse * ndotl + specular * ndotl) * light.color * (light.intensity * attenuation);
}

Vec3 shadeOpaque(const Scene& scene, const Hit& hit, const Vec3& rayDirection,
                 const Light& light) {
    (void)scene;
    return lightRadiance(light, hit.position, hit.normal, -rayDirection, hit.primitive->material);
}

Vec3 shadeSurface(const Scene& scene, const Hit& hit, const Vec3& rayDirection,
                  int maxLights) {
    Vec3 color{0.028f, 0.035f, 0.052f};
    color += hit.primitive->material.albedo * 0.045f;
    const int count = std::min(static_cast<int>(scene.lights.size()), maxLights);
    for (int i = 0; i < count; ++i) {
        color += shadeOpaque(scene, hit, rayDirection, scene.lights[static_cast<std::size_t>(i)]);
    }
    return color;
}

Vec3 skyColor(const Vec3& direction) {
    const float t = std::clamp(direction.y * 0.5f + 0.5f, 0.0f, 1.0f);
    return mix(Vec3{0.018f, 0.025f, 0.045f}, Vec3{0.07f, 0.12f, 0.21f}, t);
}

Vec3 fogInscatter(const Scene& scene, const Vec3& position, const Vec3& rayDirection,
                  int maxLights) {
    Vec3 result{0.0f};
    const int count = std::min(static_cast<int>(scene.lights.size()), maxLights);
    for (int i = 0; i < count; ++i) {
        const Light& light = scene.lights[static_cast<std::size_t>(i)];
        if (light.type == LightType::Directional) {
            const float phase = 0.45f + 0.25f * std::max(dot(rayDirection, -light.direction), 0.0f);
            result += light.color * (light.intensity * 0.085f * phase);
        } else {
            const Vec3 delta = light.position - position;
            const float distanceSq = std::max(dot(delta, delta), 0.05f);
            const Vec3 dir = normalize(delta);
            const float phase = 0.65f + 0.35f * std::max(dot(rayDirection, dir), 0.0f);
            result += light.color * (light.intensity * 0.42f * phase / (1.0f + distanceSq));
        }
    }
    return result;
}

std::uint8_t toSrgb8(float linear) {
    const float mapped = linear / (1.0f + linear);
    const float gamma = std::pow(std::clamp(mapped, 0.0f, 1.0f), 1.0f / 2.2f);
    return static_cast<std::uint8_t>(std::lround(gamma * 255.0f));
}

} // namespace

FrameOutput CpuRenderer::render(const Scene& scene, const RenderConfig& config, bool offline,
                                bool collectGBuffer) {
    const auto start = std::chrono::steady_clock::now();
    const int width = config.width;
    const int height = config.height;
    const int steps = effectiveFogSteps(config, offline);
    const int maxLights = config.lights.maxLights;

    lightSnapshot_.sync(scene.lights);
    ensureLightingUpload(lightSnapshot_, uploadStats_);
    ensureFogUpload(lightSnapshot_, uploadStats_);
    lightSnapshot_.clearDirty();

    FrameOutput output;
    output.width = width;
    output.height = height;
    output.camera = scene.camera;
    output.colorRgb.resize(static_cast<std::size_t>(width) * height * 3u);
    output.linearDepth.assign(static_cast<std::size_t>(width) * height, scene.camera.farPlane);
    output.transparent = sortTransparent(scene, config.transparent, scene.camera);

    if (collectGBuffer) {
        const std::size_t pixelCount = static_cast<std::size_t>(width) * height;
        output.gbuffer.width = width;
        output.gbuffer.height = height;
        output.gbuffer.gPosition.assign(pixelCount, Vec4{});
        output.gbuffer.gNormal.assign(pixelCount, Vec4{});
        output.gbuffer.gAlbedo.assign(pixelCount, Vec4{});
        output.gbuffer.gMaterial.assign(pixelCount, Vec4{});
    }

    std::unordered_map<int, const Primitive*> transparentById;
    transparentById.reserve(scene.transparent.size() * 2);
    for (const Primitive& primitive : scene.transparent) {
        transparentById[primitive.id] = &primitive;
    }

    std::vector<bool> transparentHit(output.transparent.objects.size(), false);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            const Ray ray = cameraRay(scene.camera, width, height, x, y);
            Hit opaqueHit = intersectOpaque(scene, ray, scene.camera.farPlane);
            Vec3 color;
            float viewZ = scene.camera.farPlane;
            if (opaqueHit.hit) {
                color = shadeSurface(scene, opaqueHit, ray.direction, maxLights);
                viewZ = viewDepth(scene.camera, opaqueHit.position);
                output.linearDepth[pixel] = std::max(0.0f, viewZ);
                if (collectGBuffer) {
                    const auto& material = opaqueHit.primitive->material;
                    output.gbuffer.gPosition[pixel] =
                        {opaqueHit.position.x, opaqueHit.position.y, opaqueHit.position.z, viewZ};
                    output.gbuffer.gNormal[pixel] =
                        {opaqueHit.normal.x, opaqueHit.normal.y, opaqueHit.normal.z,
                         static_cast<float>(opaqueHit.primitive->id)};
                    output.gbuffer.gAlbedo[pixel] =
                        {material.albedo.x, material.albedo.y, material.albedo.z, material.roughness};
                    output.gbuffer.gMaterial[pixel] =
                        {material.metallic, 0.0f, 0.0f, 1.0f};
                }
            } else {
                color = skyColor(ray.direction);
            }

            if (config.volumetric.enabled && steps > 0) {
                const float rayLimit = std::min(
                    opaqueHit.hit ? opaqueHit.t : scene.camera.farPlane,
                    config.volumetric.maxDistance);
                const float dt = rayLimit / static_cast<float>(steps);
                float transmittance = 1.0f;
                Vec3 fogColor{0.0f};
                for (int step = 0; step < steps; ++step) {
                    const float t = (static_cast<float>(step) + 0.5f) * dt;
                    const Vec3 sample = ray.origin + ray.direction * t;
                    const float density =
                        densityAt(sample, config.volumetric.densityScale,
                                  config.volumetric.noiseScale);
                    if (density <= 0.0f) continue;
                    const float sigma =
                        density * (config.volumetric.absorption + config.volumetric.scattering);
                    const float sampleTransmittance = std::exp(-sigma * dt);
                    const float integrated = (1.0f - sampleTransmittance) /
                                             std::max(sigma, 1.0e-6f);
                    fogColor += fogInscatter(scene, sample, ray.direction, maxLights) *
                                (transmittance * density * config.volumetric.scattering *
                                 integrated);
                    transmittance *= sampleTransmittance;
                    if (transmittance < 0.002f) break;
                }
                color = fogColor + color * transmittance;
            }

            for (std::size_t sortedIndex = 0;
                 sortedIndex < output.transparent.objects.size(); ++sortedIndex) {
                const SortedTransparent& sorted = output.transparent.objects[sortedIndex];
                const auto found = transparentById.find(sorted.id);
                if (found == transparentById.end()) continue;
                const Primitive& primitive = *found->second;
                Hit hit = intersectPrimitive(primitive, ray, scene.camera.farPlane);
                if (!hit.hit || (opaqueHit.hit && hit.t >= opaqueHit.t)) continue;
                Vec3 transparentColor =
                    shadeSurface(scene, hit, ray.direction, maxLights) * primitive.tint;
                color = transparentColor * primitive.opacity +
                        color * (1.0f - primitive.opacity);
                transparentHit[sortedIndex] = true;
            }

            output.colorRgb[pixel * 3u + 0u] = toSrgb8(color.x);
            output.colorRgb[pixel * 3u + 1u] = toSrgb8(color.y);
            output.colorRgb[pixel * 3u + 2u] = toSrgb8(color.z);
        }
    }

    output.stats.backend = "cpu-reference";
    output.stats.fogSteps = steps;
    output.stats.activeLights = std::min(static_cast<int>(scene.lights.size()), maxLights);
    output.stats.transparentDrawn = static_cast<int>(
        std::count(transparentHit.begin(), transparentHit.end(), true));
    output.stats.transparentDepthWrites = 0;
    output.stats.lightVersion = lightSnapshot_.version();
    output.stats.lightingUploads = uploadStats_.lightingUploads;
    output.stats.fogUploads = uploadStats_.fogUploads;
    const auto end = std::chrono::steady_clock::now();
    output.stats.cpuMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
    return output;
}

} // namespace df
