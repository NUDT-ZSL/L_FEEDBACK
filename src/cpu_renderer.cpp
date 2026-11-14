#include "renderer.hpp"

#include "scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cr {
namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kBackgroundTop = 0.035F;
constexpr float kBackgroundBottom = 0.010F;

struct GBufferPixel {
    Vec3 baseColor;
    float metallic = 0.0F;
    float roughness = 0.0F;
    Vec3 worldNormal;
    float depth = 0.0F;
    bool hit = false;
};

struct GBuffer {
    int width = 0;
    int height = 0;
    Camera camera;
    CameraFrame cameraFrame;
    std::vector<GBufferPixel> pixels;
};

float distributionGGX(float normalDotHalf, float roughness) {
    const float alpha = roughness * roughness;
    const float alphaSquared = alpha * alpha;
    const float denominator =
        normalDotHalf * normalDotHalf * (alphaSquared - 1.0F) + 1.0F;
    return alphaSquared / std::max(kPi * denominator * denominator, 1.0e-6F);
}

float geometrySchlickGGX(float normalDotView, float roughness) {
    const float a = roughness;
    const float k = (a * a) / 8.0F;
    return normalDotView / std::max(normalDotView * (1.0F - k) + k, 1.0e-5F);
}

float geometrySmith(float normalDotView, float normalDotLight, float roughness) {
    return geometrySchlickGGX(normalDotView, roughness) *
           geometrySchlickGGX(normalDotLight, roughness);
}

Vec3 fresnelSchlick(float cosTheta, const Vec3& f0) {
    const float factor = std::pow(clamp(1.0F - cosTheta, 0.0F, 1.0F), 5.0F);
    return f0 + (Vec3(1.0F, 1.0F, 1.0F) - f0) * factor;
}

float lightAttenuation(const Vec3& lightToSurface, float radius) {
    const float squaredDistance = std::max(lengthSquared(lightToSurface), 1.0e-5F);
    const float normalized = std::sqrt(squaredDistance) / radius;
    if (normalized >= 1.0F) {
        return 0.0F;
    }
    const float window = 1.0F - std::pow(normalized, 4.0F);
    return window * window / (squaredDistance + 0.04F);
}

Vec3 shadeSurface(const GBufferPixel& pixel, const Vec3& worldPosition, const Vec3& view,
                  const std::vector<Light>& lights) {
    const Vec3 normal = normalize(pixel.worldNormal);
    const Vec3 baseColor = clamp(pixel.baseColor, 0.0F, 1.0F);
    const float metallic = clamp(pixel.metallic, 0.0F, 1.0F);
    const float roughness = clamp(pixel.roughness, 0.04F, 1.0F);
    const Vec3 f0 = mix(Vec3(0.040F, 0.040F, 0.040F), baseColor, metallic);
    Vec3 radianceOut = baseColor * (0.022F + 0.018F * clamp(normal.y, 0.0F, 1.0F));

    for (const Light& light : lights) {
        const Vec3 toLight = light.position - worldPosition;
        const float lightDistance = length(toLight);
        const Vec3 lightDirection = toLight / std::max(lightDistance, 1.0e-5F);
        const float normalDotLight = dot(normal, lightDirection);
        if (normalDotLight <= 0.0F) {
            continue;
        }
        const float attenuation = lightAttenuation(toLight, light.radius);
        if (attenuation <= 0.0F || light.intensity <= 0.0F) {
            continue;
        }
        const Vec3 halfDirection = normalize(lightDirection + view);
        const float normalDotHalf = clamp(dot(normal, halfDirection), 0.0F, 1.0F);
        const float viewDotHalf = clamp(dot(view, halfDirection), 0.0F, 1.0F);
        const float normalDotView = clamp(dot(normal, view), 0.0F, 1.0F);

        const float distribution = distributionGGX(normalDotHalf, roughness);
        const float geometry = geometrySmith(normalDotView, normalDotLight, roughness);
        const Vec3 fresnel = fresnelSchlick(viewDotHalf, f0);
        const Vec3 specular =
            fresnel * (distribution * geometry /
                       std::max(4.0F * normalDotView * normalDotLight, 1.0e-4F));
        const Vec3 diffuse =
            baseColor *
            ((Vec3(1.0F, 1.0F, 1.0F) - fresnel) * ((1.0F - metallic) / kPi));
        const Vec3 incoming = light.color * (light.intensity * attenuation);
        radianceOut += (diffuse + specular) * incoming * normalDotLight;
    }
    return radianceOut;
}

float toneMapChannel(float value) {
    return value / (1.0F + value);
}

Vec3 toneMap(const Vec3& value) {
    const Vec3 mapped{toneMapChannel(value.x), toneMapChannel(value.y),
                      toneMapChannel(value.z)};
    return Vec3(std::pow(clamp(mapped.x, 0.0F, 1.0F), 1.0F / 2.2F),
                std::pow(clamp(mapped.y, 0.0F, 1.0F), 1.0F / 2.2F),
                std::pow(clamp(mapped.z, 0.0F, 1.0F), 1.0F / 2.2F));
}

std::uint8_t toByte(float value) {
    return static_cast<std::uint8_t>(
        std::lround(clamp(value, 0.0F, 1.0F) * 255.0F));
}

GBuffer buildGBuffer(const MaterialLibrary& materials, const Camera& camera) {
    GBuffer gbuffer;
    gbuffer.width = camera.width;
    gbuffer.height = camera.height;
    gbuffer.camera = camera;
    gbuffer.cameraFrame = makeCameraFrame(camera);
    gbuffer.pixels.resize(static_cast<std::size_t>(camera.width) *
                          static_cast<std::size_t>(camera.height));

    for (int y = 0; y < camera.height; ++y) {
        for (int x = 0; x < camera.width; ++x) {
            const Vec3 direction =
                rayDirection(gbuffer.cameraFrame, x, y, camera.width, camera.height);
            MaterialZone zone = MaterialZone::Skin;
            const float hitDistance =
                traceScene(camera.position, direction, camera.farPlane, &zone);
            GBufferPixel& pixel =
                gbuffer.pixels[static_cast<std::size_t>(y) *
                                   static_cast<std::size_t>(camera.width) +
                               static_cast<std::size_t>(x)];
            if (hitDistance < 0.0F) {
                continue;
            }
            const Vec3 worldPosition = camera.position + direction * hitDistance;
            const Vec3 geometricNormal = sceneNormal(worldPosition);
            const SurfaceMapping mapping = surfaceMapping(worldPosition, geometricNormal, zone);
            const LoadedMaterial& material = materials.require(mapping.materialName);
            const MaterialSurface surface = composeMaterial(material, mapping.u, mapping.v);
            pixel.baseColor = surface.baseColor;
            pixel.metallic = surface.metallic;
            pixel.roughness = surface.roughness;
            pixel.worldNormal =
                worldNormalFromTangent(mapping, surface.tangentNormal);
            pixel.depth = dot(worldPosition - camera.position, gbuffer.cameraFrame.forward);
            pixel.hit = true;
        }
    }
    return gbuffer;
}

Image shadeGBuffer(const GBuffer& gbuffer, const std::vector<Light>& lights) {
    Image output(gbuffer.width, gbuffer.height, 4);
    for (int y = 0; y < gbuffer.height; ++y) {
        const float backgroundMix =
            static_cast<float>(y) / static_cast<float>(std::max(gbuffer.height - 1, 1));
        const Vec3 background =
            Vec3(0.020F, 0.026F, 0.040F) * (0.60F + backgroundMix * 0.65F) +
            Vec3(kBackgroundTop, 0.0F, kBackgroundBottom) * (1.0F - backgroundMix);
        for (int x = 0; x < gbuffer.width; ++x) {
            const GBufferPixel& pixel =
                gbuffer.pixels[static_cast<std::size_t>(y) *
                                   static_cast<std::size_t>(gbuffer.width) +
                               static_cast<std::size_t>(x)];
            Vec3 color = background;
            if (pixel.hit) {
                const Vec3 direction = rayDirection(gbuffer.cameraFrame, x, y, gbuffer.width,
                                                    gbuffer.height);
                const Vec3 worldPosition = positionFromLinearDepth(
                    gbuffer.cameraFrame, direction, pixel.depth, gbuffer.camera);
                const Vec3 view = normalize(gbuffer.camera.position - worldPosition);
                color = toneMap(shadeSurface(pixel, worldPosition, view, lights));
            }
            std::uint8_t* out =
                output.pixels.data() +
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(gbuffer.width) +
                 static_cast<std::size_t>(x)) *
                    4U;
            out[0] = toByte(color.x);
            out[1] = toByte(color.y);
            out[2] = toByte(color.z);
            out[3] = 255U;
        }
    }
    return output;
}

}  // namespace

RenderBackend parseRenderBackend(const std::string& value) {
    if (value == "auto") {
        return RenderBackend::Auto;
    }
    if (value == "cpu") {
        return RenderBackend::Cpu;
    }
    if (value == "opengl") {
        return RenderBackend::OpenGL;
    }
    throw AppError(E_DATA_INVALID, "unknown render backend '" + value + "'");
}

const char* renderBackendName(RenderBackend backend) {
    switch (backend) {
        case RenderBackend::Auto:
            return "auto";
        case RenderBackend::Cpu:
            return "cpu";
        case RenderBackend::OpenGL:
            return "opengl";
    }
    return "auto";
}

Image renderCpuDeferred(const MaterialLibrary& materials, const Camera& camera,
                        const std::vector<Light>& lights) {
    GBuffer gbuffer = buildGBuffer(materials, camera);
    return shadeGBuffer(gbuffer, lights);
}

}  // namespace cr

