#include "renderer.h"

#include "math.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace cloud {
namespace {

bool intersectUnitBox(const Ray& ray, float& entry, float& exit) {
    float tMin = 0.0f;
    float tMax = std::numeric_limits<float>::infinity();
    for (int axis = 0; axis < 3; ++axis) {
        const float origin = axis == 0 ? ray.origin.x : (axis == 1 ? ray.origin.y : ray.origin.z);
        const float direction =
            axis == 0 ? ray.direction.x : (axis == 1 ? ray.direction.y : ray.direction.z);
        if (std::abs(direction) < 1.0e-8f) {
            if (origin < 0.0f || origin > 1.0f) {
                return false;
            }
            continue;
        }
        const float inverse = 1.0f / direction;
        float t0 = -origin * inverse;
        float t1 = (1.0f - origin) * inverse;
        if (t0 > t1) {
            std::swap(t0, t1);
        }
        tMin = std::max(tMin, t0);
        tMax = std::min(tMax, t1);
        if (tMin > tMax) {
            return false;
        }
    }
    entry = tMin;
    exit = tMax;
    return tMax > 0.0f;
}

bool insideUnitBox(const Vec3& point) {
    return point.x >= 0.0f && point.x <= 1.0f && point.y >= 0.0f && point.y <= 1.0f &&
           point.z >= 0.0f && point.z <= 1.0f;
}

float henyeyGreenstein(float cosine, float g) {
    const float g2 = g * g;
    const float denominator = 1.0f + g2 - 2.0f * g * cosine;
    if (denominator < 1.0e-5f) {
        return 1.0f;
    }
    return (1.0f - g2) / (4.0f * kPi * std::pow(denominator, 1.5f));
}

std::uint8_t floatToByte(float value) {
    const float clamped = clampf(value, 0.0f, 1.0f);
    return static_cast<std::uint8_t>(std::floor(clamped * 255.0f + 0.5f));
}

}  // namespace

Light defaultLight() {
    Light light;
    light.direction = normalize(light.direction);
    return light;
}

void RenderState::configure(int width, int height) {
    const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    accumulatedColor.resize(pixels, Vec3{});
    depth.assign(pixels, kSkyDepth);
    counts.assign(pixels, 0);
}

Renderer::Renderer(const RenderSettings& settings, const DensityField& density)
    : settings_(settings), density_(density) {
    if (settings_.width <= 0 || settings_.height <= 0) {
        throw std::invalid_argument("render target dimensions must be positive");
    }
    if (settings_.renderScale < 1 || settings_.renderScale > 4) {
        throw std::invalid_argument("renderScale must be in [1,4]");
    }
}

Vec3 Renderer::toneMap(const Vec3& color, float exposure) {
    Vec3 mapped = clamp(color * exposure, 0.0f, 1000.0f);
    mapped = Vec3{mapped.x / (1.0f + mapped.x), mapped.y / (1.0f + mapped.y),
                  mapped.z / (1.0f + mapped.z)};
    return {
        std::pow(clampf(mapped.x, 0.0f, 1.0f), 1.0f / 2.2f),
        std::pow(clampf(mapped.y, 0.0f, 1.0f), 1.0f / 2.2f),
        std::pow(clampf(mapped.z, 0.0f, 1.0f), 1.0f / 2.2f),
    };
}

Vec3 Renderer::sampleSky(const Ray& ray, const Light& light) const {
    const float horizon = clampf(ray.direction.y * 0.5f + 0.5f, 0.0f, 1.0f);
    const Vec3 low{0.075f, 0.105f, 0.16f};
    const Vec3 high{0.33f, 0.49f, 0.72f};
    Vec3 sky = lerp(low, high, smoothstep(0.1f, 0.95f, horizon));

    const float sunCosine = std::max(0.0f, dot(ray.direction, light.direction));
    sky += light.color * (0.35f * std::pow(sunCosine, 24.0f) + 0.12f * std::pow(sunCosine, 5.0f));
    sky += Vec3{light.ambient * 0.15f, light.ambient * 0.12f, light.ambient * 0.1f};
    return sky;
}

Renderer::ScatterResult Renderer::traceRay(const Ray& ray, const Light& light) const {
    ScatterResult result;
    float entry = 0.0f;
    float exit = 0.0f;
    if (!intersectUnitBox(ray, entry, exit)) {
        result.color = sampleSky(ray, light);
        return result;
    }

    const int steps = std::max(1, settings_.maxSteps);
    const float stepSize = (exit - entry) / static_cast<float>(steps);
    if (stepSize <= 0.0f) {
        result.color = sampleSky(ray, light);
        return result;
    }

    Vec3 transmittance{1.0f, 1.0f, 1.0f};
    Vec3 accumulated{};
    bool depthCaptured = false;
    const float cosine = dot(ray.direction, light.direction);
    const float phase = 0.22f + henyeyGreenstein(cosine, 0.38f) * 3.1f;

    for (int step = 0; step < steps; ++step) {
        const float t = entry + (static_cast<float>(step) + 0.5f) * stepSize;
        const Vec3 position = ray.origin + ray.direction * t;
        const float density = density_.sampleNormalized(position.x, position.y, position.z);
        if (density <= 1.0f / 255.0f) {
            continue;
        }
        if (!depthCaptured) {
            result.depth = t;
            depthCaptured = true;
        }

        float opticalDepth = 0.0f;
        const int lightSteps = std::max(1, settings_.lightSteps);
        const float lightStep = 0.38f / static_cast<float>(lightSteps);
        for (int lightStepIndex = 0; lightStepIndex < lightSteps; ++lightStepIndex) {
            const Vec3 lightPosition =
                position + light.direction * ((static_cast<float>(lightStepIndex) + 0.5f) * lightStep);
            if (insideUnitBox(lightPosition)) {
                opticalDepth += density_.sampleNormalized(lightPosition.x, lightPosition.y,
                                                          lightPosition.z) *
                                lightStep;
            }
        }

        const float sigma = density * settings_.densityExtinction;
        const float stepTransmittance = std::exp(-sigma * stepSize);
        const Vec3 radiance = light.color * (phase * std::exp(-opticalDepth *
                                                               settings_.lightExtinction)) +
                              Vec3{light.ambient * 0.55f, light.ambient * 0.62f,
                                   light.ambient * 0.75f};
        accumulated += transmittance * radiance * (1.0f - stepTransmittance);
        transmittance *= Vec3{stepTransmittance, stepTransmittance, stepTransmittance};
        if (transmittance.x + transmittance.y + transmittance.z < 0.012f) {
            break;
        }
    }

    result.color = accumulated + transmittance * sampleSky(ray, light);
    return result;
}

SampleFrame Renderer::renderLowResolution(const Camera& camera, const Light& light, int scale) const {
    SampleFrame low;
    low.width = (settings_.width + scale - 1) / scale;
    low.height = (settings_.height + scale - 1) / scale;
    low.renderScale = scale;
    low.color.resize(static_cast<std::size_t>(low.width) * low.height);
    low.depth.resize(low.color.size(), kSkyDepth);

    for (int y = 0; y < low.height; ++y) {
        for (int x = 0; x < low.width; ++x) {
            const float pixelX = static_cast<float>(x * scale) + static_cast<float>(scale - 1) * 0.5f;
            const float pixelY = static_cast<float>(y * scale) + static_cast<float>(scale - 1) * 0.5f;
            const Ray ray = camera.rayForPixel(pixelX, pixelY, settings_.width, settings_.height);
            const ScatterResult scatter = traceRay(ray, light);
            const std::size_t index = static_cast<std::size_t>(y * low.width + x);
            low.color[index] = scatter.color;
            low.depth[index] = scatter.depth;
        }
    }
    return low;
}

SampleFrame Renderer::upsampleDepthAware(const SampleFrame& low, int fullWidth, int fullHeight,
                                         int scale) const {
    SampleFrame full;
    full.width = fullWidth;
    full.height = fullHeight;
    full.renderScale = 1;
    full.color.resize(static_cast<std::size_t>(fullWidth) * fullHeight);
    full.depth.resize(full.color.size(), kSkyDepth);

    for (int y = 0; y < fullHeight; ++y) {
        for (int x = 0; x < fullWidth; ++x) {
            const float lowX = (static_cast<float>(x) + 0.5f) / static_cast<float>(scale) - 0.5f;
            const float lowY = (static_cast<float>(y) + 0.5f) / static_cast<float>(scale) - 0.5f;
            const int x0 = static_cast<int>(std::floor(lowX));
            const int y0 = static_cast<int>(std::floor(lowY));
            const float fx = lowX - static_cast<float>(x0);
            const float fy = lowY - static_cast<float>(y0);

            const auto indexAt = [&](int sx, int sy) -> std::size_t {
                sx = std::clamp(sx, 0, low.width - 1);
                sy = std::clamp(sy, 0, low.height - 1);
                return static_cast<std::size_t>(sy * low.width + sx);
            };
            const std::size_t indices[4] = {
                indexAt(x0, y0), indexAt(x0 + 1, y0), indexAt(x0, y0 + 1), indexAt(x0 + 1, y0 + 1)};
            const float spatialWeights[4] = {
                (1.0f - fx) * (1.0f - fy), fx * (1.0f - fy), (1.0f - fx) * fy, fx * fy};

            float foregroundDepth = std::numeric_limits<float>::infinity();
            for (const std::size_t index : indices) {
                foregroundDepth = std::min(foregroundDepth, low.depth[index]);
            }
            const bool foregroundIsSky = foregroundDepth >= kSkyDepth * 0.5f;
            const float depthTolerance =
                foregroundIsSky ? 1.0f : std::max(0.008f, foregroundDepth * 0.012f);

            float totalWeight = 0.0f;
            Vec3 color{};
            float depth = 0.0f;
            for (int candidate = 0; candidate < 4; ++candidate) {
                const std::size_t index = indices[candidate];
                if (!foregroundIsSky &&
                    low.depth[index] > foregroundDepth + depthTolerance) {
                    continue;
                }
                const float weight = spatialWeights[candidate];
                if (weight <= 0.0f) {
                    continue;
                }
                totalWeight += weight;
                color += low.color[index] * weight;
                depth += low.depth[index] * weight;
            }
            if (totalWeight <= 0.0f) {
                color = low.color[indices[0]];
                depth = low.depth[indices[0]];
            } else {
                color = color / totalWeight;
                depth /= totalWeight;
            }
            const std::size_t outputIndex = static_cast<std::size_t>(y * fullWidth + x);
            full.color[outputIndex] = color;
            full.depth[outputIndex] = depth;
        }
    }
    return full;
}

SampleFrame Renderer::renderSample(const Camera& camera, const Light& light, int scaleOverride) const {
    const int scale = scaleOverride > 0 ? scaleOverride : settings_.renderScale;
    if (scale == 1) {
        return renderLowResolution(camera, light, 1);
    }
    const SampleFrame low = renderLowResolution(camera, light, scale);
    return upsampleDepthAware(low, settings_.width, settings_.height, scale);
}

FrameStats Renderer::accumulate(RenderState& state, const SampleFrame& sample, const Camera& camera,
                                std::uint32_t frameIndex) {
    if (sample.width != settings_.width || sample.height != settings_.height) {
        throw std::invalid_argument("sample dimensions do not match renderer target");
    }
    if (state.accumulatedColor.size() != sample.color.size()) {
        state.configure(sample.width, sample.height);
    }

    FrameStats stats;
    stats.frameIndex = frameIndex;
    stats.reprojectionAttempts = static_cast<std::uint64_t>(sample.color.size());

    if (!state.hasHistory) {
        state.accumulatedColor = sample.color;
        state.depth = sample.depth;
        std::fill(state.counts.begin(), state.counts.end(), 1u);
        stats.reprojectionMisses = stats.reprojectionAttempts;
        stats.reprojectionHits = 0;
        stats.minimumCount = 1;
        stats.maximumCount = 1;
        stats.countsStrictlyMonotonic = true;
    } else {
        Camera previousCamera;
        previousCamera.view = state.previousView;
        previousCamera.projection = state.previousProjection;
        previousCamera.inverseViewProjection =
            (previousCamera.projection * previousCamera.view).inverse();
        previousCamera.position = state.previousCameraPosition;
        const Mat4 previousViewProjection = previousCamera.projection * previousCamera.view;
        std::vector<Vec3> nextColor(sample.color.size());
        std::vector<std::uint32_t> nextCounts(sample.color.size());
        std::uint32_t minimumCount = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maximumCount = 0;
        bool monotonic = true;
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;

        for (int y = 0; y < sample.height; ++y) {
            for (int x = 0; x < sample.width; ++x) {
                const std::size_t pixel = static_cast<std::size_t>(y * sample.width + x);
                const Ray ray = camera.rayForPixel(static_cast<float>(x), static_cast<float>(y),
                                                   sample.width, sample.height);
                const float currentDepth = sample.depth[pixel];
                const bool currentSky = currentDepth >= kSkyDepth * 0.5f;
                const Vec3 currentWorld =
                    camera.position + ray.direction * (currentSky ? 100.0f : currentDepth);

                const Vec4 clip = previousViewProjection * Vec4{currentWorld.x, currentWorld.y,
                                                                currentWorld.z, 1.0f};
                bool hit = false;
                std::size_t sourcePixel = 0;
                if (clip.w > 1.0e-8f) {
                    const float ndcX = clip.x / clip.w;
                    const float ndcY = clip.y / clip.w;
                    const float uvX = ndcX * 0.5f + 0.5f;
                    const float uvY = 0.5f - ndcY * 0.5f;
                    const int sourceX = static_cast<int>(std::floor(uvX * static_cast<float>(sample.width)));
                    const int sourceY = static_cast<int>(std::floor(uvY * static_cast<float>(sample.height)));
                    if (sourceX >= 0 && sourceX < sample.width && sourceY >= 0 &&
                        sourceY < sample.height) {
                        sourcePixel = static_cast<std::size_t>(sourceY * sample.width + sourceX);
                        const bool previousSky = state.depth[sourcePixel] >= kSkyDepth * 0.5f;
                        if (currentSky == previousSky) {
                            if (currentSky) {
                                hit = true;
                            } else {
                                const Ray previousRay =
                                    previousCamera.rayForPixel(static_cast<float>(sourceX),
                                                               static_cast<float>(sourceY),
                                                               sample.width, sample.height);
                                const Vec3 previousWorld =
                                    previousCamera.position +
                                    previousRay.direction * state.depth[sourcePixel];
                                const float tolerance =
                                    0.025f + 0.018f * std::min(currentDepth, state.depth[sourcePixel]);
                                hit = length(currentWorld - previousWorld) <= tolerance;
                            }
                        }
                    }
                }

                const std::uint32_t previousCount = state.counts[pixel];
                const std::uint32_t newCount =
                    previousCount == std::numeric_limits<std::uint32_t>::max()
                        ? previousCount
                        : previousCount + 1u;
                if (newCount <= previousCount) {
                    monotonic = false;
                }
                nextCounts[pixel] = newCount;
                minimumCount = std::min(minimumCount, newCount);
                maximumCount = std::max(maximumCount, newCount);

                if (hit) {
                    ++hits;
                    const std::uint32_t historySamples =
                        std::min(state.counts[sourcePixel],
                                 static_cast<std::uint32_t>(std::max(0, settings_.maxHistoryLength)));
                    const float historyWeight = static_cast<float>(historySamples);
                    nextColor[pixel] =
                        (state.accumulatedColor[sourcePixel] * historyWeight + sample.color[pixel]) /
                        (historyWeight + 1.0f);
                } else {
                    ++misses;
                    nextColor[pixel] = sample.color[pixel];
                }
            }
        }

        state.accumulatedColor = std::move(nextColor);
        state.counts = std::move(nextCounts);
        state.depth = sample.depth;
        stats.reprojectionHits = hits;
        stats.reprojectionMisses = misses;
        stats.minimumCount = minimumCount;
        stats.maximumCount = maximumCount;
        stats.countsStrictlyMonotonic = monotonic;
    }

    state.hasHistory = true;
    state.frameIndex = frameIndex;
    state.previousView = camera.view;
    state.previousProjection = camera.projection;
    state.previousCameraPosition =
        camera.view.inverse().transformPoint(Vec3{0.0f, 0.0f, 0.0f});
    stats.reprojectionHitRate =
        stats.reprojectionAttempts == 0
            ? 0.0
            : static_cast<double>(stats.reprojectionHits) /
                  static_cast<double>(stats.reprojectionAttempts);
    return stats;
}

std::vector<std::uint8_t> Renderer::encodeRGBA(const std::vector<Vec3>& hdrColor) const {
    std::vector<std::uint8_t> rgba(hdrColor.size() * 4);
    for (std::size_t pixel = 0; pixel < hdrColor.size(); ++pixel) {
        const Vec3 mapped = toneMap(hdrColor[pixel], settings_.exposure);
        rgba[pixel * 4 + 0] = floatToByte(mapped.x);
        rgba[pixel * 4 + 1] = floatToByte(mapped.y);
        rgba[pixel * 4 + 2] = floatToByte(mapped.z);
        rgba[pixel * 4 + 3] = 255;
    }
    return rgba;
}

double Renderer::psnrRGB8(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
    if (a.size() != b.size() || a.size() % 4 != 0) {
        throw std::invalid_argument("PSNR inputs must be equally sized RGBA8 buffers");
    }
    long double squaredError = 0.0L;
    std::uint64_t componentCount = 0;
    for (std::size_t i = 0; i < a.size(); i += 4) {
        for (int channel = 0; channel < 3; ++channel) {
            const long double delta = static_cast<long double>(a[i + channel]) -
                                      static_cast<long double>(b[i + channel]);
            squaredError += delta * delta;
            ++componentCount;
        }
    }
    if (componentCount == 0) {
        return std::numeric_limits<double>::infinity();
    }
    const long double mse = squaredError / static_cast<long double>(componentCount);
    if (mse <= 1.0e-30L) {
        return std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(10.0L * std::log10((255.0L * 255.0L) / mse));
}

}  // namespace cloud

