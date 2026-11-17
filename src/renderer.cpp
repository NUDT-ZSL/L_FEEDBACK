#include "renderer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <tuple>

namespace cv {
namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

float srgbEncode(float linear) {
    const float c = std::max(0.0f, linear);
    if (c <= 0.0031308f) {
        return 12.92f * c;
    }
    return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

uint8_t quantize(float linear) {
    const float tonemapped = linear / (1.0f + std::max(0.0f, linear));
    return static_cast<uint8_t>(
        std::clamp(static_cast<int>(std::lround(srgbEncode(tonemapped) * 255.0f)), 0, 255));
}

int wrapCoordinate(int value, int size) {
    int result = value % size;
    return result < 0 ? result + size : result;
}

Vec3 sampleTexel(const NormalTexture& texture, int x, int y) {
    const int wrappedX = wrapCoordinate(x, texture.width);
    const int wrappedY = wrapCoordinate(y, texture.height);
    const size_t offset = static_cast<size_t>(
        (wrappedY * texture.width + wrappedX) * 3);
    return {texture.rgb[offset + 0] / 255.0f, texture.rgb[offset + 1] / 255.0f,
            texture.rgb[offset + 2] / 255.0f};
}

Vec3 smoothstepAmbient(Vec3 ground, Vec3 sky, float t) {
    const float s = clamp01(t);
    return ground + (sky - ground) * s;
}

}  // namespace

Renderer::Renderer(const Scene& scene, const Lighting& lighting,
                   const RendererConfig& config)
    : config_(config), grid_(1.25f), normalMap_(scene.normalMap) {
    config_.width = std::clamp(config_.width, 1, 4096);
    config_.height = std::clamp(config_.height, 1, 4096);
    config_.cascadeResolution = std::clamp(config_.cascadeResolution, 64, 2048);
    config_.pointShadowResolution = std::clamp(config_.pointShadowResolution, 32, 1024);
    config_.pcfRadius = std::clamp(config_.pcfRadius, 0, 2);

    raster_.resize(config_.width, config_.height);
    for (DepthRasterizer& cascade : cascades_) {
        cascade.resize(config_.cascadeResolution, config_.cascadeResolution);
    }
    for (PointShadow& cube : pointShadowViews_) {
        for (DepthRasterizer& face : cube.faces) {
            face.resize(config_.pointShadowResolution, config_.pointShadowResolution);
        }
    }
    grid_.build(scene);
    configureShadowMatrices(lighting);
    image_.width = config_.width;
    image_.height = config_.height;
    image_.rgb.resize(static_cast<size_t>(config_.width) * config_.height * 3, 0);
}

void Renderer::resize(int width, int height) {
    const int newWidth = std::clamp(width, 1, 4096);
    const int newHeight = std::clamp(height, 1, 4096);
    if (newWidth == config_.width && newHeight == config_.height) {
        return;
    }
    config_.width = newWidth;
    config_.height = newHeight;
    raster_.resize(config_.width, config_.height);
    image_.width = config_.width;
    image_.height = config_.height;
    image_.rgb.assign(static_cast<size_t>(config_.width) * config_.height * 3, 0);
}

void Renderer::configureShadowMatrices(const Lighting& lighting) {
    const Vec3 direction = normalize(lighting.directionalDirection);
    const Vec3 up = std::abs(dot(direction, {0.0f, 1.0f, 0.0f})) > 0.95f
                        ? Vec3{0.0f, 0.0f, 1.0f}
                        : Vec3{0.0f, 1.0f, 0.0f};
    const Vec3 center{0.0f, 1.05f, 0.0f};
    const std::array<float, 2> extents{4.3f, 9.0f};
    for (size_t i = 0; i < cascadeViewProjection_.size(); ++i) {
        const Vec3 eye = center - direction * 16.0f;
        const Mat4 view = Mat4::lookAtLH(eye, center, up);
        const Mat4 projection =
            Mat4::orthographicD3D(-extents[i], extents[i], -extents[i], extents[i],
                                  0.1f, 35.0f);
        cascadeViewProjection_[i] = projection * view;
    }

    const std::array<Vec3, 6> faceDirections{{
        {1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
        {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f},
    }};
    const std::array<Vec3, 6> faceUps{{
        {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
    }};
    for (size_t lightIndex = 0; lightIndex < pointShadowViews_.size(); ++lightIndex) {
        const Vec3 eye = lighting.pointLights[lightIndex].position;
        for (size_t face = 0; face < 6; ++face) {
            const Mat4 view =
                Mat4::lookAtLH(eye, eye + faceDirections[face], faceUps[face]);
            const Mat4 projection = Mat4::perspectiveD3D(kPi * 0.5f, 1.0f, 0.05f, 20.0f);
            pointShadowViews_[lightIndex].viewProjection[face] = projection * view;
        }
    }
}

void Renderer::renderShadows(
    const Scene& scene, const std::vector<std::vector<FrameVertex>>& frameVertices,
    const Lighting&) {
    stats_.shadowMaps = 0;
    stats_.shadowTriangles = 0;
    if (!config_.shadows) {
        return;
    }

    auto drawTo = [&](DepthRasterizer& map, const Mat4& viewProjection) {
        map.clear();
        for (size_t meshIndex = 0; meshIndex < scene.meshes.size(); ++meshIndex) {
            const Mesh& mesh = scene.meshes[meshIndex];
            const std::vector<FrameVertex>& verts = frameVertices[meshIndex];
            const size_t triangleCount = mesh.indices.size() / 3;
            for (size_t tri = 0; tri < triangleCount; ++tri) {
                const Material& material =
                    scene.materials[mesh.triangleMaterial[tri]];
                if (material.transparent || !material.castsShadow) {
                    continue;
                }
                const size_t base = tri * 3;
                const Vec3 p0 = verts[mesh.indices[base + 0]].worldPosition;
                const Vec3 p1 = verts[mesh.indices[base + 1]].worldPosition;
                const Vec3 p2 = verts[mesh.indices[base + 2]].worldPosition;
                map.drawTriangle(viewProjection, p0, p1, p2, CullMode::Back);
                ++stats_.shadowTriangles;
            }
        }
        ++stats_.shadowMaps;
    };

    for (size_t cascade = 0; cascade < cascades_.size(); ++cascade) {
        drawTo(cascades_[cascade], cascadeViewProjection_[cascade]);
    }
    for (size_t lightIndex = 0; lightIndex < pointShadowViews_.size(); ++lightIndex) {
        for (size_t face = 0; face < 6; ++face) {
            drawTo(pointShadowViews_[lightIndex].faces[face],
                   pointShadowViews_[lightIndex].viewProjection[face]);
        }
    }
}

void Renderer::buildCommands(
    const Scene& scene, const std::vector<std::vector<FrameVertex>>& frameVertices,
    const Camera& camera, const std::vector<FaceRef>& visibleFaces, bool wireframe) {
    commands_.clear();
    stats_.opaqueTriangles = 0;
    stats_.transparentTriangles = 0;
    stats_.opaqueCommands = 0;
    stats_.transparentCommands = 0;
    stats_.wireCommands = 0;
    stats_.stateSwitches = 0;
    stats_.transparentOrderValid = true;

    if (wireframe) {
        Command command;
        command.kind = Command::Kind::Wire;
        command.faces = visibleFaces;
        if (!command.faces.empty()) {
            commands_.push_back(std::move(command));
            stats_.wireCommands = 1;
        }
        stats_.submittedTriangles = static_cast<uint32_t>(visibleFaces.size());
        return;
    }

    std::vector<FaceRef> opaque;
    std::vector<FaceRef> transparent;
    opaque.reserve(visibleFaces.size());
    for (const FaceRef& face : visibleFaces) {
        if (scene.materials[face.material].transparent) {
            transparent.push_back(face);
        } else {
            opaque.push_back(face);
        }
    }

    auto opaqueLess = [&](const FaceRef& a, const FaceRef& b) {
        const Material& ma = scene.materials[a.material];
        const Material& mb = scene.materials[b.material];
        return std::tie(ma.stateKey, a.material, a.globalIndex) <
               std::tie(mb.stateKey, b.material, b.globalIndex);
    };
    std::sort(opaque.begin(), opaque.end(), opaqueLess);

    uint64_t lastState = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < opaque.size();) {
        const uint16_t material = opaque[i].material;
        const uint64_t state = scene.materials[material].stateKey;
        Command command;
        command.kind = Command::Kind::Opaque;
        command.material = material;
        while (i < opaque.size() && opaque[i].material == material) {
            command.faces.push_back(opaque[i]);
            ++i;
        }
        if (lastState != std::numeric_limits<uint64_t>::max() && lastState != state) {
            ++stats_.stateSwitches;
        }
        lastState = state;
        stats_.opaqueTriangles += static_cast<uint32_t>(command.faces.size());
        ++stats_.opaqueCommands;
        commands_.push_back(std::move(command));
    }

    const Vec3 cameraPosition = camera.position();
    const Vec3 cameraForward = camera.forward();
    std::vector<std::pair<float, FaceRef>> transparentDepth;
    transparentDepth.reserve(transparent.size());
    for (const FaceRef& face : transparent) {
        const std::vector<FrameVertex>& verts = frameVertices[face.mesh];
        Vec3 center{};
        for (uint32_t index : face.vertexIndex) {
            center += verts[index].worldPosition;
        }
        center = center / 3.0f;
        const float depth = dot(center - cameraPosition, cameraForward);
        transparentDepth.emplace_back(depth, face);
    }
    auto transparentLess = [](const std::pair<float, FaceRef>& a,
                              const std::pair<float, FaceRef>& b) {
        if (a.first != b.first) {
            return a.first > b.first;
        }
        if (a.second.material != b.second.material) {
            return a.second.material < b.second.material;
        }
        return a.second.globalIndex < b.second.globalIndex;
    };
    std::sort(transparentDepth.begin(), transparentDepth.end(), transparentLess);
    for (const auto& entry : transparentDepth) {
        Command command;
        command.kind = Command::Kind::Transparent;
        command.material = entry.second.material;
        command.faces.push_back(entry.second);
        commands_.push_back(std::move(command));
        ++stats_.transparentCommands;
        ++stats_.transparentTriangles;
    }

    for (size_t i = 0; i < transparentDepth.size(); ++i) {
        if (i > 0 &&
            transparentLess(transparentDepth[i], transparentDepth[i - 1])) {
            stats_.transparentOrderValid = false;
            break;
        }
    }
    stats_.submittedTriangles =
        stats_.opaqueTriangles + stats_.transparentTriangles;
}

Vec3 Renderer::sampleNormalMap(Vec2 uv) const {
    const float x = uv.x * static_cast<float>(normalMap_.width) - 0.5f;
    const float y = uv.y * static_cast<float>(normalMap_.height) - 0.5f;
    const float fx = std::floor(x);
    const float fy = std::floor(y);
    const int x0 = static_cast<int>(fx);
    const int y0 = static_cast<int>(fy);
    const float tx = x - fx;
    const float ty = y - fy;
    const Vec3 c00 = sampleTexel(normalMap_, x0, y0);
    const Vec3 c10 = sampleTexel(normalMap_, x0 + 1, y0);
    const Vec3 c01 = sampleTexel(normalMap_, x0, y0 + 1);
    const Vec3 c11 = sampleTexel(normalMap_, x0 + 1, y0 + 1);
    const Vec3 top = c00 + (c10 - c00) * tx;
    const Vec3 bottom = c01 + (c11 - c01) * tx;
    const Vec3 encoded = top + (bottom - top) * ty;
    return normalize(encoded * 2.0f - Vec3{1.0f, 1.0f, 1.0f});
}

float Renderer::sampleShadow(const DepthRasterizer& map, Vec2 uv, float depth,
                             float bias) const {
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        return 1.0f;
    }
    const int radius = config_.pcfRadius;
    const float texelX = 1.0f / static_cast<float>(map.width());
    const float texelY = 1.0f / static_cast<float>(map.height());
    float lit = 0.0f;
    int taps = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const float u = uv.x + static_cast<float>(dx) * texelX;
            const float v = uv.y + static_cast<float>(dy) * texelY;
            if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
                lit += 1.0f;
            } else {
                const int x = std::clamp(static_cast<int>(u * map.width()), 0,
                                         map.width() - 1);
                const int y = std::clamp(static_cast<int>(v * map.height()), 0,
                                         map.height() - 1);
                const size_t offset =
                    static_cast<size_t>(y) * static_cast<size_t>(map.width()) +
                    static_cast<size_t>(x);
                const float stored = map.depthBuffer()[offset];
                lit += depth - bias <= stored + 1.0e-6f ? 1.0f : 0.0f;
            }
            ++taps;
        }
    }
    return taps > 0 ? lit / static_cast<float>(taps) : 1.0f;
}

float Renderer::directionalShadow(Vec3 position, Vec3 normal, Vec3 lightDirection,
                                  float ndotl, const Camera& camera) const {
    if (!config_.shadows) {
        return 1.0f;
    }
    const float cameraDepth = dot(position - camera.position(), camera.forward());
    if (cameraDepth < 0.0f || cameraDepth >= camera.farZ) {
        return 1.0f;
    }
    const size_t cascade = cameraDepth <= config_.cascadeSplit ? 0 : 1;
    const Vec3 offsetPosition = position + normal * 0.018f;
    const Vec4 clip = cascadeViewProjection_[cascade].transform(
        {offsetPosition.x, offsetPosition.y, offsetPosition.z, 1.0f});
    if (clip.w <= 1.0e-7f) {
        return 1.0f;
    }
    const float invW = 1.0f / clip.w;
    const Vec2 uv{(clip.x * invW) * 0.5f + 0.5f, 0.5f - (clip.y * invW) * 0.5f};
    const float depth = clip.z * invW;
    const float bias = 0.0012f + (1.0f - clamp01(ndotl)) * 0.0045f;
    (void)lightDirection;
    return sampleShadow(cascades_[cascade], uv, depth, bias);
}

float Renderer::pointShadow(size_t lightIndex, const Light& light, Vec3 position,
                            Vec3 normal, float ndotl) const {
    if (!config_.shadows) {
        return 1.0f;
    }
    const Vec3 delta = position - light.position;
    size_t face = 0;
    if (std::abs(delta.x) >= std::abs(delta.y) &&
        std::abs(delta.x) >= std::abs(delta.z)) {
        face = delta.x >= 0.0f ? 0 : 1;
    } else if (std::abs(delta.y) >= std::abs(delta.z)) {
        face = delta.y >= 0.0f ? 2 : 3;
    } else {
        face = delta.z >= 0.0f ? 4 : 5;
    }
    if (lightIndex >= pointShadowViews_.size()) {
        return 1.0f;
    }
    const Vec3 offsetPosition = position + normal * 0.012f;
    const Vec4 clip = pointShadowViews_[lightIndex].viewProjection[face].transform(
        {offsetPosition.x, offsetPosition.y, offsetPosition.z, 1.0f});
    if (clip.w <= 1.0e-7f) {
        return 1.0f;
    }
    const float invW = 1.0f / clip.w;
    const Vec2 uv{(clip.x * invW) * 0.5f + 0.5f, 0.5f - (clip.y * invW) * 0.5f};
    const float depth = clip.z * invW;
    const float bias = 0.0018f + (1.0f - clamp01(ndotl)) * 0.0075f;
    return sampleShadow(pointShadowViews_[lightIndex].faces[face], uv, depth, bias);
}

ShadeResult Renderer::shadeFragment(const FragmentInput& fragment,
                                    const Material& material, const Camera& camera,
                                    const Lighting& lighting) const {
    Vec3 normal = normalize(fragment.worldNormal);
    const Vec3 viewDirection = normalize(camera.position() - fragment.worldPosition);
    if (material.doubleSided && dot(normal, viewDirection) < 0.0f) {
        normal = -normal;
    }
    Vec3 tangent{fragment.worldTangent.x, fragment.worldTangent.y,
                 fragment.worldTangent.z};
    tangent = tangent - normal * dot(normal, tangent);
    if (lengthSquared(tangent) < 1.0e-10f) {
        const Vec3 fallback = std::abs(normal.y) < 0.9f ? Vec3{0.0f, 1.0f, 0.0f}
                                                        : Vec3{1.0f, 0.0f, 0.0f};
        tangent = normalize(cross(fallback, normal));
    } else {
        tangent = normalize(tangent);
    }
    const float handedness = fragment.worldTangent.w < 0.0f ? -1.0f : 1.0f;
    Vec3 bitangent = cross(normal, tangent) * handedness;

    const Vec3 tangentNormal = sampleNormalMap(fragment.uv);
    const float strength = material.normalStrength;
    normal = normalize(normal + tangent * (tangentNormal.x * strength) +
                       bitangent * (tangentNormal.y * strength));
    if (dot(normal, viewDirection) < 0.0f && material.doubleSided) {
        normal = -normal;
        bitangent = cross(normal, tangent) * handedness;
    }

    const float ndotv = clamp01(dot(normal, viewDirection));
    const Vec3 hemi = smoothstepAmbient(lighting.ambientGround, lighting.ambientSky,
                                        normal.y * 0.5f + 0.5f);
    Vec3 color = material.baseColor * hemi;

    const Vec3 directionalL = normalize(-lighting.directionalDirection);
    const float directionalNdotL = dot(normal, directionalL);
    if (directionalNdotL > 0.0f) {
        const float shadow = directionalShadow(fragment.worldPosition, normal,
                                               directionalL, directionalNdotL, camera);
        const float diffuse = directionalNdotL;
        const Vec3 halfVector = normalize(directionalL + viewDirection);
        const float shininess = lerp(10.0f, 240.0f, 1.0f - material.roughness);
        const float specular =
            std::pow(clamp01(dot(normal, halfVector)), shininess) *
            (1.0f - material.roughness * 0.7f);
        const Vec3 specColor =
            Vec3{1.0f, 1.0f, 1.0f} * (1.0f - material.metallic) +
            material.baseColor * material.metallic;
        color += (material.baseColor * (1.0f - material.metallic) * diffuse +
                  specColor * specular) *
                 lighting.directionalColor * lighting.directionalIntensity * shadow;
    }

    for (size_t lightIndex = 0; lightIndex < lighting.pointLights.size(); ++lightIndex) {
        const Light& light = lighting.pointLights[lightIndex];
        const Vec3 toLight = light.position - fragment.worldPosition;
        const float distanceToLight = length(toLight);
        if (distanceToLight < 1.0e-5f) {
            continue;
        }
        const Vec3 lightDirection = toLight / distanceToLight;
        const float ndotl = dot(normal, lightDirection);
        if (ndotl <= 0.0f) {
            continue;
        }
        const float attenuation =
            1.0f / (1.0f + 0.16f * distanceToLight +
                    0.045f * distanceToLight * distanceToLight);
        const float shadow =
            pointShadow(lightIndex, light, fragment.worldPosition, normal, ndotl);
        const Vec3 halfVector = normalize(lightDirection + viewDirection);
        const float shininess = lerp(12.0f, 220.0f, 1.0f - material.roughness);
        const float specular =
            std::pow(clamp01(dot(normal, halfVector)), shininess) *
            (1.0f - material.roughness * 0.7f);
        const Vec3 specColor =
            Vec3{1.0f, 1.0f, 1.0f} * (1.0f - material.metallic) +
            material.baseColor * material.metallic;
        color += (material.baseColor * (1.0f - material.metallic) * ndotl +
                  specColor * specular) *
                 light.color * (light.intensity * attenuation * shadow);
    }

    const float rim = std::pow(1.0f - ndotv, 3.0f);
    color += material.baseColor * rim * 0.12f;
    color += material.emissive;
    return {color, material.transparent ? material.opacity : 1.0f};
}

void Renderer::executeCommands(
    const Scene& scene, const std::vector<std::vector<FrameVertex>>& frameVertices,
    const Camera& camera, const Lighting& lighting) {
    const Mat4 viewProjection =
        camera.viewProjection(static_cast<float>(config_.width) /
                              static_cast<float>(config_.height));

    auto makeVertex = [&](const FrameVertex& source) {
        RasterVertex vertex;
        vertex.clip = viewProjection.transform(
            {source.worldPosition.x, source.worldPosition.y, source.worldPosition.z, 1.0f});
        vertex.worldPosition = source.worldPosition;
        vertex.worldNormal = source.worldNormal;
        vertex.worldTangent = source.worldTangent;
        vertex.uv = source.uv;
        return vertex;
    };

    for (const Command& command : commands_) {
        for (const FaceRef& face : command.faces) {
            const Material& material = scene.materials[face.material];
            const std::vector<FrameVertex>& verts = frameVertices[face.mesh];
            const RasterVertex a = makeVertex(verts[face.vertexIndex[0]]);
            const RasterVertex b = makeVertex(verts[face.vertexIndex[1]]);
            const RasterVertex c = makeVertex(verts[face.vertexIndex[2]]);
            if (command.kind == Command::Kind::Wire) {
                raster_.drawWireTriangle(
                    a, b, c, material.doubleSided ? CullMode::None : CullMode::Back,
                    material.baseColor * 1.5f + Vec3{0.04f, 0.42f, 0.22f});
                continue;
            }
            RasterState state;
            state.cull = material.doubleSided ? CullMode::None : CullMode::Back;
            state.depthWrite = command.kind == Command::Kind::Opaque;
            state.blend = command.kind == Command::Kind::Transparent;
            raster_.drawTriangle(
                a, b, c, state,
                [&](const FragmentInput& fragment) {
                    return shadeFragment(fragment, material, camera, lighting);
                });
        }
    }
}

void Renderer::resolveImage() {
    const std::vector<Vec3>& color = raster_.colorBuffer();
    for (size_t i = 0; i < color.size(); ++i) {
        const Vec3& c = color[i];
        image_.rgb[i * 3 + 0] = quantize(c.x);
        image_.rgb[i * 3 + 1] = quantize(c.y);
        image_.rgb[i * 3 + 2] = quantize(c.z);
    }
}

void Renderer::render(const Scene& scene, const Lighting& lighting,
                      const Camera& camera, uint32_t tick, bool wireframe) {
    const auto frameStart = Clock::now();
    const std::vector<Mat4> boneMatrices = buildBoneMatrices(scene.skeleton, tick);
    std::vector<std::vector<FrameVertex>> frameVertices(scene.meshes.size());
    for (size_t i = 0; i < scene.meshes.size(); ++i) {
        frameVertices[i] = skinVertices(scene.meshes[i], boneMatrices);
    }
    const auto cullStart = Clock::now();
    const Frustum frustum = extractFrustum(
        camera.viewProjection(static_cast<float>(config_.width) /
                              static_cast<float>(config_.height)));
    VisibilityStats visibility;
    const std::vector<FaceRef> visibleFaces =
        grid_.query(frustum, frameVertices, visibility);
    const auto cullEnd = Clock::now();
    stats_.totalTriangles = static_cast<uint32_t>(scene.totalTriangles());
    stats_.gridCandidateTriangles = visibility.cellCandidateFaces;
    stats_.gridVisibleTriangles = visibility.frustumVisibleFaces;
    stats_.visibleCells = visibility.visibleCells;
    stats_.totalCells = visibility.totalCells;
    stats_.cullRatio = static_cast<float>(visibility.frustumVisibleFaces) /
                       static_cast<float>(std::max(1u, visibility.totalFaces));
    stats_.cullMs = elapsedMs(cullStart, cullEnd);

    configureShadowMatrices(lighting);
    const auto shadowStart = Clock::now();
    if (!wireframe && config_.shadows && shadowCacheTick_ != static_cast<int64_t>(tick)) {
        renderShadows(scene, frameVertices, lighting);
        shadowCacheTick_ = static_cast<int64_t>(tick);
    } else if (!wireframe && config_.shadows) {
        stats_.shadowMaps = 2 + 12;
    } else {
        stats_.shadowMaps = 0;
        stats_.shadowTriangles = 0;
    }
    const auto shadowEnd = Clock::now();
    stats_.shadowMs = elapsedMs(shadowStart, shadowEnd);

    buildCommands(scene, frameVertices, camera, visibleFaces, wireframe);

    raster_.clear({0.006f, 0.010f, 0.018f});
    const auto colorStart = Clock::now();
    executeCommands(scene, frameVertices, camera, lighting);
    const auto colorEnd = Clock::now();
    stats_.colorMs = elapsedMs(colorStart, colorEnd);
    resolveImage();
    stats_.totalMs = elapsedMs(frameStart, Clock::now());
}

}  // namespace cv

