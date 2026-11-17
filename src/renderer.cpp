#include "renderer.hpp"

#include "errors.hpp"
#include "image_io.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace terrain {
namespace {

constexpr float kRenderEpsilon = 1.0e-6f;
constexpr float kSignaturePixelQuantum = 64.0f;
constexpr float kSignatureDepthQuantum = 1000000.0f;
constexpr int kCancelPollMask = 4095;

struct MeshVertex {
  Vec3 position;
  Vec3 normal;
};

struct Mesh {
  std::vector<MeshVertex> vertices;
  std::vector<uint32_t> indices;
  Vec3 boundsMin;
  Vec3 boundsMax;
  int64_t triangleCount = 0;
};

struct Material {
  Vec3 albedo;
  float specularPower;
  float roughness;
};

constexpr Material kGrassMaterial{
    {0.20f, 0.47f, 0.12f},
    28.0f,
    0.72f,
};
constexpr Material kRockMaterial{
    {0.39f, 0.38f, 0.34f},
    52.0f,
    0.52f,
};
constexpr Material kSnowMaterial{
    {0.87f, 0.92f, 0.97f},
    90.0f,
    0.25f,
};
constexpr Material kFallbackMaterial{
    {0.50f, 0.45f, 0.39f},
    34.0f,
    0.68f,
};

struct MaterialMix {
  Vec3 albedo;
  float specularPower = 32.0f;
  float roughness = 0.6f;
  bool fallback = false;
};

struct ClipVertex {
  Vec4 clip;
  Vec3 world;
  Vec3 normal;
};

struct ClipPolygon {
  std::array<ClipVertex, 16> vertices;
  int count = 0;
};

struct ScreenVertex {
  float x = 0.0f;
  float y = 0.0f;
  float ndcZ = 1.0f;
  float invW = 1.0f;
  Vec3 world;
  Vec3 normal;
};

struct Rect {
  int minX = 0;
  int minY = 0;
  int maxX = -1;
  int maxY = -1;

  bool empty() const {
    return maxX < minX || maxY < minY;
  }
};

struct TileSignature {
  uint32_t count = 0;
  uint64_t sum1 = 0;
  uint64_t xor1 = 0;
  uint64_t sum2 = 0;

  bool operator==(const TileSignature& rhs) const {
    return count == rhs.count && sum1 == rhs.sum1 && xor1 == rhs.xor1 &&
           sum2 == rhs.sum2;
  }

  bool operator!=(const TileSignature& rhs) const {
    return !(*this == rhs);
  }
};

struct RenderStats {
  uint64_t rasterizedTriangles = 0;
  uint64_t shadedPixels = 0;
  uint64_t shadowSamples = 0;
  uint64_t backlitPixels = 0;
  uint64_t materialFallbackSamples = 0;
  uint64_t shadowLookupsOutside = 0;
  uint64_t shadowOccludedSamples = 0;
  float shadowMinDelta = std::numeric_limits<float>::max();
  float shadowMaxDelta = std::numeric_limits<float>::lowest();
  uint64_t scratch = 0;
};

struct CameraMatrices {
  Mat4 view;
  Mat4 projection;
  Mat4 viewProjection;
};

struct ShadowProjection {
  Mat4 view;
  Mat4 projection;
  Mat4 viewProjection;
  float nearPlane = 0.0f;
  float farPlane = 0.0f;
  float worldTexelSize = 0.0f;
};

inline uint64_t rotateLeft(uint64_t value, int amount) {
  const int shift = amount & 63;
  return shift == 0 ? value : ((value << shift) | (value >> (64 - shift)));
}

float signedArea(float ax, float ay, float bx, float by, float px, float py) {
  return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

Rect triangleBounds(const ScreenVertex& a, const ScreenVertex& b,
                    const ScreenVertex& c, int width, int height) {
  const float minXf = std::min({a.x, b.x, c.x});
  const float maxXf = std::max({a.x, b.x, c.x});
  const float minYf = std::min({a.y, b.y, c.y});
  const float maxYf = std::max({a.y, b.y, c.y});
  if (!std::isfinite(minXf) || !std::isfinite(maxXf) ||
      !std::isfinite(minYf) || !std::isfinite(maxYf)) {
    return {};
  }
  Rect result;
  result.minX = std::max(0, static_cast<int>(std::floor(minXf)));
  result.minY = std::max(0, static_cast<int>(std::floor(minYf)));
  result.maxX = std::min(width - 1, static_cast<int>(std::ceil(maxXf)) - 1);
  result.maxY = std::min(height - 1, static_cast<int>(std::ceil(maxYf)) - 1);
  return result;
}

Vec3 backgroundColor(int x, int y, int width, int height) {
  const float v = static_cast<float>(y) / static_cast<float>(std::max(1, height - 1));
  const float u = static_cast<float>(x) / static_cast<float>(std::max(1, width - 1));
  const Vec3 top{0.055f, 0.085f, 0.145f};
  const Vec3 horizon{0.38f, 0.49f, 0.60f};
  const Vec3 lower{0.20f, 0.28f, 0.32f};
  const float horizonBlend = smoothstep(0.30f, 0.72f, v);
  Vec3 color = lerp(top, horizon, horizonBlend);
  color = lerp(color, lower, smoothstep(0.72f, 1.0f, v));
  const float glow = std::exp(-((u - 0.82f) * (u - 0.82f) * 55.0f +
                                (v - 0.23f) * (v - 0.23f) * 30.0f));
  color += Vec3(0.30f, 0.20f, 0.08f) * (0.35f * glow);
  return color;
}

uint8_t encodeChannel(float linearValue) {
  const float mapped = 1.0f - std::exp(-std::max(0.0f, linearValue) * 1.12f);
  const float gammaCorrected = std::pow(mapped, 1.0f / 2.2f);
  return static_cast<uint8_t>(
      std::clamp(static_cast<int>(std::lround(gammaCorrected * 255.0f)), 0, 255));
}

void encodeColor(const Vec3& color, uint8_t* destination) {
  destination[0] = encodeChannel(color.x);
  destination[1] = encodeChannel(color.y);
  destination[2] = encodeChannel(color.z);
}

CameraMatrices buildCameraMatrices(const CameraState& camera, int width, int height) {
  if (!isFinite(camera.eye) || !isFinite(camera.target) ||
      !std::isfinite(camera.fovYDegrees) || !std::isfinite(camera.nearPlane) ||
      !std::isfinite(camera.farPlane)) {
    throw businessError("DEGENERATE_PROJ", "Camera parameters are not finite");
  }
  if (lengthSquared(camera.target - camera.eye) <= 1.0e-8f) {
    throw businessError("DEGENERATE_PROJ", "Camera eye and target coincide");
  }
  if (camera.fovYDegrees <= 1.0f || camera.fovYDegrees >= 179.0f ||
      camera.nearPlane <= 0.0f || camera.farPlane <= camera.nearPlane) {
    throw businessError("DEGENERATE_PROJ", "Camera projection parameters are degenerate");
  }

  CameraMatrices matrices;
  matrices.view = lookAtRH(camera.eye, camera.target, Vec3(0.0f, 1.0f, 0.0f));
  matrices.projection =
      perspectiveRH(radians(camera.fovYDegrees),
                    static_cast<float>(width) / static_cast<float>(height),
                    camera.nearPlane, camera.farPlane);
  matrices.viewProjection = matrices.projection * matrices.view;

  if (!isFinite(matrices.view) || !isFinite(matrices.projection) ||
      !isFinite(matrices.viewProjection) ||
      std::abs(determinant(matrices.projection)) < 1.0e-16f ||
      std::abs(determinant(matrices.viewProjection)) < 1.0e-18f) {
    throw businessError("DEGENERATE_PROJ", "Projection matrix is singular");
  }
  return matrices;
}

float clipPlaneDistance(const Vec4& clip, int plane) {
  switch (plane) {
    case 0:
      return clip.x + clip.w;
    case 1:
      return clip.w - clip.x;
    case 2:
      return clip.y + clip.w;
    case 3:
      return clip.w - clip.y;
    case 4:
      return clip.z + clip.w;
    case 5:
      return clip.w - clip.z;
    default:
      return -1.0f;
  }
}

ClipVertex interpolateClipVertex(const ClipVertex& a, const ClipVertex& b, float t) {
  ClipVertex result;
  result.clip = a.clip + (b.clip - a.clip) * t;
  result.world = lerp(a.world, b.world, t);
  result.normal = lerp(a.normal, b.normal, t);
  return result;
}

bool clipPolygonAgainstPlane(const ClipPolygon& input, int plane, ClipPolygon& output) {
  output.count = 0;
  if (input.count == 0) {
    return true;
  }
  ClipVertex previous = input.vertices[static_cast<size_t>(input.count - 1)];
  float previousDistance = clipPlaneDistance(previous.clip, plane);
  bool previousInside = previousDistance >= -kRenderEpsilon;

  for (int index = 0; index < input.count; ++index) {
    const ClipVertex current = input.vertices[static_cast<size_t>(index)];
    const float currentDistance = clipPlaneDistance(current.clip, plane);
    const bool currentInside = currentDistance >= -kRenderEpsilon;

    if (currentInside != previousInside) {
      const float denominator = previousDistance - currentDistance;
      if (std::abs(denominator) <= 1.0e-20f) {
        return false;
      }
      const float t = previousDistance / denominator;
      if (output.count >= static_cast<int>(output.vertices.size())) {
        return false;
      }
      output.vertices[static_cast<size_t>(output.count++)] =
          interpolateClipVertex(previous, current, clamp01(t));
    }
    if (currentInside) {
      if (output.count >= static_cast<int>(output.vertices.size())) {
        return false;
      }
      output.vertices[static_cast<size_t>(output.count++)] = current;
    }
    previous = current;
    previousDistance = currentDistance;
    previousInside = currentInside;
  }
  return true;
}

bool projectTriangle(const Mesh& mesh, uint32_t index0, uint32_t index1,
                     uint32_t index2, const Mat4& viewProjection,
                     int width, int height,
                     std::array<ScreenVertex, 16>& projected,
                     int& projectedCount) {
  ClipPolygon current;
  ClipPolygon clipped;
  const uint32_t indices[3] = {index0, index1, index2};
  for (int i = 0; i < 3; ++i) {
    const MeshVertex& vertex = mesh.vertices[indices[i]];
    ClipVertex clipVertex;
    clipVertex.clip =
        viewProjection * Vec4(vertex.position, 1.0f);
    clipVertex.world = vertex.position;
    clipVertex.normal = vertex.normal;
    current.vertices[static_cast<size_t>(current.count++)] = clipVertex;
  }

  for (int plane = 0; plane < 6; ++plane) {
    if (!clipPolygonAgainstPlane(current, plane, clipped)) {
      projectedCount = 0;
      return false;
    }
    current = clipped;
    if (current.count == 0) {
      projectedCount = 0;
      return false;
    }
  }

  projectedCount = 0;
  for (int i = 0; i < current.count; ++i) {
    const ClipVertex& vertex = current.vertices[static_cast<size_t>(i)];
    if (vertex.clip.w <= 1.0e-8f || !isFinite(vertex.clip)) {
      projectedCount = 0;
      return false;
    }
    const float inverseW = 1.0f / vertex.clip.w;
    const float ndcX = vertex.clip.x * inverseW;
    const float ndcY = vertex.clip.y * inverseW;
    const float ndcZ = vertex.clip.z * inverseW;
    if (!std::isfinite(ndcX) || !std::isfinite(ndcY) || !std::isfinite(ndcZ)) {
      projectedCount = 0;
      return false;
    }
    ScreenVertex screen;
    screen.x = (ndcX * 0.5f + 0.5f) * static_cast<float>(width);
    screen.y = (0.5f - ndcY * 0.5f) * static_cast<float>(height);
    screen.ndcZ = ndcZ;
    screen.invW = inverseW;
    screen.world = vertex.world;
    screen.normal = vertex.normal;
    if (!std::isfinite(screen.x) || !std::isfinite(screen.y)) {
      projectedCount = 0;
      return false;
    }
    projected[static_cast<size_t>(projectedCount++)] = screen;
  }
  return projectedCount >= 3;
}

uint64_t trianglePresentationHash(const ScreenVertex& a, const ScreenVertex& b,
                                  const ScreenVertex& c, uint32_t triangleId) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  const auto add = [&hash](int64_t value) {
    hash ^= static_cast<uint64_t>(value);
    hash *= 0x100000001b3ULL;
  };
  add(static_cast<int64_t>(triangleId));
  const ScreenVertex* vertices[3] = {&a, &b, &c};
  for (const ScreenVertex* vertex : vertices) {
    add(static_cast<int64_t>(
        std::llround(static_cast<double>(vertex->x) * kSignaturePixelQuantum)));
    add(static_cast<int64_t>(
        std::llround(static_cast<double>(vertex->y) * kSignaturePixelQuantum)));
    add(static_cast<int64_t>(
        std::llround(static_cast<double>(vertex->ndcZ) * kSignatureDepthQuantum)));
  }
  return hashMix(hash);
}

void addTriangleToTileSignatures(std::vector<TileSignature>& signatures,
                                 const ScreenVertex& a, const ScreenVertex& b,
                                 const ScreenVertex& c, uint32_t triangleId,
                                 int width, int height) {
  const Rect bounds = triangleBounds(a, b, c, width, height);
  if (bounds.empty()) {
    return;
  }
  const int tileMinX = bounds.minX / kTileSize;
  const int tileMaxX = bounds.maxX / kTileSize;
  const int tileMinY = bounds.minY / kTileSize;
  const int tileMaxY = bounds.maxY / kTileSize;
  const uint64_t hash = trianglePresentationHash(a, b, c, triangleId);
  const uint64_t hash2 = hashMix(hash ^ 0xd6e8feb86659fd93ULL);
  for (int tileY = tileMinY; tileY <= tileMaxY; ++tileY) {
    for (int tileX = tileMinX; tileX <= tileMaxX; ++tileX) {
      const size_t tileIndex =
          static_cast<size_t>(tileY) * static_cast<size_t>(kTilesX) +
          static_cast<size_t>(tileX);
      TileSignature& signature = signatures[tileIndex];
      ++signature.count;
      signature.sum1 += hash;
      signature.xor1 ^= rotateLeft(hash, tileX * 7 + tileY * 13);
      signature.sum2 += hash2;
    }
  }
}

std::vector<TileSignature> computeTileSignatures(
    const Mesh& mesh, const CameraMatrices& camera, int width, int height,
    CancelToken& cancelToken) {
  std::vector<TileSignature> signatures(static_cast<size_t>(kTileCount));
  if (mesh.indices.empty()) {
    return signatures;
  }
  const size_t triangleCount = mesh.indices.size() / 3u;
  for (size_t triangle = 0; triangle < triangleCount; ++triangle) {
    if ((triangle & static_cast<size_t>(kCancelPollMask)) == 0u) {
      cancelToken.poll(4096);
    }
    std::array<ScreenVertex, 16> projected{};
    int projectedCount = 0;
    if (!projectTriangle(mesh, mesh.indices[triangle * 3u],
                         mesh.indices[triangle * 3u + 1u],
                         mesh.indices[triangle * 3u + 2u],
                         camera.viewProjection, width, height, projected,
                         projectedCount)) {
      continue;
    }
    for (int fan = 1; fan + 1 < projectedCount; ++fan) {
      addTriangleToTileSignatures(signatures, projected[0],
                                  projected[static_cast<size_t>(fan)],
                                  projected[static_cast<size_t>(fan + 1)],
                                  static_cast<uint32_t>(triangle), width, height);
    }
  }
  return signatures;
}

void clearTile(std::vector<uint8_t>& color, std::vector<float>& depth,
               int width, int height, int tileX, int tileY) {
  const int startX = tileX * kTileSize;
  const int startY = tileY * kTileSize;
  for (int y = startY; y < startY + kTileSize; ++y) {
    for (int x = startX; x < startX + kTileSize; ++x) {
      const size_t index =
          static_cast<size_t>(y) * static_cast<size_t>(width) +
          static_cast<size_t>(x);
      depth[index] = 1.0f;
      encodeColor(backgroundColor(x, y, width, height), &color[index * 3u]);
    }
  }
}

MaterialMix sampleMaterial(const HeightWeightData& data, const Vec3& world,
                           RenderStats& stats) {
  const float u =
      (world.x / kTerrainWorldSize + 0.5f) * static_cast<float>(data.width - 1);
  const float v =
      (world.z / kTerrainWorldSize + 0.5f) * static_cast<float>(data.height - 1);
  const float clampedU = clampf(u, 0.0f, static_cast<float>(data.width - 1));
  const float clampedV = clampf(v, 0.0f, static_cast<float>(data.height - 1));
  const int x0 = static_cast<int>(std::floor(clampedU));
  const int y0 = static_cast<int>(std::floor(clampedV));
  const int x1 = std::min(data.width - 1, x0 + 1);
  const int y1 = std::min(data.height - 1, y0 + 1);
  const float tx = clampedU - static_cast<float>(x0);
  const float ty = clampedV - static_cast<float>(y0);

  const Vec3 w00 = data.weightsAt(x0, y0);
  const Vec3 w10 = data.weightsAt(x1, y0);
  const Vec3 w01 = data.weightsAt(x0, y1);
  const Vec3 w11 = data.weightsAt(x1, y1);
  Vec3 weights = lerp(lerp(w00, w10, tx), lerp(w01, w11, tx), ty);
  const float weightSum = weights.x + weights.y + weights.z;
  if (weightSum <= 1.0e-6f || !std::isfinite(weightSum)) {
    ++stats.materialFallbackSamples;
    return {kFallbackMaterial.albedo, kFallbackMaterial.specularPower,
            kFallbackMaterial.roughness, true};
  }
  weights *= 1.0f / weightSum;

  MaterialMix result;
  result.albedo = kGrassMaterial.albedo * weights.x +
                  kRockMaterial.albedo * weights.y +
                  kSnowMaterial.albedo * weights.z;
  result.specularPower = kGrassMaterial.specularPower * weights.x +
                         kRockMaterial.specularPower * weights.y +
                         kSnowMaterial.specularPower * weights.z;
  result.roughness = kGrassMaterial.roughness * weights.x +
                     kRockMaterial.roughness * weights.y +
                     kSnowMaterial.roughness * weights.z;
  return result;
}

Vec3 shadeSurface(const HeightWeightData& data, const LightState& light,
                  const CameraState& camera, const Vec3& worldPosition,
                  const Vec3& normal, float shadowVisibility,
                  RenderStats& stats) {
  const MaterialMix material = sampleMaterial(data, worldPosition, stats);
  const Vec3 normalDirection = normalize(normal);
  const Vec3 toLight = normalize(-light.direction);
  const Vec3 toEye = normalize(camera.eye - worldPosition);
  const float nDotL = dot(normalDirection, toLight);

  const Vec3 skyAmbient = material.albedo * Vec3(0.21f, 0.25f, 0.31f);
  const Vec3 groundAmbient =
      material.albedo * Vec3(0.10f, 0.09f, 0.07f) *
      (0.5f - 0.5f * normalDirection.y);
  Vec3 color = skyAmbient + groundAmbient;

  if (nDotL <= 0.0f) {
    ++stats.backlitPixels;
    const float rim = 0.14f * (1.0f - std::abs(dot(normalDirection, toEye)));
    color += material.albedo * Vec3(0.15f, 0.20f, 0.29f) * (0.55f + rim);
  } else {
    const float visibility = clamp01(shadowVisibility);
    const Vec3 sunColor{1.13f, 1.05f, 0.89f};
    color += material.albedo * sunColor * (nDotL * 1.18f * visibility);

    const Vec3 halfway = normalize(toLight + toEye);
    const float nDotH = std::max(0.0f, dot(normalDirection, halfway));
    const float specularStrength = (1.0f - material.roughness) * 0.62f;
    if (specularStrength > 0.0f) {
      const float specular =
          std::pow(nDotH, std::max(1.0f, material.specularPower)) *
          specularStrength * nDotL * visibility;
      color += sunColor * specular;
    }
  }
  return color;
}

}  // namespace

void CancelToken::poll(int64_t workUnits) {
  workUnits_ += std::max<int64_t>(1, workUnits);
  if (cancelled_.load(std::memory_order_relaxed) ||
      globalCancellationRequested()) {
    throw businessError("CANCELED", "Rendering was canceled");
  }
  if (cancelAfterWorkUnits_ >= 0 && workUnits_ >= cancelAfterWorkUnits_) {
    cancelled_.store(true, std::memory_order_relaxed);
    throw businessError("CANCELED", "Rendering was canceled by the deterministic test trigger");
  }
}

namespace {
std::atomic<bool> gGlobalCancellation{false};
}

void CancelToken::requestGlobalCancellation() {
  gGlobalCancellation.store(true, std::memory_order_relaxed);
}

bool CancelToken::globalCancellationRequested() {
  return gGlobalCancellation.load(std::memory_order_relaxed);
}

CameraState makeOrbitCamera(float yawDegrees, float panX) {
  constexpr float orbitRadius = 410.0f;
  constexpr float cameraHeight = 180.0f;
  constexpr float targetHeight = 18.0f;
  CameraState camera;
  camera.yawDegrees = yawDegrees;
  camera.panX = panX;
  const float yawRadians = radians(yawDegrees);
  camera.target = {panX, targetHeight, 0.0f};
  camera.eye = {
      panX + std::sin(yawRadians) * orbitRadius,
      cameraHeight,
      std::cos(yawRadians) * orbitRadius,
  };
  return camera;
}

std::string shadowModeName(ShadowMode mode) {
  return mode == ShadowMode::Hard ? "hard" : "soft";
}

struct TerrainRenderer::Impl {
  HeightWeightData data;
  Mesh mesh;
  LightState light;
  int shadowSize = kShadowMapSize;
  std::vector<float> shadowDepth;
  ShadowProjection shadowProjection;
  bool shadowBuilt = false;
  uint64_t shadowMapFragments = 0;

  std::vector<uint8_t> color;
  std::vector<float> depth;
  std::vector<TileSignature> tileSignatures;
  bool haveSignatures = false;
  bool havePreviousSettings = false;
  ShadowMode previousShadowMode = ShadowMode::Soft;
  float previousBiasTexels = 2.0f;
  float previousPcfRadiusTexels = 8.0f;

  Impl(HeightWeightData inputData, LightState inputLight, int inputShadowSize)
      : data(std::move(inputData)),
        light(inputLight),
        shadowSize(inputShadowSize),
        color(static_cast<size_t>(kOutputWidth) * kOutputHeight * 3u),
        depth(static_cast<size_t>(kOutputWidth) * kOutputHeight, 1.0f),
        tileSignatures(static_cast<size_t>(kTileCount)) {
    buildMesh();
  }

  void buildMesh() {
    if (data.width < 2 || data.height < 2) {
      throw businessError("DIM_MISMATCH", "Cannot build a terrain mesh from fewer than 2x2 samples");
    }
    mesh.vertices.resize(static_cast<size_t>(data.width) * static_cast<size_t>(data.height));
    const float stepX = kTerrainWorldSize / static_cast<float>(data.width - 1);
    const float stepZ = kTerrainWorldSize / static_cast<float>(data.height - 1);
    mesh.boundsMin = {std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max()};
    mesh.boundsMax = {std::numeric_limits<float>::lowest(),
                      std::numeric_limits<float>::lowest(),
                      std::numeric_limits<float>::lowest()};

    for (int v = 0; v < data.height; ++v) {
      for (int u = 0; u < data.width; ++u) {
        const size_t index =
            static_cast<size_t>(v) * static_cast<size_t>(data.width) +
            static_cast<size_t>(u);
        MeshVertex& vertex = mesh.vertices[index];
        vertex.position = {
            (static_cast<float>(u) / static_cast<float>(data.width - 1) - 0.5f) *
                kTerrainWorldSize,
            data.heightAt(u, v),
            (static_cast<float>(v) / static_cast<float>(data.height - 1) - 0.5f) *
                kTerrainWorldSize,
        };
        const float dx =
            (data.heightAt(u + 1, v) - data.heightAt(u - 1, v)) / (2.0f * stepX);
        const float dz =
            (data.heightAt(u, v + 1) - data.heightAt(u, v - 1)) / (2.0f * stepZ);
        vertex.normal = normalize(Vec3(-dx, 1.0f, -dz));
        mesh.boundsMin = vmin(mesh.boundsMin, vertex.position);
        mesh.boundsMax = vmax(mesh.boundsMax, vertex.position);
      }
    }

    mesh.indices.reserve(static_cast<size_t>(data.width - 1) *
                         static_cast<size_t>(data.height - 1) * 6u);
    const auto vertexIndex = [this](int u, int v) {
      return static_cast<uint32_t>(v * data.width + u);
    };
    for (int v = 0; v < data.height - 1; ++v) {
      for (int u = 0; u < data.width - 1; ++u) {
        const uint32_t i00 = vertexIndex(u, v);
        const uint32_t i10 = vertexIndex(u + 1, v);
        const uint32_t i01 = vertexIndex(u, v + 1);
        const uint32_t i11 = vertexIndex(u + 1, v + 1);
        mesh.indices.push_back(i00);
        mesh.indices.push_back(i01);
        mesh.indices.push_back(i10);
        mesh.indices.push_back(i10);
        mesh.indices.push_back(i01);
        mesh.indices.push_back(i11);
      }
    }
    mesh.triangleCount = static_cast<int64_t>(mesh.indices.size() / 3u);
  }

  void buildShadowMap(CancelToken& cancelToken) {
    if (shadowBuilt) {
      return;
    }
    if (shadowSize < 64 || shadowSize > 8192) {
      throw businessError("SHADOW_ALLOC", "Shadow map size is outside the supported range");
    }
    try {
      shadowDepth.assign(static_cast<size_t>(shadowSize) * static_cast<size_t>(shadowSize),
                         1.0f);
    } catch (const std::bad_alloc&) {
      throw businessError("SHADOW_ALLOC", "Unable to allocate the shadow depth map");
    } catch (const std::length_error&) {
      throw businessError("SHADOW_ALLOC", "Shadow depth map size is not representable");
    }

    const Vec3 center = (mesh.boundsMin + mesh.boundsMax) * 0.5f;
    const float sceneExtent =
        std::max({mesh.boundsMax.x - mesh.boundsMin.x,
                  mesh.boundsMax.y - mesh.boundsMin.y,
                  mesh.boundsMax.z - mesh.boundsMin.z});
    const Vec3 lightPosition = center - normalize(light.direction) * (sceneExtent * 2.5f + 100.0f);
    Vec3 up = Vec3(0.0f, 1.0f, 0.0f);
    if (std::abs(dot(normalize(light.direction), up)) > 0.97f) {
      up = Vec3(0.0f, 0.0f, 1.0f);
    }
    const Mat4 lightView = lookAtRH(lightPosition, center, up);

    Vec3 minimum{std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()};
    Vec3 maximum{std::numeric_limits<float>::lowest(),
                 std::numeric_limits<float>::lowest(),
                 std::numeric_limits<float>::lowest()};
    for (int corner = 0; corner < 8; ++corner) {
      const Vec3 point{
          (corner & 1) ? mesh.boundsMax.x : mesh.boundsMin.x,
          (corner & 2) ? mesh.boundsMax.y : mesh.boundsMin.y,
          (corner & 4) ? mesh.boundsMax.z : mesh.boundsMin.z,
      };
      const Vec3 viewPoint = lightView.transformPoint(point);
      minimum = vmin(minimum, viewPoint);
      maximum = vmax(maximum, viewPoint);
    }
    constexpr float margin = 3.0f;
    const float orthoNear = std::max(0.01f, -maximum.z - margin);
    const float orthoFar = -minimum.z + margin;
    if (!(orthoFar > orthoNear) || !isFinite(minimum) || !isFinite(maximum)) {
      throw businessError("DEGENERATE_PROJ", "Light-space bounds are degenerate");
    }
    const Mat4 lightProjection =
        orthoRH(minimum.x - margin, maximum.x + margin,
                minimum.y - margin, maximum.y + margin, orthoNear, orthoFar);
    shadowProjection.view = lightView;
    shadowProjection.projection = lightProjection;
    shadowProjection.viewProjection = lightProjection * lightView;
    shadowProjection.nearPlane = orthoNear;
    shadowProjection.farPlane = orthoFar;
    shadowProjection.worldTexelSize =
        std::max(maximum.x - minimum.x, maximum.y - minimum.y) +
        2.0f * margin;
    shadowProjection.worldTexelSize /= static_cast<float>(shadowSize);
    if (!isFinite(shadowProjection.viewProjection) ||
        std::abs(determinant(shadowProjection.viewProjection)) < 1.0e-18f) {
      throw businessError("DEGENERATE_PROJ", "Light projection matrix is singular");
    }

    shadowMapFragments = 0;
    const size_t triangleCount = mesh.indices.size() / 3u;
    for (size_t triangle = 0; triangle < triangleCount; ++triangle) {
      if ((triangle & static_cast<size_t>(kCancelPollMask)) == 0u) {
        cancelToken.poll(4096);
      }
      std::array<ScreenVertex, 16> projected{};
      int projectedCount = 0;
      if (!projectTriangle(mesh, mesh.indices[triangle * 3u],
                           mesh.indices[triangle * 3u + 1u],
                           mesh.indices[triangle * 3u + 2u],
                           shadowProjection.viewProjection, shadowSize,
                           shadowSize, projected, projectedCount)) {
        continue;
      }
      for (int fan = 1; fan + 1 < projectedCount; ++fan) {
        const ScreenVertex& a = projected[0];
        const ScreenVertex& b = projected[static_cast<size_t>(fan)];
        const ScreenVertex& c = projected[static_cast<size_t>(fan + 1)];
        const Rect bounds = triangleBounds(a, b, c, shadowSize, shadowSize);
        if (bounds.empty()) {
          continue;
        }
        const float area = signedArea(a.x, a.y, b.x, b.y, c.x, c.y);
        if (std::abs(area) <= 1.0e-10f) {
          continue;
        }
        for (int y = bounds.minY; y <= bounds.maxY; ++y) {
          const float py = static_cast<float>(y) + 0.5f;
          for (int x = bounds.minX; x <= bounds.maxX; ++x) {
            const float px = static_cast<float>(x) + 0.5f;
            const float w0 = signedArea(b.x, b.y, c.x, c.y, px, py) / area;
            const float w1 = signedArea(c.x, c.y, a.x, a.y, px, py) / area;
            const float w2 = 1.0f - w0 - w1;
            if (w0 < -kRenderEpsilon || w1 < -kRenderEpsilon ||
                w2 < -kRenderEpsilon) {
              continue;
            }
            const float depthValue =
                (w0 * a.ndcZ + w1 * b.ndcZ + w2 * c.ndcZ) * 0.5f + 0.5f;
            const size_t index =
                static_cast<size_t>(y) * static_cast<size_t>(shadowSize) +
                static_cast<size_t>(x);
            if (depthValue < shadowDepth[index]) {
              shadowDepth[index] = depthValue;
              ++shadowMapFragments;
            }
          }
        }
      }
    }
    shadowBuilt = true;
  }

  float readShadowDepthBilinear(float u, float v) const {
    const float x = u * static_cast<float>(shadowSize) - 0.5f;
    const float y = v * static_cast<float>(shadowSize) - 0.5f;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const auto sample = [this](int sx, int sy) {
      sx = std::max(0, std::min(shadowSize - 1, sx));
      sy = std::max(0, std::min(shadowSize - 1, sy));
      return shadowDepth[static_cast<size_t>(sy) * static_cast<size_t>(shadowSize) +
                         static_cast<size_t>(sx)];
    };
    const float d00 = sample(x0, y0);
    const float d10 = sample(x0 + 1, y0);
    const float d01 = sample(x0, y0 + 1);
    const float d11 = sample(x0 + 1, y0 + 1);
    const float dx0 = d00 + (d10 - d00) * tx;
    const float dx1 = d01 + (d11 - d01) * tx;
    return dx0 + (dx1 - dx0) * ty;
  }

  float sampleShadowVisibility(const RenderSettings& settings, const Vec3& world,
                               const Vec3& normal, int pixelX, int pixelY,
                               RenderStats& stats) const {
    const Vec3 receiver =
        world + normalize(normal) *
                    (std::max(0.0f, settings.biasTexels) *
                     shadowProjection.worldTexelSize);
    const Vec4 lightClip =
        shadowProjection.viewProjection * Vec4(receiver, 1.0f);
    if (lightClip.w <= 1.0e-8f) {
      return 1.0f;
    }
    const float u = (lightClip.x / lightClip.w) * 0.5f + 0.5f;
    const float v = 0.5f - (lightClip.y / lightClip.w) * 0.5f;
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
      ++stats.shadowLookupsOutside;
      return 1.0f;
    }
    const float receiverDepth =
        (lightClip.z / lightClip.w) * 0.5f + 0.5f;
    const float depthRange =
        shadowProjection.farPlane - shadowProjection.nearPlane;
    const float depthEpsilonTexels =
        std::min(2.0f, 0.75f + 0.125f * std::max(0.0f, settings.biasTexels));
    const float depthBias =
        depthEpsilonTexels * depthRange / static_cast<float>(shadowSize);
    const float compareDepth = receiverDepth - depthBias;

    if (settings.shadowMode == ShadowMode::Hard) {
      ++stats.shadowSamples;
      const float stored = readShadowDepthBilinear(u, v);
      const float delta = compareDepth - stored;
      stats.shadowMinDelta = std::min(stats.shadowMinDelta, delta);
      stats.shadowMaxDelta = std::max(stats.shadowMaxDelta, delta);
      const bool lit = compareDepth <= stored + 1.0e-7f;
      if (!lit) {
        ++stats.shadowOccludedSamples;
      }
      return lit ? 1.0f : 0.0f;
    }

    const uint64_t pixelHash =
        hashCombine(static_cast<uint64_t>(pixelX) * 0x9e3779b1ULL,
                    static_cast<uint64_t>(pixelY) * 0x85ebca6bULL ^
                        static_cast<uint64_t>(data.contentHash));
    const float angle =
        hashToUnitFloat(hashMix(pixelHash)) * 2.0f * kPi;
    const float subTexelJitter =
        (hashToUnitFloat(hashMix(pixelHash ^ 0xa5a5a5a5ULL)) - 0.5f) * 0.42f;
    const float cosine = std::cos(angle);
    const float sine = std::sin(angle);
    const float radius = std::max(0.0f, settings.pcfRadiusTexels) /
                         static_cast<float>(shadowSize);
    float visibility = 0.0f;
    float totalWeight = 0.0f;
    for (int oy = -1; oy <= 1; ++oy) {
      for (int ox = -1; ox <= 1; ++ox) {
        const float localX = static_cast<float>(ox) + subTexelJitter;
        const float localY = static_cast<float>(oy) + subTexelJitter;
        const float rotatedX = localX * cosine - localY * sine;
        const float rotatedY = localX * sine + localY * cosine;
        const float sampleU = clamp01(u + rotatedX * radius);
        const float sampleV = clamp01(v + rotatedY * radius);
        const float stored = readShadowDepthBilinear(sampleU, sampleV);
        const float delta = compareDepth - stored;
        stats.shadowMinDelta = std::min(stats.shadowMinDelta, delta);
        stats.shadowMaxDelta = std::max(stats.shadowMaxDelta, delta);
        const float lit = compareDepth <= stored + 1.0e-7f ? 1.0f : 0.0f;
        if (lit <= 0.0f) {
          ++stats.shadowOccludedSamples;
        }
        const float weight =
            1.0f / (1.0f + 0.18f * static_cast<float>(ox * ox + oy * oy));
        visibility += lit * weight;
        totalWeight += weight;
      }
    }
    stats.shadowSamples += 9;
    return totalWeight > 0.0f ? visibility / totalWeight : 1.0f;
  }

  void rasterizeMainTriangle(const RenderSettings& settings,
                             const ScreenVertex& a, const ScreenVertex& b,
                             const ScreenVertex& c, const Rect& tileRect,
                             RenderStats& stats, CancelToken& cancelToken) {
    Rect bounds = triangleBounds(a, b, c, kOutputWidth, kOutputHeight);
    bounds.minX = std::max(bounds.minX, tileRect.minX);
    bounds.minY = std::max(bounds.minY, tileRect.minY);
    bounds.maxX = std::min(bounds.maxX, tileRect.maxX);
    bounds.maxY = std::min(bounds.maxY, tileRect.maxY);
    if (bounds.empty()) {
      return;
    }
    const float area = signedArea(a.x, a.y, b.x, b.y, c.x, c.y);
    if (std::abs(area) <= 1.0e-10f) {
      return;
    }

    for (int y = bounds.minY; y <= bounds.maxY; ++y) {
      const float py = static_cast<float>(y) + 0.5f;
      for (int x = bounds.minX; x <= bounds.maxX; ++x) {
        const float px = static_cast<float>(x) + 0.5f;
        const float w0 = signedArea(b.x, b.y, c.x, c.y, px, py) / area;
        const float w1 = signedArea(c.x, c.y, a.x, a.y, px, py) / area;
        const float w2 = 1.0f - w0 - w1;
        if (w0 < -kRenderEpsilon || w1 < -kRenderEpsilon ||
            w2 < -kRenderEpsilon) {
          continue;
        }

        const float depthValue = w0 * a.ndcZ + w1 * b.ndcZ + w2 * c.ndcZ;
        const size_t pixelIndex =
            static_cast<size_t>(y) * static_cast<size_t>(kOutputWidth) +
            static_cast<size_t>(x);
        if (!(depthValue < depth[pixelIndex])) {
          continue;
        }
        const float inverseDenominator =
            1.0f / std::max(1.0e-20f, w0 * a.invW + w1 * b.invW + w2 * c.invW);
        const Vec3 world =
            (a.world * (w0 * a.invW) + b.world * (w1 * b.invW) +
             c.world * (w2 * c.invW)) *
            inverseDenominator;
        Vec3 normal =
            (a.normal * (w0 * a.invW) + b.normal * (w1 * b.invW) +
             c.normal * (w2 * c.invW)) *
            inverseDenominator;
        if (lengthSquared(normal) <= 1.0e-12f) {
          normal = Vec3(0.0f, 1.0f, 0.0f);
        } else {
          normal = normalize(normal);
        }

        const Vec3 toLight = normalize(-light.direction);
        float visibility = 1.0f;
        if (dot(normal, toLight) > 0.0f) {
          visibility =
              sampleShadowVisibility(settings, world, normal, x, y, stats);
        }
        const Vec3 surfaceColor =
            shadeSurface(data, light, settings.camera, world, normal, visibility, stats);
        depth[pixelIndex] = depthValue;
        encodeColor(surfaceColor, &color[pixelIndex * 3u]);
        ++stats.shadedPixels;
        if ((stats.shadedPixels & static_cast<uint64_t>(kCancelPollMask)) == 0u) {
          cancelToken.poll(4096);
        }
      }
    }
  }

  void rasterizeDirtyTiles(const RenderSettings& settings,
                           const CameraMatrices& camera,
                           const std::vector<uint8_t>& dirtyTiles,
                           RenderStats& stats, CancelToken& cancelToken) {
    const size_t triangleCount = mesh.indices.size() / 3u;
    for (size_t triangle = 0; triangle < triangleCount; ++triangle) {
      if ((triangle & static_cast<size_t>(kCancelPollMask)) == 0u) {
        cancelToken.poll(4096);
      }
      std::array<ScreenVertex, 16> projected{};
      int projectedCount = 0;
      if (!projectTriangle(mesh, mesh.indices[triangle * 3u],
                           mesh.indices[triangle * 3u + 1u],
                           mesh.indices[triangle * 3u + 2u],
                           camera.viewProjection, kOutputWidth, kOutputHeight,
                           projected, projectedCount)) {
        continue;
      }
      for (int fan = 1; fan + 1 < projectedCount; ++fan) {
        const ScreenVertex& a = projected[0];
        const ScreenVertex& b = projected[static_cast<size_t>(fan)];
        const ScreenVertex& c = projected[static_cast<size_t>(fan + 1)];
        const Rect bounds = triangleBounds(a, b, c, kOutputWidth, kOutputHeight);
        if (bounds.empty()) {
          continue;
        }
        const int tileMinX = bounds.minX / kTileSize;
        const int tileMaxX = bounds.maxX / kTileSize;
        const int tileMinY = bounds.minY / kTileSize;
        const int tileMaxY = bounds.maxY / kTileSize;
        for (int tileY = tileMinY; tileY <= tileMaxY; ++tileY) {
          for (int tileX = tileMinX; tileX <= tileMaxX; ++tileX) {
            const size_t tileIndex =
                static_cast<size_t>(tileY) * static_cast<size_t>(kTilesX) +
                static_cast<size_t>(tileX);
            if (!dirtyTiles[tileIndex]) {
              continue;
            }
            ++stats.rasterizedTriangles;
            const Rect tileRect{
                tileX * kTileSize,
                tileY * kTileSize,
                tileX * kTileSize + kTileSize - 1,
                tileY * kTileSize + kTileSize - 1,
            };
            rasterizeMainTriangle(settings, a, b, c, tileRect, stats, cancelToken);
          }
        }
      }
    }
  }

  RenderResult render(const RenderSettings& settings, bool forceAll,
                      const std::vector<uint8_t>* forcedTiles,
                      CancelToken& cancelToken) {
    buildShadowMap(cancelToken);
    const CameraMatrices camera =
        buildCameraMatrices(settings.camera, kOutputWidth, kOutputHeight);

    std::vector<TileSignature> currentSignatures =
        computeTileSignatures(mesh, camera, kOutputWidth, kOutputHeight, cancelToken);
    std::vector<uint8_t> dirtyTiles(static_cast<size_t>(kTileCount), 0);
    int redrawnBlocks = 0;
    if (forceAll || !haveSignatures) {
      std::fill(dirtyTiles.begin(), dirtyTiles.end(), 1);
      redrawnBlocks = kTileCount;
    } else if (forcedTiles != nullptr) {
      for (int i = 0; i < kTileCount; ++i) {
        dirtyTiles[static_cast<size_t>(i)] =
            static_cast<uint8_t>((*forcedTiles)[static_cast<size_t>(i)] != 0);
        redrawnBlocks += dirtyTiles[static_cast<size_t>(i)];
      }
    } else {
      for (int i = 0; i < kTileCount; ++i) {
        const bool shadingChanged =
            !havePreviousSettings ||
            previousShadowMode != settings.shadowMode ||
            previousBiasTexels != settings.biasTexels ||
            previousPcfRadiusTexels != settings.pcfRadiusTexels;
        dirtyTiles[static_cast<size_t>(i)] =
            static_cast<uint8_t>(shadingChanged ||
                                 tileSignatures[static_cast<size_t>(i)] !=
                                     currentSignatures[static_cast<size_t>(i)]);
        redrawnBlocks += dirtyTiles[static_cast<size_t>(i)];
      }
    }

    for (int tileY = 0; tileY < kTilesY; ++tileY) {
      for (int tileX = 0; tileX < kTilesX; ++tileX) {
        const size_t index =
            static_cast<size_t>(tileY) * static_cast<size_t>(kTilesX) +
            static_cast<size_t>(tileX);
        if (dirtyTiles[index]) {
          clearTile(color, depth, kOutputWidth, kOutputHeight, tileX, tileY);
        }
      }
    }

    RenderStats stats;
    rasterizeDirtyTiles(settings, camera, dirtyTiles, stats, cancelToken);
    tileSignatures = std::move(currentSignatures);
    haveSignatures = true;
    havePreviousSettings = true;
    previousShadowMode = settings.shadowMode;
    previousBiasTexels = settings.biasTexels;
    previousPcfRadiusTexels = settings.pcfRadiusTexels;

    RenderResult result;
    result.redrawnBlocks = redrawnBlocks;
    result.reusedBlocks = kTileCount - redrawnBlocks;
    result.triangleCount = mesh.triangleCount;
    result.rasterizedTriangles = stats.rasterizedTriangles;
    result.shadedPixels = stats.shadedPixels;
    result.shadowSamples = stats.shadowSamples;
    result.backlitPixels = stats.backlitPixels;
    result.materialFallbackSamples = stats.materialFallbackSamples;
    result.shadowMapFragments = shadowMapFragments;
    result.shadowLookupsOutside = stats.shadowLookupsOutside;
    result.shadowOccludedSamples = stats.shadowOccludedSamples;
    result.shadowMinDelta =
        stats.shadowMinDelta < std::numeric_limits<float>::max()
            ? stats.shadowMinDelta : 0.0f;
    result.shadowMaxDelta =
        stats.shadowMaxDelta > std::numeric_limits<float>::lowest()
            ? stats.shadowMaxDelta : 0.0f;

    const double work =
        static_cast<double>(mesh.triangleCount) * 0.00009 +
        static_cast<double>(stats.rasterizedTriangles) * 0.00018 +
        static_cast<double>(stats.shadedPixels) * 0.00011 +
        static_cast<double>(stats.shadowSamples) * 0.000032 +
        static_cast<double>(shadowMapFragments) * 0.00007 +
        static_cast<double>(redrawnBlocks) * 0.018;
    result.deterministicRenderTimeMs =
        std::max<int64_t>(1, static_cast<int64_t>(std::llround(work)));
    return result;
  }
};

TerrainRenderer::TerrainRenderer(HeightWeightData data, LightState light,
                                 int shadowMapSize)
    : impl_(std::make_unique<Impl>(std::move(data), light, shadowMapSize)) {}

TerrainRenderer::~TerrainRenderer() = default;

void TerrainRenderer::buildShadowMap(CancelToken& cancelToken) {
  impl_->buildShadowMap(cancelToken);
}

RenderResult TerrainRenderer::renderFull(const RenderSettings& settings,
                                         CancelToken& cancelToken) {
  return impl_->render(settings, true, nullptr, cancelToken);
}

RenderResult TerrainRenderer::renderIncremental(const RenderSettings& settings,
                                                CancelToken& cancelToken) {
  return impl_->render(settings, false, nullptr, cancelToken);
}

RenderResult TerrainRenderer::redrawTiles(
    const RenderSettings& settings, const std::vector<uint8_t>& forcedTiles,
    CancelToken& cancelToken) {
  if (forcedTiles.size() != static_cast<size_t>(kTileCount)) {
    throw businessError("TILE_MASK", "Forced tile mask has the wrong size");
  }
  return impl_->render(settings, false, &forcedTiles, cancelToken);
}

const std::vector<uint8_t>& TerrainRenderer::colorBuffer() const {
  return impl_->color;
}

const std::vector<float>& TerrainRenderer::depthBuffer() const {
  return impl_->depth;
}

const HeightWeightData& TerrainRenderer::data() const {
  return impl_->data;
}

const LightState& TerrainRenderer::light() const {
  return impl_->light;
}

int TerrainRenderer::shadowMapSize() const {
  return impl_->shadowSize;
}

int64_t TerrainRenderer::triangleCount() const {
  return impl_->mesh.triangleCount;
}

uint64_t TerrainRenderer::shadowMapFragments() const {
  return impl_->shadowMapFragments;
}

int TerrainRenderer::zeroWeightTexels() const {
  return impl_->data.zeroWeightTexels;
}

int TerrainRenderer::firstZeroU() const {
  return impl_->data.firstZeroU;
}

int TerrainRenderer::firstZeroV() const {
  return impl_->data.firstZeroV;
}

uint64_t TerrainRenderer::sceneContentHash() const {
  return impl_->data.contentHash;
}

void TerrainRenderer::saveShadowMapPreview(
    const std::filesystem::path& path) const {
  if (!impl_->shadowBuilt) {
    throw businessError("SHADOW_MISSING", "Shadow map has not been built");
  }
  std::vector<uint8_t> rgb(
      static_cast<size_t>(impl_->shadowSize) *
      static_cast<size_t>(impl_->shadowSize) * 3u);
  for (size_t i = 0; i < impl_->shadowDepth.size(); ++i) {
    const float depth = impl_->shadowDepth[i];
    const uint8_t value =
        depth >= 1.0f ? 0
                      : static_cast<uint8_t>(std::clamp(
                            static_cast<int>(std::lround(depth * 255.0f)), 0, 255));
    rgb[i * 3u + 0u] = value;
    rgb[i * 3u + 1u] = value;
    rgb[i * 3u + 2u] = value;
  }
  savePngAtomic(path, impl_->shadowSize, impl_->shadowSize, rgb);
}

}  // namespace terrain
