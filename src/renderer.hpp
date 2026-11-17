#pragma once

#include "data.hpp"
#include "math.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace terrain {

constexpr int kOutputWidth = 1024;
constexpr int kOutputHeight = 768;
constexpr int kTileSize = 64;
constexpr int kShadowMapSize = 2048;
constexpr int kTilesX = kOutputWidth / kTileSize;
constexpr int kTilesY = kOutputHeight / kTileSize;
constexpr int kTileCount = kTilesX * kTilesY;

enum class ShadowMode {
  Hard,
  Soft,
};

struct CameraState {
  Vec3 eye{0.0f, 180.0f, 410.0f};
  Vec3 target{0.0f, 18.0f, 0.0f};
  float yawDegrees = 0.0f;
  float panX = 0.0f;
  float fovYDegrees = 36.0f;
  float nearPlane = 1.0f;
  float farPlane = 1200.0f;
};

CameraState makeOrbitCamera(float yawDegrees, float panX);

struct LightState {
  Vec3 direction = normalize(Vec3(0.80f, -0.30f, 0.20f));
};

struct RenderSettings {
  CameraState camera;
  ShadowMode shadowMode = ShadowMode::Soft;
  float biasTexels = 2.0f;
  float pcfRadiusTexels = 8.0f;
};

struct RenderResult {
  int width = kOutputWidth;
  int height = kOutputHeight;
  int totalBlocks = kTileCount;
  int redrawnBlocks = 0;
  int reusedBlocks = 0;
  int64_t triangleCount = 0;
  uint64_t rasterizedTriangles = 0;
  uint64_t shadedPixels = 0;
  uint64_t shadowSamples = 0;
  uint64_t backlitPixels = 0;
  uint64_t materialFallbackSamples = 0;
  uint64_t shadowMapFragments = 0;
  uint64_t shadowLookupsOutside = 0;
  uint64_t shadowOccludedSamples = 0;
  float shadowMinDelta = 0.0f;
  float shadowMaxDelta = 0.0f;
  int64_t deterministicRenderTimeMs = 0;
};

class CancelToken {
 public:
  explicit CancelToken(int64_t cancelAfterWorkUnits = -1)
      : cancelAfterWorkUnits_(cancelAfterWorkUnits) {}

  void requestCancel() {
    cancelled_.store(true, std::memory_order_relaxed);
  }

  bool isCancelled() const {
    return cancelled_.load(std::memory_order_relaxed);
  }

  void poll(int64_t workUnits = 1);

  static void requestGlobalCancellation();
  static bool globalCancellationRequested();

 private:
  std::atomic<bool> cancelled_{false};
  int64_t cancelAfterWorkUnits_ = -1;
  int64_t workUnits_ = 0;
};

class TerrainRenderer {
 public:
  TerrainRenderer(HeightWeightData data, LightState light, int shadowMapSize);
  ~TerrainRenderer();

  TerrainRenderer(const TerrainRenderer&) = delete;
  TerrainRenderer& operator=(const TerrainRenderer&) = delete;

  void buildShadowMap(CancelToken& cancelToken);

  RenderResult renderFull(const RenderSettings& settings, CancelToken& cancelToken);
  RenderResult renderIncremental(const RenderSettings& settings, CancelToken& cancelToken);
  RenderResult redrawTiles(const RenderSettings& settings,
                           const std::vector<uint8_t>& forcedTiles,
                           CancelToken& cancelToken);

  const std::vector<uint8_t>& colorBuffer() const;
  const std::vector<float>& depthBuffer() const;

  const HeightWeightData& data() const;
  const LightState& light() const;
  int shadowMapSize() const;
  int64_t triangleCount() const;
  uint64_t shadowMapFragments() const;
  int zeroWeightTexels() const;
  int firstZeroU() const;
  int firstZeroV() const;
  uint64_t sceneContentHash() const;
  void saveShadowMapPreview(const std::filesystem::path& path) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::string shadowModeName(ShadowMode mode);

}  // namespace terrain
