#include "data.hpp"
#include "errors.hpp"
#include "image_io.hpp"
#include "json_util.hpp"
#include "renderer.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace terrain {
namespace {

constexpr int kSnapshotVersion = 3;
constexpr const char* kSnapshotDefault = "render_snapshot.json";

volatile std::sig_atomic_t gInterrupted = 0;

void signalHandler(int) {
  gInterrupted = 1;
  CancelToken::requestGlobalCancellation();
}

#ifdef _WIN32
BOOL windowsConsoleHandler(DWORD event) {
  if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT ||
      event == CTRL_CLOSE_EVENT) {
    gInterrupted = 1;
    CancelToken::requestGlobalCancellation();
    return TRUE;
  }
  return FALSE;
}
#endif

void installCancellationHandler() {
#ifdef _WIN32
  SetConsoleCtrlHandler(windowsConsoleHandler, TRUE);
#else
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);
#endif
}

struct CommonOptions {
  std::filesystem::path dataDirectory = "generated_data";
  int seed = 1337;
  int shadowMapSize = kShadowMapSize;
  std::filesystem::path snapshotPath = kSnapshotDefault;
  std::filesystem::path reportPath;
  bool noRestore = false;
  bool noSaveSnapshot = false;
  bool includeWallTime = false;
  std::string fault = "none";
  int64_t cancelAfterTriangles = -1;
  LightState defaultLight;
  bool lightExplicit = false;
};

struct RenderOptions {
  CommonOptions common;
  CameraState camera = makeOrbitCamera(0.0f, 0.0f);
  bool yawExplicit = false;
  bool panExplicit = false;
  bool fovExplicit = false;
  bool nearExplicit = false;
  bool farExplicit = false;
  ShadowMode shadowMode = ShadowMode::Soft;
  float biasTexels = 2.0f;
  float pcfRadiusTexels = 8.0f;
  std::filesystem::path outputPath = "outputs/terrain.png";
  std::filesystem::path shadowPreviewPath;
};

struct AnglesOptions {
  CommonOptions common;
  std::filesystem::path outputDirectory = "outputs/angles";
  ShadowMode defaultShadowMode = ShadowMode::Soft;
  float baseBiasTexels = 2.0f;
  float pcfRadiusTexels = 8.0f;
};

struct IncrementalOptions {
  CommonOptions common;
  std::filesystem::path outputDirectory = "outputs/incremental";
  float yawDegrees = 0.0f;
  float stepWorldUnits = 0.65f;
  ShadowMode shadowMode = ShadowMode::Soft;
  float biasTexels = 2.0f;
  float pcfRadiusTexels = 8.0f;
};

enum class SnapshotRestoreStatus {
  None,
  Restored,
  VersionMismatchIgnored,
  InvalidIgnored,
};

struct SnapshotData {
  int version = kSnapshotVersion;
  CameraState camera;
  LightState light;
  int seed = 1337;
  int shadowMapSize = kShadowMapSize;
  float biasTexels = 2.0f;
  ShadowMode shadowMode = ShadowMode::Soft;
};

std::string commandLineHelp() {
  return
      "terrain_renderer - deterministic offline software terrain renderer\n"
      "\n"
      "Commands:\n"
      "  render        Render one 1024x768 PNG (default command).\n"
      "  angles        Render yaw 0/45/90 hard+soft and bias 2/4 comparisons.\n"
      "  incremental   Render five small pans plus one large rollback.\n"
      "  generate-data Generate the deterministic 256x256 input maps.\n"
      "\n"
      "Common options:\n"
      "  --data-dir DIR --seed N --snapshot PATH --no-restore\n"
      "  --no-save-snapshot --shadow-size 2048 --bias-texels F\n"
      "  --light-x F --light-y F --light-z F\n"
      "  --test-fault none|dim_mismatch|degenerate_proj|shadow_alloc|cancel\n"
      "  --cancel-after-triangles N --include-wall-time\n"
      "\n"
      "render options:\n"
      "  --out PNG --report JSON --yaw DEG --pan-x F --fov-y DEG\n"
      "  --near F --far F --shadow hard|soft --pcf-radius-texels F\n"
      "\n"
      "angles options:\n"
      "  --out-dir DIR\n"
      "\n"
      "incremental options:\n"
      "  --out-dir DIR --yaw DEG --step F --shadow hard|soft\n"
      "\n"
      "generate-data options:\n"
      "  --data-dir DIR --width N --height N\n";
}

bool isOption(const std::string& value) {
  return value == "-h" ||
         (value.size() >= 2 && value[0] == '-' && value[1] == '-');
}

const std::string& requireValue(const std::vector<std::string>& arguments,
                                size_t& index, const std::string& option) {
  if (index + 1 >= arguments.size() || isOption(arguments[index + 1])) {
    throw usageError("Option " + option + " requires a value");
  }
  ++index;
  return arguments[index];
}

float parseFloat(const std::string& text, const std::string& option) {
  if (text.empty()) {
    throw usageError("Option " + option + " requires a numeric value");
  }
  char* end = nullptr;
  errno = 0;
  const float value = std::strtof(text.c_str(), &end);
  if (errno != 0 || end == nullptr || *end != '\0' || !std::isfinite(value)) {
    throw usageError("Invalid numeric value for " + option + ": " + text);
  }
  return value;
}

int parseInt(const std::string& text, const std::string& option,
             int minimum = std::numeric_limits<int>::min(),
             int maximum = std::numeric_limits<int>::max()) {
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0' || value < minimum || value > maximum) {
    throw usageError("Invalid integer value for " + option + ": " + text);
  }
  return static_cast<int>(value);
}

int64_t parseInt64(const std::string& text, const std::string& option) {
  char* end = nullptr;
  errno = 0;
  const long long value = std::strtoll(text.c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0' || value < 0) {
    throw usageError("Invalid non-negative integer for " + option + ": " + text);
  }
  return static_cast<int64_t>(value);
}

ShadowMode parseShadowMode(const std::string& value, const std::string& option) {
  if (value == "hard") {
    return ShadowMode::Hard;
  }
  if (value == "soft") {
    return ShadowMode::Soft;
  }
  throw usageError("Invalid value for " + option + ": " + value + " (expected hard or soft)");
}

std::string pathString(const std::filesystem::path& path) {
  return path.generic_string();
}

std::filesystem::path defaultReportPathFor(const std::filesystem::path& imagePath) {
  std::filesystem::path report = imagePath;
  report.replace_extension();
  report += "_report.json";
  return report;
}

std::string wallTimeString(int64_t microseconds) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::fixed << std::setprecision(3)
         << static_cast<double>(microseconds) / 1000.0;
  return stream.str();
}

void validateCommonOptions(CommonOptions& options) {
  if (options.shadowMapSize != kShadowMapSize) {
    throw usageError("--shadow-size must be 2048 in the validated pipeline");
  }
  const std::vector<std::string> faults = {
      "none", "dim_mismatch", "degenerate_proj", "shadow_alloc", "cancel"};
  if (std::find(faults.begin(), faults.end(), options.fault) == faults.end()) {
    throw usageError("Unknown --test-fault value: " + options.fault);
  }
}

void applyFaultIfRequested(const CommonOptions& options) {
  if (options.fault == "dim_mismatch") {
    throw businessError("DIM_MISMATCH", "Injected dimension mismatch fault");
  }
  if (options.fault == "degenerate_proj") {
    throw businessError("DEGENERATE_PROJ", "Injected degenerate projection fault");
  }
  if (options.fault == "shadow_alloc") {
    throw businessError("SHADOW_ALLOC", "Injected shadow map allocation fault");
  }
}

SnapshotRestoreStatus loadSnapshot(const std::filesystem::path& path,
                                   SnapshotData& snapshot) {
  const std::optional<std::string> document = readTextFile(path);
  if (!document.has_value()) {
    return SnapshotRestoreStatus::InvalidIgnored;
  }
  const std::optional<long long> version = jsonReadInteger(*document, "version");
  if (!version.has_value()) {
    return SnapshotRestoreStatus::InvalidIgnored;
  }
  if (*version != kSnapshotVersion) {
    return SnapshotRestoreStatus::VersionMismatchIgnored;
  }

  const std::optional<Vec3> eye = jsonReadVec3(*document, "eye");
  const std::optional<Vec3> target = jsonReadVec3(*document, "target");
  const std::optional<Vec3> lightDirection = jsonReadVec3(*document, "direction");
  const std::optional<double> yaw = jsonReadNumber(*document, "yaw_deg");
  const std::optional<double> pan = jsonReadNumber(*document, "pan_x");
  const std::optional<double> fov = jsonReadNumber(*document, "fov_y_deg");
  const std::optional<double> nearPlane = jsonReadNumber(*document, "near");
  const std::optional<double> farPlane = jsonReadNumber(*document, "far");
  const std::optional<long long> seed = jsonReadInteger(*document, "seed");
  const std::optional<long long> shadowSize = jsonReadInteger(*document, "shadow_map_size");
  const std::optional<double> bias = jsonReadNumber(*document, "bias_texels");
  const std::optional<std::string> shadowMode = jsonReadString(*document, "shadow_mode");

  if (!eye.has_value() || !target.has_value() || !lightDirection.has_value() ||
      !yaw.has_value() || !pan.has_value() || !fov.has_value() ||
      !nearPlane.has_value() || !farPlane.has_value() || !seed.has_value() ||
      !shadowSize.has_value() || !bias.has_value() || !shadowMode.has_value()) {
    return SnapshotRestoreStatus::InvalidIgnored;
  }
  if (!isFinite(*eye) || !isFinite(*target) || !isFinite(*lightDirection) ||
      lengthSquared(*target - *eye) <= 1.0e-8f ||
      lengthSquared(*lightDirection) <= 1.0e-8f ||
      *fov <= 1.0 || *fov >= 179.0 || *nearPlane <= 0.0 ||
      *farPlane <= *nearPlane || *shadowSize != kShadowMapSize ||
      *bias < 0.0 ||
      (*shadowMode != "hard" && *shadowMode != "soft")) {
    return SnapshotRestoreStatus::InvalidIgnored;
  }

  snapshot.camera.eye = *eye;
  snapshot.camera.target = *target;
  snapshot.camera.yawDegrees = static_cast<float>(*yaw);
  snapshot.camera.panX = static_cast<float>(*pan);
  snapshot.camera.fovYDegrees = static_cast<float>(*fov);
  snapshot.camera.nearPlane = static_cast<float>(*nearPlane);
  snapshot.camera.farPlane = static_cast<float>(*farPlane);
  snapshot.light.direction = normalize(*lightDirection);
  snapshot.seed = static_cast<int>(*seed);
  snapshot.shadowMapSize = static_cast<int>(*shadowSize);
  snapshot.biasTexels = static_cast<float>(*bias);
  snapshot.shadowMode = parseShadowMode(*shadowMode, "snapshot shadow_mode");
  snapshot.version = kSnapshotVersion;
  return SnapshotRestoreStatus::Restored;
}

std::string snapshotJson(const RenderSettings& settings, const LightState& light,
                         int seed, int shadowMapSize, uint64_t dataHash) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << "{\n"
         << "  \"version\": " << kSnapshotVersion << ",\n"
         << "  \"seed\": " << seed << ",\n"
         << "  \"data_hash\": \"" << hashToHex(dataHash) << "\",\n"
         << "  \"camera\": {\n"
         << "    \"eye\": " << jsonVec3(settings.camera.eye, 9) << ",\n"
         << "    \"target\": " << jsonVec3(settings.camera.target, 9) << ",\n"
         << "    \"yaw_deg\": " << jsonNumber(settings.camera.yawDegrees, 9) << ",\n"
         << "    \"pan_x\": " << jsonNumber(settings.camera.panX, 9) << ",\n"
         << "    \"fov_y_deg\": " << jsonNumber(settings.camera.fovYDegrees, 9) << ",\n"
         << "    \"near\": " << jsonNumber(settings.camera.nearPlane, 9) << ",\n"
         << "    \"far\": " << jsonNumber(settings.camera.farPlane, 9) << "\n"
         << "  },\n"
         << "  \"light\": {\n"
         << "    \"direction\": " << jsonVec3(light.direction, 9) << "\n"
         << "  },\n"
         << "  \"shadow_map_size\": " << shadowMapSize << ",\n"
         << "  \"bias_texels\": " << jsonNumber(settings.biasTexels, 9) << ",\n"
         << "  \"shadow_mode\": \"" << shadowModeName(settings.shadowMode) << "\"\n"
         << "}\n";
  return stream.str();
}

void saveSnapshot(const CommonOptions& common, const RenderSettings& settings,
                  const LightState& light, uint64_t dataHash) {
  if (common.noSaveSnapshot || common.snapshotPath.empty()) {
    return;
  }
  writeTextFileAtomic(
      common.snapshotPath,
      snapshotJson(settings, light, common.seed, common.shadowMapSize, dataHash));
}

std::string materialWarningsJson(int zeroTexels, int firstU, int firstV,
                                 uint64_t fallbackSamples) {
  if (zeroTexels <= 0 && fallbackSamples == 0) {
    return "[]";
  }
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << "[{\"code\":\"MATW_ZERO\",\"zero_texels\":" << zeroTexels
         << ",\"sampled_fallbacks\":" << fallbackSamples
         << ",\"first_uv\":[" << firstU << "," << firstV << "]}]";
  return stream.str();
}

std::string renderBodyJson(const RenderSettings& settings, const LightState& light,
                           const RenderResult& result, const TerrainRenderer& renderer,
                           const std::filesystem::path& imagePath,
                           const std::optional<int64_t>& wallMicroseconds) {
  const float depthEpsilonTexels =
      std::min(2.0f, 0.75f + 0.125f * std::max(0.0f, settings.biasTexels));
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << "    \"image\": \"" << jsonEscape(pathString(imagePath)) << "\",\n"
         << "    \"width\": " << result.width << ",\n"
         << "    \"height\": " << result.height << ",\n"
         << "    \"camera\": {\n"
         << "      \"eye\": " << jsonVec3(settings.camera.eye, 6) << ",\n"
         << "      \"target\": " << jsonVec3(settings.camera.target, 6) << ",\n"
         << "      \"yaw_deg\": " << jsonNumber(settings.camera.yawDegrees, 6) << ",\n"
         << "      \"pan_x\": " << jsonNumber(settings.camera.panX, 6) << ",\n"
         << "      \"fov_y_deg\": " << jsonNumber(settings.camera.fovYDegrees, 6) << ",\n"
         << "      \"near\": " << jsonNumber(settings.camera.nearPlane, 6) << ",\n"
         << "      \"far\": " << jsonNumber(settings.camera.farPlane, 6) << "\n"
         << "    },\n"
         << "    \"light\": {\"direction\": " << jsonVec3(light.direction, 6) << "},\n"
         << "    \"shadow\": {\n"
         << "      \"mode\": \"" << shadowModeName(settings.shadowMode) << "\",\n"
         << "      \"bias_texels\": " << jsonNumber(settings.biasTexels, 6) << ",\n"
         << "      \"normal_offset_texels\": "
         << jsonNumber(settings.biasTexels, 6) << ",\n"
         << "      \"depth_epsilon_texels\": "
         << jsonNumber(depthEpsilonTexels, 6) << ",\n"
         << "      \"pcf_kernel\": \"3x3\",\n"
         << "      \"pcf_radius_texels\": " << jsonNumber(settings.pcfRadiusTexels, 6) << ",\n"
         << "      \"map_size\": " << renderer.shadowMapSize() << "\n"
         << "    },\n"
         << "    \"render_time_ms\": " << result.deterministicRenderTimeMs << ",\n"
         << "    \"render_time_model\": \"fixed_work_v1\",\n"
         << "    \"triangle_count\": " << result.triangleCount << ",\n"
         << "    \"redrawn_blocks\": " << result.redrawnBlocks << ",\n"
         << "    \"reused_blocks\": " << result.reusedBlocks << ",\n"
         << "    \"total_blocks\": " << result.totalBlocks << ",\n"
         << "    \"rasterized_triangles\": " << result.rasterizedTriangles << ",\n"
         << "    \"shaded_pixels\": " << result.shadedPixels << ",\n"
         << "    \"shadow_samples\": " << result.shadowSamples << ",\n"
         << "    \"backlit_pixels\": " << result.backlitPixels << ",\n"
         << "    \"shadow_map_fragments\": " << result.shadowMapFragments << ",\n"
         << "    \"shadow_lookups_outside\": " << result.shadowLookupsOutside << ",\n"
         << "    \"shadow_occluded_samples\": " << result.shadowOccludedSamples << ",\n"
         << "    \"shadow_min_delta\": " << jsonNumber(result.shadowMinDelta, 8) << ",\n"
         << "    \"shadow_max_delta\": " << jsonNumber(result.shadowMaxDelta, 8) << ",\n"
         << "    \"material_warnings\": "
         << materialWarningsJson(renderer.zeroWeightTexels(), renderer.firstZeroU(),
                                 renderer.firstZeroV(), result.materialFallbackSamples)
         << ",\n"
         << "    \"data_hash\": \"" << hashToHex(renderer.sceneContentHash()) << "\"";
  if (wallMicroseconds.has_value()) {
    stream << ",\n    \"wall_time_ms\": "
           << wallTimeString(*wallMicroseconds);
  }
  stream << "\n";
  return stream.str();
}

std::string composeSingleReport(const std::string& command, int seed,
                                const RenderSettings& settings,
                                const LightState& light,
                                const RenderResult& result,
                                const TerrainRenderer& renderer,
                                const std::filesystem::path& imagePath,
                                const std::optional<int64_t>& wallMicroseconds) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << "{\n"
         << "  \"ok\": true,\n"
         << "  \"version\": " << kSnapshotVersion << ",\n"
         << "  \"command\": \"" << jsonEscape(command) << "\",\n"
         << "  \"seed\": " << seed << ",\n"
         << renderBodyJson(settings, light, result, renderer, imagePath,
                           wallMicroseconds)
         << "}\n";
  return stream.str();
}

struct AngleEntry {
  float yawDegrees = 0.0f;
  float biasTexels = 2.0f;
  ShadowMode shadowMode = ShadowMode::Soft;
  std::filesystem::path imagePath;
  RenderResult result;
};

std::string composeAnglesReport(int seed, const LightState& light,
                                const std::vector<AngleEntry>& entries,
                                const TerrainRenderer& renderer) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  int totalRedrawn = 0;
  uint64_t totalFallbacks = 0;
  for (const AngleEntry& entry : entries) {
    totalRedrawn += entry.result.redrawnBlocks;
    totalFallbacks += entry.result.materialFallbackSamples;
  }
  stream << "{\n"
         << "  \"ok\": true,\n"
         << "  \"version\": " << kSnapshotVersion << ",\n"
         << "  \"command\": \"angles\",\n"
         << "  \"seed\": " << seed << ",\n"
         << "  \"light\": {\"direction\": " << jsonVec3(light.direction, 6) << "},\n"
         << "  \"triangle_count\": " << renderer.triangleCount() << ",\n"
         << "  \"total_redrawn_blocks\": " << totalRedrawn << ",\n"
         << "  \"material_warnings\": "
         << materialWarningsJson(renderer.zeroWeightTexels(), renderer.firstZeroU(),
                                 renderer.firstZeroV(), totalFallbacks)
         << ",\n"
         << "  \"data_hash\": \"" << hashToHex(renderer.sceneContentHash()) << "\",\n"
         << "  \"outputs\": [\n";
  for (size_t i = 0; i < entries.size(); ++i) {
    const AngleEntry& entry = entries[i];
    stream << "    {\"yaw_deg\":" << jsonNumber(entry.yawDegrees, 6)
           << ",\"shadow_mode\":\"" << shadowModeName(entry.shadowMode)
           << "\",\"bias_texels\":" << jsonNumber(entry.biasTexels, 6)
           << ",\"image\":\"" << jsonEscape(pathString(entry.imagePath))
           << "\",\"redrawn_blocks\":" << entry.result.redrawnBlocks
           << ",\"shadow_samples\":" << entry.result.shadowSamples
           << ",\"triangle_count\":" << entry.result.triangleCount
           << ",\"render_time_ms\":" << entry.result.deterministicRenderTimeMs
           << "}";
    if (i + 1 < entries.size()) {
      stream << ",";
    }
    stream << "\n";
  }
  stream << "  ]\n"
         << "}\n";
  return stream.str();
}

std::string composeGenerateReport(int seed, int width, int height,
                                  const std::filesystem::path& directory,
                                  const HeightWeightData& data, bool generated) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << "{\n"
         << "  \"ok\": true,\n"
         << "  \"version\": " << kSnapshotVersion << ",\n"
         << "  \"command\": \"generate-data\",\n"
         << "  \"seed\": " << seed << ",\n"
         << "  \"width\": " << width << ",\n"
         << "  \"height\": " << height << ",\n"
         << "  \"data_dir\": \"" << jsonEscape(pathString(directory)) << "\",\n"
         << "  \"data_hash\": \"" << hashToHex(data.contentHash) << "\",\n"
         << "  \"zero_weight_texels\": " << data.zeroWeightTexels << ",\n"
         << "  \"material_warnings\": "
         << materialWarningsJson(data.zeroWeightTexels, data.firstZeroU,
                                 data.firstZeroV, 0)
         << ",\n"
         << "  \"generated\": " << (generated ? "true" : "false") << "\n"
         << "}\n";
  return stream.str();
}

void applyTestFaultCancel(const CommonOptions& options, CancelToken& token) {
  if (options.fault == "cancel") {
    token.requestCancel();
  }
}

void reportWallTimeIfRequested(const CommonOptions& common, int64_t microseconds) {
  if (common.includeWallTime) {
    std::cerr << "wall_time_ms=" << wallTimeString(microseconds) << "\n";
  }
}

}  // namespace
}  // namespace terrain

int main(int argc, char** argv) {
  using namespace terrain;
  installCancellationHandler();
  try {
    std::vector<std::string> arguments;
    arguments.reserve(static_cast<size_t>(std::max(1, argc - 1)));
    for (int i = 1; i < argc; ++i) {
      arguments.emplace_back(argv[i]);
    }

    if (!arguments.empty() && (arguments[0] == "--help" || arguments[0] == "-h")) {
      std::cout << commandLineHelp();
      return 0;
    }
    if (!arguments.empty() && arguments[0] == "--version") {
      std::cout << "terrain_renderer 1.0.0\n";
      return 0;
    }

    std::string command = "render";
    size_t startIndex = 0;
    if (!arguments.empty() && !isOption(arguments[0])) {
      command = arguments[0];
      startIndex = 1;
    }

    if (command == "help") {
      std::cout << commandLineHelp();
      return 0;
    }
    if (command == "generate-data") {
      CommonOptions common;
      common.snapshotPath = "";
      int width = kDefaultTerrainDimension;
      int height = kDefaultTerrainDimension;
      for (size_t i = startIndex; i < arguments.size(); ++i) {
        const std::string& option = arguments[i];
        if (option == "--data-dir") {
          common.dataDirectory = requireValue(arguments, i, option);
        } else if (option == "--seed") {
          common.seed = parseInt(requireValue(arguments, i, option), option, -2147483647, 2147483647);
        } else if (option == "--width") {
          width = parseInt(requireValue(arguments, i, option), option, 2, 8192);
        } else if (option == "--height") {
          height = parseInt(requireValue(arguments, i, option), option, 2, 8192);
        } else if (option == "--help") {
          std::cout << commandLineHelp();
          return 0;
        } else {
          throw usageError("Unknown option for generate-data: " + option);
        }
      }
      validateCommonOptions(common);
      const HeightWeightData data =
          generateHeightWeightData({common.seed, width, height});
      writeHeightWeightData(data, common.dataDirectory);
      const std::string report =
          composeGenerateReport(common.seed, width, height, common.dataDirectory,
                                data, true);
      if (!common.reportPath.empty()) {
        writeTextFileAtomic(common.reportPath, report);
      }
      std::cout << report;
      return 0;
    }

    if (command != "render" && command != "angles" && command != "incremental") {
      throw usageError("Unknown command: " + command);
    }

    CommonOptions common;
    RenderOptions renderOptions;
    AnglesOptions anglesOptions;
    IncrementalOptions incrementalOptions;
    RenderOptions* sharedRenderOptions =
        command == "render" ? &renderOptions : nullptr;
    if (sharedRenderOptions != nullptr) {
      sharedRenderOptions->common = common;
    }
    anglesOptions.common = common;
    incrementalOptions.common = common;

    bool widthExplicit = false;
    bool heightExplicit = false;
    std::optional<float> lightX;
    std::optional<float> lightY;
    std::optional<float> lightZ;

    for (size_t i = startIndex; i < arguments.size(); ++i) {
      const std::string& option = arguments[i];
      CommonOptions* target = &common;
      if (command == "render") {
        target = &renderOptions.common;
      } else if (command == "angles") {
        target = &anglesOptions.common;
      } else {
        target = &incrementalOptions.common;
      }
      if (option == "--data-dir") {
        target->dataDirectory = requireValue(arguments, i, option);
      } else if (option == "--seed") {
        target->seed = parseInt(requireValue(arguments, i, option), option,
                               -2147483647, 2147483647);
      } else if (option == "--snapshot") {
        target->snapshotPath = requireValue(arguments, i, option);
      } else if (option == "--no-restore") {
        target->noRestore = true;
      } else if (option == "--no-save-snapshot") {
        target->noSaveSnapshot = true;
      } else if (option == "--include-wall-time") {
        target->includeWallTime = true;
      } else if (option == "--shadow-size") {
        target->shadowMapSize = parseInt(requireValue(arguments, i, option), option, 64, 8192);
      } else if (option == "--test-fault") {
        target->fault = requireValue(arguments, i, option);
      } else if (option == "--cancel-after-triangles") {
        target->cancelAfterTriangles =
            parseInt64(requireValue(arguments, i, option), option);
      } else if (option == "--light-x") {
        lightX = parseFloat(requireValue(arguments, i, option), option);
        target->lightExplicit = true;
      } else if (option == "--light-y") {
        lightY = parseFloat(requireValue(arguments, i, option), option);
        target->lightExplicit = true;
      } else if (option == "--light-z") {
        lightZ = parseFloat(requireValue(arguments, i, option), option);
        target->lightExplicit = true;
      } else if (option == "--width") {
        parseInt(requireValue(arguments, i, option), option, 2, 8192);
        widthExplicit = true;
      } else if (option == "--height") {
        parseInt(requireValue(arguments, i, option), option, 2, 8192);
        heightExplicit = true;
      } else if (command == "render" && option == "--out") {
        renderOptions.outputPath = requireValue(arguments, i, option);
      } else if (command == "render" && option == "--report") {
        renderOptions.common.reportPath = requireValue(arguments, i, option);
      } else if (command == "render" && option == "--yaw") {
        renderOptions.camera = makeOrbitCamera(
            parseFloat(requireValue(arguments, i, option), option),
            renderOptions.camera.panX);
        renderOptions.yawExplicit = true;
      } else if (command == "render" && option == "--pan-x") {
        renderOptions.camera = makeOrbitCamera(
            renderOptions.camera.yawDegrees,
            parseFloat(requireValue(arguments, i, option), option));
        renderOptions.panExplicit = true;
      } else if (command == "render" && option == "--fov-y") {
        renderOptions.camera.fovYDegrees =
            parseFloat(requireValue(arguments, i, option), option);
        renderOptions.fovExplicit = true;
      } else if (command == "render" && option == "--near") {
        renderOptions.camera.nearPlane =
            parseFloat(requireValue(arguments, i, option), option);
        renderOptions.nearExplicit = true;
      } else if (command == "render" && option == "--far") {
        renderOptions.camera.farPlane =
            parseFloat(requireValue(arguments, i, option), option);
        renderOptions.farExplicit = true;
      } else if (command == "render" && option == "--shadow") {
        renderOptions.shadowMode =
            parseShadowMode(requireValue(arguments, i, option), option);
      } else if (command == "render" && option == "--bias-texels") {
        renderOptions.biasTexels =
            parseFloat(requireValue(arguments, i, option), option);
      } else if (command == "render" && option == "--pcf-radius-texels") {
        renderOptions.pcfRadiusTexels =
            parseFloat(requireValue(arguments, i, option), option);
      } else if (command == "render" && option == "--dump-shadow") {
        renderOptions.shadowPreviewPath = requireValue(arguments, i, option);
      } else if (command == "angles" && option == "--out-dir") {
        anglesOptions.outputDirectory = requireValue(arguments, i, option);
      } else if (command == "angles" && option == "--shadow") {
        anglesOptions.defaultShadowMode =
            parseShadowMode(requireValue(arguments, i, option), option);
      } else if (command == "angles" && option == "--bias-texels") {
        anglesOptions.baseBiasTexels =
            parseFloat(requireValue(arguments, i, option), option);
      } else if (command == "angles" && option == "--pcf-radius-texels") {
        anglesOptions.pcfRadiusTexels =
            parseFloat(requireValue(arguments, i, option), option);
      } else if (command == "incremental" && option == "--out-dir") {
        incrementalOptions.outputDirectory = requireValue(arguments, i, option);
      } else if (command == "incremental" && option == "--yaw") {
        incrementalOptions.yawDegrees =
            parseFloat(requireValue(arguments, i, option), option);
      } else if (command == "incremental" && option == "--step") {
        incrementalOptions.stepWorldUnits =
            parseFloat(requireValue(arguments, i, option), option);
      } else if (command == "incremental" && option == "--bias-texels") {
        incrementalOptions.biasTexels =
            parseFloat(requireValue(arguments, i, option), option);
      } else if (command == "incremental" && option == "--shadow") {
        incrementalOptions.shadowMode =
            parseShadowMode(requireValue(arguments, i, option), option);
      } else if (command == "incremental" && option == "--pcf-radius-texels") {
        incrementalOptions.pcfRadiusTexels =
            parseFloat(requireValue(arguments, i, option), option);
      } else if (option == "--help") {
        std::cout << commandLineHelp();
        return 0;
      } else {
        throw usageError("Unknown option: " + option);
      }
    }

    if (widthExplicit || heightExplicit) {
      throw usageError("Render uses fixed 256x256 inputs; --width/--height are only for generate-data");
    }
    if (renderOptions.biasTexels < 0.0f || incrementalOptions.biasTexels < 0.0f ||
        anglesOptions.baseBiasTexels < 0.0f) {
      throw usageError("--bias-texels must be non-negative");
    }
    if (renderOptions.pcfRadiusTexels < 0.0f || incrementalOptions.pcfRadiusTexels < 0.0f ||
        anglesOptions.pcfRadiusTexels < 0.0f) {
      throw usageError("--pcf-radius-texels must be non-negative");
    }
    if (incrementalOptions.stepWorldUnits <= 0.0f ||
        incrementalOptions.stepWorldUnits > 16.0f) {
      throw usageError("--step must be in (0, 16] world units");
    }
    validateCommonOptions(common);
    if (command == "render") {
      validateCommonOptions(renderOptions.common);
    } else if (command == "angles") {
      validateCommonOptions(anglesOptions.common);
    } else {
      validateCommonOptions(incrementalOptions.common);
    }

    CommonOptions& activeCommon =
        command == "render" ? renderOptions.common :
        (command == "angles" ? anglesOptions.common : incrementalOptions.common);
    if (activeCommon.dataDirectory.empty()) {
      throw usageError("--data-dir requires a non-empty path");
    }

    applyFaultIfRequested(activeCommon);

    SnapshotData snapshot;
    SnapshotRestoreStatus restoreStatus = SnapshotRestoreStatus::None;
    if (command == "render" && !renderOptions.common.noRestore &&
        !renderOptions.common.snapshotPath.empty() &&
        std::filesystem::exists(renderOptions.common.snapshotPath)) {
      restoreStatus = loadSnapshot(renderOptions.common.snapshotPath, snapshot);
      if (restoreStatus == SnapshotRestoreStatus::Restored) {
        if (!renderOptions.yawExplicit && !renderOptions.panExplicit) {
          renderOptions.camera = snapshot.camera;
        } else {
          const float yaw = renderOptions.yawExplicit
                                ? renderOptions.camera.yawDegrees
                                : snapshot.camera.yawDegrees;
          const float pan = renderOptions.panExplicit
                                ? renderOptions.camera.panX
                                : snapshot.camera.panX;
          renderOptions.camera = makeOrbitCamera(yaw, pan);
        }
        if (!renderOptions.fovExplicit) {
          renderOptions.camera.fovYDegrees = snapshot.camera.fovYDegrees;
        }
        if (!renderOptions.nearExplicit) {
          renderOptions.camera.nearPlane = snapshot.camera.nearPlane;
        }
        if (!renderOptions.farExplicit) {
          renderOptions.camera.farPlane = snapshot.camera.farPlane;
        }
        if (!activeCommon.lightExplicit) {
          activeCommon.defaultLight = snapshot.light;
        }
      }
    }

    if (activeCommon.lightExplicit) {
      const Vec3 direction{
          lightX.value_or(activeCommon.defaultLight.direction.x),
          lightY.value_or(activeCommon.defaultLight.direction.y),
          lightZ.value_or(activeCommon.defaultLight.direction.z),
      };
      if (lengthSquared(direction) <= 1.0e-10f) {
        throw usageError("Light direction cannot be all zero");
      }
      activeCommon.defaultLight.direction = normalize(direction);
    }

    bool generated = false;
    HeightWeightData data = ensureHeightWeightData(
        activeCommon.dataDirectory, {activeCommon.seed, kDefaultTerrainDimension,
                                     kDefaultTerrainDimension},
        true, &generated);

    if (command == "render") {
      RenderSettings settings;
      settings.camera = renderOptions.camera;
      settings.shadowMode = renderOptions.shadowMode;
      settings.biasTexels = renderOptions.biasTexels;
      settings.pcfRadiusTexels = renderOptions.pcfRadiusTexels;
      if (renderOptions.common.reportPath.empty()) {
        renderOptions.common.reportPath =
            defaultReportPathFor(renderOptions.outputPath);
      }
      const auto start = std::chrono::steady_clock::now();
      CancelToken token(renderOptions.common.cancelAfterTriangles);
      applyTestFaultCancel(renderOptions.common, token);
      TerrainRenderer renderer(std::move(data), renderOptions.common.defaultLight,
                               renderOptions.common.shadowMapSize);
      const RenderResult result = renderer.renderFull(settings, token);
      if (!renderOptions.shadowPreviewPath.empty()) {
        renderer.saveShadowMapPreview(renderOptions.shadowPreviewPath);
      }
      savePngAtomic(renderOptions.outputPath, result.width, result.height,
                    renderer.colorBuffer());
      const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start);
      const std::optional<int64_t> wall =
          renderOptions.common.includeWallTime
              ? std::optional<int64_t>(elapsed.count())
              : std::nullopt;
      const std::string report = composeSingleReport(
          "render", renderOptions.common.seed, settings,
          renderOptions.common.defaultLight, result, renderer,
          renderOptions.outputPath, wall);
      writeTextFileAtomic(renderOptions.common.reportPath, report);
      saveSnapshot(renderOptions.common, settings, renderOptions.common.defaultLight,
                   renderer.sceneContentHash());
      reportWallTimeIfRequested(renderOptions.common, elapsed.count());
      if (restoreStatus == SnapshotRestoreStatus::VersionMismatchIgnored) {
        std::cerr << "snapshot_version_mismatch_ignored\n";
      } else if (restoreStatus == SnapshotRestoreStatus::InvalidIgnored) {
        std::cerr << "snapshot_invalid_ignored\n";
      }
      std::cout << report;
      return 0;
    }

    if (command == "angles") {
      std::filesystem::create_directories(anglesOptions.outputDirectory);
      TerrainRenderer renderer(std::move(data), anglesOptions.common.defaultLight,
                               anglesOptions.common.shadowMapSize);
      std::vector<AngleEntry> entries;
      const float yaws[3] = {0.0f, 45.0f, 90.0f};
      for (float yaw : yaws) {
        for (ShadowMode mode : {ShadowMode::Hard, ShadowMode::Soft}) {
          RenderSettings settings;
          settings.camera = makeOrbitCamera(yaw, 0.0f);
          settings.shadowMode = mode;
          settings.biasTexels = anglesOptions.baseBiasTexels;
          settings.pcfRadiusTexels = anglesOptions.pcfRadiusTexels;
          CancelToken token(anglesOptions.common.cancelAfterTriangles);
          applyTestFaultCancel(anglesOptions.common, token);
          const RenderResult result = renderer.renderFull(settings, token);
          const std::string base =
              "yaw" + std::to_string(static_cast<int>(std::lround(yaw))) +
              "_" + shadowModeName(mode);
          const std::filesystem::path image =
              anglesOptions.outputDirectory / (base + ".png");
          const std::filesystem::path report =
              anglesOptions.outputDirectory / (base + ".json");
          savePngAtomic(image, result.width, result.height, renderer.colorBuffer());
          writeTextFileAtomic(
              report,
              composeSingleReport("angles", anglesOptions.common.seed, settings,
                                  anglesOptions.common.defaultLight, result,
                                  renderer, image, std::nullopt));
          entries.push_back({yaw, settings.biasTexels, mode, image, result});
          if (std::abs(yaw - 45.0f) <= 1.0e-6f) {
            const std::filesystem::path biasImage =
                anglesOptions.outputDirectory /
                ("bias2_" + shadowModeName(mode) + ".png");
            savePngAtomic(biasImage, result.width, result.height,
                          renderer.colorBuffer());
            writeTextFileAtomic(
                anglesOptions.outputDirectory /
                    ("bias2_" + shadowModeName(mode) + ".json"),
                composeSingleReport("angles", anglesOptions.common.seed, settings,
                                    anglesOptions.common.defaultLight, result,
                                    renderer, biasImage, std::nullopt));
          }
          saveSnapshot(anglesOptions.common, settings,
                       anglesOptions.common.defaultLight,
                       renderer.sceneContentHash());
        }
      }
      for (ShadowMode mode : {ShadowMode::Hard, ShadowMode::Soft}) {
        RenderSettings settings;
        settings.camera = makeOrbitCamera(45.0f, 0.0f);
        settings.shadowMode = mode;
        settings.biasTexels = 4.0f;
        settings.pcfRadiusTexels = anglesOptions.pcfRadiusTexels;
        CancelToken token(anglesOptions.common.cancelAfterTriangles);
        applyTestFaultCancel(anglesOptions.common, token);
        const RenderResult result = renderer.renderFull(settings, token);
        const std::string base = "bias4_" + shadowModeName(mode);
        const std::filesystem::path image =
            anglesOptions.outputDirectory / (base + ".png");
        const std::filesystem::path report =
            anglesOptions.outputDirectory / (base + ".json");
        savePngAtomic(image, result.width, result.height, renderer.colorBuffer());
        writeTextFileAtomic(
            report,
            composeSingleReport("angles", anglesOptions.common.seed, settings,
                                anglesOptions.common.defaultLight, result,
                                renderer, image, std::nullopt));
        entries.push_back({45.0f, 4.0f, mode, image, result});
        saveSnapshot(anglesOptions.common, settings,
                     anglesOptions.common.defaultLight,
                     renderer.sceneContentHash());
      }
      const std::string report =
          composeAnglesReport(anglesOptions.common.seed,
                              anglesOptions.common.defaultLight, entries, renderer);
      const std::filesystem::path reportPath =
          anglesOptions.outputDirectory / "angles_report.json";
      writeTextFileAtomic(reportPath, report);
      std::cout << report;
      return 0;
    }

    std::filesystem::create_directories(incrementalOptions.outputDirectory);
    RenderSettings settings;
    settings.camera = makeOrbitCamera(incrementalOptions.yawDegrees, 0.0f);
    settings.shadowMode = incrementalOptions.shadowMode;
    settings.biasTexels = incrementalOptions.biasTexels;
    settings.pcfRadiusTexels = incrementalOptions.pcfRadiusTexels;

    TerrainRenderer renderer(std::move(data), incrementalOptions.common.defaultLight,
                             incrementalOptions.common.shadowMapSize);
    CancelToken token(incrementalOptions.common.cancelAfterTriangles);
    applyTestFaultCancel(incrementalOptions.common, token);

    struct FrameOutput {
      int index = 0;
      float panX = 0.0f;
      std::filesystem::path image;
      std::filesystem::path report;
      RenderResult result;
    };
    std::vector<FrameOutput> frameOutputs;
    const auto saveFrame = [&](FrameOutput frame) {
      const std::string base =
          "frame_" + (frame.index < 10 ? std::string("0") : std::string()) +
          std::to_string(frame.index);
      frame.image = incrementalOptions.outputDirectory / (base + ".png");
      frame.report = incrementalOptions.outputDirectory / (base + ".json");
      savePngAtomic(frame.image, frame.result.width, frame.result.height,
                    renderer.colorBuffer());
      writeTextFileAtomic(
          frame.report,
          composeSingleReport("incremental", incrementalOptions.common.seed, settings,
                              incrementalOptions.common.defaultLight, frame.result,
                              renderer, frame.image, std::nullopt));
      frameOutputs.push_back(frame);
    };

    FrameOutput first;
    first.index = 0;
    first.panX = settings.camera.panX;
    first.result = renderer.renderFull(settings, token);
    saveFrame(first);

    for (int step = 1; step <= 5; ++step) {
      settings.camera = makeOrbitCamera(
          incrementalOptions.yawDegrees,
          incrementalOptions.stepWorldUnits * static_cast<float>(step));
      FrameOutput frame;
      frame.index = step;
      frame.panX = settings.camera.panX;
      frame.result = renderer.renderIncremental(settings, token);
      saveFrame(frame);
    }

    settings.camera = makeOrbitCamera(incrementalOptions.yawDegrees, 0.0f);
    FrameOutput rollback;
    rollback.index = 6;
    rollback.panX = 0.0f;
    rollback.result = renderer.renderIncremental(settings, token);
    saveFrame(rollback);

    TerrainRenderer referenceRenderer(
        ensureHeightWeightData(activeCommon.dataDirectory,
                               {activeCommon.seed, kDefaultTerrainDimension,
                                kDefaultTerrainDimension},
                               false, nullptr),
        incrementalOptions.common.defaultLight,
        incrementalOptions.common.shadowMapSize);
    referenceRenderer.buildShadowMap(token);
    const RenderResult referenceResult = referenceRenderer.renderFull(settings, token);
    const std::filesystem::path referenceImage =
        incrementalOptions.outputDirectory / "reference_full.png";
    savePngAtomic(referenceImage, referenceResult.width, referenceResult.height,
                  referenceRenderer.colorBuffer());

    const std::vector<uint8_t>& incrementalColor = renderer.colorBuffer();
    const std::vector<uint8_t>& referenceColor = referenceRenderer.colorBuffer();
    int maxDiff = 0;
    uint64_t differingPixels = 0;
    uint64_t pixelsOverTolerance = 0;
    std::vector<uint8_t> mismatchTiles(static_cast<size_t>(kTileCount), 0);
    for (int y = 0; y < kOutputHeight; ++y) {
      for (int x = 0; x < kOutputWidth; ++x) {
        const size_t pixel =
            static_cast<size_t>(y) * static_cast<size_t>(kOutputWidth) +
            static_cast<size_t>(x);
        int pixelMax = 0;
        for (int channel = 0; channel < 3; ++channel) {
          const int diff = std::abs(
              static_cast<int>(incrementalColor[pixel * 3u + static_cast<size_t>(channel)]) -
              static_cast<int>(referenceColor[pixel * 3u + static_cast<size_t>(channel)]));
          pixelMax = std::max(pixelMax, diff);
          maxDiff = std::max(maxDiff, diff);
        }
        if (pixelMax > 0) {
          ++differingPixels;
        }
        if (pixelMax > 1) {
          ++pixelsOverTolerance;
          const size_t tileIndex =
              static_cast<size_t>(y / kTileSize) * static_cast<size_t>(kTilesX) +
              static_cast<size_t>(x / kTileSize);
          mismatchTiles[tileIndex] = 1;
        }
      }
    }
    int correctionBlocks = 0;
    if (pixelsOverTolerance > 0) {
      const RenderResult correction = renderer.redrawTiles(settings, mismatchTiles, token);
      correctionBlocks = correction.redrawnBlocks;
      savePngAtomic(frameOutputs.back().image, correction.width, correction.height,
                    renderer.colorBuffer());
      writeTextFileAtomic(
          frameOutputs.back().report,
          composeSingleReport("incremental", incrementalOptions.common.seed, settings,
                              incrementalOptions.common.defaultLight, correction,
                              renderer, frameOutputs.back().image, std::nullopt));
      frameOutputs.back().result = correction;
      maxDiff = 0;
      differingPixels = 0;
      pixelsOverTolerance = 0;
      for (int y = 0; y < kOutputHeight; ++y) {
        for (int x = 0; x < kOutputWidth; ++x) {
          const size_t pixel =
              static_cast<size_t>(y) * static_cast<size_t>(kOutputWidth) +
              static_cast<size_t>(x);
          int pixelMax = 0;
          for (int channel = 0; channel < 3; ++channel) {
            const int diff = std::abs(
                static_cast<int>(renderer.colorBuffer()[pixel * 3u + static_cast<size_t>(channel)]) -
                static_cast<int>(referenceRenderer.colorBuffer()[pixel * 3u + static_cast<size_t>(channel)]));
            pixelMax = std::max(pixelMax, diff);
            maxDiff = std::max(maxDiff, diff);
          }
          if (pixelMax > 0) {
            ++differingPixels;
          }
          if (pixelMax > 1) {
            ++pixelsOverTolerance;
          }
        }
      }
    }

    int totalRedrawn = 0;
    for (const FrameOutput& frame : frameOutputs) {
      totalRedrawn += frame.result.redrawnBlocks;
    }
    const int fullEquivalent = static_cast<int>(frameOutputs.size()) * kTileCount;
    const bool withinTolerance = maxDiff <= 1;
    const int savingsPercentMilli =
        std::max(0, static_cast<int>(std::llround(
                        100000.0 * static_cast<double>(fullEquivalent - totalRedrawn) /
                        static_cast<double>(fullEquivalent))));

    std::ostringstream summary;
    summary.imbue(std::locale::classic());
    summary << "{\n"
            << "  \"ok\": true,\n"
            << "  \"version\": " << kSnapshotVersion << ",\n"
            << "  \"command\": \"incremental\",\n"
            << "  \"seed\": " << incrementalOptions.common.seed << ",\n"
            << "  \"yaw_deg\": " << jsonNumber(incrementalOptions.yawDegrees, 6) << ",\n"
            << "  \"small_step_world_units\": "
            << jsonNumber(incrementalOptions.stepWorldUnits, 6) << ",\n"
            << "  \"frame_count\": " << frameOutputs.size() << ",\n"
            << "  \"total_redrawn_blocks\": " << totalRedrawn << ",\n"
            << "  \"full_equivalent_blocks\": " << fullEquivalent << ",\n"
            << "  \"savings_percent\": "
            << jsonNumber(static_cast<double>(savingsPercentMilli) / 1000.0, 3) << ",\n"
            << "  \"correction_blocks\": " << correctionBlocks << ",\n"
            << "  \"final_max_channel_diff\": " << maxDiff << ",\n"
            << "  \"final_differing_pixels\": " << differingPixels << ",\n"
            << "  \"final_pixels_over_tolerance\": " << pixelsOverTolerance << ",\n"
            << "  \"final_within_tolerance\": " << (withinTolerance ? "true" : "false") << ",\n"
            << "  \"reference_image\": \"" << jsonEscape(pathString(referenceImage))
            << "\",\n"
            << "  \"material_warnings\": "
            << materialWarningsJson(renderer.zeroWeightTexels(), renderer.firstZeroU(),
                                    renderer.firstZeroV(),
                                    frameOutputs.back().result.materialFallbackSamples)
            << ",\n"
            << "  \"data_hash\": \"" << hashToHex(renderer.sceneContentHash()) << "\",\n"
            << "  \"frames\": [\n";
    for (size_t i = 0; i < frameOutputs.size(); ++i) {
      const FrameOutput& frame = frameOutputs[i];
      summary << "    {\"index\":" << frame.index
              << ",\"pan_x\":" << jsonNumber(frame.panX, 6)
              << ",\"image\":\"" << jsonEscape(pathString(frame.image))
              << "\",\"redrawn_blocks\":" << frame.result.redrawnBlocks
              << ",\"reused_blocks\":" << frame.result.reusedBlocks
              << ",\"shaded_pixels\":" << frame.result.shadedPixels
              << ",\"shadow_samples\":" << frame.result.shadowSamples
              << ",\"render_time_ms\":" << frame.result.deterministicRenderTimeMs
              << "}";
      if (i + 1 < frameOutputs.size()) {
        summary << ",";
      }
      summary << "\n";
    }
    summary << "  ]\n"
            << "}\n";

    const std::filesystem::path summaryPath =
        incrementalOptions.outputDirectory / "incremental_report.json";
    writeTextFileAtomic(summaryPath, summary.str());
    saveSnapshot(incrementalOptions.common, settings,
                 incrementalOptions.common.defaultLight,
                 renderer.sceneContentHash());
    std::cout << summary.str();
    return withinTolerance ? 0 : 1;
  } catch (const AppError& error) {
    std::cout << errorJson(error.code, error.what(), error.exitCode);
    return error.exitCode;
  } catch (const std::bad_alloc&) {
    std::cout << errorJson("OUT_OF_MEMORY", "Memory allocation failed", 1);
    return 1;
  } catch (const std::exception& error) {
    std::cout << errorJson("INTERNAL_ERROR", error.what(), 1);
    return 1;
  } catch (...) {
    std::cout << errorJson("INTERNAL_ERROR", "Unknown failure", 1);
    return 1;
  }
}
