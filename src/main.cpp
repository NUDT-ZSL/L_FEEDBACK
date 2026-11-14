#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "lf/brdf.h"
#include "lf/config_store.h"
#include "lf/error_code.h"
#include "lf/gl_api.h"
#include "lf/log.h"
#include "lf/renderer.h"
#include "lf/samples.h"
#include "lf/snapshot.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace lf;

struct Options {
  std::filesystem::path dataDirectory = "lf_runtime";
  std::filesystem::path shaderDirectory;
  int width = 1280;
  int height = 720;
  bool verify = false;
  int frameLimit = 0;
  double runSeconds = 0.0;
};

struct GlfwGuard {
  bool initialized = false;
  ~GlfwGuard() {
    if (initialized) {
      glfwTerminate();
    }
  }
};

struct WindowGuard {
  GLFWwindow* window = nullptr;
  ~WindowGuard() {
    if (window != nullptr) {
      glfwDestroyWindow(window);
    }
  }
};

struct InputActions {
  bool saveSnapshot = false;
  bool restoreSnapshot = false;
  bool togglePause = false;
  bool reloadConfig = false;
};

void glfw_error_callback(int code, const char* description) {
  log_warn(ErrorCode::E_RENDER_WINDOW_CREATE,
           "GLFW error " + std::to_string(code) + ": " +
               (description == nullptr ? "unknown" : description));
}

void key_callback(GLFWwindow* window, int key, int, int action, int) {
  if (action != GLFW_PRESS) {
    return;
  }
  auto* actions = static_cast<InputActions*>(glfwGetWindowUserPointer(window));
  if (actions == nullptr) {
    return;
  }
  switch (key) {
    case GLFW_KEY_F5:
      actions->saveSnapshot = true;
      break;
    case GLFW_KEY_F9:
      actions->restoreSnapshot = true;
      break;
    case GLFW_KEY_F6:
      actions->togglePause = true;
      break;
    case GLFW_KEY_R:
      actions->reloadConfig = true;
      break;
    case GLFW_KEY_ESCAPE:
      glfwSetWindowShouldClose(window, GLFW_TRUE);
      break;
    default:
      break;
  }
}

std::string gl_string(gl::GLenum name) {
  const gl::GLubyte* value = gl::gl.GetString(name);
  return value == nullptr ? "unknown" : reinterpret_cast<const char*>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto require_value = [&](const char* name) -> std::string {
      if (index + 1 >= argc) {
        throw AppError(ErrorCode::E_ARGUMENT,
                       std::string("missing value for ") + name);
      }
      return argv[++index];
    };
    if (argument == "--help" || argument == "-h") {
      std::cout
          << "lf_character_engine options:\n"
          << "  --data-dir PATH    runtime samples, config, snapshots, logs\n"
          << "  --shader-dir PATH  directory containing GLSL shaders\n"
          << "  --width N          window width (default 1280)\n"
          << "  --height N         window height (default 720)\n"
          << "  --frames N         close automatically after N rendered frames\n"
          << "  --run-seconds S    close automatically after S seconds\n"
          << "  --verify           hidden-window deterministic pixel verification\n";
      std::exit(0);
    }
    if (argument == "--data-dir") {
      options.dataDirectory = require_value("--data-dir");
    } else if (argument == "--shader-dir") {
      options.shaderDirectory = require_value("--shader-dir");
    } else if (argument == "--width") {
      try {
        options.width = std::stoi(require_value("--width"));
      } catch (const std::exception&) {
        throw AppError(ErrorCode::E_ARGUMENT, "--width expects an integer");
      }
    } else if (argument == "--height") {
      try {
        options.height = std::stoi(require_value("--height"));
      } catch (const std::exception&) {
        throw AppError(ErrorCode::E_ARGUMENT, "--height expects an integer");
      }
    } else if (argument == "--frames") {
      try {
        options.frameLimit = std::stoi(require_value("--frames"));
      } catch (const std::exception&) {
        throw AppError(ErrorCode::E_ARGUMENT, "--frames expects an integer");
      }
    } else if (argument == "--run-seconds") {
      try {
        options.runSeconds = std::stod(require_value("--run-seconds"));
      } catch (const std::exception&) {
        throw AppError(ErrorCode::E_ARGUMENT, "--run-seconds expects a number");
      }
    } else if (argument == "--verify") {
      options.verify = true;
    } else {
      throw AppError(ErrorCode::E_ARGUMENT, "unknown argument: " + argument);
    }
  }
  if (options.width < 64 || options.height < 64 || options.width > 8192 ||
      options.height > 8192) {
    throw AppError(ErrorCode::E_ARGUMENT, "window dimensions must be in [64, 8192]");
  }
  if (options.frameLimit < 0) {
    throw AppError(ErrorCode::E_ARGUMENT, "--frames must be non-negative");
  }
  if (!std::isfinite(options.runSeconds) || options.runSeconds < 0.0) {
    throw AppError(ErrorCode::E_ARGUMENT, "--run-seconds must be finite and non-negative");
  }
  return options;
}

std::filesystem::path resolve_shaders(const Options& options, const char* executablePath) {
  if (!options.shaderDirectory.empty()) {
    return std::filesystem::absolute(options.shaderDirectory);
  }
  const std::filesystem::path executableDirectory =
      std::filesystem::absolute(executablePath).parent_path();
  const std::array<std::filesystem::path, 3> candidates = {
      executableDirectory / "shaders", std::filesystem::current_path() / "shaders",
      std::filesystem::current_path() / "assets" / "shaders"};
  for (const auto& candidate : candidates) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(candidate / "gbuffer.vert", ec) && !ec) {
      return candidate;
    }
  }
  throw AppError(ErrorCode::E_RESOURCE_MISSING,
                 "GLSL shader directory was not found next to the executable");
}

LightingState state_from_config(const LightingConfig& config, float animationTime) {
  LightingState state;
  state.lights = config.lights;
  state.materialWeights = material_layer_weights(config.material);
  state.animationTime = animationTime;
  return state;
}

void apply_reloaded_config(const ConfigStore& store, Renderer& renderer,
                           LightingState& state,
                           const std::filesystem::path& assetRoot) {
  const LightingConfig& config = store.config();
  renderer.apply_material(config.material, assetRoot);
  state.lights = config.lights;
  state.materialWeights = material_layer_weights(config.material);
  renderer.set_lighting_state(state);
  log_info("reloaded light set and material weights replaced atomically");
}

bool save_current_snapshot(const std::filesystem::path& snapshotFile,
                           const LightingState& state) {
  try {
    const Snapshot snapshot = make_snapshot(state);
    save_snapshot(snapshotFile, snapshot);
    log_info("snapshot saved: " + snapshotFile.string() + " at " + snapshot.savedAt);
    return true;
  } catch (const AppError& error) {
    log_error(error.code(), error.what());
    return false;
  }
}

bool restore_snapshot(const std::filesystem::path& snapshotFile, Renderer& renderer,
                      LightingState& state, bool& paused, float& accumulator) {
  try {
    const SnapshotLoadResult loaded = load_snapshot(snapshotFile);
    state = loaded.snapshot.state;
    renderer.set_lighting_state(state);
    paused = true;
    accumulator = 0.0F;
    log_info("snapshot restored atomically: " + snapshotFile.string() +
             " (version " + std::to_string(loaded.snapshot.formatVersion) + ", " +
             std::to_string(loaded.snapshot.state.lights.size()) + " lights); animation paused");
    return true;
  } catch (const AppError& error) {
    log_error(error.code(), error.what());
    return false;
  }
}

std::uint64_t hash_pixels(const std::vector<std::uint8_t>& pixels) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::uint8_t byte : pixels) {
    hash ^= static_cast<std::uint64_t>(byte);
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string hash_string(std::uint64_t hash) {
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << hash;
  return output.str();
}

struct ImageStats {
  float minimumLuminance = 1.0F;
  float maximumLuminance = 0.0F;
  float meanLuminance = 0.0F;
  std::uint64_t hash = 0;
  bool nonUniform = false;
};

ImageStats analyze_pixels(const std::vector<std::uint8_t>& pixels) {
  ImageStats stats;
  double luminanceSum = 0.0;
  std::uint8_t first[3] = {};
  bool sawDifferentPixel = false;
  for (std::size_t offset = 0; offset + 3U < pixels.size(); offset += 4U) {
    const float luminance =
        (0.2126F * static_cast<float>(pixels[offset]) +
         0.7152F * static_cast<float>(pixels[offset + 1U]) +
         0.0722F * static_cast<float>(pixels[offset + 2U])) /
        255.0F;
    stats.minimumLuminance = std::min(stats.minimumLuminance, luminance);
    stats.maximumLuminance = std::max(stats.maximumLuminance, luminance);
    luminanceSum += static_cast<double>(luminance);
    if (offset == 0U) {
      first[0] = pixels[0];
      first[1] = pixels[1];
      first[2] = pixels[2];
    } else if (!sawDifferentPixel &&
               (pixels[offset] != first[0] || pixels[offset + 1U] != first[1] ||
                pixels[offset + 2U] != first[2])) {
      sawDifferentPixel = true;
    }
  }
  const double pixelCount = static_cast<double>(pixels.size() / 4U);
  stats.meanLuminance =
      static_cast<float>(luminanceSum / std::max(pixelCount, 1.0));
  stats.hash = hash_pixels(pixels);
  stats.nonUniform =
      sawDifferentPixel && stats.maximumLuminance - stats.minimumLuminance > 0.03F;
  return stats;
}

void write_bmp(const std::filesystem::path& path, int width, int height,
               const std::vector<std::uint8_t>& rgba) {
  const int rowBytes = width * 3;
  const int padding = (4 - (rowBytes % 4)) % 4;
  const std::uint32_t imageSize =
      static_cast<std::uint32_t>((rowBytes + padding) * height);
  const std::uint32_t fileSize = 54U + imageSize;
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw AppError(ErrorCode::E_RESOURCE_MISSING,
                   "cannot write verification image: " + path.string());
  }
  std::array<std::uint8_t, 54> header{};
  auto put_u16 = [&](std::size_t offset, std::uint16_t value) {
    header[offset] = static_cast<std::uint8_t>(value & 0xFFU);
    header[offset + 1U] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
  };
  auto put_u32 = [&](std::size_t offset, std::uint32_t value) {
    for (std::size_t byte = 0; byte < 4U; ++byte) {
      header[offset + byte] =
          static_cast<std::uint8_t>((value >> (byte * 8U)) & 0xFFU);
    }
  };
  header[0] = 'B';
  header[1] = 'M';
  put_u32(2U, fileSize);
  put_u32(10U, 54U);
  put_u32(14U, 40U);
  put_u32(18U, static_cast<std::uint32_t>(width));
  put_u32(22U, static_cast<std::uint32_t>(height));
  put_u16(26U, 1U);
  put_u16(28U, 24U);
  put_u32(30U, 0U);
  put_u32(34U, imageSize);
  put_u32(38U, 2835U);
  put_u32(42U, 2835U);
  output.write(reinterpret_cast<const char*>(header.data()), 54);
  const std::array<std::uint8_t, 3> zero = {0, 0, 0};
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t source =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x)) *
          4U;
      const std::array<std::uint8_t, 3> bgr = {rgba[source + 2U], rgba[source + 1U],
                                               rgba[source]};
      output.write(reinterpret_cast<const char*>(bgr.data()), 3);
    }
    for (int pad = 0; pad < padding; ++pad) {
      output.write(reinterpret_cast<const char*>(zero.data()), 1);
    }
  }
  if (!output) {
    throw AppError(ErrorCode::E_RESOURCE_MISSING,
                   "verification image write failed: " + path.string());
  }
}

std::uint32_t crc32_bytes(const std::vector<std::uint8_t>& bytes,
                          std::uint32_t crc = 0xFFFFFFFFU) {
  for (std::uint8_t byte : bytes) {
    crc ^= static_cast<std::uint32_t>(byte);
    for (int bit = 0; bit < 8; ++bit) {
      const std::uint32_t mask = 0U - (crc & 1U);
      crc = (crc >> 1U) ^ (0xEDB88320U & mask);
    }
  }
  return crc;
}

void append_u32_be(std::vector<std::uint8_t>& output, std::uint32_t value) {
  output.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
  output.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
  output.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
  output.push_back(static_cast<std::uint8_t>(value & 0xFFU));
}

void append_png_chunk(std::vector<std::uint8_t>& png, const std::array<char, 4>& type,
                      const std::vector<std::uint8_t>& payload) {
  append_u32_be(png, static_cast<std::uint32_t>(payload.size()));
  const std::size_t crcStart = png.size();
  for (char character : type) {
    png.push_back(static_cast<std::uint8_t>(character));
  }
  png.insert(png.end(), payload.begin(), payload.end());
  const std::uint32_t crc = crc32_bytes(
      std::vector<std::uint8_t>(png.begin() + static_cast<std::ptrdiff_t>(crcStart),
                                png.end())) ^
                            0xFFFFFFFFU;
  append_u32_be(png, crc);
}

void write_png(const std::filesystem::path& path, int width, int height,
               const std::vector<std::uint8_t>& rgba) {
  std::vector<std::uint8_t> raw;
  raw.reserve(static_cast<std::size_t>(height) *
              (static_cast<std::size_t>(width) * 3U + 1U));
  for (int y = 0; y < height; ++y) {
    const int sourceY = height - 1 - y;
    raw.push_back(0U);
    for (int x = 0; x < width; ++x) {
      const std::size_t source =
          (static_cast<std::size_t>(sourceY) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x)) *
          4U;
      raw.push_back(rgba[source]);
      raw.push_back(rgba[source + 1U]);
      raw.push_back(rgba[source + 2U]);
    }
  }

  std::vector<std::uint8_t> zlibBytes = {0x78U, 0x01U};
  std::size_t offset = 0;
  while (offset < raw.size()) {
    const std::size_t blockSize = std::min<std::size_t>(65535U, raw.size() - offset);
    const bool finalBlock = offset + blockSize == raw.size();
    zlibBytes.push_back(finalBlock ? 1U : 0U);
    const auto length = static_cast<std::uint16_t>(blockSize);
    const auto inverseLength = static_cast<std::uint16_t>(~length);
    zlibBytes.push_back(static_cast<std::uint8_t>(length & 0xFFU));
    zlibBytes.push_back(static_cast<std::uint8_t>((length >> 8U) & 0xFFU));
    zlibBytes.push_back(static_cast<std::uint8_t>(inverseLength & 0xFFU));
    zlibBytes.push_back(static_cast<std::uint8_t>((inverseLength >> 8U) & 0xFFU));
    zlibBytes.insert(zlibBytes.end(),
                     raw.begin() + static_cast<std::ptrdiff_t>(offset),
                     raw.begin() + static_cast<std::ptrdiff_t>(offset + blockSize));
    offset += blockSize;
  }
  std::uint32_t adlerA = 1U;
  std::uint32_t adlerB = 0U;
  for (std::uint8_t byte : raw) {
    adlerA = (adlerA + static_cast<std::uint32_t>(byte)) % 65521U;
    adlerB = (adlerB + adlerA) % 65521U;
  }
  append_u32_be(zlibBytes, (adlerB << 16U) | adlerA);

  std::vector<std::uint8_t> png = {0x89U, 0x50U, 0x4EU, 0x47U,
                                   0x0DU, 0x0AU, 0x1AU, 0x0AU};
  std::vector<std::uint8_t> ihdr;
  append_u32_be(ihdr, static_cast<std::uint32_t>(width));
  append_u32_be(ihdr, static_cast<std::uint32_t>(height));
  ihdr.insert(ihdr.end(), {8U, 2U, 0U, 0U, 0U});
  append_png_chunk(png, {'I', 'H', 'D', 'R'}, ihdr);
  append_png_chunk(png, {'I', 'D', 'A', 'T'}, zlibBytes);
  append_png_chunk(png, {'I', 'E', 'N', 'D'}, {});

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw AppError(ErrorCode::E_RESOURCE_MISSING,
                   "cannot write verification PNG: " + path.string());
  }
  output.write(reinterpret_cast<const char*>(png.data()),
               static_cast<std::streamsize>(png.size()));
  if (!output) {
    throw AppError(ErrorCode::E_RESOURCE_MISSING,
                   "verification PNG write failed: " + path.string());
  }
}

void write_verification_image(const std::filesystem::path& directory,
                              const std::string& stem, int width, int height,
                              const std::vector<std::uint8_t>& pixels) {
  write_bmp(directory / (stem + ".bmp"), width, height, pixels);
  write_png(directory / (stem + ".png"), width, height, pixels);
}

bool run_verification(Renderer& renderer, ConfigStore& store, LightingState state,
                      const std::filesystem::path& snapshotFile,
                      const std::filesystem::path& artifactDirectory) {
  using json = nlohmann::json;
  std::error_code ec;
  std::filesystem::create_directories(artifactDirectory, ec);
  if (ec) {
    throw AppError(ErrorCode::E_RESOURCE_MISSING,
                   "cannot create verification artifact directory");
  }

  json report;
  report["verificationVersion"] = 1;
  report["generatedAt"] = utc_timestamp_now();
  report["openGLVersion"] = gl_string(gl::GL_VERSION);
  report["glslVersion"] = gl_string(gl::GL_SHADING_LANGUAGE_VERSION);
  report["renderer"] = gl_string(gl::GL_RENDERER);
  report["pathTrace"] = {
      {"deferred", true},
      {"gbufferAttachments", 5},
      {"brdfLayers", {"diffuse", "ggx_specular", "cloth_sheen"}},
      {"lightCount", state.lights.size()},
  };
  json checks = json::array();
  bool allPassed = true;

  auto add_check = [&](const std::string& name, bool passed, json details) {
    checks.push_back({{"name", name}, {"passed", passed}, {"details", std::move(details)}});
    allPassed = allPassed && passed;
  };

  bool hasPoint = false;
  bool hasDirectional = false;
  bool hasSpot = false;
  for (const Light& light : state.lights) {
    hasPoint = hasPoint || light.type == LightType::Point;
    hasDirectional = hasDirectional || light.type == LightType::Directional;
    hasSpot = hasSpot || light.type == LightType::Spot;
  }
  add_check("all_light_types_present", hasPoint && hasDirectional && hasSpot,
            {{"point", hasPoint}, {"directional", hasDirectional}, {"spot", hasSpot}});

  SurfaceSample sample;
  const glm::vec3 radiance = evaluate_layered_brdf(
      sample, glm::normalize(glm::vec3(-0.3F, 0.8F, 0.6F)), glm::vec3(1.0F));
  auto finite = [](const glm::vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
  };
  add_check("cpu_layered_brdf_finite",
            finite(radiance) && glm::length(radiance) > 0.0F,
            {{"result", {radiance.x, radiance.y, radiance.z}}});

  state.animationTime = 0.420F;
  renderer.set_lighting_state(state);
  renderer.render();
  const std::vector<std::uint8_t> savedPixels = renderer.read_pixels();
  const ImageStats savedStats = analyze_pixels(savedPixels);
  write_verification_image(artifactDirectory, "snapshot_saved_frame", renderer.width(),
                           renderer.height(), savedPixels);
  add_check("rendered_character_visible", savedStats.nonUniform &&
                savedStats.meanLuminance > 0.018F,
            {{"meanLuminance", savedStats.meanLuminance},
             {"minLuminance", savedStats.minimumLuminance},
             {"maxLuminance", savedStats.maximumLuminance},
             {"hash", hash_string(savedStats.hash)}});

  const bool snapshotSaved = save_current_snapshot(snapshotFile, state);
  add_check("snapshot_saved", snapshotSaved, {{"path", snapshotFile.string()}});
  if (snapshotSaved) {
    state.materialWeights = glm::vec3(0.22F, 0.53F, 0.25F);
    state.animationTime = 0.91F;
    for (Light& light : state.lights) {
      light.color *= glm::vec3(0.55F, 1.4F, 0.72F);
    }
    renderer.set_lighting_state(state);
    renderer.render();
    const std::vector<std::uint8_t> mutatedPixels = renderer.read_pixels();
    const ImageStats mutatedStats = analyze_pixels(mutatedPixels);
    write_verification_image(artifactDirectory, "snapshot_mutated_frame",
                             renderer.width(), renderer.height(), mutatedPixels);
    add_check("mutation_changes_pixels", mutatedStats.hash != savedStats.hash,
              {{"mutatedHash", hash_string(mutatedStats.hash)}});

    bool restoredPause = false;
    float restoredAccumulator = 0.0F;
    const bool restored = restore_snapshot(snapshotFile, renderer, state,
                                           restoredPause, restoredAccumulator);
    (void)restored;
    (void)restoredPause;
    (void)restoredAccumulator;
    renderer.render();
    const std::vector<std::uint8_t> restoredPixels = renderer.read_pixels();
    const ImageStats restoredStats = analyze_pixels(restoredPixels);
    write_verification_image(artifactDirectory, "snapshot_restored_frame",
                             renderer.width(), renderer.height(), restoredPixels);
    const bool exactParity = restoredPixels == savedPixels;
    size_t differingBytes = 0;
    if (!exactParity && restoredPixels.size() == savedPixels.size()) {
      for (std::size_t index = 0; index < savedPixels.size(); ++index) {
        differingBytes += savedPixels[index] != restoredPixels[index] ? 1U : 0U;
      }
    }
    add_check("snapshot_restore_pixel_parity", exactParity,
              {{"savedHash", hash_string(savedStats.hash)},
               {"restoredHash", hash_string(restoredStats.hash)},
               {"differingBytes", differingBytes},
               {"pixelCount", renderer.width() * renderer.height()}});
  } else {
    allPassed = false;
  }

  const std::array<glm::vec3, 3> layerWeights = {
      glm::vec3(1.0F, 0.0F, 0.0F), glm::vec3(0.0F, 1.0F, 0.0F),
      glm::vec3(0.0F, 0.0F, 1.0F)};
  const std::array<const char*, 3> layerNames = {"diffuse", "ggx_specular", "cloth_sheen"};
  std::array<ImageStats, 3> layerStats{};
  std::array<std::vector<std::uint8_t>, 3> layerPixels{};
  LightingState layerState = state_from_config(store.config(), 0.33F);
  for (std::size_t index = 0; index < layerWeights.size(); ++index) {
    layerState.materialWeights = layerWeights[index];
    renderer.set_lighting_state(layerState);
    renderer.render();
    layerPixels[index] = renderer.read_pixels();
    layerStats[index] = analyze_pixels(layerPixels[index]);
    write_verification_image(artifactDirectory,
                             std::string("layer_") + layerNames[index], renderer.width(),
                             renderer.height(), layerPixels[index]);
  }
  const bool layerImagesDiffer =
      layerStats[0].hash != layerStats[1].hash && layerStats[1].hash != layerStats[2].hash &&
      layerStats[0].hash != layerStats[2].hash;
  const bool layerImagesNonUniform =
      layerStats[0].nonUniform && layerStats[1].nonUniform && layerStats[2].nonUniform;
  add_check("three_layer_brdf_render_distinct", layerImagesDiffer && layerImagesNonUniform,
            {{"diffuseHash", hash_string(layerStats[0].hash)},
             {"ggxHash", hash_string(layerStats[1].hash)},
             {"clothHash", hash_string(layerStats[2].hash)},
             {"diffuseMean", layerStats[0].meanLuminance},
             {"ggxMean", layerStats[1].meanLuminance},
             {"clothMean", layerStats[2].meanLuminance}});

  report["checks"] = std::move(checks);
  report["allPassed"] = allPassed;
  report["snapshotPath"] = snapshotFile.string();
  std::ofstream output(artifactDirectory / "verification.json", std::ios::trunc);
  output << report.dump(2) << '\n';
  if (!output) {
    throw AppError(ErrorCode::E_RESOURCE_MISSING, "cannot write verification report");
  }
  std::cout << report.dump(2) << '\n';
  return allPassed;
}

int run_application(const Options& options, const char* executablePath) {
  std::error_code ec;
  const std::filesystem::path dataDirectory = std::filesystem::absolute(options.dataDirectory);
  std::filesystem::create_directories(dataDirectory, ec);
  if (ec) {
    throw AppError(ErrorCode::E_RESOURCE_MISSING,
                   "cannot create runtime data directory: " + dataDirectory.string());
  }
  Logger::instance().open(dataDirectory / "logs" / "lf_engine.log");
  log_info("process start; data directory: " + dataDirectory.string());

  const std::filesystem::path configFile = dataDirectory / "lighting.json";
  const std::filesystem::path snapshotFile = dataDirectory / "snapshots" / "light_snapshot.json";
  ensure_samples(dataDirectory, configFile);

  ConfigStore configStore(configFile, dataDirectory);
  try {
    configStore.load_initial();
  } catch (const AppError& error) {
    log_error(error.code(), error.what());
    if (error.code() != ErrorCode::E_CONFIG_INVALID) {
      throw;
    }
    configStore.adopt_fallback(make_default_config());
  }

  const std::filesystem::path shaderDirectory = resolve_shaders(options, executablePath);
  glfwSetErrorCallback(glfw_error_callback);
  GlfwGuard glfwGuard;
  if (glfwInit() != GLFW_TRUE) {
    throw AppError(ErrorCode::E_RENDER_WINDOW_CREATE,
                   "GLFW initialization failed; an OpenGL-capable desktop is required");
  }
  glfwGuard.initialized = true;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  glfwWindowHint(GLFW_VISIBLE, options.verify ? GLFW_FALSE : GLFW_TRUE);
  WindowGuard windowGuard;
  windowGuard.window = glfwCreateWindow(options.width, options.height,
                                        "LF Layered Character Engine", nullptr, nullptr);
  if (windowGuard.window == nullptr) {
    throw AppError(ErrorCode::E_RENDER_WINDOW_CREATE,
                   "window or OpenGL 4.3 context creation failed");
  }
  glfwMakeContextCurrent(windowGuard.window);
  glfwSwapInterval(1);

  std::string missingEntryPoint;
  if (!gl::load_api(missingEntryPoint)) {
    throw AppError(ErrorCode::E_RENDER_GL_LOAD,
                   "missing OpenGL entry point: " + missingEntryPoint);
  }
  glfwSetWindowUserPointer(windowGuard.window, nullptr);

  int framebufferWidth = options.width;
  int framebufferHeight = options.height;
  glfwGetFramebufferSize(windowGuard.window, &framebufferWidth, &framebufferHeight);
  framebufferWidth = std::max(framebufferWidth, 64);
  framebufferHeight = std::max(framebufferHeight, 64);
  auto renderer =
      std::make_unique<Renderer>(shaderDirectory, framebufferWidth, framebufferHeight);
  renderer->initialize(CharacterSettings{});
  const LightingConfig& initialConfig = configStore.config();
  renderer->apply_material(initialConfig.material, dataDirectory);
  LightingState state = state_from_config(initialConfig, 0.0F);
  renderer->set_lighting_state(state);
  log_info("renderer initialized through deferred G-Buffer pipeline");

  if (options.verify) {
    renderer->resize(framebufferWidth, framebufferHeight);
    const bool passed =
        run_verification(*renderer, configStore, state, snapshotFile,
                         dataDirectory / "artifacts");
    renderer.reset();
    glfwDestroyWindow(windowGuard.window);
    windowGuard.window = nullptr;
    Logger::instance().close();
    return passed ? 0 : 1;
  }

  InputActions actions;
  glfwSetWindowUserPointer(windowGuard.window, &actions);
  glfwSetKeyCallback(windowGuard.window, key_callback);
  glfwSetWindowTitle(windowGuard.window,
                     "LF Layered Character | F5 Save  F9 Restore  F6 Pause  R Reload");
  constexpr float kFixedStep = 1.0F / 60.0F;
  float accumulator = 0.0F;
  bool paused = false;
  bool pendingRedraw = true;
  int renderedFrames = 0;
  const double interactiveStart = glfwGetTime();
  double previousTime = glfwGetTime();
  double lastReloadPoll = previousTime;

  while (glfwWindowShouldClose(windowGuard.window) == GLFW_FALSE) {
    glfwPollEvents();

    if (actions.restoreSnapshot) {
      actions.restoreSnapshot = false;
      restore_snapshot(snapshotFile, *renderer, state, paused, accumulator);
      pendingRedraw = true;
    }
    if (actions.togglePause) {
      actions.togglePause = false;
      paused = !paused;
      accumulator = 0.0F;
      log_info(paused ? "animation paused" : "animation resumed");
    }
    if (actions.reloadConfig) {
      actions.reloadConfig = false;
      const ReloadOutcome outcome = configStore.reload_now();
      if (outcome.status == ReloadStatus::Reloaded) {
        apply_reloaded_config(configStore, *renderer, state, dataDirectory);
        pendingRedraw = true;
      }
    }

    const double now = glfwGetTime();
    if (now - lastReloadPoll >= 0.35) {
      lastReloadPoll = now;
      const ReloadOutcome outcome = configStore.reload_if_changed();
      if (outcome.status == ReloadStatus::Reloaded) {
        apply_reloaded_config(configStore, *renderer, state, dataDirectory);
        pendingRedraw = true;
      }
    }

    int windowWidth = 0;
    int windowHeight = 0;
    glfwGetFramebufferSize(windowGuard.window, &windowWidth, &windowHeight);
    if (windowWidth > 0 && windowHeight > 0) {
      renderer->resize(windowWidth, windowHeight);
    }

    const float frameTime =
        std::min(static_cast<float>(std::max(now - previousTime, 0.0)), 0.10F);
    previousTime = now;
    if (!paused) {
      accumulator += frameTime;
      int steps = 0;
      while (accumulator >= kFixedStep && steps < 5) {
        state.animationTime += kFixedStep;
        accumulator -= kFixedStep;
        ++steps;
      }
      if (steps == 5) {
        accumulator = 0.0F;
      }
      pendingRedraw = true;
    }

    if (pendingRedraw && windowWidth > 0 && windowHeight > 0) {
      renderer->set_lighting_state(state);
      renderer->render();
      glfwSwapBuffers(windowGuard.window);
      pendingRedraw = false;
      ++renderedFrames;
      if (actions.saveSnapshot) {
        actions.saveSnapshot = false;
        save_current_snapshot(snapshotFile, state);
      }
    }

    if (actions.saveSnapshot && paused && windowWidth > 0 && windowHeight > 0) {
      actions.saveSnapshot = false;
      renderer->set_lighting_state(state);
      renderer->render();
      glfwSwapBuffers(windowGuard.window);
      save_current_snapshot(snapshotFile, state);
    }
    if (options.frameLimit > 0 && renderedFrames >= options.frameLimit) {
      glfwSetWindowShouldClose(windowGuard.window, GLFW_TRUE);
    }
    if (options.runSeconds > 0.0 &&
        glfwGetTime() - interactiveStart >= options.runSeconds) {
      glfwSetWindowShouldClose(windowGuard.window, GLFW_TRUE);
    }
  }

  log_info("window close requested; shutting down");
  renderer.reset();
  glfwDestroyWindow(windowGuard.window);
  windowGuard.window = nullptr;
  Logger::instance().close();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    return run_application(options, argv[0]);
  } catch (const AppError& error) {
    if (Logger::instance().file_path().empty()) {
      Logger::instance().open("lf_engine_error.log");
    }
    log_error(error.code(), error.what());
    Logger::instance().close();
    return error_code_number(error.code());
  } catch (const std::exception& error) {
    if (Logger::instance().file_path().empty()) {
      Logger::instance().open("lf_engine_error.log");
    }
    log_error(ErrorCode::E_INTERNAL, error.what());
    Logger::instance().close();
    return error_code_number(ErrorCode::E_INTERNAL);
  }
}

