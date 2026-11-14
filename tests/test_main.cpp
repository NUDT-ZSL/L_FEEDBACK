#include "lf/brdf.h"
#include "lf/config_store.h"
#include "lf/error_code.h"
#include "lf/log.h"
#include "lf/mesh.h"
#include "lf/samples.h"
#include "lf/snapshot.h"

#include <nlohmann/json.hpp>

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace lf;
using json = nlohmann::json;

int g_checks = 0;
int g_failures = 0;
std::string g_currentTest;

void start_test(const std::string& name) {
  g_currentTest = name;
  std::cout << "[ RUN      ] " << name << '\n';
}

void check(bool condition, const std::string& message) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::cerr << "[  FAILED  ] " << g_currentTest << ": " << message << '\n';
  }
}

template <typename T>
void check_equal(const T& actual, const T& expected, const std::string& message) {
  ++g_checks;
  if (!(actual == expected)) {
    ++g_failures;
    std::cerr << "[  FAILED  ] " << g_currentTest << ": " << message << '\n';
  }
}

void end_test() { std::cout << "[       OK ] " << g_currentTest << '\n'; }

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(input)),
                                   std::istreambuf_iterator<char>());
}

void write_text(const std::filesystem::path& path, const std::string& content) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output << content;
}

ErrorCode capture_code(const std::function<void()>& action) {
  try {
    action();
  } catch (const AppError& error) {
    return error.code();
  } catch (...) {
    return ErrorCode::E_INTERNAL;
  }
  return ErrorCode::Ok;
}

void test_error_codes() {
  start_test("business_error_codes");
  check_equal(std::string(error_code_name(ErrorCode::E_RENDER_WINDOW_CREATE)),
              std::string("E_RENDER_WINDOW_CREATE"), "window error name");
  check_equal(std::string(error_code_name(ErrorCode::E_SHADER_COMPILE)),
              std::string("E_SHADER_COMPILE"), "shader error name");
  check_equal(std::string(error_code_name(ErrorCode::E_TEXTURE_LOAD)),
              std::string("E_TEXTURE_LOAD"), "texture error name");
  check_equal(std::string(error_code_name(ErrorCode::E_SNAPSHOT_CORRUPT)),
              std::string("E_SNAPSHOT_CORRUPT"), "snapshot error name");
  check_equal(std::string(error_code_name(ErrorCode::E_RESOURCE_MISSING)),
              std::string("E_RESOURCE_MISSING"), "resource error name");
  check_equal(error_code_number(ErrorCode::E_SNAPSHOT_VERSION_UNSUPPORTED), 1202,
              "version unsupported numeric code");
  end_test();
}

void test_deterministic_samples(const std::filesystem::path& scratch) {
  start_test("deterministic_sample_generation");
  const std::filesystem::path first = scratch / "samples_a";
  const std::filesystem::path second = scratch / "samples_b";
  const std::filesystem::path firstConfig = first / "lighting.json";
  const std::filesystem::path secondConfig = second / "lighting.json";

  const SampleGenerationResult generatedA = generate_samples(first, firstConfig, true);
  const SampleGenerationResult generatedB = generate_samples(second, secondConfig, true);
  check(generatedA.texturesGenerated && generatedB.texturesGenerated,
        "texture generation reported");
  check(generatedA.configGenerated && generatedB.configGenerated, "config generation reported");

  const std::vector<std::string> files = expected_texture_files();
  check_equal(files.size(), std::size_t{12}, "twelve layer textures are generated");
  for (const std::string& relative : files) {
    const std::vector<std::uint8_t> bytesA = read_bytes(first / relative);
    const std::vector<std::uint8_t> bytesB = read_bytes(second / relative);
    check_equal(bytesA.size(), std::size_t{18 + kSampleTextureSize * kSampleTextureSize * 4},
                "TGA payload size for " + relative);
    check(bytesA == bytesB, "deterministic bytes for " + relative);
  }
  const std::vector<std::uint8_t> configA = read_bytes(firstConfig);
  const std::vector<std::uint8_t> configB = read_bytes(secondConfig);
  check(!configA.empty(), "default config is not empty");
  check(configA == configB, "default config bytes are deterministic");

  generate_samples(first, firstConfig, true);
  check(read_bytes(firstConfig) == configA, "re-generating config is byte-identical");
  check(read_bytes(first / files[0]) == read_bytes(second / files[0]),
        "re-generating textures is byte-identical");

  const LightingConfig config = ConfigLoader(first).load(firstConfig);
  check_equal(config.material.layers.size(), std::size_t{3}, "three material layers");
  check_equal(std::string(brdf_type_name(config.material.layers[0].brdf)),
              std::string("diffuse"), "layer 0 diffuse");
  check_equal(std::string(brdf_type_name(config.material.layers[1].brdf)),
              std::string("ggx_specular"), "layer 1 GGX");
  check_equal(std::string(brdf_type_name(config.material.layers[2].brdf)),
              std::string("cloth_sheen"), "layer 2 cloth");
  check_equal(config.lights.size(), std::size_t{3}, "three generated lights");
  end_test();
}

void test_config_validation(const std::filesystem::path& samplesRoot,
                            const std::filesystem::path& scratch) {
  start_test("config_validation_error_codes");
  const std::filesystem::path validConfig = samplesRoot / "lighting.json";
  const LightingConfig valid = ConfigLoader(samplesRoot).load(validConfig);
  check_equal(valid.formatVersion, 1, "config format version");

  const std::filesystem::path bad = scratch / "bad_config.json";
  const std::string original = [&] {
    const std::vector<std::uint8_t> bytes = read_bytes(validConfig);
    return std::string(bytes.begin(), bytes.end());
  }();
  const ConfigLoader loader(samplesRoot);

  json document = json::parse(original);
  document["formatVersion"] = 1.5;
  write_text(bad, document.dump(2));
  check_equal(capture_code([&] { (void)loader.load(bad); }), ErrorCode::E_CONFIG_INVALID,
              "non-integer config version rejected");

  document = json::parse(original);
  document["material"]["layers"][0]["weight"] = 1.75;
  write_text(bad, document.dump(2));
  check_equal(capture_code([&] { (void)loader.load(bad); }), ErrorCode::E_CONFIG_INVALID,
              "weight range rejected");

  document = json::parse(original);
  document["lights"][1]["id"] = document["lights"][0]["id"];
  write_text(bad, document.dump(2));
  check_equal(capture_code([&] { (void)loader.load(bad); }), ErrorCode::E_CONFIG_INVALID,
              "duplicate light id rejected");

  document = json::parse(original);
  document["material"]["layers"][2]["normalTexture"] = "textures/does_not_exist.tga";
  write_text(bad, document.dump(2));
  check_equal(capture_code([&] { (void)loader.load(bad); }), ErrorCode::E_CONFIG_INVALID,
              "missing texture reference rejected");

  document = json::parse(original);
  document["material"]["layers"][1]["type"] = "cloth_sheen";
  write_text(bad, document.dump(2));
  check_equal(capture_code([&] { (void)loader.load(bad); }), ErrorCode::E_CONFIG_INVALID,
              "non-fixed BRDF layer order rejected");

  document = json::parse(original);
  for (json& light : document["lights"]) {
    if (light["type"] == "spot") {
      light["innerConeDegrees"] = 44.0;
      light["outerConeDegrees"] = 31.0;
    }
  }
  write_text(bad, document.dump(2));
  check_equal(capture_code([&] { (void)loader.load(bad); }), ErrorCode::E_CONFIG_INVALID,
              "invalid spot cone range rejected");

  document = json::parse(original);
  document["lights"][0]["intensity"] = -3.0;
  write_text(bad, document.dump(2));
  check_equal(capture_code([&] { (void)loader.load(bad); }), ErrorCode::E_CONFIG_INVALID,
              "negative intensity rejected");

  document = json::parse(original);
  document["lights"][0]["position"] = {1.0, 2.0};
  write_text(bad, document.dump(2));
  check_equal(capture_code([&] { (void)loader.load(bad); }), ErrorCode::E_CONFIG_INVALID,
              "malformed vector rejected");
  end_test();
}

void test_hot_reload_retention(const std::filesystem::path& samplesRoot,
                               const std::filesystem::path& scratch) {
  start_test("config_hot_reload_atomic_retention");
  const std::filesystem::path working = scratch / "reload";
  std::filesystem::create_directories(working);
  const std::filesystem::path configPath = working / "lighting.json";
  const std::vector<std::uint8_t> source = read_bytes(samplesRoot / "lighting.json");
  const std::string validText(source.begin(), source.end());
  write_text(configPath, validText);

  ConfigStore store(configPath, samplesRoot);
  store.load_initial();
  const LightingConfig before = store.config();
  const std::string beforeFirstId = before.lights.front().id;
  const float beforeWeight = before.material.layers[0].weight;

  json invalid = json::parse(validText);
  invalid["lights"][0]["color"] = {2.0, -1.0, 0.0};
  write_text(configPath, invalid.dump(2));
  const ReloadOutcome invalidOutcome = store.reload_now();
  check_equal(invalidOutcome.status, ReloadStatus::Invalid, "invalid reload status");
  check_equal(invalidOutcome.code, ErrorCode::E_CONFIG_RELOAD_INVALID,
              "invalid reload business code");
  check_equal(store.config().lights.front().id, beforeFirstId,
              "old light set retained after invalid reload");
  check_equal(store.config().material.layers[0].weight, beforeWeight,
              "old material retained after invalid reload");

  json valid = json::parse(validText);
  valid["material"]["layers"][0]["weight"] = 0.61;
  valid["material"]["layers"][1]["weight"] = 0.25;
  valid["material"]["layers"][2]["weight"] = 0.14;
  valid["lights"][0]["intensity"] = 51.5;
  write_text(configPath, valid.dump(2));
  const ReloadOutcome validOutcome = store.reload_now();
  check_equal(validOutcome.status, ReloadStatus::Reloaded, "valid reload status");
  check_equal(store.config().lights.front().intensity, 51.5F, "new light applied");
  check_equal(store.config().material.layers[0].weight, 0.61F, "new material weight applied");
  end_test();
}

bool same_state(const LightingState& left, const LightingState& right) {
  if (left.lights.size() != right.lights.size() ||
      glm::any(glm::notEqual(left.materialWeights, right.materialWeights)) ||
      left.animationTime != right.animationTime) {
    return false;
  }
  for (std::size_t index = 0; index < left.lights.size(); ++index) {
    const Light& a = left.lights[index];
    const Light& b = right.lights[index];
    if (a.id != b.id || a.type != b.type || a.enabled != b.enabled ||
        glm::any(glm::notEqual(a.position, b.position)) ||
        glm::any(glm::notEqual(a.direction, b.direction)) ||
        glm::any(glm::notEqual(a.color, b.color)) ||
        a.intensity != b.intensity || a.range != b.range ||
        a.innerConeDegrees != b.innerConeDegrees ||
        a.outerConeDegrees != b.outerConeDegrees ||
        a.constantAttenuation != b.constantAttenuation ||
        a.linearAttenuation != b.linearAttenuation ||
        a.quadraticAttenuation != b.quadraticAttenuation) {
      return false;
    }
  }
  return true;
}

void test_snapshots(const std::filesystem::path& samplesRoot,
                    const std::filesystem::path& scratch) {
  start_test("snapshot_roundtrip_migration_and_rejection");
  const LightingConfig config = ConfigLoader(samplesRoot).load(samplesRoot / "lighting.json");
  LightingState state;
  state.lights = config.lights;
  state.materialWeights = material_layer_weights(config.material);
  state.animationTime = 0.625F;

  const std::filesystem::path snapshotPath = scratch / "snapshots" / "state.json";
  const Snapshot snapshot = make_snapshot(state);
  save_snapshot(snapshotPath, snapshot);
  check(std::filesystem::is_regular_file(snapshotPath), "snapshot file created");

  const json saved = json::parse([&] {
    const auto bytes = read_bytes(snapshotPath);
    return std::string(bytes.begin(), bytes.end());
  }());
  check_equal(saved.at("formatVersion").get<int>(), kSnapshotFormatVersion,
              "snapshot current format version");
  check(!saved.at("savedAt").get<std::string>().empty(), "snapshot timestamp persisted");

  const SnapshotLoadResult loaded = load_snapshot(snapshotPath);
  check_equal(loaded.snapshot.formatVersion, kSnapshotFormatVersion,
              "loaded current snapshot version");
  check(same_state(loaded.snapshot.state, state), "snapshot state round-trips exactly");
  check(loaded.migrationLog.empty(), "current snapshot requires no migration");

  const std::filesystem::path unsupported = scratch / "snapshots" / "newer.json";
  write_text(unsupported,
             R"({"formatVersion":99,"savedAt":"2030-01-01T00:00:00Z"})");
  check_equal(capture_code([&] { (void)load_snapshot(unsupported); }),
              ErrorCode::E_SNAPSHOT_VERSION_UNSUPPORTED,
              "newer snapshot version rejected with dedicated code");

  const std::filesystem::path corrupt = scratch / "snapshots" / "corrupt.json";
  write_text(corrupt, R"({"formatVersion":2,"savedAt":)");
  check_equal(capture_code([&] { (void)load_snapshot(corrupt); }),
              ErrorCode::E_SNAPSHOT_CORRUPT, "truncated JSON is corrupt");

  write_text(corrupt,
             R"({"formatVersion":2,"savedAt":"2026-01-01T00:00:00Z","materialWeights":[1,0,0],"renderState":{"animationTime":0},"lights":[]})");
  check_equal(capture_code([&] { (void)load_snapshot(corrupt); }),
              ErrorCode::E_SNAPSHOT_CORRUPT, "empty light set is corrupt");

  write_text(corrupt,
             R"({"formatVersion":2,"savedAt":"yesterday","materialWeights":[1,0,0],"renderState":{"animationTime":0},"lights":[]})");
  check_equal(capture_code([&] { (void)load_snapshot(corrupt); }),
              ErrorCode::E_SNAPSHOT_CORRUPT, "invalid timestamp is corrupt");

  const std::filesystem::path versionOne = scratch / "snapshots" / "version1.json";
  const json v1 = {
      {"formatVersion", 1},
      {"savedAt", "2024-05-01T12:00:00Z"},
      {"lights",
       json::array({{
           {"id", "legacy_point"},
           {"type", "point"},
           {"position", {1.0, 2.0, 3.0}},
           {"color", {1.0, 0.5, 0.25}},
           {"intensity", 20.0},
       }})},
  };
  write_text(versionOne, v1.dump(2));
  const SnapshotLoadResult migrated = load_snapshot(versionOne);
  check_equal(migrated.snapshot.formatVersion, kSnapshotFormatVersion,
              "v1 migrated to current");
  check(!migrated.migrationLog.empty(), "v1 migration log is populated");
  check_equal(migrated.snapshot.state.materialWeights,
              glm::vec3(0.55F, 0.28F, 0.17F), "v1 default material weights added");
  check_equal(migrated.snapshot.state.animationTime, 0.0F, "v1 animation time defaulted");
  check_equal(migrated.snapshot.state.lights.front().range, 10.0F,
              "v1 point range defaulted");
  check_equal(migrated.snapshot.state.lights.front().constantAttenuation, 1.0F,
              "v1 attenuation defaulted");
  check(migrated.snapshot.state.lights.front().enabled, "v1 enabled flag defaulted");

  LightingState replaced = state;
  replaced.lights.clear();
  replaced.materialWeights = glm::vec3(0.1F, 0.2F, 0.7F);
  replaced.animationTime = 9.0F;
  replaced = load_snapshot(snapshotPath).snapshot.state;
  check(same_state(replaced, state), "restore replaces state without blending");
  end_test();
}

void test_mesh_and_skin() {
  start_test("procedural_skinned_capsule_mesh");
  const SkinnedMesh mesh = generate_character_mesh(CharacterSettings{});
  check(mesh.vertices.size() > 1000U, "capsule character has substantial geometry");
  check(!mesh.indices.empty() && mesh.indices.size() % 3U == 0U,
        "index buffer contains triangles");
  check_equal(mesh.bones.size(), std::size_t{kBoneCount}, "bone count");

  std::vector<glm::vec3> positions;
  positions.reserve(mesh.vertices.size());
  for (const SkinnedVertex& vertex : mesh.vertices) {
    positions.push_back(vertex.position);
    const float weightSum = vertex.weights.x + vertex.weights.y + vertex.weights.z +
                            vertex.weights.w;
    check(std::abs(weightSum - 1.0F) <= 1.0e-4F, "skin weights normalized");
    check(vertex.joints.x >= 0 && vertex.joints.x < kBoneCount &&
              vertex.joints.y >= 0 && vertex.joints.y < kBoneCount,
          "joint indices valid");
    check(std::isfinite(vertex.normal.x) && std::isfinite(vertex.normal.y) &&
              std::isfinite(vertex.normal.z),
          "vertex normals finite");
  }
  const auto [minIt, maxIt] =
      std::minmax_element(positions.begin(), positions.end(),
                          [](const glm::vec3& left, const glm::vec3& right) {
                            return left.y < right.y;
                          });
  check(maxIt->y - minIt->y > 1.6F, "generated character spans configured height");

  const auto& keyframes = built_in_pose_keyframes();
  check_equal(keyframes.size(), std::size_t{2}, "two built-in pose keyframes");
  check_equal(std::string(keyframes[0].name), std::string("rest"), "rest keyframe");
  check_equal(std::string(keyframes[1].name), std::string("walk"), "walk keyframe");
  bool poseDifference = false;
  for (const glm::vec3& rotation : keyframes[1].localEulerRadians) {
    poseDifference = poseDifference || glm::length(rotation) > 0.01F;
  }
  check(poseDifference, "walk keyframe contains rotation data");

  const auto restMatrices = compute_skin_matrices(mesh, 0.0F);
  const auto walkMatrices = compute_skin_matrices(mesh, 0.6F);
  bool matricesDiffer = false;
  for (std::size_t index = 0; index < restMatrices.size(); ++index) {
    matricesDiffer = matricesDiffer || restMatrices[index] != walkMatrices[index];
    for (int column = 0; column < 4; ++column) {
      check(std::isfinite(walkMatrices[index][column].x) &&
                std::isfinite(walkMatrices[index][column].y) &&
                std::isfinite(walkMatrices[index][column].z) &&
                std::isfinite(walkMatrices[index][column].w),
            "skin matrix finite");
    }
  }
  check(matricesDiffer, "walk animation changes skin matrices");

  CharacterSettings taller;
  taller.height = 2.05F;
  taller.shoulderWidth = 0.82F;
  taller.limbProportion = 1.12F;
  const SkinnedMesh scaled = generate_character_mesh(taller);
  check(!scaled.vertices.empty(), "parameterized proportions generate a mesh");
  end_test();
}

void test_brdf() {
  start_test("layered_brdf_reference");
  SurfaceSample surface;
  const glm::vec3 light = glm::normalize(glm::vec3(-0.35F, 0.8F, 0.48F));
  surface.layerWeights = glm::vec3(1.0F, 0.0F, 0.0F);
  const glm::vec3 diffuse = evaluate_layered_brdf(surface, light, glm::vec3(1.0F));
  surface.layerWeights = glm::vec3(0.0F, 1.0F, 0.0F);
  const glm::vec3 specular = evaluate_layered_brdf(surface, light, glm::vec3(1.0F));
  surface.layerWeights = glm::vec3(0.0F, 0.0F, 1.0F);
  const glm::vec3 cloth = evaluate_layered_brdf(surface, light, glm::vec3(1.0F));

  auto finite_nonnegative = [](const glm::vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) &&
           value.x >= 0.0F && value.y >= 0.0F && value.z >= 0.0F;
  };
  check(finite_nonnegative(diffuse), "diffuse layer finite and non-negative");
  check(finite_nonnegative(specular), "GGX layer finite and non-negative");
  check(finite_nonnegative(cloth), "cloth layer finite and non-negative");
  check(glm::length(diffuse) > 0.0F && glm::length(specular) > 0.0F &&
            glm::length(cloth) > 0.0F,
        "all three layers contribute energy");
  check(glm::any(glm::notEqual(diffuse, specular)) &&
            glm::any(glm::notEqual(specular, cloth)) &&
            glm::any(glm::notEqual(diffuse, cloth)),
        "BRDF layers produce distinct values");

  surface.layerWeights = glm::vec3(0.55F, 0.28F, 0.17F);
  const glm::vec3 mixed = evaluate_layered_brdf(surface, light, glm::vec3(1.0F));
  check(finite_nonnegative(mixed), "weighted mixed BRDF finite");
  check(mixed.x + mixed.y + mixed.z > 0.0F, "weighted mixed BRDF contributes");
  end_test();
}

void write_report(const std::filesystem::path& path) {
  json report = {
      {"checks", g_checks},
      {"failures", g_failures},
      {"passed", g_failures == 0},
      {"timestamp", utc_timestamp_now()},
  };
  std::ofstream output(path, std::ios::trunc);
  output << report.dump(2) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path reportPath = "lf_test_report.json";
  if (argc >= 2) {
    reportPath = argv[1];
  }
  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path scratch =
      std::filesystem::temp_directory_path() /
      ("lf_layered_character_tests_" + std::to_string(unique));
  std::filesystem::create_directories(scratch);
  Logger::instance().open(scratch / "tests.log");

  test_error_codes();
  test_deterministic_samples(scratch);
  test_config_validation(scratch / "samples_a", scratch);
  test_hot_reload_retention(scratch / "samples_a", scratch);
  test_snapshots(scratch / "samples_a", scratch);
  test_mesh_and_skin();
  test_brdf();

  write_report(reportPath);
  std::cout << "\n" << (g_failures == 0 ? "PASS" : "FAIL") << ": " << g_checks
            << " checks, " << g_failures << " failures\n";
  Logger::instance().close();
  if (g_failures == 0) {
    std::error_code ec;
    std::filesystem::remove_all(scratch, ec);
  } else {
    std::cerr << "test data retained at " << scratch << '\n';
  }
  return g_failures == 0 ? 0 : 1;
}

