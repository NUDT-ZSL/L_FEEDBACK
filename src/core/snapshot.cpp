#include "lf/snapshot.h"

#include "lf/log.h"

#include <glm/geometric.hpp>

#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#else
#include <cstdio>
#endif

namespace lf {

namespace {

using json = nlohmann::json;

[[noreturn]] void corrupt(const std::string& message) {
  throw AppError(ErrorCode::E_SNAPSHOT_CORRUPT, message);
}

std::string now_utc() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &time);
#else
  gmtime_r(&time, &tm);
#endif
  std::ostringstream output;
  output << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

bool valid_utc_timestamp(const std::string& value) {
  if (value.size() != 20U || value[4] != '-' || value[7] != '-' || value[10] != 'T' ||
      value[13] != ':' || value[16] != ':' || value[19] != 'Z') {
    return false;
  }
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (index == 4U || index == 7U || index == 10U || index == 13U || index == 16U ||
        index == 19U) {
      continue;
    }
    if (value[index] < '0' || value[index] > '9') {
      return false;
    }
  }
  auto number = [&](std::size_t offset) {
    return (value[offset] - '0') * 10 + (value[offset + 1U] - '0');
  };
  const int month = number(5U);
  const int day = number(8U);
  const int hour = number(11U);
  const int minute = number(14U);
  const int second = number(17U);
  return month >= 1 && month <= 12 && day >= 1 && day <= 31 && hour >= 0 &&
         hour <= 23 && minute >= 0 && minute <= 59 && second >= 0 && second <= 60;
}

int read_version(const json& root) {
  if (!root.is_object()) {
    corrupt("snapshot root must be an object");
  }
  if (!root.contains("formatVersion")) {
    corrupt("snapshot is missing formatVersion");
  }
  if (!root.at("formatVersion").is_number_integer()) {
    corrupt("snapshot formatVersion must be an integer");
  }
  std::int64_t version = 0;
  try {
    version = root.at("formatVersion").get<std::int64_t>();
  } catch (const nlohmann::json::exception&) {
    corrupt("snapshot formatVersion cannot be represented as an integer");
  }
  if (version < std::numeric_limits<int>::min() ||
      version > std::numeric_limits<int>::max()) {
    corrupt("snapshot formatVersion is outside the supported integer range");
  }
  return static_cast<int>(version);
}

glm::vec3 parse_weights(const json& value) {
  if (!value.is_array() || value.size() != 3U) {
    corrupt("materialWeights must be a three-element array");
  }
  glm::vec3 result{0.0F};
  for (std::size_t index = 0; index < 3U; ++index) {
    if (!value.at(index).is_number()) {
      corrupt("materialWeights components must be numeric");
    }
    const double component = value.at(index).get<double>();
    if (!std::isfinite(component) || component < 0.0 || component > 1.0) {
      corrupt("materialWeights components must be finite and in [0, 1]");
    }
    result[static_cast<glm::length_t>(index)] = static_cast<float>(component);
  }
  if (result.x + result.y + result.z <= 1.0e-5F) {
    corrupt("materialWeights sum must be positive");
  }
  return result;
}

float parse_animation_time(const json& renderState) {
  if (!renderState.is_object()) {
    corrupt("renderState must be an object");
  }
  if (!renderState.contains("animationTime")) {
    corrupt("renderState.animationTime is missing");
  }
  if (!renderState.at("animationTime").is_number()) {
    corrupt("renderState.animationTime must be numeric");
  }
  const double value = renderState.at("animationTime").get<double>();
  if (!std::isfinite(value) || value < 0.0 || value > 86400.0) {
    corrupt("renderState.animationTime is outside [0, 86400]");
  }
  return static_cast<float>(value);
}

LightingState parse_state(const json& root) {
  LightingState state;
  if (!root.contains("materialWeights")) {
    corrupt("snapshot is missing materialWeights");
  }
  state.materialWeights = parse_weights(root.at("materialWeights"));

  if (!root.contains("renderState")) {
    corrupt("snapshot is missing renderState");
  }
  state.animationTime = parse_animation_time(root.at("renderState"));

  if (!root.contains("lights") || !root.at("lights").is_array()) {
    corrupt("snapshot lights must be an array");
  }
  const json& lights = root.at("lights");
  if (lights.empty() || lights.size() > 32U) {
    corrupt("snapshot light count must be in [1, 32]");
  }
  std::set<std::string> ids;
  for (std::size_t index = 0; index < lights.size(); ++index) {
    const std::string context = "$.lights[" + std::to_string(index) + "]";
    try {
      Light light = parse_light_json(lights.at(index), context);
      if (!ids.insert(light.id).second) {
        corrupt("duplicate light id in snapshot: " + light.id);
      }
      state.lights.push_back(std::move(light));
    } catch (const JsonValidationError& error) {
      corrupt(std::string("invalid light: ") + error.what());
    }
  }
  return state;
}

json serialize_state(const LightingState& state) {
  if (state.lights.empty() || state.lights.size() > 32U) {
    throw AppError(ErrorCode::E_SNAPSHOT_IO,
                   "cannot save snapshot with an invalid light count");
  }
  glm::vec3 weights = state.materialWeights;
  if (!std::isfinite(weights.x) || !std::isfinite(weights.y) ||
      !std::isfinite(weights.z) || weights.x < 0.0F || weights.y < 0.0F ||
      weights.z < 0.0F || weights.x > 1.0F || weights.y > 1.0F || weights.z > 1.0F ||
      weights.x + weights.y + weights.z <= 1.0e-5F) {
    throw AppError(ErrorCode::E_SNAPSHOT_IO,
                   "cannot save snapshot with invalid material weights");
  }
  if (!std::isfinite(state.animationTime) || state.animationTime < 0.0F) {
    throw AppError(ErrorCode::E_SNAPSHOT_IO,
                   "cannot save snapshot with invalid animation time");
  }

  json lights = json::array();
  std::set<std::string> ids;
  for (const Light& light : state.lights) {
    try {
      validate_light(light, "$.lights");
      if (!ids.insert(light.id).second) {
        throw JsonValidationError("duplicate light id '" + light.id + "'");
      }
    } catch (const JsonValidationError& error) {
      throw AppError(ErrorCode::E_SNAPSHOT_IO,
                     std::string("cannot save snapshot: ") + error.what());
    }
    lights.push_back(light_to_json(light));
  }

  return {
      {"materialWeights", {weights.x, weights.y, weights.z}},
      {"renderState", {{"animationTime", state.animationTime}}},
      {"lights", std::move(lights)},
  };
}

json migrate_v1_to_v2(json document, std::vector<std::string>& log) {
  if (!document.is_object()) {
    corrupt("version 1 snapshot root must be an object");
  }
  if (!document.contains("materialWeights")) {
    document["materialWeights"] = {0.55, 0.28, 0.17};
    log.emplace_back("1->2: added absent materialWeights with default [0.55,0.28,0.17]");
  } else {
    log.emplace_back("1->2: preserved existing legacy materialWeights");
  }

  if (!document.contains("renderState") || !document.at("renderState").is_object()) {
    document["renderState"] = {{"animationTime", 0.0}};
    log.emplace_back("1->2: added absent renderState.animationTime as 0");
  } else if (!document.at("renderState").contains("animationTime")) {
    document["renderState"]["animationTime"] = 0.0;
    log.emplace_back("1->2: added absent renderState.animationTime as 0");
  }

  if (document.contains("lights") && document.at("lights").is_array()) {
    for (json& light : document["lights"]) {
      if (!light.is_object()) {
        continue;
      }
      if (!light.contains("enabled")) {
        light["enabled"] = true;
        log.emplace_back("1->2: defaulted light.enabled=true");
      }
      const std::string type = light.value("type", "");
      if ((type == "point" || type == "spot") && !light.contains("range")) {
        light["range"] = 10.0;
        log.emplace_back("1->2: defaulted point/spot light.range=10");
      }
      if ((type == "point" || type == "spot") && !light.contains("constantAttenuation")) {
        light["constantAttenuation"] = 1.0;
        log.emplace_back("1->2: defaulted light.constantAttenuation=1");
      }
      if ((type == "point" || type == "spot") && !light.contains("linearAttenuation")) {
        light["linearAttenuation"] = 0.0;
        log.emplace_back("1->2: defaulted light.linearAttenuation=0");
      }
      if ((type == "point" || type == "spot") && !light.contains("quadraticAttenuation")) {
        light["quadraticAttenuation"] = 0.0;
        log.emplace_back("1->2: defaulted light.quadraticAttenuation=0");
      }
    }
  }
  document["formatVersion"] = 2;
  log.emplace_back("1->2: formatVersion upgraded to 2");
  return document;
}

}  // namespace

std::string utc_timestamp_now() { return now_utc(); }

Snapshot make_snapshot(const LightingState& state) {
  Snapshot snapshot;
  snapshot.formatVersion = kSnapshotFormatVersion;
  snapshot.savedAt = now_utc();
  snapshot.state = state;
  return snapshot;
}

void save_snapshot(const std::filesystem::path& path, const Snapshot& snapshot) {
  if (snapshot.formatVersion != kSnapshotFormatVersion) {
    throw AppError(ErrorCode::E_SNAPSHOT_VERSION_UNSUPPORTED,
                   "refusing to save non-current snapshot version " +
                       std::to_string(snapshot.formatVersion));
  }

  json document = serialize_state(snapshot.state);
  document["formatVersion"] = kSnapshotFormatVersion;
  const std::string savedAt = snapshot.savedAt.empty() ? now_utc() : snapshot.savedAt;
  if (!valid_utc_timestamp(savedAt)) {
    throw AppError(ErrorCode::E_SNAPSHOT_IO,
                   "cannot save snapshot with an invalid savedAt timestamp");
  }
  document["savedAt"] = savedAt;

  std::error_code ec;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
      throw AppError(ErrorCode::E_SNAPSHOT_IO,
                     "cannot create snapshot directory: " + path.parent_path().string());
    }
  }

  const std::filesystem::path temporary = path.string() + ".tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw AppError(ErrorCode::E_SNAPSHOT_IO,
                     "cannot create snapshot temporary file: " + temporary.string());
    }
    output << document.dump(2) << '\n';
    output.flush();
    if (!output) {
      throw AppError(ErrorCode::E_SNAPSHOT_IO,
                     "failed writing snapshot temporary file: " + temporary.string());
    }
  }

#ifdef _WIN32
  if (!MoveFileExW(temporary.c_str(), path.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    std::filesystem::remove(temporary, ec);
    throw AppError(ErrorCode::E_SNAPSHOT_IO,
                   "cannot atomically replace snapshot file: " + path.string());
  }
#else
  if (std::rename(temporary.c_str(), path.c_str()) != 0) {
    std::filesystem::remove(temporary, ec);
    throw AppError(ErrorCode::E_SNAPSHOT_IO,
                   "cannot atomically replace snapshot file: " + path.string());
  }
#endif
}

SnapshotLoadResult load_snapshot(const std::filesystem::path& path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec) || ec) {
    throw AppError(ErrorCode::E_RESOURCE_MISSING,
                   "snapshot file is missing: " + path.string());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw AppError(ErrorCode::E_SNAPSHOT_CORRUPT,
                   "cannot open snapshot file: " + path.string());
  }

  json document;
  try {
    input >> document;
  } catch (const std::exception& error) {
    throw AppError(ErrorCode::E_SNAPSHOT_CORRUPT,
                   "snapshot JSON parse failure: " + std::string(error.what()));
  }

  const int version = read_version(document);
  if (version > kSnapshotFormatVersion) {
    throw AppError(ErrorCode::E_SNAPSHOT_VERSION_UNSUPPORTED,
                   "snapshot version " + std::to_string(version) +
                       " is newer than supported version " +
                       std::to_string(kSnapshotFormatVersion));
  }
  if (version < 1) {
    corrupt("snapshot version must be at least 1");
  }

  SnapshotLoadResult result;
  if (version == 1) {
    document = migrate_v1_to_v2(std::move(document), result.migrationLog);
    for (const std::string& entry : result.migrationLog) {
      log_info("snapshot migration: " + entry);
    }
  }
  result.snapshot.formatVersion = kSnapshotFormatVersion;
  if (!document.contains("savedAt") || !document.at("savedAt").is_string()) {
    corrupt("snapshot savedAt must be a string");
  }
  result.snapshot.savedAt = document.at("savedAt").get<std::string>();
  if (!valid_utc_timestamp(result.snapshot.savedAt)) {
    corrupt("snapshot savedAt must be an ISO-8601 UTC timestamp");
  }
  try {
    result.snapshot.state = parse_state(document);
  } catch (const AppError&) {
    throw;
  } catch (const std::exception& error) {
    corrupt(std::string("snapshot semantic validation failed: ") + error.what());
  }
  return result;
}

}  // namespace lf

