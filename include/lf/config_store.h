#pragma once

#include "lf/config.h"
#include "lf/error_code.h"

#include <filesystem>
#include <cstdint>
#include <optional>

namespace lf {

enum class ReloadStatus { NoChange, Reloaded, Invalid };

struct ReloadOutcome {
  ReloadStatus status = ReloadStatus::NoChange;
  ErrorCode code = ErrorCode::Ok;
  std::string message;
};

class ConfigStore {
 public:
  ConfigStore(std::filesystem::path configFile, std::filesystem::path assetRoot);

  void load_initial();
  void adopt_fallback(LightingConfig fallback);
  ReloadOutcome reload_if_changed();
  ReloadOutcome reload_now();

  [[nodiscard]] const LightingConfig& config() const;
  [[nodiscard]] bool has_config() const noexcept { return current_.has_value(); }

 private:
  ReloadOutcome reload_impl(bool force);
  void remember_file_stamp(const std::filesystem::path& path);
  [[nodiscard]] bool file_stamp_changed(const std::filesystem::path& path) const;

  std::filesystem::path configFile_;
  std::filesystem::path assetRoot_;
  ConfigLoader loader_;
  std::optional<LightingConfig> current_;
  std::filesystem::file_time_type lastWriteTime_{};
  std::uintmax_t lastFileSize_ = 0;
  bool hasStamp_ = false;
};

}  // namespace lf

