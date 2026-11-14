#include "lf/config_store.h"

#include "lf/log.h"

#include <system_error>

namespace lf {

ConfigStore::ConfigStore(std::filesystem::path configFile, std::filesystem::path assetRoot)
    : configFile_(std::move(configFile)),
      assetRoot_(std::move(assetRoot)),
      loader_(assetRoot_) {}

void ConfigStore::load_initial() {
  current_ = loader_.load(configFile_);
  remember_file_stamp(configFile_);
  log_info("loaded configuration: " + configFile_.string() + " (" +
           std::to_string(current_->lights.size()) + " lights)");
}

void ConfigStore::adopt_fallback(LightingConfig fallback) {
  try {
    current_ = std::move(fallback);
    remember_file_stamp(configFile_);
    log_warn(ErrorCode::E_CONFIG_INVALID,
             "using in-memory deterministic fallback configuration; invalid file retained");
  } catch (const std::exception& error) {
    throw AppError(ErrorCode::E_CONFIG_INVALID,
                   std::string("fallback configuration is invalid: ") + error.what());
  }
}

ReloadOutcome ConfigStore::reload_if_changed() { return reload_impl(false); }

ReloadOutcome ConfigStore::reload_now() { return reload_impl(true); }

ReloadOutcome ConfigStore::reload_impl(bool force) {
  if (!force && has_config() && !file_stamp_changed(configFile_)) {
    return {ReloadStatus::NoChange, ErrorCode::Ok, {}};
  }
  if (!has_config() && !force) {
    return {ReloadStatus::NoChange, ErrorCode::Ok, {}};
  }

  try {
    LightingConfig replacement = loader_.load(configFile_);
    current_ = std::move(replacement);
    remember_file_stamp(configFile_);
    log_info("configuration hot-reloaded atomically: " + configFile_.string());
    return {ReloadStatus::Reloaded, ErrorCode::Ok, {}};
  } catch (const AppError& error) {
    remember_file_stamp(configFile_);
    const std::string message = "configuration reload rejected; previous valid state retained: " +
                                std::string(error_code_name(error.code())) + ": " + error.what();
    log_error(ErrorCode::E_CONFIG_RELOAD_INVALID, message);
    return {ReloadStatus::Invalid, ErrorCode::E_CONFIG_RELOAD_INVALID,
            std::string(error.what())};
  } catch (const std::exception& error) {
    remember_file_stamp(configFile_);
    const std::string message =
        "configuration reload rejected; previous valid state retained: " +
        std::string(error.what());
    log_error(ErrorCode::E_CONFIG_RELOAD_INVALID, message);
    return {ReloadStatus::Invalid, ErrorCode::E_CONFIG_RELOAD_INVALID, error.what()};
  }
}

const LightingConfig& ConfigStore::config() const {
  if (!current_) {
    throw AppError(ErrorCode::E_CONFIG_INVALID, "configuration has not been loaded");
  }
  return *current_;
}

void ConfigStore::remember_file_stamp(const std::filesystem::path& path) {
  std::error_code ec;
  lastWriteTime_ = std::filesystem::last_write_time(path, ec);
  if (ec) {
    hasStamp_ = false;
    lastFileSize_ = 0;
    return;
  }
  lastFileSize_ = std::filesystem::file_size(path, ec);
  hasStamp_ = !ec;
}

bool ConfigStore::file_stamp_changed(const std::filesystem::path& path) const {
  std::error_code ec;
  const auto writeTime = std::filesystem::last_write_time(path, ec);
  if (ec) {
    return true;
  }
  const std::uintmax_t fileSize = std::filesystem::file_size(path, ec);
  if (ec) {
    return true;
  }
  return !hasStamp_ || writeTime != lastWriteTime_ || fileSize != lastFileSize_;
}

}  // namespace lf

