#pragma once

#include "types.h"

#include <cstdint>
#include <string>

namespace dr {

constexpr std::uint32_t kSnapshotHeaderSize = 160;
constexpr std::uint64_t kSnapshotMaxFileBytes = 256ULL * 1024ULL * 1024ULL;

std::uint64_t snapshotChecksum(const std::uint8_t* data, std::size_t size);
bool saveSnapshot(const std::string& path, const SnapshotData& snapshot, SnapshotError& error);
bool loadSnapshot(const std::string& path, SnapshotData& snapshot, SnapshotError& error);

std::string versionedSnapshotPath(const std::string& basePath, std::uint64_t version);
std::uint64_t nextAvailableLightVersion(std::uint64_t desiredVersion, const std::string& basePath);

} // namespace dr
