#pragma once

#include "sha256.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace rigskin {

constexpr int kManifestVersion = 2;

struct FrameEntry {
    std::uint32_t frame = 0;
    int manifestVersion = kManifestVersion;
    std::string inputHash;
    std::string outputChecksum;
    std::string meshSha256;
    std::string ppmSha256;
    std::uint64_t meshBytes = 0;
    std::uint64_t ppmBytes = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t triangleCount = 0;
    std::uint32_t degenerateTriangles = 0;
    std::uint32_t zeroWeightVertices = 0;
};

struct Manifest {
    int version = kManifestVersion;
    std::string inputHash;
    std::uint32_t frameCount = 0;
    bool complete = false;
    std::uint32_t zeroWeightVertices = 0;
    std::vector<FrameEntry> entries;
};

enum class ManifestReadStatus {
    Missing,
    Parsed,
    Corrupt,
    UnsupportedVersion,
};

struct ManifestReadResult {
    ManifestReadStatus status = ManifestReadStatus::Missing;
    Manifest manifest;
    int detectedVersion = 0;
    std::string error;
};

struct ArtifactValidation {
    bool valid = false;
    std::string errorCode;
    std::string detail;
};

ManifestReadResult readManifest(const std::filesystem::path& path);
std::string serializeManifest(const Manifest& manifest);
bool writeManifest(
    const std::filesystem::path& path,
    const Manifest& manifest,
    std::string& error);

std::filesystem::path frameMeshPath(
    const std::filesystem::path& snapshotDirectory,
    std::uint32_t frame);
std::filesystem::path framePpmPath(
    const std::filesystem::path& snapshotDirectory,
    std::uint32_t frame);

bool readFileBinary(
    const std::filesystem::path& path,
    std::vector<std::uint8_t>& bytes,
    std::string& error);
bool writeFileAtomic(
    const std::filesystem::path& path,
    const std::vector<std::uint8_t>& bytes,
    std::string& error);
bool writeFileAtomic(
    const std::filesystem::path& path,
    const std::string& text,
    std::string& error);

std::string combinedChecksum(
    const std::vector<std::uint8_t>& meshBytes,
    const std::vector<std::uint8_t>& ppmBytes);

ArtifactValidation validateFrameArtifacts(
    const std::filesystem::path& snapshotDirectory,
    const FrameEntry& entry,
    const std::string& expectedInputHash,
    int width,
    int height);

bool archiveSnapshotForRebuild(
    const std::filesystem::path& snapshotDirectory,
    const std::string& reason,
    const std::string& reportText,
    std::filesystem::path& archivePath,
    std::string& error);

}  // namespace rigskin

