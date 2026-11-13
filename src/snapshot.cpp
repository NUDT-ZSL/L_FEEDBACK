#include "skinbake/snapshot.hpp"

#include "skinbake/error.hpp"
#include "skinbake/fs.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>

namespace skinbake {

namespace {

constexpr std::array<std::uint8_t, 8> kMagic{{'S', 'K', 'N', 'B', 'K', 'E', '0', '1'}};
constexpr std::size_t kVersionOffset = 8;
constexpr std::size_t kFpsOffset = 12;
constexpr std::size_t kFingerprintOffset = 16;
constexpr std::size_t kTotalFramesOffset = 24;
constexpr std::size_t kFrameCountOffset = 28;
constexpr std::size_t kBoneCountOffset = 32;
constexpr std::size_t kVertexCountOffset = 36;
constexpr std::size_t kFlagsOffset = 40;
constexpr std::size_t kChecksumOffset = 44;

void put_u32(std::vector<std::uint8_t>& out, std::size_t offset,
             std::uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        out[offset + static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>((value >> (i * 8)) & 0xffu);
    }
}

void put_u64(std::vector<std::uint8_t>& out, std::size_t offset,
             std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        out[offset + static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>((value >> (i * 8)) & 0xffu);
    }
}

std::uint32_t get_u32(const std::vector<std::uint8_t>& bytes,
                      std::size_t offset) {
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(
                     bytes[offset + static_cast<std::size_t>(i)])
                 << (i * 8);
    }
    return value;
}

std::uint64_t get_u64(const std::vector<std::uint8_t>& bytes,
                      std::size_t offset) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(
                     bytes[offset + static_cast<std::size_t>(i)])
                 << (i * 8);
    }
    return value;
}

std::uint32_t crc32_bytes(const std::vector<std::uint8_t>& bytes) {
    std::uint32_t crc = 0xffffffffu;
    for (std::uint8_t byte : bytes) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask =
                static_cast<std::uint32_t>(-(static_cast<int>(crc & 1u)));
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return ~crc;
}

std::string hex_u32(std::uint32_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(8) << std::setfill('0') << value;
    return out.str();
}

std::size_t checked_payload_size(std::uint32_t frameCount,
                                 std::uint32_t boneCount,
                                 std::uint32_t vertexCount) {
    const std::uint64_t record =
        static_cast<std::uint64_t>(frame_record_size(boneCount, vertexCount));
    const std::uint64_t payload = record * frameCount;
    const std::uint64_t total = payload + kSnapshotFormatHeaderSize;
    if (total > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
        throw_error(ErrorCode::SnapshotDimensionMismatch,
                    "snapshot dimensions overflow host size");
    }
    return static_cast<std::size_t>(total);
}

}

std::size_t frame_record_size(std::uint32_t boneCount,
                              std::uint32_t vertexCount) {
    return 8u + static_cast<std::size_t>(boneCount) * 12u * sizeof(float) +
           static_cast<std::size_t>(vertexCount) * 2u * sizeof(float);
}

std::vector<std::uint8_t> serialize_snapshot(const Snapshot& snapshot) {
    if (snapshot.frames.size() > snapshot.totalFrames) {
        throw_error(ErrorCode::SnapshotDataInvalid,
                    "snapshot contains more frames than its declared total");
    }
    std::vector<std::uint8_t> bytes(
        checked_payload_size(static_cast<std::uint32_t>(snapshot.frames.size()),
                             snapshot.boneCount, snapshot.vertexCount),
        0);
    std::copy(kMagic.begin(), kMagic.end(), bytes.begin());
    put_u32(bytes, kVersionOffset, snapshot.version);
    put_u32(bytes, kFpsOffset, snapshot.fps);
    put_u64(bytes, kFingerprintOffset, snapshot.fingerprint);
    put_u32(bytes, kTotalFramesOffset, snapshot.totalFrames);
    put_u32(bytes, kFrameCountOffset,
            static_cast<std::uint32_t>(snapshot.frames.size()));
    put_u32(bytes, kBoneCountOffset, snapshot.boneCount);
    put_u32(bytes, kVertexCountOffset, snapshot.vertexCount);
    put_u32(bytes, kFlagsOffset,
            snapshot.complete ? kSnapshotFlagComplete : 0u);
    put_u32(bytes, kChecksumOffset, 0u);

    std::size_t offset = kSnapshotFormatHeaderSize;
    for (const FrameRecord& frame : snapshot.frames) {
        if (frame.boneMatrices.size() !=
            static_cast<std::size_t>(snapshot.boneCount) * 12u) {
            throw_error(ErrorCode::SnapshotDimensionMismatch,
                        "frame bone matrix storage has the wrong size",
                        {{"frame", std::to_string(frame.frameIndex)}});
        }
        if (frame.vertices.size() !=
            static_cast<std::size_t>(snapshot.vertexCount) * 2u) {
            throw_error(ErrorCode::SnapshotDimensionMismatch,
                        "frame vertex storage has the wrong size",
                        {{"frame", std::to_string(frame.frameIndex)}});
        }
        put_u32(bytes, offset, frame.frameIndex);
        put_u32(bytes, offset + 4, frame.timeMs);
        offset += 8;
        for (float value : frame.boneMatrices) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            put_u32(bytes, offset, bits);
            offset += 4;
        }
        for (float value : frame.vertices) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            put_u32(bytes, offset, bits);
            offset += 4;
        }
    }
    put_u32(bytes, kChecksumOffset, snapshot_checksum(bytes));
    return bytes;
}

std::uint32_t snapshot_checksum(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() < kSnapshotFormatHeaderSize) {
        return 0;
    }
    std::vector<std::uint8_t> copy = bytes;
    put_u32(copy, kChecksumOffset, 0u);
    return crc32_bytes(copy);
}

void write_snapshot_atomic(const std::string& path,
                           const Snapshot& snapshot) {
    atomic_write_bytes(path, serialize_snapshot(snapshot));
}

Snapshot load_snapshot(const std::string& path,
                       std::optional<std::uint32_t> expectedVersion,
                       std::optional<std::uint64_t> expectedFingerprint) {
    const std::vector<std::uint8_t> bytes = read_file_bytes(path);
    if (bytes.empty()) {
        throw_error(ErrorCode::SnapshotTruncated,
                    "snapshot file is empty", {{"path", path}});
    }
    if (bytes.size() < kMagic.size()) {
        throw_error(ErrorCode::SnapshotTruncated,
                    "snapshot is shorter than its magic",
                    {{"path", path}, {"size", std::to_string(bytes.size())}});
    }
    if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
        std::ostringstream actual;
        actual << std::hex << std::setw(2) << std::setfill('0');
        for (std::size_t i = 0; i < 8; ++i) {
            actual << static_cast<unsigned>(bytes[i]);
        }
        throw_error(ErrorCode::SnapshotMagicMismatch,
                    "snapshot magic mismatch",
                    {{"path", path},
                     {"actualMagic", actual.str()},
                     {"expectedMagic", "SKNBKE01"}});
    }
    if (bytes.size() < kVersionOffset + sizeof(std::uint32_t)) {
        throw_error(ErrorCode::SnapshotTruncated,
                    "snapshot is shorter than its version field",
                    {{"path", path}});
    }
    const std::uint32_t actualVersion = get_u32(bytes, kVersionOffset);
    if (expectedVersion.has_value() && actualVersion != *expectedVersion) {
        throw_error(
            ErrorCode::SnapshotVersionMismatch,
            "snapshot version mismatch: actual=" +
                std::to_string(actualVersion) +
                " expected=" + std::to_string(*expectedVersion),
            {{"actualVersion", std::to_string(actualVersion)},
             {"expectedVersion", std::to_string(*expectedVersion)}});
    }
    if (bytes.size() < kSnapshotFormatHeaderSize) {
        throw_error(ErrorCode::SnapshotTruncated,
                    "snapshot header is truncated",
                    {{"path", path}, {"size", std::to_string(bytes.size())}});
    }

    const std::uint32_t storedChecksum = get_u32(bytes, kChecksumOffset);
    const std::uint32_t computedChecksum = snapshot_checksum(bytes);
    if (storedChecksum != computedChecksum) {
        throw_error(
            ErrorCode::SnapshotChecksumMismatch,
            "snapshot checksum mismatch: stored=" + hex_u32(storedChecksum) +
                " computed=" + hex_u32(computedChecksum),
            {{"storedChecksum", std::to_string(storedChecksum)},
             {"computedChecksum", std::to_string(computedChecksum)}});
    }

    Snapshot snapshot;
    snapshot.version = actualVersion;
    snapshot.fps = get_u32(bytes, kFpsOffset);
    snapshot.fingerprint = get_u64(bytes, kFingerprintOffset);
    snapshot.totalFrames = get_u32(bytes, kTotalFramesOffset);
    snapshot.boneCount = get_u32(bytes, kBoneCountOffset);
    snapshot.vertexCount = get_u32(bytes, kVertexCountOffset);
    const std::uint32_t frameCount = get_u32(bytes, kFrameCountOffset);
    const std::uint32_t flags = get_u32(bytes, kFlagsOffset);
    if ((flags & ~kSnapshotFlagComplete) != 0) {
        throw_error(ErrorCode::SnapshotDataInvalid,
                    "snapshot contains unknown flag bits",
                    {{"flags", std::to_string(flags)}});
    }
    snapshot.complete = (flags & kSnapshotFlagComplete) != 0;
    if (snapshot.fps == 0 || snapshot.boneCount == 0 ||
        snapshot.vertexCount == 0) {
        throw_error(ErrorCode::SnapshotDataInvalid,
                    "snapshot metadata contains zero-valued dimensions");
    }
    if (frameCount > snapshot.totalFrames) {
        throw_error(ErrorCode::SnapshotDataInvalid,
                    "snapshot frame count exceeds total frame count");
    }
    if (snapshot.complete && frameCount != snapshot.totalFrames) {
        throw_error(ErrorCode::SnapshotDataInvalid,
                    "complete snapshot does not contain every frame");
    }
    if (expectedFingerprint.has_value() &&
        snapshot.fingerprint != *expectedFingerprint) {
        throw_error(ErrorCode::SnapshotFingerprintMismatch,
                    "snapshot fingerprint does not match the requested scene",
                    {{"actualFingerprint", std::to_string(snapshot.fingerprint)},
                     {"expectedFingerprint",
                      std::to_string(*expectedFingerprint)}});
    }

    const std::size_t expectedSize =
        checked_payload_size(frameCount, snapshot.boneCount,
                             snapshot.vertexCount);
    if (bytes.size() != expectedSize) {
        throw_error(ErrorCode::SnapshotTruncated,
                    "snapshot byte length does not match its frame metadata",
                    {{"expectedSize", std::to_string(expectedSize)},
                     {"actualSize", std::to_string(bytes.size())}});
    }

    snapshot.frames.reserve(frameCount);
    const std::size_t matrixFloats =
        static_cast<std::size_t>(snapshot.boneCount) * 12u;
    const std::size_t vertexFloats =
        static_cast<std::size_t>(snapshot.vertexCount) * 2u;
    std::size_t offset = kSnapshotFormatHeaderSize;
    for (std::uint32_t index = 0; index < frameCount; ++index) {
        FrameRecord frame;
        frame.frameIndex = get_u32(bytes, offset);
        frame.timeMs = get_u32(bytes, offset + 4);
        offset += 8;
        if (frame.frameIndex != index) {
            throw_error(ErrorCode::SnapshotDataInvalid,
                        "snapshot frame indices are not contiguous",
                        {{"expectedFrame", std::to_string(index)},
                         {"actualFrame", std::to_string(frame.frameIndex)}});
        }
        frame.boneMatrices.resize(matrixFloats);
        for (float& value : frame.boneMatrices) {
            const std::uint32_t bits = get_u32(bytes, offset);
            std::memcpy(&value, &bits, sizeof(value));
            offset += 4;
            if (!std::isfinite(value)) {
                throw_error(ErrorCode::SnapshotDataInvalid,
                            "snapshot matrix contains a non-finite value",
                            {{"frame", std::to_string(index)}});
            }
        }
        frame.vertices.resize(vertexFloats);
        for (float& value : frame.vertices) {
            const std::uint32_t bits = get_u32(bytes, offset);
            std::memcpy(&value, &bits, sizeof(value));
            offset += 4;
            if (!std::isfinite(value)) {
                throw_error(ErrorCode::SnapshotDataInvalid,
                            "snapshot vertex contains a non-finite value",
                            {{"frame", std::to_string(index)}});
            }
        }
        snapshot.frames.push_back(std::move(frame));
    }
    return snapshot;
}

Snapshot snapshot_from_progress(const Snapshot& progress, bool complete) {
    Snapshot snapshot = progress;
    snapshot.complete = complete;
    return snapshot;
}

}

