#include "baker/snapshot.h"

#include <bit>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>

#include "baker/byte_io.h"
#include "baker/scene.h"

namespace baker {
namespace {

constexpr std::uint64_t kMaxSnapshotBytes = 2ull * 1024ull * 1024ull * 1024ull;

[[noreturn]] void corrupt(const std::string& detail) {
    throw BakerError(ErrorCode::kLength, detail);
}

std::uint64_t expected_snapshot_size(std::uint32_t version,
                                     std::uint32_t layer_count,
                                     std::uint32_t width,
                                     std::uint32_t height) {
    const std::uint64_t fixed = version == kLegacySnapshotVersion ? 32ull : 40ull;
    return fixed + static_cast<std::uint64_t>(layer_count) * 8ull +
           static_cast<std::uint64_t>(width) * height * 4ull * sizeof(float);
}

}  // namespace

const char* snapshot_status_name(SnapshotStatus status) noexcept {
    switch (status) {
    case SnapshotStatus::kMissing: return "missing";
    case SnapshotStatus::kLoaded: return "hit";
    case SnapshotStatus::kVersionMismatch: return "version_mismatch";
    case SnapshotStatus::kCorrupt: return "corrupt";
    }
    return "unknown";
}

SnapshotReadResult read_snapshot_file(const std::filesystem::path& path) {
    SnapshotReadResult result;

    std::error_code exists_error;
    const bool exists = std::filesystem::exists(path, exists_error);
    if (exists_error || !exists) {
        result.status = SnapshotStatus::kMissing;
        result.detail = exists_error ? exists_error.message() : "missing";
        return result;
    }

    std::vector<std::uint8_t> bytes;
    try {
        bytes = read_file(path, kMaxSnapshotBytes, "snapshot");
    } catch (const BakerError&) {
        result.status = SnapshotStatus::kCorrupt;
        result.detail = "snapshot read failed";
        return result;
    }

    try {
        ByteReader reader(bytes);
        if (reader.remaining() < 8) {
            corrupt("truncated snapshot header");
        }
        if (reader.u32("snapshot magic") != kSnapshotMagic) {
            corrupt("bad snapshot magic");
        }

        const std::uint32_t version = reader.u32("snapshot version");
        if (version != kSnapshotVersion && version != kLegacySnapshotVersion) {
            result.status = SnapshotStatus::kVersionMismatch;
            result.detail = "unsupported snapshot version " + std::to_string(version);
            return result;
        }

        Snapshot snapshot;
        snapshot.version = version;
        snapshot.input_hash = reader.u64("snapshot input hash");
        snapshot.width = reader.u32("snapshot width");
        snapshot.height = reader.u32("snapshot height");
        snapshot.layer_count = reader.u32("snapshot layer count");
        snapshot.completed_layers = reader.u32("snapshot completed layer count");
        if (version >= kSnapshotVersion) {
            snapshot.time_bits = reader.u32("snapshot time");
            snapshot.degenerate_layers = reader.u32("snapshot degenerate layer count");
            snapshot.encoded_degenerate_count = true;
        } else {
            snapshot.time_bits = std::bit_cast<std::uint32_t>(0.0f);
            snapshot.legacy_stats = true;
        }

        if (snapshot.width == 0 || snapshot.width > kMaxDimension ||
            snapshot.height == 0 || snapshot.height > kMaxDimension) {
            corrupt("illegal snapshot resolution");
        }
        if (snapshot.layer_count == 0 || snapshot.layer_count > kMaxLayers ||
            snapshot.completed_layers > snapshot.layer_count) {
            corrupt("illegal snapshot layer counts");
        }
        if (snapshot.degenerate_layers > snapshot.layer_count) {
            corrupt("illegal snapshot degenerate layer count");
        }
        const std::uint64_t pixels =
            static_cast<std::uint64_t>(snapshot.width) * snapshot.height;
        if (pixels > kMaxPixels) {
            corrupt("snapshot resolution is too large");
        }

        const std::uint64_t expected =
            expected_snapshot_size(version, snapshot.layer_count, snapshot.width,
                                   snapshot.height);
        if (expected != bytes.size()) {
            corrupt("snapshot payload length mismatch");
        }

        snapshot.layer_pixels.resize(snapshot.layer_count);
        for (std::uint32_t index = 0; index < snapshot.layer_count; ++index) {
            const std::uint64_t stored = reader.u64("snapshot layer pixel count");
            std::uint64_t count = stored;
            if (version == kLegacySnapshotVersion && index + 1 == snapshot.layer_count &&
                stored > std::numeric_limits<std::uint32_t>::max()) {
                // Legacy writers placed the scalar degenerate-layer count in the
                // high word of the final 64-bit statistics slot.
                count = stored & 0xffffffffull;
                snapshot.degenerate_layers =
                    static_cast<std::uint32_t>(stored >> 32);
                if (snapshot.degenerate_layers > snapshot.layer_count) {
                    corrupt("illegal legacy snapshot degenerate layer count");
                }
                snapshot.legacy_stats = true;
                snapshot.encoded_degenerate_count = true;
            }
            if (count > pixels) {
                corrupt("illegal snapshot layer pixel count");
            }
            snapshot.layer_pixels[index] = count;
        }
        const std::uint64_t float_count = pixels * 4ull;
        snapshot.pixels.resize(static_cast<std::size_t>(float_count));
        for (float& value : snapshot.pixels) {
            value = reader.f32("snapshot pixel data");
            if (!is_finite(value) || value < 0.0f || value > 1.0f) {
                corrupt("illegal snapshot pixel value");
            }
        }
        if (!reader.at_end()) {
            corrupt("trailing snapshot bytes");
        }

        result.status = SnapshotStatus::kLoaded;
        result.detail = "loaded";
        result.snapshot = std::move(snapshot);
        return result;
    } catch (const BakerError& error) {
        result.status = SnapshotStatus::kCorrupt;
        result.detail = error.what();
        return result;
    }
}

std::vector<std::uint8_t> encode_snapshot(const Snapshot& snapshot) {
    if (snapshot.version != kLegacySnapshotVersion &&
        snapshot.version != kSnapshotVersion) {
        throw BakerError(ErrorCode::kVersion, "cannot encode unsupported snapshot version");
    }
    if (snapshot.width == 0 || snapshot.height == 0 ||
        snapshot.layer_count == 0 || snapshot.completed_layers > snapshot.layer_count ||
        snapshot.layer_pixels.size() != snapshot.layer_count) {
        throw BakerError(ErrorCode::kValue, "cannot encode inconsistent snapshot header");
    }
    const std::uint64_t pixel_floats =
        static_cast<std::uint64_t>(snapshot.width) * snapshot.height * 4ull;
    if (snapshot.pixels.size() != pixel_floats) {
        throw BakerError(ErrorCode::kValue, "cannot encode inconsistent snapshot pixels");
    }

    std::vector<std::uint8_t> bytes;
    bytes.reserve(static_cast<std::size_t>(
        expected_snapshot_size(snapshot.version, snapshot.layer_count, snapshot.width,
                               snapshot.height)));
    append_u32(bytes, kSnapshotMagic);
    append_u32(bytes, snapshot.version);
    append_u64(bytes, snapshot.input_hash);
    append_u32(bytes, snapshot.width);
    append_u32(bytes, snapshot.height);
    append_u32(bytes, snapshot.layer_count);
    append_u32(bytes, snapshot.completed_layers);
    if (snapshot.version >= kSnapshotVersion) {
        append_u32(bytes, snapshot.time_bits);
        append_u32(bytes, snapshot.degenerate_layers);
    }
    for (const std::uint64_t count : snapshot.layer_pixels) {
        append_u64(bytes, count);
    }
    for (const float value : snapshot.pixels) {
        append_f32(bytes, value);
    }
    return bytes;
}

Snapshot make_snapshot(const Scene& scene,
                       std::uint32_t completed_layers,
                       float time,
                       const Stats& stats,
                       const PixelBuffer& pixels) {
    Snapshot snapshot;
    snapshot.version = kSnapshotVersion;
    snapshot.input_hash = scene.input_hash;
    snapshot.width = scene.width;
    snapshot.height = scene.height;
    snapshot.layer_count = static_cast<std::uint32_t>(scene.layers.size());
    snapshot.completed_layers = completed_layers;
    snapshot.time_bits = std::bit_cast<std::uint32_t>(time);
    snapshot.degenerate_layers = stats.degenerate_layers;
    snapshot.layer_pixels = stats.layer_pixels;
    snapshot.layer_pixels.resize(scene.layers.size(), 0);
    snapshot.pixels = pixels.rgba;
    return snapshot;
}

}  // namespace baker
