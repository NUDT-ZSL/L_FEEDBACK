#include "snapshot.h"

#include "crc32.h"
#include "errors.h"
#include "json.h"

#include <array>
#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace cloud {
namespace {

constexpr std::array<char, 8> kMagic{'C', 'L', 'D', 'S', 'N', 'A', 'P', '0'};
constexpr std::size_t kHeaderSize = 24;
constexpr std::size_t kMaximumDimensions = 16384;

struct SnapshotLayout {
    std::uint32_t version;
    std::size_t seed;
    std::size_t cloudType;
    std::size_t densityResolution;
    std::size_t frameIndex;
    std::size_t width;
    std::size_t height;
    std::size_t cameraHistoryCount;
    std::size_t viewMatrix;
    std::size_t projectionMatrix;
    std::size_t sunDirection;
    std::size_t sunColor;
    std::size_t ambient;
    std::size_t historyLength;
    std::size_t depth;
    const char* name;
};

constexpr SnapshotLayout kV1Layout{
    1,
    0,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    96,
    std::numeric_limits<std::size_t>::max(),
    std::numeric_limits<std::size_t>::max(),
    std::numeric_limits<std::size_t>::max(),
    std::numeric_limits<std::size_t>::max(),
    160,
    "v1",
};

constexpr SnapshotLayout kV2Layout{
    2,
    0,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    96,
    160,
    172,
    184,
    188,
    192,
    "v2",
};

const SnapshotLayout& layoutFor(std::uint32_t version) {
    if (version == 1) {
        return kV1Layout;
    }
    if (version == 2) {
        return kV2Layout;
    }
    throw BusinessError("E_VERSION_UNSUPPORTED", kExitVersionUnsupported,
                        "snapshot version is not supported",
                        json::Value::Object{{"foundVersion", version},
                                            {"supportedVersions", json::Value::Array{1, 2}}});
}

std::uint32_t readU32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    if (offset + 4 > bytes.size()) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot field is outside payload");
    }
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8u) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16u) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24u);
}

std::uint64_t readU64(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    if (offset + 8 > bytes.size()) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot field is outside payload");
    }
    std::uint64_t value = 0;
    for (int byte = 0; byte < 8; ++byte) {
        value |= static_cast<std::uint64_t>(bytes[offset + static_cast<std::size_t>(byte)])
                 << (static_cast<unsigned>(byte) * 8u);
    }
    return value;
}

float readF32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return std::bit_cast<float>(readU32(bytes, offset));
}

void writeU32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    if (offset + 4 > bytes.size()) {
        bytes.resize(offset + 4);
    }
    bytes[offset] = static_cast<std::uint8_t>(value & 0xffu);
    bytes[offset + 1] = static_cast<std::uint8_t>((value >> 8u) & 0xffu);
    bytes[offset + 2] = static_cast<std::uint8_t>((value >> 16u) & 0xffu);
    bytes[offset + 3] = static_cast<std::uint8_t>((value >> 24u) & 0xffu);
}

void writeU64(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint64_t value) {
    if (offset + 8 > bytes.size()) {
        bytes.resize(offset + 8);
    }
    for (int byte = 0; byte < 8; ++byte) {
        bytes[offset + static_cast<std::size_t>(byte)] = static_cast<std::uint8_t>(
            (value >> (static_cast<unsigned>(byte) * 8u)) & 0xffu);
    }
}

void writeF32(std::vector<std::uint8_t>& bytes, std::size_t offset, float value) {
    writeU32(bytes, offset, std::bit_cast<std::uint32_t>(value));
}

void writeMatrix(std::vector<std::uint8_t>& bytes, std::size_t offset, const Mat4& matrix) {
    for (int index = 0; index < 16; ++index) {
        writeF32(bytes, offset + static_cast<std::size_t>(index) * 4, matrix.m[index]);
    }
}

Mat4 readMatrix(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    Mat4 matrix;
    for (int index = 0; index < 16; ++index) {
        matrix.m[index] = readF32(bytes, offset + static_cast<std::size_t>(index) * 4);
    }
    return matrix;
}

std::vector<std::uint8_t> readFile(const std::string& path) {
    std::error_code error;
    if (!std::filesystem::exists(path, error) || error) {
        throw BusinessError("E_SNAPSHOT_MISSING", kExitSnapshotMissing,
                            "snapshot file does not exist", json::Value::Object{{"path", path}});
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw BusinessError("E_SNAPSHOT_MISSING", kExitSnapshotMissing,
                            "snapshot file cannot be opened", json::Value::Object{{"path", path}});
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0 || size > static_cast<std::streamoff>(2ull * 1024ull * 1024ull * 1024ull)) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot file size is invalid");
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!input) {
            throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                                "snapshot file is truncated");
        }
    }
    return bytes;
}

void writeFileAtomically(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    const std::filesystem::path target(path);
    std::error_code error;
    if (target.has_parent_path()) {
        std::filesystem::create_directories(target.parent_path(), error);
        if (error) {
            throw BusinessError("E_OUTPUT", kExitOutput, "cannot create snapshot directory",
                                json::Value::Object{{"path", target.parent_path().string()}});
        }
    }
    const std::filesystem::path temporary = target.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw BusinessError("E_OUTPUT", kExitOutput, "cannot open temporary snapshot",
                                json::Value::Object{{"path", temporary.string()}});
        }
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        output.flush();
        if (!output) {
            throw BusinessError("E_OUTPUT", kExitOutput, "failed while writing snapshot",
                                json::Value::Object{{"path", temporary.string()}});
        }
    }
    std::filesystem::remove(target, error);
    error.clear();
    std::filesystem::rename(temporary, target, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        throw BusinessError("E_OUTPUT", kExitOutput, "cannot publish snapshot atomically",
                            json::Value::Object{{"path", path}});
    }
}

std::size_t checkedElementCount(std::uint32_t width, std::uint32_t height) {
    if (width == 0 || height == 0 || width > kMaximumDimensions || height > kMaximumDimensions) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot dimensions are invalid");
    }
    const std::uint64_t count = static_cast<std::uint64_t>(width) * height;
    if (count > 100'000'000ull) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot pixel count is too large");
    }
    return static_cast<std::size_t>(count);
}

std::size_t expectedPayloadSize(const SnapshotLayout& layout, std::size_t pixelCount) {
    constexpr std::size_t bytesPerPixel = 4 + 12 + 4;
    if (pixelCount > (std::numeric_limits<std::size_t>::max() - layout.depth) / bytesPerPixel) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot payload size overflows");
    }
    return layout.depth + pixelCount * bytesPerPixel;
}

}  // namespace

std::uint32_t snapshotVersionFromDisk(const std::string& path) {
    const std::vector<std::uint8_t> file = readFile(path);
    if (file.size() < kHeaderSize) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot header is truncated");
    }
    if (!std::equal(kMagic.begin(), kMagic.end(), file.begin())) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot magic does not match");
    }
    return readU32(file, 8);
}

SnapshotData loadSnapshot(const std::string& path) {
    const std::vector<std::uint8_t> file = readFile(path);
    if (file.size() < kHeaderSize) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot header is truncated",
                            json::Value::Object{{"path", path}});
    }
    if (!std::equal(kMagic.begin(), kMagic.end(), file.begin())) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot magic does not match",
                            json::Value::Object{{"path", path}});
    }

    const std::uint32_t version = readU32(file, 8);
    const SnapshotLayout& layout = layoutFor(version);
    const std::uint64_t payloadSize = readU64(file, 12);
    const std::uint32_t storedCrc = readU32(file, 20);
    if (payloadSize != file.size() - kHeaderSize) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot payload length does not match header",
                            json::Value::Object{{"version", version},
                                                {"headerPayloadSize", payloadSize},
                                                {"actualPayloadSize", file.size() - kHeaderSize}});
    }

    const std::vector<std::uint8_t> payload(file.begin() + static_cast<std::ptrdiff_t>(kHeaderSize),
                                            file.end());
    const std::uint32_t actualCrc = crc32(payload.data(), payload.size());
    if (actualCrc != storedCrc) {
        std::ostringstream expected;
        std::ostringstream actual;
        expected << std::hex << storedCrc;
        actual << std::hex << actualCrc;
        throw BusinessError("E_SNAPSHOT_CRC", kExitSnapshotCrc, "snapshot CRC32 mismatch",
                            json::Value::Object{{"expectedCrc32", expected.str()},
                                                {"actualCrc32", actual.str()},
                                                {"version", version}});
    }

    SnapshotData data;
    data.version = version;
    data.seed = readU64(payload, layout.seed);
    data.cloudType = readU32(payload, layout.cloudType);
    data.densityResolution = readU32(payload, layout.densityResolution);
    data.frameIndex = readU32(payload, layout.frameIndex);
    data.width = readU32(payload, layout.width);
    data.height = readU32(payload, layout.height);
    const std::uint32_t cameraHistoryCount = readU32(payload, layout.cameraHistoryCount);
    if (cameraHistoryCount != 1) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot must contain exactly one camera-history entry");
    }
    data.view = readMatrix(payload, layout.viewMatrix);
    data.projection = readMatrix(payload, layout.projectionMatrix);

    if (version == 2) {
        data.hasLight = true;
        data.nextLight.direction = normalize(
            Vec3{readF32(payload, layout.sunDirection),
                 readF32(payload, layout.sunDirection + 4),
                 readF32(payload, layout.sunDirection + 8)});
        data.nextLight.color = {
            readF32(payload, layout.sunColor),
            readF32(payload, layout.sunColor + 4),
            readF32(payload, layout.sunColor + 8),
        };
        data.nextLight.ambient = readF32(payload, layout.ambient);
        data.historyLength = readU32(payload, layout.historyLength);
    } else {
        data.hasLight = true;
        data.upgradedFromV1 = true;
        data.nextLight = defaultLight();
        data.historyLength = 8;
    }

    if (data.cloudType > 1) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot cloud type is invalid");
    }
    if (data.densityResolution != 128 && data.densityResolution != 64 &&
        data.densityResolution != 32) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot density resolution is invalid");
    }
    if (data.historyLength == 0 || data.historyLength > 4096) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot history length is invalid");
    }

    const std::size_t pixelCount = checkedElementCount(data.width, data.height);
    const std::size_t expectedSize = expectedPayloadSize(layout, pixelCount);
    if (payload.size() != expectedSize) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot payload does not match version layout",
                            json::Value::Object{{"version", version},
                                                {"expectedPayloadSize", expectedSize},
                                                {"actualPayloadSize", payload.size()}});
    }

    data.depth.resize(pixelCount);
    data.accumulatedColor.resize(pixelCount);
    data.counts.resize(pixelCount);
    std::size_t offset = layout.depth;
    for (std::size_t pixel = 0; pixel < pixelCount; ++pixel) {
        data.depth[pixel] = readF32(payload, offset);
        offset += 4;
    }
    for (std::size_t pixel = 0; pixel < pixelCount; ++pixel) {
        data.accumulatedColor[pixel] = {
            readF32(payload, offset),
            readF32(payload, offset + 4),
            readF32(payload, offset + 8),
        };
        offset += 12;
    }
    for (std::size_t pixel = 0; pixel < pixelCount; ++pixel) {
        data.counts[pixel] = readU32(payload, offset);
        offset += 4;
    }
    return data;
}

std::uint32_t saveSnapshot(const std::string& path, const SnapshotData& data, std::uint32_t version) {
    const SnapshotLayout& layout = layoutFor(version);
    const std::size_t pixelCount = checkedElementCount(data.width, data.height);
    if (data.accumulatedColor.size() != pixelCount || data.depth.size() != pixelCount ||
        data.counts.size() != pixelCount) {
        throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                            "snapshot state buffers do not match dimensions");
    }

    const std::size_t payloadSize = expectedPayloadSize(layout, pixelCount);
    std::vector<std::uint8_t> payload(payloadSize, 0);
    writeU64(payload, layout.seed, data.seed);
    writeU32(payload, layout.cloudType, data.cloudType);
    writeU32(payload, layout.densityResolution, data.densityResolution);
    writeU32(payload, layout.frameIndex, data.frameIndex);
    writeU32(payload, layout.width, data.width);
    writeU32(payload, layout.height, data.height);
    writeU32(payload, layout.cameraHistoryCount, 1);
    writeMatrix(payload, layout.viewMatrix, data.view);
    writeMatrix(payload, layout.projectionMatrix, data.projection);
    if (version == 2) {
        const Vec3 direction = normalize(data.nextLight.direction);
        writeF32(payload, layout.sunDirection, direction.x);
        writeF32(payload, layout.sunDirection + 4, direction.y);
        writeF32(payload, layout.sunDirection + 8, direction.z);
        writeF32(payload, layout.sunColor, data.nextLight.color.x);
        writeF32(payload, layout.sunColor + 4, data.nextLight.color.y);
        writeF32(payload, layout.sunColor + 8, data.nextLight.color.z);
        writeF32(payload, layout.ambient, data.nextLight.ambient);
        writeU32(payload, layout.historyLength, data.historyLength);
    }

    std::size_t offset = layout.depth;
    for (const float depth : data.depth) {
        writeF32(payload, offset, depth);
        offset += 4;
    }
    for (const Vec3& color : data.accumulatedColor) {
        writeF32(payload, offset, color.x);
        writeF32(payload, offset + 4, color.y);
        writeF32(payload, offset + 8, color.z);
        offset += 12;
    }
    for (const std::uint32_t count : data.counts) {
        writeU32(payload, offset, count);
        offset += 4;
    }

    const std::uint32_t checksum = crc32(payload.data(), payload.size());
    std::vector<std::uint8_t> file(kHeaderSize + payload.size(), 0);
    std::copy(kMagic.begin(), kMagic.end(), file.begin());
    writeU32(file, 8, version);
    writeU64(file, 12, payload.size());
    writeU32(file, 20, checksum);
    std::copy(payload.begin(), payload.end(), file.begin() + static_cast<std::ptrdiff_t>(kHeaderSize));
    writeFileAtomically(path, file);
    return checksum;
}

}  // namespace cloud

