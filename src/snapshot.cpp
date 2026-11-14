#include "snapshot.h"

#include "scene.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

namespace dr {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {'D', 'G', 'S', 'N', 'A', 'P', '0', '1'};

void appendU32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
}

void appendFloat(std::vector<std::uint8_t>& bytes, float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    appendU32(bytes, bits);
}

std::uint32_t readU32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24U);
}

std::uint64_t readU64(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    std::uint64_t value = 0;
    for (int index = 7; index >= 0; --index) {
        value = (value << 8U) | bytes[offset + static_cast<std::size_t>(index)];
    }
    return value;
}

float readFloat(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    const std::uint32_t bits = readU32(bytes, offset);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void writeU32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        bytes[offset + static_cast<std::size_t>(shift / 8)] =
            static_cast<std::uint8_t>((value >> shift) & 0xffU);
    }
}

void writeU64(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        bytes[offset + static_cast<std::size_t>(shift / 8)] =
            static_cast<std::uint8_t>((value >> shift) & 0xffU);
    }
}

void writeFloat(std::vector<std::uint8_t>& bytes, std::size_t offset, float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    writeU32(bytes, offset, bits);
}

SnapshotError makeError(const std::string& code, const std::string& message,
                        const std::string& path) {
    SnapshotError error;
    error.code = code;
    error.message = message;
    error.path = path;
    return error;
}

void setHeaderVersions(SnapshotError& error, std::uint32_t formatVersion, std::uint64_t lightVersion) {
    error.formatVersion = formatVersion;
    error.lightVersion = lightVersion;
}

class TempFileGuard {
public:
    explicit TempFileGuard(std::filesystem::path path) : path_(std::move(path)) {}
    ~TempFileGuard() {
        if (!committed_) {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }
    }
    void commit() { committed_ = true; }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
    bool committed_ = false;
};

bool checkedProduct(std::uint64_t a, std::uint64_t b, std::uint64_t& result) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

} // namespace

std::uint64_t snapshotChecksum(const std::uint8_t* data, std::size_t size) {
    constexpr std::uint64_t offsetBasis = 14695981039346656037ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t hash = offsetBasis;
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= data[index];
        hash *= prime;
    }
    return hash;
}

bool saveSnapshot(const std::string& pathValue, const SnapshotData& snapshot, SnapshotError& error) {
    error = makeError("", "", pathValue);
    error.formatVersion = snapshot.formatVersion;
    error.lightVersion = snapshot.lightVersion;

    if (snapshot.width <= 0 || snapshot.height <= 0) {
        error = makeError("ERR_SNAPSHOT_INVALID_GBUFFER", "invalid snapshot dimensions", pathValue);
        setHeaderVersions(error, snapshot.formatVersion, snapshot.lightVersion);
        return false;
    }

    std::size_t pointCount = 0;
    std::size_t directionalCount = 0;
    for (const Light& light : snapshot.lights) {
        if (light.type == LightType::Point) {
            ++pointCount;
        } else {
            ++directionalCount;
        }
        if (light.id.size() > 64) {
            error = makeError("ERR_SNAPSHOT_WRITE", "light id is too long", pathValue);
            setHeaderVersions(error, snapshot.formatVersion, snapshot.lightVersion);
            return false;
        }
    }
    if (pointCount > static_cast<std::size_t>(kMaxPointLights) ||
        directionalCount > static_cast<std::size_t>(kMaxDirectionalLights)) {
        error = makeError("ERR_SNAPSHOT_WRITE", "too many lights for snapshot", pathValue);
        setHeaderVersions(error, snapshot.formatVersion, snapshot.lightVersion);
        return false;
    }

    const std::uint64_t pixels = static_cast<std::uint64_t>(snapshot.width) *
                                 static_cast<std::uint64_t>(snapshot.height);
    std::uint64_t worldBytes = 0;
    std::uint64_t normalBytes = 0;
    std::uint64_t albedoBytes = 0;
    std::uint64_t depthBytes = 0;
    std::uint64_t resultBytes = 0;
    if (!checkedProduct(pixels, sizeof(float) * 4U, worldBytes) ||
        !checkedProduct(pixels, sizeof(float) * 4U, normalBytes) ||
        !checkedProduct(pixels, 4U, albedoBytes) ||
        !checkedProduct(pixels, sizeof(float), depthBytes) ||
        !checkedProduct(pixels, 4U, resultBytes)) {
        error = makeError("ERR_SNAPSHOT_INVALID_GBUFFER", "snapshot dimensions overflow", pathValue);
        setHeaderVersions(error, snapshot.formatVersion, snapshot.lightVersion);
        return false;
    }

    const std::size_t expectedWorldFloats = static_cast<std::size_t>(pixels) * 4U;
    const std::size_t expectedNormalFloats = static_cast<std::size_t>(pixels) * 4U;
    const std::size_t expectedAlbedoBytes = static_cast<std::size_t>(pixels) * 4U;
    const std::size_t expectedDepthFloats = static_cast<std::size_t>(pixels);
    const std::size_t expectedResultBytes = static_cast<std::size_t>(pixels) * 4U;
    if (snapshot.gbuffer.world.size() != expectedWorldFloats ||
        snapshot.gbuffer.normal.size() != expectedNormalFloats ||
        snapshot.gbuffer.albedo.size() != expectedAlbedoBytes ||
        snapshot.gbuffer.depth.size() != expectedDepthFloats ||
        snapshot.gbuffer.resultRgbaTopDown.size() != expectedResultBytes) {
        error = makeError(
            "ERR_SNAPSHOT_INVALID_GBUFFER",
            "G-Buffer payload size mismatch: world=" + std::to_string(snapshot.gbuffer.world.size()) +
                "/" + std::to_string(expectedWorldFloats) + " normal=" +
                std::to_string(snapshot.gbuffer.normal.size()) + "/" +
                std::to_string(expectedNormalFloats) + " albedo=" +
                std::to_string(snapshot.gbuffer.albedo.size()) + "/" +
                std::to_string(expectedAlbedoBytes) + " depth=" +
                std::to_string(snapshot.gbuffer.depth.size()) + "/" +
                std::to_string(expectedDepthFloats) + " result=" +
                std::to_string(snapshot.gbuffer.resultRgbaTopDown.size()) + "/" +
                std::to_string(expectedResultBytes),
            pathValue);
        setHeaderVersions(error, snapshot.formatVersion, snapshot.lightVersion);
        return false;
    }

    std::vector<std::uint8_t> payload;
    payload.reserve(static_cast<std::size_t>(worldBytes + normalBytes + albedoBytes + depthBytes +
                                            resultBytes + 4096));
    for (const Light& light : snapshot.lights) {
        appendU32(payload, static_cast<std::uint32_t>(light.type));
        appendU32(payload, static_cast<std::uint32_t>(light.id.size()));
        payload.insert(payload.end(), light.id.begin(), light.id.end());
        appendFloat(payload, light.positionOrDirection.x);
        appendFloat(payload, light.positionOrDirection.y);
        appendFloat(payload, light.positionOrDirection.z);
        appendFloat(payload, light.color.x);
        appendFloat(payload, light.color.y);
        appendFloat(payload, light.color.z);
        appendFloat(payload, light.intensity);
        appendFloat(payload, light.radius);
    }

    auto appendFloats = [&payload](const std::vector<float>& values) {
        for (float value : values) {
            appendFloat(payload, value);
        }
    };
    appendFloats(snapshot.gbuffer.world);
    appendFloats(snapshot.gbuffer.normal);
    payload.insert(payload.end(), snapshot.gbuffer.albedo.begin(), snapshot.gbuffer.albedo.end());
    appendFloats(snapshot.gbuffer.depth);
    payload.insert(payload.end(), snapshot.gbuffer.resultRgbaTopDown.begin(),
                   snapshot.gbuffer.resultRgbaTopDown.end());

    const std::uint64_t payloadSize = payload.size();
    const std::uint64_t checksum = snapshotChecksum(payload.data(), payload.size());
    const std::uint64_t fileSize = kSnapshotHeaderSize + payloadSize;
    if (fileSize > kSnapshotMaxFileBytes) {
        error = makeError("ERR_SNAPSHOT_WRITE", "snapshot exceeds maximum supported size", pathValue);
        setHeaderVersions(error, snapshot.formatVersion, snapshot.lightVersion);
        return false;
    }

    std::vector<std::uint8_t> header(kSnapshotHeaderSize, 0);
    std::copy(kMagic.begin(), kMagic.end(), header.begin());
    writeU32(header, 8, snapshot.formatVersion);
    writeU32(header, 12, kSnapshotHeaderSize);
    writeU64(header, 16, fileSize);
    writeU64(header, 24, snapshot.lightVersion);
    writeU32(header, 32, static_cast<std::uint32_t>(snapshot.width));
    writeU32(header, 36, static_cast<std::uint32_t>(snapshot.height));
    writeU32(header, 40, static_cast<std::uint32_t>(pointCount));
    writeU32(header, 44, static_cast<std::uint32_t>(directionalCount));
    writeU32(header, 48, 3);
    writeU32(header, 52, 1);
    writeU64(header, 56, payloadSize);
    writeU64(header, 64, checksum);

    const auto sceneBytes = std::min<std::size_t>(snapshot.sceneId.size(), 31);
    std::copy_n(snapshot.sceneId.data(), sceneBytes, header.begin() + 72);
    writeFloat(header, 104, snapshot.cameraPosition.x);
    writeFloat(header, 108, snapshot.cameraPosition.y);
    writeFloat(header, 112, snapshot.cameraPosition.z);
    writeFloat(header, 116, snapshot.exposure);
    writeFloat(header, 128, snapshot.ambient.x);
    writeFloat(header, 132, snapshot.ambient.y);
    writeFloat(header, 136, snapshot.ambient.z);

    const std::filesystem::path destination(pathValue);
    if (std::filesystem::exists(destination)) {
        error = makeError("ERR_SNAPSHOT_EXISTS", "refusing to overwrite existing snapshot", pathValue);
        setHeaderVersions(error, snapshot.formatVersion, snapshot.lightVersion);
        return false;
    }
    if (destination.has_parent_path()) {
        std::error_code directoryError;
        std::filesystem::create_directories(destination.parent_path(), directoryError);
        if (directoryError) {
            error = makeError("ERR_SNAPSHOT_WRITE", "cannot create snapshot directory", pathValue);
            setHeaderVersions(error, snapshot.formatVersion, snapshot.lightVersion);
            return false;
        }
    }

    const auto suffix = static_cast<unsigned long long>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const std::filesystem::path temporary =
        destination.parent_path() / (destination.filename().string() + ".tmp." + std::to_string(suffix));
    TempFileGuard guard(temporary);

    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            error = makeError("ERR_SNAPSHOT_WRITE", "cannot open temporary snapshot file", pathValue);
            setHeaderVersions(error, snapshot.formatVersion, snapshot.lightVersion);
            return false;
        }
        output.write(reinterpret_cast<const char*>(header.data()),
                     static_cast<std::streamsize>(header.size()));
        output.write(reinterpret_cast<const char*>(payload.data()),
                     static_cast<std::streamsize>(payload.size()));
        output.flush();
        if (!output) {
            error = makeError("ERR_SNAPSHOT_WRITE", "snapshot write failed", pathValue);
            setHeaderVersions(error, snapshot.formatVersion, snapshot.lightVersion);
            return false;
        }
    }

    std::error_code renameError;
    std::filesystem::rename(temporary, destination, renameError);
    if (renameError) {
        error = makeError("ERR_SNAPSHOT_WRITE", "cannot commit snapshot file", pathValue);
        setHeaderVersions(error, snapshot.formatVersion, snapshot.lightVersion);
        return false;
    }
    guard.commit();
    return true;
}

bool loadSnapshot(const std::string& path, SnapshotData& snapshot, SnapshotError& error) {
    error = makeError("", "", path);
    snapshot = SnapshotData{};

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = makeError("ERR_SNAPSHOT_NOT_FOUND", "cannot open snapshot file", path);
        return false;
    }

    const std::streamoff end = input.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) > kSnapshotMaxFileBytes) {
        error = makeError("ERR_SNAPSHOT_CORRUPT", "snapshot size is invalid", path);
        return false;
    }
    const std::size_t actualSize = static_cast<std::size_t>(end);
    if (actualSize < kSnapshotHeaderSize) {
        error = makeError("ERR_SNAPSHOT_CORRUPT", "snapshot is truncated before header", path);
        return false;
    }

    std::vector<std::uint8_t> file(actualSize);
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(file.data()), static_cast<std::streamsize>(file.size()));
    if (!input) {
        error = makeError("ERR_SNAPSHOT_READ", "snapshot read failed", path);
        return false;
    }

    if (!std::equal(kMagic.begin(), kMagic.end(), file.begin())) {
        error = makeError("ERR_SNAPSHOT_CORRUPT", "snapshot magic does not match", path);
        return false;
    }

    const std::uint32_t formatVersion = readU32(file, 8);
    const std::uint32_t headerSize = readU32(file, 12);
    const std::uint64_t lightVersion = readU64(file, 24);
    if (headerSize != kSnapshotHeaderSize) {
        error = makeError("ERR_SNAPSHOT_CORRUPT", "snapshot header size does not match", path);
        setHeaderVersions(error, formatVersion, lightVersion);
        return false;
    }
    if (formatVersion != kSnapshotFormatVersion) {
        error = makeError("ERR_SNAPSHOT_VERSION", "snapshot format version is unsupported", path);
        setHeaderVersions(error, formatVersion, lightVersion);
        return false;
    }

    const std::uint64_t declaredFileSize = readU64(file, 16);
    const std::uint32_t width = readU32(file, 32);
    const std::uint32_t height = readU32(file, 36);
    const std::uint32_t pointCount = readU32(file, 40);
    const std::uint32_t directionalCount = readU32(file, 44);
    const std::uint64_t payloadSize = readU64(file, 56);
    const std::uint64_t expectedChecksum = readU64(file, 64);

    setHeaderVersions(error, formatVersion, lightVersion);
    if (declaredFileSize != actualSize || payloadSize != actualSize - kSnapshotHeaderSize) {
        error = makeError("ERR_SNAPSHOT_CORRUPT", "snapshot file size metadata mismatch", path);
        setHeaderVersions(error, formatVersion, lightVersion);
        return false;
    }
    if (width == 0 || height == 0 || width > 4096U || height > 4096U) {
        error = makeError("ERR_SNAPSHOT_INVALID_GBUFFER", "snapshot dimensions are invalid", path);
        setHeaderVersions(error, formatVersion, lightVersion);
        return false;
    }
    if (pointCount > static_cast<std::uint32_t>(kMaxPointLights) ||
        directionalCount > static_cast<std::uint32_t>(kMaxDirectionalLights)) {
        error = makeError("ERR_SNAPSHOT_CORRUPT", "snapshot light count exceeds capacity", path);
        setHeaderVersions(error, formatVersion, lightVersion);
        return false;
    }

    const std::uint8_t* payload = file.data() + kSnapshotHeaderSize;
    if (snapshotChecksum(payload, static_cast<std::size_t>(payloadSize)) != expectedChecksum) {
        error = makeError("ERR_SNAPSHOT_CORRUPT", "snapshot payload checksum mismatch", path);
        setHeaderVersions(error, formatVersion, lightVersion);
        return false;
    }

    const std::uint64_t pixels = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    std::uint64_t worldBytes = 0;
    std::uint64_t normalBytes = 0;
    std::uint64_t albedoBytes = 0;
    std::uint64_t depthBytes = 0;
    std::uint64_t resultBytes = 0;
    if (!checkedProduct(pixels, sizeof(float) * 4U, worldBytes) ||
        !checkedProduct(pixels, sizeof(float) * 4U, normalBytes) ||
        !checkedProduct(pixels, 4U, albedoBytes) ||
        !checkedProduct(pixels, sizeof(float), depthBytes) ||
        !checkedProduct(pixels, 4U, resultBytes)) {
        error = makeError("ERR_SNAPSHOT_INVALID_GBUFFER", "snapshot dimensions overflow", path);
        setHeaderVersions(error, formatVersion, lightVersion);
        return false;
    }

    std::size_t cursor = 0;
    std::vector<Light> lights;
    lights.reserve(static_cast<std::size_t>(pointCount + directionalCount));
    for (std::uint32_t index = 0; index < pointCount + directionalCount; ++index) {
        if (cursor + 8 > payloadSize) {
            error = makeError("ERR_SNAPSHOT_CORRUPT", "truncated light record", path);
            setHeaderVersions(error, formatVersion, lightVersion);
            return false;
        }
        const std::uint32_t type = readU32(file, kSnapshotHeaderSize + cursor);
        const std::uint32_t idLength = readU32(file, kSnapshotHeaderSize + cursor + 4);
        cursor += 8;
        if (idLength == 0 || idLength > 64 || cursor + idLength + 32 > payloadSize) {
            error = makeError("ERR_SNAPSHOT_CORRUPT", "invalid light record length", path);
            setHeaderVersions(error, formatVersion, lightVersion);
            return false;
        }
        if (type > static_cast<std::uint32_t>(LightType::Directional)) {
            error = makeError("ERR_SNAPSHOT_CORRUPT", "invalid light type", path);
            setHeaderVersions(error, formatVersion, lightVersion);
            return false;
        }
        Light light;
        light.id.assign(reinterpret_cast<const char*>(payload + cursor), idLength);
        cursor += idLength;
        light.type = static_cast<LightType>(type);
        light.positionOrDirection = {readFloat(file, kSnapshotHeaderSize + cursor),
                                     readFloat(file, kSnapshotHeaderSize + cursor + 4),
                                     readFloat(file, kSnapshotHeaderSize + cursor + 8)};
        light.color = {readFloat(file, kSnapshotHeaderSize + cursor + 12),
                       readFloat(file, kSnapshotHeaderSize + cursor + 16),
                       readFloat(file, kSnapshotHeaderSize + cursor + 20)};
        light.intensity = readFloat(file, kSnapshotHeaderSize + cursor + 24);
        light.radius = readFloat(file, kSnapshotHeaderSize + cursor + 28);
        cursor += 32;
        lights.push_back(std::move(light));
    }

    const std::uint64_t gbufferBytes = worldBytes + normalBytes + albedoBytes + depthBytes + resultBytes;
    if (cursor + gbufferBytes != payloadSize) {
        error = makeError("ERR_SNAPSHOT_CORRUPT", "snapshot payload length mismatch", path);
        setHeaderVersions(error, formatVersion, lightVersion);
        return false;
    }
    if (width != static_cast<std::uint32_t>(kRenderWidth) ||
        height != static_cast<std::uint32_t>(kRenderHeight)) {
        error = makeError("ERR_SNAPSHOT_INVALID_GBUFFER", "snapshot resolution is not 800x600", path);
        setHeaderVersions(error, formatVersion, lightVersion);
        return false;
    }

    snapshot.formatVersion = formatVersion;
    snapshot.lightVersion = lightVersion;
    snapshot.width = static_cast<int>(width);
    snapshot.height = static_cast<int>(height);
    const char* sceneIdBegin = reinterpret_cast<const char*>(file.data() + 72);
    std::size_t sceneIdLength = 0;
    while (sceneIdLength < 32 && sceneIdBegin[sceneIdLength] != '\0') {
        ++sceneIdLength;
    }
    snapshot.sceneId.assign(sceneIdBegin, sceneIdLength);
    snapshot.cameraPosition = {readFloat(file, 104), readFloat(file, 108), readFloat(file, 112)};
    snapshot.exposure = readFloat(file, 116);
    snapshot.ambient = {readFloat(file, 128), readFloat(file, 132), readFloat(file, 136)};
    snapshot.lights = std::move(lights);

    auto readFloats = [&](std::size_t byteCount, std::vector<float>& values) {
        values.resize(static_cast<std::size_t>(byteCount / sizeof(float)));
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = readFloat(file, kSnapshotHeaderSize + cursor + index * sizeof(float));
        }
        cursor += static_cast<std::size_t>(byteCount);
    };

    readFloats(worldBytes, snapshot.gbuffer.world);
    readFloats(normalBytes, snapshot.gbuffer.normal);
    snapshot.gbuffer.albedo.resize(static_cast<std::size_t>(albedoBytes));
    std::copy_n(payload + cursor, snapshot.gbuffer.albedo.size(), snapshot.gbuffer.albedo.begin());
    cursor += static_cast<std::size_t>(albedoBytes);
    readFloats(depthBytes, snapshot.gbuffer.depth);
    snapshot.gbuffer.resultRgbaTopDown.resize(static_cast<std::size_t>(resultBytes));
    std::copy_n(payload + cursor, snapshot.gbuffer.resultRgbaTopDown.size(),
                snapshot.gbuffer.resultRgbaTopDown.begin());
    snapshot.gbuffer.width = snapshot.width;
    snapshot.gbuffer.height = snapshot.height;

    std::string validationError;
    if (!validateLights(snapshot.lights, validationError)) {
        error = makeError("ERR_SNAPSHOT_CORRUPT", validationError, path);
        setHeaderVersions(error, formatVersion, lightVersion);
        return false;
    }
    return true;
}

std::string versionedSnapshotPath(const std::string& basePathValue, std::uint64_t version) {
    const std::filesystem::path base(basePathValue);
    std::string filename = base.stem().string();
    const std::string suffix = base.has_extension() ? base.extension().string() : ".dgsnap";

    const std::string marker = ".v";
    const std::size_t markerPosition = filename.rfind(marker);
    if (markerPosition != std::string::npos) {
        const std::string tail = filename.substr(markerPosition + marker.size());
        if (!tail.empty() && std::all_of(tail.begin(), tail.end(),
                                         [](char value) { return value >= '0' && value <= '9'; })) {
            filename = filename.substr(0, markerPosition);
        }
    }
    return (base.parent_path() / (filename + ".v" + std::to_string(version) + suffix)).string();
}

std::uint64_t nextAvailableLightVersion(std::uint64_t desiredVersion, const std::string& basePathValue) {
    std::uint64_t version = std::max<std::uint64_t>(desiredVersion, 1);
    while (std::filesystem::exists(versionedSnapshotPath(basePathValue, version))) {
        ++version;
    }
    return version;
}

} // namespace dr
