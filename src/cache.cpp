#include "cache.hpp"

#include "atomic_io.hpp"
#include "errors.hpp"
#include "json.hpp"
#include "sha256.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <utility>

namespace lfb {

namespace {

std::string frameStem(std::uint32_t frameIndex) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "frame_%06u", frameIndex);
    return buffer;
}

bool parseIntegral(const json::Value* value, std::uint64_t& output) {
    if (value == nullptr || !value->isNumber()) {
        return false;
    }
    const double number = value->asNumber();
    if (number < 0.0 || number > 9007199254740992.0 ||
        static_cast<double>(static_cast<std::uint64_t>(number)) != number) {
        return false;
    }
    output = static_cast<std::uint64_t>(number);
    return true;
}

}  // namespace

FrameCache::FrameCache(std::filesystem::path root, std::string globalKey)
    : root_(std::move(root)), globalKey_(std::move(globalKey)) {}

std::filesystem::path FrameCache::entryDirectory(const std::string& subKey) const {
    return root_ / globalKey_ / subKey;
}

void FrameCache::cleanHalfProducts() const {
    io::cleanupTemporaryFiles(root_);
}

std::optional<Image> FrameCache::load(std::uint32_t frameIndex,
                                      const std::string& subKey,
                                      int width,
                                      int height) const {
    const std::filesystem::path directory = entryDirectory(subKey);
    const std::uint32_t targetFrame = frameIndex;
    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec) || ec) {
        return std::nullopt;
    }

    const std::string stem = frameStem(targetFrame);
    const std::filesystem::path metadataPath = directory / (stem + ".meta.json");
    const std::filesystem::path pixelsPath = directory / (stem + ".rgba");
    if (!std::filesystem::is_regular_file(metadataPath, ec) || ec ||
        !std::filesystem::is_regular_file(pixelsPath, ec) || ec) {
        return std::nullopt;
    }

    std::string metadataText;
    std::string readError;
    if (!io::readText(metadataPath, metadataText, readError)) {
        return std::nullopt;
    }

    json::Value metadata;
    try {
        metadata = json::parse(metadataText);
    } catch (const json::ParseError&) {
        return std::nullopt;
    }
    if (!metadata.isObject()) {
        return std::nullopt;
    }

    std::uint64_t recordedIndex = 0;
    std::uint64_t recordedWidth = 0;
    std::uint64_t recordedHeight = 0;
    std::uint64_t recordedBytes = 0;
    const json::Value* recordedSubKey = metadata.find("subkey");
    const json::Value* recordedChecksum = metadata.find("pixel_sha256");
    if (!parseIntegral(metadata.find("frame_index"), recordedIndex) ||
        !parseIntegral(metadata.find("width"), recordedWidth) ||
        !parseIntegral(metadata.find("height"), recordedHeight) ||
        !parseIntegral(metadata.find("bytes"), recordedBytes) ||
        recordedSubKey == nullptr || !recordedSubKey->isString() ||
        recordedChecksum == nullptr || !recordedChecksum->isString()) {
        return std::nullopt;
    }
    if (recordedIndex != frameIndex ||
        recordedSubKey->asString() != subKey ||
        recordedWidth != static_cast<std::uint64_t>(width) ||
        recordedHeight != static_cast<std::uint64_t>(height)) {
        return std::nullopt;
    }

    const std::uint64_t expectedSize =
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) * 4u;
    if (recordedBytes != expectedSize) {
        return std::nullopt;
    }

    std::vector<std::uint8_t> pixels;
    if (!io::readBytes(pixelsPath, pixels, readError)) {
        return std::nullopt;
    }
    if (pixels.size() != expectedSize) {
        return std::nullopt;
    }
    if (Sha256::hash(pixels.data(), pixels.size()) != recordedChecksum->asString()) {
        return std::nullopt;
    }

    Image image(width, height);
    image.rgba = std::move(pixels);
    return image;
}

void FrameCache::commit(std::uint32_t frameIndex,
                        const std::string& subKey,
                        const Image& image) const {
    const std::filesystem::path directory = entryDirectory(subKey);
    const std::string stem = frameStem(frameIndex);
    const std::string checksum = pixelChecksum(image);

    std::ostringstream metadata;
    metadata << "{\n";
    metadata << "  \"version\": 1,\n";
    metadata << "  \"frame_index\": " << frameIndex << ",\n";
    metadata << "  \"subkey\": \"" << subKey << "\",\n";
    metadata << "  \"width\": " << image.width << ",\n";
    metadata << "  \"height\": " << image.height << ",\n";
    metadata << "  \"bytes\": " << image.rgba.size() << ",\n";
    metadata << "  \"pixel_sha256\": \"" << checksum << "\"\n";
    metadata << "}\n";
    const std::string metadataText = metadata.str();

    std::string writeError;
    if (!io::atomicWriteBytes(directory / (stem + ".rgba"), image.rgba, writeError)) {
        throw BusinessError(ExitCode::E_CACHE_IO,
                            "cache write failed for frame " + std::to_string(frameIndex) +
                                ": " + writeError);
    }
    const std::vector<std::uint8_t> metadataBytes(metadataText.begin(), metadataText.end());
    if (!io::atomicWriteBytes(directory / (stem + ".meta.json"), metadataBytes, writeError)) {
        throw BusinessError(ExitCode::E_CACHE_IO,
                            "cache metadata write failed for frame " +
                                std::to_string(frameIndex) + ": " + writeError);
    }
}

}  // namespace lfb
