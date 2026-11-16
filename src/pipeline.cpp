#include "pipeline.hpp"

#include "json.hpp"
#include "render.hpp"
#include "scene.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace tilegen {

namespace {

namespace fs = std::filesystem;

class OutputFailure : public std::runtime_error {
public:
    explicit OutputFailure(const std::string& message) : std::runtime_error(message) {}
};

class DiskFullFailure : public OutputFailure {
public:
    explicit DiskFullFailure(const std::string& message) : OutputFailure(message) {}
};

class Logger {
public:
    Logger(const fs::path& path, bool append) : path_(path) {
        std::ios::openmode mode = std::ios::out | std::ios::trunc;
        if (append) {
            mode = std::ios::out | std::ios::app;
        }
        stream_.open(path_, mode);
    }

    bool available() const noexcept { return stream_.is_open(); }

    void info(const std::string& stage, const std::string& message) {
        write("INFO", stage, message);
    }

    void warning(const std::string& stage, const std::string& message) {
        write("WARN", stage, message);
    }

    void error(const std::string& stage, const std::string& message) {
        write("ERROR", stage, message);
        std::cerr << "[ERROR] [" << stage << "] " << message << '\n';
    }

private:
    void write(const char* level, const std::string& stage,
               const std::string& message) {
        const std::string line =
            std::string("[") + level + "] [" + stage + "] " + message;
        if (stream_.is_open()) {
            stream_ << line << '\n';
            stream_.flush();
        }
    }

    fs::path path_;
    std::ofstream stream_;
};

struct TileKey {
    int level = 0;
    int x = 0;
    int y = 0;

    bool operator<(const TileKey& other) const {
        return std::tie(level, x, y) < std::tie(other.level, other.x, other.y);
    }
};

struct TileRecord {
    int level = 0;
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    Bounds bbox;
    std::vector<std::string> layerIds;
    std::map<std::string, Bounds> layerBboxes;
    bool bboxCovered = false;
    bool empty = true;
    bool present = false;
    std::string path;
    std::string pngPath;
    std::uint32_t crc32Value = 0;
    std::uint32_t pngCrc32Value = 0;
    std::uint64_t byteLength = 0;
};

struct LevelInfo {
    int level = 0;
    double scale = 1.0;
    int tilesX = 0;
    int tilesY = 0;
    bool completed = false;
};

struct ManifestState {
    bool loaded = false;
    bool validForResume = false;
    int startLevel = 0;
    int tileSize = 0;
    int levels = 0;
    std::uint32_t sceneCrc32 = 0;
    std::set<int> completedLevels;
    std::map<TileKey, TileRecord> tiles;
};

std::string escapeJson(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch) << std::dec;
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

void writeJsonNumber(std::ostringstream& out, double value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("cannot serialize non-finite number");
    }
    if (std::abs(value - std::round(value)) < 1e-9) {
        out << std::fixed << std::setprecision(0) << std::round(value);
    } else {
        out << std::fixed << std::setprecision(6) << value;
    }
}

std::string formatBounds(const Bounds& bounds) {
    std::ostringstream out;
    out << '[';
    writeJsonNumber(out, bounds.minX);
    out << ',';
    writeJsonNumber(out, bounds.minY);
    out << ',';
    writeJsonNumber(out, bounds.maxX);
    out << ',';
    writeJsonNumber(out, bounds.maxY);
    out << ']';
    return out.str();
}

std::string relativePathString(const fs::path& path, const fs::path& root) {
    std::error_code error;
    const fs::path relative = fs::relative(path, root, error);
    if (error) {
        return path.string();
    }
    return relative.generic_string();
}

std::string tileRawPath(const fs::path& output, int level, int x, int y) {
    return (output / "tiles" / std::to_string(level) / std::to_string(x) /
            (std::to_string(y) + ".rgba"))
        .string();
}

std::string tilePngPath(const fs::path& output, int level, int x, int y) {
    return (output / "tiles" / std::to_string(level) / std::to_string(x) /
            (std::to_string(y) + ".png"))
        .string();
}

std::uint32_t parseCrc32(const JsonValue& value, const std::string& path) {
    if (!value.isString()) {
        throw SceneError(path, "checksum must be an 8-digit hexadecimal string");
    }
    const std::string& text = value.asString();
    if (text.empty() || text.size() > 8) {
        throw SceneError(path, "checksum must contain up to 8 hexadecimal digits");
    }
    std::uint32_t result = 0;
    for (char ch : text) {
        result <<= 4U;
        if (ch >= '0' && ch <= '9') {
            result += static_cast<std::uint32_t>(ch - '0');
        } else if (ch >= 'a' && ch <= 'f') {
            result += static_cast<std::uint32_t>(ch - 'a' + 10);
        } else if (ch >= 'A' && ch <= 'F') {
            result += static_cast<std::uint32_t>(ch - 'A' + 10);
        } else {
            throw SceneError(path, "invalid hexadecimal checksum");
        }
    }
    return result;
}

int requireIntegerField(const JsonValue& object, const std::string& key,
                        const std::string& path) {
    const JsonValue* value = object.find(key);
    if (value == nullptr || !value->isNumber()) {
        throw SceneError(jsonPath(path, key), "required integer field is missing");
    }
    const double number = value->asNumber();
    if (!std::isfinite(number) || number < 0.0 ||
        std::floor(number) != number || number > 2147483647.0) {
        throw SceneError(jsonPath(path, key), "must be a non-negative integer");
    }
    return static_cast<int>(number);
}

Bounds parseBounds(const JsonValue& value, const std::string& path) {
    if (!value.isArray() || value.asArray().size() != 4) {
        throw SceneError(path, "bbox must be [minX, minY, maxX, maxY]");
    }
    Bounds result;
    for (std::size_t i = 0; i < 4; ++i) {
        if (!value.asArray()[i].isNumber()) {
            throw SceneError(jsonPath(path, i), "bbox component must be a number");
        }
    }
    result.minX = value.asArray()[0].asNumber();
    result.minY = value.asArray()[1].asNumber();
    result.maxX = value.asArray()[2].asNumber();
    result.maxY = value.asArray()[3].asNumber();
    result.valid = true;
    return result;
}

ManifestState readManifest(const fs::path& path, const Scene& scene,
                           const PipelineOptions& options, Logger& logger) {
    ManifestState state;
    JsonParser parser;
    JsonValue root = parser.parseFile(path.string());
    if (!root.isObject()) {
        throw SceneError("$", "manifest root must be an object");
    }
    const JsonValue* schema = root.find("schema");
    if (schema == nullptr || !schema->isString() ||
        schema->asString() != "tilegen-manifest-v1") {
        throw SceneError("$.schema", "unsupported manifest schema");
    }
    const JsonValue* tileSize = root.find("tile_size");
    const JsonValue* levels = root.find("levels");
    const JsonValue* sceneObject = root.find("scene");
    if (tileSize == nullptr || !tileSize->isNumber() || levels == nullptr ||
        !levels->isNumber() || sceneObject == nullptr || !sceneObject->isObject()) {
        throw SceneError("$", "manifest is missing tile_size, levels, or scene");
    }
    state.tileSize = static_cast<int>(tileSize->asNumber());
    state.levels = static_cast<int>(levels->asNumber());
    const JsonValue* crc = sceneObject->find("crc32");
    if (crc == nullptr) {
        throw SceneError("$.scene.crc32", "required checksum is missing");
    }
    state.sceneCrc32 = parseCrc32(*crc, "$.scene.crc32");
    if (state.tileSize != options.tileSize || state.levels != options.levels) {
        throw SceneError(
            "$",
            "manifest configuration does not match requested tile size or levels");
    }
    if (state.sceneCrc32 != scene.sourceCrc32) {
        throw SceneError(
            "$.scene.crc32",
            "manifest scene checksum does not match the requested scene");
    }

    if (const JsonValue* completed = root.find("completed_levels")) {
        if (!completed->isArray()) {
            throw SceneError("$.completed_levels", "must be an array");
        }
        for (const JsonValue& value : completed->asArray()) {
            if (!value.isNumber()) {
                throw SceneError("$.completed_levels", "entries must be integers");
            }
            const int level = static_cast<int>(value.asNumber());
            if (level >= 0 && level < options.levels) {
                state.completedLevels.insert(level);
            }
        }
    }

    if (const JsonValue* tiles = root.find("tiles")) {
        if (!tiles->isArray()) {
            throw SceneError("$.tiles", "must be an array");
        }
        for (std::size_t i = 0; i < tiles->asArray().size(); ++i) {
            const std::string tilePath = jsonPath("$.tiles", i);
            const JsonValue& value = tiles->asArray()[i];
            if (!value.isObject()) {
                throw SceneError(tilePath, "tile record must be an object");
            }
            TileRecord record;
            record.level = requireIntegerField(value, "level", tilePath);
            record.x = requireIntegerField(value, "x", tilePath);
            record.y = requireIntegerField(value, "y", tilePath);
            if (record.level >= options.levels || record.level < 0) {
                continue;
            }
            if (const JsonValue* bbox = value.find("bbox")) {
                record.bbox = parseBounds(*bbox, jsonPath(tilePath, "bbox"));
            }
            if (const JsonValue* ids = value.find("layer_ids")) {
                if (!ids->isArray()) {
                    throw SceneError(jsonPath(tilePath, "layer_ids"),
                                     "must be an array");
                }
                for (const JsonValue& id : ids->asArray()) {
                    if (id.isString()) {
                        record.layerIds.push_back(id.asString());
                    }
                }
            }
            if (const JsonValue* bboxes = value.find("layer_bboxes")) {
                if (bboxes->isObject()) {
                    for (const auto& entry : bboxes->asObject()) {
                        record.layerBboxes[entry.first] =
                            parseBounds(entry.second,
                                        jsonPath(jsonPath(tilePath, "layer_bboxes"),
                                                 entry.first));
                    }
                }
            }
            if (const JsonValue* covered = value.find("bbox_covered")) {
                record.bboxCovered =
                    covered->isBool() ? covered->asBool() : false;
            }
            if (const JsonValue* empty = value.find("empty")) {
                record.empty = empty->isBool() ? empty->asBool() : true;
            }
            if (const JsonValue* present = value.find("present")) {
                record.present = present->isBool() ? present->asBool() : false;
            }
            if (const JsonValue* rawPath = value.find("path")) {
                if (rawPath->isString()) {
                    record.path = rawPath->asString();
                }
            }
            if (const JsonValue* pngPath = value.find("png_path")) {
                if (pngPath->isString()) {
                    record.pngPath = pngPath->asString();
                }
            }
            if (const JsonValue* crcValue = value.find("crc32")) {
                if (crcValue->isString()) {
                    record.crc32Value =
                        parseCrc32(*crcValue, jsonPath(tilePath, "crc32"));
                }
            }
            if (const JsonValue* pngCrc = value.find("png_crc32")) {
                if (pngCrc->isString()) {
                    record.pngCrc32Value =
                        parseCrc32(*pngCrc, jsonPath(tilePath, "png_crc32"));
                }
            }
            if (const JsonValue* bytes = value.find("byte_length")) {
                if (bytes->isNumber() &&
                    bytes->asNumber() >= 0.0 &&
                    bytes->asNumber() <
                        static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
                    record.byteLength =
                        static_cast<std::uint64_t>(bytes->asNumber());
                }
            }
            if (record.x >= 0 && record.y >= 0) {
                state.tiles[TileKey{record.level, record.x, record.y}] = record;
            }
        }
    }

    state.loaded = true;
    state.validForResume = true;
    logger.info("resume", "loaded manifest with " +
                              std::to_string(state.tiles.size()) + " tile records");
    return state;
}

const TileRecord* findReusableTile(const ManifestState& state,
                                   const PipelineOptions& options,
                                   const fs::path& output, const TileKey& key,
                                   Logger& logger) {
    if (!state.validForResume) {
        return nullptr;
    }
    auto it = state.tiles.find(key);
    if (it == state.tiles.end()) {
        return nullptr;
    }
    const TileRecord& record = it->second;
    if (!record.present) {
        return record.empty ? &record : nullptr;
    }
    if (record.path.empty()) {
        return nullptr;
    }
    const fs::path rawPath =
        fs::path(record.path).is_absolute() ? fs::path(record.path)
                                            : output / record.path;
    std::error_code error;
    if (!fs::is_regular_file(rawPath, error)) {
        logger.warning("resume", "tile file is missing: " + rawPath.string());
        return nullptr;
    }
    const std::uint64_t expectedBytes =
        static_cast<std::uint64_t>(options.tileSize) * options.tileSize * 4U;
    const std::uint64_t size = fs::file_size(rawPath, error);
    if (error || size != expectedBytes) {
        logger.warning("resume", "tile size mismatch: " + rawPath.string());
        return nullptr;
    }
    std::vector<std::uint8_t> bytes;
    try {
        bytes = readBinaryFile(rawPath.string());
    } catch (const std::exception& errorValue) {
        logger.warning("resume", std::string("cannot read tile: ") + errorValue.what());
        return nullptr;
    }
    if (crc32(bytes) != record.crc32Value) {
        logger.warning("resume", "tile checksum mismatch: " + rawPath.string());
        return nullptr;
    }

    const fs::path pngPath =
        record.pngPath.empty()
            ? fs::path(tilePngPath(output, key.level, key.x, key.y))
            : (fs::path(record.pngPath).is_absolute() ? fs::path(record.pngPath)
                                                      : output / record.pngPath);
    bool needPng = true;
    if (fs::is_regular_file(pngPath, error)) {
        try {
            const std::vector<std::uint8_t> pngBytes =
                readBinaryFile(pngPath.string());
            if (crc32(pngBytes) == record.pngCrc32Value) {
                needPng = false;
            }
        } catch (const std::exception&) {
            needPng = true;
        }
    }
    if (needPng) {
        std::vector<Rgba8> pixels;
        pixels.reserve(bytes.size() / 4U);
        for (std::size_t i = 0; i + 3 < bytes.size(); i += 4) {
            pixels.push_back(
                {bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]});
        }
        try {
            const std::vector<std::uint8_t> png =
                encodePngRgba(options.tileSize, options.tileSize, pixels);
            writeBinaryFile(pngPath.string(), png);
        } catch (const std::exception& errorValue) {
            logger.warning("resume", std::string("cannot regenerate tile PNG: ") +
                                         errorValue.what());
        }
    }
    return &record;
}

void validateCompletedLevels(ManifestState& state, const Scene& scene,
                             const PipelineOptions& options,
                             const fs::path& output, Logger& logger) {
    state.startLevel = 0;
    for (int level = 0; level < options.levels; ++level) {
        if (state.completedLevels.find(level) == state.completedLevels.end()) {
            break;
        }
        const double scale = std::ldexp(1.0, -level);
        const int tilesX = static_cast<int>(std::ceil(
            (static_cast<double>(scene.width) * scale) / options.tileSize));
        const int tilesY = static_cast<int>(std::ceil(
            (static_cast<double>(scene.height) * scale) / options.tileSize));
        bool valid = true;
        for (const auto& entry : state.tiles) {
            if (entry.first.level != level) {
                continue;
            }
            if (findReusableTile(state, options, output, entry.first, logger) ==
                nullptr) {
                valid = false;
                break;
            }
        }
        std::size_t levelRecordCount = 0;
        for (const auto& entry : state.tiles) {
            if (entry.first.level == level) {
                ++levelRecordCount;
            }
        }
        if (levelRecordCount !=
            static_cast<std::size_t>(tilesX) * static_cast<std::size_t>(tilesY)) {
            valid = false;
        }
        if (!valid) {
            logger.warning("resume", "level " + std::to_string(level) +
                                         " is incomplete on disk; resuming there");
            break;
        }
        ++state.startLevel;
    }
    for (auto it = state.completedLevels.begin(); it != state.completedLevels.end();) {
        if (*it >= state.startLevel) {
            it = state.completedLevels.erase(it);
        } else {
            ++it;
        }
    }
}

std::string makeManifest(const Scene& scene, const PipelineOptions& options,
                         const PipelineOptions& effective,
                         const std::map<TileKey, TileRecord>& tiles,
                         const std::set<int>& completedLevels,
                         const fs::path& output, const std::string& previewPath,
                         int previewLevel, int previewWidth, int previewHeight,
                         std::uint32_t previewCrc) {
    (void)effective;
    std::ostringstream out;
    out << "{\n";
    out << "  \"schema\": \"tilegen-manifest-v1\",\n";
    out << "  \"generator\": \"tilegen 1.0.0\",\n";
    out << "  \"scene\": {\"path\": \"" << escapeJson(scene.sourcePath)
        << "\", \"crc32\": \"" << crc32Hex(scene.sourceCrc32)
        << "\", \"bytes\": " << scene.sourceBytes << ", \"width\": " << scene.width
        << ", \"height\": " << scene.height << "},\n";
    out << "  \"tile_size\": " << options.tileSize << ",\n";
    out << "  \"levels\": " << options.levels << ",\n";
    out << "  \"alpha_mode\": \"straight-rgba8\",\n";
    out << "  \"checksum\": {\"algorithm\": \"crc32\", \"scope\": "
           "\"raw-rgba8-straight\", \"byte_order\": \"RGBA\"},\n";
    out << "  \"completed_levels\": [";
    bool firstCompleted = true;
    for (int level : completedLevels) {
        if (!firstCompleted) {
            out << ", ";
        }
        firstCompleted = false;
        out << level;
    }
    out << "],\n";
    out << "  \"levels_info\": [\n";
    for (int level = 0; level < options.levels; ++level) {
        const double scale = std::ldexp(1.0, -level);
        const double levelWidth = static_cast<double>(scene.width) * scale;
        const double levelHeight = static_cast<double>(scene.height) * scale;
        const int tilesX = static_cast<int>(
            std::ceil(levelWidth / static_cast<double>(options.tileSize)));
        const int tilesY = static_cast<int>(
            std::ceil(levelHeight / static_cast<double>(options.tileSize)));
        if (level != 0) {
            out << ",\n";
        }
        out << "    {\"level\": " << level << ", \"scale\": ";
        writeJsonNumber(out, scale);
        out << ", \"tiles_x\": " << tilesX << ", \"tiles_y\": " << tilesY
            << ", \"completed\": "
            << (completedLevels.count(level) != 0 ? "true" : "false") << "}";
    }
    out << "\n  ],\n";
    out << "  \"tiles\": [\n";
    bool firstTile = true;
    for (const auto& entry : tiles) {
        const TileRecord& tile = entry.second;
        if (!firstTile) {
            out << ",\n";
        }
        firstTile = false;
        out << "    {\"level\": " << tile.level << ", \"x\": " << tile.x
            << ", \"y\": " << tile.y << ", \"tile_size\": " << options.tileSize
            << ", \"pixel_width\": " << tile.width
            << ", \"pixel_height\": " << tile.height << ", \"bbox\": "
            << formatBounds(tile.bbox) << ", \"bbox_covered\": "
            << (tile.bboxCovered ? "true" : "false") << ", \"empty\": "
            << (tile.empty ? "true" : "false") << ", \"present\": "
            << (tile.present ? "true" : "false") << ", \"layer_ids\": [";
        for (std::size_t i = 0; i < tile.layerIds.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << '"' << escapeJson(tile.layerIds[i]) << '"';
        }
        out << "], \"layer_bboxes\": {";
        bool firstBbox = true;
        for (const auto& bbox : tile.layerBboxes) {
            if (!firstBbox) {
                out << ", ";
            }
            firstBbox = false;
            out << '"' << escapeJson(bbox.first) << "\": "
                << formatBounds(bbox.second);
        }
        out << "}";
        if (tile.present) {
            out << ", \"path\": \""
                << escapeJson(relativePathString(tile.path, output))
                << "\", \"png_path\": \""
                << escapeJson(relativePathString(tile.pngPath, output))
                << "\", \"crc32\": \"" << crc32Hex(tile.crc32Value)
                << "\", \"checksum\": \"" << crc32Hex(tile.crc32Value)
                << "\", \"png_crc32\": \"" << crc32Hex(tile.pngCrc32Value)
                << "\", \"byte_length\": " << tile.byteLength;
        }
        out << "}";
    }
    out << "\n  ],\n";
    out << "  \"preview\": ";
    if (previewPath.empty()) {
        out << "null\n";
    } else {
        out << "{\"level\": " << previewLevel
            << ", \"scale\": " << std::setprecision(10)
            << std::ldexp(1.0, -previewLevel) << ", \"path\": \""
            << escapeJson(relativePathString(previewPath, output))
            << "\", \"width\": " << previewWidth
            << ", \"height\": " << previewHeight
            << ", \"crc32\": \"" << crc32Hex(previewCrc)
            << "\", \"projection\": \"equirectangular\", "
               "\"projection_parameters\": {\"lon_min\": -180, \"lon_max\": 180, "
               "\"lat_min\": -90, \"lat_max\": 90, \"level\": "
            << previewLevel << "}}\n";
    }
    out << "}\n";
    return out.str();
}

void writeTextAtomic(const fs::path& path, const std::string& text) {
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw DiskFullFailure("cannot open temporary file for writing: " +
                              temporary.string());
    }
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    stream.flush();
    if (!stream) {
        stream.close();
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw DiskFullFailure("cannot write temporary file: " + temporary.string());
    }
    stream.close();
    std::error_code error;
    fs::rename(temporary, path, error);
    if (error) {
        fs::remove(path, error);
        error.clear();
        fs::rename(temporary, path, error);
        if (error) {
            throw OutputFailure("cannot replace manifest " + path.string() + ": " +
                                error.message());
        }
    }
}

void saveManifest(const fs::path& output, const std::string& manifest) {
    writeTextAtomic(output / "manifest.json", manifest);
    writeTextAtomic(output / "index.json", manifest);
}

void writeTileFiles(const fs::path& output, const TileRecord& tile,
                    const std::vector<std::uint8_t>& rgbaBytes) {
    const fs::path rawPath = fs::path(tile.path).is_absolute()
                                 ? fs::path(tile.path)
                                 : output / tile.path;
    const fs::path pngPath = fs::path(tile.pngPath).is_absolute()
                                 ? fs::path(tile.pngPath)
                                 : output / tile.pngPath;
    std::error_code error;
    fs::create_directories(rawPath.parent_path(), error);
    if (error) {
        throw OutputFailure("cannot create tile directory " +
                            rawPath.parent_path().string() + ": " +
                            error.message());
    }
    const fs::path rawTemporary = rawPath.string() + ".tmp";
    try {
        writeBinaryFile(rawTemporary.string(), rgbaBytes);
        fs::rename(rawTemporary, rawPath, error);
        if (error) {
            fs::remove(rawPath, error);
            error.clear();
            fs::rename(rawTemporary, rawPath, error);
            if (error) {
                throw OutputFailure("cannot install tile " + rawPath.string() +
                                    ": " + error.message());
            }
        }
    } catch (const std::runtime_error& failure) {
        std::error_code ignored;
        fs::remove(rawTemporary, ignored);
        throw DiskFullFailure(failure.what());
    }

    std::vector<Rgba8> pixels;
    pixels.reserve(rgbaBytes.size() / 4U);
    for (std::size_t i = 0; i + 3 < rgbaBytes.size(); i += 4) {
        pixels.push_back(
            {rgbaBytes[i], rgbaBytes[i + 1], rgbaBytes[i + 2], rgbaBytes[i + 3]});
    }
    const std::vector<std::uint8_t> png =
        encodePngRgba(tile.width, tile.height, pixels);
    const fs::path pngTemporary = pngPath.string() + ".tmp";
    try {
        writeBinaryFile(pngTemporary.string(), png);
        fs::rename(pngTemporary, pngPath, error);
        if (error) {
            fs::remove(pngPath, error);
            error.clear();
            fs::rename(pngTemporary, pngPath, error);
            if (error) {
                throw OutputFailure("cannot install tile PNG " + pngPath.string() +
                                    ": " + error.message());
            }
        }
    } catch (const std::runtime_error& failure) {
        std::error_code ignored;
        fs::remove(pngTemporary, ignored);
        throw DiskFullFailure(failure.what());
    }
}

void writeTextAtomicBinary(const fs::path& path,
                           const std::vector<std::uint8_t>& bytes);

Image resampleEquirectangular(const Image& source, int width, int height) {
    Image output(width, height);
    auto sampleClamped = [&source](double x, double y) {
        const double clampedX =
            std::clamp(x, 0.0, static_cast<double>(source.width - 1));
        const double clampedY =
            std::clamp(y, 0.0, static_cast<double>(source.height - 1));
        const int x0 = static_cast<int>(std::floor(clampedX));
        const int y0 = static_cast<int>(std::floor(clampedY));
        const int x1 = std::min(x0 + 1, source.width - 1);
        const int y1 = std::min(y0 + 1, source.height - 1);
        const double fx = clampedX - x0;
        const double fy = clampedY - y0;
        auto mix = [fx, fy](double p00, double p10, double p01, double p11) {
            return (p00 * (1.0 - fx) + p10 * fx) * (1.0 - fy) +
                   (p01 * (1.0 - fx) + p11 * fx) * fy;
        };
        const PremulPixel& p00 = source.at(x0, y0);
        const PremulPixel& p10 = source.at(x1, y0);
        const PremulPixel& p01 = source.at(x0, y1);
        const PremulPixel& p11 = source.at(x1, y1);
        return PremulPixel{
            static_cast<float>(mix(p00.r, p10.r, p01.r, p11.r)),
            static_cast<float>(mix(p00.g, p10.g, p01.g, p11.g)),
            static_cast<float>(mix(p00.b, p10.b, p01.b, p11.b)),
            static_cast<float>(mix(p00.a, p10.a, p01.a, p11.a))};
    };
    for (int y = 0; y < height; ++y) {
        const double latitude =
            90.0 - (static_cast<double>(y) + 0.5) / height * 180.0;
        const double v = (90.0 - latitude) / 180.0;
        for (int x = 0; x < width; ++x) {
            const double longitude =
                -180.0 + (static_cast<double>(x) + 0.5) / width * 360.0;
            const double u = (longitude + 180.0) / 360.0;
            output.at(x, y) =
                sampleClamped(u * source.width - 0.5, v * source.height - 0.5);
        }
    }
    return output;
}

std::string generatePreview(const Scene& scene, const PipelineOptions& options,
                            const fs::path& output, Logger& logger,
                            std::uint32_t& previewCrc) {
    const int level = options.previewLevel;
    const double scale = std::ldexp(1.0, -level);
    const int levelWidth =
        std::max(1, static_cast<int>(std::ceil(scene.width * scale)));
    const int levelHeight =
        std::max(1, static_cast<int>(std::ceil(scene.height * scale)));
    const std::uint64_t sourcePixels =
        static_cast<std::uint64_t>(levelWidth) * levelHeight;
    if (sourcePixels > 268435456ULL) {
        throw std::runtime_error(
            "preview level raster would exceed 268 million pixels; choose a "
            "higher preview level");
    }
    logger.info("preview", "rendering level " + std::to_string(level) +
                               " source raster " + std::to_string(levelWidth) +
                               "x" + std::to_string(levelHeight));
    RenderSettings settings;
    settings.samplesPerAxis = options.samplesPerAxis;
    RegionRenderResult levelImage =
        renderSceneRegion(scene, scale,
                          Bounds{0.0, 0.0, static_cast<double>(scene.width),
                                 static_cast<double>(scene.height), true},
                          levelWidth, levelHeight, settings, true);
    Image preview = resampleEquirectangular(levelImage.image, options.previewWidth,
                                            options.previewHeight);
    const std::vector<Rgba8> straight = imageToStraightRgba8(preview);
    const std::vector<std::uint8_t> png =
        encodePngRgba(options.previewWidth, options.previewHeight, straight);
    previewCrc = crc32(png);
    const fs::path previewPath = output / "preview.png";
    writeTextAtomicBinary(previewPath, png);
    logger.info("preview", "wrote " + previewPath.string() + " (" +
                               std::to_string(options.previewWidth) + "x" +
                               std::to_string(options.previewHeight) +
                               ", equirectangular, crc32=" +
                               crc32Hex(previewCrc) + ")");
    return previewPath.string();
}

void writeTextAtomicBinary(const fs::path& path,
                           const std::vector<std::uint8_t>& bytes) {
    const fs::path temporary = path.string() + ".tmp";
    try {
        writeBinaryFile(temporary.string(), bytes);
        std::error_code error;
        fs::rename(temporary, path, error);
        if (error) {
            fs::remove(path, error);
            error.clear();
            fs::rename(temporary, path, error);
            if (error) {
                throw OutputFailure("cannot install file " + path.string());
            }
        }
    } catch (const std::runtime_error& error) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw DiskFullFailure(error.what());
    }
}

int exitCodeForException(const std::exception& error, int level) {
    if (dynamic_cast<const DiskFullFailure*>(&error) != nullptr) {
        return 3;
    }
    if (dynamic_cast<const OutputFailure*>(&error) != nullptr) {
        return 2;
    }
    const SceneError* sceneError = dynamic_cast<const SceneError*>(&error);
    if (sceneError != nullptr) {
        if (sceneError->path().find("texture") != std::string::npos) {
            return 4;
        }
        return 5;
    }
    if (level >= 0) {
        return 6;
    }
    return 1;
}

}  // namespace

int runPipeline(const PipelineOptions& options) {
    if (options.scenePath.empty() || options.outputDirectory.empty()) {
        std::cerr << "[ERROR] [config] scene and output are required\n";
        return 1;
    }
    if (options.tileSize <= 0 || options.levels <= 0 ||
        options.previewWidth <= 0 || options.previewHeight <= 0 ||
        options.samplesPerAxis < 1 || options.samplesPerAxis > 16) {
        std::cerr << "[ERROR] [config] invalid numeric option\n";
        return 1;
    }
    if (options.previewLevel < 0 || options.previewLevel >= options.levels) {
        std::cerr << "[ERROR] [config] preview level is outside the pyramid\n";
        return 1;
    }

    const fs::path output = fs::absolute(options.outputDirectory);
    std::error_code error;
    fs::create_directories(output, error);
    if (error) {
        std::cerr << "[ERROR] [output] cannot create output directory "
                  << output.string() << ": " << error.message() << '\n';
        return 2;
    }
    const fs::path probe = output / ".tilegen-write-probe";
    {
        std::ofstream test(probe, std::ios::binary | std::ios::trunc);
        if (!test) {
            std::cerr << "[ERROR] [output] output directory is not writable: "
                      << output.string() << '\n';
            return 2;
        }
        test << "ok";
        test.close();
        fs::remove(probe, error);
    }

    Logger logger(output / "tilegen.log", options.resume);
    if (!logger.available()) {
        std::cerr << "[ERROR] [output] cannot open log file in output directory\n";
        return 2;
    }
    logger.info("run", "scene=" + options.scenePath + ", output=" +
                           output.string() + ", tile_size=" +
                           std::to_string(options.tileSize) + ", levels=" +
                           std::to_string(options.levels) + ", resume=" +
                           (options.resume ? "true" : "false"));

    try {
        Scene scene = loadScene(options.scenePath);
        logger.info("scene", "validated " + std::to_string(scene.width) + "x" +
                                 std::to_string(scene.height) + ", " +
                                 std::to_string(scene.layers.size()) +
                                 " layers, crc32=" + crc32Hex(scene.sourceCrc32));

        ManifestState state;
        if (options.resume) {
            const fs::path manifestPath = output / "manifest.json";
            if (fs::is_regular_file(manifestPath, error)) {
                state = readManifest(manifestPath, scene, options, logger);
                validateCompletedLevels(state, scene, options, output, logger);
            } else {
                logger.warning("resume", "manifest.json does not exist; starting fresh");
            }
        }

        std::map<TileKey, TileRecord> records;
        if (state.validForResume) {
            records = state.tiles;
        }
        std::set<int> completedLevels = state.completedLevels;
        const int startLevel = state.validForResume ? state.startLevel : 0;
        if (options.resume) {
            logger.info("resume", "starting at level " + std::to_string(startLevel));
        }

        auto emitManifest = [&](const std::string& previewPath, int previewLevel,
                                int previewWidth, int previewHeight,
                                std::uint32_t previewCrc) {
            const std::string manifest =
                makeManifest(scene, options, options, records, completedLevels, output,
                             previewPath, previewLevel, previewWidth, previewHeight,
                             previewCrc);
            saveManifest(output, manifest);
        };

        emitManifest("", options.previewLevel, options.previewWidth,
                     options.previewHeight, 0);

        RenderSettings renderSettings;
        renderSettings.samplesPerAxis = options.samplesPerAxis;

        for (int level = startLevel; level < options.levels; ++level) {
            if (options.simulateDiskFullLevel >= 0 &&
                level >= options.simulateDiskFullLevel) {
                logger.error("level " + std::to_string(level),
                             "simulated ENOSPC before level generation");
                emitManifest("", options.previewLevel, options.previewWidth,
                             options.previewHeight, 0);
                logger.info("manifest",
                            "preserved completed levels and existing tile records");
                return 3;
            }
            const double scale = std::ldexp(1.0, -level);
            const double levelWidth = scene.width * scale;
            const double levelHeight = scene.height * scale;
            const int tilesX = static_cast<int>(
                std::ceil(levelWidth / static_cast<double>(options.tileSize)));
            const int tilesY = static_cast<int>(
                std::ceil(levelHeight / static_cast<double>(options.tileSize)));
            logger.info("level " + std::to_string(level),
                        "generating " + std::to_string(tilesX) + "x" +
                            std::to_string(tilesY) + " tiles at scale " +
                            std::to_string(scale));
            try {
                for (int tileYIndex = 0; tileYIndex < tilesY; ++tileYIndex) {
                    for (int tileXIndex = 0; tileXIndex < tilesX; ++tileXIndex) {
                        const TileKey key{level, tileXIndex, tileYIndex};
                        const double worldPerPixel = 1.0 / scale;
                        const Bounds tileBounds{
                            tileXIndex * options.tileSize * worldPerPixel,
                            tileYIndex * options.tileSize * worldPerPixel,
                            (tileXIndex + 1) * options.tileSize * worldPerPixel,
                            (tileYIndex + 1) * options.tileSize * worldPerPixel, true};
                        if (options.resume) {
                            const TileRecord* reusable =
                                findReusableTile(state, options, output, key, logger);
                            if (reusable != nullptr) {
                                records[key] = *reusable;
                                continue;
                            }
                        }

                        RegionRenderResult rendered = renderSceneRegion(
                            scene, scale, tileBounds, options.tileSize,
                            options.tileSize, renderSettings, false);
                        TileRecord record;
                        record.level = level;
                        record.x = tileXIndex;
                        record.y = tileYIndex;
                        record.width = options.tileSize;
                        record.height = options.tileSize;
                        record.bbox = tileBounds;
                        record.layerIds = rendered.contributingLayerIds;
                        for (std::size_t i = 0;
                             i < rendered.contributingLayerIds.size(); ++i) {
                            record.layerBboxes[rendered.contributingLayerIds[i]] =
                                rendered.contributingLayerBounds[i];
                        }
                        bool bboxCovered = false;
                        for (const Layer& layer : scene.layers) {
                            if (!layer.visible) {
                                continue;
                            }
                            const Bounds bounds = layerWorldBounds(layer);
                            if (bounds.valid && bounds.intersects(tileBounds)) {
                                bboxCovered = true;
                                break;
                            }
                        }
                        record.bboxCovered = bboxCovered;
                        const std::vector<Rgba8> straight =
                            imageToStraightRgba8(rendered.image);
                        bool anyAlpha = false;
                        std::vector<std::uint8_t> raw;
                        raw.reserve(straight.size() * 4U);
                        for (const Rgba8& pixel : straight) {
                            raw.push_back(pixel.r);
                            raw.push_back(pixel.g);
                            raw.push_back(pixel.b);
                            raw.push_back(pixel.a);
                            anyAlpha = anyAlpha || pixel.a != 0;
                        }
                        record.empty = !anyAlpha;
                        record.present = anyAlpha;
                        if (!record.empty) {
                            record.path =
                                tileRawPath(output, level, tileXIndex, tileYIndex);
                            record.pngPath =
                                tilePngPath(output, level, tileXIndex, tileYIndex);
                            record.crc32Value = crc32(raw);
                            std::vector<Rgba8> pixels = straight;
                            record.pngCrc32Value =
                                crc32(encodePngRgba(options.tileSize,
                                                    options.tileSize, pixels));
                            record.byteLength = raw.size();
                            writeTileFiles(output, record, raw);
                        }
                        records[key] = std::move(record);
                    }
                }
                completedLevels.insert(level);
                emitManifest("", options.previewLevel, options.previewWidth,
                             options.previewHeight, 0);
                logger.info("level " + std::to_string(level), "completed");
            } catch (const std::exception& levelError) {
                logger.error("level " + std::to_string(level), levelError.what());
                try {
                    emitManifest("", options.previewLevel, options.previewWidth,
                                 options.previewHeight, 0);
                    logger.info("manifest",
                                "preserved completed levels and partial tile records");
                } catch (const std::exception& manifestError) {
                    logger.error("manifest",
                                 std::string("cannot preserve manifest: ") +
                                     manifestError.what());
                }
                return exitCodeForException(levelError, level);
            }
        }

        try {
            std::uint32_t previewCrc = 0;
            const std::string previewPath =
                generatePreview(scene, options, output, logger, previewCrc);
            emitManifest(previewPath, options.previewLevel, options.previewWidth,
                         options.previewHeight, previewCrc);
            logger.info("complete", "pyramid, manifest, index, and preview generated");
            return 0;
        } catch (const std::exception& previewError) {
            logger.error("preview", previewError.what());
            try {
                emitManifest("", options.previewLevel, options.previewWidth,
                             options.previewHeight, 0);
            } catch (const std::exception&) {
            }
            return exitCodeForException(previewError, -1);
        }
    } catch (const std::exception& errorValue) {
        logger.error("scene", errorValue.what());
        return exitCodeForException(errorValue, -1);
    }
}

}  // namespace tilegen
