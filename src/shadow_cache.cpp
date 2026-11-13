#include "shadow_cache.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#endif

namespace dungeon {
namespace {

constexpr char kMagic[8] = {'D', 'D', 'S', 'H', 'A', 'D', 'O', 'W'};
constexpr uint32_t kFormatVersion = 1;
constexpr uint32_t kMaxEntries = 4096;
constexpr uint32_t kMaxTileSize = 1024;

void writeU32(std::ostream& out, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.put(static_cast<char>((value >> shift) & 0xFFu));
    }
}

void writeU64(std::ostream& out, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        out.put(static_cast<char>((value >> shift) & 0xFFu));
    }
}

bool readU32(std::istream& in, uint32_t& value) {
    value = 0;
    for (int shift = 0; shift < 32; shift += 8) {
        char byte = 0;
        if (!in.get(byte)) {
            return false;
        }
        value |= static_cast<uint32_t>(static_cast<uint8_t>(byte)) << shift;
    }
    return true;
}

bool readU64(std::istream& in, uint64_t& value) {
    value = 0;
    for (int shift = 0; shift < 64; shift += 8) {
        char byte = 0;
        if (!in.get(byte)) {
            return false;
        }
        value |= static_cast<uint64_t>(static_cast<uint8_t>(byte)) << shift;
    }
    return true;
}

void writeDoubleBits(std::ostream& out, double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    writeU64(out, bits);
}

bool readDoubleBits(std::istream& in, double& value) {
    uint64_t bits = 0;
    if (!readU64(in, bits)) {
        return false;
    }
    std::memcpy(&value, &bits, sizeof(value));
    return true;
}

std::filesystem::path temporaryPathFor(const std::filesystem::path& path) {
    return path.string() + ".tmp";
}

bool replaceFileAtomically(const std::filesystem::path& temp, const std::filesystem::path& destination) {
#ifdef _WIN32
    return MoveFileExW(
               temp.wstring().c_str(),
               destination.wstring().c_str(),
               MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    std::error_code ec;
    std::filesystem::rename(temp, destination, ec);
    return !ec;
#endif
}

}  // namespace

bool loadShadowCache(
    const std::string& path,
    int expectedColumns,
    int expectedRows,
    int expectedTileSize,
    ShadowCache& cache) {
    cache = {};
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    char magic[8] = {};
    if (!in.read(magic, sizeof(magic)) || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        return false;
    }

    uint32_t version = 0;
    uint32_t columns = 0;
    uint32_t rows = 0;
    uint32_t tileSize = 0;
    uint32_t entryCount = 0;
    if (!readU32(in, version) || !readU32(in, columns) || !readU32(in, rows) ||
        !readU32(in, tileSize) || !readU32(in, entryCount)) {
        return false;
    }
    if (version != kFormatVersion ||
        static_cast<int>(columns) != expectedColumns ||
        static_cast<int>(rows) != expectedRows ||
        static_cast<int>(tileSize) != expectedTileSize ||
        entryCount > kMaxEntries ||
        tileSize > kMaxTileSize) {
        return false;
    }

    ShadowCache loaded;
    loaded.columns = expectedColumns;
    loaded.rows = expectedRows;
    loaded.tileSize = expectedTileSize;
    const uint64_t texelCount = static_cast<uint64_t>(tileSize) * tileSize;

    for (uint32_t entryIndex = 0; entryIndex < entryCount; ++entryIndex) {
        ShadowCacheEntry entry;
        uint32_t lightId = 0;
        uint32_t slot = 0;
        uint32_t geometryVersion = 0;
        uint32_t reserved = 0;
        uint32_t depthCount = 0;
        if (!readU32(in, lightId) || !readU32(in, slot) || !readU32(in, geometryVersion) ||
            !readU32(in, reserved) || !readU64(in, entry.lightHash) ||
            !readU64(in, entry.sceneHash) || !readU32(in, depthCount)) {
            return false;
        }
        if (reserved != 0 || static_cast<uint64_t>(depthCount) != texelCount ||
            static_cast<int32_t>(slot) < 0 ||
            static_cast<int>(slot) >= expectedColumns * expectedRows) {
            return false;
        }
        entry.lightId = lightId;
        entry.slot = static_cast<int32_t>(slot);
        entry.geometryVersion = static_cast<int32_t>(geometryVersion);
        entry.depth.resize(depthCount);
        for (double& depth : entry.depth) {
            if (!readDoubleBits(in, depth)) {
                return false;
            }
        }
        loaded.entries.push_back(std::move(entry));
    }

    cache = std::move(loaded);
    return true;
}

bool saveShadowCache(const std::string& path, const ShadowCache& cache) {
    try {
        const std::filesystem::path destination(path);
        if (destination.has_parent_path()) {
            std::filesystem::create_directories(destination.parent_path());
        }
        const std::filesystem::path temp = temporaryPathFor(destination);

        {
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            if (!out) {
                return false;
            }
            out.write(kMagic, sizeof(kMagic));
            writeU32(out, kFormatVersion);
            writeU32(out, static_cast<uint32_t>(cache.columns));
            writeU32(out, static_cast<uint32_t>(cache.rows));
            writeU32(out, static_cast<uint32_t>(cache.tileSize));
            writeU32(out, static_cast<uint32_t>(cache.entries.size()));
            for (const ShadowCacheEntry& entry : cache.entries) {
                writeU32(out, entry.lightId);
                writeU32(out, static_cast<uint32_t>(entry.slot));
                writeU32(out, static_cast<uint32_t>(entry.geometryVersion));
                writeU32(out, 0);
                writeU64(out, entry.lightHash);
                writeU64(out, entry.sceneHash);
                writeU32(out, static_cast<uint32_t>(entry.depth.size()));
                for (double depth : entry.depth) {
                    writeDoubleBits(out, depth);
                }
            }
            out.flush();
            if (!out) {
                out.close();
                std::filesystem::remove(temp);
                return false;
            }
        }

        if (!replaceFileAtomically(temp, destination)) {
            std::filesystem::remove(temp);
            return false;
        }
        return true;
    } catch (const std::filesystem::filesystem_error&) {
        return false;
    }
}

}  // namespace dungeon
