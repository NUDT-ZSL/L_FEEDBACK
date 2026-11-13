#include "snapshot.hpp"

#include "mesh_io.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;

namespace rigskin {
namespace {

std::atomic<std::uint64_t> g_temporaryFileCounter{0};

std::vector<std::string> splitWhitespace(const std::string& line) {
    std::istringstream stream(line);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) {
        tokens.push_back(std::move(token));
    }
    return tokens;
}

bool parseUnsigned(const std::string& token, std::uint64_t& value) {
    if (token.empty() || token[0] == '-') {
        return false;
    }
    std::size_t consumed = 0;
    try {
        value = std::stoull(token, &consumed, 10);
    } catch (const std::exception&) {
        return false;
    }
    return consumed == token.size();
}

bool parseSigned(const std::string& token, int& value) {
    std::size_t consumed = 0;
    try {
        value = std::stoi(token, &consumed, 10);
    } catch (const std::exception&) {
        return false;
    }
    return consumed == token.size();
}

bool ensureParentDirectory(
    const fs::path& path,
    std::string& error) {
    const fs::path parent = path.parent_path();
    if (parent.empty()) {
        return true;
    }
    std::error_code code;
    fs::create_directories(parent, code);
    if (code) {
        error = "cannot create directory " + parent.string() + ": " + code.message();
        return false;
    }
    return true;
}

bool replaceFile(
    const fs::path& temporary,
    const fs::path& destination,
    std::string& error) {
    std::error_code code;
    fs::rename(temporary, destination, code);
    if (!code) {
        return true;
    }

    std::error_code removeCode;
    fs::remove(destination, removeCode);
    code.clear();
    fs::rename(temporary, destination, code);
    if (code) {
        error = "cannot replace " + destination.string() + ": " + code.message();
        return false;
    }
    return true;
}

std::string makeTempPath(const fs::path& destination) {
    const std::uint64_t counter =
        g_temporaryFileCounter.fetch_add(1, std::memory_order_relaxed);
    return destination.string() + ".tmp." + std::to_string(counter);
}

bool isSha256Hex(const std::string& value);

bool parseEntry(const std::vector<std::string>& tokens, FrameEntry& entry, std::string& error) {
    if (tokens.size() < 2 || tokens[0] != "entry") {
        error = "expected entry line";
        return false;
    }

    std::uint64_t number = 0;
    if (!parseUnsigned(tokens[1], number) || number > UINT32_MAX) {
        error = "invalid manifest frame number";
        return false;
    }
    entry.frame = static_cast<std::uint32_t>(number);

    bool sawVersion = false;
    bool sawInputHash = false;
    bool sawOutputChecksum = false;
    bool sawMeshSha = false;
    bool sawMeshBytes = false;
    bool sawPpmSha = false;
    bool sawPpmBytes = false;
    bool sawVertexCount = false;
    bool sawTriangleCount = false;
    bool sawDegenerateTriangles = false;
    bool sawZeroWeightVertices = false;

    for (std::size_t index = 2; index < tokens.size(); ++index) {
        const std::string& token = tokens[index];
        const std::size_t equals = token.find('=');
        if (equals == std::string::npos || equals == 0) {
            error = "entry field is not key=value";
            return false;
        }
        const std::string key = token.substr(0, equals);
        const std::string value = token.substr(equals + 1);

        if (key == "manifest_version") {
            if (sawVersion) {
                error = "duplicate entry manifest_version";
                return false;
            }
            int version = 0;
            if (!parseSigned(value, version)) {
                error = "invalid entry manifest_version";
                return false;
            }
            entry.manifestVersion = version;
            sawVersion = true;
        } else if (key == "input_hash") {
            if (sawInputHash) {
                error = "duplicate entry input_hash";
                return false;
            }
            entry.inputHash = value;
            sawInputHash = true;
        } else if (key == "output_checksum") {
            if (sawOutputChecksum) {
                error = "duplicate entry output_checksum";
                return false;
            }
            entry.outputChecksum = value;
            sawOutputChecksum = true;
        } else if (key == "mesh_sha256") {
            if (sawMeshSha) {
                error = "duplicate entry mesh_sha256";
                return false;
            }
            entry.meshSha256 = value;
            sawMeshSha = true;
        } else if (key == "ppm_sha256") {
            if (sawPpmSha) {
                error = "duplicate entry ppm_sha256";
                return false;
            }
            entry.ppmSha256 = value;
            sawPpmSha = true;
        } else if (key == "mesh_bytes") {
            if (sawMeshBytes) {
                error = "duplicate entry mesh_bytes";
                return false;
            }
            if (!parseUnsigned(value, entry.meshBytes)) {
                error = "invalid entry mesh_bytes";
                return false;
            }
            sawMeshBytes = true;
        } else if (key == "ppm_bytes") {
            if (sawPpmBytes) {
                error = "duplicate entry ppm_bytes";
                return false;
            }
            if (!parseUnsigned(value, entry.ppmBytes)) {
                error = "invalid entry ppm_bytes";
                return false;
            }
            sawPpmBytes = true;
        } else if (key == "vertex_count") {
            if (sawVertexCount) {
                error = "duplicate entry vertex_count";
                return false;
            }
            std::uint64_t parsed = 0;
            if (!parseUnsigned(value, parsed) || parsed > UINT32_MAX) {
                error = "invalid entry vertex_count";
                return false;
            }
            entry.vertexCount = static_cast<std::uint32_t>(parsed);
            sawVertexCount = true;
        } else if (key == "triangle_count") {
            if (sawTriangleCount) {
                error = "duplicate entry triangle_count";
                return false;
            }
            std::uint64_t parsed = 0;
            if (!parseUnsigned(value, parsed) || parsed > UINT32_MAX) {
                error = "invalid entry triangle_count";
                return false;
            }
            entry.triangleCount = static_cast<std::uint32_t>(parsed);
            sawTriangleCount = true;
        } else if (key == "degenerate_triangles") {
            if (sawDegenerateTriangles) {
                error = "duplicate entry degenerate_triangles";
                return false;
            }
            std::uint64_t parsed = 0;
            if (!parseUnsigned(value, parsed) || parsed > UINT32_MAX) {
                error = "invalid entry degenerate_triangles";
                return false;
            }
            entry.degenerateTriangles = static_cast<std::uint32_t>(parsed);
            sawDegenerateTriangles = true;
        } else if (key == "zero_weight_vertices") {
            if (sawZeroWeightVertices) {
                error = "duplicate entry zero_weight_vertices";
                return false;
            }
            std::uint64_t parsed = 0;
            if (!parseUnsigned(value, parsed) || parsed > UINT32_MAX) {
                error = "invalid entry zero_weight_vertices";
                return false;
            }
            entry.zeroWeightVertices = static_cast<std::uint32_t>(parsed);
            sawZeroWeightVertices = true;
        } else {
            error = "unknown entry field " + key;
            return false;
        }
    }

    if (!sawVersion || !sawInputHash || !sawOutputChecksum || !sawMeshSha ||
        !sawMeshBytes || !sawPpmSha || !sawPpmBytes || !sawVertexCount ||
        !sawTriangleCount || !sawDegenerateTriangles ||
        !sawZeroWeightVertices) {
        error = "entry is missing a required field";
        return false;
    }
    if (entry.manifestVersion != kManifestVersion) {
        error = "entry manifest version does not match current version";
        return false;
    }
    if (!isSha256Hex(entry.inputHash) ||
        !isSha256Hex(entry.outputChecksum) ||
        !isSha256Hex(entry.meshSha256) ||
        !isSha256Hex(entry.ppmSha256)) {
        error = "entry checksum is not a SHA-256 string";
        return false;
    }
    return true;
}

bool isSha256Hex(const std::string& value) {
    if (value.size() != 64) {
        return false;
    }
    return std::all_of(
        value.begin(), value.end(), [](char character) {
            return (character >= '0' && character <= '9') ||
                   (character >= 'a' && character <= 'f');
        });
}

}  // namespace

ManifestReadResult readManifest(const fs::path& path) {
    ManifestReadResult result;
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        result.status = fs::exists(path)
            ? ManifestReadStatus::Corrupt
            : ManifestReadStatus::Missing;
        if (result.status == ManifestReadStatus::Corrupt) {
            result.error = "manifest exists but cannot be opened";
        }
        return result;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    if (!input.eof()) {
        result.status = ManifestReadStatus::Corrupt;
        result.error = "I/O failure while reading manifest";
        return result;
    }
    if (lines.empty()) {
        result.status = ManifestReadStatus::Corrupt;
        result.error = "manifest is empty";
        return result;
    }

    for (const std::string& candidate : lines) {
        const auto tokens = splitWhitespace(candidate);
        const bool isVersionField =
            tokens.size() == 2 && tokens[0] == "manifest_version";
        const bool isVersionHeader =
            tokens.size() == 2 && tokens[0] == "FMSNAP_MANIFEST";
        if (isVersionField || isVersionHeader) {
            int version = 0;
            if (!parseSigned(tokens[1], version)) {
                result.status = ManifestReadStatus::Corrupt;
                result.error = "manifest version is not an integer";
                return result;
            }
            result.detectedVersion = version;
            if (version != kManifestVersion) {
                result.status = ManifestReadStatus::UnsupportedVersion;
                result.error =
                    "unsupported manifest version " + std::to_string(version);
                return result;
            }
        }
    }
    if (result.detectedVersion == 0) {
        result.status = ManifestReadStatus::Corrupt;
        result.error = "manifest_version field is missing";
        return result;
    }

    const auto headerTokens = splitWhitespace(lines.front());
    if (headerTokens.size() != 2 || headerTokens[0] != "FMSNAP_MANIFEST" ||
        headerTokens[1] != std::to_string(kManifestVersion)) {
        result.status = ManifestReadStatus::Corrupt;
        result.error = "manifest header is invalid";
        return result;
    }

    Manifest manifest;
    manifest.version = kManifestVersion;
    bool sawVersion = false;
    bool sawHash = false;
    bool sawFrameCount = false;
    bool sawComplete = false;
    bool sawZeroWeights = false;
    bool sawFooter = false;
    std::map<std::uint32_t, FrameEntry> entries;

    for (std::size_t lineIndex = 1; lineIndex < lines.size(); ++lineIndex) {
        const auto tokens = splitWhitespace(lines[lineIndex]);
        if (tokens.empty()) {
            continue;
        }
        if (tokens[0] == "FMSNAP_END") {
            if (tokens.size() != 1 || sawFooter) {
                result.status = ManifestReadStatus::Corrupt;
                result.error = "invalid manifest footer";
                return result;
            }
            sawFooter = true;
            continue;
        }
        if (sawFooter) {
            result.status = ManifestReadStatus::Corrupt;
            result.error = "data follows the manifest footer";
            return result;
        }
        if (tokens[0] == "entry") {
            FrameEntry entry;
            std::string fieldError;
            if (!parseEntry(tokens, entry, fieldError)) {
                result.status = ManifestReadStatus::Corrupt;
                result.error = fieldError;
                return result;
            }
            if (entries.count(entry.frame) != 0) {
                result.status = ManifestReadStatus::Corrupt;
                result.error = "duplicate frame entry " + std::to_string(entry.frame);
                return result;
            }
            entries.emplace(entry.frame, std::move(entry));
            continue;
        }
        if (tokens.size() != 2) {
            result.status = ManifestReadStatus::Corrupt;
            result.error = "malformed manifest line";
            return result;
        }

        std::uint64_t number = 0;
        if (tokens[0] == "manifest_version") {
            int version = 0;
            if (sawVersion || !parseSigned(tokens[1], version) ||
                version != kManifestVersion) {
                result.status = ManifestReadStatus::Corrupt;
                result.error = "invalid manifest_version line";
                return result;
            }
            manifest.version = version;
            sawVersion = true;
        } else if (tokens[0] == "input_hash") {
            if (sawHash || !isSha256Hex(tokens[1])) {
                result.status = ManifestReadStatus::Corrupt;
                result.error = "invalid input_hash line";
                return result;
            }
            manifest.inputHash = tokens[1];
            sawHash = true;
        } else if (tokens[0] == "frame_count") {
            if (sawFrameCount || !parseUnsigned(tokens[1], number) ||
                number > UINT32_MAX || number == 0) {
                result.status = ManifestReadStatus::Corrupt;
                result.error = "invalid frame_count line";
                return result;
            }
            manifest.frameCount = static_cast<std::uint32_t>(number);
            sawFrameCount = true;
        } else if (tokens[0] == "complete") {
            if (sawComplete || !parseUnsigned(tokens[1], number) || number > 1) {
                result.status = ManifestReadStatus::Corrupt;
                result.error = "invalid complete line";
                return result;
            }
            manifest.complete = number != 0;
            sawComplete = true;
        } else if (tokens[0] == "zero_weight_vertices") {
            if (sawZeroWeights || !parseUnsigned(tokens[1], number) ||
                number > UINT32_MAX) {
                result.status = ManifestReadStatus::Corrupt;
                result.error = "invalid zero_weight_vertices line";
                return result;
            }
            manifest.zeroWeightVertices = static_cast<std::uint32_t>(number);
            sawZeroWeights = true;
        } else {
            result.status = ManifestReadStatus::Corrupt;
            result.error = "unknown manifest field " + tokens[0];
            return result;
        }
    }

    if (!sawVersion || !sawHash || !sawFrameCount || !sawComplete ||
        !sawZeroWeights || !sawFooter) {
        result.status = ManifestReadStatus::Corrupt;
        result.error = "manifest is missing a required field or footer";
        return result;
    }
    for (const auto& item : entries) {
        if (item.first >= manifest.frameCount) {
            result.status = ManifestReadStatus::Corrupt;
            result.error = "frame entry is outside frame_count";
            return result;
        }
        manifest.entries.push_back(item.second);
    }

    result.status = ManifestReadStatus::Parsed;
    result.manifest = std::move(manifest);
    return result;
}

std::string serializeManifest(const Manifest& inputManifest) {
    Manifest manifest = inputManifest;
    std::sort(
        manifest.entries.begin(),
        manifest.entries.end(),
        [](const FrameEntry& lhs, const FrameEntry& rhs) {
            return lhs.frame < rhs.frame;
        });

    std::ostringstream output;
    output << "FMSNAP_MANIFEST " << kManifestVersion << '\n';
    output << "manifest_version " << kManifestVersion << '\n';
    output << "input_hash " << manifest.inputHash << '\n';
    output << "frame_count " << manifest.frameCount << '\n';
    output << "complete " << (manifest.complete ? 1 : 0) << '\n';
    output << "zero_weight_vertices " << manifest.zeroWeightVertices << '\n';
    for (const FrameEntry& entry : manifest.entries) {
        output << "entry " << entry.frame;
        output << " manifest_version=" << entry.manifestVersion;
        output << " input_hash=" << entry.inputHash;
        output << " output_checksum=" << entry.outputChecksum;
        output << " mesh_sha256=" << entry.meshSha256;
        output << " mesh_bytes=" << entry.meshBytes;
        output << " ppm_sha256=" << entry.ppmSha256;
        output << " ppm_bytes=" << entry.ppmBytes;
        output << " vertex_count=" << entry.vertexCount;
        output << " triangle_count=" << entry.triangleCount;
        output << " degenerate_triangles=" << entry.degenerateTriangles;
        output << " zero_weight_vertices=" << entry.zeroWeightVertices;
        output << '\n';
    }
    output << "FMSNAP_END\n";
    return output.str();
}

bool writeManifest(
    const fs::path& path,
    const Manifest& manifest,
    std::string& error) {
    return writeFileAtomic(path, serializeManifest(manifest), error);
}

fs::path frameMeshPath(const fs::path& snapshotDirectory, std::uint32_t frame) {
    std::ostringstream name;
    name << "frame_" << std::setw(4) << std::setfill('0') << frame << ".mesh";
    return snapshotDirectory / "frames" / name.str();
}

fs::path framePpmPath(const fs::path& snapshotDirectory, std::uint32_t frame) {
    std::ostringstream name;
    name << "frame_" << std::setw(4) << std::setfill('0') << frame << ".ppm";
    return snapshotDirectory / "frames" / name.str();
}

bool readFileBinary(
    const fs::path& path,
    std::vector<std::uint8_t>& bytes,
    std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open " + path.string();
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff length = input.tellg();
    if (length < 0) {
        error = "cannot determine size of " + path.string();
        return false;
    }
    input.seekg(0, std::ios::beg);
    bytes.resize(static_cast<std::size_t>(length));
    if (!bytes.empty()) {
        input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    }
    if (!input || input.gcount() != static_cast<std::streamsize>(bytes.size())) {
        error = "short read from " + path.string();
        return false;
    }
    return true;
}

bool writeFileAtomic(
    const fs::path& path,
    const std::vector<std::uint8_t>& bytes,
    std::string& error) {
    if (!ensureParentDirectory(path, error)) {
        return false;
    }
    const std::string temporary = makeTempPath(path);
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            error = "cannot create temporary file " + temporary;
            return false;
        }
        if (!bytes.empty()) {
            output.write(
                reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
        }
        output.flush();
        if (!output) {
            output.close();
            std::error_code removeCode;
            fs::remove(temporary, removeCode);
            error = "cannot write temporary file " + temporary;
            return false;
        }
    }
    return replaceFile(temporary, path, error);
}

bool writeFileAtomic(
    const fs::path& path,
    const std::string& text,
    std::string& error) {
    std::vector<std::uint8_t> bytes(text.begin(), text.end());
    return writeFileAtomic(path, bytes, error);
}

std::string combinedChecksum(
    const std::vector<std::uint8_t>& meshBytes,
    const std::vector<std::uint8_t>& ppmBytes) {
    Sha256 context;
    context.update(meshBytes);
    context.update(ppmBytes);
    return Sha256::hex(context.finish());
}

ArtifactValidation validateFrameArtifacts(
    const fs::path& snapshotDirectory,
    const FrameEntry& entry,
    const std::string& expectedInputHash,
    int width,
    int height) {
    ArtifactValidation result;
    if (entry.manifestVersion != kManifestVersion) {
        result.errorCode = "E3001";
        result.detail = "entry manifest version mismatch";
        return result;
    }
    if (entry.inputHash != expectedInputHash) {
        result.errorCode = "E3001";
        result.detail = "entry input hash mismatch";
        return result;
    }

    const fs::path meshPath = frameMeshPath(snapshotDirectory, entry.frame);
    const fs::path ppmPath = framePpmPath(snapshotDirectory, entry.frame);
    std::error_code code;
    if (!fs::exists(meshPath, code) || !fs::exists(ppmPath, code)) {
        result.errorCode = "E2001";
        result.detail = "frame payload is missing";
        return result;
    }
    const std::uint64_t meshFileSize =
        static_cast<std::uint64_t>(fs::file_size(meshPath, code));
    if (code) {
        result.errorCode = "E2001";
        result.detail = "cannot read mesh file size";
        return result;
    }
    const std::uint64_t ppmFileSize =
        static_cast<std::uint64_t>(fs::file_size(ppmPath, code));
    if (code) {
        result.errorCode = "E2001";
        result.detail = "cannot read PPM file size";
        return result;
    }
    if (meshFileSize != entry.meshBytes || ppmFileSize != entry.ppmBytes) {
        result.errorCode = "E2002";
        result.detail = "frame payload size does not match manifest";
        return result;
    }

    std::vector<std::uint8_t> meshBytes;
    std::vector<std::uint8_t> ppmBytes;
    std::string readError;
    if (!readFileBinary(meshPath, meshBytes, readError) ||
        !readFileBinary(ppmPath, ppmBytes, readError)) {
        result.errorCode = "E2001";
        result.detail = readError;
        return result;
    }
    if (meshBytes.size() != entry.meshBytes || ppmBytes.size() != entry.ppmBytes) {
        result.errorCode = "E2002";
        result.detail = "frame payload changed while it was being read";
        return result;
    }

    const std::string meshSha = Sha256::digestHex(meshBytes);
    const std::string ppmSha = Sha256::digestHex(ppmBytes);
    const std::string combined = combinedChecksum(meshBytes, ppmBytes);
    if (meshSha != entry.meshSha256 || ppmSha != entry.ppmSha256 ||
        combined != entry.outputChecksum) {
        result.errorCode = "E2003";
        result.detail = "frame payload checksum mismatch";
        return result;
    }

    MeshHeader meshHeader;
    std::string headerError;
    if (!decodeMeshHeader(meshBytes, meshHeader, headerError)) {
        result.errorCode = "E2005";
        result.detail = headerError;
        return result;
    }
    if (meshHeader.frame != entry.frame ||
        meshHeader.vertexCount != entry.vertexCount ||
        meshHeader.triangleCount != entry.triangleCount ||
        meshHeader.degenerateTriangleCount != entry.degenerateTriangles ||
        meshHeader.zeroWeightVertexCount != entry.zeroWeightVertices) {
        result.errorCode = "E3001";
        result.detail = "mesh header statistics do not match manifest";
        return result;
    }

    std::string ppmError;
    if (!validatePpmHeader(ppmBytes, width, height, ppmError)) {
        result.errorCode = "E2004";
        result.detail = ppmError;
        return result;
    }

    result.valid = true;
    return result;
}

bool archiveSnapshotForRebuild(
    const fs::path& snapshotDirectory,
    const std::string& reason,
    const std::string& reportText,
    fs::path& archivePath,
    std::string& error) {
    std::error_code code;
    fs::create_directories(snapshotDirectory / "errors", code);
    if (code) {
        error = "cannot create errors directory: " + code.message();
        return false;
    }

    if (!writeFileAtomic(
            snapshotDirectory / "errors" / (reason + ".txt"),
            reportText,
            error)) {
        return false;
    }

    std::vector<fs::path> entriesToMove;
    for (const fs::directory_entry& item : fs::directory_iterator(snapshotDirectory, code)) {
        if (code) {
            error = "cannot inspect snapshot directory: " + code.message();
            return false;
        }
        if (item.path().filename() == "errors" ||
            item.path().filename() == "quarantine" ||
            item.path().filename() == ".tmp") {
            continue;
        }
        entriesToMove.push_back(item.path());
    }
    if (entriesToMove.empty()) {
        archivePath.clear();
        return true;
    }

    const fs::path quarantineRoot = snapshotDirectory / "quarantine";
    fs::create_directories(quarantineRoot, code);
    if (code) {
        error = "cannot create quarantine directory: " + code.message();
        return false;
    }
    archivePath = quarantineRoot / reason;
    if (fs::exists(archivePath)) {
        std::uint64_t counter = 1;
        do {
            archivePath = quarantineRoot / (reason + "-" + std::to_string(counter));
            ++counter;
        } while (fs::exists(archivePath));
    }
    fs::create_directories(archivePath, code);
    if (code) {
        error = "cannot create archive directory: " + code.message();
        return false;
    }

    for (const fs::path& item : entriesToMove) {
        code.clear();
        fs::rename(item, archivePath / item.filename(), code);
        if (code) {
            error =
                "cannot archive " + item.string() + ": " + code.message();
            return false;
        }
    }
    return true;
}

}  // namespace rigskin
