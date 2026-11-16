#include "asset_reader.hpp"

#include "binary_io.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace skel {
namespace {

constexpr float kMinimumQuaternionLength = 1.0e-8F;

Error truncated(std::size_t offset, const char* what) {
    std::ostringstream stream;
    stream << "unexpected end of file while reading " << what << " at byte "
           << offset;
    return makeError(ErrorCode::Truncated, stream.str());
}

bool readMatrix(ByteReader& reader, glm::mat4& matrix, Error& error) {
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            if (!reader.readF32(matrix[column][row])) {
                error = truncated(reader.offset(), "inverse bind matrix");
                return false;
            }
        }
    }
    if (!isFinite(matrix)) {
        error = makeError(ErrorCode::InvalidNumericValue,
                          "inverse bind matrix contains a non-finite value");
        return false;
    }
    return true;
}

bool readVec3(ByteReader& reader, glm::vec3& value, const char* field,
              Error& error) {
    if (!reader.readF32(value.x) || !reader.readF32(value.y) ||
        !reader.readF32(value.z)) {
        error = truncated(reader.offset(), field);
        return false;
    }
    if (!isFinite(value)) {
        error = makeError(ErrorCode::InvalidNumericValue,
                          std::string(field) + " contains a non-finite value");
        return false;
    }
    return true;
}

bool validateParents(const Asset& asset, Error& error) {
    const std::size_t count = asset.nodes.size();
    std::vector<std::uint8_t> state(count, 0);

    for (std::size_t start = 0; start < count; ++start) {
        if (state[start] == 2) {
            continue;
        }
        std::vector<std::uint32_t> chain;
        std::int32_t current = static_cast<std::int32_t>(start);
        while (current >= 0 && state[static_cast<std::size_t>(current)] == 0) {
            const auto index = static_cast<std::size_t>(current);
            state[index] = 1;
            chain.push_back(static_cast<std::uint32_t>(current));
            current = asset.nodes[index].parent;
        }
        if (current >= 0 &&
            state[static_cast<std::size_t>(current)] == 1) {
            error = makeError(
                ErrorCode::ParentCycle,
                "parent cycle detected at node " + std::to_string(current));
            return false;
        }
        for (const std::uint32_t index : chain) {
            state[index] = 2;
        }
    }
    return true;
}

bool validateSamples(const Asset& asset, Error& error) {
    std::vector<bool> claimed(asset.samples.size(), false);

    for (std::size_t nodeIndex = 0; nodeIndex < asset.nodes.size();
         ++nodeIndex) {
        const Node& node = asset.nodes[nodeIndex];
        const std::uint64_t end =
            static_cast<std::uint64_t>(node.firstSample) + node.sampleCount;
        if (end > asset.samples.size()) {
            error = makeError(
                ErrorCode::InvalidTrackNode,
                "sample range for node " + std::to_string(nodeIndex) +
                    " exceeds the sample table");
            return false;
        }

        float previousTime = -std::numeric_limits<float>::infinity();
        for (std::uint32_t offset = 0; offset < node.sampleCount; ++offset) {
            const std::size_t sampleIndex =
                static_cast<std::size_t>(node.firstSample) + offset;
            if (claimed[sampleIndex]) {
                error = makeError(
                    ErrorCode::InvalidTrackNode,
                    "sample " + std::to_string(sampleIndex) +
                        " is claimed by more than one node");
                return false;
            }
            claimed[sampleIndex] = true;

            const AnimationSample& sample = asset.samples[sampleIndex];
            if (!std::isfinite(sample.time)) {
                error = makeError(
                    ErrorCode::InvalidNumericValue,
                    "sample time is not finite at sample " +
                        std::to_string(sampleIndex));
                return false;
            }
            if (!(sample.time > previousTime)) {
                error = makeError(
                    ErrorCode::NonMonotonicTime,
                    "sample times must be strictly increasing for node " +
                        std::to_string(nodeIndex) + " (sample " +
                        std::to_string(sampleIndex) + ")");
                return false;
            }
            previousTime = sample.time;

            if (!isFinite(sample.translation) || !isFinite(sample.scale) ||
                !isFinite(sample.rotation)) {
                error = makeError(
                    ErrorCode::InvalidNumericValue,
                    "sample contains a non-finite TRS component at sample " +
                        std::to_string(sampleIndex));
                return false;
            }
            const float quaternionLength = glm::length(sample.rotation);
            if (!std::isfinite(quaternionLength) ||
                quaternionLength < kMinimumQuaternionLength) {
                error = makeError(
                    ErrorCode::ZeroQuaternion,
                    "zero-length quaternion at sample " +
                        std::to_string(sampleIndex));
                return false;
            }
        }
    }

    const auto unclaimed = std::find(claimed.begin(), claimed.end(), false);
    if (unclaimed != claimed.end()) {
        const std::size_t index =
            static_cast<std::size_t>(std::distance(claimed.begin(), unclaimed));
        error = makeError(ErrorCode::InvalidTrackNode,
                          "sample " + std::to_string(index) +
                              " is not assigned to a node track");
        return false;
    }
    return true;
}

bool validateMeshes(const Asset& asset, Error& error) {
    std::size_t totalVertices = 0;
    std::size_t totalIndices = 0;
    for (std::size_t meshIndex = 0; meshIndex < asset.meshes.size();
         ++meshIndex) {
        const Mesh& mesh = asset.meshes[meshIndex];
        totalVertices += mesh.vertices.size();
        totalIndices += mesh.indices.size();
        if (mesh.indices.size() % 3U != 0U) {
            error = makeError(
                ErrorCode::InvalidCount,
                "mesh " + std::to_string(meshIndex) +
                    " index count is not divisible by three");
            return false;
        }

        for (std::size_t vertexIndex = 0; vertexIndex < mesh.vertices.size();
             ++vertexIndex) {
            const Vertex& vertex = mesh.vertices[vertexIndex];
            if (!isFinite(vertex.position) || !isFinite(vertex.normal) ||
                !isFinite(vertex.color)) {
                error = makeError(
                    ErrorCode::InvalidNumericValue,
                    "mesh " + std::to_string(meshIndex) + " vertex " +
                        std::to_string(vertexIndex) +
                        " contains a non-finite attribute");
                return false;
            }
            if (vertex.weightCount == 0U || vertex.weightCount > 4U) {
                error = makeError(
                    ErrorCode::InvalidWeightCount,
                    "mesh " + std::to_string(meshIndex) + " vertex " +
                        std::to_string(vertexIndex) +
                        " must have between one and four weights");
                return false;
            }

            float weightSum = 0.0F;
            for (std::uint32_t slot = 0; slot < vertex.weightCount; ++slot) {
                const std::uint32_t bone = vertex.boneIndices[slot];
                const float weight = vertex.weights[slot];
                if (bone >= asset.nodes.size()) {
                    error = makeError(
                        ErrorCode::InvalidBoneIndex,
                        "mesh " + std::to_string(meshIndex) + " vertex " +
                            std::to_string(vertexIndex) +
                            " references an out-of-range bone");
                    return false;
                }
                if (!std::isfinite(weight) || weight < 0.0F) {
                    error = makeError(
                        ErrorCode::InvalidWeight,
                        "mesh " + std::to_string(meshIndex) + " vertex " +
                            std::to_string(vertexIndex) +
                            " has an invalid weight");
                    return false;
                }
                weightSum += weight;
            }
            for (std::uint32_t slot = vertex.weightCount; slot < 4U; ++slot) {
                if (!std::isfinite(vertex.weights[slot]) ||
                    vertex.weights[slot] != 0.0F) {
                    error = makeError(
                        ErrorCode::InvalidWeight,
                        "mesh " + std::to_string(meshIndex) + " vertex " +
                            std::to_string(vertexIndex) +
                            " has a non-zero weight outside weightCount");
                    return false;
                }
            }
            if (!std::isfinite(weightSum) || weightSum <= 0.0F) {
                error = makeError(
                    ErrorCode::WeightSumZero,
                    "mesh " + std::to_string(meshIndex) + " vertex " +
                        std::to_string(vertexIndex) +
                        " has a non-positive weight sum");
                return false;
            }
        }

        for (std::size_t index = 0; index < mesh.indices.size(); ++index) {
            if (mesh.indices[index] >= mesh.vertices.size()) {
                error = makeError(
                    ErrorCode::InvalidTriangleIndex,
                    "mesh " + std::to_string(meshIndex) + " index " +
                        std::to_string(index) + " is out of range");
                return false;
            }
        }
    }

    if (totalVertices > std::numeric_limits<std::uint32_t>::max() ||
        totalIndices >
            static_cast<std::size_t>(
                std::numeric_limits<std::int32_t>::max())) {
        error = makeError(ErrorCode::ResourceTooLarge,
                          "total mesh data exceeds index/buffer limits");
        return false;
    }
    return true;
}

} // namespace

bool AssetReader::load(const std::filesystem::path& path, Asset& asset,
                       Error& error) {
    asset = Asset{};
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        error = makeError(ErrorCode::OpenFailed,
                          "cannot open asset: " + path.string());
        return false;
    }
    const std::streamoff size = stream.tellg();
    if (size < 0) {
        error = makeError(ErrorCode::ReadFailed,
                          "cannot determine asset size: " + path.string());
        return false;
    }
    if (static_cast<std::uint64_t>(size) > kMaxAssetBytes) {
        error = makeError(ErrorCode::ResourceTooLarge,
                          "asset exceeds the 512 MiB safety limit");
        return false;
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    stream.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        stream.read(reinterpret_cast<char*>(bytes.data()), size);
    }
    if (!stream) {
        error = makeError(ErrorCode::ReadFailed,
                          "failed while reading asset: " + path.string());
        return false;
    }
    return parse(bytes, asset, error);
}

bool AssetReader::parse(const std::vector<std::uint8_t>& bytes, Asset& asset,
                        Error& error) {
    asset = Asset{};
    if (bytes.size() < 32U) {
        error = truncated(0, "asset header");
        return false;
    }
    if (bytes.size() > kMaxAssetBytes) {
        error = makeError(ErrorCode::ResourceTooLarge,
                          "asset exceeds the 512 MiB safety limit");
        return false;
    }

    ByteReader reader(bytes.data(), bytes.size());
    char magic[4]{};
    if (!reader.readBytes(magic, sizeof(magic))) {
        error = truncated(0, "magic");
        return false;
    }
    if (magic[0] != 'S' || magic[1] != 'K' || magic[2] != 'A' ||
        magic[3] != '1') {
        error = makeError(ErrorCode::InvalidMagic,
                          "expected SKA1 asset magic");
        return false;
    }

    Asset parsed;
    std::uint32_t order = 0;
    std::uint32_t nodeCount = 0;
    std::uint32_t meshCount = 0;
    std::uint32_t sampleCount = 0;
    if (!reader.readU32(parsed.version) || !reader.readU32(order) ||
        !reader.readF32(parsed.framesPerSecond) ||
        !reader.readU32(parsed.frameCount) ||
        !reader.readU32(nodeCount) || !reader.readU32(meshCount) ||
        !reader.readU32(sampleCount)) {
        error = truncated(reader.offset(), "asset header fields");
        return false;
    }

    if (parsed.version != kAssetVersion) {
        error = makeError(ErrorCode::UnsupportedVersion,
                          "unsupported asset version " +
                              std::to_string(parsed.version));
        return false;
    }
    if (order > static_cast<std::uint32_t>(CompositionOrder::SRT)) {
        error = makeError(ErrorCode::InvalidCompositionOrder,
                          "composition order must be in [0, 5]");
        return false;
    }
    parsed.compositionOrder = static_cast<CompositionOrder>(order);

    if (!std::isfinite(parsed.framesPerSecond) ||
        parsed.framesPerSecond <= 0.0F) {
        error = makeError(ErrorCode::InvalidNumericValue,
                          "framesPerSecond must be finite and positive");
        return false;
    }
    if (parsed.frameCount == 0U) {
        error = makeError(ErrorCode::InvalidCount,
                          "frameCount must be positive");
        return false;
    }
    if (nodeCount == 0U || nodeCount > kMaxNodes) {
        error = makeError(ErrorCode::InvalidCount,
                          "nodeCount must be in [1, 65536]");
        return false;
    }
    if (meshCount > kMaxMeshes) {
        error = makeError(ErrorCode::InvalidCount,
                          "meshCount exceeds 256");
        return false;
    }
    if (sampleCount > kMaxSamples) {
        error = makeError(ErrorCode::InvalidCount,
                          "sampleCount exceeds 4194304");
        return false;
    }

    parsed.nodes.resize(nodeCount);
    for (std::uint32_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        Node& node = parsed.nodes[nodeIndex];
        if (!reader.readString32(node.name) || !reader.readI32(node.parent) ||
            !reader.readU32(node.firstSample) ||
            !reader.readU32(node.sampleCount)) {
            error = truncated(reader.offset(), "node record");
            return false;
        }
        if (node.parent < -1 ||
            node.parent >= static_cast<std::int32_t>(nodeCount)) {
            error = makeError(
                ErrorCode::InvalidParentIndex,
                "node " + std::to_string(nodeIndex) +
                    " has an out-of-range parent index");
            return false;
        }
        if (!readMatrix(reader, node.inverseBind, error)) {
            return false;
        }
    }

    parsed.samples.resize(sampleCount);
    for (std::uint32_t sampleIndex = 0; sampleIndex < sampleCount;
         ++sampleIndex) {
        AnimationSample& sample = parsed.samples[sampleIndex];
        if (!reader.readF32(sample.time) ||
            !readVec3(reader, sample.translation, "sample translation",
                      error)) {
            if (!error) {
                error = truncated(reader.offset(), "sample time");
            }
            return false;
        }
        float x = 0.0F;
        float y = 0.0F;
        float z = 0.0F;
        float w = 0.0F;
        if (!reader.readF32(x) || !reader.readF32(y) ||
            !reader.readF32(z) || !reader.readF32(w)) {
            error = truncated(reader.offset(), "sample rotation");
            return false;
        }
        sample.rotation = glm::quat(w, x, y, z);
        if (!readVec3(reader, sample.scale, "sample scale", error)) {
            return false;
        }
    }

    parsed.meshes.resize(meshCount);
    for (std::uint32_t meshIndex = 0; meshIndex < meshCount; ++meshIndex) {
        Mesh& mesh = parsed.meshes[meshIndex];
        std::uint32_t vertexCount = 0;
        std::uint32_t indexCount = 0;
        if (!reader.readString32(mesh.name) ||
            !reader.readU32(vertexCount) ||
            !reader.readU32(indexCount)) {
            error = truncated(reader.offset(), "mesh header");
            return false;
        }
        if (vertexCount > kMaxVerticesPerMesh ||
            indexCount > kMaxIndicesPerMesh) {
            error = makeError(
                ErrorCode::ResourceTooLarge,
                "mesh " + std::to_string(meshIndex) +
                    " exceeds the per-mesh vertex or index limit");
            return false;
        }

        mesh.vertices.resize(vertexCount);
        for (std::uint32_t vertexIndex = 0; vertexIndex < vertexCount;
             ++vertexIndex) {
            Vertex& vertex = mesh.vertices[vertexIndex];
            if (!readVec3(reader, vertex.position, "vertex position", error) ||
                !readVec3(reader, vertex.normal, "vertex normal", error) ||
                !readVec3(reader, vertex.color, "vertex color", error)) {
                return false;
            }
            for (std::uint32_t slot = 0; slot < 4U; ++slot) {
                if (!reader.readU32(vertex.boneIndices[slot])) {
                    error = truncated(reader.offset(), "vertex bone index");
                    return false;
                }
            }
            if (!reader.readU32(vertex.weightCount)) {
                error = truncated(reader.offset(), "vertex weight count");
                return false;
            }
            for (std::uint32_t slot = 0; slot < 4U; ++slot) {
                if (!reader.readF32(vertex.weights[slot])) {
                    error = truncated(reader.offset(), "vertex weight");
                    return false;
                }
            }
        }

        mesh.indices.resize(indexCount);
        for (std::uint32_t index = 0; index < indexCount; ++index) {
            if (!reader.readU32(mesh.indices[index])) {
                error = truncated(reader.offset(), "mesh index");
                return false;
            }
        }
    }

    if (!reader.exhausted()) {
        error = makeError(ErrorCode::InvalidCount,
                          "asset contains trailing bytes");
        return false;
    }
    if (!validateParents(parsed, error) ||
        !validateSamples(parsed, error) ||
        !validateMeshes(parsed, error)) {
        return false;
    }

    asset = std::move(parsed);
    error = {};
    return true;
}

} // namespace skel
