#include "mesh_io.hpp"

#include <array>
#include <cstring>
#include <stdexcept>

namespace rigskin {
namespace {

constexpr std::array<std::uint8_t, 8> kMeshMagic = {
    'F', 'M', 'S', 'H', '0', '0', '0', '1'};
constexpr std::size_t kHeaderBytes = 32;
constexpr std::size_t kVertexBytes = 24;
constexpr std::size_t kTriangleBytes = 24;

void appendUint32(std::vector<std::uint8_t>& output, std::uint32_t value) {
    output.push_back(static_cast<std::uint8_t>((value >> 0U) & 0xffU));
    output.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
    output.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xffU));
    output.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xffU));
}

void appendFloat(std::vector<std::uint8_t>& output, double value) {
    const float stored = static_cast<float>(value);
    std::uint32_t bits = 0;
    std::memcpy(&bits, &stored, sizeof(bits));
    appendUint32(output, bits);
}

std::uint32_t readUint32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24U);
}

}  // namespace

std::vector<std::uint8_t> encodeMeshBinary(
    const Model& model,
    std::uint32_t frameIndex,
    const DeformedMesh& mesh) {
    if (mesh.positions.size() != model.vertices.size() ||
        mesh.vertexNormals.size() != model.vertices.size() ||
        mesh.faceNormals.size() != model.triangles.size()) {
        throw std::runtime_error("deformed mesh shape does not match model");
    }

    std::vector<std::uint8_t> output;
    output.reserve(
        kHeaderBytes + mesh.positions.size() * kVertexBytes +
        mesh.faceNormals.size() * kTriangleBytes);
    output.insert(output.end(), kMeshMagic.begin(), kMeshMagic.end());
    appendUint32(output, frameIndex);
    appendUint32(output, static_cast<std::uint32_t>(mesh.positions.size()));
    appendUint32(output, static_cast<std::uint32_t>(model.triangles.size()));
    appendUint32(output, mesh.degenerateTriangleCount);
    appendUint32(
        output, static_cast<std::uint32_t>(model.zeroWeightVertices.size()));
    appendUint32(output, 0U);

    for (std::size_t vertexIndex = 0; vertexIndex < mesh.positions.size();
         ++vertexIndex) {
        const Vec3& position = mesh.positions[vertexIndex];
        const Vec3& normal = mesh.vertexNormals[vertexIndex];
        appendFloat(output, position.x);
        appendFloat(output, position.y);
        appendFloat(output, position.z);
        appendFloat(output, normal.x);
        appendFloat(output, normal.y);
        appendFloat(output, normal.z);
    }

    for (std::size_t triangleIndex = 0; triangleIndex < model.triangles.size();
         ++triangleIndex) {
        const Triangle& triangle = model.triangles[triangleIndex];
        const Vec3& normal = mesh.faceNormals[triangleIndex];
        appendUint32(output, triangle.a);
        appendUint32(output, triangle.b);
        appendUint32(output, triangle.c);
        appendFloat(output, normal.x);
        appendFloat(output, normal.y);
        appendFloat(output, normal.z);
    }

    return output;
}

bool decodeMeshHeader(
    const std::vector<std::uint8_t>& bytes,
    MeshHeader& header,
    std::string& error) {
    if (bytes.size() < kHeaderBytes) {
        error = "mesh file is shorter than its header";
        return false;
    }
    if (!std::equal(kMeshMagic.begin(), kMeshMagic.end(), bytes.begin())) {
        error = "mesh magic/version is invalid";
        return false;
    }

    MeshHeader parsed;
    parsed.frame = readUint32(bytes, 8);
    parsed.vertexCount = readUint32(bytes, 12);
    parsed.triangleCount = readUint32(bytes, 16);
    parsed.degenerateTriangleCount = readUint32(bytes, 20);
    parsed.zeroWeightVertexCount = readUint32(bytes, 24);
    const std::uint32_t reserved = readUint32(bytes, 28);
    if (reserved != 0) {
        error = "mesh header reserved field is nonzero";
        return false;
    }

    const std::uint64_t expected =
        static_cast<std::uint64_t>(kHeaderBytes) +
        static_cast<std::uint64_t>(parsed.vertexCount) * kVertexBytes +
        static_cast<std::uint64_t>(parsed.triangleCount) * kTriangleBytes;
    if (expected != bytes.size()) {
        error = "mesh file length does not match embedded counts";
        return false;
    }

    header = parsed;
    return true;
}

std::vector<std::uint8_t> encodePpm(
    int width,
    int height,
    const std::vector<std::uint8_t>& rgb) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("invalid PPM dimensions");
    }
    const std::size_t expected =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;
    if (rgb.size() != expected) {
        throw std::runtime_error("RGB buffer size does not match PPM dimensions");
    }

    const std::string header =
        "P6\n" + std::to_string(width) + " " + std::to_string(height) + "\n255\n";
    std::vector<std::uint8_t> output;
    output.reserve(header.size() + rgb.size());
    output.insert(output.end(), header.begin(), header.end());
    output.insert(output.end(), rgb.begin(), rgb.end());
    return output;
}

bool validatePpmHeader(
    const std::vector<std::uint8_t>& bytes,
    int expectedWidth,
    int expectedHeight,
    std::string& error) {
    const std::string expectedHeader =
        "P6\n" + std::to_string(expectedWidth) + " " +
        std::to_string(expectedHeight) + "\n255\n";
    const std::size_t expectedSize =
        expectedHeader.size() +
        static_cast<std::size_t>(expectedWidth) *
            static_cast<std::size_t>(expectedHeight) * 3U;
    if (bytes.size() != expectedSize) {
        error = "PPM file size does not match expected dimensions";
        return false;
    }
    if (bytes.size() < expectedHeader.size() ||
        !std::equal(expectedHeader.begin(), expectedHeader.end(), bytes.begin())) {
        error = "PPM header does not match expected fixed header";
        return false;
    }
    return true;
}

}  // namespace rigskin
