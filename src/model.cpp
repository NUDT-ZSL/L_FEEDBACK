#include "model.hpp"

#include "error.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <glm/geometric.hpp>

namespace preview {
namespace {

struct FaceVertex {
    int position = -1;
    int uv = -1;
    int normal = -1;
};

struct Face {
    std::vector<FaceVertex> vertices;
    std::string materialName;
};

struct ParsedMaterial {
    Material material;
    bool hasDiffuseMap = false;
    bool hasNormalMap = false;
};

std::string unicodePath(const std::filesystem::path& path) {
    return path.u8string();
}

std::string trim(const std::string& text) {
    const std::size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const std::size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::string stripComment(const std::string& line) {
    const std::size_t comment = line.find('#');
    return comment == std::string::npos ? line : line.substr(0, comment);
}

std::pair<std::string, std::string> splitKeyword(const std::string& line) {
    const std::size_t separator = line.find_first_of(" \t");
    if (separator == std::string::npos) {
        return {line, {}};
    }
    return {line.substr(0, separator), trim(line.substr(separator + 1))};
}

std::string parsePathArgument(const std::string& argument) {
    const std::string value = trim(argument);
    if (value.empty()) {
        return {};
    }
    if (value.front() == '"') {
        std::string result;
        for (std::size_t i = 1; i < value.size(); ++i) {
            const char c = value[i];
            if (c == '\\' && i + 1 < value.size() &&
                (value[i + 1] == '"' || value[i + 1] == '\\')) {
                result.push_back(value[++i]);
            } else if (c == '"') {
                return result;
            } else {
                result.push_back(c);
            }
        }
        return {};
    }
    return value;
}

float parseFloat(const std::string& token, const std::string& context) {
    float value = 0.0f;
    const char* first = token.data();
    const char* last = token.data() + token.size();
    const auto result = std::from_chars(first, last, value, std::chars_format::general);
    if (result.ec != std::errc() || result.ptr != last || !std::isfinite(value)) {
        throw AppError(ErrorCode::ObjMissing, "invalid floating-point value in " + context);
    }
    return value;
}

int parseIndex(
    const std::string& token,
    int count,
    const std::string& attributeName,
    std::size_t lineNumber) {
    if (token.empty()) {
        throw AppError(
            ErrorCode::ObjMissing,
            "face on line " + std::to_string(lineNumber) + " is missing " + attributeName);
    }
    int value = 0;
    const char* first = token.data();
    const char* last = token.data() + token.size();
    const auto result = std::from_chars(first, last, value);
    if (result.ec != std::errc() || result.ptr != last || value == 0) {
        throw AppError(
            ErrorCode::IndexInvalid,
            "invalid " + attributeName + " index '" + token + "' on line " +
                std::to_string(lineNumber));
    }
    int resolved = value > 0 ? value - 1 : count + value;
    if (resolved < 0 || resolved >= count) {
        throw AppError(
            ErrorCode::IndexInvalid,
            attributeName + " index " + token + " is out of range on line " +
                std::to_string(lineNumber));
    }
    return resolved;
}

FaceVertex parseFaceVertex(
    const std::string& token,
    int positionCount,
    int uvCount,
    int normalCount,
    std::size_t lineNumber) {
    std::array<std::string, 3> fields{};
    int fieldCount = 1;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= token.size(); ++i) {
        if (i == token.size() || token[i] == '/') {
            if (fieldCount > 3) {
                throw AppError(
                    ErrorCode::IndexInvalid,
                    "too many components in face vertex on line " + std::to_string(lineNumber));
            }
            fields[static_cast<std::size_t>(fieldCount - 1)] = token.substr(start, i - start);
            ++fieldCount;
            start = i + 1;
        }
    }
    --fieldCount;
    if (fieldCount < 3) {
        throw AppError(
            ErrorCode::ObjMissing,
            "face vertex on line " + std::to_string(lineNumber) +
                " must contain position/texture/normal indices");
    }

    FaceVertex result;
    result.position = parseIndex(fields[0], positionCount, "position", lineNumber);
    result.uv = parseIndex(fields[1], uvCount, "texture coordinate", lineNumber);
    result.normal = parseIndex(fields[2], normalCount, "normal", lineNumber);
    return result;
}

std::filesystem::path resolveExistingAsset(
    const std::filesystem::path& preferredDirectory,
    const std::filesystem::path& fallbackDirectory,
    const std::string& value) {
    if (value.empty()) {
        return {};
    }
    std::filesystem::path path = std::filesystem::u8path(value);
    if (path.is_absolute()) {
        return path;
    }
    path = preferredDirectory / path;
    if (std::filesystem::exists(path)) {
        return path;
    }
    const std::filesystem::path fallback = fallbackDirectory / std::filesystem::u8path(value);
    return std::filesystem::exists(fallback) ? fallback : path;
}

void parseMtl(
    const std::filesystem::path& mtlPath,
    const std::filesystem::path& objDirectory,
    std::map<std::string, ParsedMaterial>& materials) {
    std::ifstream input(mtlPath, std::ios::binary);
    if (!input) {
        throw AppError(
            ErrorCode::MatMissing,
            "cannot open MTL file: " + unicodePath(mtlPath));
    }

    ParsedMaterial* current = nullptr;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        line = trim(stripComment(line));
        if (line.empty()) {
            continue;
        }
        const auto [keyword, argument] = splitKeyword(line);
        if (keyword == "newmtl") {
            if (argument.empty()) {
                throw AppError(
                    ErrorCode::MatMissing,
                    "empty material name on MTL line " + std::to_string(lineNumber));
            }
            if (materials.find(argument) != materials.end()) {
                throw AppError(
                    ErrorCode::MatMissing,
                    "duplicate material name '" + argument + "'");
            }
            ParsedMaterial parsed;
            parsed.material.name = argument;
            auto inserted = materials.emplace(argument, std::move(parsed));
            current = &inserted.first->second;
        } else if (keyword == "map_Kd") {
            if (current == nullptr) {
                throw AppError(
                    ErrorCode::MatMissing,
                    "map_Kd appears before newmtl on MTL line " + std::to_string(lineNumber));
            }
            const std::string pathText = parsePathArgument(argument);
            if (pathText.empty()) {
                throw AppError(
                    ErrorCode::MatMissing,
                    "empty map_Kd path on MTL line " + std::to_string(lineNumber));
            }
            current->material.diffusePath =
                resolveExistingAsset(mtlPath.parent_path(), objDirectory, pathText);
            current->hasDiffuseMap = true;
        } else if (keyword == "norm" || keyword == "map_Bump" || keyword == "bump") {
            if (current == nullptr) {
                throw AppError(
                    ErrorCode::MatMissing,
                    keyword + " appears before newmtl on MTL line " + std::to_string(lineNumber));
            }
            const std::string pathText = parsePathArgument(argument);
            current->material.normalPath =
                resolveExistingAsset(mtlPath.parent_path(), objDirectory, pathText);
            current->hasNormalMap = !pathText.empty();
        } else if (keyword == "Kd") {
            if (current != nullptr) {
                std::istringstream stream(argument);
                float red = 1.0f;
                float green = 1.0f;
                float blue = 1.0f;
                if (stream >> red >> green >> blue) {
                    current->material.diffuseTint = glm::vec3(red, green, blue);
                }
            }
        } else if (keyword == "Ks") {
            if (current != nullptr) {
                std::istringstream stream(argument);
                float red = 0.0f;
                float green = 0.0f;
                float blue = 0.0f;
                if (stream >> red >> green >> blue) {
                    current->material.specularStrength =
                        std::clamp((red + green + blue) / 3.0f, 0.0f, 1.0f);
                }
            }
        } else if (keyword == "Ns") {
            if (current != nullptr) {
                std::istringstream stream(argument);
                float shininess = 0.0f;
                if (stream >> shininess && shininess > 0.0f) {
                    current->material.shininess = std::clamp(shininess, 1.0f, 4096.0f);
                }
            }
        }
    }
}

glm::vec3 fallbackTangent(const glm::vec3& normal) {
    const glm::vec3 reference =
        std::abs(normal.x) < 0.9f ? glm::vec3(1.0f, 0.0f, 0.0f)
                                  : glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 tangent = glm::cross(reference, normal);
    if (glm::length(tangent) <= 1.0e-8f) {
        return glm::vec3(1.0f, 0.0f, 0.0f);
    }
    return glm::normalize(tangent);
}

struct VertexKey {
    int position = 0;
    int uv = 0;
    int normal = 0;

    bool operator<(const VertexKey& other) const noexcept {
        return std::tie(position, uv, normal) <
               std::tie(other.position, other.uv, other.normal);
    }
};

} // namespace

Model loadModel(const std::filesystem::path& objPath) {
    std::ifstream input(objPath, std::ios::binary);
    if (!input) {
        throw AppError(
            ErrorCode::ObjMissing,
            "cannot open OBJ file: " + unicodePath(objPath));
    }

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> positionTangents;
    std::vector<bool> hasPositionTangent;
    std::vector<Face> faces;
    std::vector<std::string> mtlLibraries;
    std::string currentMaterial;

    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        line = trim(stripComment(line));
        if (lineNumber == 1 && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF) {
            line.erase(0, 3);
        }
        if (line.empty()) {
            continue;
        }

        const auto [keyword, argument] = splitKeyword(line);
        if (keyword == "v") {
            std::istringstream stream(argument);
            std::string x;
            std::string y;
            std::string z;
            if (!(stream >> x >> y >> z)) {
                throw AppError(
                    ErrorCode::ObjMissing,
                    "incomplete vertex on OBJ line " + std::to_string(lineNumber));
            }
            positions.push_back(glm::vec3(
                parseFloat(x, "vertex"),
                parseFloat(y, "vertex"),
                parseFloat(z, "vertex")));
            positionTangents.push_back(glm::vec3(0.0f));
            hasPositionTangent.push_back(false);
        } else if (keyword == "vt") {
            std::istringstream stream(argument);
            std::string u;
            std::string v;
            if (!(stream >> u >> v)) {
                throw AppError(
                    ErrorCode::ObjMissing,
                    "incomplete texture coordinate on OBJ line " +
                        std::to_string(lineNumber));
            }
            uvs.emplace_back(
                parseFloat(u, "texture coordinate"),
                parseFloat(v, "texture coordinate"));
        } else if (keyword == "vn") {
            std::istringstream stream(argument);
            std::string x;
            std::string y;
            std::string z;
            if (!(stream >> x >> y >> z)) {
                throw AppError(
                    ErrorCode::ObjMissing,
                    "incomplete normal on OBJ line " + std::to_string(lineNumber));
            }
            normals.push_back(glm::vec3(
                parseFloat(x, "normal"),
                parseFloat(y, "normal"),
                parseFloat(z, "normal")));
        } else if (keyword == "vtangent" || keyword == "tangent") {
            if (positions.empty()) {
                throw AppError(
                    ErrorCode::ObjMissing,
                    "tangent appears before any vertex on OBJ line " +
                        std::to_string(lineNumber));
            }
            std::istringstream stream(argument);
            std::string x;
            std::string y;
            std::string z;
            if (!(stream >> x >> y >> z)) {
                throw AppError(
                    ErrorCode::ObjMissing,
                    "incomplete vertex tangent on OBJ line " +
                        std::to_string(lineNumber));
            }
            const glm::vec3 tangent(
                parseFloat(x, "tangent"),
                parseFloat(y, "tangent"),
                parseFloat(z, "tangent"));
            positionTangents.back() = tangent;
            hasPositionTangent.back() = glm::length(tangent) > 1.0e-8f;
        } else if (keyword == "f") {
            std::istringstream stream(argument);
            std::vector<FaceVertex> faceVertices;
            std::string token;
            while (stream >> token) {
                faceVertices.push_back(parseFaceVertex(
                    token,
                    static_cast<int>(positions.size()),
                    static_cast<int>(uvs.size()),
                    static_cast<int>(normals.size()),
                    lineNumber));
            }
            if (faceVertices.size() < 3) {
                throw AppError(
                    ErrorCode::ObjMissing,
                    "face on OBJ line " + std::to_string(lineNumber) +
                        " has fewer than three vertices");
            }
            if (currentMaterial.empty()) {
                throw AppError(
                    ErrorCode::MatMissing,
                    "face on OBJ line " + std::to_string(lineNumber) +
                        " does not follow a usemtl statement");
            }
            faces.push_back(Face{std::move(faceVertices), currentMaterial});
        } else if (keyword == "mtllib") {
            const std::string pathText = parsePathArgument(argument);
            if (pathText.empty()) {
                throw AppError(
                    ErrorCode::MatMissing,
                    "empty mtllib path on OBJ line " + std::to_string(lineNumber));
            }
            mtlLibraries.push_back(pathText);
        } else if (keyword == "usemtl") {
            currentMaterial = argument;
            if (currentMaterial.empty()) {
                throw AppError(
                    ErrorCode::MatMissing,
                    "empty usemtl name on OBJ line " + std::to_string(lineNumber));
            }
        }
    }

    if (positions.empty() || uvs.empty() || normals.empty()) {
        throw AppError(
            ErrorCode::ObjMissing,
            "OBJ requires at least one v, vt, and vn record");
    }
    if (faces.empty()) {
        throw AppError(ErrorCode::ObjMissing, "OBJ contains no faces");
    }
    if (mtlLibraries.empty()) {
        throw AppError(ErrorCode::MatMissing, "OBJ does not declare an mtllib");
    }

    std::map<std::string, ParsedMaterial> parsedMaterials;
    for (const std::string& library : mtlLibraries) {
        std::filesystem::path mtlPath = std::filesystem::u8path(library);
        if (!mtlPath.is_absolute()) {
            mtlPath = objPath.parent_path() / mtlPath;
        }
        parseMtl(mtlPath, objPath.parent_path(), parsedMaterials);
    }
    if (parsedMaterials.empty()) {
        throw AppError(ErrorCode::MatMissing, "MTL does not define any materials");
    }

    std::map<std::string, int> materialIndices;
    Model model;
    model.materials.reserve(parsedMaterials.size());
    for (auto& [name, parsed] : parsedMaterials) {
        if (!parsed.hasDiffuseMap || parsed.material.diffusePath.empty()) {
            throw AppError(
                ErrorCode::MatMissing,
                "material '" + name + "' is missing a map_Kd diffuse texture");
        }
        if (!std::filesystem::is_regular_file(parsed.material.diffusePath)) {
            throw AppError(
                ErrorCode::MatMissing,
                "diffuse texture does not exist for material '" + name + "': " +
                    unicodePath(parsed.material.diffusePath));
        }

        parsed.material.diffuseTexture = loadTexture(parsed.material.diffusePath);
        if (parsed.hasNormalMap && !parsed.material.normalPath.empty() &&
            std::filesystem::is_regular_file(parsed.material.normalPath)) {
            parsed.material.normalTexture = loadTexture(parsed.material.normalPath);
        }
        materialIndices.emplace(name, static_cast<int>(model.materials.size()));
        model.materials.push_back(std::move(parsed.material));
    }

    std::vector<glm::vec3> accumulatedTangents;
    std::vector<glm::vec3> accumulatedBitangents;
    std::map<VertexKey, unsigned int> vertexMap;
    const auto resolveVertex = [&](const FaceVertex& faceVertex) -> unsigned int {
        const VertexKey key{faceVertex.position, faceVertex.uv, faceVertex.normal};
        const auto existing = vertexMap.find(key);
        if (existing != vertexMap.end()) {
            return existing->second;
        }

        Vertex vertex;
        vertex.position = positions[static_cast<std::size_t>(faceVertex.position)];
        vertex.uv = uvs[static_cast<std::size_t>(faceVertex.uv)];
        vertex.normal = normals[static_cast<std::size_t>(faceVertex.normal)];
        if (glm::length(vertex.normal) <= 1.0e-8f) {
            throw AppError(ErrorCode::ObjMissing, "OBJ contains a zero-length normal");
        }
        vertex.normal = glm::normalize(vertex.normal);
        if (hasPositionTangent[static_cast<std::size_t>(faceVertex.position)]) {
            vertex.tangent =
                positionTangents[static_cast<std::size_t>(faceVertex.position)];
        }

        const unsigned int index = static_cast<unsigned int>(model.vertices.size());
        model.vertices.push_back(vertex);
        vertexMap.emplace(key, index);
        accumulatedTangents.emplace_back(0.0f);
        accumulatedBitangents.emplace_back(0.0f);
        return index;
    };

    for (const Face& face : faces) {
        const auto material = materialIndices.find(face.materialName);
        if (material == materialIndices.end()) {
            throw AppError(
                ErrorCode::MatMissing,
                "OBJ references undefined material '" + face.materialName + "'");
        }
        const int materialIndex = material->second;
        for (std::size_t i = 1; i + 1 < face.vertices.size(); ++i) {
            Triangle triangle;
            triangle.indices[0] = resolveVertex(face.vertices[0]);
            triangle.indices[1] = resolveVertex(face.vertices[i]);
            triangle.indices[2] = resolveVertex(face.vertices[i + 1]);
            triangle.materialIndex = materialIndex;
            model.triangles.push_back(triangle);
        }
    }

    for (const Triangle& triangle : model.triangles) {
        const Vertex& v0 = model.vertices[triangle.indices[0]];
        const Vertex& v1 = model.vertices[triangle.indices[1]];
        const Vertex& v2 = model.vertices[triangle.indices[2]];

        const glm::vec3 edge1 = v1.position - v0.position;
        const glm::vec3 edge2 = v2.position - v0.position;
        const glm::vec2 deltaUv1 = v1.uv - v0.uv;
        const glm::vec2 deltaUv2 = v2.uv - v0.uv;
        const float determinant =
            deltaUv1.x * deltaUv2.y - deltaUv1.y * deltaUv2.x;
        if (std::abs(determinant) <= 1.0e-12f) {
            continue;
        }

        const glm::vec3 tangent =
            (edge1 * deltaUv2.y - edge2 * deltaUv1.y) / determinant;
        const glm::vec3 bitangent =
            (edge2 * deltaUv1.x - edge1 * deltaUv2.x) / determinant;
        for (unsigned int index : triangle.indices) {
            accumulatedTangents[index] += tangent;
            accumulatedBitangents[index] += bitangent;
        }
    }

    // A second pass is required after all unique vertices exist: identify supplied
    // tangents by their original position index and orthonormalize each basis.
    for (const auto& [key, vertexIndex] : vertexMap) {
        Vertex& vertex = model.vertices[vertexIndex];
        const bool supplied =
            hasPositionTangent[static_cast<std::size_t>(key.position)];
        glm::vec3 tangent = supplied
                                ? positionTangents[static_cast<std::size_t>(key.position)]
                                : accumulatedTangents[vertexIndex];
        tangent -= vertex.normal * glm::dot(tangent, vertex.normal);
        if (glm::length(tangent) <= 1.0e-8f) {
            tangent = fallbackTangent(vertex.normal);
        } else {
            tangent = glm::normalize(tangent);
        }
        vertex.tangent = tangent;

        const glm::vec3 bitangent = accumulatedBitangents[vertexIndex];
        if (glm::length(bitangent) > 1.0e-10f) {
            const float handedness =
                glm::dot(glm::cross(vertex.normal, tangent), bitangent);
            vertex.tangentSign = handedness < 0.0f ? -1.0f : 1.0f;
        } else {
            vertex.tangentSign = 1.0f;
        }
    }

    return model;
}

} // namespace preview
