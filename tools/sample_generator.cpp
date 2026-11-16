#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/geometric.hpp>
#include <glm/glm.hpp>

namespace {

using Bytes = std::vector<std::uint8_t>;

void appendU32Big(Bytes& bytes, std::uint32_t value) {
    bytes.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>(value & 0xFFU));
}

std::uint32_t crc32(const Bytes& data) {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (const std::uint8_t byte : data) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1U) ^ (0xEDB88320U & (0U - (crc & 1U)));
        }
    }
    return crc ^ 0xFFFFFFFFU;
}

std::uint32_t adler32(const Bytes& data) {
    constexpr std::uint32_t modulus = 65521U;
    std::uint32_t a = 1U;
    std::uint32_t b = 0U;
    for (const std::uint8_t byte : data) {
        a = (a + byte) % modulus;
        b = (b + a) % modulus;
    }
    return (b << 16U) | a;
}

void appendChunk(Bytes& png, const std::array<char, 4>& type, const Bytes& data) {
    appendU32Big(png, static_cast<std::uint32_t>(data.size()));
    Bytes crcInput(type.begin(), type.end());
    crcInput.insert(crcInput.end(), data.begin(), data.end());
    png.insert(png.end(), type.begin(), type.end());
    png.insert(png.end(), data.begin(), data.end());
    appendU32Big(png, crc32(crcInput));
}

Bytes zlibStored(const Bytes& raw) {
    Bytes zlib;
    zlib.push_back(0x78U);
    zlib.push_back(0x01U);
    std::size_t offset = 0;
    do {
        const std::size_t length = std::min<std::size_t>(65535U, raw.size() - offset);
        const bool finalBlock = offset + length == raw.size();
        zlib.push_back(finalBlock ? 0x01U : 0x00U);
        const std::uint16_t length16 = static_cast<std::uint16_t>(length);
        const std::uint16_t inverse = static_cast<std::uint16_t>(~length16);
        zlib.push_back(static_cast<std::uint8_t>(length16 & 0xFFU));
        zlib.push_back(static_cast<std::uint8_t>((length16 >> 8U) & 0xFFU));
        zlib.push_back(static_cast<std::uint8_t>(inverse & 0xFFU));
        zlib.push_back(static_cast<std::uint8_t>((inverse >> 8U) & 0xFFU));
        zlib.insert(
            zlib.end(),
            raw.begin() + static_cast<std::ptrdiff_t>(offset),
            raw.begin() + static_cast<std::ptrdiff_t>(offset + length));
        offset += length;
    } while (offset < raw.size());
    appendU32Big(zlib, adler32(raw));
    return zlib;
}

void writePng(
    const std::filesystem::path& path,
    int width,
    int height,
    const std::vector<std::array<std::uint8_t, 3>>& pixels) {
    if (pixels.size() !=
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {
        throw std::runtime_error("invalid PNG pixel buffer");
    }

    Bytes raw;
    raw.reserve((static_cast<std::size_t>(width) * 3U + 1U) *
                static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0U);
        for (int x = 0; x < width; ++x) {
            const auto& pixel = pixels[
                static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(x)];
            raw.insert(raw.end(), pixel.begin(), pixel.end());
        }
    }

    Bytes png = {0x89U, 0x50U, 0x4EU, 0x47U, 0x0DU, 0x0AU, 0x1AU, 0x0AU};
    Bytes ihdr;
    appendU32Big(ihdr, static_cast<std::uint32_t>(width));
    appendU32Big(ihdr, static_cast<std::uint32_t>(height));
    ihdr.push_back(8U);
    ihdr.push_back(2U);
    ihdr.push_back(0U);
    ihdr.push_back(0U);
    ihdr.push_back(0U);
    appendChunk(png, {'I', 'H', 'D', 'R'}, ihdr);
    appendChunk(png, {'I', 'D', 'A', 'T'}, zlibStored(raw));
    appendChunk(png, {'I', 'E', 'N', 'D'}, {});

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot write " + path.u8string());
    }
    output.write(
        reinterpret_cast<const char*>(png.data()),
        static_cast<std::streamsize>(png.size()));
    if (!output) {
        throw std::runtime_error("failed writing " + path.u8string());
    }
}

std::uint8_t toByte(float value) {
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    return static_cast<std::uint8_t>(clamped * 255.0f + 0.5f);
}

float triangleWave(float value) {
    const float repeating = value - std::floor(value);
    return 4.0f * std::abs(repeating - 0.5f) - 1.0f;
}

void writeTextFile(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot write " + path.u8string());
    }
    output << text;
    if (!output) {
        throw std::runtime_error("failed writing " + path.u8string());
    }
}

struct Corner {
    glm::vec3 position;
    glm::vec2 uv;
    glm::vec3 normal;
    glm::vec3 tangent;
};

std::vector<std::array<Corner, 3>> buildFaces() {
    const std::array<glm::vec3, 4> positions = {
        glm::vec3(0.0f, 1.2f, 0.0f),
        glm::vec3(-1.2f, -1.0f, 1.2f),
        glm::vec3(1.2f, -1.0f, 1.2f),
        glm::vec3(0.0f, -1.0f, -1.2f),
    };
    const std::array<std::array<int, 3>, 4> indices = {{
        {{0, 1, 2}},
        {{0, 2, 3}},
        {{0, 3, 1}},
        {{1, 3, 2}},
    }};
    const std::array<glm::vec2, 3> uvs = {
        glm::vec2(0.04f, 0.04f),
        glm::vec2(0.96f, 0.04f),
        glm::vec2(0.50f, 0.96f),
    };

    std::vector<std::array<Corner, 3>> faces;
    for (const auto& faceIndices : indices) {
        std::array<Corner, 3> face;
        for (int corner = 0; corner < 3; ++corner) {
            face[static_cast<std::size_t>(corner)].position =
                positions[static_cast<std::size_t>(
                    faceIndices[static_cast<std::size_t>(corner)])];
            face[static_cast<std::size_t>(corner)].uv =
                uvs[static_cast<std::size_t>(corner)];
        }

        const glm::vec3 normal = glm::normalize(glm::cross(
            face[1].position - face[0].position,
            face[2].position - face[0].position));
        const glm::vec3 edge1 = face[1].position - face[0].position;
        const glm::vec3 edge2 = face[2].position - face[0].position;
        const glm::vec2 deltaUv1 = face[1].uv - face[0].uv;
        const glm::vec2 deltaUv2 = face[2].uv - face[0].uv;
        const float determinant =
            deltaUv1.x * deltaUv2.y - deltaUv1.y * deltaUv2.x;
        glm::vec3 tangent =
            (edge1 * deltaUv2.y - edge2 * deltaUv1.y) / determinant;
        tangent -= normal * glm::dot(tangent, normal);
        tangent = glm::normalize(tangent);

        for (Corner& corner : face) {
            corner.normal = normal;
            corner.tangent = tangent;
        }
        faces.push_back(face);
    }
    return faces;
}

std::string buildObj(
    const std::vector<std::array<Corner, 3>>& faces,
    bool includeTangents,
    const std::string& materialFile) {
    std::ostringstream output;
    output << "# Deterministic tetrahedron fixture generated by make_sample.\n";
    output << "mtllib " << materialFile << "\n\n";

    for (const auto& face : faces) {
        for (const Corner& corner : face) {
            output << "v " << corner.position.x << ' ' << corner.position.y << ' '
                   << corner.position.z << '\n';
            if (includeTangents) {
                output << "vtangent " << corner.tangent.x << ' ' << corner.tangent.y
                       << ' ' << corner.tangent.z << '\n';
            }
        }
    }
    output << '\n';
    for (const auto& face : faces) {
        for (const Corner& corner : face) {
            output << "vt " << corner.uv.x << ' ' << corner.uv.y << '\n';
        }
    }
    output << '\n';
    for (const auto& face : faces) {
        for (const Corner& corner : face) {
            output << "vn " << corner.normal.x << ' ' << corner.normal.y << ' '
                   << corner.normal.z << '\n';
        }
    }
    output << "\nusemtl tetra\n";
    int index = 1;
    for (std::size_t face = 0; face < faces.size(); ++face) {
        output << "f";
        for (int corner = 0; corner < 3; ++corner) {
            output << ' ' << index << '/' << index << '/' << index;
            ++index;
        }
        output << '\n';
    }
    return output.str();
}

std::string buildMtl(bool includeNormalMap) {
    std::ostringstream output;
    output << "# Deterministic tetrahedron material.\n";
    output << "newmtl tetra\n";
    output << "Kd 1.0 1.0 1.0\n";
    output << "Ks 0.55 0.55 0.55\n";
    output << "Ns 48\n";
    output << "map_Kd tetra_diffuse.png\n";
    if (includeNormalMap) {
        output << "norm tetra_normal.png\n";
    }
    return output.str();
}

std::vector<std::array<std::uint8_t, 3>> makeDiffuse(int width, int height) {
    std::vector<std::array<std::uint8_t, 3>> pixels(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const bool checker = ((x / 8) + (y / 8)) % 2 == 0;
            const float ramp = static_cast<float>(x + y) /
                               static_cast<float>(width + height - 2);
            glm::vec3 color = checker
                                  ? glm::vec3(0.86f, 0.38f, 0.12f)
                                  : glm::vec3(0.12f, 0.55f, 0.82f);
            color += glm::vec3(0.10f, 0.08f, 0.05f) * ramp;
            if (x >= width / 2 - 2 && x <= width / 2 + 2) {
                color = glm::mix(color, glm::vec3(0.95f, 0.92f, 0.80f), 0.55f);
            }
            pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)] = {
                toByte(color.r), toByte(color.g), toByte(color.b)};
        }
    }
    return pixels;
}

std::vector<std::array<std::uint8_t, 3>> makeNormalMap(int width, int height) {
    std::vector<std::array<std::uint8_t, 3>> pixels(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) /
                            static_cast<float>(width);
            const float v = (static_cast<float>(y) + 0.5f) /
                            static_cast<float>(height);
            float nx = 0.38f * triangleWave(u * 2.0f);
            float ny = 0.34f * triangleWave(v * 3.0f);
            float nz = std::sqrt(std::max(0.0f, 1.0f - nx * nx - ny * ny));
            const glm::vec3 normal = glm::normalize(glm::vec3(nx, ny, nz));
            const glm::vec3 encoded = normal * 0.5f + 0.5f;
            pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)] = {
                toByte(encoded.x), toByte(encoded.y), toByte(encoded.z)};
        }
    }
    return pixels;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path outputDirectory =
            argc == 2 ? std::filesystem::u8path(argv[1])
                      : throw std::runtime_error(
                            "usage: make_sample <output-directory>");
        std::filesystem::create_directories(outputDirectory);

        const auto faces = buildFaces();
        writeTextFile(
            outputDirectory / "tetra.obj",
            buildObj(faces, false, "tetra.mtl"));
        writeTextFile(
            outputDirectory / "tetra_with_tangents.obj",
            buildObj(faces, true, "tetra.mtl"));
        writeTextFile(
            outputDirectory / "tetra_no_normal.obj",
            buildObj(faces, false, "tetra_no_normal.mtl"));
        writeTextFile(outputDirectory / "tetra.mtl", buildMtl(true));
        writeTextFile(outputDirectory / "tetra_no_normal.mtl", buildMtl(false));

        constexpr int textureSize = 64;
        writePng(
            outputDirectory / "tetra_diffuse.png",
            textureSize,
            textureSize,
            makeDiffuse(textureSize, textureSize));
        writePng(
            outputDirectory / "tetra_normal.png",
            textureSize,
            textureSize,
            makeNormalMap(textureSize, textureSize));

        std::cout << "Generated deterministic sample in "
                  << outputDirectory.u8string() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "make_sample: " << error.what() << "\n";
        return 1;
    }
}
