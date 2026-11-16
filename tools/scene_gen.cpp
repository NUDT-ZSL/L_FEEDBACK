#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct Vec2 {
  float x;
  float y;
};

struct Vec3 {
  float x;
  float y;
  float z;
};

struct MeshData {
  std::string name;
  std::vector<Vec3> positions;
  std::vector<Vec3> normals;
  std::vector<Vec3> tangents;
  std::vector<Vec2> uvs;
  std::vector<std::uint32_t> indices;
  Vec3 baseColor;
  float roughness;
  float metallic;
  float anisotropyStrength;
  float anisotropyRotation;
};

std::string number(float value) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(9) << value;
  std::string result = stream.str();
  if (result == "-0") {
    result = "0";
  }
  return result;
}

template <typename Value, typename Writer>
void writeArray(std::ostream& stream, const std::vector<Value>& values, Writer writer) {
  stream << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      stream << ",";
    }
    writer(values[i]);
  }
  stream << "]";
}

void writeVec3Array(std::ostream& stream, const std::vector<Vec3>& values) {
  writeArray(stream, values, [&](const Vec3& value) {
    stream << "[" << number(value.x) << "," << number(value.y) << ","
           << number(value.z) << "]";
  });
}

void writeVec2Array(std::ostream& stream, const std::vector<Vec2>& values) {
  writeArray(stream, values, [&](const Vec2& value) {
    stream << "[" << number(value.x) << "," << number(value.y) << "]";
  });
}

void writeIndexArray(std::ostream& stream, const std::vector<std::uint32_t>& values) {
  writeArray(stream, values,
             [&](std::uint32_t value) { stream << value; });
}

MeshData makeGround() {
  MeshData ground;
  ground.name = "ground";
  ground.positions = {
      {-7.0f, -0.75f, -7.0f}, {7.0f, -0.75f, -7.0f},
      {7.0f, -0.75f, 7.0f},   {-7.0f, -0.75f, 7.0f}};
  ground.normals = {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}};
  ground.tangents = {{1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
  ground.uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  ground.indices = {0, 3, 2, 0, 2, 1};
  ground.baseColor = {0.18f, 0.22f, 0.26f};
  ground.roughness = 0.68f;
  ground.metallic = 0.08f;
  ground.anisotropyStrength = 0.25f;
  ground.anisotropyRotation = 0.0f;
  return ground;
}

MeshData makeTorus() {
  constexpr int majorSegments = 72;
  constexpr int tubeSegments = 32;
  constexpr float majorRadius = 1.45f;
  constexpr float tubeRadius = 0.52f;
  constexpr float centerY = 0.35f;

  MeshData torus;
  torus.name = "anisotropic_torus";
  torus.positions.reserve((majorSegments + 1) * (tubeSegments + 1));
  torus.normals.reserve(torus.positions.capacity());
  torus.tangents.reserve(torus.positions.capacity());
  torus.uvs.reserve(torus.positions.capacity());
  torus.indices.reserve(majorSegments * tubeSegments * 6);

  for (int i = 0; i <= majorSegments; ++i) {
    const float u = static_cast<float>(i) / static_cast<float>(majorSegments);
    const float phi = u * 2.0f * kPi;
    for (int j = 0; j <= tubeSegments; ++j) {
      const float v = static_cast<float>(j) / static_cast<float>(tubeSegments);
      const float theta = v * 2.0f * kPi;
      const float radial = majorRadius + tubeRadius * std::cos(theta);
      torus.positions.push_back(
          {radial * std::cos(phi), centerY + tubeRadius * std::sin(theta),
           radial * std::sin(phi)});
      torus.normals.push_back(
          {std::cos(theta) * std::cos(phi), std::sin(theta),
           std::cos(theta) * std::sin(phi)});
      torus.tangents.push_back({-std::sin(phi), 0.0f, std::cos(phi)});
      torus.uvs.push_back({u, v});
    }
  }

  const int columns = tubeSegments + 1;
  for (int i = 0; i < majorSegments; ++i) {
    for (int j = 0; j < tubeSegments; ++j) {
      const auto a = static_cast<std::uint32_t>(i * columns + j);
      const auto b = static_cast<std::uint32_t>((i + 1) * columns + j);
      const auto c = static_cast<std::uint32_t>((i + 1) * columns + j + 1);
      const auto d = static_cast<std::uint32_t>(i * columns + j + 1);
      torus.indices.insert(torus.indices.end(), {a, d, c, a, c, b});
    }
  }

  torus.baseColor = {0.82f, 0.55f, 0.23f};
  torus.roughness = 0.24f;
  torus.metallic = 0.92f;
  torus.anisotropyStrength = 0.9f;
  torus.anisotropyRotation = 0.35f;
  return torus;
}

void writeMaterial(std::ostream& stream, const MeshData& mesh, bool trailingComma) {
  stream << "      \"material\":{\"base_color\":[" << number(mesh.baseColor.x) << ","
         << number(mesh.baseColor.y) << "," << number(mesh.baseColor.z)
         << "],\"roughness\":" << number(mesh.roughness)
         << ",\"metallic\":" << number(mesh.metallic)
         << ",\"anisotropy_strength\":" << number(mesh.anisotropyStrength)
         << ",\"anisotropy_rotation_rad\":" << number(mesh.anisotropyRotation)
         << "}" << (trailingComma ? "," : "") << "\n";
}

void writeMesh(std::ostream& stream, const MeshData& mesh, bool trailingComma) {
  stream << "    {\n";
  stream << "      \"name\":\"" << mesh.name << "\",\n";
  stream << "      \"model\":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],\n";
  stream << "      \"vertices\":{\n";
  stream << "        \"positions\":";
  writeVec3Array(stream, mesh.positions);
  stream << ",\n        \"normals\":";
  writeVec3Array(stream, mesh.normals);
  stream << ",\n        \"tangents\":";
  writeVec3Array(stream, mesh.tangents);
  stream << ",\n        \"uvs\":";
  writeVec2Array(stream, mesh.uvs);
  stream << "\n      },\n";
  stream << "      \"indices\":";
  writeIndexArray(stream, mesh.indices);
  stream << ",\n";
  writeMaterial(stream, mesh, false);
  stream << "    }" << (trailingComma ? "," : "") << "\n";
}

bool writeScene(const std::filesystem::path& path, bool noLights) {
  const MeshData ground = makeGround();
  const MeshData torus = makeTorus();
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    return false;
  }
  stream.imbue(std::locale::classic());
  stream << std::setprecision(9);
  stream << "{\n";
  stream << "  \"camera\":{\n";
  stream << "    \"position\":[5.2,4.4,7.2],\n";
  stream << "    \"target\":[0.0,0.25,0.0],\n";
  stream << "    \"up\":[0.0,1.0,0.0],\n";
  stream << "    \"projection\":\"perspective\",\n";
  stream << "    \"fov_y_deg\":43.0,\n";
  stream << "    \"near\":0.1,\n";
  stream << "    \"far\":100.0\n";
  stream << "  },\n";
  stream << "  \"meshes\":[\n";
  writeMesh(stream, ground, true);
  writeMesh(stream, torus, false);
  stream << "  ],\n";
  stream << "  \"lights\":{\n";
  if (noLights) {
    stream << "    \"directional\":null,\n";
    stream << "    \"point_lights\":[]\n";
  } else {
    stream << "    \"directional\":{\"type\":\"directional\","
              "\"direction\":[-0.45,-1.0,-0.55],"
              "\"color\":[1.0,0.95,0.82],\"intensity\":3.4},\n";
    stream << "    \"point_lights\":[\n";
    stream << "      {\"type\":\"point\",\"position\":[3.0,2.1,2.1],"
              "\"color\":[1.0,0.32,0.12],\"intensity\":32.0,\"range\":11.0},\n";
    stream << "      {\"type\":\"point\",\"position\":[-3.2,1.6,1.2],"
              "\"color\":[0.12,0.42,1.0],\"intensity\":28.0,\"range\":10.0},\n";
    stream << "      {\"type\":\"point\",\"position\":[0.2,3.1,-2.7],"
              "\"color\":[0.25,1.0,0.48],\"intensity\":24.0,\"range\":9.0}\n";
    stream << "    ]\n";
  }
  stream << "  }\n";
  stream << "}\n";
  stream.flush();
  return static_cast<bool>(stream);
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path output;
  bool noLights = false;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--output" && i + 1 < argc) {
      output = argv[++i];
    } else if (argument == "--variant" && i + 1 < argc) {
      const std::string variant = argv[++i];
      if (variant == "no-lights") {
        noLights = true;
      } else if (variant != "default") {
        std::cerr << "{\"ok\":false,\"code\":\"GEN_ERR_INVALID_VARIANT\"}\n";
        return 2;
      }
    } else {
      std::cerr << "{\"ok\":false,\"code\":\"GEN_ERR_INVALID_ARGUMENT\","
                   "\"argument\":\""
                << argument << "\"}\n";
      return 2;
    }
  }
  if (output.empty()) {
    std::cerr << "{\"ok\":false,\"code\":\"GEN_ERR_MISSING_OUTPUT\"}\n";
    return 2;
  }
  std::error_code ec;
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path(), ec);
  }
  if (ec || !writeScene(output, noLights)) {
    std::cerr << "{\"ok\":false,\"code\":\"GEN_ERR_WRITE\"}\n";
    return 1;
  }
  std::cout << "{\"ok\":true,\"output\":\"" << output.string()
            << "\",\"variant\":\"" << (noLights ? "no-lights" : "default")
            << "\"}\n";
  return 0;
}
