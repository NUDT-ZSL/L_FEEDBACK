#include "scene.hpp"

#include "errors.hpp"
#include "hash.hpp"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace vlr {
namespace {

[[noreturn]] void schemaError(const std::string& message) {
    fail(ErrorCode::Schema, "schema: " + message);
}

const json::Value& requireMember(const json::Value& object, const std::string& key,
                                 const std::string& context) {
    if (!object.isObject()) {
        schemaError(context + " must be an object");
    }
    const json::Value* value = object.find(key);
    if (value == nullptr) {
        schemaError(context + "." + key + " is required");
    }
    return *value;
}

double requireNumber(const json::Value& object, const std::string& key,
                     const std::string& context) {
    const json::Value& value = requireMember(object, key, context);
    if (!value.isNumber() || !std::isfinite(value.asNumber())) {
        schemaError(context + "." + key + " must be a finite number");
    }
    return value.asNumber();
}

int requireInteger(const json::Value& object, const std::string& key,
                   const std::string& context) {
    const double value = requireNumber(object, key, context);
    if (std::floor(value) != value ||
        value < static_cast<double>(std::numeric_limits<int>::min()) ||
        value > static_cast<double>(std::numeric_limits<int>::max())) {
        schemaError(context + "." + key + " must be an integer");
    }
    return static_cast<int>(value);
}

std::string requireString(const json::Value& object, const std::string& key,
                          const std::string& context) {
    const json::Value& value = requireMember(object, key, context);
    if (!value.isString() || value.asString().empty()) {
        schemaError(context + "." + key + " must be a non-empty string");
    }
    return value.asString();
}

std::string optionalString(const json::Value& object, const std::string& key,
                           const std::string& context, const std::string& fallback) {
    const json::Value* value = object.find(key);
    if (value == nullptr) {
        return fallback;
    }
    if (!value->isString() || value->asString().empty()) {
        schemaError(context + "." + key + " must be a non-empty string");
    }
    return value->asString();
}

Vec3 requireVec3(const json::Value& object, const std::string& key,
                 const std::string& context) {
    const json::Value& value = requireMember(object, key, context);
    if (!value.isArray() || value.asArray().size() != 3) {
        schemaError(context + "." + key + " must be an array of 3 numbers");
    }
    Vec3 result{};
    for (std::size_t i = 0; i < 3; ++i) {
        const auto& item = value.asArray()[i];
        if (!item.isNumber() || !std::isfinite(item.asNumber())) {
            schemaError(context + "." + key + " must contain finite numbers");
        }
        result[i] = static_cast<float>(item.asNumber());
    }
    return result;
}

void requireRange(double value, double minimum, double maximum,
                  const std::string& label) {
    const double epsilon = 1.0e-9;
    if (value < minimum - epsilon || value > maximum + epsilon) {
        std::ostringstream message;
        message << label << " must be in [" << minimum << ", " << maximum
                << "], got " << value;
        schemaError(message.str());
    }
}

void requireFinite(Vec3 value, const std::string& label) {
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z)) {
        schemaError(label + " must contain finite values");
    }
}

std::uint32_t mixBits(std::uint32_t value) {
    value += 0x9e3779b9u;
    value = (value ^ (value >> 16u)) * 0x21f0aaadu;
    value = (value ^ (value >> 15u)) * 0x735a2d97u;
    return value ^ (value >> 15u);
}

std::uint32_t latticeHash(int x, int y, int z, int period, std::uint32_t seed) {
    auto wrap = [period](int value) {
        const int result = value % period;
        return result < 0 ? result + period : result;
    };
    const auto ux = static_cast<std::uint32_t>(wrap(x));
    const auto uy = static_cast<std::uint32_t>(wrap(y));
    const auto uz = static_cast<std::uint32_t>(wrap(z));
    std::uint32_t hash = seed ^ (ux * 0x8da6b343u) ^ (uy * 0xd8163841u) ^
                         (uz * 0xcb1ab31fu);
    hash = mixBits(hash);
    return hash;
}

float latticeValue(int x, int y, int z, int period, std::uint32_t seed) {
    return static_cast<float>(latticeHash(x, y, z, period, seed) & 0x00ffffffu) /
           static_cast<float>(0x01000000u);
}

float smooth(float t) {
    return t * t * (3.0f - 2.0f * t);
}

float valueNoise(float x, float y, float z, int period, std::uint32_t seed) {
    const int ix = static_cast<int>(std::floor(x));
    const int iy = static_cast<int>(std::floor(y));
    const int iz = static_cast<int>(std::floor(z));
    const float fx = smooth(x - static_cast<float>(ix));
    const float fy = smooth(y - static_cast<float>(iy));
    const float fz = smooth(z - static_cast<float>(iz));

    const float c000 = latticeValue(ix, iy, iz, period, seed);
    const float c100 = latticeValue(ix + 1, iy, iz, period, seed);
    const float c010 = latticeValue(ix, iy + 1, iz, period, seed);
    const float c110 = latticeValue(ix + 1, iy + 1, iz, period, seed);
    const float c001 = latticeValue(ix, iy, iz + 1, period, seed);
    const float c101 = latticeValue(ix + 1, iy, iz + 1, period, seed);
    const float c011 = latticeValue(ix, iy + 1, iz + 1, period, seed);
    const float c111 = latticeValue(ix + 1, iy + 1, iz + 1, period, seed);

    const float x00 = c000 + (c100 - c000) * fx;
    const float x10 = c010 + (c110 - c010) * fx;
    const float x01 = c001 + (c101 - c001) * fx;
    const float x11 = c011 + (c111 - c011) * fx;
    const float y0 = x00 + (x10 - x00) * fy;
    const float y1 = x01 + (x11 - x01) * fy;
    return y0 + (y1 - y0) * fz;
}

} // namespace

std::filesystem::path resolveRelative(const std::filesystem::path& base,
                                      const std::filesystem::path& candidate) {
    if (candidate.is_absolute()) {
        return candidate;
    }
    return (base / candidate).lexically_normal();
}

std::vector<std::uint8_t> readFileBytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail(ErrorCode::Io, "cannot open file: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) {
        fail(ErrorCode::Io, "cannot determine file size: " + path.string());
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), size);
    }
    if (!input) {
        fail(ErrorCode::Io, "cannot read file: " + path.string());
    }
    return bytes;
}

Scene loadScene(const std::filesystem::path& path) {
    const std::vector<std::uint8_t> bytes = readFileBytes(path);
    const std::string text(reinterpret_cast<const char*>(bytes.data()), bytes.size());

    json::Value root;
    try {
        root = json::parse(text);
    } catch (const std::exception& exception) {
        schemaError(exception.what());
    }
    if (!root.isObject()) {
        schemaError("root must be an object");
    }
    if (root.find("schemaVersion") == nullptr) {
        schemaError("schemaVersion is required");
    }
    const int version = requireInteger(root, "schemaVersion", "root");
    if (version != 1) {
        schemaError("unsupported schemaVersion " + std::to_string(version) +
                    " (expected 1)");
    }

    Scene scene;
    scene.schemaVersion = version;
    scene.name = optionalString(root, "name", "root", "unnamed");
    scene.sourcePath = std::filesystem::absolute(path).lexically_normal();
    scene.sourceDirectory = scene.sourcePath.parent_path();
    scene.sourceSha256 = sha256Hex(bytes);

    const json::Value& camera = requireMember(root, "camera", "root");
    scene.camera.position = requireVec3(camera, "position", "camera");
    scene.camera.target = requireVec3(camera, "target", "camera");
    scene.camera.up = requireVec3(camera, "up", "camera");
    scene.camera.fovYDegrees =
        static_cast<float>(requireNumber(camera, "fovYDegrees", "camera"));
    scene.camera.nearMeters =
        static_cast<float>(requireNumber(camera, "nearMeters", "camera"));
    scene.camera.farMeters =
        static_cast<float>(requireNumber(camera, "farMeters", "camera"));
    requireFinite(scene.camera.position, "camera.position");
    requireFinite(scene.camera.target, "camera.target");
    requireFinite(scene.camera.up, "camera.up");
    requireRange(scene.camera.fovYDegrees, 10.0, 120.0, "camera.fovYDegrees");
    requireRange(scene.camera.nearMeters, 0.01, 10.0, "camera.nearMeters");
    requireRange(scene.camera.farMeters, 1.0, 1000.0, "camera.farMeters");
    if (scene.camera.farMeters <= scene.camera.nearMeters) {
        schemaError("camera.farMeters must be greater than camera.nearMeters");
    }
    if (lengthSquared(scene.camera.target - scene.camera.position) < 1.0e-8f) {
        schemaError("camera.target must differ from camera.position");
    }
    if (lengthSquared(cross(normalize(scene.camera.target - scene.camera.position),
                            scene.camera.up)) < 1.0e-8f) {
        schemaError("camera.up must not be parallel to the view direction");
    }

    const json::Value& output = requireMember(root, "output", "root");
    scene.output.width = requireInteger(output, "width", "output");
    scene.output.height = requireInteger(output, "height", "output");
    scene.output.exposure =
        static_cast<float>(requireNumber(output, "exposure", "output"));
    scene.output.gain = static_cast<float>(requireNumber(output, "gain", "output"));
    requireRange(scene.output.width, 64.0, 3840.0, "output.width");
    requireRange(scene.output.height, 64.0, 2160.0, "output.height");
    if (static_cast<std::int64_t>(scene.output.width) * scene.output.height > 8294400) {
        schemaError("output pixel count must not exceed 8,294,400");
    }
    requireRange(scene.output.exposure, 0.01, 32.0, "output.exposure");
    requireRange(scene.output.gain, 0.0, 8.0, "output.gain");
    scene.output.pngPath = requireString(output, "pngPath", "output");
    scene.output.statsPath =
        optionalString(output, "statsPath", "output", "stats.json");
    scene.output.referencePngPath =
        optionalString(output, "referencePngPath", "output", "render.reference.png");
    if (const json::Value* background = output.find("background")) {
        if (!background->isArray() || background->asArray().size() != 3) {
            schemaError("output.background must be an array of 3 numbers");
        }
        for (int i = 0; i < 3; ++i) {
            const auto& item = background->asArray()[static_cast<std::size_t>(i)];
            if (!item.isNumber()) {
                schemaError("output.background must contain numbers");
            }
            scene.output.background[static_cast<std::size_t>(i)] =
                static_cast<float>(item.asNumber());
            requireRange(scene.output.background[static_cast<std::size_t>(i)],
                         0.0, 1.0, "output.background component");
        }
    }

    const json::Value& noise = requireMember(root, "noise", "root");
    const std::string noisePath = requireString(noise, "path", "noise");
    scene.noisePath = resolveRelative(scene.sourceDirectory, noisePath);
    scene.noiseDimension = requireInteger(noise, "dimension", "noise");
    if (scene.noiseDimension != 32) {
        schemaError("noise.dimension must be exactly 32");
    }
    const int seed = requireInteger(noise, "seed", "noise");
    if (seed < 0) {
        schemaError("noise.seed must be a non-negative 32-bit integer");
    }
    scene.noiseSeed = static_cast<std::uint32_t>(seed);

    const json::Value& volume = requireMember(root, "volume", "root");
    scene.volume.boundsMin = requireVec3(volume, "boundsMin", "volume");
    scene.volume.boundsMax = requireVec3(volume, "boundsMax", "volume");
    scene.volume.density =
        static_cast<float>(requireNumber(volume, "density", "volume"));
    scene.volume.absorption =
        static_cast<float>(requireNumber(volume, "absorption", "volume"));
    scene.volume.scattering =
        static_cast<float>(requireNumber(volume, "scattering", "volume"));
    scene.volume.anisotropy =
        static_cast<float>(requireNumber(volume, "anisotropy", "volume"));
    scene.volume.stepMeters =
        static_cast<float>(requireNumber(volume, "stepMeters", "volume"));
    scene.volume.maxSteps = requireInteger(volume, "maxSteps", "volume");
    scene.volume.noiseFrequencyPerMeter = static_cast<float>(
        requireNumber(volume, "noiseFrequencyPerMeter", "volume"));
    requireFinite(scene.volume.boundsMin, "volume.boundsMin");
    requireFinite(scene.volume.boundsMax, "volume.boundsMax");
    for (int i = 0; i < 3; ++i) {
        if (scene.volume.boundsMin[static_cast<std::size_t>(i)] >=
            scene.volume.boundsMax[static_cast<std::size_t>(i)]) {
            schemaError("volume.boundsMin must be strictly less than volume.boundsMax");
        }
    }
    requireRange(scene.volume.density, 0.0001, 2.0, "volume.density");
    requireRange(scene.volume.absorption, 0.0, 20.0, "volume.absorption");
    requireRange(scene.volume.scattering, 0.0, 20.0, "volume.scattering");
    requireRange(scene.volume.anisotropy, -0.95, 0.95, "volume.anisotropy");
    requireRange(scene.volume.stepMeters, 0.01, 1.0, "volume.stepMeters");
    requireRange(scene.volume.maxSteps, 8.0, 2048.0, "volume.maxSteps");
    requireRange(scene.volume.noiseFrequencyPerMeter, 0.001, 4.0,
                 "volume.noiseFrequencyPerMeter");

    const json::Value& directional =
        requireMember(root, "directionalLight", "root");
    scene.directionalLight.direction =
        normalize(requireVec3(directional, "direction", "directionalLight"));
    scene.directionalLight.color =
        requireVec3(directional, "color", "directionalLight");
    scene.directionalLight.intensity = static_cast<float>(
        requireNumber(directional, "intensity", "directionalLight"));
    if (lengthSquared(scene.directionalLight.direction) < 1.0e-8f) {
        schemaError("directionalLight.direction must be non-zero");
    }
    for (int i = 0; i < 3; ++i) {
        requireRange(scene.directionalLight.color[static_cast<std::size_t>(i)],
                     0.0, 1.0, "directionalLight.color component");
    }
    requireRange(scene.directionalLight.intensity, 0.0, 100.0,
                 "directionalLight.intensity");

    const json::Value& point = requireMember(root, "pointLight", "root");
    scene.pointLight.position = requireVec3(point, "position", "pointLight");
    scene.pointLight.color = requireVec3(point, "color", "pointLight");
    scene.pointLight.intensity =
        static_cast<float>(requireNumber(point, "intensity", "pointLight"));
    scene.pointLight.rangeMeters =
        static_cast<float>(requireNumber(point, "rangeMeters", "pointLight"));
    requireFinite(scene.pointLight.position, "pointLight.position");
    for (int i = 0; i < 3; ++i) {
        requireRange(scene.pointLight.color[static_cast<std::size_t>(i)],
                     0.0, 1.0, "pointLight.color component");
    }
    requireRange(scene.pointLight.intensity, 0.0, 1000.0, "pointLight.intensity");
    requireRange(scene.pointLight.rangeMeters, 0.1, 100.0,
                 "pointLight.rangeMeters");

    const json::Value& shadows = requireMember(root, "shadows", "root");
    scene.shadows.resolution =
        requireInteger(shadows, "resolution", "shadows");
    scene.shadows.pcfKernel =
        requireInteger(shadows, "pcfKernel", "shadows");
    scene.shadows.directionalBias = static_cast<float>(
        requireNumber(shadows, "directionalBias", "shadows"));
    scene.shadows.pointBias = static_cast<float>(
        requireNumber(shadows, "pointBias", "shadows"));
    requireRange(scene.shadows.resolution, 128.0, 4096.0, "shadows.resolution");
    requireRange(scene.shadows.pcfKernel, 1.0, 9.0, "shadows.pcfKernel");
    if (scene.shadows.pcfKernel % 2 == 0) {
        schemaError("shadows.pcfKernel must be odd");
    }
    requireRange(scene.shadows.directionalBias, 0.0, 0.05,
                 "shadows.directionalBias");
    requireRange(scene.shadows.pointBias, 0.0, 0.05, "shadows.pointBias");

    const json::Value& geometry = requireMember(root, "geometry", "root");
    const json::Value& ground = requireMember(geometry, "ground", "geometry");
    scene.ground.height =
        static_cast<float>(requireNumber(ground, "height", "geometry.ground"));
    scene.ground.sizeMeters =
        static_cast<float>(requireNumber(ground, "sizeMeters", "geometry.ground"));
    scene.ground.albedo =
        requireVec3(ground, "albedo", "geometry.ground");
    requireRange(scene.ground.sizeMeters, 1.0, 100.0, "geometry.ground.sizeMeters");
    for (int i = 0; i < 3; ++i) {
        requireRange(scene.ground.albedo[static_cast<std::size_t>(i)],
                     0.0, 1.0, "geometry.ground.albedo component");
    }

    const json::Value& spheres = requireMember(geometry, "spheres", "geometry");
    if (!spheres.isArray() || spheres.asArray().size() > 16) {
        schemaError("geometry.spheres must be an array of at most 16 spheres");
    }
    for (std::size_t i = 0; i < spheres.asArray().size(); ++i) {
        const auto& item = spheres.asArray()[i];
        const std::string context = "geometry.spheres[" + std::to_string(i) + "]";
        Sphere sphere;
        sphere.center = requireVec3(item, "center", context);
        sphere.radius =
            static_cast<float>(requireNumber(item, "radius", context));
        sphere.albedo = requireVec3(item, "albedo", context);
        requireFinite(sphere.center, context + ".center");
        requireRange(sphere.radius, 0.01, 20.0, context + ".radius");
        for (int channel = 0; channel < 3; ++channel) {
            requireRange(sphere.albedo[static_cast<std::size_t>(channel)],
                         0.0, 1.0, context + ".albedo component");
        }
        scene.spheres.push_back(sphere);
    }

    return scene;
}

NoiseVolume loadNoise(const std::filesystem::path& path, int expectedDimension) {
    if (expectedDimension != 32) {
        schemaError("noise dimension must be 32");
    }
    const std::vector<std::uint8_t> bytes = readFileBytes(path);
    const std::size_t expected =
        static_cast<std::size_t>(expectedDimension) * expectedDimension *
        expectedDimension * sizeof(float);
    if (bytes.size() != expected) {
        schemaError("noise .raw size is " + std::to_string(bytes.size()) +
                    " bytes; expected exactly " + std::to_string(expected));
    }
    NoiseVolume volume;
    volume.dimension = expectedDimension;
    volume.values.resize(static_cast<std::size_t>(expectedDimension) * expectedDimension *
                         expectedDimension);
    for (std::size_t i = 0; i < volume.values.size(); ++i) {
        std::uint32_t bits = 0;
        bits |= static_cast<std::uint32_t>(bytes[i * 4]) << 0u;
        bits |= static_cast<std::uint32_t>(bytes[i * 4 + 1]) << 8u;
        bits |= static_cast<std::uint32_t>(bytes[i * 4 + 2]) << 16u;
        bits |= static_cast<std::uint32_t>(bytes[i * 4 + 3]) << 24u;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(float));
        if (!std::isfinite(value) || value < -0.0001f || value > 1.0001f) {
            schemaError("noise sample " + std::to_string(i) +
                        " is outside the required finite [0,1] range");
        }
        volume.values[i] = std::clamp(value, 0.0f, 1.0f);
    }
    return volume;
}

NoiseVolume generateNoise(int dimension, std::uint32_t seed) {
    if (dimension != 32) {
        schemaError("deterministic noise generator only supports dimension 32");
    }
    NoiseVolume volume;
    volume.dimension = dimension;
    volume.values.resize(static_cast<std::size_t>(dimension) * dimension * dimension);
    for (int z = 0; z < dimension; ++z) {
        for (int y = 0; y < dimension; ++y) {
            for (int x = 0; x < dimension; ++x) {
                const float px = (static_cast<float>(x) + 0.5f) /
                                 static_cast<float>(dimension);
                const float py = (static_cast<float>(y) + 0.5f) /
                                 static_cast<float>(dimension);
                const float pz = (static_cast<float>(z) + 0.5f) /
                                 static_cast<float>(dimension);
                float value = 0.0f;
                float amplitude = 1.0f;
                float total = 0.0f;
                constexpr std::array<int, 4> periods = {4, 8, 16, 32};
                for (std::size_t octave = 0; octave < periods.size(); ++octave) {
                    const int period = periods[octave];
                    value += amplitude *
                             valueNoise(px * static_cast<float>(period),
                                        py * static_cast<float>(period),
                                        pz * static_cast<float>(period),
                                        period,
                                        seed + static_cast<std::uint32_t>(octave) * 0x9e3779b9u);
                    total += amplitude;
                    amplitude *= 0.5f;
                }
                value /= total;
                const std::size_t index =
                    static_cast<std::size_t>(x) +
                    static_cast<std::size_t>(y) * dimension +
                    static_cast<std::size_t>(z) * dimension * dimension;
                volume.values[index] = std::clamp(value, 0.0f, 1.0f);
            }
        }
    }
    return volume;
}

std::vector<std::uint8_t> generateNoiseRaw(int dimension, std::uint32_t seed) {
    const NoiseVolume volume = generateNoise(dimension, seed);
    std::vector<std::uint8_t> bytes(volume.values.size() * sizeof(float));
    for (std::size_t i = 0; i < volume.values.size(); ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &volume.values[i], sizeof(float));
        bytes[i * 4] = static_cast<std::uint8_t>((bits >> 0u) & 0xffu);
        bytes[i * 4 + 1] = static_cast<std::uint8_t>((bits >> 8u) & 0xffu);
        bytes[i * 4 + 2] = static_cast<std::uint8_t>((bits >> 16u) & 0xffu);
        bytes[i * 4 + 3] = static_cast<std::uint8_t>((bits >> 24u) & 0xffu);
    }
    return bytes;
}

} // namespace vlr
