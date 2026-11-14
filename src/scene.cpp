#include "baker/scene.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>

#include "baker/byte_io.h"
#include "baker/interrupt.h"

namespace baker {
namespace {

[[noreturn]] void invalid(const std::string& what) {
    throw BakerError(ErrorCode::kValue, "illegal " + what);
}

void require_finite(float value, const char* what) {
    if (!is_finite(value)) {
        invalid(std::string(what) + " (non-finite)");
    }
}

void require_unit(float value, const char* what) {
    require_finite(value, what);
    if (value < 0.0f || value > 1.0f) {
        invalid(std::string(what) + " (outside [0,1])");
    }
}

void require_bounded(float value, float limit, const char* what) {
    require_finite(value, what);
    if (value < -limit || value > limit) {
        invalid(std::string(what) + " (outside numeric range)");
    }
}

void require_key_times(const std::vector<VKey>& keys) {
    float previous = -1.0f;
    for (const auto& key : keys) {
        require_unit(key.time, "vertex key time");
        if (key.time < previous) {
            invalid("vertex key order");
        }
        previous = key.time;
    }
}

void require_key_times(const std::vector<OKey>& keys) {
    float previous = -1.0f;
    for (const auto& key : keys) {
        require_unit(key.time, "opacity key time");
        if (key.time < previous) {
            invalid("opacity key order");
        }
        previous = key.time;
    }
}

}  // namespace

std::uint64_t fnv1a64(std::span<const std::uint8_t> bytes) noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    for (const std::uint8_t byte : bytes) {
        hash ^= static_cast<std::uint64_t>(byte);
        hash *= 1099511628211ull;
    }
    return hash;
}

bool is_finite(float value) noexcept {
    return std::isfinite(value);
}

bool nearly_equal(float lhs, float rhs) noexcept {
    return std::fabs(lhs - rhs) <= 1.0e-6f;
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path,
                                    std::uint64_t maximum_size,
                                    const char* description) {
    std::error_code size_error;
    const std::uintmax_t file_size = std::filesystem::file_size(path, size_error);
    if (size_error) {
        throw BakerError(ErrorCode::kIo,
                         std::string("cannot read ") + description + " '" + path.string() +
                             "': " + size_error.message());
    }
    if (file_size > maximum_size ||
        file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw BakerError(ErrorCode::kLength,
                         std::string(description) + " exceeds the supported size");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw BakerError(ErrorCode::kIo,
                         std::string("cannot open ") + description + " '" + path.string() + "'");
    }

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
    }
    if (!input && !bytes.empty()) {
        throw BakerError(ErrorCode::kIo,
                         std::string("cannot read ") + description + " '" + path.string() + "'");
    }
    throw_if_interrupted();
    return bytes;
}

Scene parse_scene(std::span<const std::uint8_t> bytes) {
    ByteReader reader(bytes);

    if (reader.remaining() < 4) {
        throw BakerError(ErrorCode::kLength, "truncated scene header");
    }
    const std::uint32_t magic = reader.u32("scene magic");
    if (magic != kSceneMagic) {
        throw BakerError(ErrorCode::kMagic, "bad magic number (want QBSC)");
    }
    if (reader.remaining() < 4) {
        throw BakerError(ErrorCode::kLength, "truncated scene version");
    }
    const std::uint32_t version = reader.u32("scene version");
    if (version != kSceneVersion) {
        throw BakerError(ErrorCode::kVersion,
                         "unsupported scene version " + std::to_string(version));
    }

    Scene scene;
    scene.width = reader.u32("scene width");
    scene.height = reader.u32("scene height");
    const std::uint32_t layer_count = reader.u32("scene layer count");
    scene.tex_size = reader.u32("scene texture size");

    if (scene.width == 0 || scene.width > kMaxDimension ||
        scene.height == 0 || scene.height > kMaxDimension) {
        invalid("resolution");
    }
    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(scene.width) * static_cast<std::uint64_t>(scene.height);
    if (pixel_count > kMaxPixels) {
        invalid("resolution (pixel count too large)");
    }
    if (layer_count == 0 || layer_count > kMaxLayers) {
        invalid("layer count");
    }
    if (scene.tex_size == 0 || scene.tex_size > kMaxTextureSize) {
        invalid("texture size");
    }

    const std::uint64_t texel_count =
        static_cast<std::uint64_t>(scene.tex_size) * scene.tex_size * 4ull;
    if (texel_count > std::numeric_limits<std::size_t>::max()) {
        invalid("texture size");
    }
    scene.texels.reserve(static_cast<std::size_t>(texel_count));
    for (std::uint64_t index = 0; index < texel_count; ++index) {
        const float value = reader.f32("texture data");
        require_unit(value, "texture component");
        scene.texels.push_back(value);
    }

    scene.layers.reserve(layer_count);
    for (std::uint32_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        Layer layer;

        const std::uint32_t vertex_key_count = reader.u32("vertex key count");
        if (vertex_key_count == 0 || vertex_key_count > kMaxKeys) {
            invalid("vertex key count in layer " + std::to_string(layer_index));
        }
        layer.vertex_keys.resize(vertex_key_count);
        for (VKey& key : layer.vertex_keys) {
            key.time = reader.f32("vertex key time");
            for (auto& vertex : key.displacement) {
                for (float& component : vertex) {
                    component = reader.f32("vertex displacement");
                    require_bounded(component, 1000000.0f, "vertex displacement");
                }
            }
        }
        require_key_times(layer.vertex_keys);

        const std::uint32_t opacity_key_count = reader.u32("opacity key count");
        if (opacity_key_count == 0 || opacity_key_count > kMaxKeys) {
            invalid("opacity key count in layer " + std::to_string(layer_index));
        }
        layer.opacity_keys.resize(opacity_key_count);
        for (OKey& key : layer.opacity_keys) {
            key.time = reader.f32("opacity key time");
            key.opacity = reader.f32("opacity key value");
            require_unit(key.opacity, "opacity key value");
        }
        require_key_times(layer.opacity_keys);

        const std::uint32_t playback = reader.u32("playback mode");
        if (playback > 1) {
            invalid("playback mode in layer " + std::to_string(layer_index));
        }
        layer.playback = static_cast<Playback>(playback);

        for (Vec3& vertex : layer.base) {
            vertex.x = reader.f32("base vertex");
            vertex.y = reader.f32("base vertex");
            vertex.z = reader.f32("base vertex");
            require_bounded(vertex.x, 1000000.0f, "base vertex x");
            require_bounded(vertex.y, 1000000.0f, "base vertex y");
            require_bounded(vertex.z, 1000000.0f, "base vertex z");
        }

        for (Vec2& uv : layer.uv) {
            uv.x = reader.f32("texture coordinate");
            uv.y = reader.f32("texture coordinate");
            require_bounded(uv.x, 64.0f, "texture coordinate u");
            require_bounded(uv.y, 64.0f, "texture coordinate v");
        }

        layer.albedo.x = reader.f32("albedo");
        layer.albedo.y = reader.f32("albedo");
        layer.albedo.z = reader.f32("albedo");
        require_unit(layer.albedo.x, "albedo red");
        require_unit(layer.albedo.y, "albedo green");
        require_unit(layer.albedo.z, "albedo blue");

        layer.roughness = reader.f32("roughness");
        layer.specular = reader.f32("specular");
        require_unit(layer.roughness, "roughness");
        require_unit(layer.specular, "specular");

        const std::uint32_t premultiplied = reader.u32("premultiplied flag");
        if (premultiplied > 1) {
            invalid("premultiplied flag in layer " + std::to_string(layer_index));
        }
        layer.premultiplied = premultiplied != 0;

        scene.layers.push_back(std::move(layer));
    }

    if (!reader.at_end()) {
        throw BakerError(ErrorCode::kLength,
                         "trailing bytes after last layer at byte " +
                             std::to_string(reader.offset()));
    }
    return scene;
}

Scene load_scene(const std::filesystem::path& path) {
    const std::vector<std::uint8_t> bytes = read_file(path, kMaxSceneBytes, "scene");
    Scene scene = parse_scene(bytes);
    scene.input_hash = fnv1a64(bytes);
    return scene;
}

}  // namespace baker
