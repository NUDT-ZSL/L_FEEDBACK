#include "baker/render.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "baker/byte_io.h"
#include "baker/interrupt.h"

namespace baker {
namespace {

constexpr double kDegenerateArea = 1.0e-7;
constexpr double kTriangleArea = 1.0e-12;
constexpr double kBarycentricTolerance = 1.0e-9;
constexpr float kAmbient = 0.08f;
constexpr float kDiffuse = 0.92f;

struct Projected {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    Vec2 uv{};
};

float wrap_playback_time(float time, float first, float last, Playback playback) {
    if (playback == Playback::kLoop) {
        const float span = last - first;
        if (span > 0.0f) {
            float wrapped = std::fmod(time - first, span);
            if (wrapped < 0.0f) {
                wrapped += span;
            }
            return first + wrapped;
        }
        return first;
    }

    if (time <= first) {
        return first;
    }
    if (time >= last) {
        return last;
    }
    return time;
}

template <typename Key>
std::size_t upper_key_index(const std::vector<Key>& keys, float time) {
    return static_cast<std::size_t>(
        std::upper_bound(keys.begin(), keys.end(), time,
                         [](float value, const Key& key) { return value < key.time; }) -
        keys.begin());
}

std::array<std::array<float, 2>, 4> evaluate_displacement(const Layer& layer, float time) {
    const auto& keys = layer.vertex_keys;
    if (keys.empty()) {
        return {};
    }
    const float first = keys.front().time;
    const float last = keys.back().time;
    const float played = wrap_playback_time(time, first, last, layer.playback);

    if (keys.size() == 1) {
        return keys.front().displacement;
    }

    const std::size_t upper = upper_key_index(keys, played);
    if (upper == 0) {
        return keys.front().displacement;
    }
    if (upper >= keys.size()) {
        return keys.back().displacement;
    }

    const VKey& left = keys[upper - 1];
    const VKey& right = keys[upper];
    const float span = right.time - left.time;
    const float weight = span > 0.0f ? (played - left.time) / span : 1.0f;

    std::array<std::array<float, 2>, 4> result{};
    for (std::size_t vertex = 0; vertex < 4; ++vertex) {
        for (std::size_t component = 0; component < 2; ++component) {
            result[vertex][component] =
                left.displacement[vertex][component] +
                (right.displacement[vertex][component] -
                 left.displacement[vertex][component]) *
                    weight;
        }
    }
    return result;
}

float evaluate_opacity(const Layer& layer, float time) {
    const auto& keys = layer.opacity_keys;
    if (keys.empty()) {
        return 0.0f;
    }
    const float first = keys.front().time;
    const float last = keys.back().time;
    const float played = wrap_playback_time(time, first, last, layer.playback);

    if (keys.size() == 1) {
        return keys.front().opacity;
    }

    const std::size_t upper = upper_key_index(keys, played);
    if (upper == 0) {
        return keys.front().opacity;
    }
    if (upper >= keys.size()) {
        return keys.back().opacity;
    }

    const OKey& left = keys[upper - 1];
    const OKey& right = keys[upper];
    const float span = right.time - left.time;
    const float weight = span > 0.0f ? (played - left.time) / span : 1.0f;
    return left.opacity + (right.opacity - left.opacity) * weight;
}

Vec3 subtract(Vec3 lhs, Vec3 rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

Vec3 cross(Vec3 lhs, Vec3 rhs) {
    return {lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.x * rhs.y - lhs.y * rhs.x};
}

float dot(Vec3 lhs, Vec3 rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

float length(Vec3 value) {
    return std::sqrt(dot(value, value));
}

Vec3 normalize(Vec3 value) {
    const float len = length(value);
    if (!(len > 1.0e-20f) || !is_finite(len)) {
        return {0.0f, 0.0f, 0.0f};
    }
    const float inverse = 1.0f / len;
    return {value.x * inverse, value.y * inverse, value.z * inverse};
}

Vec3 negate(Vec3 value) {
    return {-value.x, -value.y, -value.z};
}

double edge(double ax, double ay, double bx, double by, double px, double py) {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

double signed_triangle_area(const Projected& a, const Projected& b, const Projected& c) {
    return edge(a.x, a.y, b.x, b.y, c.x, c.y);
}

int positive_mod(int value, int modulus) {
    int result = value % modulus;
    if (result < 0) {
        result += modulus;
    }
    return result;
}

std::array<float, 4> sample_bilinear(const Scene& scene, float u, float v) {
    const int size = static_cast<int>(scene.tex_size);
    const double wrapped_u = static_cast<double>(u) - std::floor(static_cast<double>(u));
    const double wrapped_v = static_cast<double>(v) - std::floor(static_cast<double>(v));

    const double x = wrapped_u * static_cast<double>(size) - 0.5;
    const double y = wrapped_v * static_cast<double>(size) - 0.5;
    const double x_floor = std::floor(x);
    const double y_floor = std::floor(y);
    const float fx = static_cast<float>(x - x_floor);
    const float fy = static_cast<float>(y - y_floor);

    const int x0 = positive_mod(static_cast<int>(x_floor), size);
    const int y0 = positive_mod(static_cast<int>(y_floor), size);
    const int x1 = positive_mod(x0 + 1, size);
    const int y1 = positive_mod(y0 + 1, size);

    const auto texel = [&](int tx, int ty, int channel) -> float {
        const std::size_t index =
            (static_cast<std::size_t>(ty) * static_cast<std::size_t>(size) +
             static_cast<std::size_t>(tx)) *
                4u +
            static_cast<std::size_t>(channel);
        return scene.texels[index];
    };

    std::array<float, 4> result{};
    for (int channel = 0; channel < 4; ++channel) {
        const float top =
            texel(x0, y0, channel) * (1.0f - fx) + texel(x1, y0, channel) * fx;
        const float bottom =
            texel(x0, y1, channel) * (1.0f - fx) + texel(x1, y1, channel) * fx;
        result[static_cast<std::size_t>(channel)] =
            top * (1.0f - fy) + bottom * fy;
    }
    return result;
}

bool project_vertex(const Scene& scene, Vec3 world, Vec2 uv, Projected& output) {
    if (!is_finite(world.x) || !is_finite(world.y) || !is_finite(world.z) ||
        !(world.z > 1.0e-6f)) {
        return false;
    }

    const double focal = static_cast<double>(scene.height) * 0.5;
    output.x = static_cast<double>(scene.width) * 0.5 +
               (static_cast<double>(world.x) / static_cast<double>(world.z)) * focal;
    output.y = static_cast<double>(scene.height) * 0.5 -
               (static_cast<double>(world.y) / static_cast<double>(world.z)) * focal;
    output.z = static_cast<double>(world.z);
    output.uv = uv;
    return std::isfinite(output.x) && std::isfinite(output.y);
}

float clamp_unit(float value) {
    if (!(value > 0.0f)) {
        return 0.0f;
    }
    if (value >= 1.0f) {
        return 1.0f;
    }
    return value;
}

std::uint8_t to_unorm8(float value) {
    if (!(value > 0.0f)) {
        return 0;
    }
    if (value >= 1.0f) {
        return 255;
    }
    return static_cast<std::uint8_t>(value * 255.0f);
}

}  // namespace

PixelBuffer make_empty_buffer(std::uint32_t width, std::uint32_t height) {
    PixelBuffer buffer;
    buffer.width = width;
    buffer.height = height;
    buffer.rgba.assign(static_cast<std::size_t>(width) * height * 4u, 0.0f);
    return buffer;
}

void render_scene(const Scene& scene,
                  const RenderOptions& options,
                  std::uint32_t first_layer,
                  PixelBuffer& state,
                  Stats& stats,
                  const LayerComplete& on_layer_complete) {
    if (state.rgba.size() !=
        static_cast<std::size_t>(scene.width) * scene.height * 4u) {
        state = make_empty_buffer(scene.width, scene.height);
    }
    if (stats.layer_pixels.size() != scene.layers.size()) {
        stats.layer_pixels.resize(scene.layers.size(), 0);
    }

    const std::size_t pixel_count =
        static_cast<std::size_t>(scene.width) * scene.height;
    std::vector<std::uint8_t> covered(pixel_count);

    for (std::uint32_t layer_index = first_layer; layer_index < scene.layers.size();
         ++layer_index) {
        throw_if_interrupted();
        const Layer& layer = scene.layers[layer_index];
        stats.layer_pixels[layer_index] = 0;
        std::fill(covered.begin(), covered.end(), 0);

        const auto displacement = evaluate_displacement(layer, options.time);
        const float opacity = clamp_unit(evaluate_opacity(layer, options.time));

        std::array<Vec3, 4> world{};
        for (std::size_t vertex = 0; vertex < 4; ++vertex) {
            world[vertex] = {
                layer.base[vertex].x + displacement[vertex][0],
                layer.base[vertex].y + displacement[vertex][1],
                layer.base[vertex].z,
            };
        }

        const Vec3 layer_normal = normalize(cross(subtract(world[1], world[0]),
                                                  subtract(world[2], world[0])));
        std::array<Projected, 4> projected{};
        bool projected_ok = true;
        for (std::size_t vertex = 0; vertex < 4; ++vertex) {
            projected_ok =
                project_vertex(scene, world[vertex], layer.uv[vertex], projected[vertex]) &&
                projected_ok;
        }

        double quad_area = 0.0;
        if (projected_ok) {
            for (std::size_t vertex = 0; vertex < 4; ++vertex) {
                const std::size_t next = (vertex + 1) % 4;
                quad_area += projected[vertex].x * projected[next].y -
                             projected[next].x * projected[vertex].y;
            }
            quad_area = std::fabs(quad_area) * 0.5;
        }

        const Vec3 light_direction =
            normalize({0.44721359f, 0.71554178f, -0.53665632f});
        const float roughness = std::clamp(layer.roughness, 0.05f, 1.0f);
        const float shininess =
            std::clamp(2.0f / (roughness * roughness) - 2.0f, 2.0f, 2048.0f);

        if (!projected_ok || !(quad_area > kDegenerateArea) ||
            length(layer_normal) <= 1.0e-12f) {
            ++stats.degenerate_layers;
        } else {
            static constexpr std::array<std::array<int, 3>, 2> triangles{{
                {{0, 1, 2}},
                {{0, 2, 3}},
            }};

            for (const auto& triangle : triangles) {
                const Projected& a = projected[static_cast<std::size_t>(triangle[0])];
                const Projected& b = projected[static_cast<std::size_t>(triangle[1])];
                const Projected& c = projected[static_cast<std::size_t>(triangle[2])];
                const double area = signed_triangle_area(a, b, c);
                if (!(std::fabs(area) > kTriangleArea) || !std::isfinite(area)) {
                    continue;
                }

                const double min_x = std::min({a.x, b.x, c.x});
                const double max_x = std::max({a.x, b.x, c.x});
                const double min_y = std::min({a.y, b.y, c.y});
                const double max_y = std::max({a.y, b.y, c.y});
                if (!std::isfinite(min_x) || !std::isfinite(max_x) ||
                    !std::isfinite(min_y) || !std::isfinite(max_y)) {
                    continue;
                }

                const double width_limit = static_cast<double>(scene.width - 1u);
                const double height_limit = static_cast<double>(scene.height - 1u);
                const int x_begin = static_cast<int>(
                    std::floor(std::clamp(min_x, 0.0, width_limit)));
                const int x_end = static_cast<int>(
                    std::ceil(std::clamp(max_x, 0.0, width_limit)));
                const int y_begin = static_cast<int>(
                    std::floor(std::clamp(min_y, 0.0, height_limit)));
                const int y_end = static_cast<int>(
                    std::ceil(std::clamp(max_y, 0.0, height_limit)));
                if (x_begin > x_end || y_begin > y_end) {
                    continue;
                }

                const double inv_z_a = 1.0 / a.z;
                const double inv_z_b = 1.0 / b.z;
                const double inv_z_c = 1.0 / c.z;
                const Vec3 world_a = world[static_cast<std::size_t>(triangle[0])];
                const Vec3 world_b = world[static_cast<std::size_t>(triangle[1])];
                const Vec3 world_c = world[static_cast<std::size_t>(triangle[2])];
                Vec3 triangle_normal =
                    normalize(cross(subtract(world_b, world_a),
                                    subtract(world_c, world_a)));
                const Vec3 triangle_center{
                    (world_a.x + world_b.x + world_c.x) / 3.0f,
                    (world_a.y + world_b.y + world_c.y) / 3.0f,
                    (world_a.z + world_b.z + world_c.z) / 3.0f,
                };
                if (dot(triangle_normal, normalize(negate(triangle_center))) < 0.0f) {
                    triangle_normal = negate(triangle_normal);
                }

                for (int y = y_begin; y <= y_end; ++y) {
                    if ((y & 15) == 0) {
                        throw_if_interrupted();
                    }
                    const double py = static_cast<double>(y) + 0.5;
                    for (int x = x_begin; x <= x_end; ++x) {
                        const std::size_t pixel_index =
                            static_cast<std::size_t>(y) * scene.width +
                            static_cast<std::size_t>(x);
                        if (covered[pixel_index] != 0) {
                            continue;
                        }
                        const double px = static_cast<double>(x) + 0.5;
                        const double w0 = edge(b.x, b.y, c.x, c.y, px, py) / area;
                        const double w1 = edge(c.x, c.y, a.x, a.y, px, py) / area;
                        const double w2 = edge(a.x, a.y, b.x, b.y, px, py) / area;
                        if (w0 < -kBarycentricTolerance ||
                            w1 < -kBarycentricTolerance ||
                            w2 < -kBarycentricTolerance) {
                            continue;
                        }

                        const double weight_sum =
                            w0 * inv_z_a + w1 * inv_z_b + w2 * inv_z_c;
                        if (!(weight_sum > 1.0e-20) || !std::isfinite(weight_sum)) {
                            continue;
                        }
                        const double u =
                            (w0 * static_cast<double>(a.uv.x) * inv_z_a +
                             w1 * static_cast<double>(b.uv.x) * inv_z_b +
                             w2 * static_cast<double>(c.uv.x) * inv_z_c) /
                            weight_sum;
                        const double v =
                            (w0 * static_cast<double>(a.uv.y) * inv_z_a +
                             w1 * static_cast<double>(b.uv.y) * inv_z_b +
                             w2 * static_cast<double>(c.uv.y) * inv_z_c) /
                            weight_sum;
                        if (!std::isfinite(u) || !std::isfinite(v)) {
                            continue;
                        }

                        covered[pixel_index] = 1;
                        ++stats.layer_pixels[layer_index];

                        if (opacity <= 0.0f) {
                            continue;
                        }

                        const std::array<float, 4> texel =
                            sample_bilinear(scene,
                                            static_cast<float>(u),
                                            static_cast<float>(v));
                        const float alpha = clamp_unit(texel[3] * opacity);
                        if (!(alpha > 0.0f)) {
                            continue;
                        }

                        const Vec3 world_position{
                            static_cast<float>(
                                (w0 * static_cast<double>(world_a.x) * inv_z_a +
                                 w1 * static_cast<double>(world_b.x) * inv_z_b +
                                 w2 * static_cast<double>(world_c.x) * inv_z_c) /
                                weight_sum),
                            static_cast<float>(
                                (w0 * static_cast<double>(world_a.y) * inv_z_a +
                                 w1 * static_cast<double>(world_b.y) * inv_z_b +
                                 w2 * static_cast<double>(world_c.y) * inv_z_c) /
                                weight_sum),
                            static_cast<float>(
                                (w0 * static_cast<double>(world_a.z) * inv_z_a +
                                 w1 * static_cast<double>(world_b.z) * inv_z_b +
                                 w2 * static_cast<double>(world_c.z) * inv_z_c) /
                                weight_sum),
                        };
                        const Vec3 view = normalize(negate(world_position));
                        const Vec3 half_vector = normalize({
                            light_direction.x + view.x,
                            light_direction.y + view.y,
                            light_direction.z + view.z,
                        });
                        const float ndotl =
                            std::max(0.0f, dot(triangle_normal, light_direction));
                        const float ndoth =
                            std::max(0.0f, dot(triangle_normal, half_vector));
                        const float diffuse = kAmbient + kDiffuse * ndotl;
                        const float specular =
                            layer.specular * std::pow(ndoth, shininess);
                        const std::array<float, 3> albedo{
                            layer.albedo.x, layer.albedo.y, layer.albedo.z};
                        std::array<float, 3> lit{};
                        for (std::size_t channel = 0; channel < 3; ++channel) {
                            lit[channel] =
                                albedo[channel] * texel[channel] * diffuse + specular;
                        }
                        if (!is_finite(lit[0]) || !is_finite(lit[1]) ||
                            !is_finite(lit[2])) {
                            continue;
                        }

                        float source_r = lit[0];
                        float source_g = lit[1];
                        float source_b = lit[2];
                        if (!layer.premultiplied) {
                            source_r *= alpha;
                            source_g *= alpha;
                            source_b *= alpha;
                        }

                        const std::size_t base = pixel_index * 4u;
                        const float inverse_alpha = 1.0f - alpha;
                        state.rgba[base + 0] =
                            clamp_unit(source_r + state.rgba[base + 0] * inverse_alpha);
                        state.rgba[base + 1] =
                            clamp_unit(source_g + state.rgba[base + 1] * inverse_alpha);
                        state.rgba[base + 2] =
                            clamp_unit(source_b + state.rgba[base + 2] * inverse_alpha);
                        state.rgba[base + 3] =
                            clamp_unit(alpha + state.rgba[base + 3] * inverse_alpha);
                    }
                }
            }
        }

        if (on_layer_complete) {
            on_layer_complete(layer_index + 1u, state, stats);
        }
    }
}

std::vector<std::uint8_t> encode_output(const PixelBuffer& pixels) {
    const std::uint64_t byte_count =
        static_cast<std::uint64_t>(pixels.width) * pixels.height * 4ull;
    if (byte_count > std::numeric_limits<std::size_t>::max()) {
        throw BakerError(ErrorCode::kValue, "output buffer is too large");
    }

    std::vector<std::uint8_t> output;
    output.reserve(static_cast<std::size_t>(16 + byte_count));
    append_u32(output, kOutputMagic);
    append_u32(output, kOutputVersion);
    append_u32(output, pixels.width);
    append_u32(output, pixels.height);
    for (const float value : pixels.rgba) {
        output.push_back(to_unorm8(value));
    }
    return output;
}

}  // namespace baker
