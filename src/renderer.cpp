#include "renderer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ct {
namespace {

struct ClipVertex {
    Vec4 clip;
    Vec3 world;
    Vec3 normal;
    Vec2 uv;
    std::uint32_t material = 0;
};

struct ScreenVertex {
    double x = 0.0;
    double y = 0.0;
    double inv_w = 0.0;
    const ClipVertex* source = nullptr;
};

struct GBufferSample {
    float px = 0.0f;
    float py = 0.0f;
    float pz = 0.0f;
    float nx = 0.0f;
    float ny = 1.0f;
    float nz = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float dudx = 0.0f;
    float dvdx = 0.0f;
    float dudy = 0.0f;
    float dvdy = 0.0f;
    float inv_depth = 0.0f;
    std::uint32_t material = std::numeric_limits<std::uint32_t>::max();

    bool valid() const {
        return material != std::numeric_limits<std::uint32_t>::max() && inv_depth > 0.0f;
    }
};

struct AnisoSampleResult {
    Vec3 color;
    int samples = 0;
};

class ProceduralTexture {
public:
    explicit ProceduralTexture(int size) : size_(size), texels_(
        static_cast<std::size_t>(size) * static_cast<std::size_t>(size)) {
        for (int y = 0; y < size_; ++y) {
            for (int x = 0; x < size_; ++x) {
                const double fx = static_cast<double>(x) / static_cast<double>(size_);
                const double fy = static_cast<double>(y) / static_cast<double>(size_);
                const double broad =
                    0.5 + 0.5 * std::sin(2.0 * kPi * (fx * 5.0 + fy * 3.0) + 0.7) *
                                    std::cos(2.0 * kPi * (fx * 3.0 - fy * 7.0));
                const double medium =
                    0.5 + 0.5 * std::sin(2.0 * kPi * (fx * 17.0 - fy * 11.0) + 1.3);
                const double fine =
                    0.5 + 0.5 * std::cos(2.0 * kPi * (fx * 47.0 + fy * 29.0) + 2.1);
                const std::uint32_t hash =
                    static_cast<std::uint32_t>(x * 374761393u + y * 668265263u);
                const double speckle =
                    static_cast<double>((hash ^ (hash >> 13)) & 255u) / 255.0;
                const double value =
                    std::clamp(0.56 + 0.17 * broad + 0.12 * medium + 0.09 * fine +
                                   0.06 * speckle,
                               0.35, 1.0);
                texels_[static_cast<std::size_t>(y) * size_ + x] =
                    Vec3{value * 1.02, value, value * 0.94};
            }
        }
    }

    Vec3 sampleBilinear(double u, double v) const {
        const double scaled_u = u * static_cast<double>(size_);
        const double scaled_v = v * static_cast<double>(size_);
        const double floor_u = std::floor(scaled_u);
        const double floor_v = std::floor(scaled_v);
        const double tx = scaled_u - floor_u;
        const double ty = scaled_v - floor_v;
        const int x0 = wrap(static_cast<int>(floor_u));
        const int y0 = wrap(static_cast<int>(floor_v));
        const int x1 = wrap(x0 + 1);
        const int y1 = wrap(y0 + 1);
        const Vec3 a = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
        const Vec3 b = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
        return a * (1.0 - ty) + b * ty;
    }

private:
    int wrap(int value) const {
        value %= size_;
        return value < 0 ? value + size_ : value;
    }

    const Vec3& at(int x, int y) const {
        return texels_[static_cast<std::size_t>(y) * size_ + x];
    }

    int size_;
    std::vector<Vec3> texels_;
};

Vec3 materialTint(std::uint32_t material) {
    switch (material) {
        case 0:
            return {0.70, 0.60, 0.38};
        case 2:
            return {0.48, 0.43, 0.35};
        case 3:
            return {0.86, 0.91, 0.96};
        case 1:
        default:
            return {0.34, 0.55, 0.25};
    }
}

class Renderer {
public:
    Renderer(const HeightField& heightfield, const TerrainMesh& mesh, const AppConfig& config,
             ColorBuffer& output, RenderStats& stats)
        : heightfield_(heightfield),
          mesh_(mesh),
          config_(config),
          output_(output),
          stats_(stats),
          texture_(config.texture.size),
          internal_width_(config.render.width * config.render.supersampling),
          internal_height_(config.render.height * config.render.supersampling),
          gbuffer_(static_cast<std::size_t>(internal_width_) *
                   static_cast<std::size_t>(internal_height_)),
          view_(lookAt(config.camera.position, config.camera.target, config.camera.up)),
          projection_(perspective(config.camera.fov_degrees * kPi / 180.0,
                                  static_cast<double>(internal_width_) /
                                      static_cast<double>(internal_height_),
                                  config.camera.near_plane, config.camera.far_plane)),
          view_projection_(projection_ * view_) {
        stats_.width = config.render.width;
        stats_.height = config.render.height;
        stats_.internal_width = internal_width_;
        stats_.internal_height = internal_height_;
        stats_.supersampling = config.render.supersampling;
        stats_.gbuffer_bytes_per_pixel = sizeof(GBufferSample);
        stats_.gbuffer_pixel_count = gbuffer_.size();
        stats_.gbuffer_bytes = gbuffer_.size() * sizeof(GBufferSample);
        for (int y = 0; y < output_.height; ++y) {
            for (int x = 0; x < output_.width; ++x) {
                output_.at(x, y) = {};
            }
        }
    }

    void render() {
        stats_.triangles_total = mesh_.triangles.size();
        for (const TerrainTriangle& triangle : mesh_.triangles) {
            processTriangle(triangle);
        }
        shadeAndResolve();
        if (stats_.covered_samples > 0) {
            stats_.average_anisotropic_samples =
                static_cast<double>(stats_.anisotropic_samples) /
                static_cast<double>(stats_.covered_samples);
        }
    }

private:
    static Vec4 clipPlane(int plane, const Vec4& p) {
        switch (plane) {
            case 0:
                return {p.x + p.w, 0.0, 0.0, 0.0};
            case 1:
                return {p.w - p.x, 0.0, 0.0, 0.0};
            case 2:
                return {p.y + p.w, 0.0, 0.0, 0.0};
            case 3:
                return {p.w - p.y, 0.0, 0.0, 0.0};
            case 4:
                return {p.z + p.w, 0.0, 0.0, 0.0};
            default:
                return {p.w - p.z, 0.0, 0.0, 0.0};
        }
    }

    static ClipVertex interpolate(const ClipVertex& a, const ClipVertex& b, double t) {
        ClipVertex result;
        result.clip = a.clip + (b.clip - a.clip) * t;
        result.world = a.world + (b.world - a.world) * t;
        result.normal = a.normal + (b.normal - a.normal) * t;
        result.uv = a.uv + (b.uv - a.uv) * t;
        result.material = t < 0.5 ? a.material : b.material;
        return result;
    }

    static bool samePosition(const ClipVertex& a, const ClipVertex& b) {
        return std::abs(a.clip.x - b.clip.x) < 1e-11 &&
               std::abs(a.clip.y - b.clip.y) < 1e-11 &&
               std::abs(a.clip.z - b.clip.z) < 1e-11 &&
               std::abs(a.clip.w - b.clip.w) < 1e-11;
    }

    static std::vector<ClipVertex> clipAgainstPlane(const std::vector<ClipVertex>& input,
                                                    int plane) {
        std::vector<ClipVertex> output;
        output.reserve(input.size() + 1);
        if (input.empty()) {
            return output;
        }
        ClipVertex previous = input.back();
        double previous_distance = clipPlane(plane, previous.clip).x;
        bool previous_inside = previous_distance >= 0.0;
        for (const ClipVertex& current : input) {
            const double current_distance = clipPlane(plane, current.clip).x;
            const bool current_inside = current_distance >= 0.0;
            if (current_inside != previous_inside) {
                const double denominator = previous_distance - current_distance;
                const double t = std::abs(denominator) > 1e-20
                                     ? previous_distance / denominator
                                     : 0.0;
                output.push_back(interpolate(previous, current, std::clamp(t, 0.0, 1.0)));
            }
            if (current_inside) {
                if (output.empty() || !samePosition(output.back(), current)) {
                    output.push_back(current);
                }
            }
            previous = current;
            previous_distance = current_distance;
            previous_inside = current_inside;
        }
        if (output.size() > 1 && samePosition(output.front(), output.back())) {
            output.pop_back();
        }
        return output;
    }

    void processTriangle(const TerrainTriangle& triangle) {
        const TerrainVertex& a = mesh_.vertices[triangle.a];
        const TerrainVertex& b = mesh_.vertices[triangle.b];
        const TerrainVertex& c = mesh_.vertices[triangle.c];
        std::vector<ClipVertex> polygon;
        polygon.reserve(3);
        const TerrainVertex* source[3] = {&a, &b, &c};
        for (const TerrainVertex* vertex : source) {
            ClipVertex clipped;
            clipped.clip = view_projection_ * Vec4(vertex->position, 1.0);
            clipped.world = vertex->position;
            clipped.normal = vertex->normal;
            clipped.uv = vertex->uv;
            clipped.material = vertex->material;
            polygon.push_back(clipped);
        }

        for (int plane = 0; plane < 6 && !polygon.empty(); ++plane) {
            polygon = clipAgainstPlane(polygon, plane);
        }
        if (polygon.size() < 3) {
            ++stats_.triangles_frustum_culled;
            return;
        }
        if (polygon.size() > 3) {
            ++stats_.triangles_partially_clipped;
        }
        stats_.triangles_after_frustum_clip += polygon.size() - 2;

        for (std::size_t i = 1; i + 1 < polygon.size(); ++i) {
            const ClipVertex& v0 = polygon[0];
            const ClipVertex& v1 = polygon[i];
            const ClipVertex& v2 = polygon[i + 1];
            const Vec3 face_normal =
                cross(v1.world - v0.world, v2.world - v0.world);
            const Vec3 centroid = (v0.world + v1.world + v2.world) / 3.0;
            if (!triangle.transition &&
                dot(face_normal, config_.camera.position - centroid) <= 0.0) {
                ++stats_.triangles_backface_culled;
                continue;
            }
            if (length(face_normal) < 1e-12) {
                continue;
            }
            ++stats_.triangles_after_clipping;
            rasterize(v0, v1, v2);
        }
    }

    void rasterize(const ClipVertex& c0, const ClipVertex& c1, const ClipVertex& c2) {
        const ClipVertex* source[3] = {&c0, &c1, &c2};
        ScreenVertex vertices[3];
        for (int i = 0; i < 3; ++i) {
            if (source[i]->clip.w <= 1e-12) {
                return;
            }
            const double inv_w = 1.0 / source[i]->clip.w;
            vertices[i].x =
                (source[i]->clip.x * inv_w * 0.5 + 0.5) * static_cast<double>(internal_width_);
            vertices[i].y =
                (0.5 - source[i]->clip.y * inv_w * 0.5) *
                static_cast<double>(internal_height_);
            vertices[i].inv_w = inv_w;
            vertices[i].source = source[i];
        }

        const double area = (vertices[1].x - vertices[0].x) *
                                (vertices[2].y - vertices[0].y) -
                            (vertices[1].y - vertices[0].y) *
                                (vertices[2].x - vertices[0].x);
        if (std::abs(area) < 1e-12) {
            return;
        }
        ++stats_.triangles_rasterized;
        const double inverse_area = 1.0 / area;
        const double minimum_x =
            std::min({vertices[0].x, vertices[1].x, vertices[2].x});
        const double maximum_x =
            std::max({vertices[0].x, vertices[1].x, vertices[2].x});
        const double minimum_y =
            std::min({vertices[0].y, vertices[1].y, vertices[2].y});
        const double maximum_y =
            std::max({vertices[0].y, vertices[1].y, vertices[2].y});
        const int x_begin = std::max(0, static_cast<int>(std::floor(minimum_x)));
        const int x_end =
            std::min(internal_width_ - 1, static_cast<int>(std::ceil(maximum_x)));
        const int y_begin = std::max(0, static_cast<int>(std::floor(minimum_y)));
        const int y_end =
            std::min(internal_height_ - 1, static_cast<int>(std::ceil(maximum_y)));

        const double db1dx = (vertices[2].y - vertices[0].y) * inverse_area;
        const double db1dy = -(vertices[2].x - vertices[0].x) * inverse_area;
        const double db2dx = -(vertices[1].y - vertices[0].y) * inverse_area;
        const double db2dy = (vertices[1].x - vertices[0].x) * inverse_area;
        const double db0dx = -db1dx - db2dx;
        const double db0dy = -db1dy - db2dy;

        const double iw0 = vertices[0].inv_w;
        const double iw1 = vertices[1].inv_w;
        const double iw2 = vertices[2].inv_w;
        const double d_depth_dx = db0dx * iw0 + db1dx * iw1 + db2dx * iw2;
        const double d_depth_dy = db0dy * iw0 + db1dy * iw1 + db2dy * iw2;

        for (int y = y_begin; y <= y_end; ++y) {
            const double py = static_cast<double>(y) + 0.5;
            for (int x = x_begin; x <= x_end; ++x) {
                const double px = static_cast<double>(x) + 0.5;
                double b1 = ((px - vertices[0].x) * (vertices[2].y - vertices[0].y) -
                             (py - vertices[0].y) * (vertices[2].x - vertices[0].x)) *
                            inverse_area;
                double b2 = ((vertices[1].x - vertices[0].x) * (py - vertices[0].y) -
                             (vertices[1].y - vertices[0].y) * (px - vertices[0].x)) *
                            inverse_area;
                double b0 = 1.0 - b1 - b2;
                if (b0 < -1e-9 || b1 < -1e-9 || b2 < -1e-9) {
                    continue;
                }
                b0 = std::max(b0, 0.0);
                b1 = std::max(b1, 0.0);
                b2 = std::max(b2, 0.0);
                const double inverse_sum = 1.0 / (b0 + b1 + b2);
                b0 *= inverse_sum;
                b1 *= inverse_sum;
                b2 *= inverse_sum;

                const double depth = b0 * iw0 + b1 * iw1 + b2 * iw2;
                if (!(depth > 0.0)) {
                    continue;
                }
                GBufferSample& target =
                    gbuffer_[static_cast<std::size_t>(y) * internal_width_ + x];
                if (target.valid() && static_cast<float>(depth) <= target.inv_depth) {
                    continue;
                }

                const ClipVertex& s0 = *vertices[0].source;
                const ClipVertex& s1 = *vertices[1].source;
                const ClipVertex& s2 = *vertices[2].source;
                const Vec2 uv0 = s0.uv * iw0;
                const Vec2 uv1 = s1.uv * iw1;
                const Vec2 uv2 = s2.uv * iw2;
                const Vec2 weighted_uv =
                    uv0 * b0 + uv1 * b1 + uv2 * b2;
                const Vec2 uv = weighted_uv / depth;
                const Vec2 d_uv_dx =
                    (uv0 * db0dx + uv1 * db1dx + uv2 * db2dx - uv * d_depth_dx) / depth;
                const Vec2 d_uv_dy =
                    (uv0 * db0dy + uv1 * db1dy + uv2 * db2dy - uv * d_depth_dy) / depth;

                const Vec3 weighted_normal =
                    s0.normal * (b0 * iw0) + s1.normal * (b1 * iw1) +
                    s2.normal * (b2 * iw2);
                const Vec3 weighted_world =
                    s0.world * (b0 * iw0) + s1.world * (b1 * iw1) +
                    s2.world * (b2 * iw2);
                const Vec3 normal = normalize(weighted_normal / depth);
                const Vec3 world = weighted_world / depth;

                std::uint32_t material = s0.material;
                if (b1 >= b0 && b1 >= b2) {
                    material = s1.material;
                } else if (b2 >= b0 && b2 >= b1) {
                    material = s2.material;
                }

                target.px = static_cast<float>(world.x);
                target.py = static_cast<float>(world.y);
                target.pz = static_cast<float>(world.z);
                target.nx = static_cast<float>(normal.x);
                target.ny = static_cast<float>(normal.y);
                target.nz = static_cast<float>(normal.z);
                target.u = static_cast<float>(uv.x);
                target.v = static_cast<float>(uv.y);
                target.dudx = static_cast<float>(d_uv_dx.x);
                target.dvdx = static_cast<float>(d_uv_dx.y);
                target.dudy = static_cast<float>(d_uv_dy.x);
                target.dvdy = static_cast<float>(d_uv_dy.y);
                target.inv_depth = static_cast<float>(depth);
                target.material = material;
            }
        }
    }

    AnisoSampleResult sampleAnisotropic(const GBufferSample& sample) const {
        const Vec3 normal = normalize(
            {sample.nx, sample.ny, sample.nz});
        const Vec3 world{sample.px, sample.py, sample.pz};
        const Vec3 view = normalize(config_.camera.position - world);
        const double incidence = std::clamp(std::abs(dot(normal, view)), 0.08, 1.0);
        const double stretch = std::clamp(1.0 / incidence, 1.0, 10.0);

        Vec2 dx{sample.dudx, sample.dvdx};
        Vec2 dy{sample.dudy, sample.dvdy};
        const double length_x = length(dx);
        const double length_y = length(dy);
        Vec2 major = length_x >= length_y ? dx : dy;
        Vec2 minor = length_x >= length_y ? dy : dx;
        double major_length = std::max(length_x, length_y) *
                              static_cast<double>(config_.texture.size) * stretch;
        double minor_length =
            std::min(length_x, length_y) * static_cast<double>(config_.texture.size);
        if (length(major) < 1e-20) {
            major = {1.0, 0.0};
        } else {
            major = normalize(major);
        }
        if (length(minor) < 1e-20) {
            minor = {-major.y, major.x};
        } else {
            minor = normalize(minor - major * dot(minor, major));
            if (length(minor) < 1e-20) {
                minor = {-major.y, major.x};
            }
        }
        major_length = std::clamp(major_length, 0.0, 64.0);
        minor_length = std::clamp(minor_length, 0.0, 16.0);

        const int major_taps =
            std::clamp(static_cast<int>(std::ceil(std::min(major_length, 4.0))), 1, 4);
        const int minor_taps =
            std::clamp(static_cast<int>(std::ceil(std::min(minor_length, 4.0))), 1, 4);
        const int total_taps = major_taps * minor_taps;

        Vec3 accum{};
        double weight_sum = 0.0;
        const double inv_size = 1.0 / static_cast<double>(config_.texture.size);
        const Vec2 major_uv = major * (major_length * inv_size);
        const Vec2 minor_uv = minor * (minor_length * inv_size);
        for (int iy = 0; iy < minor_taps; ++iy) {
            const double ty =
                minor_taps == 1 ? 0.0
                                : (static_cast<double>(iy) / (minor_taps - 1) - 0.5);
            for (int ix = 0; ix < major_taps; ++ix) {
                const double tx =
                    major_taps == 1 ? 0.0
                                    : (static_cast<double>(ix) / (major_taps - 1) - 0.5);
                const Vec2 offset = major_uv * tx + minor_uv * ty;
                const double radius_sq = tx * tx * 2.2 + ty * ty * 2.2;
                const double weight = std::exp(-radius_sq);
                accum += texture_.sampleBilinear(sample.u + offset.x, sample.v + offset.y) *
                         weight;
                weight_sum += weight;
            }
        }
        return {accum / weight_sum, total_taps};
    }

    Vec3 shadeSky(int x, int y) const {
        const double t = static_cast<double>(y) /
                         std::max(1.0, static_cast<double>(internal_height_ - 1));
        const double azimuth = static_cast<double>(x) /
                               std::max(1.0, static_cast<double>(internal_width_ - 1));
        const Vec3 horizon{0.46 + 0.10 * azimuth, 0.56 + 0.06 * azimuth, 0.70};
        const Vec3 zenith{0.12, 0.25, 0.52};
        const double blend = std::pow(t, 0.55);
        return horizon * (1.0 - blend) + zenith * blend;
    }

    Vec3 shadeSurface(const GBufferSample& sample, int& aniso_samples) const {
        Vec3 normal = normalize({sample.nx, sample.ny, sample.nz});
        const Vec3 world{sample.px, sample.py, sample.pz};
        const Vec3 view = normalize(config_.camera.position - world);
        if (dot(normal, view) < 0.0) {
            normal = -normal;
        }

        const AnisoSampleResult texture = sampleAnisotropic(sample);
        aniso_samples = texture.samples;
        const Vec3 tint = materialTint(sample.material);
        const Vec3 albedo{
            texture.color.x * tint.x * 1.7,
            texture.color.y * tint.y * 1.7,
            texture.color.z * tint.z * 1.7,
        };

        Vec3 color{
            albedo.x * config_.ambient.x,
            albedo.y * config_.ambient.y,
            albedo.z * config_.ambient.z,
        };
        for (const DirectionalLight& light : config_.directional_lights) {
            const Vec3 light_dir = normalize(light.direction);
            const double n_dot_l = std::max(0.0, dot(normal, light_dir));
            if (n_dot_l <= 0.0) {
                continue;
            }
            const Vec3 half_vector = normalize(light_dir + view);
            const double specular =
                std::pow(std::max(0.0, dot(normal, half_vector)), 42.0) * 0.13;
            color += Vec3{
                         albedo.x * light.color.x + specular * light.color.x,
                         albedo.y * light.color.y + specular * light.color.y,
                         albedo.z * light.color.z + specular * light.color.z,
                     } *
                     (light.intensity * n_dot_l);
        }
        for (const PointLight& light : config_.point_lights) {
            const Vec3 light_vector = light.position - world;
            const double distance = length(light_vector);
            if (distance >= light.range || distance < 1e-9) {
                continue;
            }
            const Vec3 light_dir = light_vector / distance;
            const double n_dot_l = std::max(0.0, dot(normal, light_dir));
            if (n_dot_l <= 0.0) {
                continue;
            }
            const double falloff = std::max(0.0, 1.0 - distance / light.range);
            const double attenuation =
                falloff * falloff /
                (1.0 + 0.12 * distance + 0.018 * distance * distance);
            const Vec3 half_vector = normalize(light_dir + view);
            const double specular =
                std::pow(std::max(0.0, dot(normal, half_vector)), 36.0) * 0.09;
            color += Vec3{
                         albedo.x * light.color.x + specular * light.color.x,
                         albedo.y * light.color.y + specular * light.color.y,
                         albedo.z * light.color.z + specular * light.color.z,
                     } *
                     (light.intensity * n_dot_l * attenuation);
        }

        const auto curve = [](double value) {
            value = std::max(0.0, value);
            return (value * (2.51 * value + 0.03)) /
                   (value * (2.43 * value + 0.59) + 0.14);
        };
        return {curve(color.x), curve(color.y), curve(color.z)};
    }

    void shadeAndResolve() {
        const int ss = config_.render.supersampling;
        const double normalization = 1.0 / static_cast<double>(ss * ss);
        for (int y = 0; y < internal_height_; ++y) {
            for (int x = 0; x < internal_width_; ++x) {
                const GBufferSample& sample =
                    gbuffer_[static_cast<std::size_t>(y) * internal_width_ + x];
                Vec3 color;
                if (sample.valid()) {
                    int aniso_samples = 0;
                    color = shadeSurface(sample, aniso_samples);
                    ++stats_.covered_samples;
                    stats_.anisotropic_samples += static_cast<std::uint64_t>(aniso_samples);
                    stats_.max_anisotropic_samples =
                        std::max(stats_.max_anisotropic_samples, aniso_samples);
                } else {
                    color = shadeSky(x, y);
                }
                output_.at(x / ss, y / ss) += color * normalization;
            }
        }
    }

    const HeightField& heightfield_;
    const TerrainMesh& mesh_;
    const AppConfig& config_;
    ColorBuffer& output_;
    RenderStats& stats_;
    ProceduralTexture texture_;
    int internal_width_;
    int internal_height_;
    std::vector<GBufferSample> gbuffer_;
    Mat4 view_;
    Mat4 projection_;
    Mat4 view_projection_;
};

}  // namespace

bool renderTerrain(const HeightField& heightfield, const TerrainMesh& mesh,
                   const AppConfig& config, ColorBuffer& output, RenderStats& stats,
                   std::string& error) {
    error.clear();
    if (mesh.vertices.empty() || mesh.triangles.empty()) {
        error = "cannot render an empty terrain mesh";
        return false;
    }
    output = ColorBuffer(config.render.width, config.render.height);
    stats = {};
    Renderer renderer(heightfield, mesh, config, output, stats);
    renderer.render();
    return true;
}

}  // namespace ct
