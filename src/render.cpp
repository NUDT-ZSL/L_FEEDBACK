#include "render.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace lfb {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsilon = 1e-12;
constexpr int kLatitudeSegments = 48;
constexpr int kLongitudeSegments = 96;
constexpr double kCameraDistance = 3.2;
constexpr double kFieldOfViewYDegrees = 45.0;
constexpr double kNearPlane = 0.1;
constexpr double kFarPlane = 100.0;

struct Vec4 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 0.0;
};

struct MeshVertex {
    Vec3 position;
    Vec3 normal;
    Vec3 tangent;
};

struct Mesh {
    std::vector<MeshVertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct ClipVertex {
    Vec4 clip;
    Vec3 world;
    Vec3 normal;
    Vec3 tangent;
};

struct ScreenVertex {
    double x = 0.0;
    double y = 0.0;
    double zOverW = 0.0;
    double inverseW = 0.0;
    Vec3 worldOverW;
    Vec3 normalOverW;
    Vec3 tangentOverW;
};

double dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(Vec3 a, Vec3 b) {
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

Vec3 add(Vec3 a, Vec3 b) {
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 sub(Vec3 a, Vec3 b) {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 mul(Vec3 a, double scalar) {
    return Vec3{a.x * scalar, a.y * scalar, a.z * scalar};
}

Vec3 divide(Vec3 a, double scalar) {
    return Vec3{a.x / scalar, a.y / scalar, a.z / scalar};
}

Vec3 lerp(Vec3 a, Vec3 b, double t) {
    return Vec3{
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
    };
}

Vec4 lerp(Vec4 a, Vec4 b, double t) {
    return Vec4{
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
        a.w + (b.w - a.w) * t,
    };
}

ClipVertex lerp(const ClipVertex& a, const ClipVertex& b, double t) {
    ClipVertex result;
    result.clip = lerp(a.clip, b.clip, t);
    result.world = lerp(a.world, b.world, t);
    result.normal = lerp(a.normal, b.normal, t);
    result.tangent = lerp(a.tangent, b.tangent, t);
    return result;
}

Vec3 normalizeSafe(Vec3 value, Vec3 fallback) {
    const double len = length(value);
    if (len <= 1e-12) {
        return fallback;
    }
    return divide(value, len);
}

const Mesh& sphereMesh() {
    static const Mesh mesh = [] {
        Mesh result;
        result.vertices.reserve(
            static_cast<std::size_t>(kLatitudeSegments + 1) *
            static_cast<std::size_t>(kLongitudeSegments + 1));
        for (int lat = 0; lat <= kLatitudeSegments; ++lat) {
            const double theta = kPi * static_cast<double>(lat) /
                                 static_cast<double>(kLatitudeSegments);
            const double sinTheta = std::sin(theta);
            const double cosTheta = std::cos(theta);
            for (int lon = 0; lon <= kLongitudeSegments; ++lon) {
                const double phi = 2.0 * kPi * static_cast<double>(lon) /
                                   static_cast<double>(kLongitudeSegments);
                const double sinPhi = std::sin(phi);
                const double cosPhi = std::cos(phi);
                const Vec3 position{sinTheta * cosPhi, cosTheta, sinTheta * sinPhi};
                const Vec3 tangent{-sinPhi, 0.0, cosPhi};
                result.vertices.push_back(MeshVertex{position, position, tangent});
            }
        }

        const int rowStride = kLongitudeSegments + 1;
        for (int lat = 0; lat < kLatitudeSegments; ++lat) {
            for (int lon = 0; lon < kLongitudeSegments; ++lon) {
                const std::uint32_t a =
                    static_cast<std::uint32_t>(lat * rowStride + lon);
                const std::uint32_t d =
                    static_cast<std::uint32_t>(lat * rowStride + lon + 1);
                const std::uint32_t b =
                    static_cast<std::uint32_t>((lat + 1) * rowStride + lon);
                const std::uint32_t c =
                    static_cast<std::uint32_t>((lat + 1) * rowStride + lon + 1);
                result.indices.push_back(a);
                result.indices.push_back(c);
                result.indices.push_back(b);
                result.indices.push_back(a);
                result.indices.push_back(d);
                result.indices.push_back(c);
            }
        }
        return result;
    }();
    return mesh;
}

double clipPlaneDistance(const Vec4& p, int plane) {
    switch (plane) {
        case 0: return p.x + p.w;
        case 1: return p.w - p.x;
        case 2: return p.y + p.w;
        case 3: return p.w - p.y;
        case 4: return p.z + p.w;
        case 5: return p.w - p.z;
        default: return -1.0;
    }
}

int clipTriangle(const ClipVertex input[3], ClipVertex output[16]) {
    std::array<ClipVertex, 16> current{};
    std::array<ClipVertex, 16> next{};
    int count = 3;
    current[0] = input[0];
    current[1] = input[1];
    current[2] = input[2];

    for (int plane = 0; plane < 6 && count > 0; ++plane) {
        int nextCount = 0;
        int previous = count - 1;
        for (int i = 0; i < count; ++i) {
            const double previousDistance = clipPlaneDistance(current[previous].clip, plane);
            const double currentDistance = clipPlaneDistance(current[i].clip, plane);
            const bool previousInside = previousDistance >= -kEpsilon;
            const bool currentInside = currentDistance >= -kEpsilon;

            if (previousInside != currentInside) {
                const double denominator = previousDistance - currentDistance;
                const double t = std::clamp(previousDistance / denominator, 0.0, 1.0);
                if (nextCount < 16) {
                    next[nextCount++] = lerp(current[previous], current[i], t);
                }
            }
            if (currentInside && nextCount < 16) {
                next[nextCount++] = current[i];
            }
            previous = i;
        }
        count = nextCount;
        current = next;
    }

    for (int i = 0; i < count; ++i) {
        output[i] = current[i];
    }
    return count;
}

bool projectVertex(const MeshVertex& vertex,
                   Vec3 eye,
                   Vec3 xAxis,
                   Vec3 yAxis,
                   Vec3 zAxis,
                   double aspect,
                   ClipVertex& output) {
    const Vec3 relative = sub(vertex.position, eye);
    const Vec4 view{
        dot(xAxis, relative),
        dot(yAxis, relative),
        dot(zAxis, relative),
        1.0,
    };

    const double f = 1.0 / std::tan((kFieldOfViewYDegrees * kPi / 180.0) * 0.5);
    const double projectionA = (kFarPlane + kNearPlane) / (kNearPlane - kFarPlane);
    const double projectionB =
        (2.0 * kFarPlane * kNearPlane) / (kNearPlane - kFarPlane);

    output.clip = Vec4{
        view.x * (f / aspect),
        view.y * f,
        projectionA * view.z + projectionB,
        -view.z,
    };
    output.world = vertex.position;
    output.normal = vertex.normal;
    output.tangent = vertex.tangent;
    return output.clip.w > kEpsilon;
}

ScreenVertex makeScreenVertex(const ClipVertex& vertex, int width, int height) {
    const double inverseW = 1.0 / vertex.clip.w;
    const double ndcX = vertex.clip.x * inverseW;
    const double ndcY = vertex.clip.y * inverseW;
    const double ndcZ = vertex.clip.z * inverseW;
    ScreenVertex result;
    result.x = (ndcX + 1.0) * 0.5 * static_cast<double>(width);
    result.y = (1.0 - ndcY) * 0.5 * static_cast<double>(height);
    result.zOverW = ndcZ * inverseW;
    result.inverseW = inverseW;
    result.worldOverW = mul(vertex.world, inverseW);
    result.normalOverW = mul(vertex.normal, inverseW);
    result.tangentOverW = mul(vertex.tangent, inverseW);
    return result;
}

bool barycentric(double px,
                 double py,
                 const ScreenVertex& a,
                 const ScreenVertex& b,
                 const ScreenVertex& c,
                 double& weightA,
                 double& weightB,
                 double& weightC) {
    const double area =
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if (std::abs(area) <= 1e-20) {
        return false;
    }
    const double edgeAB_AC = 1.0 / area;
    weightA = ((b.x - px) * (c.y - py) - (b.y - py) * (c.x - px)) * edgeAB_AC;
    weightB = ((c.x - px) * (a.y - py) - (c.y - py) * (a.x - px)) * edgeAB_AC;
    weightC = 1.0 - weightA - weightB;
    const double tolerance = -1e-9;
    return weightA >= tolerance && weightB >= tolerance && weightC >= tolerance;
}

std::uint8_t quantize(double linear) {
    if (!(linear > 0.0)) {
        return 0;
    }
    double mapped = 1.0;
    if (std::isfinite(linear)) {
        mapped = linear / (1.0 + linear);
    }
    if (!(mapped > 0.0)) {
        return 0;
    }
    const double clamped = std::clamp(mapped, 0.0, 1.0);
    const int value = static_cast<int>(clamped * 255.0 + 0.5);
    return static_cast<std::uint8_t>(value);
}

void shadePixel(const Scene& scene,
                Vec3 worldPosition,
                Vec3 interpolatedNormal,
                Vec3 interpolatedTangent,
                Vec3 eye,
                std::uint8_t* pixel) {
    Vec3 normal = normalizeSafe(interpolatedNormal, Vec3{0.0, 1.0, 0.0});
    Vec3 tangent = sub(
        interpolatedTangent,
        mul(normal, dot(interpolatedTangent, normal)));
    tangent = normalizeSafe(
        tangent,
        normalizeSafe(cross(Vec3{0.0, 1.0, 0.0}, normal), Vec3{1.0, 0.0, 0.0}));
    const Vec3 viewDirection = normalizeSafe(sub(eye, worldPosition), Vec3{0.0, 0.0, 1.0});

    const double u =
        std::atan2(worldPosition.z, worldPosition.x) / (2.0 * kPi) + 0.5;
    const double v =
        std::acos(std::clamp(worldPosition.y, -1.0, 1.0)) / kPi;

    double diffuse[3] = {0.0, 0.0, 0.0};
    double specular[3] = {0.0, 0.0, 0.0};
    double emission[3] = {0.0, 0.0, 0.0};

    for (std::size_t layerIndex = 0; layerIndex < scene.layers.size(); ++layerIndex) {
        const MaterialLayer& layer = scene.layers[layerIndex];
        const Vec3 bitangent = cross(normal, tangent);
        const double phase = 0.37 * static_cast<double>(layerIndex + 1);
        const double waveU =
            std::sin((u * 8.0 + v * 3.0 + phase) * 2.0 * kPi);
        const double waveV =
            std::cos((v * 10.0 - u * 2.0 + phase) * 2.0 * kPi);
        const Vec3 tangentSpaceNormal = normalizeSafe(
            add(add(mul(tangent, layer.normalStrength * waveU),
                    mul(bitangent, layer.normalStrength * waveV)),
                normal),
            normal);
        normal = tangentSpaceNormal;
        tangent = normalizeSafe(
            sub(tangent, mul(normal, dot(tangent, normal))),
            normalizeSafe(cross(Vec3{0.0, 1.0, 0.0}, normal),
                          Vec3{1.0, 0.0, 0.0}));

        const double shininess = 2.0 / (layer.roughness * layer.roughness + 1e-4);
        const double specularStrength = 0.15 + 0.85 * (1.0 - layer.roughness);
        for (const Light& light : scene.lights) {
            const double nDotL = std::max(0.0, dot(normal, light.direction));
            const Vec3 halfVector =
                normalizeSafe(add(light.direction, viewDirection), light.direction);
            const double nDotH = std::max(0.0, dot(normal, halfVector));
            double specularTerm = 0.0;
            if (nDotH > 0.0) {
                specularTerm =
                    specularStrength * std::pow(nDotH, shininess) * light.intensity;
            }
            const double diffuseTerm = nDotL * light.intensity;
            diffuse[0] += layer.baseColor.r * light.color.r * diffuseTerm;
            diffuse[1] += layer.baseColor.g * light.color.g * diffuseTerm;
            diffuse[2] += layer.baseColor.b * light.color.b * diffuseTerm;
            specular[0] += light.color.r * specularTerm;
            specular[1] += light.color.g * specularTerm;
            specular[2] += light.color.b * specularTerm;
        }
        emission[0] += layer.emissionColor.r;
        emission[1] += layer.emissionColor.g;
        emission[2] += layer.emissionColor.b;
    }

    pixel[0] = quantize(diffuse[0] + specular[0] + emission[0]);
    pixel[1] = quantize(diffuse[1] + specular[1] + emission[1]);
    pixel[2] = quantize(diffuse[2] + specular[2] + emission[2]);
    pixel[3] = 255;
}

void rasterizeTriangle(const Scene& scene,
                       const ScreenVertex screen[3],
                       Vec3 eye,
                       Image& image,
                       std::vector<double>& depthBuffer) {
    const double minX = std::min({screen[0].x, screen[1].x, screen[2].x});
    const double maxX = std::max({screen[0].x, screen[1].x, screen[2].x});
    const double minY = std::min({screen[0].y, screen[1].y, screen[2].y});
    const double maxY = std::max({screen[0].y, screen[1].y, screen[2].y});

    const int x0 = std::max(0, static_cast<int>(std::ceil(minX - 0.5)));
    const int x1 = std::min(image.width - 1, static_cast<int>(std::floor(maxX - 0.5)));
    const int y0 = std::max(0, static_cast<int>(std::ceil(minY - 0.5)));
    const int y1 = std::min(image.height - 1, static_cast<int>(std::floor(maxY - 0.5)));
    if (x0 > x1 || y0 > y1) {
        return;
    }

    for (int y = y0; y <= y1; ++y) {
        const double py = static_cast<double>(y) + 0.5;
        for (int x = x0; x <= x1; ++x) {
            const double px = static_cast<double>(x) + 0.5;
            double weightA = 0.0;
            double weightB = 0.0;
            double weightC = 0.0;
            if (!barycentric(px, py, screen[0], screen[1], screen[2],
                             weightA, weightB, weightC)) {
                continue;
            }

            const double inverseW =
                weightA * screen[0].inverseW +
                weightB * screen[1].inverseW +
                weightC * screen[2].inverseW;
            if (!(inverseW > 0.0)) {
                continue;
            }
            const double ndcDepth =
                (weightA * screen[0].zOverW +
                 weightB * screen[1].zOverW +
                 weightC * screen[2].zOverW) /
                inverseW;
            const std::size_t pixelIndex =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width) +
                static_cast<std::size_t>(x);
            if (!(ndcDepth < depthBuffer[pixelIndex])) {
                continue;
            }
            depthBuffer[pixelIndex] = ndcDepth;

            const Vec3 worldPosition = divide(
                add(add(mul(screen[0].worldOverW, weightA),
                        mul(screen[1].worldOverW, weightB)),
                    mul(screen[2].worldOverW, weightC)),
                inverseW);
            const Vec3 normal = divide(
                add(add(mul(screen[0].normalOverW, weightA),
                        mul(screen[1].normalOverW, weightB)),
                    mul(screen[2].normalOverW, weightC)),
                inverseW);
            const Vec3 tangent = divide(
                add(add(mul(screen[0].tangentOverW, weightA),
                        mul(screen[1].tangentOverW, weightB)),
                    mul(screen[2].tangentOverW, weightC)),
                inverseW);

            shadePixel(scene, worldPosition, normal, tangent, eye,
                       &image.rgba[pixelIndex * 4]);
        }
    }
}

}  // namespace

Image renderFrame(const Scene& scene, const FrameDesc& frame) {
    Image image(scene.width, scene.height);
    std::vector<double> depthBuffer(
        static_cast<std::size_t>(scene.width) * static_cast<std::size_t>(scene.height),
        std::numeric_limits<double>::infinity());

    const Vec3 zAxis = normalized(frame.direction);
    const Vec3 eye = mul(zAxis, kCameraDistance);
    const Vec3 reference =
        std::abs(zAxis.y) > 0.98 ? Vec3{1.0, 0.0, 0.0} : Vec3{0.0, 1.0, 0.0};
    const Vec3 xAxis = normalizeSafe(cross(reference, zAxis), Vec3{1.0, 0.0, 0.0});
    const Vec3 yAxis = normalizeSafe(cross(zAxis, xAxis), Vec3{0.0, 1.0, 0.0});
    const double aspect = static_cast<double>(scene.width) /
                          static_cast<double>(scene.height);

    const Mesh& mesh = sphereMesh();
    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        ClipVertex projected[3];
        bool valid = true;
        for (int vertexIndex = 0; vertexIndex < 3; ++vertexIndex) {
            if (!projectVertex(mesh.vertices[mesh.indices[i + vertexIndex]],
                               eye, xAxis, yAxis, zAxis, aspect, projected[vertexIndex])) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            continue;
        }

        ClipVertex clipped[16];
        const int count = clipTriangle(projected, clipped);
        if (count < 3) {
            continue;
        }
        for (int fan = 1; fan + 1 < count; ++fan) {
            ScreenVertex screen[3] = {
                makeScreenVertex(clipped[0], scene.width, scene.height),
                makeScreenVertex(clipped[fan], scene.width, scene.height),
                makeScreenVertex(clipped[fan + 1], scene.width, scene.height),
            };
            rasterizeTriangle(scene, screen, eye, image, depthBuffer);
        }
    }
    return image;
}

}  // namespace lfb
