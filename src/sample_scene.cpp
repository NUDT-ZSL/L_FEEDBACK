#include "sample_scene.h"

#include "json.h"
#include "math.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace softraster {
namespace {

using Array = JsonValue::Array;
using Object = JsonValue::Object;

JsonValue number(double value) { return JsonValue(value); }

void appendIndex(Array& indices, std::uint32_t value) {
    indices.push_back(number(static_cast<double>(value)));
}

JsonValue vec2(double x, double y) {
    return JsonValue(Array{number(x), number(y)});
}

JsonValue vec3(double x, double y, double z) {
    return JsonValue(Array{number(x), number(y), number(z)});
}

JsonValue vec3(Vec3 value) {
    return vec3(static_cast<double>(value.x), static_cast<double>(value.y),
                static_cast<double>(value.z));
}

JsonValue matrixJson(const Mat4& matrix) {
    Array values;
    values.reserve(16);
    for (const float value : matrix.m) {
        values.emplace_back(static_cast<double>(value));
    }
    return JsonValue(std::move(values));
}

JsonValue vertexJson(Vec3 position, Vec3 normal, Vec2 uv) {
    Object vertex;
    vertex.emplace("position", vec3(position));
    vertex.emplace("normal", vec3(normal));
    vertex.emplace("uv", vec2(static_cast<double>(uv.x), static_cast<double>(uv.y)));
    return JsonValue(std::move(vertex));
}

JsonValue meshJson(std::string name, Array vertices, Array indices) {
    Object mesh;
    mesh.emplace("name", JsonValue(std::move(name)));
    mesh.emplace("vertices", JsonValue(std::move(vertices)));
    mesh.emplace("indices", JsonValue(std::move(indices)));
    return JsonValue(std::move(mesh));
}

JsonValue normalTextureJson(int width, int height, float frequency, float amplitude,
                            float phase) {
    Object texture;
    texture.emplace("width", number(width));
    texture.emplace("height", number(height));
    texture.emplace("strength", number(1.0));
    Array data;
    data.reserve(static_cast<std::size_t>(width * height * 3));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double u = static_cast<double>(x) / static_cast<double>(width);
            const double v = static_cast<double>(y) / static_cast<double>(height);
            const double nx =
                amplitude * std::sin(6.283185307179586 * u * frequency + phase) *
                std::cos(6.283185307179586 * v * (frequency - 0.5));
            const double ny =
                amplitude * std::cos(6.283185307179586 * u * (frequency + 0.5)) *
                std::sin(6.283185307179586 * v * frequency - phase);
            const double nz =
                std::sqrt(std::max(0.05, 1.0 - nx * nx - ny * ny));
            data.emplace_back(nx);
            data.emplace_back(ny);
            data.emplace_back(nz);
        }
    }
    texture.emplace("data", JsonValue(std::move(data)));
    return JsonValue(std::move(texture));
}

JsonValue materialJson(std::string name, Vec3 baseColor, Vec3 specularColor,
                       double shininess, double opacity, bool withNormalMap,
                       float frequency, float amplitude, float phase) {
    Object material;
    material.emplace("name", JsonValue(std::move(name)));
    material.emplace("baseColor", vec3(baseColor));
    material.emplace("specularColor", vec3(specularColor));
    material.emplace("shininess", number(shininess));
    material.emplace("opacity", number(opacity));
    if (withNormalMap) {
        material.emplace("normalMap",
                         normalTextureJson(16, 16, frequency, amplitude, phase));
    }
    return JsonValue(std::move(material));
}

void addCylinder(Array& vertices, Array& indices, float bottomY, float topY,
                 float bottomRadius, float topRadius, int segments, bool caps) {
    const std::uint32_t base = static_cast<std::uint32_t>(vertices.size());
    for (int ring = 0; ring < 2; ++ring) {
        const float y = ring == 0 ? bottomY : topY;
        const float radius = ring == 0 ? bottomRadius : topRadius;
        for (int i = 0; i < segments; ++i) {
            const double theta =
                6.283185307179586 * static_cast<double>(i) / static_cast<double>(segments);
            const float x = radius * static_cast<float>(std::cos(theta));
            const float z = radius * static_cast<float>(std::sin(theta));
            const Vec3 normal = normalize(Vec3{x, 0.0F, z}, {0.0F, 0.0F, 1.0F});
            vertices.push_back(vertexJson(
                {x, y, z}, normal,
                {static_cast<float>(i) / static_cast<float>(segments),
                 ring == 0 ? 0.0F : 1.0F}));
        }
    }
    for (int i = 0; i < segments; ++i) {
        const std::uint32_t lower = base + static_cast<std::uint32_t>(i);
        const std::uint32_t lowerNext =
            base + static_cast<std::uint32_t>((i + 1) % segments);
        const std::uint32_t upper =
            base + static_cast<std::uint32_t>(segments + i);
        const std::uint32_t upperNext =
            base + static_cast<std::uint32_t>(segments + (i + 1) % segments);
        appendIndex(indices, lower);
        appendIndex(indices, upper);
        appendIndex(indices, upperNext);
        appendIndex(indices, lower);
        appendIndex(indices, upperNext);
        appendIndex(indices, lowerNext);
    }
    if (caps) {
        const std::uint32_t topCenter = static_cast<std::uint32_t>(vertices.size());
        vertices.push_back(vertexJson({0.0F, topY, 0.0F}, {0.0F, 1.0F, 0.0F},
                                      {0.5F, 0.5F}));
        const std::uint32_t bottomCenter = static_cast<std::uint32_t>(vertices.size());
        vertices.push_back(vertexJson({0.0F, bottomY, 0.0F}, {0.0F, -1.0F, 0.0F},
                                      {0.5F, 0.5F}));
        for (int i = 0; i < segments; ++i) {
            const std::uint32_t lower = base + static_cast<std::uint32_t>(i);
            const std::uint32_t lowerNext =
                base + static_cast<std::uint32_t>((i + 1) % segments);
            const std::uint32_t upper =
                base + static_cast<std::uint32_t>(segments + i);
            const std::uint32_t upperNext =
                base + static_cast<std::uint32_t>(segments + (i + 1) % segments);
            appendIndex(indices, topCenter);
            appendIndex(indices, upperNext);
            appendIndex(indices, upper);
            appendIndex(indices, bottomCenter);
            appendIndex(indices, lower);
            appendIndex(indices, lowerNext);
        }
    }
}

void addSphere(Array& vertices, Array& indices, float radius, int slices, int stacks) {
    const std::uint32_t top = static_cast<std::uint32_t>(vertices.size());
    vertices.push_back(vertexJson({0.0F, radius, 0.0F}, {0.0F, 1.0F, 0.0F}, {0.5F, 0.0F}));

    const int rings = stacks - 1;
    const std::uint32_t firstRing = static_cast<std::uint32_t>(vertices.size());
    for (int row = 1; row < stacks; ++row) {
        const double phi = 3.141592653589793 * static_cast<double>(row) /
                           static_cast<double>(stacks);
        const float ringRadius = radius * static_cast<float>(std::sin(phi));
        const float y = radius * static_cast<float>(std::cos(phi));
        for (int i = 0; i < slices; ++i) {
            const double theta =
                6.283185307179586 * static_cast<double>(i) / static_cast<double>(slices);
            const float x = ringRadius * static_cast<float>(std::cos(theta));
            const float z = ringRadius * static_cast<float>(std::sin(theta));
            const Vec3 normal = normalize(Vec3{x, y, z}, {0.0F, 1.0F, 0.0F});
            vertices.push_back(vertexJson(
                {x, y, z}, normal,
                {static_cast<float>(i) / static_cast<float>(slices),
                 1.0F - static_cast<float>(row) / static_cast<float>(stacks)}));
        }
    }
    const std::uint32_t bottom = static_cast<std::uint32_t>(vertices.size());
    vertices.push_back(
        vertexJson({0.0F, -radius, 0.0F}, {0.0F, -1.0F, 0.0F}, {0.5F, 1.0F}));

    for (int i = 0; i < slices; ++i) {
        const std::uint32_t current = firstRing + static_cast<std::uint32_t>(i);
        const std::uint32_t next =
            firstRing + static_cast<std::uint32_t>((i + 1) % slices);
        appendIndex(indices, top);
        appendIndex(indices, next);
        appendIndex(indices, current);
    }
    for (int row = 0; row + 1 < rings; ++row) {
        const std::uint32_t rowStart =
            firstRing + static_cast<std::uint32_t>(row * slices);
        const std::uint32_t nextStart =
            firstRing + static_cast<std::uint32_t>((row + 1) * slices);
        for (int i = 0; i < slices; ++i) {
            const std::uint32_t current = rowStart + static_cast<std::uint32_t>(i);
            const std::uint32_t next =
                rowStart + static_cast<std::uint32_t>((i + 1) % slices);
            const std::uint32_t below = nextStart + static_cast<std::uint32_t>(i);
            const std::uint32_t belowNext =
                nextStart + static_cast<std::uint32_t>((i + 1) % slices);
            appendIndex(indices, current);
            appendIndex(indices, next);
            appendIndex(indices, belowNext);
            appendIndex(indices, current);
            appendIndex(indices, belowNext);
            appendIndex(indices, below);
        }
    }
    const int lastRow = rings - 1;
    const std::uint32_t lastStart =
        firstRing + static_cast<std::uint32_t>(lastRow * slices);
    for (int i = 0; i < slices; ++i) {
        const std::uint32_t current = lastStart + static_cast<std::uint32_t>(i);
        const std::uint32_t next =
            lastStart + static_cast<std::uint32_t>((i + 1) % slices);
        appendIndex(indices, bottom);
        appendIndex(indices, current);
        appendIndex(indices, next);
    }
}

void addAura(Array& vertices, Array& indices) {
    constexpr int kSegments = 16;
    constexpr int kRows = 5;
    constexpr float kBottom = 0.25F;
    constexpr float kTop = 1.62F;
    for (int row = 0; row < kRows; ++row) {
        const float t = static_cast<float>(row) / static_cast<float>(kRows - 1);
        const float y = kBottom + (kTop - kBottom) * t;
        for (int i = 0; i < kSegments; ++i) {
            const double theta =
                6.283185307179586 * static_cast<double>(i) / static_cast<double>(kSegments);
            const float radius =
                0.76F + 0.075F * static_cast<float>(std::sin(2.399 * i + 0.7 * row)) +
                0.035F * static_cast<float>(std::cos(0.77 * i - 1.1 * row));
            const float x = radius * static_cast<float>(std::cos(theta));
            const float z = radius * static_cast<float>(std::sin(theta));
            const Vec3 normal = normalize(Vec3{x, 0.12F, z}, {0.0F, 0.0F, 1.0F});
            vertices.push_back(
                vertexJson({x, y, z}, normal,
                           {static_cast<float>(i) / static_cast<float>(kSegments), t}));
        }
    }
    for (int row = 0; row + 1 < kRows; ++row) {
        const std::uint32_t currentRow =
            static_cast<std::uint32_t>(row * kSegments);
        const std::uint32_t nextRow =
            static_cast<std::uint32_t>((row + 1) * kSegments);
        for (int i = 0; i < kSegments; ++i) {
            const std::uint32_t lower = currentRow + static_cast<std::uint32_t>(i);
            const std::uint32_t lowerNext =
                currentRow + static_cast<std::uint32_t>((i + 1) % kSegments);
            const std::uint32_t upper = nextRow + static_cast<std::uint32_t>(i);
            const std::uint32_t upperNext =
                nextRow + static_cast<std::uint32_t>((i + 1) % kSegments);
            appendIndex(indices, lower);
            appendIndex(indices, upper);
            appendIndex(indices, upperNext);
            appendIndex(indices, lower);
            appendIndex(indices, upperNext);
            appendIndex(indices, lowerNext);
        }
    }
}

JsonValue objectJson(std::string name, int mesh, int material, const Mat4& model) {
    Object object;
    object.emplace("name", JsonValue(std::move(name)));
    object.emplace("mesh", number(mesh));
    object.emplace("material", number(material));
    object.emplace("model", matrixJson(model));
    return JsonValue(std::move(object));
}

}  // namespace

std::string buildSampleSceneJson() {
    Array bodyVertices;
    Array bodyIndices;
    addCylinder(bodyVertices, bodyIndices, 0.25F, 1.35F, 0.55F, 0.43F, 14, true);

    Array headVertices;
    Array headIndices;
    addSphere(headVertices, headIndices, 0.42F, 16, 9);

    Array auraVertices;
    Array auraIndices;
    addAura(auraVertices, auraIndices);

    Array floorVertices;
    Array floorIndices;
    floorVertices.push_back(vertexJson({-3.0F, 0.0F, -3.0F}, {0.0F, 1.0F, 0.0F}, {0.0F, 0.0F}));
    floorVertices.push_back(vertexJson({-3.0F, 0.0F, 3.0F}, {0.0F, 1.0F, 0.0F}, {0.0F, 1.0F}));
    floorVertices.push_back(vertexJson({3.0F, 0.0F, 3.0F}, {0.0F, 1.0F, 0.0F}, {1.0F, 1.0F}));
    floorVertices.push_back(vertexJson({3.0F, 0.0F, -3.0F}, {0.0F, 1.0F, 0.0F}, {1.0F, 0.0F}));
    appendIndex(floorIndices, 0);
    appendIndex(floorIndices, 1);
    appendIndex(floorIndices, 2);
    appendIndex(floorIndices, 0);
    appendIndex(floorIndices, 2);
    appendIndex(floorIndices, 3);

    Array meshes;
    meshes.push_back(meshJson("body_cylinder", std::move(bodyVertices), std::move(bodyIndices)));
    meshes.push_back(meshJson("head_sphere", std::move(headVertices), std::move(headIndices)));
    meshes.push_back(meshJson("aura_shell", std::move(auraVertices), std::move(auraIndices)));
    meshes.push_back(meshJson("floor_plane", std::move(floorVertices), std::move(floorIndices)));

    Array materials;
    materials.push_back(materialJson("torso_ceramic", {0.10F, 0.38F, 0.52F},
                                     {0.55F, 0.82F, 0.95F}, 72.0, 1.0, true, 3.0F, 0.38F,
                                     0.4F));
    materials.push_back(materialJson("head_ceramic", {0.78F, 0.55F, 0.34F},
                                     {0.75F, 0.62F, 0.45F}, 36.0, 1.0, true, 2.0F, 0.20F,
                                     1.7F));
    materials.push_back(materialJson("spectral_aura", {0.03F, 0.62F, 0.88F},
                                     {0.40F, 0.95F, 1.0F}, 96.0, 0.34, true, 4.0F, 0.52F,
                                     2.1F));
    materials.push_back(materialJson("floor_slate", {0.10F, 0.11F, 0.13F},
                                     {0.22F, 0.24F, 0.28F}, 20.0, 1.0, true, 6.0F, 0.13F,
                                     0.2F));

    Array objects;
    objects.push_back(objectJson("torso", 0, 0, Mat4::identity()));
    objects.push_back(objectJson(
        "head", 1, 1, translation({0.0F, 1.71F, 0.02F}) * scale({1.0F, 1.05F, 1.0F})));
    objects.push_back(objectJson("aura", 2, 2, Mat4::identity()));
    objects.push_back(objectJson("floor", 3, 3, Mat4::identity()));

    Array pointLights;
    const auto pointLight = [](Vec3 position, Vec3 color, double intensity) {
        Object light;
        light.emplace("position", vec3(position));
        light.emplace("color", vec3(color));
        light.emplace("intensity", number(intensity));
        return JsonValue(std::move(light));
    };
    pointLights.push_back(pointLight({-1.8F, 1.65F, 1.6F}, {0.15F, 0.8F, 1.0F}, 10.0));
    pointLights.push_back(pointLight({2.0F, 1.35F, 0.4F}, {1.0F, 0.18F, 0.52F}, 8.0));
    pointLights.push_back(pointLight({0.0F, 0.85F, -1.8F}, {1.0F, 0.48F, 0.12F}, 12.0));

    Object shadow;
    shadow.emplace("enabled", JsonValue(true));
    shadow.emplace("mapSize", number(512));
    shadow.emplace("depthBias", number(0.0015));
    shadow.emplace("normalBias", number(0.035));
    shadow.emplace("pcfRadius", number(1));

    Object directional;
    directional.emplace("direction", vec3(0.42, 0.88, 0.55));
    directional.emplace("color", vec3(1.0, 0.88, 0.70));
    directional.emplace("intensity", number(1.7));
    directional.emplace("shadow", JsonValue(std::move(shadow)));

    Object camera;
    camera.emplace("position", vec3(3.15, 2.25, 4.85));
    camera.emplace("target", vec3(0.0, 1.0, 0.0));
    camera.emplace("up", vec3(0.0, 1.0, 0.0));
    camera.emplace("fovYDegrees", number(43.0));
    camera.emplace("near", number(0.1));
    camera.emplace("far", number(40.0));

    Object viewport;
    viewport.emplace("x", number(0));
    viewport.emplace("y", number(0));
    viewport.emplace("width", number(320));
    viewport.emplace("height", number(320));

    Object image;
    image.emplace("width", number(320));
    image.emplace("height", number(320));
    image.emplace("viewport", JsonValue(std::move(viewport)));

    Object transparency;
    transparency.emplace("depthWriteMaxOpacity", number(0.5));

    Object limits;
    limits.emplace("maxMemoryBytes", number(536870912.0));

    Object root;
    root.emplace("version", number(1));
    root.emplace("image", JsonValue(std::move(image)));
    root.emplace("camera", JsonValue(std::move(camera)));
    root.emplace("ambientColor", vec3(0.035, 0.045, 0.065));
    root.emplace("background", vec3(0.005, 0.008, 0.015));
    root.emplace("directionalLight", JsonValue(std::move(directional)));
    root.emplace("pointLights", JsonValue(std::move(pointLights)));
    root.emplace("materials", JsonValue(std::move(materials)));
    root.emplace("meshes", JsonValue(std::move(meshes)));
    root.emplace("objects", JsonValue(std::move(objects)));
    root.emplace("transparency", JsonValue(std::move(transparency)));
    root.emplace("limits", JsonValue(std::move(limits)));
    return writeJson(JsonValue(std::move(root)), true);
}

}  // namespace softraster
