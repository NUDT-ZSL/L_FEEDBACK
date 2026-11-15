#include "scene.hpp"

#include "error.hpp"

#include <array>

namespace scene {
namespace {

class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) : state_(seed) {}

    uint64_t next() {
        uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    float unit() {
        return static_cast<float>(next() >> 40) * (1.0f / 16777216.0f);
    }

    float range(float minimum, float maximum) {
        return minimum + (maximum - minimum) * unit();
    }

    int rangeInt(int minimum, int maximumInclusive) {
        return minimum + static_cast<int>(next() %
               static_cast<uint64_t>(maximumInclusive - minimum + 1));
    }

private:
    uint64_t state_;
};

void appendVertex(std::vector<float>& data, Vec3 position, Vec3 normal) {
    data.push_back(position.x);
    data.push_back(position.y);
    data.push_back(position.z);
    data.push_back(normal.x);
    data.push_back(normal.y);
    data.push_back(normal.z);
}

Mat4 composeModel(Vec3 position, Vec3 rotation, Vec3 scaleValue) {
    return translate(position) * rotateY(rotation.y) * rotateX(rotation.x) *
           rotateZ(rotation.z) * scale(scaleValue);
}

} // namespace

Mesh::~Mesh() {
    release();
}

Mesh::Mesh(Mesh&& other) noexcept
    : vao_(other.vao_), vbo_(other.vbo_), ebo_(other.ebo_),
      indexCount_(other.indexCount_) {
    other.vao_ = 0;
    other.vbo_ = 0;
    other.ebo_ = 0;
    other.indexCount_ = 0;
}

Mesh& Mesh::operator=(Mesh&& other) noexcept {
    if (this != &other) {
        release();
        vao_ = other.vao_;
        vbo_ = other.vbo_;
        ebo_ = other.ebo_;
        indexCount_ = other.indexCount_;
        other.vao_ = 0;
        other.vbo_ = 0;
        other.ebo_ = 0;
        other.indexCount_ = 0;
    }
    return *this;
}

void Mesh::upload(const std::vector<float>& vertices,
                  const std::vector<uint32_t>& indices) {
    release();
    if (vertices.empty() || vertices.size() % 6 != 0 || indices.empty()) {
        throw AppError(ErrorType::Internal, "invalid CPU mesh data");
    }

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glGenBuffers(1, &ebo_);
    if (vao_ == 0 || vbo_ == 0 || ebo_ == 0) {
        release();
        throw AppError(ErrorType::GlRuntime, "failed to allocate mesh buffers");
    }

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
                 vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(indices.size() * sizeof(uint32_t)),
                 indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          reinterpret_cast<const void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          reinterpret_cast<const void*>(3 * sizeof(float)));
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    indexCount_ = static_cast<GLsizei>(indices.size());
    checkGlError("uploading mesh");
}

void Mesh::draw() const {
    glBindVertexArray(vao_);
    glDrawElements(GL_TRIANGLES, indexCount_, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

void Mesh::release() {
    if (ebo_ != 0) glDeleteBuffers(1, &ebo_);
    if (vbo_ != 0) glDeleteBuffers(1, &vbo_);
    if (vao_ != 0) glDeleteVertexArrays(1, &vao_);
    ebo_ = 0;
    vbo_ = 0;
    vao_ = 0;
    indexCount_ = 0;
}

Scene::Scene(uint64_t seed, bool includeLights) : seed_(seed) {
    buildMeshes();
    generateObjects();
    if (includeLights) {
        generateLights();
    }
    buildCamera();
    buildShadowCamera();
    checkGlError("constructing scene");
}

std::vector<float> Scene::cubeVertices() {
    const std::array<Vec3, 6> normals = {
        Vec3{1, 0, 0}, Vec3{-1, 0, 0}, Vec3{0, 1, 0},
        Vec3{0, -1, 0}, Vec3{0, 0, 1}, Vec3{0, 0, -1}
    };
    const std::array<std::array<Vec3, 4>, 6> corners = {{
        {{Vec3{1, -1, -1}, Vec3{1, 1, -1}, Vec3{1, 1, 1}, Vec3{1, -1, 1}}},
        {{Vec3{-1, -1, 1}, Vec3{-1, 1, 1}, Vec3{-1, 1, -1}, Vec3{-1, -1, -1}}},
        {{Vec3{-1, 1, -1}, Vec3{-1, 1, 1}, Vec3{1, 1, 1}, Vec3{1, 1, -1}}},
        {{Vec3{-1, -1, 1}, Vec3{-1, -1, -1}, Vec3{1, -1, -1}, Vec3{1, -1, 1}}},
        {{Vec3{-1, -1, 1}, Vec3{1, -1, 1}, Vec3{1, 1, 1}, Vec3{-1, 1, 1}}},
        {{Vec3{1, -1, -1}, Vec3{-1, -1, -1}, Vec3{-1, 1, -1}, Vec3{1, 1, -1}}}
    }};

    std::vector<float> vertices;
    vertices.reserve(24 * 6);
    for (size_t face = 0; face < 6; ++face) {
        for (const Vec3& corner : corners[face]) {
            appendVertex(vertices, corner, normals[face]);
        }
    }
    return vertices;
}

std::vector<uint32_t> Scene::cubeIndices() {
    std::vector<uint32_t> indices;
    indices.reserve(36);
    for (uint32_t face = 0; face < 6; ++face) {
        const uint32_t base = face * 4;
        indices.insert(indices.end(),
                       {base, base + 1, base + 2, base, base + 2, base + 3});
    }
    return indices;
}

std::vector<float> Scene::sphereVertices() {
    constexpr int latitudeSegments = 24;
    constexpr int longitudeSegments = 48;
    std::vector<float> vertices;
    vertices.reserve(static_cast<size_t>((latitudeSegments + 1) *
                                         (longitudeSegments + 1) * 6));
    for (int latitude = 0; latitude <= latitudeSegments; ++latitude) {
        const float v = static_cast<float>(latitude) /
                        static_cast<float>(latitudeSegments);
        const float theta = v * kPi;
        const float sinTheta = std::sin(theta);
        const float cosTheta = std::cos(theta);
        for (int longitude = 0; longitude <= longitudeSegments; ++longitude) {
            const float u = static_cast<float>(longitude) /
                            static_cast<float>(longitudeSegments);
            const float phi = u * 2.0f * kPi;
            const Vec3 normal{sinTheta * std::cos(phi), cosTheta,
                              sinTheta * std::sin(phi)};
            appendVertex(vertices, normal, normal);
        }
    }
    return vertices;
}

std::vector<uint32_t> Scene::sphereIndices() {
    constexpr int latitudeSegments = 24;
    constexpr int longitudeSegments = 48;
    std::vector<uint32_t> indices;
    indices.reserve(static_cast<size_t>(latitudeSegments * longitudeSegments * 6));
    const uint32_t stride = longitudeSegments + 1;
    for (int latitude = 0; latitude < latitudeSegments; ++latitude) {
        for (int longitude = 0; longitude < longitudeSegments; ++longitude) {
            const uint32_t a = static_cast<uint32_t>(latitude) * stride +
                               static_cast<uint32_t>(longitude);
            const uint32_t b = a + stride;
            indices.insert(indices.end(), {a, b, a + 1, a + 1, b, b + 1});
        }
    }
    return indices;
}

std::vector<float> Scene::planeVertices() {
    std::vector<float> vertices;
    appendVertex(vertices, {-1, 0, -1}, {0, 1, 0});
    appendVertex(vertices, {1, 0, -1}, {0, 1, 0});
    appendVertex(vertices, {1, 0, 1}, {0, 1, 0});
    appendVertex(vertices, {-1, 0, 1}, {0, 1, 0});
    return vertices;
}

std::vector<uint32_t> Scene::planeIndices() {
    return {0, 1, 2, 0, 2, 3};
}

void Scene::buildMeshes() {
    meshes_.reserve(3);
    meshes_.emplace_back();
    meshes_.back().upload(cubeVertices(), cubeIndices());
    meshes_.emplace_back();
    meshes_.back().upload(sphereVertices(), sphereIndices());
    meshes_.emplace_back();
    meshes_.back().upload(planeVertices(), planeIndices());
}

void Scene::generateObjects() {
    SplitMix64 random(seed_ ^ 0xd1b54a32d192ed03ULL);
    const std::array<Vec3, 8> palette = {
        Vec3{0.72f, 0.20f, 0.14f},
        Vec3{0.16f, 0.43f, 0.72f},
        Vec3{0.78f, 0.56f, 0.16f},
        Vec3{0.15f, 0.62f, 0.35f},
        Vec3{0.62f, 0.28f, 0.72f},
        Vec3{0.78f, 0.76f, 0.70f},
        Vec3{0.14f, 0.66f, 0.70f},
        Vec3{0.66f, 0.34f, 0.18f}
    };

    objects_.push_back(SceneObject{
        MeshType::Plane,
        composeModel({0.0f, 0.0f, 0.3f}, {0.0f, 0.0f, 0.0f}, {8.8f, 1.0f, 8.0f}),
        upperLeft3(composeModel({0.0f, 0.0f, 0.3f}, {0.0f, 0.0f, 0.0f},
                                {8.8f, 1.0f, 8.0f})),
        Material{Vec3{0.16f, 0.18f, 0.20f}, 0.72f, 0.05f}
    });

    constexpr int columns = 4;
    constexpr int rows = 3;
    for (int index = 0; index < columns * rows; ++index) {
        const int column = index % columns;
        const int row = index / columns;
        const float size = random.range(0.42f, 0.78f);
        const Vec3 position{
            -4.05f + static_cast<float>(column) * 2.65f + random.range(-0.22f, 0.22f),
            size * random.range(1.0f, 1.45f),
            -2.55f + static_cast<float>(row) * 2.55f + random.range(-0.22f, 0.22f)
        };
        const Vec3 rotation{
            random.range(-0.55f, 0.55f),
            random.range(0.0f, kPi * 2.0f),
            random.range(-0.35f, 0.35f)
        };
        const MeshType mesh = index == 0
                            ? MeshType::Cube
                            : (index == 1
                                   ? MeshType::Sphere
                                   : (random.rangeInt(0, 4) == 0
                                          ? MeshType::Sphere
                                          : MeshType::Cube));
        const Vec3 scaleValue = mesh == MeshType::Sphere
                              ? Vec3{size, size, size}
                              : Vec3{size, size * random.range(0.8f, 1.35f), size};
        const Mat4 model = composeModel(position, rotation, scaleValue);

        Vec3 albedo = palette[static_cast<size_t>(index) % palette.size()];
        albedo.x = std::clamp(albedo.x * random.range(0.88f, 1.08f), 0.02f, 0.95f);
        albedo.y = std::clamp(albedo.y * random.range(0.88f, 1.08f), 0.02f, 0.95f);
        albedo.z = std::clamp(albedo.z * random.range(0.88f, 1.08f), 0.02f, 0.95f);

        const bool metallicObject = (index % 3 == 0) || random.rangeInt(0, 2) == 0;
        Material material{
            albedo,
            random.range(0.12f, metallicObject ? 0.42f : 0.88f),
            metallicObject ? random.range(0.55f, 0.95f) : random.range(0.0f, 0.22f)
        };
        objects_.push_back(SceneObject{mesh, model, upperLeft3(model), material});
    }
}

void Scene::generateLights() {
    SplitMix64 random(seed_ ^ 0xa0761d6478bd642fULL);

    const Vec3 directionalDirection = normalize(Vec3{
        random.range(-0.48f, 0.48f),
        -1.0f,
        random.range(-0.36f, 0.52f)
    });
    lights_.push_back(Light{
        LightType::Directional,
        {},
        directionalDirection,
        Vec3{1.0f, random.range(0.82f, 0.95f), random.range(0.68f, 0.84f)},
        random.range(2.1f, 2.8f),
        1.0f,
        0.0f
    });
    shadowLightIndex_ = 0;

    for (int i = 0; i < 2; ++i) {
        const Vec3 color = i == 0
            ? Vec3{random.range(0.18f, 0.35f), random.range(0.48f, 0.72f), 1.0f}
            : Vec3{1.0f, random.range(0.28f, 0.48f), random.range(0.12f, 0.30f)};
        lights_.push_back(Light{
            LightType::Point,
            Vec3{random.range(-4.5f, 4.5f), random.range(2.0f, 3.6f),
                 random.range(-2.8f, 3.2f)},
            {0.0f, -1.0f, 0.0f},
            color,
            random.range(13.0f, 22.0f),
            1.0f,
            0.0f
        });
    }

    const Vec3 spotTarget{random.range(-1.4f, 1.4f), 0.35f,
                          random.range(-0.8f, 1.2f)};
    const Vec3 spotPosition{random.range(-1.0f, 1.0f), 5.0f,
                            random.range(4.3f, 5.5f)};
    lights_.push_back(Light{
        LightType::Spot,
        spotPosition,
        normalize(spotTarget - spotPosition),
        Vec3{1.0f, random.range(0.88f, 0.97f), random.range(0.78f, 0.9f)},
        random.range(24.0f, 34.0f),
        std::cos(10.0f * kPi / 180.0f),
        std::cos(25.0f * kPi / 180.0f)
    });
}

void Scene::buildCamera() {
    const Vec3 target{0.0f, 1.15f, 0.25f};
    view_ = lookAt(cameraPosition_, target, {0.0f, 1.0f, 0.0f});
    projection_ = projectionForAspect(16.0f / 9.0f);
}

void Scene::buildShadowCamera() {
    if (shadowLightIndex_ < 0 ||
        shadowLightIndex_ >= static_cast<int>(lights_.size())) {
        lightViewProjection_ = Mat4::identity();
        return;
    }
    const Light& light = lights_[static_cast<size_t>(shadowLightIndex_)];
    const Vec3 center{0.0f, 0.9f, 0.25f};
    const Vec3 eye = center - normalize(light.direction) * 18.0f;
    Vec3 up{0.0f, 1.0f, 0.0f};
    if (std::abs(dot(normalize(light.direction), up)) > 0.98f) {
        up = {0.0f, 0.0f, 1.0f};
    }
    const Mat4 lightView = lookAt(eye, center, up);
    const Mat4 lightProjection = orthographic(-9.5f, 9.5f, -9.5f, 9.5f,
                                               0.1f, 40.0f);
    lightViewProjection_ = lightProjection * lightView;
}

} // namespace scene

