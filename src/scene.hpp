#pragma once

#include "gl.hpp"
#include "math.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace scene {

enum class MeshType : int {
    Cube = 0,
    Sphere = 1,
    Plane = 2
};

enum class LightType : int {
    Directional = 0,
    Point = 1,
    Spot = 2
};

struct Material {
    Vec3 albedo{0.7f, 0.7f, 0.7f};
    float roughness = 0.5f;
    float metallic = 0.0f;
};

struct SceneObject {
    MeshType mesh = MeshType::Cube;
    Mat4 model = Mat4::identity();
    Mat3 normalMatrix = Mat3::identity();
    Material material;
};

struct Light {
    LightType type = LightType::Point;
    Vec3 position{};
    Vec3 direction{0.0f, -1.0f, 0.0f};
    Vec3 color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    float innerConeCosine = 1.0f;
    float outerConeCosine = 0.0f;
};

class Mesh {
public:
    Mesh() = default;
    ~Mesh();
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    Mesh(Mesh&& other) noexcept;
    Mesh& operator=(Mesh&& other) noexcept;

    void upload(const std::vector<float>& vertices,
                const std::vector<uint32_t>& indices);
    void draw() const;
    GLuint vao() const noexcept { return vao_; }

private:
    void release();
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    GLuint ebo_ = 0;
    GLsizei indexCount_ = 0;
};

class Scene {
public:
    Scene(uint64_t seed, bool includeLights);
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    uint64_t seed() const noexcept { return seed_; }
    const std::vector<SceneObject>& objects() const noexcept { return objects_; }
    const std::vector<Light>& lights() const noexcept { return lights_; }
    const std::vector<Mesh>& meshes() const noexcept { return meshes_; }

    Vec3 cameraPosition() const noexcept { return cameraPosition_; }
    const Mat4& view() const noexcept { return view_; }
    const Mat4& projection() const noexcept { return projection_; }
    Mat4 projectionForAspect(float aspect) const {
        return perspective(fovYRadians_, aspect, nearPlane_, farPlane_);
    }
    const Mat4& lightViewProjection() const noexcept {
        return lightViewProjection_;
    }
    int shadowLightIndex() const noexcept { return shadowLightIndex_; }

private:
    static std::vector<float> cubeVertices();
    static std::vector<uint32_t> cubeIndices();
    static std::vector<float> sphereVertices();
    static std::vector<uint32_t> sphereIndices();
    static std::vector<float> planeVertices();
    static std::vector<uint32_t> planeIndices();

    void buildMeshes();
    void generateObjects();
    void generateLights();
    void buildCamera();
    void buildShadowCamera();

    uint64_t seed_ = 0;
    std::vector<Mesh> meshes_;
    std::vector<SceneObject> objects_;
    std::vector<Light> lights_;
    Vec3 cameraPosition_{7.4f, 5.6f, 9.4f};
    Mat4 view_ = Mat4::identity();
    Mat4 projection_ = Mat4::identity();
    Mat4 lightViewProjection_ = Mat4::identity();
    int shadowLightIndex_ = -1;
    float fovYRadians_ = 43.0f * kPi / 180.0f;
    float nearPlane_ = 0.08f;
    float farPlane_ = 80.0f;
};

} // namespace scene

