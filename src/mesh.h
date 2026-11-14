#pragma once

#include <glad/gl.h>

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

struct Vertex {
    glm::vec3 position{};
    glm::vec3 normal{};
};

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

class GpuMesh {
public:
    GpuMesh() = default;
    ~GpuMesh();

    GpuMesh(const GpuMesh&) = delete;
    GpuMesh& operator=(const GpuMesh&) = delete;
    GpuMesh(GpuMesh&& other) noexcept;
    GpuMesh& operator=(GpuMesh&& other) noexcept;

    void create(const MeshData& data);
    void destroy();
    void draw() const;

    GLsizei indexCount() const { return indexCount_; }

private:
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    GLuint ebo_ = 0;
    GLsizei indexCount_ = 0;
};

MeshData makeCubeMesh();
MeshData makePlaneMesh(float halfExtent, int subdivisions);
MeshData makeSphereMesh(int longitudeSegments, int latitudeSegments);
MeshData makeCylinderMesh(int segments, float halfHeight, float radius);

