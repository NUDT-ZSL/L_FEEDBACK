#include "mesh.h"

#include <cmath>

#include <glm/gtc/constants.hpp>

#include "gl_utils.h"

GpuMesh::~GpuMesh() {
    destroy();
}

GpuMesh::GpuMesh(GpuMesh&& other) noexcept
    : vao_(other.vao_), vbo_(other.vbo_), ebo_(other.ebo_), indexCount_(other.indexCount_) {
    other.vao_ = 0;
    other.vbo_ = 0;
    other.ebo_ = 0;
    other.indexCount_ = 0;
}

GpuMesh& GpuMesh::operator=(GpuMesh&& other) noexcept {
    if (this != &other) {
        destroy();
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

void GpuMesh::create(const MeshData& data) {
    destroy();

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glGenBuffers(1, &ebo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(data.vertices.size() * sizeof(Vertex)),
                 data.vertices.data(),
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(data.indices.size() * sizeof(uint32_t)),
                 data.indices.data(),
                 GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, normal)));

    glBindVertexArray(0);
    indexCount_ = static_cast<GLsizei>(data.indices.size());
    checkGLError("GpuMesh::create");
}

void GpuMesh::destroy() {
    if (ebo_ != 0) {
        glDeleteBuffers(1, &ebo_);
        ebo_ = 0;
    }
    if (vbo_ != 0) {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    if (vao_ != 0) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    indexCount_ = 0;
}

void GpuMesh::draw() const {
    glBindVertexArray(vao_);
    glDrawElements(GL_TRIANGLES, indexCount_, GL_UNSIGNED_INT, nullptr);
}

MeshData makeCubeMesh() {
    MeshData data;
    const glm::vec3 faceNormals[6] = {
        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}
    };
    const glm::vec3 tangents[6][2] = {
        {{0, 0, 1}, {0, 1, 0}},
        {{0, 0, -1}, {0, 1, 0}},
        {{1, 0, 0}, {0, 0, 1}},
        {{1, 0, 0}, {0, 0, -1}},
        {{-1, 0, 0}, {0, 1, 0}},
        {{1, 0, 0}, {0, 1, 0}}
    };

    for (int face = 0; face < 6; ++face) {
        const glm::vec3 n = faceNormals[face];
        const glm::vec3 t = tangents[face][0];
        const glm::vec3 b = tangents[face][1];
        for (int i = 0; i < 4; ++i) {
            const float u = (i == 1 || i == 2) ? 1.0f : -1.0f;
            const float v = (i >= 2) ? 1.0f : -1.0f;
            Vertex vertex;
            vertex.position = n + t * u + b * v;
            vertex.normal = n;
            data.vertices.push_back(vertex);
        }
        const uint32_t base = static_cast<uint32_t>(face * 4);
        data.indices.insert(data.indices.end(), {
            base + 0, base + 2, base + 1,
            base + 0, base + 3, base + 2
        });
    }
    return data;
}

MeshData makePlaneMesh(float halfExtent, int subdivisions) {
    MeshData data;
    subdivisions = glm::max(subdivisions, 1);
    const int vertexSide = subdivisions + 1;
    for (int z = 0; z <= subdivisions; ++z) {
        for (int x = 0; x <= subdivisions; ++x) {
            const float fx = static_cast<float>(x) / static_cast<float>(subdivisions);
            const float fz = static_cast<float>(z) / static_cast<float>(subdivisions);
            Vertex vertex;
            vertex.position = {glm::mix(-halfExtent, halfExtent, fx), 0.0f,
                               glm::mix(-halfExtent, halfExtent, fz)};
            vertex.normal = {0, 1, 0};
            data.vertices.push_back(vertex);
        }
    }
    for (int z = 0; z < subdivisions; ++z) {
        for (int x = 0; x < subdivisions; ++x) {
            const uint32_t a = static_cast<uint32_t>(z * vertexSide + x);
            const uint32_t b = a + 1;
            const uint32_t c = a + static_cast<uint32_t>(vertexSide);
            const uint32_t d = c + 1;
            data.indices.insert(data.indices.end(), {a, c, b, b, c, d});
        }
    }
    return data;
}

MeshData makeSphereMesh(int longitudeSegments, int latitudeSegments) {
    MeshData data;
    for (int y = 0; y <= latitudeSegments; ++y) {
        const float v = static_cast<float>(y) / static_cast<float>(latitudeSegments);
        const float latitude = v * glm::pi<float>();
        const float sinLat = std::sin(latitude);
        const float cosLat = std::cos(latitude);
        for (int x = 0; x <= longitudeSegments; ++x) {
            const float u = static_cast<float>(x) / static_cast<float>(longitudeSegments);
            const float longitude = u * glm::two_pi<float>();
            Vertex vertex;
            vertex.position = {
                sinLat * std::cos(longitude),
                cosLat,
                sinLat * std::sin(longitude)
            };
            vertex.normal = vertex.position;
            data.vertices.push_back(vertex);
        }
    }
    for (int y = 0; y < latitudeSegments; ++y) {
        for (int x = 0; x < longitudeSegments; ++x) {
            const uint32_t a = static_cast<uint32_t>(y * (longitudeSegments + 1) + x);
            const uint32_t b = a + static_cast<uint32_t>(longitudeSegments + 1);
            const uint32_t c = b + 1;
            const uint32_t d = a + 1;
            if (y != 0) {
                data.indices.insert(data.indices.end(), {a, d, b});
            }
            if (y != latitudeSegments - 1) {
                data.indices.insert(data.indices.end(), {d, c, b});
            }
        }
    }
    return data;
}

MeshData makeCylinderMesh(int segments, float halfHeight, float radius) {
    MeshData data;
    for (int side = 0; side < 2; ++side) {
        const float y = side == 0 ? -halfHeight : halfHeight;
        const float ny = side == 0 ? -1.0f : 1.0f;
        const uint32_t base = static_cast<uint32_t>(data.vertices.size());
        for (int i = 0; i <= segments; ++i) {
            const float angle = static_cast<float>(i) / static_cast<float>(segments) *
                                glm::two_pi<float>();
            Vertex vertex;
            vertex.position = {std::cos(angle) * radius, y, std::sin(angle) * radius};
            vertex.normal = {0, ny, 0};
            data.vertices.push_back(vertex);
        }
        Vertex center;
        center.position = {0, y, 0};
        center.normal = {0, ny, 0};
        const uint32_t centerIndex = static_cast<uint32_t>(data.vertices.size());
        data.vertices.push_back(center);
        for (int i = 0; i < segments; ++i) {
            const uint32_t a = base + static_cast<uint32_t>(i);
            const uint32_t b = a + 1;
            if (side == 0) {
                data.indices.insert(data.indices.end(), {centerIndex, a, b});
            } else {
                data.indices.insert(data.indices.end(), {centerIndex, b, a});
            }
        }
    }

    const uint32_t sideBase = static_cast<uint32_t>(data.vertices.size());
    for (int i = 0; i <= segments; ++i) {
        const float angle = static_cast<float>(i) / static_cast<float>(segments) *
                            glm::two_pi<float>();
        const glm::vec3 n = {std::cos(angle), 0, std::sin(angle)};
        Vertex bottom;
        bottom.position = {n.x * radius, -halfHeight, n.z * radius};
        bottom.normal = n;
        data.vertices.push_back(bottom);
        Vertex top = bottom;
        top.position.y = halfHeight;
        data.vertices.push_back(top);
    }
    for (int i = 0; i < segments; ++i) {
        const uint32_t a = sideBase + static_cast<uint32_t>(i * 2);
        const uint32_t b = a + 1;
        const uint32_t c = a + 2;
        const uint32_t d = a + 3;
        data.indices.insert(data.indices.end(), {a, b, c, b, d, c});
    }
    return data;
}

