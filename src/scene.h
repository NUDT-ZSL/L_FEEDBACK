#pragma once

#include "core.h"
#include "json.h"

#include <cstdint>
#include <string>
#include <vector>

struct Camera {
    Vec3 position;
    Vec3 target;
    Vec3 up;
    double fovYDeg = 55.0;
};

struct HeightfieldParams {
    int resolutionX = 72;
    int resolutionZ = 72;
    double minX = -32.0;
    double maxX = 32.0;
    double minZ = -40.0;
    double maxZ = 38.0;
    double heightMin = 0.8;
    double heightMax = 14.0;
    double wallStart = 5.0;
    double wallEnd = 24.0;
    int noiseOctaves = 4;
    double erosionStrength = 1.8;
};

struct MaterialLayer {
    std::string name;
    Vec3 color;
    Vec3 slopeTint;
    double heightStart = 0.0;
    double heightEnd = 1.0;
};

enum class LightType { Directional, Point };

struct Light {
    LightType type = LightType::Directional;
    Vec3 direction;
    Vec3 position;
    Vec3 color{1.0, 1.0, 1.0};
    double intensity = 1.0;
    double range = 100.0;
};

struct Transform {
    Vec3 translation;
    Vec3 rotationDeg;
    Vec3 scale{1.0, 1.0, 1.0};
};

struct RockMaterial {
    Vec3 baseColor{0.55, 0.42, 0.30};
    double roughness = 0.8;
};

struct MeshParams {
    double radius = 1.0;
    int subdivisions = 1;
};

struct RockLocalMesh {
    std::vector<Vec3> vertices;
    std::vector<int> indices;
};

struct TerrainTriangle {
    Vec3 p[3];
    Vec3 n[3];
    uint32_t primitiveId = 0;
};

struct RockTriangle {
    Vec3 p[3];
    Vec3 normal;
    uint32_t primitiveId = 0;
};

struct RockNode {
    std::string id;
    int parent = -1;
    std::vector<int> children;
    Transform transform;
    RockMaterial material;
    MeshParams mesh;
    RockLocalMesh localMesh;
    Mat4 world = Mat4::identity();
    std::vector<RockTriangle> worldTriangles;
};

enum class ChangeKind { Transform, Material };

struct TransformPatch {
    bool hasTranslation = false;
    bool hasRotation = false;
    bool hasScale = false;
    Vec3 translation;
    Vec3 rotationDeg;
    Vec3 scale;
};

struct MaterialPatch {
    bool hasBaseColor = false;
    bool hasRoughness = false;
    Vec3 baseColor;
    double roughness = 0.0;
};

struct Change {
    int target = -1;
    ChangeKind kind = ChangeKind::Transform;
    TransformPatch transform;
    MaterialPatch material;
};

struct Scene {
    uint32_t seed = 0;
    int width = 640;
    int height = 360;
    int tileSize = 16;
    int depthBits = 24;
    int stencilBits = 8;
    double nearPlane = 0.1;
    double farPlane = 500.0;
    double shadowExtrusion = 240.0;
    uint32_t shadowDepthBiasUnits = 8;
    Camera camera;
    HeightfieldParams heightfield;
    std::vector<MaterialLayer> layers;
    Vec3 ambient{0.16, 0.18, 0.21};
    std::vector<Light> lights;
    std::vector<RockNode> rocks;
    std::vector<TerrainTriangle> terrainTriangles;
    std::vector<Change> changes;

    uint32_t depthMax() const;
    Mat4 viewMatrix() const;
    Mat4 projectionMatrix() const;
    Mat4 viewProjectionMatrix() const;
    std::vector<int> subtree(int rootIndex) const;
};

Scene loadScene(const std::string& path);
void rebuildWorldTransforms(Scene& scene);
void rebuildRockGeometry(Scene& scene, int rockIndex);
void rebuildAllRockGeometry(Scene& scene);
void applyChange(Scene& scene, const Change& change);
void applyAllChanges(Scene& scene);

std::string changeKindName(ChangeKind kind);
