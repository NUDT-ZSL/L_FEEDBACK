#pragma once

#include "types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dr {

struct Vertex {
    float position[3];
    float normal[3];
};

struct CpuMesh {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct RenderObject {
    std::size_t meshIndex = 0;
    Mat4 model = Mat4::identity();
    Material material;
};

struct SceneAssets {
    std::vector<CpuMesh> meshes;
    std::vector<RenderObject> objects;
    Mat4 viewProjection = Mat4::identity();
};

bool loadScene(const std::string& path, Scene& scene, std::string& error);
SceneAssets buildSceneAssets(const Scene& scene);
LightSet lightSetFromScene(const Scene& scene);
bool validateLights(const std::vector<Light>& lights, std::string& error);

} // namespace dr
