#pragma once

#include "math.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace rigskin {

struct Keyframe {
    double time = 0.0;
    Vec3 translation;
    Quat rotation;
};

struct Bone {
    std::string name;
    int parent = -1;
    Vec3 bindTranslation;
    Quat bindRotation;
    std::vector<Keyframe> keyframes;
};

struct SkinWeight {
    std::uint32_t bone = 0;
    double weight = 0.0;
};

struct MeshVertex {
    Vec3 bindPosition;
    Vec3 bindNormal;
    std::vector<SkinWeight> weights;
};

struct Triangle {
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    std::uint32_t c = 0;
};

struct Model {
    std::string sampleName;
    double duration = 0.0;
    std::vector<Bone> bones;
    std::vector<MeshVertex> vertices;
    std::vector<Triangle> triangles;
    std::vector<int> evalOrder;
    std::vector<Mat4> bindGlobals;
    std::vector<std::uint32_t> zeroWeightVertices;
    Bounds bindBounds;
};

struct PipelineConfig {
    int frameCount = 24;
    int width = 640;
    int height = 480;
};

std::string generateAsciiSample(const std::string& sampleName);
Model parseAsciiModel(const std::string& text);
void finalizeModel(Model& model);

std::vector<Mat4> evaluatePoseGlobals(
    const Model& model,
    double time,
    std::vector<Mat4>* localTransforms = nullptr);

double frameTime(double duration, int frameIndex, int frameCount);
std::string pipelineDescriptor(const PipelineConfig& config);

}  // namespace rigskin

