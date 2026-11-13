#pragma once

#include "terrain/cli.hpp"
#include "terrain/ppm.hpp"
#include "terrain/render_data.hpp"
#include "terrain/terrain.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace terrain {

class GlRenderer {
public:
    GlRenderer(const Terrain& terrain, int width, int height);
    ~GlRenderer();

    bool initialize(std::string& error);
    bool render(
        const SceneFrame& frame,
        DebugView view,
        std::string& error);
    bool capturePpm(const std::string& filename, std::string& error);
    void presentToDefaultFramebuffer() const;
    void shutdown();

    int width() const { return width_; }
    int height() const { return height_; }

private:
    struct Batch {
        int nodeOffset = 0;
        int nodeCount = 0;
    };

    bool createShaders(std::string& error);
    bool createStaticBuffers(std::string& error);
    bool createFramebuffer(std::string& error);
    void buildGpuMetadata(const SceneFrame& frame, DebugView view);
    void issueMeshDraws(const SceneFrame& frame, DebugView view);
    void issueSeamOverlay(const SceneFrame& frame);

    const Terrain& terrain_;
    int width_ = 0;
    int height_ = 0;
    bool initialized_ = false;
    uint64_t terrainVersion_ = 0;

    unsigned int terrainProgram_ = 0;
    unsigned int lineProgram_ = 0;
    unsigned int vao_ = 0;
    unsigned int indexBuffer_ = 0;
    unsigned int heightTexture_ = 0;
    unsigned int nodeBuffer_ = 0;
    unsigned int seamVertexBuffer_ = 0;
    unsigned int colorTexture_ = 0;
    unsigned int depthRenderbuffer_ = 0;
    unsigned int framebuffer_ = 0;

    int viewProjectionLocation_ = -1;
    int heightMapLocation_ = -1;
    int extentLocation_ = -1;
    int gridLocation_ = -1;
    int nodeOffsetLocation_ = -1;
    int viewModeLocation_ = -1;
    int amplitudeLocation_ = -1;
    int lineViewProjectionLocation_ = -1;
    int lineBiasLocation_ = -1;

    std::vector<uint8_t> readback_;
    std::vector<unsigned int> gpuNodesWords_;
    std::vector<float> seamVertices_;
    std::vector<Batch> batches_;
};

}  // namespace terrain

