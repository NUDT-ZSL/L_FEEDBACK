#pragma once

#include "renderer/math.hpp"
#include "renderer/scene.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace renderer {

class Logger;

struct ResourceCounts {
    int shaderPrograms = 0;
    int textures = 0;
    int framebuffers = 0;
    int renderbuffers = 0;
    int buffers = 0;
    int vertexArrays = 0;
    int queries = 0;

    int total() const {
        return shaderPrograms + textures + framebuffers + renderbuffers
             + buffers + vertexArrays + queries;
    }
};

struct ShadowIntensityStats {
    double minimum = 0.0;
    double mean = 0.0;
    double maximum = 0.0;
    std::size_t geometryPixels = 0;
    std::size_t shadowedPixels = 0;
};

struct LightAtlasInfo {
    int lightIndex = 0;
    std::string type;
    bool castShadow = false;
    int firstSlot = -1;
    int faceCount = 0;
    std::vector<int> slots;
};

struct RenderStats {
    double geometryGpuMs = 0.0;
    double shadowGpuMs = 0.0;
    double lightingGpuMs = 0.0;
    double nonBackgroundPercent = 0.0;
    std::size_t nonBackgroundPixels = 0;
    std::size_t totalPixels = 0;
    int shadowSamples = 0;
    int postprocessIterations = 0;
    bool gaussianFilterApplied = false;
    int tileResolution = 0;
    int atlasWidth = 0;
    int atlasHeight = 0;
    ShadowIntensityStats shadowBefore;
    ShadowIntensityStats shadowAfter;
    std::vector<LightAtlasInfo> lights;
};

class Renderer {
public:
    Renderer() = default;
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;

    void initialize(const Scene& scene, const std::string& mode, Logger& logger);
    RenderStats render(const Scene& scene, const std::string& mode,
                       const std::string& outputPath, Logger& logger);
    void shutdown();

    bool initialized() const {
        return initialized_;
    }

    const ResourceCounts& resource_counts() const {
        return resourceCounts_;
    }

private:
    struct SlotPlan {
        int lightIndex = 0;
        int faceIndex = 0;
        int column = 0;
        int row = 0;
        Mat4 viewProjection;
    };

    struct LightPlan {
        int firstSlot = -1;
        int faceCount = 0;
        std::vector<int> slots;
        std::vector<int> slotColumns;
        std::vector<int> slotRows;
        std::vector<Mat4> slotMatrices;
    };

    struct MeshGpu {
        unsigned int vao = 0;
        unsigned int vertexBuffer = 0;
        unsigned int indexBuffer = 0;
        int indexCount = 0;
        int materialIndex = 0;
    };

    void build_pipeline(const Scene& scene, const std::string& mode, Logger& logger);
    void build_shadow_plan(const Scene& scene);
    void create_meshes(const Scene& scene);
    void create_programs();
    void create_targets(const Scene& scene);

    unsigned int create_texture_2d(int width, int height, unsigned int internalFormat,
                                   unsigned int format, unsigned int type,
                                   unsigned int minFilter, unsigned int magFilter);
    unsigned int create_framebuffer();
    unsigned int create_renderbuffer();
    unsigned int create_buffer();
    unsigned int create_vertex_array();
    unsigned int create_query();
    unsigned int create_program(const char* vertexSource, const char* fragmentSource,
                                const char* label);
    void delete_texture(unsigned int& texture);
    void delete_framebuffer(unsigned int& framebuffer);
    void delete_renderbuffer(unsigned int& renderbuffer);
    void delete_buffer(unsigned int& buffer);
    void delete_vertex_array(unsigned int& vertexArray);
    void delete_query(unsigned int& query);
    void delete_program(unsigned int& program);

    void log_light_plan(const Scene& scene, Logger& logger) const;
    void upload_lights(const Scene& scene);
    void upload_material(const Scene& scene, const MeshGpu& mesh);
    void draw_meshes_geometry(const Scene& scene);
    void draw_meshes_depth(const Scene& scene);
    void draw_fullscreen() const;
    void render_shadow_atlas(const Scene& scene);
    void render_geometry_pass(const Scene& scene);
    void render_lighting_pass(const Scene& scene);
    void render_denoise(const Scene& scene, unsigned int& finalVisibility);
    void render_composite_and_tonemap(const Scene& scene, unsigned int finalVisibility);
    void write_png(const std::string& outputPath);
    void check_gl(const char* stage) const;

    bool initialized_ = false;
    ResourceCounts resourceCounts_;
    int width_ = 0;
    int height_ = 0;
    int atlasWidth_ = 0;
    int atlasHeight_ = 0;
    int tileResolution_ = 0;
    int atlasColumns_ = 0;
    int atlasRows_ = 0;
    int pcfKernel_ = 0;
    float pcfStep_ = 0.0f;
    int shadowSamples_ = 0;
    int postprocessIterations_ = 0;
    bool gaussianFilter_ = false;
    int filterRadius_ = 0;
    std::vector<float> filterWeights_;

    std::vector<MeshGpu> meshes_;
    std::vector<SlotPlan> slots_;
    std::vector<LightPlan> lightPlans_;

    unsigned int fullscreenVao_ = 0;
    unsigned int geometryFbo_ = 0;
    unsigned int geometryDepthRbo_ = 0;
    unsigned int gPosition_ = 0;
    unsigned int gNormal_ = 0;
    unsigned int gAlbedo_ = 0;
    unsigned int gMaterial_ = 0;

    unsigned int shadowFbo_ = 0;
    unsigned int shadowAtlas_ = 0;

    unsigned int lightFbo_ = 0;
    unsigned int hdrColor_ = 0;
    unsigned int shadowVisibility_ = 0;
    unsigned int filterScratchA_ = 0;
    unsigned int filterScratchB_ = 0;
    unsigned int denoiseFbo_ = 0;

    unsigned int compositeFbo_ = 0;
    unsigned int hdrFinal_ = 0;
    unsigned int tonemapFbo_ = 0;
    unsigned int outputTexture_ = 0;

    unsigned int geometryProgram_ = 0;
    unsigned int shadowProgram_ = 0;
    unsigned int lightingProgram_ = 0;
    unsigned int filterProgram_ = 0;
    unsigned int compositeProgram_ = 0;
    unsigned int tonemapProgram_ = 0;

    unsigned int geometryQuery_ = 0;
    unsigned int shadowQuery_ = 0;
    unsigned int lightingQuery_ = 0;
};

}  // namespace renderer
