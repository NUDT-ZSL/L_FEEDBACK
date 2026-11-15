#pragma once

#include "environment.h"
#include "image.h"
#include "scene.h"

#include <string>

namespace pbr {

struct RenderSettings {
    int samples = 256;
    int subframes = 8;
    unsigned int seed = 0x105u;
};

class Renderer {
public:
    Renderer();
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool initialize(std::string& error);
    bool render(const Scene& scene,
                const RenderSettings& settings,
                Image& image,
                std::string& error);

private:
    unsigned int program_ = 0;
    unsigned int vao_ = 0;
    unsigned int accumulationTexture_ = 0;
    unsigned int accumulationFbo_ = 0;
    unsigned int auxiliaryTexture_ = 0;
    unsigned int auxiliaryFbo_ = 0;
    bool initialized_ = false;
};

}  // namespace pbr
