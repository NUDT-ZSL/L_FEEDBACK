#pragma once

#include "math.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ct {

struct HeightfieldConfig {
    int size = 192;
    double base_frequency = 0.016;
    double amplitude = 15.0;
    std::uint32_t seed = 1337;
    int octaves = 5;
};

struct CameraConfig {
    Vec3 position{96.0, 54.0, 226.0};
    Vec3 target{96.0, 8.0, 82.0};
    Vec3 up{0.0, 1.0, 0.0};
    double fov_degrees = 60.0;
    double near_plane = 0.4;
    double far_plane = 650.0;
};

struct RenderConfig {
    int width = 800;
    int height = 450;
    int supersampling = 2;
};

struct LodConfig {
    int block_size = 32;
    double lod0_screen_area = 0.024;
    double lod1_screen_area = 0.019;
    double lod0_distance = 58.0;
    double lod1_distance = 150.0;
};

struct DirectionalLight {
    Vec3 direction{0.0, 1.0, 0.0};
    Vec3 color{1.0, 1.0, 1.0};
    double intensity = 1.0;
};

struct PointLight {
    Vec3 position;
    Vec3 color{1.0, 1.0, 1.0};
    double intensity = 1.0;
    double range = 40.0;
};

struct TextureConfig {
    int size = 256;
    double scale = 0.08;
};

struct OutputConfig {
    std::string image_path;
    std::string report_path;
};

struct AppConfig {
    HeightfieldConfig heightfield;
    CameraConfig camera;
    RenderConfig render;
    LodConfig lod;
    TextureConfig texture;
    std::vector<DirectionalLight> directional_lights;
    std::vector<PointLight> point_lights;
    Vec3 ambient{0.055, 0.075, 0.105};
    OutputConfig output;
};

bool loadConfig(const std::string& path, AppConfig& config, std::string& error);

}  // namespace ct
