#pragma once

#include <cstdint>
#include <string>

namespace cv {

struct AppOptions {
    int width = 960;
    int height = 640;
    uint32_t startTick = 0;
    int cascadeResolution = 1024;
    int pointShadowResolution = 256;
    float yawDegrees = 0.0f;
    float pitchDegrees = 0.0f;
    float cameraDistance = 5.1f;
    bool hasYaw = false;
    bool hasPitch = false;
    bool hasCameraDistance = false;
    bool shadows = true;
    bool startPaused = false;
    bool wireframe = false;
    std::string ppmPath = "frame.ppm";
    std::string bmpPath = "frame.bmp";
    char ppmKey = 'P';
    char bmpKey = 'B';
    bool helpRequested = false;
    bool valid = true;
    std::string error;
};

AppOptions parseOptions(int argc, char** argv);
void printUsage(const char* executableName);

}  // namespace cv

