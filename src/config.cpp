#include "config.h"

#include "errors.h"
#include "json.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cloud {
namespace {

double numberField(const json::Value& object, const std::string& key, double fallback) {
    const json::Value* value = object.find(key);
    return value == nullptr ? fallback : value->asNumber();
}

int intField(const json::Value& object, const std::string& key, int fallback) {
    const json::Value* value = object.find(key);
    return value == nullptr ? fallback : static_cast<int>(value->asNumber());
}

std::string stringField(const json::Value& object, const std::string& key, const std::string& fallback) {
    const json::Value* value = object.find(key);
    return value == nullptr ? fallback : value->asString();
}

bool boolField(const json::Value& object, const std::string& key, bool fallback) {
    const json::Value* value = object.find(key);
    return value == nullptr ? fallback : value->asBool();
}

Vec3 vec3Field(const json::Value& object, const std::string& key, const Vec3& fallback) {
    const json::Value* value = object.find(key);
    if (value == nullptr) {
        return fallback;
    }
    const auto& array = value->asArray();
    if (array.size() != 3) {
        throw BusinessError("E_CONFIG", kExitConfig,
                            "vector field '" + key + "' must have three numbers");
    }
    return {static_cast<float>(array[0].asNumber()), static_cast<float>(array[1].asNumber()),
            static_cast<float>(array[2].asNumber())};
}

Mat4 matrixField(const json::Value& object, const std::string& key) {
    const json::Value* value = object.find(key);
    if (value == nullptr) {
        throw BusinessError("E_CONFIG", kExitConfig,
                            "missing matrix field '" + key + "'");
    }
    const auto& array = value->asArray();
    if (array.size() != 16) {
        throw BusinessError("E_CONFIG", kExitConfig,
                            "matrix field '" + key + "' must have 16 row-major numbers");
    }
    Mat4 matrix;
    for (int index = 0; index < 16; ++index) {
        matrix.m[index] = static_cast<float>(array[static_cast<std::size_t>(index)].asNumber());
    }
    return matrix;
}

const json::Value* optionalObject(const json::Value& parent, const std::string& key) {
    return parent.find(key);
}

}  // namespace

Camera CameraConfig::cameraAt(std::uint32_t frameIndex, int width, int height) const {
    const auto overrideIt = frameOverrides.find(frameIndex);
    if (overrideIt != frameOverrides.end()) {
        const ExplicitCamera& override = overrideIt->second;
        if (override.usesMatrices) {
            return Camera::fromMatrices(override.view, override.projection);
        }
        return Camera::fromLookAt(override.position, override.target, override.up,
                                  override.fovDegrees, static_cast<float>(width) /
                                                          static_cast<float>(height),
                                  nearPlane, farPlane);
    }

    Vec3 selectedPosition = position;
    Vec3 selectedTarget = target;
    if (orbitEnabled) {
        const float angle =
            (orbitStartDegrees + orbitDegreesPerFrame * static_cast<float>(frameIndex)) * kPi /
            180.0f;
        selectedPosition = orbitCenter +
                           Vec3{orbitRadius * std::cos(angle), orbitHeight,
                                orbitRadius * std::sin(angle)};
        selectedTarget = orbitCenter;
    }
    return Camera::fromLookAt(selectedPosition, selectedTarget, up, fovDegrees,
                              static_cast<float>(width) / static_cast<float>(height), nearPlane,
                              farPlane);
}

Light SunConfig::lightAt(std::uint32_t frameIndex) const {
    Light light;
    Vec3 lightDirection = normalize(direction);
    if (degreesPerFrame != 0.0f) {
        const float angle = degreesPerFrame * static_cast<float>(frameIndex) * kPi / 180.0f;
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        lightDirection =
            normalize(Vec3{lightDirection.x * cosine - lightDirection.z * sine, lightDirection.y,
                           lightDirection.x * sine + lightDirection.z * cosine});
    }
    light.direction = lightDirection;
    light.color = color;
    light.ambient = ambient;
    return light;
}

RenderSettings AppConfig::renderSettings() const {
    RenderSettings settings;
    settings.width = width;
    settings.height = height;
    settings.renderScale = renderScale;
    settings.maxSteps = maxSteps;
    settings.lightSteps = lightSteps;
    settings.densityExtinction = densityExtinction;
    settings.lightExtinction = lightExtinction;
    settings.exposure = exposure;
    settings.maxHistoryLength = maxHistoryLength;
    return settings;
}

AppConfig loadConfig(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw BusinessError("E_CONFIG", kExitConfig, "cannot open JSON config",
                            json::Value::Object{{"path", path}});
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    json::Value root;
    try {
        root = json::parse(contents.str());
    } catch (const std::exception& error) {
        throw BusinessError("E_CONFIG", kExitConfig, std::string("invalid JSON config: ") + error.what(),
                            json::Value::Object{{"path", path}});
    }
    if (!root.isObject()) {
        throw BusinessError("E_CONFIG", kExitConfig, "JSON config root must be an object");
    }

    AppConfig config;
    config.configPath = path;
    config.outputDirectory = stringField(root, "outputDir", config.outputDirectory);
    config.outputName = stringField(root, "outputName", config.outputName);
    config.width = intField(root, "width", config.width);
    config.height = intField(root, "height", config.height);
    config.startFrame = static_cast<std::uint32_t>(numberField(root, "startFrame", 0.0));
    config.endFrame = static_cast<std::uint32_t>(numberField(root, "endFrame", 1.0));
    config.seed = root.find("seed") == nullptr ? config.seed
                                               : root.find("seed")->asUInt64();
    config.cloudType = parseCloudType(stringField(root, "cloudType", "cumulus"));
    config.renderScale = intField(root, "renderScale", config.renderScale);
    config.maxSteps = intField(root, "maxSteps", config.maxSteps);
    config.lightSteps = intField(root, "lightSteps", config.lightSteps);
    config.densityExtinction =
        static_cast<float>(numberField(root, "densityExtinction", config.densityExtinction));
    config.lightExtinction =
        static_cast<float>(numberField(root, "lightExtinction", config.lightExtinction));
    config.exposure = static_cast<float>(numberField(root, "exposure", config.exposure));
    config.maxHistoryLength = intField(root, "maxHistoryLength", config.maxHistoryLength);
    config.asynchronous = boolField(root, "async", config.asynchronous);
    config.workerThreads =
        static_cast<std::uint32_t>(numberField(root, "threads", config.workerThreads));

    if (const json::Value* cameraValue = optionalObject(root, "camera")) {
        const json::Value& camera = *cameraValue;
        config.camera.position = vec3Field(camera, "position", config.camera.position);
        config.camera.target = vec3Field(camera, "target", config.camera.target);
        config.camera.up = vec3Field(camera, "up", config.camera.up);
        config.camera.fovDegrees =
            static_cast<float>(numberField(camera, "fovDeg", config.camera.fovDegrees));
        config.camera.nearPlane =
            static_cast<float>(numberField(camera, "near", config.camera.nearPlane));
        config.camera.farPlane =
            static_cast<float>(numberField(camera, "far", config.camera.farPlane));
        if (const json::Value* orbitValue = optionalObject(camera, "orbit")) {
            const json::Value& orbit = *orbitValue;
            config.camera.orbitEnabled = boolField(orbit, "enabled", true);
            config.camera.orbitCenter = vec3Field(orbit, "center", config.camera.orbitCenter);
            config.camera.orbitRadius =
                static_cast<float>(numberField(orbit, "radius", config.camera.orbitRadius));
            config.camera.orbitHeight =
                static_cast<float>(numberField(orbit, "height", config.camera.orbitHeight));
            config.camera.orbitStartDegrees =
                static_cast<float>(numberField(orbit, "startDegrees",
                                               config.camera.orbitStartDegrees));
            config.camera.orbitDegreesPerFrame =
                static_cast<float>(numberField(orbit, "degreesPerFrame",
                                               config.camera.orbitDegreesPerFrame));
        }
    }

    if (const json::Value* framesValue = optionalObject(root, "cameraFrames")) {
        const auto& frames = framesValue->asArray();
        for (const json::Value& frameValue : frames) {
            if (!frameValue.isObject() || frameValue.find("frame") == nullptr) {
                throw BusinessError("E_CONFIG", kExitConfig,
                                    "each cameraFrames entry needs a frame number");
            }
            ExplicitCamera camera;
            camera.usesMatrices = frameValue.find("viewMatrix") != nullptr ||
                                  frameValue.find("projectionMatrix") != nullptr;
            if (camera.usesMatrices) {
                camera.view = matrixField(frameValue, "viewMatrix");
                camera.projection = matrixField(frameValue, "projectionMatrix");
            } else {
                camera.position = vec3Field(frameValue, "position", config.camera.position);
                camera.target = vec3Field(frameValue, "target", config.camera.target);
                camera.up = vec3Field(frameValue, "up", config.camera.up);
                camera.fovDegrees =
                    static_cast<float>(numberField(frameValue, "fovDeg",
                                                  config.camera.fovDegrees));
            }
            const std::uint32_t frameIndex = static_cast<std::uint32_t>(
                frameValue.find("frame")->asNumber());
            config.camera.frameOverrides[frameIndex] = camera;
        }
    }

    if (const json::Value* sunValue = optionalObject(root, "sun")) {
        const json::Value& sun = *sunValue;
        config.sun.direction = vec3Field(sun, "direction", config.sun.direction);
        config.sun.color = vec3Field(sun, "color", config.sun.color);
        config.sun.ambient = static_cast<float>(numberField(sun, "ambient", config.sun.ambient));
        config.sun.degreesPerFrame =
            static_cast<float>(numberField(sun, "degreesPerFrame", config.sun.degreesPerFrame));
    }

    if (const json::Value* snapshotValue = optionalObject(root, "snapshot")) {
        const json::Value& snapshot = *snapshotValue;
        config.snapshot.version =
            static_cast<std::uint32_t>(numberField(snapshot, "version", config.snapshot.version));
        config.snapshot.loadPath = stringField(snapshot, "loadPath", "");
        config.snapshot.savePath = stringField(snapshot, "savePath", "");
        config.snapshot.saveEveryFrames = static_cast<std::uint32_t>(
            numberField(snapshot, "saveEveryFrames", config.snapshot.saveEveryFrames));
        config.snapshot.stopAfterSave = boolField(snapshot, "stopAfterSave", false);
    }
    if (config.snapshot.version != 1 && config.snapshot.version != 2) {
        throw BusinessError("E_VERSION_UNSUPPORTED", kExitVersionUnsupported,
                            "configured snapshot version must be 1 or 2",
                            json::Value::Object{{"foundVersion", config.snapshot.version}});
    }
    if (config.snapshot.saveEveryFrames == 0) {
        throw BusinessError("E_CONFIG", kExitConfig, "saveEveryFrames must be at least 1");
    }
    if (config.snapshot.savePath.empty()) {
        config.snapshot.savePath = config.outputDirectory + "/snapshot.cld";
    }

    if (const json::Value* memoryValue = optionalObject(root, "memory")) {
        config.memory.availableMB =
            numberField(*memoryValue, "availableMB", config.memory.availableMB);
    }
    if (const json::Value* qualityValue = optionalObject(root, "quality")) {
        config.quality.compareAgainstFullResolution =
            boolField(*qualityValue, "compareAgainstFullResolution", false);
        config.quality.writeFullResolutionReference =
            boolField(*qualityValue, "writeFullResolutionReference", false);
    }

    if (config.width <= 0 || config.height <= 0 || config.width > 16384 ||
        config.height > 16384) {
        throw BusinessError("E_CONFIG", kExitConfig, "invalid render dimensions",
                            json::Value::Object{{"width", config.width}, {"height", config.height}});
    }
    if (config.renderScale < 1 || config.renderScale > 4) {
        throw BusinessError("E_CONFIG", kExitConfig, "renderScale must be in [1,4]");
    }
    if (config.maxSteps < 1 || config.maxSteps > 1024) {
        throw BusinessError("E_CONFIG", kExitConfig, "maxSteps must be in [1,1024]");
    }
    if (config.lightSteps < 1 || config.lightSteps > 128) {
        throw BusinessError("E_CONFIG", kExitConfig, "lightSteps must be in [1,128]");
    }
    if (config.maxHistoryLength < 1 || config.maxHistoryLength > 4096) {
        throw BusinessError("E_CONFIG", kExitConfig, "maxHistoryLength must be in [1,4096]");
    }
    if (config.workerThreads < 1 || config.workerThreads > 64) {
        throw BusinessError("E_CONFIG", kExitConfig, "threads must be in [1,64]");
    }
    if (config.endFrame <= config.startFrame) {
        throw BusinessError("E_CONFIG", kExitConfig, "endFrame must be greater than startFrame");
    }
    if (!std::isfinite(config.memory.availableMB) || config.memory.availableMB < 0.0) {
        throw BusinessError("E_CONFIG", kExitConfig, "availableMB must be finite and non-negative");
    }
    if (!finite(config.sun.direction) || length(config.sun.direction) < 1.0e-6f) {
        throw BusinessError("E_CONFIG", kExitConfig, "sun direction must be a non-zero finite vector");
    }
    return config;
}

}  // namespace cloud

