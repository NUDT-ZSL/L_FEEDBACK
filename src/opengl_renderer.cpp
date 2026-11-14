#include "renderer.hpp"

#include "opengl_renderer.hpp"
#include "scene.hpp"

#include "gl_loader.hpp"

#include <glm/vec3.hpp>

#ifdef _WIN32
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace cr {
namespace {

using namespace gl;

constexpr int kMaxLayersPerMaterial = 3;
constexpr int kMaxLights = 64;

const char* kVertexShader = R"GLSL(
#version 430 core
void main() {
    vec2 vertices[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
}
)GLSL";

const char* kGBufferShader = R"GLSL(
#version 430 core

layout(location = 0) out vec4 outBaseMetal;
layout(location = 1) out vec4 outNormalHit;
layout(location = 2) out float outRoughness;
layout(location = 3) out float outDepth;

uniform sampler2DArray uBaseColorTex;
uniform sampler2DArray uMetallicTex;
uniform sampler2DArray uRoughnessTex;
uniform sampler2DArray uNormalTex;
uniform int uMaxLayers;
uniform int uLayerCount[5];
uniform int uLayerSlice[15];
uniform int uBlendMode[15];
uniform float uLayerOpacity[15];

uniform vec3 uCameraPosition;
uniform vec3 uCameraForward;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;
uniform float uTanHalfFov;
uniform float uAspect;
uniform float uNear;
uniform float uFar;
uniform float uWidth;
uniform float uHeight;

const float PI = 3.14159265358979323846;

struct MapResult {
    float distance;
    int zone;
};

float sdEllipsoid(vec3 p, vec3 center, vec3 radii) {
    vec3 local = p - center;
    vec3 scaled = local / radii;
    vec3 squaredScaled = scaled / radii;
    float k0 = length(scaled);
    float k1 = length(squaredScaled);
    if (k1 <= 0.000001) {
        return -min(min(radii.x, radii.y), radii.z);
    }
    return k0 * (k0 - 1.0) / k1;
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float radius) {
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - radius;
}

float sdTorusY(vec3 p, vec3 center, float majorRadius, float minorRadius) {
    vec3 local = p - center;
    float radial = sqrt(local.x * local.x + local.z * local.z) - majorRadius;
    return sqrt(radial * radial + local.y * local.y) - minorRadius;
}

void consider(inout MapResult result, float distance, int zone) {
    if (distance < result.distance) {
        result.distance = distance;
        result.zone = zone;
    }
}

MapResult mapScene(vec3 p) {
    MapResult result;
    result.distance = 1.0e9;
    result.zone = 0;
    consider(result, sdEllipsoid(p, vec3(0.0, 0.38, 0.0), vec3(0.64, 0.76, 0.60)), 0);
    consider(result, sdEllipsoid(p, vec3(0.0, 0.69, -0.04), vec3(0.69, 0.50, 0.65)), 3);
    consider(result, sdCapsule(p, vec3(0.0, -0.09, 0.0), vec3(0.0, -0.43, 0.0), 0.205), 0);
    consider(result, sdEllipsoid(p, vec3(0.0, -1.04, 0.0), vec3(1.03, 0.70, 0.52)), 1);
    consider(result, sdCapsule(p, vec3(-0.78, -0.68, 0.02), vec3(-1.03, -0.90, 0.02), 0.25), 1);
    consider(result, sdCapsule(p, vec3(0.78, -0.68, 0.02), vec3(1.03, -0.90, 0.02), 0.25), 1);
    consider(result, sdEllipsoid(p, vec3(-0.87, -0.66, 0.02), vec3(0.38, 0.29, 0.34)), 2);
    consider(result, sdEllipsoid(p, vec3(0.87, -0.66, 0.02), vec3(0.38, 0.29, 0.34)), 2);
    consider(result, sdEllipsoid(p, vec3(-0.24, 0.45, 0.555), vec3(0.112, 0.073, 0.052)), 4);
    consider(result, sdEllipsoid(p, vec3(0.24, 0.45, 0.555), vec3(0.112, 0.073, 0.052)), 4);
    consider(result, sdEllipsoid(p, vec3(0.0, 0.31, 0.575), vec3(0.095, 0.135, 0.105)), 0);
    consider(result, sdEllipsoid(p, vec3(-0.62, 0.38, 0.0), vec3(0.075, 0.15, 0.07)), 0);
    consider(result, sdEllipsoid(p, vec3(0.62, 0.38, 0.0), vec3(0.075, 0.15, 0.07)), 0);
    consider(result, sdTorusY(p, vec3(0.0, 0.67, 0.0), 0.69, 0.045), 2);
    consider(result, sdEllipsoid(p, vec3(0.0, -0.91, 0.50), vec3(0.17, 0.21, 0.065)), 2);
    return result;
}

vec3 sceneNormal(vec3 p) {
    const float epsilon = 0.0015;
    vec3 dx = vec3(mapScene(p + vec3(epsilon, 0.0, 0.0)).distance,
                   mapScene(p + vec3(0.0, epsilon, 0.0)).distance,
                   mapScene(p + vec3(0.0, 0.0, epsilon)).distance);
    vec3 dy = vec3(mapScene(p - vec3(epsilon, 0.0, 0.0)).distance,
                   mapScene(p - vec3(0.0, epsilon, 0.0)).distance,
                   mapScene(p - vec3(0.0, 0.0, epsilon)).distance);
    return normalize(dx - dy);
}

float traceScene(vec3 origin, vec3 direction, out int zone) {
    float t = 0.5;
    zone = 0;
    for (int step = 0; step < 170; ++step) {
        MapResult mapped = mapScene(origin + direction * t);
        if (mapped.distance < 0.0007) {
            zone = mapped.zone;
            return t;
        }
        t += max(mapped.distance * 0.62, 0.0008);
        if (t > uFar) {
            break;
        }
    }
    return -1.0;
}

vec2 surfaceUV(vec3 position, vec3 normal, int zone) {
    if (zone == 4) {
        float centerX = position.x < 0.0 ? -0.24 : 0.24;
        return vec2(clamp(0.5 + (position.x - centerX) / 0.224, 0.0, 1.0),
                    clamp(0.5 - (position.y - 0.45) / 0.146, 0.0, 1.0));
    }
    if (zone == 0 || zone == 3) {
        return vec2(atan(normal.x, normal.z) / (2.0 * PI) + 0.5,
                    clamp((position.y + 0.38) / 1.52, 0.0, 1.0));
    }
    return vec2(atan(position.x, position.z) / (2.0 * PI) + 0.5,
                clamp((position.y + 1.76) / 2.40, 0.0, 1.0));
}

vec3 tangentBasis(vec3 normal) {
    vec3 reference = abs(normal.y) < 0.92 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    return normalize(cross(reference, normal));
}

void composeLayers(int zone, vec2 uv, out vec3 baseColor, out float metallic,
                   out float roughness, out vec3 tangentNormal) {
    baseColor = vec3(0.5);
    metallic = 0.0;
    roughness = 0.5;
    tangentNormal = vec3(0.0, 0.0, 1.0);
    bool hasValue = false;
    int count = uLayerCount[zone];
    for (int layer = 0; layer < count; ++layer) {
        int index = zone * uMaxLayers + layer;
        int sliceIndex = uLayerSlice[index];
        vec4 baseSample = texture(uBaseColorTex, vec3(uv, float(sliceIndex)));
        vec4 metalSample = texture(uMetallicTex, vec3(uv, float(sliceIndex)));
        vec4 roughSample = texture(uRoughnessTex, vec3(uv, float(sliceIndex)));
        vec4 normalSample = texture(uNormalTex, vec3(uv, float(sliceIndex)));
        float weight = uLayerOpacity[index];
        if (uBlendMode[index] == 1) {
            weight *= baseSample.a;
        } else if (uBlendMode[index] == 2) {
            weight *= baseSample.a *
                      dot(baseSample.rgb, vec3(0.2126, 0.7152, 0.0722));
        }
        weight = clamp(weight, 0.0, 1.0);
        vec3 layerNormal = normalize(normalSample.rgb * 2.0 - vec3(1.0));
        if (!hasValue) {
            baseColor = baseSample.rgb;
            metallic = metalSample.r;
            roughness = roughSample.r;
            tangentNormal = layerNormal;
            hasValue = true;
        } else {
            baseColor = mix(baseColor, baseSample.rgb, weight);
            metallic = mix(metallic, metalSample.r, weight);
            roughness = mix(roughness, roughSample.r, weight);
            tangentNormal = normalize(mix(tangentNormal, layerNormal, weight));
        }
    }
    metallic = clamp(metallic, 0.0, 1.0);
    roughness = clamp(roughness, 0.03, 1.0);
    tangentNormal = normalize(tangentNormal);
}

void main() {
    float screenX = (2.0 * gl_FragCoord.x / uWidth - 1.0);
    float screenY = (2.0 * gl_FragCoord.y / uHeight - 1.0);
    vec3 direction = normalize(uCameraForward +
                               uCameraRight * (screenX * uAspect * uTanHalfFov) +
                               uCameraUp * (screenY * uTanHalfFov));
    int zone = 0;
    float hitDistance = traceScene(uCameraPosition, direction, zone);
    if (hitDistance < 0.0) {
        outBaseMetal = vec4(0.0);
        outNormalHit = vec4(0.0);
        outRoughness = 0.0;
        outDepth = uFar;
        gl_FragDepth = 1.0;
        return;
    }

    vec3 position = uCameraPosition + direction * hitDistance;
    vec3 geometricNormal = sceneNormal(position);
    vec2 uv = surfaceUV(position, geometricNormal, zone);
    vec3 baseColor;
    float metallic;
    float roughness;
    vec3 tangentNormal;
    composeLayers(zone, uv, baseColor, metallic, roughness, tangentNormal);
    vec3 tangent = tangentBasis(geometricNormal);
    vec3 bitangent = normalize(cross(geometricNormal, tangent));
    vec3 worldNormal = normalize(tangent * tangentNormal.x +
                                 bitangent * tangentNormal.y +
                                 geometricNormal * tangentNormal.z);
    float depth = dot(position - uCameraPosition, uCameraForward);
    float ndcDepth = (uFar + uNear) / (uFar - uNear) -
                     2.0 * uFar * uNear / ((uFar - uNear) * max(depth, 0.001));
    gl_FragDepth = clamp(ndcDepth * 0.5 + 0.5, 0.0, 1.0);
    outBaseMetal = vec4(clamp(baseColor, 0.0, 1.0), metallic);
    outNormalHit = vec4(worldNormal, 1.0);
    outRoughness = roughness;
    outDepth = depth;
}
)GLSL";

const char* kLightingShader = R"GLSL(
#version 430 core

layout(location = 0) out vec4 outColor;

uniform sampler2D uBaseMetalTex;
uniform sampler2D uNormalHitTex;
uniform sampler2D uRoughnessTex;
uniform sampler2D uDepthTex;
uniform vec3 uCameraPosition;
uniform vec3 uCameraForward;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;
uniform float uTanHalfFov;
uniform float uAspect;
uniform float uWidth;
uniform float uHeight;

uniform int uLightCount;
uniform vec3 uLightPosition[64];
uniform vec3 uLightColor[64];
uniform float uLightIntensity[64];
uniform float uLightRadius[64];

const float PI = 3.14159265358979323846;

float distributionGGX(float normalDotHalf, float roughness) {
    float alpha = roughness * roughness;
    float alphaSquared = alpha * alpha;
    float denominator = normalDotHalf * normalDotHalf * (alphaSquared - 1.0) + 1.0;
    return alphaSquared / max(PI * denominator * denominator, 0.000001);
}

float geometrySchlickGGX(float normalDotView, float roughness) {
    float k = roughness * roughness / 8.0;
    return normalDotView / max(normalDotView * (1.0 - k) + k, 0.00001);
}

float geometrySmith(float normalDotView, float normalDotLight, float roughness) {
    return geometrySchlickGGX(normalDotView, roughness) *
           geometrySchlickGGX(normalDotLight, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 f0) {
    float factor = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    return f0 + (vec3(1.0) - f0) * factor;
}

float lightAttenuation(vec3 lightToSurface, float radius) {
    float squaredDistance = max(dot(lightToSurface, lightToSurface), 0.00001);
    float normalized = sqrt(squaredDistance) / radius;
    if (normalized >= 1.0) {
        return 0.0;
    }
    float window = 1.0 - pow(normalized, 4.0);
    return window * window / (squaredDistance + 0.04);
}

vec3 toneMap(vec3 value) {
    vec3 mapped = value / (vec3(1.0) + value);
    return pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
}

void main() {
    vec4 baseMetal = texture(uBaseMetalTex, gl_FragCoord.xy / vec2(uWidth, uHeight));
    vec4 normalHit = texture(uNormalHitTex, gl_FragCoord.xy / vec2(uWidth, uHeight));
    float roughnessSample = texture(uRoughnessTex, gl_FragCoord.xy / vec2(uWidth, uHeight)).r;
    float depth = texture(uDepthTex, gl_FragCoord.xy / vec2(uWidth, uHeight)).r;
    float backgroundMix = 1.0 - gl_FragCoord.y / uHeight;
    vec3 background = vec3(0.020, 0.026, 0.040) * (0.60 + backgroundMix * 0.65) +
                      vec3(0.035, 0.0, 0.010) * (1.0 - backgroundMix);
    if (normalHit.w < 0.5) {
        outColor = vec4(background, 1.0);
        return;
    }

    float screenX = (2.0 * gl_FragCoord.x / uWidth - 1.0);
    float screenY = (2.0 * gl_FragCoord.y / uHeight - 1.0);
    vec3 direction = normalize(uCameraForward +
                               uCameraRight * (screenX * uAspect * uTanHalfFov) +
                               uCameraUp * (screenY * uTanHalfFov));
    vec3 worldPosition = uCameraPosition +
                         direction * (depth / max(dot(direction, uCameraForward), 0.000001));
    vec3 normal = normalize(normalHit.xyz);
    vec3 view = normalize(uCameraPosition - worldPosition);
    vec3 baseColor = clamp(baseMetal.rgb, 0.0, 1.0);
    float metallic = clamp(baseMetal.a, 0.0, 1.0);
    float roughness = clamp(roughnessSample, 0.04, 1.0);
    vec3 f0 = mix(vec3(0.040), baseColor, metallic);
    vec3 radianceOut = baseColor * (0.022 + 0.018 * clamp(normal.y, 0.0, 1.0));

    for (int lightIndex = 0; lightIndex < uLightCount; ++lightIndex) {
        vec3 toLight = uLightPosition[lightIndex] - worldPosition;
        float lightDistance = length(toLight);
        vec3 lightDirection = toLight / max(lightDistance, 0.00001);
        float normalDotLight = dot(normal, lightDirection);
        if (normalDotLight <= 0.0) {
            continue;
        }
        float attenuation = lightAttenuation(toLight, uLightRadius[lightIndex]);
        if (attenuation <= 0.0 || uLightIntensity[lightIndex] <= 0.0) {
            continue;
        }
        vec3 halfDirection = normalize(lightDirection + view);
        float normalDotHalf = clamp(dot(normal, halfDirection), 0.0, 1.0);
        float viewDotHalf = clamp(dot(view, halfDirection), 0.0, 1.0);
        float normalDotView = clamp(dot(normal, view), 0.0, 1.0);
        float distribution = distributionGGX(normalDotHalf, roughness);
        float geometry = geometrySmith(normalDotView, normalDotLight, roughness);
        vec3 fresnel = fresnelSchlick(viewDotHalf, f0);
        vec3 specular = fresnel * (distribution * geometry /
                                   max(4.0 * normalDotView * normalDotLight, 0.0001));
        vec3 diffuse = baseColor *
                       ((vec3(1.0) - fresnel) * ((1.0 - metallic) / PI));
        vec3 incoming = uLightColor[lightIndex] *
                        (uLightIntensity[lightIndex] * attenuation);
        radianceOut += (diffuse + specular) * incoming * normalDotLight;
    }
    outColor = vec4(toneMap(radianceOut), 1.0);
}
)GLSL";

std::string shaderLog(GLuint shader) {
    GLint length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    if (length <= 1) {
        return {};
    }
    std::string log(static_cast<std::size_t>(length), '\0');
    glGetShaderInfoLog(shader, length, nullptr, log.data());
    while (!log.empty() && log.back() == '\0') {
        log.pop_back();
    }
    return log;
}

std::string programLog(GLuint program) {
    GLint length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    if (length <= 1) {
        return {};
    }
    std::string log(static_cast<std::size_t>(length), '\0');
    glGetProgramInfoLog(program, length, nullptr, log.data());
    while (!log.empty() && log.back() == '\0') {
        log.pop_back();
    }
    return log;
}

GLuint compileShader(GLenum type, const char* source) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == 0) {
        const std::string log = shaderLog(shader);
        glDeleteShader(shader);
        throw AppError(E_OPENGL, "OpenGL shader compilation failed: " + log);
    }
    return shader;
}

GLuint linkProgram(const char* vertexSource, const char* fragmentSource,
                   const char* label) {
    const GLuint vertex = compileShader(GL_VERTEX_SHADER, vertexSource);
    const GLuint fragment = compileShader(GL_FRAGMENT_SHADER, fragmentSource);
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint status = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == 0) {
        const std::string log = programLog(program);
        glDeleteProgram(program);
        throw AppError(E_OPENGL, std::string("OpenGL link failed for ") + label +
                                     ": " + log);
    }
    return program;
}

GLint uniform(GLuint program, const char* name) {
    const GLint location = glGetUniformLocation(program, name);
    if (location < 0) {
        throw AppError(E_OPENGL, std::string("missing shader uniform '") + name + "'");
    }
    return location;
}

void setVec3(GLuint program, const char* name, const Vec3& value) {
    glUniform3f(uniform(program, name), value.x, value.y, value.z);
}

GLuint createTexture2D(int width, int height, GLenum internalFormat, GLenum format,
                       GLenum type) {
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(internalFormat), width, height, 0,
                 format, type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    return texture;
}

GLuint createTextureArray(int width, int height, int slices) {
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D_ARRAY, texture);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, static_cast<GLint>(GL_RGBA8), width, height,
                 slices, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER,
                    static_cast<GLint>(GL_LINEAR));
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER,
                    static_cast<GLint>(GL_LINEAR));
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S,
                    static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T,
                    static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R,
                    static_cast<GLint>(GL_CLAMP_TO_EDGE));
    return texture;
}

void uploadSlice(GLuint texture, int slice, const Image& image) {
    glBindTexture(GL_TEXTURE_2D_ARRAY, texture);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, slice, image.width, image.height, 1,
                    GL_RGBA, GL_UNSIGNED_BYTE, image.pixels.data());
}

void checkGlErrors(const char* stage) {
    std::string errors;
    for (int count = 0; count < 16; ++count) {
        const GLenum error = glGetError();
        if (error == GL_NO_ERROR) {
            break;
        }
        std::ostringstream value;
        value << "0x" << std::hex << error;
        if (!errors.empty()) {
            errors += ", ";
        }
        errors += value.str();
    }
    if (!errors.empty()) {
        throw AppError(E_OPENGL,
                       std::string("OpenGL error after ") + stage + ": " + errors);
    }
}

#ifdef _WIN32
struct GlfwSession {
    GLFWwindow* window = nullptr;
    bool initialized = false;

    ~GlfwSession() {
        if (window != nullptr) {
            glfwDestroyWindow(window);
        }
        if (initialized) {
            glfwTerminate();
        }
    }
};

void createContext(GlfwSession& session) {
    if (glfwInit() == GLFW_FALSE) {
        throw AppError(E_OPENGL_UNAVAILABLE, "GLFW could not initialize a display context");
    }
    session.initialized = true;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
    session.window = glfwCreateWindow(32, 32, "character-renderer-offscreen", nullptr,
                                      nullptr);
    if (session.window == nullptr) {
        throw AppError(E_OPENGL_UNAVAILABLE,
                       "GLFW could not create an OpenGL 4.3 core context");
    }
    glfwMakeContextCurrent(session.window);
    glfwSwapInterval(0);
}
#endif

void checkOpenGLVersion() {
    GLint major = 0;
    GLint minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    if (major < 4 || (major == 4 && minor < 3)) {
        throw AppError(E_OPENGL_UNAVAILABLE,
                       "OpenGL 4.3 or newer is required; context reports " +
                           std::to_string(major) + "." + std::to_string(minor));
    }
}

struct GpuMaterialLayout {
    std::vector<int> layerCounts;
    std::vector<int> slices;
    std::vector<int> blendModes;
    std::vector<float> opacities;
    int maxLayers = 0;
    int totalSlices = 0;
    int width = 0;
    int height = 0;
};

GpuMaterialLayout buildLayout(const MaterialLibrary& materials) {
    GpuMaterialLayout layout;
    layout.layerCounts.assign(kMaterialZoneCount, 0);
    layout.slices.assign(kMaterialZoneCount * kMaxLayersPerMaterial, -1);
    layout.blendModes.assign(kMaterialZoneCount * kMaxLayersPerMaterial, 0);
    layout.opacities.assign(kMaterialZoneCount * kMaxLayersPerMaterial, 0.0F);
    for (int zone = 0; zone < kMaterialZoneCount; ++zone) {
        const LoadedMaterial& material =
            materials.require(kRequiredMaterialNames[zone]);
        const int count = static_cast<int>(material.layers.size());
        if (count <= 0 || count > kMaxLayersPerMaterial) {
            throw AppError(E_OPENGL,
                           "material '" + material.name + "' has unsupported layer count " +
                               std::to_string(count));
        }
        layout.layerCounts[static_cast<std::size_t>(zone)] = count;
        layout.maxLayers = std::max(layout.maxLayers, count);
        for (int layer = 0; layer < count; ++layer) {
            const LoadedLayer& loadedLayer =
                material.layers[static_cast<std::size_t>(layer)];
            const Image& baseImage = loadedLayer.baseColor.image();
            if (layout.totalSlices == 0) {
                layout.width = baseImage.width;
                layout.height = baseImage.height;
            } else if (baseImage.width != layout.width || baseImage.height != layout.height) {
                throw AppError(E_OPENGL, "all layer textures must use the same dimensions");
            }
            ++layout.totalSlices;
        }
    }
    // maxLayers may change while indexing; recompute the final layout identically.
    layout.slices.assign(kMaterialZoneCount * kMaxLayersPerMaterial, -1);
    layout.blendModes.assign(kMaterialZoneCount * kMaxLayersPerMaterial, 0);
    layout.opacities.assign(kMaterialZoneCount * kMaxLayersPerMaterial, 0.0F);
    int slice = 0;
    for (int zone = 0; zone < kMaterialZoneCount; ++zone) {
        const LoadedMaterial& material =
            materials.require(kRequiredMaterialNames[zone]);
        for (std::size_t layer = 0; layer < material.layers.size(); ++layer) {
            const std::size_t index =
                static_cast<std::size_t>(zone) *
                    static_cast<std::size_t>(layout.maxLayers) +
                layer;
            layout.slices[index] = slice;
            layout.blendModes[index] =
                static_cast<int>(material.layers[layer].spec.blendMode);
            layout.opacities[index] = material.layers[layer].spec.opacity;
            ++slice;
        }
    }
    layout.totalSlices = slice;
    return layout;
}

struct UploadedTextures {
    GLuint baseColor = 0;
    GLuint metallic = 0;
    GLuint roughness = 0;
    GLuint normal = 0;
};

UploadedTextures uploadMaterialTextures(const MaterialLibrary& materials,
                                        const GpuMaterialLayout& layout) {
    UploadedTextures textures;
    textures.baseColor =
        createTextureArray(layout.width, layout.height, layout.totalSlices);
    textures.metallic =
        createTextureArray(layout.width, layout.height, layout.totalSlices);
    textures.roughness =
        createTextureArray(layout.width, layout.height, layout.totalSlices);
    textures.normal =
        createTextureArray(layout.width, layout.height, layout.totalSlices);

    int slice = 0;
    for (int zone = 0; zone < kMaterialZoneCount; ++zone) {
        const LoadedMaterial& material =
            materials.require(kRequiredMaterialNames[zone]);
        for (const LoadedLayer& layer : material.layers) {
            const Image& base = layer.baseColor.image();
            const Image& metal = layer.metallic.image();
            const Image& rough = layer.roughness.image();
            const Image& normal = layer.normal.image();
            if (base.width != layout.width || base.height != layout.height ||
                metal.width != layout.width || metal.height != layout.height ||
                rough.width != layout.width || rough.height != layout.height ||
                normal.width != layout.width || normal.height != layout.height) {
                throw AppError(E_OPENGL, "layer texture dimensions are inconsistent");
            }
            uploadSlice(textures.baseColor, slice, base);
            uploadSlice(textures.metallic, slice, metal);
            uploadSlice(textures.roughness, slice, rough);
            uploadSlice(textures.normal, slice, normal);
            ++slice;
        }
    }
    return textures;
}

struct GBufferTargets {
    GLuint framebuffer = 0;
    GLuint baseMetal = 0;
    GLuint normalHit = 0;
    GLuint roughness = 0;
    GLuint depth = 0;
    GLuint depthRenderbuffer = 0;
};

GBufferTargets createGBuffer(int width, int height) {
    GBufferTargets targets;
    targets.baseMetal =
        createTexture2D(width, height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    targets.normalHit =
        createTexture2D(width, height, GL_RGBA16F, GL_RGBA, GL_FLOAT);
    targets.roughness = createTexture2D(width, height, GL_R32F, GL_RED, GL_FLOAT);
    targets.depth = createTexture2D(width, height, GL_R32F, GL_RED, GL_FLOAT);

    glGenFramebuffers(1, &targets.framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, targets.framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           targets.baseMetal, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D,
                           targets.normalHit, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D,
                           targets.roughness, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D,
                           targets.depth, 0);
    glGenRenderbuffers(1, &targets.depthRenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, targets.depthRenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER,
                              targets.depthRenderbuffer);
    const GLenum drawBuffers[4] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
                                   GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
    glDrawBuffers(4, drawBuffers);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        throw AppError(E_OPENGL, "G-Buffer framebuffer is incomplete");
    }
    return targets;
}

struct FinalTarget {
    GLuint framebuffer = 0;
    GLuint texture = 0;
};

FinalTarget createFinalTarget(int width, int height) {
    FinalTarget target;
    target.texture =
        createTexture2D(width, height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    glGenFramebuffers(1, &target.framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           target.texture, 0);
    const GLenum drawBuffer = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuffer);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        throw AppError(E_OPENGL, "final framebuffer is incomplete");
    }
    return target;
}

void setGBufferUniforms(GLuint program, const Camera& camera, const CameraFrame& frame,
                        const GpuMaterialLayout& layout, const UploadedTextures& textures) {
    glUniform1i(uniform(program, "uBaseColorTex"), 0);
    glUniform1i(uniform(program, "uMetallicTex"), 1);
    glUniform1i(uniform(program, "uRoughnessTex"), 2);
    glUniform1i(uniform(program, "uNormalTex"), 3);
    glUniform1i(uniform(program, "uMaxLayers"), layout.maxLayers);
    glUniform1iv(uniform(program, "uLayerCount"), kMaterialZoneCount,
                 layout.layerCounts.data());
    glUniform1iv(uniform(program, "uLayerSlice"),
                 kMaterialZoneCount * kMaxLayersPerMaterial, layout.slices.data());
    glUniform1iv(uniform(program, "uBlendMode"),
                 kMaterialZoneCount * kMaxLayersPerMaterial, layout.blendModes.data());
    glUniform1fv(uniform(program, "uLayerOpacity"),
                 kMaterialZoneCount * kMaxLayersPerMaterial, layout.opacities.data());
    setVec3(program, "uCameraPosition", frame.origin);
    setVec3(program, "uCameraForward", frame.forward);
    setVec3(program, "uCameraRight", frame.right);
    setVec3(program, "uCameraUp", frame.up);
    glUniform1f(uniform(program, "uTanHalfFov"), frame.tanHalfFov);
    glUniform1f(uniform(program, "uAspect"), frame.aspect);
    glUniform1f(uniform(program, "uNear"), camera.nearPlane);
    glUniform1f(uniform(program, "uFar"), camera.farPlane);
    glUniform1f(uniform(program, "uWidth"), static_cast<float>(camera.width));
    glUniform1f(uniform(program, "uHeight"), static_cast<float>(camera.height));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, textures.baseColor);
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D_ARRAY, textures.metallic);
    glActiveTexture(GL_TEXTURE0 + 2);
    glBindTexture(GL_TEXTURE_2D_ARRAY, textures.roughness);
    glActiveTexture(GL_TEXTURE0 + 3);
    glBindTexture(GL_TEXTURE_2D_ARRAY, textures.normal);
}

void setLightingUniforms(GLuint program, const Camera& camera, const CameraFrame& frame,
                         const std::vector<Light>& lights) {
    if (lights.size() > static_cast<std::size_t>(kMaxLights)) {
        throw AppError(E_OPENGL, "OpenGL backend supports at most " +
                                     std::to_string(kMaxLights) + " point lights");
    }
    glUniform1i(uniform(program, "uBaseMetalTex"), 0);
    glUniform1i(uniform(program, "uNormalHitTex"), 1);
    glUniform1i(uniform(program, "uRoughnessTex"), 2);
    glUniform1i(uniform(program, "uDepthTex"), 3);
    setVec3(program, "uCameraPosition", frame.origin);
    setVec3(program, "uCameraForward", frame.forward);
    setVec3(program, "uCameraRight", frame.right);
    setVec3(program, "uCameraUp", frame.up);
    glUniform1f(uniform(program, "uTanHalfFov"), frame.tanHalfFov);
    glUniform1f(uniform(program, "uAspect"), frame.aspect);
    glUniform1f(uniform(program, "uWidth"), static_cast<float>(camera.width));
    glUniform1f(uniform(program, "uHeight"), static_cast<float>(camera.height));
    glUniform1i(uniform(program, "uLightCount"), static_cast<int>(lights.size()));

    std::array<glm::vec3, kMaxLights> positions{};
    std::array<glm::vec3, kMaxLights> colors{};
    std::array<float, kMaxLights> intensities{};
    std::array<float, kMaxLights> radii{};
    for (std::size_t index = 0; index < lights.size(); ++index) {
        positions[index] = glm::vec3(lights[index].position.x, lights[index].position.y,
                                     lights[index].position.z);
        colors[index] = glm::vec3(lights[index].color.x, lights[index].color.y,
                                  lights[index].color.z);
        intensities[index] = lights[index].intensity;
        radii[index] = lights[index].radius;
    }
    glUniform3fv(uniform(program, "uLightPosition"), static_cast<GLsizei>(lights.size()),
                 reinterpret_cast<const GLfloat*>(positions.data()));
    glUniform3fv(uniform(program, "uLightColor"), static_cast<GLsizei>(lights.size()),
                 reinterpret_cast<const GLfloat*>(colors.data()));
    glUniform1fv(uniform(program, "uLightIntensity"),
                 static_cast<GLsizei>(lights.size()), intensities.data());
    glUniform1fv(uniform(program, "uLightRadius"), static_cast<GLsizei>(lights.size()),
                 radii.data());
}

}  // namespace

Image renderOpenGLDeferredWithCurrentContext(const MaterialLibrary& materials,
                                             const Camera& camera,
                                             const std::vector<Light>& lights,
                                             const gl::GlProcLoader& loader) {
    loadFunctions(loader);
    checkOpenGLVersion();
    const CameraFrame frame = makeCameraFrame(camera);
    const GpuMaterialLayout layout = buildLayout(materials);
    const UploadedTextures textures = uploadMaterialTextures(materials, layout);
    const GBufferTargets gbuffer = createGBuffer(camera.width, camera.height);
    const FinalTarget finalTarget = createFinalTarget(camera.width, camera.height);
    const GLuint gbufferProgram = linkProgram(kVertexShader, kGBufferShader, "G-Buffer");
    const GLuint lightingProgram =
        linkProgram(kVertexShader, kLightingShader, "lighting");

    GLuint vertexArray = 0;
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    glBindFramebuffer(GL_FRAMEBUFFER, gbuffer.framebuffer);
    glViewport(0, 0, camera.width, camera.height);
    glClearColor(0.0F, 0.0F, 0.0F, 0.0F);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(gbufferProgram);
    setGBufferUniforms(gbufferProgram, camera, frame, layout, textures);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    checkGlErrors("G-Buffer pass");

    glBindFramebuffer(GL_FRAMEBUFFER, finalTarget.framebuffer);
    glViewport(0, 0, camera.width, camera.height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(lightingProgram);
    setLightingUniforms(lightingProgram, camera, frame, lights);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gbuffer.baseMetal);
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, gbuffer.normalHit);
    glActiveTexture(GL_TEXTURE0 + 2);
    glBindTexture(GL_TEXTURE_2D, gbuffer.roughness);
    glActiveTexture(GL_TEXTURE0 + 3);
    glBindTexture(GL_TEXTURE_2D, gbuffer.depth);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    checkGlErrors("lighting pass");

    Image output(camera.width, camera.height, 4);
    glReadPixels(0, 0, camera.width, camera.height, GL_RGBA, GL_UNSIGNED_BYTE,
                 output.pixels.data());
    glFinish();
    checkGlErrors("framebuffer readback");
    for (int y = 0; y < camera.height / 2; ++y) {
        std::uint8_t* top = output.row(y);
        std::uint8_t* bottom = output.row(camera.height - 1 - y);
        for (int byte = 0; byte < camera.width * 4; ++byte) {
            std::swap(top[byte], bottom[byte]);
        }
    }

    glDeleteVertexArrays(1, &vertexArray);
    glDeleteProgram(gbufferProgram);
    glDeleteProgram(lightingProgram);
    glDeleteFramebuffers(1, &gbuffer.framebuffer);
    glDeleteFramebuffers(1, &finalTarget.framebuffer);
    glDeleteRenderbuffers(1, &gbuffer.depthRenderbuffer);
    const GLuint gbufferTextures[4] = {gbuffer.baseMetal, gbuffer.normalHit,
                                       gbuffer.roughness, gbuffer.depth};
    glDeleteTextures(4, gbufferTextures);
    glDeleteTextures(1, &finalTarget.texture);
    const GLuint materialTextures[4] = {textures.baseColor, textures.metallic,
                                        textures.roughness, textures.normal};
    glDeleteTextures(4, materialTextures);
    return output;
}

#ifdef _WIN32
Image renderOpenGLDeferred(const MaterialLibrary& materials, const Camera& camera,
                           const std::vector<Light>& lights) {
    GlfwSession session;
    createContext(session);
    return renderOpenGLDeferredWithCurrentContext(
        materials, camera, lights,
        [](const char* name) -> GlProc { return glfwGetProcAddress(name); });
}
#endif

}  // namespace cr

