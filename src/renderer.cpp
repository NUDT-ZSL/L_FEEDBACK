#include "renderer.h"

#include <glad/gl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace pbr {
namespace {

const char* kVertexShader = R"GLSL(
#version 450 core
out vec2 vNdc;

void main() {
    vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    vec2 position = positions[gl_VertexID];
    vNdc = position;
    gl_Position = vec4(position, 0.0, 1.0);
}
)GLSL";

const char* kFragmentShader = R"GLSL(
#version 450 core

const float PI = 3.14159265358979323846;
const float INV_PI = 0.31830988618379067154;
const int MAX_MATERIALS = 16;
const int MAX_SPHERES = 24;
const int MAX_POINT_LIGHTS = 8;
const int MAX_RECT_LIGHTS = 4;
const int SH_COEFFICIENTS = 9;

layout(location = 0) out vec4 fragColor;

uniform vec2 uResolution;
uniform vec3 uCameraPosition;
uniform vec3 uCameraForward;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;
uniform float uFocalLengthMm;
uniform float uSensorWidthMm;
uniform float uSensorHeightMm;
uniform float uFocusDistance;
uniform float uApertureRadius;
uniform int uSamples;
uniform int uSubframeIndex;
uniform uint uSeed;
uniform bool uAuxPass;
uniform float uFrameWeight;

uniform int uSphereCount;
uniform vec4 uSpherePosRadius[MAX_SPHERES];
uniform int uSphereMaterial[MAX_SPHERES];

uniform int uMaterialCount;
uniform vec4 uMaterialBaseColor[MAX_MATERIALS];
uniform vec4 uMaterialParams[MAX_MATERIALS];

uniform int uPointLightCount;
uniform vec4 uPointLightPosition[MAX_POINT_LIGHTS];
uniform vec4 uPointLightColorPower[MAX_POINT_LIGHTS];

uniform int uRectLightCount;
uniform vec4 uRectLightCenter[MAX_RECT_LIGHTS];
uniform vec4 uRectLightNormal[MAX_RECT_LIGHTS];
uniform vec4 uRectLightTangent[MAX_RECT_LIGHTS];
uniform vec4 uRectLightBitangent[MAX_RECT_LIGHTS];
uniform vec4 uRectLightRadiance[MAX_RECT_LIGHTS];
uniform float uRectLightCdf[MAX_RECT_LIGHTS];

uniform vec3 uRadianceSH[SH_COEFFICIENTS];
uniform vec3 uIrradianceSH[SH_COEFFICIENTS];

uniform vec3 uEnvZenith;
uniform vec3 uEnvHorizon;
uniform vec3 uEnvGround;
uniform vec3 uEnvSunDirection;
uniform vec3 uEnvSunRadiance;
uniform float uEnvSunRadiusDegrees;

struct MaterialData {
    vec3 baseColor;
    float ior;
    float metallic;
    float roughness;
    float anisotropy;
    float rotation;
};

uint pcgHash(uint value) {
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint rngState;

float randomFloat() {
    rngState = rngState * 747796405u + 2891336453u;
    uint word = ((rngState >> ((rngState >> 28u) + 4u)) ^ rngState) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word & 0x00FFFFFFu) * (1.0 / 16777216.0);
}

vec2 sampleDisk(vec2 uniformSample) {
    float radius = sqrt(clamp(uniformSample.x, 0.0, 1.0));
    float angle = 2.0 * PI * uniformSample.y;
    return vec2(radius * cos(angle), radius * sin(angle));
}

vec3 environmentRadiance(vec3 direction) {
    vec3 dir = normalize(direction);
    vec3 color;
    if (dir.y >= 0.0) {
        float t = pow(clamp(dir.y, 0.0, 1.0), 0.5);
        color = mix(uEnvHorizon, uEnvZenith, t);
    } else {
        float t = pow(clamp(-dir.y, 0.0, 1.0), 0.45);
        color = mix(uEnvHorizon, uEnvGround, t);
    }
    float cosine = dot(dir, normalize(uEnvSunDirection));
    float inner = cos(radians(uEnvSunRadiusDegrees));
    float outer = cos(radians(uEnvSunRadiusDegrees * 1.35));
    color += uEnvSunRadiance * smoothstep(outer, inner, cosine);
    return max(color, vec3(0.0));
}

vec3 evaluateSH(vec3 coefficients[SH_COEFFICIENTS], vec3 direction) {
    return coefficients[0] * 0.2820947917738781
         + coefficients[1] * (0.4886025119029199 * direction.y)
         + coefficients[2] * (0.4886025119029199 * direction.z)
         + coefficients[3] * (0.4886025119029199 * direction.x)
         + coefficients[4] * (1.0925484305920792 * direction.x * direction.y)
         + coefficients[5] * (1.0925484305920792 * direction.y * direction.z)
         + coefficients[6] * (0.3153915652525200 * (3.0 * direction.z * direction.z - 1.0))
         + coefficients[7] * (1.0925484305920792 * direction.x * direction.z)
         + coefficients[8] * (0.5462742152960396 * (direction.x * direction.x - direction.y * direction.y));
}

MaterialData loadMaterial(int materialId) {
    MaterialData material;
    vec4 base = uMaterialBaseColor[materialId];
    vec4 parameters = uMaterialParams[materialId];
    material.baseColor = base.rgb;
    material.ior = base.w;
    material.metallic = parameters.x;
    material.roughness = max(parameters.y, 0.02);
    material.anisotropy = clamp(parameters.z, 0.0, 1.0);
    material.rotation = parameters.w;
    return material;
}

bool intersectSphere(vec3 rayOrigin,
                     vec3 rayDirection,
                     vec3 center,
                     float radius,
                     out float hitDistance) {
    vec3 oc = rayOrigin - center;
    float b = dot(oc, rayDirection);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - c;
    if (discriminant < 0.0) {
        return false;
    }
    float root = sqrt(discriminant);
    float first = -b - root;
    float second = -b + root;
    if (first > 1.0e-4) {
        hitDistance = first;
        return true;
    }
    if (second > 1.0e-4) {
        hitDistance = second;
        return true;
    }
    return false;
}

bool traceScene(vec3 rayOrigin,
                vec3 rayDirection,
                out float hitDistance,
                out vec3 normal,
                out int materialId) {
    float nearest = 1.0e30;
    int nearestMaterial = -1;
    vec3 nearestNormal = vec3(0.0);
    for (int i = 0; i < uSphereCount; ++i) {
        vec4 sphere = uSpherePosRadius[i];
        float distance;
        if (intersectSphere(rayOrigin, rayDirection, sphere.xyz, sphere.w, distance) && distance < nearest) {
            nearest = distance;
            nearestMaterial = uSphereMaterial[i];
            nearestNormal = normalize(rayOrigin + rayDirection * distance - sphere.xyz);
        }
    }
    hitDistance = nearest;
    normal = nearestNormal;
    materialId = nearestMaterial;
    return nearestMaterial >= 0;
}

bool occluded(vec3 position, vec3 normal, vec3 target) {
    vec3 origin = position + normal * 2.0e-4;
    vec3 delta = target - origin;
    float maximum = length(delta);
    if (maximum <= 2.0e-4) {
        return false;
    }
    vec3 direction = delta / maximum;
    for (int i = 0; i < uSphereCount; ++i) {
        float distance;
        if (intersectSphere(origin, direction, uSpherePosRadius[i].xyz, uSpherePosRadius[i].w, distance) &&
            distance < maximum - 1.0e-4) {
            return true;
        }
    }
    return false;
}

vec3 cameraRay(vec2 pixelCoordinate, vec2 uniformLensSample) {
    vec2 ndc = pixelCoordinate / uResolution * 2.0 - 1.0;
    vec3 direction = normalize(
        uCameraForward * uFocalLengthMm
        + uCameraRight * (ndc.x * 0.5 * uSensorWidthMm)
        + uCameraUp * (ndc.y * 0.5 * uSensorHeightMm));
    float forwardCosine = max(dot(direction, uCameraForward), 0.001);
    vec3 focalPoint = uCameraPosition + direction * (uFocusDistance / forwardCosine);
    vec2 lensPoint = sampleDisk(uniformLensSample) * uApertureRadius;
    vec3 lensOrigin = uCameraPosition + uCameraRight * lensPoint.x + uCameraUp * lensPoint.y;
    return normalize(focalPoint - lensOrigin);
}

void buildTangentFrame(vec3 normal, float rotation, out vec3 tangent, out vec3 bitangent) {
    vec3 reference = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 baseTangent = normalize(cross(reference, normal));
    vec3 baseBitangent = cross(normal, baseTangent);
    float cosine = cos(rotation);
    float sine = sin(rotation);
    tangent = baseTangent * cosine + baseBitangent * sine;
    bitangent = -baseTangent * sine + baseBitangent * cosine;
}

float smithG1(float nox, float alpha) {
    float alphaSquared = alpha * alpha;
    return 2.0 * nox / max(nox + sqrt(alphaSquared + (1.0 - alphaSquared) * nox * nox), 1.0e-6);
}

vec3 fresnelSchlick(float cosine, vec3 f0) {
    return f0 + (vec3(1.0) - f0) * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosine, vec3 f0, float roughness) {
    vec3 f90 = max(vec3(1.0 - roughness), f0);
    return f0 + (f90 - f0) * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

void evaluateBrdf(MaterialData material,
                  vec3 normal,
                  vec3 viewDirection,
                  vec3 lightDirection,
                  vec3 tangent,
                  vec3 bitangent,
                  out vec3 diffuse,
                  out vec3 specular) {
    float noL = dot(normal, lightDirection);
    float noV = max(dot(normal, viewDirection), 1.0e-4);
    if (noL <= 0.0) {
        diffuse = vec3(0.0);
        specular = vec3(0.0);
        return;
    }
    vec3 halfVector = normalize(viewDirection + lightDirection);
    float noH = max(dot(normal, halfVector), 0.0);
    float voH = max(dot(viewDirection, halfVector), 0.0);

    float alpha = material.roughness * material.roughness;
    float aspect = sqrt(max(1.0 - 0.9 * material.anisotropy, 0.05));
    float alphaT = max(alpha / aspect, 0.001);
    float alphaB = max(alpha * aspect, 0.001);
    float ht = dot(halfVector, tangent);
    float hb = dot(halfVector, bitangent);
    float hn = noH;
    float denominator = ht * ht / (alphaT * alphaT)
                      + hb * hb / (alphaB * alphaB)
                      + hn * hn;
    float distribution = 1.0 / max(PI * alphaT * alphaB * denominator * denominator, 1.0e-9);

    float geometry = smithG1(noV, alpha) * smithG1(noL, alpha);
    float dielectricF0 = pow((material.ior - 1.0) / (material.ior + 1.0), 2.0);
    vec3 f0 = mix(vec3(dielectricF0), material.baseColor, material.metallic);
    vec3 fresnel = fresnelSchlick(voH, f0);

    specular = distribution * geometry * fresnel / max(4.0 * noV * noL, 1.0e-6);
    diffuse = material.baseColor * (1.0 - material.metallic) * (1.0 - fresnel) * INV_PI;
}

vec3 shadeSurface(vec3 position, vec3 normal, vec3 viewDirection, MaterialData material) {
    vec3 tangent;
    vec3 bitangent;
    buildTangentFrame(normal, material.rotation, tangent, bitangent);

    vec3 result = vec3(0.0);
    for (int i = 0; i < uPointLightCount; ++i) {
        vec3 toLight = uPointLightPosition[i].xyz - position;
        float distanceSquared = dot(toLight, toLight);
        if (distanceSquared < 1.0e-8) {
            continue;
        }
        vec3 lightDirection = toLight * inversesqrt(distanceSquared);
        if (dot(normal, lightDirection) <= 0.0 || occluded(position, normal, uPointLightPosition[i].xyz)) {
            continue;
        }
        vec3 diffuse;
        vec3 specular;
        evaluateBrdf(material, normal, viewDirection, lightDirection, tangent, bitangent, diffuse, specular);
        vec3 radiance =
            uPointLightColorPower[i].rgb * uPointLightColorPower[i].w / (4.0 * PI * distanceSquared);
        result += (diffuse + specular) * radiance * dot(normal, lightDirection);
    }

    if (uRectLightCount > 0) {
        float selection = randomFloat();
        int lightIndex = uRectLightCount - 1;
        for (int i = 0; i < uRectLightCount; ++i) {
            if (selection <= uRectLightCdf[i]) {
                lightIndex = i;
                break;
            }
        }
        float previousCdf = lightIndex == 0 ? 0.0 : uRectLightCdf[lightIndex - 1];
        float selectionProbability = max(uRectLightCdf[lightIndex] - previousCdf, 1.0e-5);

        vec4 tangentData = uRectLightTangent[lightIndex];
        vec4 bitangentData = uRectLightBitangent[lightIndex];
        vec3 samplePoint = uRectLightCenter[lightIndex].xyz
                         + tangentData.xyz * ((2.0 * randomFloat() - 1.0) * tangentData.w)
                         + bitangentData.xyz * ((2.0 * randomFloat() - 1.0) * bitangentData.w);
        vec3 toLight = samplePoint - position;
        float distanceSquared = dot(toLight, toLight);
        if (distanceSquared > 1.0e-8) {
            vec3 lightDirection = toLight * inversesqrt(distanceSquared);
            float cosLight = dot(-lightDirection, uRectLightNormal[lightIndex].xyz);
            float noL = dot(normal, lightDirection);
            if (cosLight > 0.0 && noL > 0.0 && !occluded(position, normal, samplePoint)) {
                vec3 diffuse;
                vec3 specular;
                evaluateBrdf(material, normal, viewDirection, lightDirection, tangent, bitangent, diffuse, specular);
                float area = 4.0 * tangentData.w * bitangentData.w;
                vec3 contribution = (diffuse + specular) * uRectLightRadiance[lightIndex].rgb
                                  * (area * noL * cosLight / (distanceSquared * selectionProbability));
                result += contribution;
            }
        }
    }

    float noV = max(dot(normal, viewDirection), 1.0e-4);
    vec3 irradiance = max(evaluateSH(uIrradianceSH, normal), vec3(0.0));
    vec3 diffuseIndirect = material.baseColor * (1.0 - material.metallic) * irradiance * INV_PI;
    vec3 reflected = reflect(-viewDirection, normal);
    vec3 radianceIndirect = max(evaluateSH(uRadianceSH, reflected), vec3(0.0));
    float dielectricF0 = pow((material.ior - 1.0) / (material.ior + 1.0), 2.0);
    vec3 f0 = mix(vec3(dielectricF0), material.baseColor, material.metallic);
    vec3 environmentFresnel = fresnelSchlickRoughness(noV, f0, material.roughness);
    vec3 specularIndirect = radianceIndirect * environmentFresnel;
    return result + diffuseIndirect + specularIndirect;
}

void main() {
    uvec2 pixel = uvec2(gl_FragCoord.xy);
    rngState = pcgHash(uSeed ^ (pixel.x * 1973u) ^ (pixel.y * 9277u) ^ (uint(uSubframeIndex) * 26699u));

    if (uAuxPass) {
        vec3 rayDirection = cameraRay(gl_FragCoord.xy, vec2(0.0));
        float distance;
        vec3 normal;
        int materialId;
        if (traceScene(uCameraPosition, rayDirection, distance, normal, materialId)) {
            vec3 reflected = reflect(rayDirection, normal);
            fragColor = vec4(reflected, float(materialId));
        } else {
            fragColor = vec4(0.0, 0.0, 0.0, -1.0);
        }
        return;
    }

    vec3 sum = vec3(0.0);
    for (int sampleIndex = 0; sampleIndex < uSamples; ++sampleIndex) {
        vec2 pixelJitter = vec2(randomFloat(), randomFloat());
        vec2 lensJitter = vec2(randomFloat(), randomFloat());
        vec3 rayDirection = cameraRay(gl_FragCoord.xy + pixelJitter, lensJitter);
        float distance;
        vec3 normal;
        int materialId;
        if (traceScene(uCameraPosition, rayDirection, distance, normal, materialId)) {
            vec3 position = uCameraPosition + rayDirection * distance;
            MaterialData material = loadMaterial(materialId);
            sum += shadeSurface(position, normal, -rayDirection, material);
        } else {
            sum += environmentRadiance(rayDirection);
        }
    }
    fragColor = vec4(sum / float(uSamples) * uFrameWeight, 1.0);
}
)GLSL";

bool checkGlError(const char* stage, std::string& error) {
    const GLenum code = glGetError();
    if (code == GL_NO_ERROR) {
        return true;
    }
    std::ostringstream output;
    output << "OpenGL error 0x" << std::hex << code << " during " << stage;
    error = output.str();
    return false;
}

GLuint compileShader(GLenum type, const char* source, std::string& error) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint success = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<size_t>(std::max(length, 1)), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        error = "shader compilation failed:\n" + log;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

GLuint buildProgram(std::string& error) {
    const GLuint vertex = compileShader(GL_VERTEX_SHADER, kVertexShader, error);
    if (vertex == 0) {
        return 0;
    }
    const GLuint fragment = compileShader(GL_FRAGMENT_SHADER, kFragmentShader, error);
    if (fragment == 0) {
        glDeleteShader(vertex);
        return 0;
    }
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint success = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == GL_FALSE) {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<size_t>(std::max(length, 1)), '\0');
        glGetProgramInfoLog(program, length, nullptr, log.data());
        error = "shader program link failed:\n" + log;
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

GLuint createFloatTexture(int width, int height, std::string& error) {
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (!checkGlError("float texture creation", error)) {
        glDeleteTextures(1, &texture);
        return 0;
    }
    return texture;
}

GLuint createFramebuffer(GLuint texture, std::string& error) {
    GLuint framebuffer = 0;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        error = "cannot create a complete floating-point framebuffer";
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &framebuffer);
        return 0;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (!checkGlError("framebuffer creation", error)) {
        glDeleteFramebuffers(1, &framebuffer);
        return 0;
    }
    return framebuffer;
}

GLint uniformLocation(GLuint program, const char* name, std::string& error) {
    const GLint location = glGetUniformLocation(program, name);
    if (location < 0) {
        error = "required shader uniform '" + std::string(name) + "' was not found";
    }
    return location;
}

void setInt(GLuint program, const char* name, int value, std::string& error) {
    const GLint location = uniformLocation(program, name, error);
    if (location >= 0) {
        glUniform1i(location, value);
    }
}

void setFloat(GLuint program, const char* name, float value, std::string& error) {
    const GLint location = uniformLocation(program, name, error);
    if (location >= 0) {
        glUniform1f(location, value);
    }
}

void setVec2(GLuint program, const char* name, float x, float y, std::string& error) {
    const GLint location = uniformLocation(program, name, error);
    if (location >= 0) {
        glUniform2f(location, x, y);
    }
}

void setVec3(GLuint program, const char* name, const Vec3& value, std::string& error) {
    const GLint location = uniformLocation(program, name, error);
    if (location >= 0) {
        glUniform3f(location, value.x, value.y, value.z);
    }
}

void setVec4Array(GLuint program,
                  const char* name,
                  const std::vector<float>& values,
                  std::string& error) {
    const GLint location = uniformLocation(program, name, error);
    if (location >= 0) {
        glUniform4fv(location, static_cast<GLsizei>(values.size() / 4), values.data());
    }
}

void setFloatArray(GLuint program,
                   const char* name,
                   const std::vector<float>& values,
                   std::string& error) {
    const GLint location = uniformLocation(program, name, error);
    if (location >= 0) {
        glUniform1fv(location, static_cast<GLsizei>(values.size()), values.data());
    }
}

void setIntArray(GLuint program,
                 const char* name,
                 const std::vector<int>& values,
                 std::string& error) {
    const GLint location = uniformLocation(program, name, error);
    if (location >= 0) {
        glUniform1iv(location, static_cast<GLsizei>(values.size()), values.data());
    }
}

void setSHArray(GLuint program,
                const char* name,
                const std::array<Vec3, 9>& values,
                std::string& error) {
    const GLint location = uniformLocation(program, name, error);
    if (location >= 0) {
        glUniform3fv(location, 9, reinterpret_cast<const GLfloat*>(values.data()));
    }
}

void setSpherePositions(GLuint program,
                        const Scene& scene,
                        float normalizedTime,
                        std::string& error) {
    std::vector<float> positions(static_cast<size_t>(kMaxSpheres) * 4, 0.0f);
    for (size_t i = 0; i < scene.spheres.size(); ++i) {
        const Vec3 position = spherePositionAt(scene, static_cast<int>(i), normalizedTime);
        positions[i * 4 + 0] = position.x;
        positions[i * 4 + 1] = position.y;
        positions[i * 4 + 2] = position.z;
        positions[i * 4 + 3] = scene.spheres[i].radius;
    }
    setVec4Array(program, "uSpherePosRadius[0]", positions, error);
}

void uploadSceneUniforms(GLuint program,
                         const Scene& scene,
                         const EnvironmentSH& sh,
                         const RenderSettings& settings,
                         std::string& error) {
    const Vec3 forward = normalize(scene.camera.target - scene.camera.position);
    const Vec3 right = normalize(cross(forward, scene.camera.up));
    const Vec3 up = normalize(cross(right, forward));
    const float apertureRadius = scene.camera.focalLengthMm / (2.0f * scene.camera.fNumber) / 1000.0f;

    setVec2(program, "uResolution", static_cast<float>(kResolution), static_cast<float>(kResolution), error);
    setVec3(program, "uCameraPosition", scene.camera.position, error);
    setVec3(program, "uCameraForward", forward, error);
    setVec3(program, "uCameraRight", right, error);
    setVec3(program, "uCameraUp", up, error);
    setFloat(program, "uFocalLengthMm", scene.camera.focalLengthMm, error);
    setFloat(program, "uSensorWidthMm", scene.camera.sensorWidthMm, error);
    setFloat(program, "uSensorHeightMm", scene.camera.sensorHeightMm, error);
    setFloat(program, "uFocusDistance", scene.camera.focusDistance, error);
    setFloat(program, "uApertureRadius", apertureRadius, error);
    setInt(program, "uSamples", settings.samples, error);
    setFloat(program, "uFrameWeight", 1.0f / static_cast<float>(settings.subframes), error);
    const GLint seedLocation = uniformLocation(program, "uSeed", error);
    if (seedLocation >= 0) {
        glUniform1ui(seedLocation, settings.seed);
    }

    setInt(program, "uSphereCount", static_cast<int>(scene.spheres.size()), error);
    std::vector<int> materialIds(static_cast<size_t>(kMaxSpheres), 0);
    for (size_t i = 0; i < scene.spheres.size(); ++i) {
        materialIds[i] = scene.spheres[i].materialId;
    }
    setIntArray(program, "uSphereMaterial[0]", materialIds, error);
    setSpherePositions(program, scene, 0.0f, error);

    std::vector<float> baseColors(static_cast<size_t>(kMaxMaterials) * 4, 0.0f);
    std::vector<float> parameters(static_cast<size_t>(kMaxMaterials) * 4, 0.0f);
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        const Material& material = scene.materials[i];
        baseColors[i * 4 + 0] = material.baseColor.x;
        baseColors[i * 4 + 1] = material.baseColor.y;
        baseColors[i * 4 + 2] = material.baseColor.z;
        baseColors[i * 4 + 3] = material.ior;
        parameters[i * 4 + 0] = material.metallic;
        parameters[i * 4 + 1] = material.roughness;
        parameters[i * 4 + 2] = material.anisotropy;
        parameters[i * 4 + 3] = material.anisotropyRotationRadians;
    }
    setVec4Array(program, "uMaterialBaseColor[0]", baseColors, error);
    setVec4Array(program, "uMaterialParams[0]", parameters, error);

    setInt(program, "uPointLightCount", static_cast<int>(scene.pointLights.size()), error);
    std::vector<float> pointPositions(static_cast<size_t>(kMaxPointLights) * 4, 0.0f);
    std::vector<float> pointColors(static_cast<size_t>(kMaxPointLights) * 4, 0.0f);
    for (size_t i = 0; i < scene.pointLights.size(); ++i) {
        const PointLight& light = scene.pointLights[i];
        pointPositions[i * 4 + 0] = light.position.x;
        pointPositions[i * 4 + 1] = light.position.y;
        pointPositions[i * 4 + 2] = light.position.z;
        pointColors[i * 4 + 0] = light.color.x;
        pointColors[i * 4 + 1] = light.color.y;
        pointColors[i * 4 + 2] = light.color.z;
        pointColors[i * 4 + 3] = light.power;
    }
    setVec4Array(program, "uPointLightPosition[0]", pointPositions, error);
    setVec4Array(program, "uPointLightColorPower[0]", pointColors, error);

    setInt(program, "uRectLightCount", static_cast<int>(scene.rectLights.size()), error);
    std::vector<float> rectCenters(static_cast<size_t>(kMaxRectLights) * 4, 0.0f);
    std::vector<float> rectNormals(static_cast<size_t>(kMaxRectLights) * 4, 0.0f);
    std::vector<float> rectTangents(static_cast<size_t>(kMaxRectLights) * 4, 0.0f);
    std::vector<float> rectBitangents(static_cast<size_t>(kMaxRectLights) * 4, 0.0f);
    std::vector<float> rectRadiance(static_cast<size_t>(kMaxRectLights) * 4, 0.0f);
    std::vector<float> rectCdf(static_cast<size_t>(kMaxRectLights), 0.0f);
    double rectWeightSum = 0.0;
    for (size_t i = 0; i < scene.rectLights.size(); ++i) {
        const RectLight& light = scene.rectLights[i];
        rectCenters[i * 4 + 0] = light.center.x;
        rectCenters[i * 4 + 1] = light.center.y;
        rectCenters[i * 4 + 2] = light.center.z;
        rectNormals[i * 4 + 0] = light.normal.x;
        rectNormals[i * 4 + 1] = light.normal.y;
        rectNormals[i * 4 + 2] = light.normal.z;
        rectTangents[i * 4 + 0] = light.tangent.x;
        rectTangents[i * 4 + 1] = light.tangent.y;
        rectTangents[i * 4 + 2] = light.tangent.z;
        rectTangents[i * 4 + 3] = light.halfWidth;
        rectBitangents[i * 4 + 0] = light.bitangent.x;
        rectBitangents[i * 4 + 1] = light.bitangent.y;
        rectBitangents[i * 4 + 2] = light.bitangent.z;
        rectBitangents[i * 4 + 3] = light.halfHeight;
        rectRadiance[i * 4 + 0] = light.radiance.x;
        rectRadiance[i * 4 + 1] = light.radiance.y;
        rectRadiance[i * 4 + 2] = light.radiance.z;
        const double area = 4.0 * static_cast<double>(light.halfWidth) * static_cast<double>(light.halfHeight);
        const double weight = static_cast<double>(luminance(light.radiance)) * area;
        rectWeightSum += weight;
        rectCdf[i] = static_cast<float>(rectWeightSum);
    }
    for (size_t i = 0; i < scene.rectLights.size(); ++i) {
        rectCdf[i] = rectWeightSum > 0.0 ? static_cast<float>(rectCdf[i] / rectWeightSum) : 1.0f;
    }
    if (!scene.rectLights.empty()) {
        rectCdf.back() = 1.0f;
    }
    setVec4Array(program, "uRectLightCenter[0]", rectCenters, error);
    setVec4Array(program, "uRectLightNormal[0]", rectNormals, error);
    setVec4Array(program, "uRectLightTangent[0]", rectTangents, error);
    setVec4Array(program, "uRectLightBitangent[0]", rectBitangents, error);
    setVec4Array(program, "uRectLightRadiance[0]", rectRadiance, error);
    setFloatArray(program, "uRectLightCdf[0]", rectCdf, error);

    setSHArray(program, "uRadianceSH[0]", sh.radiance, error);
    setSHArray(program, "uIrradianceSH[0]", sh.irradiance, error);
    setVec3(program, "uEnvZenith", scene.environment.zenith, error);
    setVec3(program, "uEnvHorizon", scene.environment.horizon, error);
    setVec3(program, "uEnvGround", scene.environment.ground, error);
    setVec3(program, "uEnvSunDirection", normalize(scene.environment.sunDirection), error);
    setVec3(program, "uEnvSunRadiance", scene.environment.sunRadiance, error);
    setFloat(program, "uEnvSunRadiusDegrees", scene.environment.sunAngularRadiusDegrees, error);
}

void flipImage(const std::vector<float>& source, int width, int height, Image& image) {
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int y = 0; y < height; ++y) {
        const int sourceY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            const size_t sourceIndex =
                (static_cast<size_t>(sourceY) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4u;
            image.pixels[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] = {
                source[sourceIndex + 0],
                source[sourceIndex + 1],
                source[sourceIndex + 2],
            };
        }
    }
}

void flipAuxiliary(const std::vector<float>& source, int width, int height, Image& image) {
    image.materialIds.assign(static_cast<size_t>(width) * static_cast<size_t>(height), -1);
    image.reflectionDirections.assign(static_cast<size_t>(width) * static_cast<size_t>(height), Vec3{});
    for (int y = 0; y < height; ++y) {
        const int sourceY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            const size_t sourceIndex =
                (static_cast<size_t>(sourceY) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4u;
            const size_t destinationIndex =
                static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const int materialId = static_cast<int>(std::lround(source[sourceIndex + 3]));
            image.materialIds[destinationIndex] =
                materialId >= 0 && materialId < kMaxMaterials ? materialId : -1;
            image.reflectionDirections[destinationIndex] = {
                source[sourceIndex + 0],
                source[sourceIndex + 1],
                source[sourceIndex + 2],
            };
        }
    }
}

void denoiseDof(Image& image, int samples) {
    if (image.width <= 0 || image.height <= 0) {
        return;
    }
    const int radius = samples < 64 ? 3 : 2;
    const std::vector<Vec3> input = image.pixels;
    const std::vector<int> materials = image.materialIds;
    const float sigmaSquared = 1.25f * 1.25f;
    std::vector<Vec3> output = input;
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const size_t centerIndex =
                static_cast<size_t>(y) * static_cast<size_t>(image.width) + static_cast<size_t>(x);
            const int centerMaterial = materials[centerIndex];
            if (centerMaterial < 0) {
                continue;
            }
            const float centerLuminance = luminance(input[centerIndex]);
            Vec3 sum{};
            float weightSum = 0.0f;
            for (int dy = -radius; dy <= radius; ++dy) {
                const int sampleY = y + dy;
                if (sampleY < 0 || sampleY >= image.height) {
                    continue;
                }
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int sampleX = x + dx;
                    if (sampleX < 0 || sampleX >= image.width) {
                        continue;
                    }
                    const size_t sampleIndex =
                        static_cast<size_t>(sampleY) * static_cast<size_t>(image.width) +
                        static_cast<size_t>(sampleX);
                    if (materials[sampleIndex] != centerMaterial) {
                        continue;
                    }
                    const float spatialWeight =
                        std::exp(-(static_cast<float>(dx * dx + dy * dy)) / (2.0f * sigmaSquared));
                    const float sampleLuminance = luminance(input[sampleIndex]);
                    const float rangeWeight =
                        std::exp(-std::fabs(sampleLuminance - centerLuminance) /
                                 (0.15f + 0.20f * std::max(centerLuminance, 0.0f)));
                    const float weight = spatialWeight * rangeWeight;
                    sum += input[sampleIndex] * weight;
                    weightSum += weight;
                }
            }
            if (weightSum > 0.0f) {
                output[centerIndex] = sum / weightSum;
            }
        }
    }
    image.pixels.swap(output);
}

}  // namespace

Renderer::Renderer() = default;

Renderer::~Renderer() {
    if (accumulationFbo_ != 0) {
        glDeleteFramebuffers(1, &accumulationFbo_);
    }
    if (auxiliaryFbo_ != 0) {
        glDeleteFramebuffers(1, &auxiliaryFbo_);
    }
    if (accumulationTexture_ != 0) {
        glDeleteTextures(1, &accumulationTexture_);
    }
    if (auxiliaryTexture_ != 0) {
        glDeleteTextures(1, &auxiliaryTexture_);
    }
    if (vao_ != 0) {
        glDeleteVertexArrays(1, &vao_);
    }
    if (program_ != 0) {
        glDeleteProgram(program_);
    }
}

bool Renderer::initialize(std::string& error) {
    program_ = buildProgram(error);
    if (program_ == 0) {
        return false;
    }
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_FRAMEBUFFER_SRGB);

    accumulationTexture_ = createFloatTexture(kResolution, kResolution, error);
    if (accumulationTexture_ == 0) {
        return false;
    }
    auxiliaryTexture_ = createFloatTexture(kResolution, kResolution, error);
    if (auxiliaryTexture_ == 0) {
        return false;
    }
    accumulationFbo_ = createFramebuffer(accumulationTexture_, error);
    if (accumulationFbo_ == 0) {
        return false;
    }
    auxiliaryFbo_ = createFramebuffer(auxiliaryTexture_, error);
    if (auxiliaryFbo_ == 0) {
        return false;
    }
    initialized_ = true;
    return true;
}

bool Renderer::render(const Scene& scene,
                      const RenderSettings& settings,
                      Image& image,
                      std::string& error) {
    if (!initialized_) {
        error = "renderer was not initialized";
        return false;
    }
    const EnvironmentSH environmentSh = computeEnvironmentSH(scene.environment);

    glUseProgram(program_);
    glBindVertexArray(vao_);
    uploadSceneUniforms(program_, scene, environmentSh, settings, error);
    if (!error.empty()) {
        return false;
    }
    setInt(program_, "uAuxPass", 0, error);
    if (!checkGlError("uniform upload", error)) {
        return false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, accumulationFbo_);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glViewport(0, 0, kResolution, kResolution);
    glDisable(GL_BLEND);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE);

    for (int subframe = 0; subframe < settings.subframes; ++subframe) {
        const float normalizedTime =
            (static_cast<float>(subframe) + 0.5f) / static_cast<float>(settings.subframes);
        setSpherePositions(program_, scene, normalizedTime, error);
        setInt(program_, "uSubframeIndex", subframe, error);
        if (!error.empty()) {
            return false;
        }
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glDisable(GL_BLEND);
    glFinish();
    if (!checkGlError("main accumulation pass", error)) {
        return false;
    }

    std::vector<float> colorPixels(
        static_cast<size_t>(kResolution) * static_cast<size_t>(kResolution) * 4u);
    glBindFramebuffer(GL_FRAMEBUFFER, accumulationFbo_);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, kResolution, kResolution, GL_RGBA, GL_FLOAT, colorPixels.data());
    if (!checkGlError("framebuffer color readback", error)) {
        return false;
    }

    setInt(program_, "uAuxPass", 1, error);
    setSpherePositions(program_, scene, 0.5f, error);
    setInt(program_, "uSubframeIndex", 0, error);
    if (!error.empty()) {
        return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, auxiliaryFbo_);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glViewport(0, 0, kResolution, kResolution);
    glClearColor(0.0f, 0.0f, 0.0f, -1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    if (!checkGlError("auxiliary framebuffer pass", error)) {
        return false;
    }
    std::vector<float> auxiliaryPixels(
        static_cast<size_t>(kResolution) * static_cast<size_t>(kResolution) * 4u);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, kResolution, kResolution, GL_RGBA, GL_FLOAT, auxiliaryPixels.data());
    if (!checkGlError("framebuffer auxiliary readback", error)) {
        return false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    setInt(program_, "uAuxPass", 0, error);
    glUseProgram(0);

    flipImage(colorPixels, kResolution, kResolution, image);
    flipAuxiliary(auxiliaryPixels, kResolution, kResolution, image);
    denoiseDof(image, settings.samples);
    return true;
}

}  // namespace pbr
