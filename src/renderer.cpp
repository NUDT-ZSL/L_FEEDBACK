#include "renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace preview {
namespace {

struct ClipVertex {
    glm::vec4 clip = glm::vec4(0.0f);
    glm::vec3 world = glm::vec3(0.0f);
    glm::vec3 normal = glm::vec3(0.0f);
    glm::vec2 uv = glm::vec2(0.0f);
    glm::vec3 tangent = glm::vec3(1.0f, 0.0f, 0.0f);
    float tangentSign = 1.0f;
};

struct ScreenVertex {
    glm::vec2 screen = glm::vec2(0.0f);
    float invW = 0.0f;
    ClipVertex attributes;
};

glm::vec3 fallbackTangent(const glm::vec3& normal) {
    const glm::vec3 reference =
        std::abs(normal.x) < 0.9f ? glm::vec3(1.0f, 0.0f, 0.0f)
                                  : glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 tangent = glm::cross(reference, normal);
    return glm::length(tangent) <= 1.0e-8f
               ? glm::vec3(1.0f, 0.0f, 0.0f)
               : glm::normalize(tangent);
}

ClipVertex interpolate(const ClipVertex& a, const ClipVertex& b, float t) {
    ClipVertex result;
    result.clip = glm::mix(a.clip, b.clip, t);
    result.world = glm::mix(a.world, b.world, t);
    result.normal = glm::mix(a.normal, b.normal, t);
    result.uv = glm::mix(a.uv, b.uv, t);
    result.tangent = glm::mix(a.tangent, b.tangent, t);
    result.tangentSign = glm::mix(a.tangentSign, b.tangentSign, t);
    return result;
}

std::vector<ClipVertex> clipPolygon(
    const std::vector<ClipVertex>& input,
    const glm::vec4& plane) {
    if (input.empty()) {
        return {};
    }

    std::vector<ClipVertex> output;
    ClipVertex previous = input.back();
    float previousDistance = glm::dot(plane, previous.clip);
    bool previousInside = previousDistance >= 0.0f;

    for (const ClipVertex& current : input) {
        const float currentDistance = glm::dot(plane, current.clip);
        const bool currentInside = currentDistance >= 0.0f;
        if (currentInside != previousInside) {
            const float denominator = previousDistance - currentDistance;
            if (std::abs(denominator) > 1.0e-12f) {
                output.push_back(interpolate(
                    previous,
                    current,
                    previousDistance / denominator));
            }
        }
        if (currentInside) {
            output.push_back(current);
        }
        previous = current;
        previousDistance = currentDistance;
        previousInside = currentInside;
    }
    return output;
}

std::vector<ClipVertex> clipToFrustum(const std::array<ClipVertex, 3>& triangle) {
    const std::array<glm::vec4, 6> planes = {
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(-1.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, -1.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
        glm::vec4(0.0f, 0.0f, -1.0f, 1.0f),
    };

    std::vector<ClipVertex> polygon = {triangle[0], triangle[1], triangle[2]};
    for (const glm::vec4& plane : planes) {
        polygon = clipPolygon(polygon, plane);
        if (polygon.size() < 3) {
            return {};
        }
    }
    return polygon;
}

float edgeFunction(const glm::vec2& a, const glm::vec2& b, const glm::vec2& point) {
    return (point.x - a.x) * (b.y - a.y) -
           (point.y - a.y) * (b.x - a.x);
}

bool triangleFacesCamera(
    const glm::vec3& cameraPosition,
    const glm::vec3& p0,
    const glm::vec3& p1,
    const glm::vec3& p2) {
    const glm::vec3 faceNormal = glm::cross(p1 - p0, p2 - p0);
    if (glm::length(faceNormal) <= 1.0e-12f) {
        return false;
    }
    const glm::vec3 viewDirection = cameraPosition - p0;
    return glm::dot(faceNormal, viewDirection) > 0.0f;
}

glm::vec3 shadeFragment(
    const RenderConfig& config,
    const Material& material,
    const glm::vec3& worldPosition,
    const glm::vec3& interpolatedNormal,
    const glm::vec2& uv,
    const glm::vec3& interpolatedTangent,
    float tangentSign) {
    glm::vec3 normal = interpolatedNormal;
    if (glm::length(normal) <= 1.0e-8f) {
        return config.ambient;
    }
    normal = glm::normalize(normal);

    const glm::vec3 viewVector = config.camera.position - worldPosition;
    if (glm::length(viewVector) <= 1.0e-8f) {
        return config.ambient;
    }
    const glm::vec3 viewDirection = glm::normalize(viewVector);
    if (config.doubleSided && glm::dot(normal, viewDirection) < 0.0f) {
        normal = -normal;
    }

    glm::vec3 tangent =
        interpolatedTangent - normal * glm::dot(interpolatedTangent, normal);
    tangent = glm::length(tangent) <= 1.0e-8f
                  ? fallbackTangent(normal)
                  : glm::normalize(tangent);
    const glm::vec3 bitangent =
        glm::normalize(glm::cross(normal, tangent)) * (tangentSign < 0.0f ? -1.0f : 1.0f);

    if (material.normalTexture.valid) {
        const glm::vec3 sampled = sampleBilinearClamp(material.normalTexture, uv);
        glm::vec3 tangentNormal = sampled * 2.0f - 1.0f;
        if (glm::length(tangentNormal) <= 1.0e-8f) {
            tangentNormal = glm::vec3(0.0f, 0.0f, 1.0f);
        } else {
            tangentNormal = glm::normalize(tangentNormal);
        }
        normal = glm::normalize(
            tangent * tangentNormal.x +
            bitangent * tangentNormal.y +
            normal * tangentNormal.z);
    }

    const glm::vec3 albedo = sampleBilinearClamp(material.diffuseTexture, uv) *
                             material.diffuseTint;
    glm::vec3 color = config.ambient * albedo;

    for (const LightConfig& light : config.lights) {
        glm::vec3 lightDirection = glm::vec3(0.0f);
        float attenuation = 1.0f;
        if (light.type == LightConfig::Type::Directional) {
            lightDirection = -light.direction;
        } else {
            const glm::vec3 toLight = light.position - worldPosition;
            const float distance = glm::length(toLight);
            if (distance <= 1.0e-8f) {
                continue;
            }
            lightDirection = toLight / distance;
            attenuation =
                1.0f /
                (light.attenuation.constant +
                 light.attenuation.linear * distance +
                 light.attenuation.quadratic * distance * distance);
        }

        const float nDotL = std::max(glm::dot(normal, lightDirection), 0.0f);
        if (nDotL <= 0.0f || light.intensity <= 0.0f) {
            continue;
        }
        const glm::vec3 lightEnergy =
            light.color * (light.intensity * attenuation);
        color += albedo * lightEnergy * nDotL;

        glm::vec3 halfway = lightDirection + viewDirection;
        if (glm::length(halfway) > 1.0e-8f) {
            halfway = glm::normalize(halfway);
            const float nDotH = std::max(glm::dot(normal, halfway), 0.0f);
            const float specular =
                std::pow(nDotH, material.shininess) * material.specularStrength;
            color += albedo * lightEnergy * specular;
        }
    }

    return glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f));
}

} // namespace

Image renderScene(const RenderConfig& config, const Model& model) {
    Image image;
    image.width = config.width;
    image.height = config.height;
    image.pixels.assign(
        static_cast<std::size_t>(config.width) *
            static_cast<std::size_t>(config.height),
        config.background);

    std::vector<float> inverseDepth(
        static_cast<std::size_t>(config.width) *
            static_cast<std::size_t>(config.height),
        0.0f);

    const float aspect =
        static_cast<float>(config.width) / static_cast<float>(config.height);
    const glm::mat4 view = glm::lookAt(
        config.camera.position,
        config.camera.target,
        config.camera.up);
    const glm::mat4 projection = glm::perspective(
        glm::radians(config.camera.verticalFovDegrees),
        aspect,
        config.camera.nearPlane,
        config.camera.farPlane);
    const glm::mat4 viewProjection = projection * view;

    const auto toClipVertex = [&viewProjection](const Vertex& vertex) {
        ClipVertex result;
        result.clip = viewProjection * glm::vec4(vertex.position, 1.0f);
        result.world = vertex.position;
        result.normal = vertex.normal;
        result.uv = vertex.uv;
        result.tangent = vertex.tangent;
        result.tangentSign = vertex.tangentSign;
        return result;
    };

    for (const Triangle& triangle : model.triangles) {
        if (triangle.materialIndex < 0 ||
            static_cast<std::size_t>(triangle.materialIndex) >= model.materials.size()) {
            continue;
        }
        const Vertex& v0 = model.vertices[triangle.indices[0]];
        const Vertex& v1 = model.vertices[triangle.indices[1]];
        const Vertex& v2 = model.vertices[triangle.indices[2]];
        if (!triangleFacesCamera(
                config.camera.position,
                v0.position,
                v1.position,
                v2.position) &&
            !config.doubleSided) {
            continue;
        }

        const std::array<ClipVertex, 3> clipTriangle = {
            toClipVertex(v0),
            toClipVertex(v1),
            toClipVertex(v2),
        };
        const std::vector<ClipVertex> clipped = clipToFrustum(clipTriangle);
        if (clipped.size() < 3) {
            continue;
        }

        for (std::size_t fan = 1; fan + 1 < clipped.size(); ++fan) {
            std::array<ClipVertex, 3> polygonTriangle = {
                clipped[0],
                clipped[fan],
                clipped[fan + 1],
            };
            std::array<ScreenVertex, 3> screen;
            bool valid = true;
            for (int i = 0; i < 3; ++i) {
                const glm::vec4& clip = polygonTriangle[static_cast<std::size_t>(i)].clip;
                if (clip.w <= 1.0e-8f) {
                    valid = false;
                    break;
                }
                const glm::vec3 ndc = glm::vec3(clip) / clip.w;
                screen[static_cast<std::size_t>(i)].screen = glm::vec2(
                    (ndc.x * 0.5f + 0.5f) * static_cast<float>(config.width),
                    (1.0f - (ndc.y * 0.5f + 0.5f)) *
                        static_cast<float>(config.height));
                screen[static_cast<std::size_t>(i)].invW = 1.0f / clip.w;
                screen[static_cast<std::size_t>(i)].attributes =
                    polygonTriangle[static_cast<std::size_t>(i)];
            }
            if (!valid) {
                continue;
            }
            const float area = edgeFunction(
                screen[0].screen,
                screen[1].screen,
                screen[2].screen);
            if (std::abs(area) <= 1.0e-10f) {
                continue;
            }

            int minX = static_cast<int>(
                std::floor(std::min({screen[0].screen.x, screen[1].screen.x, screen[2].screen.x})));
            int maxX = static_cast<int>(
                std::ceil(std::max({screen[0].screen.x, screen[1].screen.x, screen[2].screen.x})));
            int minY = static_cast<int>(
                std::floor(std::min({screen[0].screen.y, screen[1].screen.y, screen[2].screen.y})));
            int maxY = static_cast<int>(
                std::ceil(std::max({screen[0].screen.y, screen[1].screen.y, screen[2].screen.y})));
            minX = std::max(minX, 0);
            minY = std::max(minY, 0);
            maxX = std::min(maxX, config.width - 1);
            maxY = std::min(maxY, config.height - 1);
            if (minX > maxX || minY > maxY) {
                continue;
            }

            const Material& material =
                model.materials[static_cast<std::size_t>(triangle.materialIndex)];
            for (int y = minY; y <= maxY; ++y) {
                for (int x = minX; x <= maxX; ++x) {
                    const glm::vec2 point(
                        static_cast<float>(x) + 0.5f,
                        static_cast<float>(y) + 0.5f);
                    const float e0 = edgeFunction(
                        screen[1].screen, screen[2].screen, point);
                    const float e1 = edgeFunction(
                        screen[2].screen, screen[0].screen, point);
                    const float e2 = edgeFunction(
                        screen[0].screen, screen[1].screen, point);
                    const bool inside =
                        area > 0.0f
                            ? (e0 >= -1.0e-7f && e1 >= -1.0e-7f && e2 >= -1.0e-7f)
                            : (e0 <= 1.0e-7f && e1 <= 1.0e-7f && e2 <= 1.0e-7f);
                    if (!inside) {
                        continue;
                    }

                    const float barycentric0 = e0 / area;
                    const float barycentric1 = e1 / area;
                    const float barycentric2 = e2 / area;
                    const float interpolatedInverseDepth =
                        barycentric0 * screen[0].invW +
                        barycentric1 * screen[1].invW +
                        barycentric2 * screen[2].invW;
                    if (interpolatedInverseDepth <= 0.0f) {
                        continue;
                    }

                    const std::size_t pixelIndex =
                        static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(config.width) +
                        static_cast<std::size_t>(x);
                    if (interpolatedInverseDepth <= inverseDepth[pixelIndex]) {
                        continue;
                    }

                    const float weight0 =
                        barycentric0 * screen[0].invW / interpolatedInverseDepth;
                    const float weight1 =
                        barycentric1 * screen[1].invW / interpolatedInverseDepth;
                    const float weight2 =
                        barycentric2 * screen[2].invW / interpolatedInverseDepth;
                    const glm::vec3 worldPosition =
                        weight0 * screen[0].attributes.world +
                        weight1 * screen[1].attributes.world +
                        weight2 * screen[2].attributes.world;
                    const glm::vec3 normal =
                        weight0 * screen[0].attributes.normal +
                        weight1 * screen[1].attributes.normal +
                        weight2 * screen[2].attributes.normal;
                    const glm::vec2 uv =
                        weight0 * screen[0].attributes.uv +
                        weight1 * screen[1].attributes.uv +
                        weight2 * screen[2].attributes.uv;
                    const glm::vec3 tangent =
                        weight0 * screen[0].attributes.tangent +
                        weight1 * screen[1].attributes.tangent +
                        weight2 * screen[2].attributes.tangent;
                    const float tangentSign =
                        weight0 * screen[0].attributes.tangentSign +
                        weight1 * screen[1].attributes.tangentSign +
                        weight2 * screen[2].attributes.tangentSign;

                    inverseDepth[pixelIndex] = interpolatedInverseDepth;
                    image.pixels[pixelIndex] = shadeFragment(
                        config,
                        material,
                        worldPosition,
                        normal,
                        uv,
                        tangent,
                        tangentSign);
                }
            }
        }
    }

    return image;
}

} // namespace preview
