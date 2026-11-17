#include "scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <sstream>

#include <glm/gtc/constants.hpp>

#include "error.h"

namespace sr {
namespace {

constexpr double kPi = glm::pi<double>();

struct Influence {
    std::array<std::uint32_t, kMaxBoneInfluences> indices{0, 0, 0, 0};
    std::array<double, kMaxBoneInfluences> weights{1.0, 0.0, 0.0, 0.0};
};

Influence singleBone(std::uint32_t bone) {
    Influence influence;
    influence.indices = {bone, 0, 0, 0};
    influence.weights = {1.0, 0.0, 0.0, 0.0};
    return influence;
}

Influence blend(std::initializer_list<std::pair<std::uint32_t, double>> values) {
    Influence influence;
    influence.indices.fill(0);
    influence.weights.fill(0.0);
    double sum = 0.0;
    int slot = 0;
    for (const auto& value : values) {
        if (slot >= kMaxBoneInfluences || value.second <= 0.0) {
            continue;
        }
        influence.indices[static_cast<std::size_t>(slot)] = value.first;
        influence.weights[static_cast<std::size_t>(slot)] = value.second;
        sum += value.second;
        ++slot;
    }
    if (sum <= 0.0) {
        influence.weights[0] = 1.0;
    } else {
        for (double& weight : influence.weights) {
            weight /= sum;
        }
    }
    return influence;
}

Influence torsoWeights(double y) {
    constexpr std::array<double, 3> centers = {0.95, 1.60, 2.15};
    constexpr std::array<double, 3> falloffs = {0.82, 0.72, 0.53};
    std::array<double, 3> values{};
    for (int i = 0; i < 3; ++i) {
        values[static_cast<std::size_t>(i)] = std::max(
            0.0, 1.0 - std::abs(y - centers[static_cast<std::size_t>(i)]) /
                             falloffs[static_cast<std::size_t>(i)]);
    }
    if (values[0] + values[1] + values[2] <= 1.0e-9) {
        return y < 1.6 ? singleBone(0) : singleBone(2);
    }
    return blend({{0, values[0]}, {1, values[1]}, {2, values[2]}});
}

class MeshBuilder {
public:
    explicit MeshBuilder(std::string name) : name_(std::move(name)) {}

    void addVertex(const glm::dvec3& position,
                   const glm::dvec3& normal,
                   const Influence& influence) {
        SkinnedVertex vertex;
        vertex.px = static_cast<float>(position.x);
        vertex.py = static_cast<float>(position.y);
        vertex.pz = static_cast<float>(position.z);
        const glm::dvec3 unitNormal =
            glm::length(normal) > 1.0e-12 ? glm::normalize(normal)
                                          : glm::dvec3(0.0, 1.0, 0.0);
        vertex.nx = static_cast<float>(unitNormal.x);
        vertex.ny = static_cast<float>(unitNormal.y);
        vertex.nz = static_cast<float>(unitNormal.z);

        double sum = 0.0;
        for (int i = 0; i < kMaxBoneInfluences; ++i) {
            const double weight = std::max(
                0.0, influence.weights[static_cast<std::size_t>(i)]);
            sum += weight;
        }
        if (sum <= 1.0e-12) {
            for (int i = 0; i < kMaxBoneInfluences; ++i) {
                vertex.boneIndices[i] = 0;
                vertex.boneWeights[i] = (i == 0) ? 1.0f : 0.0f;
            }
        } else {
            for (int i = 0; i < kMaxBoneInfluences; ++i) {
                vertex.boneIndices[i] =
                    influence.indices[static_cast<std::size_t>(i)];
                vertex.boneWeights[i] = static_cast<float>(
                    influence.weights[static_cast<std::size_t>(i)] / sum);
            }
        }
        vertices_.push_back(vertex);
    }

    void addTriangle(std::uint32_t a, std::uint32_t b, std::uint32_t c) {
        indices_.push_back(a);
        indices_.push_back(b);
        indices_.push_back(c);
    }

    void addQuad(std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t d) {
        addTriangle(a, b, c);
        addTriangle(a, c, d);
    }

    void addBox(const glm::dvec3& center,
                const glm::dvec3& halfExtents,
                const Influence& influence) {
        const glm::dvec3 lo = center - halfExtents;
        const glm::dvec3 hi = center + halfExtents;
        struct Face {
            glm::dvec3 normal;
            glm::dvec3 vertices[4];
        };
        const std::array<Face, 6> faces = {{
            {{1, 0, 0}, {{hi.x, lo.y, lo.z}, {hi.x, lo.y, hi.z},
                          {hi.x, hi.y, hi.z}, {hi.x, hi.y, lo.z}}},
            {{-1, 0, 0}, {{lo.x, lo.y, hi.z}, {lo.x, lo.y, lo.z},
                           {lo.x, hi.y, lo.z}, {lo.x, hi.y, hi.z}}},
            {{0, 1, 0}, {{lo.x, hi.y, lo.z}, {hi.x, hi.y, lo.z},
                          {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z}}},
            {{0, -1, 0}, {{lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z},
                           {hi.x, lo.y, lo.z}, {lo.x, lo.y, lo.z}}},
            {{0, 0, 1}, {{lo.x, lo.y, hi.z}, {lo.x, hi.y, hi.z},
                          {hi.x, hi.y, hi.z}, {hi.x, lo.y, hi.z}}},
            {{0, 0, -1}, {{hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z},
                           {lo.x, hi.y, lo.z}, {lo.x, lo.y, lo.z}}},
        }};
        for (const Face& face : faces) {
            const std::uint32_t base = static_cast<std::uint32_t>(vertices_.size());
            for (const glm::dvec3& point : face.vertices) {
                addVertex(point, face.normal, influence);
            }
            addQuad(base, base + 1, base + 2, base + 3);
        }
    }

    void addTube(const std::vector<glm::dvec3>& centers,
                 const std::vector<glm::dvec2>& radii,
                 int segments,
                 bool capBottom,
                 bool capTop) {
        if (centers.size() != radii.size() || centers.size() < 2 || segments < 3) {
            fail(ExitCode::Scene, "E_SCENE_MESH", "invalid tube generation parameters");
        }

        const std::size_t ringCount = centers.size();
        const std::uint32_t base = static_cast<std::uint32_t>(vertices_.size());
        for (std::size_t ring = 0; ring < ringCount; ++ring) {
            const glm::dvec3& center = centers[ring];
            const glm::dvec2& radius = radii[ring];
            const Influence influence = torsoWeights(center.y);
            for (int segment = 0; segment < segments; ++segment) {
                const double angle =
                    (static_cast<double>(segment) / static_cast<double>(segments)) *
                    2.0 * kPi;
                const double c = std::cos(angle);
                const double s = std::sin(angle);
                const glm::dvec3 point(center.x + radius.x * c,
                                       center.y,
                                       center.z + radius.y * s);
                const glm::dvec3 normal(c / std::max(radius.x, 1.0e-6),
                                        0.0,
                                        s / std::max(radius.y, 1.0e-6));
                addVertex(point, normal, influence);
            }
        }

        for (std::size_t ring = 0; ring + 1 < ringCount; ++ring) {
            for (int segment = 0; segment < segments; ++segment) {
                const int next = (segment + 1) % segments;
                const std::uint32_t a = base +
                    static_cast<std::uint32_t>(ring * static_cast<std::size_t>(segments) +
                                               static_cast<std::size_t>(segment));
                const std::uint32_t b = base +
                    static_cast<std::uint32_t>(ring * static_cast<std::size_t>(segments) +
                                               static_cast<std::size_t>(next));
                const std::uint32_t c = b + static_cast<std::uint32_t>(segments);
                const std::uint32_t d = a + static_cast<std::uint32_t>(segments);
                addQuad(a, b, c, d);
            }
        }

        if (capBottom) {
            const std::uint32_t centerIndex =
                static_cast<std::uint32_t>(vertices_.size());
            addVertex(centers.front(), {0, -1, 0}, torsoWeights(centers.front().y));
            for (int segment = 0; segment < segments; ++segment) {
                const int next = (segment + 1) % segments;
                addTriangle(centerIndex,
                            base + static_cast<std::uint32_t>(segment),
                            base + static_cast<std::uint32_t>(next));
            }
        }
        if (capTop) {
            const std::uint32_t centerIndex =
                static_cast<std::uint32_t>(vertices_.size());
            addVertex(centers.back(), {0, 1, 0}, torsoWeights(centers.back().y));
            const std::uint32_t ringBase =
                base + static_cast<std::uint32_t>((ringCount - 1) * segments);
            for (int segment = 0; segment < segments; ++segment) {
                const int next = (segment + 1) % segments;
                addTriangle(centerIndex,
                            ringBase + static_cast<std::uint32_t>(next),
                            ringBase + static_cast<std::uint32_t>(segment));
            }
        }
    }

    void addLimb(const glm::dvec3& start,
                 const glm::dvec3& end,
                 double startRadius,
                 double endRadius,
                 int segments,
                 const Influence& influence) {
        const glm::dvec3 axis = glm::normalize(end - start);
        const glm::dvec3 helper =
            std::abs(glm::dot(axis, glm::dvec3(0, 1, 0))) > 0.92
                ? glm::dvec3(1, 0, 0)
                : glm::dvec3(0, 1, 0);
        const glm::dvec3 u = glm::normalize(glm::cross(axis, helper));
        const glm::dvec3 v = glm::normalize(glm::cross(axis, u));

        const std::uint32_t base = static_cast<std::uint32_t>(vertices_.size());
        for (int ring = 0; ring < 2; ++ring) {
            const glm::dvec3 center = ring == 0 ? start : end;
            const double radius = ring == 0 ? startRadius : endRadius;
            for (int segment = 0; segment < segments; ++segment) {
                const double angle =
                    (static_cast<double>(segment) / static_cast<double>(segments)) *
                    2.0 * kPi;
                const glm::dvec3 radial =
                    u * std::cos(angle) + v * std::sin(angle);
                addVertex(center + radial * radius, radial, influence);
            }
        }

        for (int segment = 0; segment < segments; ++segment) {
            const int next = (segment + 1) % segments;
            const std::uint32_t a = base + static_cast<std::uint32_t>(segment);
            const std::uint32_t b = base + static_cast<std::uint32_t>(next);
            const std::uint32_t c = b + static_cast<std::uint32_t>(segments);
            const std::uint32_t d = a + static_cast<std::uint32_t>(segments);
            addQuad(a, b, c, d);
        }

        const std::uint32_t startCenter =
            static_cast<std::uint32_t>(vertices_.size());
        addVertex(start, -axis, influence);
        const std::uint32_t endCenter =
            static_cast<std::uint32_t>(vertices_.size());
        addVertex(end, axis, influence);
        for (int segment = 0; segment < segments; ++segment) {
            const int next = (segment + 1) % segments;
            addTriangle(startCenter,
                        base + static_cast<std::uint32_t>(next),
                        base + static_cast<std::uint32_t>(segment));
            const std::uint32_t endBase =
                base + static_cast<std::uint32_t>(segments);
            addTriangle(endCenter,
                        endBase + static_cast<std::uint32_t>(segment),
                        endBase + static_cast<std::uint32_t>(next));
        }
    }

    void addEllipsoid(const glm::dvec3& center,
                      const glm::dvec3& radii,
                      int longitudeSegments,
                      int latitudeSegments,
                      const Influence& influence) {
        if (longitudeSegments < 3 || latitudeSegments < 2) {
            fail(ExitCode::Scene, "E_SCENE_MESH", "invalid ellipsoid parameters");
        }

        const std::uint32_t top = static_cast<std::uint32_t>(vertices_.size());
        addVertex(center + glm::dvec3(0, radii.y, 0), {0, 1, 0}, influence);

        const std::uint32_t ringBase = static_cast<std::uint32_t>(vertices_.size());
        for (int lat = 1; lat < latitudeSegments; ++lat) {
            const double theta =
                -kPi * 0.5 + kPi * static_cast<double>(lat) /
                                    static_cast<double>(latitudeSegments);
            const double cy = std::sin(theta);
            const double cr = std::cos(theta);
            for (int lon = 0; lon < longitudeSegments; ++lon) {
                const double phi =
                    2.0 * kPi * static_cast<double>(lon) /
                    static_cast<double>(longitudeSegments);
                const glm::dvec3 unit(cr * std::cos(phi), cy, cr * std::sin(phi));
                const glm::dvec3 point =
                    center + glm::dvec3(unit.x * radii.x,
                                        unit.y * radii.y,
                                        unit.z * radii.z);
                const glm::dvec3 normal(unit.x / (radii.x * radii.x),
                                        unit.y / (radii.y * radii.y),
                                        unit.z / (radii.z * radii.z));
                addVertex(point, normal, influence);
            }
        }

        const std::uint32_t bottom = static_cast<std::uint32_t>(vertices_.size());
        addVertex(center - glm::dvec3(0, radii.y, 0), {0, -1, 0}, influence);

        const int ringCount = latitudeSegments - 1;
        for (int lon = 0; lon < longitudeSegments; ++lon) {
            const int next = (lon + 1) % longitudeSegments;
            addTriangle(top, ringBase + static_cast<std::uint32_t>(next),
                        ringBase + static_cast<std::uint32_t>(lon));
        }
        for (int lat = 0; lat + 1 < ringCount; ++lat) {
            const std::uint32_t lower =
                ringBase + static_cast<std::uint32_t>(lat * longitudeSegments);
            const std::uint32_t upper = lower + static_cast<std::uint32_t>(longitudeSegments);
            for (int lon = 0; lon < longitudeSegments; ++lon) {
                const int next = (lon + 1) % longitudeSegments;
                addQuad(lower + static_cast<std::uint32_t>(lon),
                        lower + static_cast<std::uint32_t>(next),
                        upper + static_cast<std::uint32_t>(next),
                        upper + static_cast<std::uint32_t>(lon));
            }
        }
        const std::uint32_t lastRing =
            ringBase + static_cast<std::uint32_t>((ringCount - 1) * longitudeSegments);
        for (int lon = 0; lon < longitudeSegments; ++lon) {
            const int next = (lon + 1) % longitudeSegments;
            addTriangle(bottom,
                        lastRing + static_cast<std::uint32_t>(lon),
                        lastRing + static_cast<std::uint32_t>(next));
        }
    }

    MeshData finish() {
        MeshData mesh;
        mesh.name = std::move(name_);
        mesh.vertices = std::move(vertices_);
        mesh.indices = std::move(indices_);
        return mesh;
    }

private:
    std::string name_;
    std::vector<SkinnedVertex> vertices_;
    std::vector<std::uint32_t> indices_;
};

void applyHeightScale(MeshData& mesh, double heightScale) {
    for (SkinnedVertex& vertex : mesh.vertices) {
        vertex.py = static_cast<float>(static_cast<double>(vertex.py) * heightScale);
        glm::dvec3 normal(vertex.nx,
                          static_cast<double>(vertex.ny) / heightScale,
                          vertex.nz);
        normal = glm::length(normal) > 1.0e-12
                     ? glm::normalize(normal)
                     : glm::dvec3(0.0, 1.0, 0.0);
        vertex.nx = static_cast<float>(normal.x);
        vertex.ny = static_cast<float>(normal.y);
        vertex.nz = static_cast<float>(normal.z);
    }
    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        if (mesh.indices[i] >= mesh.vertices.size() ||
            mesh.indices[i + 1] >= mesh.vertices.size() ||
            mesh.indices[i + 2] >= mesh.vertices.size()) {
            fail(ExitCode::Scene, "E_SCENE_MESH", "mesh index is outside the vertex buffer");
        }
    }
}

MeshData buildCharacterMesh(const MeshParams& params) {
    if (!std::isfinite(params.bodyWidth) || !std::isfinite(params.heightScale) ||
        !std::isfinite(params.headScale) || !std::isfinite(params.armSpan) ||
        params.bodyWidth < 0.5 || params.bodyWidth > 1.8 ||
        params.heightScale < 0.7 || params.heightScale > 1.4 ||
        params.headScale < 0.65 || params.headScale > 1.5 ||
        params.armSpan < 0.65 || params.armSpan > 1.6) {
        fail(ExitCode::Scene, "E_SCENE_MESH", "mesh parameters are outside safe limits");
    }

    const double bodyWidth = params.bodyWidth;
    MeshBuilder builder("character_bind");

    const Influence root = singleBone(0);
    const Influence arm =
        blend({{0, 0.12}, {1, 0.78}, {2, 0.10}});
    const Influence neck = blend({{1, 0.38}, {2, 0.62}});
    const Influence head = singleBone(2);

    builder.addBox({-0.17 * bodyWidth, 0.49, 0.0},
                   {0.115 * bodyWidth, 0.45, 0.135 * bodyWidth}, root);
    builder.addBox({0.17 * bodyWidth, 0.49, 0.0},
                   {0.115 * bodyWidth, 0.45, 0.135 * bodyWidth}, root);
    builder.addBox({-0.18 * bodyWidth, 0.065, 0.075},
                   {0.125 * bodyWidth, 0.065, 0.22 * bodyWidth}, root);
    builder.addBox({0.18 * bodyWidth, 0.065, 0.075},
                   {0.125 * bodyWidth, 0.065, 0.22 * bodyWidth}, root);
    builder.addBox({0.0, 0.88, 0.0},
                   {0.31 * bodyWidth, 0.15, 0.20 * bodyWidth},
                   blend({{0, 0.78}, {1, 0.22}}));

    std::vector<glm::dvec3> torsoCenters = {
        {0.0, 0.82, 0.0},
        {0.0, 1.01, 0.0},
        {0.0, 1.23, 0.0},
        {0.0, 1.46, 0.0},
        {0.0, 1.68, 0.0},
        {0.0, 1.86, 0.0},
    };
    std::vector<glm::dvec2> torsoRadii = {
        {0.29 * bodyWidth, 0.19 * bodyWidth},
        {0.35 * bodyWidth, 0.22 * bodyWidth},
        {0.37 * bodyWidth, 0.235 * bodyWidth},
        {0.39 * bodyWidth, 0.245 * bodyWidth},
        {0.35 * bodyWidth, 0.225 * bodyWidth},
        {0.27 * bodyWidth, 0.18 * bodyWidth},
    };
    builder.addTube(torsoCenters, torsoRadii, 10, true, true);

    builder.addLimb({-0.31 * bodyWidth, 1.76, 0.0},
                    {-0.56 * params.armSpan, 1.27, 0.08},
                    0.105 * bodyWidth, 0.072 * bodyWidth, 8, arm);
    builder.addLimb({0.31 * bodyWidth, 1.76, 0.0},
                    {0.56 * params.armSpan, 1.27, 0.08},
                    0.105 * bodyWidth, 0.072 * bodyWidth, 8, arm);
    builder.addEllipsoid({-0.56 * params.armSpan, 1.23, 0.08},
                         {0.085, 0.095, 0.080}, 8, 4, arm);
    builder.addEllipsoid({0.56 * params.armSpan, 1.23, 0.08},
                         {0.085, 0.095, 0.080}, 8, 4, arm);

    builder.addEllipsoid({0.0, 1.91, 0.0}, {0.10, 0.09, 0.09}, 8, 4, neck);
    builder.addEllipsoid({0.0, 2.23, 0.0},
                         {0.32 * params.headScale,
                          0.31 * params.headScale,
                          0.29 * params.headScale},
                         10, 6, head);
    builder.addBox({0.0, 2.205, 0.285 * params.headScale},
                   {0.045, 0.050, 0.045}, head);

    MeshData mesh = builder.finish();
    applyHeightScale(mesh, params.heightScale);
    inspectMeshWeights(mesh);
    return mesh;
}

MeshData buildOccluderMesh() {
    MeshBuilder builder("static_occluders");
    const Influence root = singleBone(0);
    builder.addBox({-2.25, 1.15, 0.95}, {0.38, 1.15, 0.38}, root);
    builder.addBox({2.15, 0.70, -0.35}, {0.75, 0.70, 0.75}, root);
    builder.addBox({0.40, 1.50, 2.55}, {1.15, 1.50, 0.18}, root);
    MeshData mesh = builder.finish();
    inspectMeshWeights(mesh);
    return mesh;
}

MeshData buildGroundMesh() {
    MeshData mesh;
    mesh.name = "ground";
    const Influence root = singleBone(0);
    const double halfSize = 7.5;
    const double y = -0.015;
    const glm::dvec3 corners[4] = {
        {-halfSize, y, -halfSize},
        {halfSize, y, -halfSize},
        {halfSize, y, halfSize},
        {-halfSize, y, halfSize},
    };
    for (const glm::dvec3& corner : corners) {
        SkinnedVertex vertex;
        vertex.px = static_cast<float>(corner.x);
        vertex.py = static_cast<float>(corner.y);
        vertex.pz = static_cast<float>(corner.z);
        vertex.nx = 0.0f;
        vertex.ny = 1.0f;
        vertex.nz = 0.0f;
        vertex.boneIndices[0] = root.indices[0];
        for (int i = 0; i < kMaxBoneInfluences; ++i) {
            vertex.boneIndices[static_cast<std::size_t>(i)] =
                root.indices[static_cast<std::size_t>(i)];
            vertex.boneWeights[static_cast<std::size_t>(i)] =
                static_cast<float>(root.weights[static_cast<std::size_t>(i)]);
        }
        mesh.vertices.push_back(vertex);
    }
    mesh.indices = {0, 1, 2, 0, 2, 3};
    inspectMeshWeights(mesh);
    return mesh;
}

} // namespace

DeterministicScene::DeterministicScene(int initialVariant)
    : skeleton_(makeSkeleton()),
      occluders_(buildOccluderMesh()),
      ground_(buildGroundMesh()),
      occluderBounds_(meshBounds(occluders_)) {
    rebuildCharacter(initialVariant);
}

void DeterministicScene::rebuildCharacter(int variant) {
    rebuildCharacter(meshParamsForVariant(variant));
}

void DeterministicScene::rebuildCharacter(const MeshParams& params) {
    MeshData mesh = buildCharacterMesh(params);
    mesh.name = "character_bind_r" + std::to_string(revision_);
    const MeshRevisionInfo weights = inspectMeshWeights(mesh);
    characterBind_ = std::move(mesh);
    revisionInfo_ = weights;
    revisionInfo_.revision = revision_;
    revisionInfo_.variant = params.variant;
    ++revision_;
}

Bounds DeterministicScene::characterBounds(const PoseResult& pose) const {
    Bounds bounds;
    for (const SkinnedVertex& vertex : characterBind_.vertices) {
        bounds.expand(skinPosition(vertex, pose.skinMatrices));
    }
    return bounds;
}

Bounds DeterministicScene::visibleCasterBounds(const PoseResult& pose) const {
    Bounds bounds = characterBounds(pose);
    bounds.expand(occluderBounds_);
    return bounds;
}

MeshData DeterministicScene::deformedCharacter(const PoseResult& pose) const {
    MeshData result = characterBind_;
    result.name = "character_deformed";
    for (SkinnedVertex& vertex : result.vertices) {
        const glm::dvec3 position = skinPosition(vertex, pose.skinMatrices);
        const glm::dvec3 normal = skinNormal(vertex, pose.skinMatrices);
        vertex.px = static_cast<float>(position.x);
        vertex.py = static_cast<float>(position.y);
        vertex.pz = static_cast<float>(position.z);
        vertex.nx = static_cast<float>(normal.x);
        vertex.ny = static_cast<float>(normal.y);
        vertex.nz = static_cast<float>(normal.z);
    }
    return result;
}

Lights DeterministicScene::lights() const {
    Lights result;
    result.directional.directionTowardScene =
        glm::normalize(glm::dvec3(-0.55, -0.72, 0.42));
    result.directional.color = {1.0, 0.93, 0.82};
    result.directional.intensity = 4.40;

    result.point0.position = {-2.0, 2.55, -1.55};
    result.point0.color = {0.25, 0.58, 1.0};
    result.point0.intensity = 2.8;
    result.point0.range = 6.0;

    result.point1.position = {2.25, 1.65, 1.7};
    result.point1.color = {1.0, 0.42, 0.19};
    result.point1.intensity = 2.2;
    result.point1.range = 5.5;

    result.ambientColor = {0.17, 0.20, 0.27};
    result.ambientIntensity = 0.18;
    return result;
}

Material DeterministicScene::characterMaterial() const {
    Material material;
    material.baseColor = {0.78, 0.32, 0.13};
    material.roughness = 0.48;
    material.specular = 0.18;
    material.checkerScale = 0.0;
    return material;
}

Material DeterministicScene::occluderMaterial() const {
    Material material;
    material.baseColor = {0.27, 0.36, 0.43};
    material.roughness = 0.68;
    material.specular = 0.09;
    material.checkerScale = 0.0;
    return material;
}

Material DeterministicScene::groundMaterial() const {
    Material material;
    material.baseColor = {0.20, 0.23, 0.20};
    material.roughness = 0.84;
    material.specular = 0.04;
    material.checkerScale = 0.62;
    return material;
}

} // namespace sr
