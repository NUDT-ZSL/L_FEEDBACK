#include "model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace rigskin {
namespace {

std::vector<std::string> splitWhitespace(const std::string& line) {
    std::istringstream stream(line);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) {
        tokens.push_back(std::move(token));
    }
    return tokens;
}

double parseDoubleToken(const std::string& token, const char* context) {
    std::size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(token, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid number for ") + context);
    }
    if (consumed != token.size() || !finite(value)) {
        throw std::runtime_error(std::string("invalid number for ") + context);
    }
    return value;
}

int parseIntToken(const std::string& token, const char* context) {
    std::size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(token, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid integer for ") + context);
    }
    if (consumed != token.size()) {
        throw std::runtime_error(std::string("invalid integer for ") + context);
    }
    return value;
}

std::uint32_t parseUintToken(const std::string& token, const char* context) {
    int value = parseIntToken(token, context);
    if (value < 0) {
        throw std::runtime_error(std::string("negative integer for ") + context);
    }
    return static_cast<std::uint32_t>(value);
}

Quat animationDelta(int boneIndex, double time, double duration) {
    const double phase = 2.0 * kPi * time / duration;
    const double secondary = 4.0 * kPi * time / duration;

    switch (boneIndex) {
        case 0:
            return axisAngle({0.0, 1.0, 0.0}, radians(2.0 * std::sin(phase)));
        case 1:
            return axisAngle({0.0, 1.0, 0.0}, radians(11.0 * std::sin(phase + 0.2))) *
                   axisAngle({1.0, 0.0, 0.0}, radians(4.0 * std::sin(secondary)));
        case 2:
            return axisAngle({0.0, 1.0, 0.0}, radians(-7.0 * std::sin(phase))) *
                   axisAngle({1.0, 0.0, 0.0}, radians(7.0 * std::sin(phase + 0.45)));
        case 3:
            return axisAngle({1.0, 0.0, 0.0}, radians(4.0 * std::sin(phase + 0.9))) *
                   axisAngle({0.0, 0.0, 1.0}, radians(3.0 * std::sin(secondary)));
        case 4:
            return axisAngle({1.0, 0.0, 0.0}, radians(9.0 * std::sin(phase + 1.1))) *
                   axisAngle({0.0, 0.0, 1.0}, radians(5.0 * std::sin(secondary + 0.5)));
        case 5:
            return axisAngle({1.0, 0.0, 0.0}, radians(29.0 * std::sin(phase + 0.15))) *
                   axisAngle({0.0, 0.0, 1.0}, radians(8.0 * std::sin(secondary)));
        case 6:
            return axisAngle({0.0, 0.0, 1.0}, radians(34.0 * std::sin(phase + 0.8)));
        case 7:
            return axisAngle({1.0, 0.0, 0.0}, radians(29.0 * std::sin(phase + kPi - 0.1))) *
                   axisAngle({0.0, 0.0, 1.0}, radians(-8.0 * std::sin(secondary)));
        case 8:
            return axisAngle({0.0, 0.0, 1.0}, radians(-34.0 * std::sin(phase + 0.8)));
        case 9:
            return axisAngle({1.0, 0.0, 0.0}, radians(21.0 * std::sin(phase)));
        case 10:
            return axisAngle({1.0, 0.0, 0.0}, radians(28.0 * std::sin(phase + 0.55)));
        case 11:
            return axisAngle({1.0, 0.0, 0.0}, radians(21.0 * std::sin(phase + kPi)));
        case 12:
            return axisAngle({1.0, 0.0, 0.0}, radians(28.0 * std::sin(phase + kPi + 0.55)));
        default:
            return {};
    }
}

Vec3 animationTranslationDelta(int boneIndex, double time, double duration) {
    if (boneIndex == 0) {
        const double bob =
            0.045 * (0.5 - 0.5 * std::cos(2.0 * kPi * time / duration));
        return {0.0, bob, 0.0};
    }
    return {};
}

void buildEvalOrderAndBindGlobals(Model& model) {
    const std::size_t boneCount = model.bones.size();
    if (boneCount == 0) {
        throw std::runtime_error("model has no bones");
    }

    int rootIndex = -1;
    const int rootCount = static_cast<int>(std::count_if(
        model.bones.begin(),
        model.bones.end(),
        [](const Bone& bone) { return bone.parent == -1; }));
    if (rootCount != 1) {
        throw std::runtime_error("model must have exactly one root bone");
    }

    for (std::size_t index = 0; index < boneCount; ++index) {
        if (model.bones[index].name.empty()) {
            throw std::runtime_error("bone name must not be empty");
        }
        if (model.bones[index].parent == -1) {
            rootIndex = static_cast<int>(index);
        } else if (model.bones[index].parent < 0 ||
                   model.bones[index].parent >= static_cast<int>(boneCount)) {
            throw std::runtime_error("bone parent index is out of range");
        }
    }

    model.evalOrder.clear();
    model.evalOrder.reserve(boneCount);
    std::vector<std::vector<int>> children(boneCount);
    for (std::size_t index = 0; index < boneCount; ++index) {
        const int parent = model.bones[index].parent;
        if (parent >= 0) {
            children[static_cast<std::size_t>(parent)].push_back(static_cast<int>(index));
        }
    }

    std::vector<std::uint8_t> state(boneCount, 0);
    std::function<void(int)> visit = [&](int boneIndex) {
        const std::size_t index = static_cast<std::size_t>(boneIndex);
        if (state[index] == 1) {
            throw std::runtime_error("bone hierarchy contains a cycle");
        }
        if (state[index] == 2) {
            throw std::runtime_error("bone hierarchy contains a duplicate parent reference");
        }
        state[index] = 1;
        model.evalOrder.push_back(boneIndex);
        for (const int child : children[index]) {
            visit(child);
        }
        state[index] = 2;
    };
    visit(rootIndex);
    if (model.evalOrder.size() != boneCount) {
        throw std::runtime_error("bone hierarchy is disconnected");
    }

    model.bindGlobals.assign(boneCount, Mat4::identity());
    for (const int boneIndex : model.evalOrder) {
        const Bone& bone = model.bones[static_cast<std::size_t>(boneIndex)];
        const Mat4 local =
            Mat4::localTransform(bone.bindTranslation, bone.bindRotation);
        if (bone.parent < 0) {
            model.bindGlobals[static_cast<std::size_t>(boneIndex)] = local;
        } else {
            model.bindGlobals[static_cast<std::size_t>(boneIndex)] =
                model.bindGlobals[static_cast<std::size_t>(bone.parent)] * local;
        }
    }
}

void normalizeVertexWeights(Model& model) {
    model.zeroWeightVertices.clear();
    for (std::size_t vertexIndex = 0; vertexIndex < model.vertices.size(); ++vertexIndex) {
        auto& weights = model.vertices[vertexIndex].weights;
        std::map<std::uint32_t, double> merged;
        for (const SkinWeight& weight : weights) {
            if (weight.bone >= model.bones.size()) {
                throw std::runtime_error("vertex weight references an unknown bone");
            }
            if (!finite(weight.weight) || weight.weight < 0.0) {
                throw std::runtime_error("vertex weight is invalid");
            }
            if (weight.weight > 0.0) {
                merged[weight.bone] += weight.weight;
            }
        }

        weights.clear();
        double sum = 0.0;
        for (const auto& item : merged) {
            weights.push_back({item.first, item.second});
            sum += item.second;
        }

        if (weights.empty() || sum <= 1e-12 || !finite(sum)) {
            weights.clear();
            model.zeroWeightVertices.push_back(static_cast<std::uint32_t>(vertexIndex));
            continue;
        }

        for (SkinWeight& weight : weights) {
            weight.weight /= sum;
        }
    }
}

void computeBindNormals(Model& model) {
    for (MeshVertex& vertex : model.vertices) {
        vertex.bindNormal = {0.0, 0.0, 0.0};
    }

    for (const Triangle& triangle : model.triangles) {
        if (triangle.a >= model.vertices.size() ||
            triangle.b >= model.vertices.size() ||
            triangle.c >= model.vertices.size()) {
            throw std::runtime_error("triangle index is out of range");
        }
        const Vec3& a = model.vertices[triangle.a].bindPosition;
        const Vec3& b = model.vertices[triangle.b].bindPosition;
        const Vec3& c = model.vertices[triangle.c].bindPosition;
        const Vec3 normal = cross(b - a, c - a);
        if (lengthSquared(normal) <= 1e-30) {
            continue;
        }
        model.vertices[triangle.a].bindNormal += normal;
        model.vertices[triangle.b].bindNormal += normal;
        model.vertices[triangle.c].bindNormal += normal;
    }

    for (MeshVertex& vertex : model.vertices) {
        if (lengthSquared(vertex.bindNormal) <= 1e-24) {
            vertex.bindNormal = {0.0, 1.0, 0.0};
        } else {
            vertex.bindNormal = normalized(vertex.bindNormal);
        }
    }
}

class MeshBuilder {
public:
    explicit MeshBuilder(Model& target) : model_(target) {}

    std::uint32_t addVertex(const Vec3& position, std::vector<SkinWeight> weights) {
        MeshVertex vertex;
        vertex.bindPosition = position;
        vertex.weights = std::move(weights);
        model_.vertices.push_back(std::move(vertex));
        return static_cast<std::uint32_t>(model_.vertices.size() - 1);
    }

    void addTriangle(
        std::uint32_t a,
        std::uint32_t b,
        std::uint32_t c,
        const Vec3& interiorPoint) {
        const Vec3& pa = model_.vertices[a].bindPosition;
        const Vec3& pb = model_.vertices[b].bindPosition;
        const Vec3& pc = model_.vertices[c].bindPosition;
        const Vec3 normal = cross(pb - pa, pc - pa);
        if (lengthSquared(normal) > 1e-30) {
            const Vec3 centroid = (pa + pb + pc) / 3.0;
            if (dot(normal, centroid - interiorPoint) < 0.0) {
                std::swap(b, c);
            }
        }
        model_.triangles.push_back({a, b, c});
    }

    void addQuad(
        std::uint32_t a,
        std::uint32_t b,
        std::uint32_t c,
        std::uint32_t d,
        const Vec3& interiorPoint) {
        addTriangle(a, b, c, interiorPoint);
        addTriangle(a, c, d, interiorPoint);
    }

private:
    Model& model_;
};

struct BoxDescription {
    Vec3 center;
    Vec3 halfExtent;
};

void addBox(
    MeshBuilder& builder,
    const BoxDescription& box,
    const std::function<std::vector<SkinWeight>(const Vec3&)>& weightAt) {
    std::array<Vec3, 8> corners;
    for (int index = 0; index < 8; ++index) {
        const double x = (index & 1) ? box.center.x + box.halfExtent.x
                                     : box.center.x - box.halfExtent.x;
        const double y = (index & 2) ? box.center.y + box.halfExtent.y
                                     : box.center.y - box.halfExtent.y;
        const double z = (index & 4) ? box.center.z + box.halfExtent.z
                                     : box.center.z - box.halfExtent.z;
        corners[static_cast<std::size_t>(index)] = {x, y, z};
    }

    std::array<std::uint32_t, 8> ids{};
    for (int index = 0; index < 8; ++index) {
        ids[static_cast<std::size_t>(index)] =
            builder.addVertex(corners[static_cast<std::size_t>(index)],
                              weightAt(corners[static_cast<std::size_t>(index)]));
    }

    const std::array<std::array<int, 4>, 6> faces = {{
        {{0, 1, 3, 2}},
        {{4, 6, 7, 5}},
        {{0, 4, 5, 1}},
        {{2, 3, 7, 6}},
        {{0, 2, 6, 4}},
        {{1, 5, 7, 3}},
    }};
    for (const auto& face : faces) {
        builder.addQuad(
            ids[static_cast<std::size_t>(face[0])],
            ids[static_cast<std::size_t>(face[1])],
            ids[static_cast<std::size_t>(face[2])],
            ids[static_cast<std::size_t>(face[3])],
            box.center);
    }
}

void addTube(
    MeshBuilder& builder,
    const Vec3& start,
    const Vec3& end,
    double startRadius,
    double endRadius,
    std::int64_t parentBone,
    std::int64_t boneIndex,
    std::int64_t childBone) {
    constexpr int sideCount = 8;
    constexpr int ringCount = 4;
    const Vec3 axis = end - start;
    const Vec3 direction = normalized(axis);
    if (lengthSquared(direction) <= 1e-20) {
        return;
    }
    const Vec3 helper =
        std::abs(direction.y) < 0.9 ? Vec3{0.0, 1.0, 0.0} : Vec3{1.0, 0.0, 0.0};
    const Vec3 u = normalized(cross(helper, direction));
    const Vec3 v = normalized(cross(direction, u));

    std::array<std::array<std::uint32_t, sideCount>, ringCount> rings{};
    std::array<Vec3, ringCount> centers{};
    for (int ring = 0; ring < ringCount; ++ring) {
        const double t = static_cast<double>(ring) / (ringCount - 1);
        const Vec3 center = lerp(start, end, t);
        centers[static_cast<std::size_t>(ring)] = center;
        const double radius = startRadius * (1.0 - t) + endRadius * t;

        std::vector<SkinWeight> weights;
        if (ring == 0 && parentBone >= 0) {
            weights.push_back(
                {static_cast<std::uint32_t>(parentBone), 0.30});
            weights.push_back(
                {static_cast<std::uint32_t>(boneIndex), 0.70});
        } else if (ring == ringCount - 1 && childBone >= 0) {
            weights.push_back(
                {static_cast<std::uint32_t>(boneIndex), 0.75});
            weights.push_back(
                {static_cast<std::uint32_t>(childBone), 0.25});
        } else {
            weights.push_back(
                {static_cast<std::uint32_t>(boneIndex), 1.0});
        }

        for (int side = 0; side < sideCount; ++side) {
            const double angle =
                2.0 * kPi * static_cast<double>(side) / sideCount;
            const Vec3 position =
                center + (u * std::cos(angle) + v * std::sin(angle)) * radius;
            rings[static_cast<std::size_t>(ring)][static_cast<std::size_t>(side)] =
                builder.addVertex(position, weights);
        }
    }

    const Vec3 interior = (start + end) * 0.5;
    for (int ring = 0; ring + 1 < ringCount; ++ring) {
        for (int side = 0; side < sideCount; ++side) {
            const int nextSide = (side + 1) % sideCount;
            const std::uint32_t a =
                rings[static_cast<std::size_t>(ring)][static_cast<std::size_t>(side)];
            const std::uint32_t b =
                rings[static_cast<std::size_t>(ring)][static_cast<std::size_t>(nextSide)];
            const std::uint32_t c =
                rings[static_cast<std::size_t>(ring + 1)][static_cast<std::size_t>(nextSide)];
            const std::uint32_t d =
                rings[static_cast<std::size_t>(ring + 1)][static_cast<std::size_t>(side)];
            builder.addQuad(a, b, c, d, interior);
        }
    }

    const Vec3 startCapCenter = start;
    const Vec3 endCapCenter = end;
    const std::vector<SkinWeight> startCapWeights =
        parentBone >= 0
            ? std::vector<SkinWeight>{
                  {static_cast<std::uint32_t>(parentBone), 0.30},
                  {static_cast<std::uint32_t>(boneIndex), 0.70}}
            : std::vector<SkinWeight>{
                  {static_cast<std::uint32_t>(boneIndex), 1.0}};
    const std::vector<SkinWeight> endCapWeights =
        childBone >= 0
            ? std::vector<SkinWeight>{
                  {static_cast<std::uint32_t>(boneIndex), 0.75},
                  {static_cast<std::uint32_t>(childBone), 0.25}}
            : std::vector<SkinWeight>{
                  {static_cast<std::uint32_t>(boneIndex), 1.0}};
    const std::uint32_t startCenterId =
        builder.addVertex(startCapCenter, startCapWeights);
    const std::uint32_t endCenterId =
        builder.addVertex(endCapCenter, endCapWeights);
    for (int side = 0; side < sideCount; ++side) {
        const int nextSide = (side + 1) % sideCount;
        builder.addTriangle(
            startCenterId, rings[0][static_cast<std::size_t>(side)],
            rings[0][static_cast<std::size_t>(nextSide)], interior);
        builder.addTriangle(
            endCenterId,
            rings[static_cast<std::size_t>(ringCount - 1)]
                 [static_cast<std::size_t>(nextSide)],
            rings[static_cast<std::size_t>(ringCount - 1)]
                 [static_cast<std::size_t>(side)],
            interior);
    }

    (void)centers;
}

std::vector<SkinWeight> oneBoneWeight(int boneIndex) {
    return {{static_cast<std::uint32_t>(boneIndex), 1.0}};
}

std::vector<SkinWeight> blendWeights(
    int firstBone,
    double firstWeight,
    int secondBone,
    double secondWeight) {
    return {
        {static_cast<std::uint32_t>(firstBone), firstWeight},
        {static_cast<std::uint32_t>(secondBone), secondWeight},
    };
}

Vec3 globalHead(const Model& model, int boneIndex) {
    return transformPoint(
        model.bindGlobals[static_cast<std::size_t>(boneIndex)], {0.0, 0.0, 0.0});
}

int firstChild(const Model& model, int boneIndex) {
    for (std::size_t index = 0; index < model.bones.size(); ++index) {
        if (model.bones[index].parent == boneIndex) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

Vec3 leafDirectionEnd(const Model& model, int boneIndex, double lengthValue) {
    const Vec3 head = globalHead(model, boneIndex);
    const int parent = model.bones[static_cast<std::size_t>(boneIndex)].parent;
    if (parent < 0) {
        return head + Vec3{0.0, -lengthValue, 0.0};
    }
    const Vec3 outgoing = normalized(head - globalHead(model, parent));
    return head + outgoing * lengthValue;
}

void buildSampleMesh(Model& model) {
    MeshBuilder builder(model);

    addBox(
        builder,
        {{0.0, 1.80, 0.0}, {0.36, 0.52, 0.19}},
        [&model](const Vec3& point) {
            const double t = std::max(
                0.0, std::min(1.0, (point.y - 1.28) / (2.32 - 1.28)));
            if (t < 0.35) {
                return blendWeights(1, 0.35, 2, 0.65);
            }
            if (t > 0.82) {
                return blendWeights(2, 0.85, 3, 0.15);
            }
            return oneBoneWeight(2);
        });

    addBox(
        builder,
        {{0.0, 2.975, 0.0}, {0.22, 0.235, 0.18}},
        [&model](const Vec3& point) {
            (void)model;
            if (point.y < 2.79) {
                return blendWeights(3, 0.25, 4, 0.75);
            }
            return oneBoneWeight(4);
        });

    const int neckChild = firstChild(model, 3);
    addTube(
        builder,
        globalHead(model, 3),
        globalHead(model, 4),
        0.090,
        0.078,
        2,
        3,
        neckChild);

    const int armLeftChild = firstChild(model, 5);
    addTube(
        builder,
        globalHead(model, 5),
        globalHead(model, armLeftChild),
        0.120,
        0.100,
        2,
        5,
        armLeftChild);
    const Vec3 leftHand =
        leafDirectionEnd(model, 6, 0.55);
    addTube(
        builder,
        globalHead(model, 6),
        leftHand,
        0.095,
        0.075,
        5,
        6,
        -1);
    addBox(
        builder,
        {{leftHand.x + 0.03, leftHand.y, leftHand.z}, {0.095, 0.070, 0.060}},
        [](const Vec3&) { return oneBoneWeight(6); });

    const int armRightChild = firstChild(model, 7);
    addTube(
        builder,
        globalHead(model, 7),
        globalHead(model, armRightChild),
        0.120,
        0.100,
        2,
        7,
        armRightChild);
    const Vec3 rightHand =
        leafDirectionEnd(model, 8, 0.55);
    addTube(
        builder,
        globalHead(model, 8),
        rightHand,
        0.095,
        0.075,
        7,
        8,
        -1);
    addBox(
        builder,
        {{rightHand.x - 0.03, rightHand.y, rightHand.z}, {0.095, 0.070, 0.060}},
        [](const Vec3&) { return oneBoneWeight(8); });

    const int leftLegChild = firstChild(model, 9);
    addTube(
        builder,
        globalHead(model, 9),
        globalHead(model, leftLegChild),
        0.155,
        0.125,
        1,
        9,
        leftLegChild);
    const Vec3 leftFoot = leafDirectionEnd(model, 10, 0.50);
    addTube(
        builder,
        globalHead(model, 10),
        leftFoot,
        0.120,
        0.098,
        9,
        10,
        -1);
    addBox(
        builder,
        {{leftFoot.x, leftFoot.y - 0.015, leftFoot.z + 0.10},
         {0.115, 0.060, 0.225}},
        [](const Vec3&) { return oneBoneWeight(10); });

    const int rightLegChild = firstChild(model, 11);
    addTube(
        builder,
        globalHead(model, 11),
        globalHead(model, rightLegChild),
        0.155,
        0.125,
        1,
        11,
        rightLegChild);
    const Vec3 rightFoot = leafDirectionEnd(model, 12, 0.50);
    addTube(
        builder,
        globalHead(model, 12),
        rightFoot,
        0.120,
        0.098,
        11,
        12,
        -1);
    addBox(
        builder,
        {{rightFoot.x, rightFoot.y - 0.015, rightFoot.z + 0.10},
         {0.115, 0.060, 0.225}},
        [](const Vec3&) { return oneBoneWeight(12); });

    builder.addVertex({0.0, 0.95, 0.0}, {});

    if (model.triangles.size() >= 2) {
        builder.addTriangle(0, 1, 1, {0.0, 1.5, 0.0});
    }
}

void addKeysForBone(Model& model, int boneIndex, const std::vector<double>& times) {
    Bone& bone = model.bones[static_cast<std::size_t>(boneIndex)];
    bone.keyframes.clear();
    for (const double time : times) {
        if (time < 0.0 || time > model.duration) {
            throw std::runtime_error("sample key time outside duration");
        }
        Keyframe key;
        key.time = time;
        key.translation =
            bone.bindTranslation + animationTranslationDelta(boneIndex, time, model.duration);
        key.rotation = normalized(
            bone.bindRotation * animationDelta(boneIndex, time, model.duration));
        bone.keyframes.push_back(key);
    }
}

}  // namespace

std::string generateAsciiSample(const std::string& sampleName) {
    if (!sampleName.empty() && sampleName != "preview") {
        throw std::runtime_error(
            "unknown built-in sample '" + sampleName + "'; expected 'preview'");
    }

    Model model;
    model.sampleName = "preview";
    model.duration = 3.4;

    auto addBone = [&model](
                       const std::string& name,
                       int parent,
                       const Vec3& translation,
                       const Quat& rotation) {
        Bone bone;
        bone.name = name;
        bone.parent = parent;
        bone.bindTranslation = translation;
        bone.bindRotation = normalized(rotation);
        model.bones.push_back(std::move(bone));
    };

    addBone("root", -1, {0.0, 0.0, 0.0}, {});
    addBone("pelvis", 0, {0.0, 1.25, 0.0}, {});
    addBone("spine", 1, {0.0, 0.55, 0.0}, {});
    addBone("neck", 2, {0.0, 0.62, 0.0}, {});
    addBone("head", 3, {0.0, 0.35, 0.0}, {});
    addBone(
        "upper_arm_l", 2, {0.36, 0.44, 0.0},
        axisAngle({0.0, 0.0, 1.0}, radians(-70.0)));
    addBone("forearm_l", 5, {0.0, 0.58, 0.0}, {});
    addBone(
        "upper_arm_r", 2, {-0.36, 0.44, 0.0},
        axisAngle({0.0, 0.0, 1.0}, radians(70.0)));
    addBone("forearm_r", 7, {0.0, 0.58, 0.0}, {});
    addBone("thigh_l", 1, {0.22, -0.10, 0.0}, {});
    addBone("shin_l", 9, {0.0, -0.60, 0.0}, {});
    addBone("thigh_r", 1, {-0.22, -0.10, 0.0}, {});
    addBone("shin_r", 11, {0.0, -0.60, 0.0}, {});

    buildEvalOrderAndBindGlobals(model);

    addKeysForBone(model, 0, {0.0, 0.45, 1.12, 1.93, 2.71, 3.4});
    addKeysForBone(model, 1, {0.0, 0.38, 0.86, 1.55, 2.18, 2.84, 3.4});
    addKeysForBone(model, 2, {0.0, 0.52, 1.25, 2.0, 2.65, 3.4});
    addKeysForBone(model, 3, {0.0, 0.63, 1.34, 2.22, 3.0, 3.4});
    addKeysForBone(model, 4, {0.0, 0.58, 1.19, 1.91, 2.74, 3.4});
    addKeysForBone(model, 5, {0.0, 0.31, 0.79, 1.42, 2.06, 2.57, 3.05, 3.4});
    addKeysForBone(model, 6, {0.0, 0.27, 0.74, 1.36, 2.13, 2.81, 3.4});
    addKeysForBone(model, 7, {0.0, 0.41, 0.97, 1.63, 2.31, 2.92, 3.4});
    addKeysForBone(model, 8, {0.0, 0.36, 0.89, 1.48, 2.17, 2.88, 3.4});
    addKeysForBone(model, 9, {0.0, 0.34, 0.93, 1.68, 2.2, 2.95, 3.4});
    addKeysForBone(model, 10, {0.0, 0.46, 1.04, 1.71, 2.36, 3.02, 3.4});
    addKeysForBone(model, 11, {0.0, 0.52, 1.06, 1.78, 2.42, 3.1, 3.4});
    addKeysForBone(model, 12, {0.0, 0.39, 0.98, 1.73, 2.28, 2.99, 3.4});

    buildSampleMesh(model);
    finalizeModel(model);

    std::ostringstream output;
    output << std::fixed << std::setprecision(12);
    output << "ASCII_MODEL 1\n";
    output << "sample " << model.sampleName << '\n';
    output << "duration " << model.duration << '\n';
    output << "bone_count " << model.bones.size() << '\n';
    for (std::size_t boneIndex = 0; boneIndex < model.bones.size(); ++boneIndex) {
        const Bone& bone = model.bones[boneIndex];
        output << "bone " << boneIndex << ' ' << bone.name << ' ' << bone.parent;
        output << ' ' << bone.bindTranslation.x << ' ' << bone.bindTranslation.y
               << ' ' << bone.bindTranslation.z;
        output << ' ' << bone.bindRotation.w << ' ' << bone.bindRotation.x
               << ' ' << bone.bindRotation.y << ' ' << bone.bindRotation.z << '\n';
    }
    for (std::size_t boneIndex = 0; boneIndex < model.bones.size(); ++boneIndex) {
        const Bone& bone = model.bones[boneIndex];
        output << "keys " << boneIndex << ' ' << bone.keyframes.size() << '\n';
        for (const Keyframe& key : bone.keyframes) {
            output << "key " << key.time;
            output << ' ' << key.translation.x << ' ' << key.translation.y
                   << ' ' << key.translation.z;
            output << ' ' << key.rotation.w << ' ' << key.rotation.x
                   << ' ' << key.rotation.y << ' ' << key.rotation.z << '\n';
        }
    }

    output << "vertex_count " << model.vertices.size() << '\n';
    for (std::size_t vertexIndex = 0; vertexIndex < model.vertices.size();
         ++vertexIndex) {
        const MeshVertex& vertex = model.vertices[vertexIndex];
        output << "vertex " << vertexIndex;
        output << ' ' << vertex.bindPosition.x << ' ' << vertex.bindPosition.y
               << ' ' << vertex.bindPosition.z;
        output << ' ' << vertex.weights.size();
        for (const SkinWeight& weight : vertex.weights) {
            output << ' ' << weight.bone << ' ' << weight.weight;
        }
        output << '\n';
    }

    output << "triangle_count " << model.triangles.size() << '\n';
    for (std::size_t triangleIndex = 0; triangleIndex < model.triangles.size();
         ++triangleIndex) {
        const Triangle& triangle = model.triangles[triangleIndex];
        output << "triangle " << triangleIndex << ' ' << triangle.a << ' '
               << triangle.b << ' ' << triangle.c << '\n';
    }
    output << "end\n";
    return output.str();
}

Model parseAsciiModel(const std::string& text) {
    std::istringstream input(text);
    std::vector<std::pair<std::size_t, std::string>> lines;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto tokens = splitWhitespace(line);
        if (tokens.empty()) {
            continue;
        }
        lines.emplace_back(lineNumber, line);
    }

    if (lines.empty()) {
        throw std::runtime_error("empty model description");
    }

    std::size_t cursor = 0;
    auto nextTokens = [&](const char* context) {
        if (cursor >= lines.size()) {
            throw std::runtime_error(std::string("unexpected end of file reading ") + context);
        }
        auto tokens = splitWhitespace(lines[cursor].second);
        ++cursor;
        return tokens;
    };
    auto fail = [&](const char* message) {
        throw std::runtime_error(
            std::string("model parse error at line ") +
            std::to_string(lines[cursor > 0 ? cursor - 1 : 0].first) + ": " + message);
    };

    {
        const auto tokens = nextTokens("header");
        if (tokens.size() != 2 || tokens[0] != "ASCII_MODEL" ||
            parseIntToken(tokens[1], "format version") != 1) {
            fail("expected ASCII_MODEL 1");
        }
    }

    Model model;
    {
        const auto tokens = nextTokens("sample");
        if (tokens.size() != 2 || tokens[0] != "sample") {
            fail("expected sample name");
        }
        model.sampleName = tokens[1];
    }
    {
        const auto tokens = nextTokens("duration");
        if (tokens.size() != 2 || tokens[0] != "duration") {
            fail("expected duration");
        }
        model.duration = parseDoubleToken(tokens[1], "duration");
        if (model.duration <= 0.0) {
            fail("duration must be positive");
        }
    }
    {
        const auto tokens = nextTokens("bone_count");
        if (tokens.size() != 2 || tokens[0] != "bone_count") {
            fail("expected bone_count");
        }
        const int boneCount = parseIntToken(tokens[1], "bone count");
        if (boneCount <= 0) {
            fail("bone count must be positive");
        }
        model.bones.resize(static_cast<std::size_t>(boneCount));
    }

    for (std::size_t boneIndex = 0; boneIndex < model.bones.size(); ++boneIndex) {
        const auto tokens = nextTokens("bone");
        if (tokens.size() != 11 || tokens[0] != "bone") {
            fail("expected bone record with 11 tokens");
        }
        const int recordIndex = parseIntToken(tokens[1], "bone index");
        if (recordIndex != static_cast<int>(boneIndex)) {
            fail("bone index is out of sequence");
        }
        Bone& bone = model.bones[boneIndex];
        bone.name = tokens[2];
        bone.parent = parseIntToken(tokens[3], "bone parent");
        bone.bindTranslation = {
            parseDoubleToken(tokens[4], "bind translation x"),
            parseDoubleToken(tokens[5], "bind translation y"),
            parseDoubleToken(tokens[6], "bind translation z"),
        };
        bone.bindRotation = normalized({
            parseDoubleToken(tokens[7], "bind rotation w"),
            parseDoubleToken(tokens[8], "bind rotation x"),
            parseDoubleToken(tokens[9], "bind rotation y"),
            parseDoubleToken(tokens[10], "bind rotation z"),
        });
    }

    for (std::size_t boneIndex = 0; boneIndex < model.bones.size(); ++boneIndex) {
        const auto header = nextTokens("keys");
        if (header.size() != 3 || header[0] != "keys") {
            fail("expected keys header");
        }
        const int recordIndex = parseIntToken(header[1], "keys bone index");
        if (recordIndex != static_cast<int>(boneIndex)) {
            fail("keys bone index is out of sequence");
        }
        const int keyCount = parseIntToken(header[2], "key count");
        if (keyCount < 0) {
            fail("negative key count");
        }
        for (int keyIndex = 0; keyIndex < keyCount; ++keyIndex) {
            const auto tokens = nextTokens("key");
            if (tokens.size() != 9 || tokens[0] != "key") {
                fail("expected key record with 9 tokens");
            }
            Keyframe key;
            key.time = parseDoubleToken(tokens[1], "key time");
            key.translation = {
                parseDoubleToken(tokens[2], "key translation x"),
                parseDoubleToken(tokens[3], "key translation y"),
                parseDoubleToken(tokens[4], "key translation z"),
            };
            key.rotation = normalized({
                parseDoubleToken(tokens[5], "key rotation w"),
                parseDoubleToken(tokens[6], "key rotation x"),
                parseDoubleToken(tokens[7], "key rotation y"),
                parseDoubleToken(tokens[8], "key rotation z"),
            });
            if (key.time < 0.0) {
                fail("key time must not be negative");
            }
            if (!model.bones[boneIndex].keyframes.empty() &&
                key.time <= model.bones[boneIndex].keyframes.back().time) {
                fail("key times must be strictly increasing");
            }
            model.bones[boneIndex].keyframes.push_back(key);
        }
    }

    {
        const auto tokens = nextTokens("vertex_count");
        if (tokens.size() != 2 || tokens[0] != "vertex_count") {
            fail("expected vertex_count");
        }
        const int vertexCount = parseIntToken(tokens[1], "vertex count");
        if (vertexCount <= 0) {
            fail("vertex count must be positive");
        }
        model.vertices.resize(static_cast<std::size_t>(vertexCount));
    }

    for (std::size_t vertexIndex = 0; vertexIndex < model.vertices.size(); ++vertexIndex) {
        const auto tokens = nextTokens("vertex");
        if (tokens.size() < 6 || tokens[0] != "vertex") {
            fail("expected vertex record");
        }
        const int recordIndex = parseIntToken(tokens[1], "vertex index");
        if (recordIndex != static_cast<int>(vertexIndex)) {
            fail("vertex index is out of sequence");
        }
        MeshVertex& vertex = model.vertices[vertexIndex];
        vertex.bindPosition = {
            parseDoubleToken(tokens[2], "vertex x"),
            parseDoubleToken(tokens[3], "vertex y"),
            parseDoubleToken(tokens[4], "vertex z"),
        };
        const int weightCount = parseIntToken(tokens[5], "weight count");
        if (weightCount < 0) {
            fail("negative weight count");
        }
        if (tokens.size() != static_cast<std::size_t>(6 + weightCount * 2)) {
            fail("weight token count does not match weight count");
        }
        for (int weightIndex = 0; weightIndex < weightCount; ++weightIndex) {
            const std::size_t offset = 6 + static_cast<std::size_t>(weightIndex * 2);
            SkinWeight weight;
            weight.bone = parseUintToken(tokens[offset], "weight bone index");
            weight.weight = parseDoubleToken(tokens[offset + 1], "weight value");
            vertex.weights.push_back(weight);
        }
    }

    {
        const auto tokens = nextTokens("triangle_count");
        if (tokens.size() != 2 || tokens[0] != "triangle_count") {
            fail("expected triangle_count");
        }
        const int triangleCount = parseIntToken(tokens[1], "triangle count");
        if (triangleCount <= 0) {
            fail("triangle count must be positive");
        }
        model.triangles.resize(static_cast<std::size_t>(triangleCount));
    }

    for (std::size_t triangleIndex = 0; triangleIndex < model.triangles.size();
         ++triangleIndex) {
        const auto tokens = nextTokens("triangle");
        if (tokens.size() != 5 || tokens[0] != "triangle") {
            fail("expected triangle record");
        }
        const int recordIndex = parseIntToken(tokens[1], "triangle index");
        if (recordIndex != static_cast<int>(triangleIndex)) {
            fail("triangle index is out of sequence");
        }
        model.triangles[triangleIndex] = {
            parseUintToken(tokens[2], "triangle a"),
            parseUintToken(tokens[3], "triangle b"),
            parseUintToken(tokens[4], "triangle c"),
        };
    }

    {
        const auto tokens = nextTokens("end");
        if (tokens.size() != 1 || tokens[0] != "end") {
            fail("expected end marker");
        }
    }
    if (cursor != lines.size()) {
        throw std::runtime_error("trailing data after model end marker");
    }

    finalizeModel(model);
    return model;
}

void finalizeModel(Model& model) {
    buildEvalOrderAndBindGlobals(model);
    normalizeVertexWeights(model);
    computeBindNormals(model);

    model.bindBounds = {};
    for (const MeshVertex& vertex : model.vertices) {
        model.bindBounds.include(vertex.bindPosition);
    }
    if (!model.bindBounds.valid) {
        throw std::runtime_error("model has no valid bounds");
    }
}

std::vector<Mat4> evaluatePoseGlobals(
    const Model& model,
    double time,
    std::vector<Mat4>* localTransforms) {
    const std::size_t boneCount = model.bones.size();
    std::vector<Mat4> locals(boneCount, Mat4::identity());

    for (std::size_t boneIndex = 0; boneIndex < boneCount; ++boneIndex) {
        const Bone& bone = model.bones[boneIndex];
        if (bone.keyframes.empty()) {
            locals[boneIndex] =
                Mat4::localTransform(bone.bindTranslation, bone.bindRotation);
            continue;
        }

        const auto& keys = bone.keyframes;
        if (time <= keys.front().time) {
            locals[boneIndex] =
                Mat4::localTransform(keys.front().translation, keys.front().rotation);
            continue;
        }
        if (time >= keys.back().time) {
            locals[boneIndex] =
                Mat4::localTransform(keys.back().translation, keys.back().rotation);
            continue;
        }

        const auto upper = std::upper_bound(
            keys.begin(),
            keys.end(),
            time,
            [](double value, const Keyframe& key) { return value < key.time; });
        const std::size_t upperIndex =
            static_cast<std::size_t>(upper - keys.begin());
        const Keyframe& right = keys[upperIndex];
        const Keyframe& left = keys[upperIndex - 1];
        const double span = right.time - left.time;
        if (span <= 0.0) {
            throw std::runtime_error("keyframe times are not increasing");
        }
        const double t = (time - left.time) / span;
        locals[boneIndex] = Mat4::localTransform(
            lerp(left.translation, right.translation, t),
            slerp(left.rotation, right.rotation, t));
    }

    std::vector<Mat4> globals(boneCount, Mat4::identity());
    for (const int boneIndex : model.evalOrder) {
        const std::size_t index = static_cast<std::size_t>(boneIndex);
        const int parent = model.bones[index].parent;
        if (parent < 0) {
            globals[index] = locals[index];
        } else {
            globals[index] =
                globals[static_cast<std::size_t>(parent)] * locals[index];
        }
    }

    if (localTransforms != nullptr) {
        *localTransforms = std::move(locals);
    }
    return globals;
}

double frameTime(double duration, int frameIndex, int frameCount) {
    if (frameCount <= 1) {
        return 0.0;
    }
    return duration * static_cast<double>(frameIndex) /
           static_cast<double>(frameCount - 1);
}

std::string pipelineDescriptor(const PipelineConfig& config) {
    std::ostringstream output;
    output << "FMSKEL_PIPELINE 1\n";
    output << "frame_count " << config.frameCount << '\n';
    output << "width " << config.width << '\n';
    output << "height " << config.height << '\n';
    output << "camera orthographic_fixed\n";
    output << "light directional_fixed\n";
    return output.str();
}

}  // namespace rigskin
