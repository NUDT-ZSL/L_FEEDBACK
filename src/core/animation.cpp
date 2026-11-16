#include "core/animation.hpp"

#include "core/error.hpp"
#include "core/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace uid166 {

namespace {

constexpr char kMagic[8] = {'U', 'I', 'D', '1', '6', '6', 'A', 'N'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kHeaderSize = 64;
constexpr size_t kMaxPayload = 128u * 1024u * 1024u;
constexpr uint32_t kMaxBones = 128;
constexpr uint32_t kMaxVertices = 1'000'000;
constexpr uint32_t kMaxIndices = 3'000'000;
constexpr uint32_t kMaxKeys = 1'000'000;

static_assert(sizeof(SkinnedVertex) == 52, "binary vertex layout must remain stable");

class LittleWriter {
public:
    void u8(uint8_t v) { data_.push_back(v); }
    void u16(uint16_t v) {
        data_.push_back(static_cast<uint8_t>(v));
        data_.push_back(static_cast<uint8_t>(v >> 8));
    }
    void u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) data_.push_back(static_cast<uint8_t>(v >> (i * 8)));
    }
    void i32(int32_t v) { u32(static_cast<uint32_t>(v)); }
    void u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) data_.push_back(static_cast<uint8_t>(v >> (i * 8)));
    }
    void f32(float v) {
        const double scaled = std::round(static_cast<double>(v) * 1000000.0);
        const float canonical = std::abs(scaled) < 0.5
            ? 0.0f
            : static_cast<float>(scaled / 1000000.0);
        uint32_t bits = 0;
        std::memcpy(&bits, &canonical, sizeof(bits));
        u32(bits);
    }
    void bytes(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        data_.insert(data_.end(), b, b + n);
    }
    void string(const std::string& s) {
        if (s.size() > 255) throw AppError(ErrorCode::internal, "bone name too long");
        u16(static_cast<uint16_t>(s.size()));
        bytes(s.data(), s.size());
    }

    [[nodiscard]] const std::vector<uint8_t>& data() const { return data_; }

private:
    std::vector<uint8_t> data_;
};

class LittleReader {
public:
    explicit LittleReader(std::string_view bytes) : bytes_(bytes) {}

    uint8_t u8() {
        need(1);
        return static_cast<uint8_t>(bytes_[pos_++]);
    }
    uint16_t u16() {
        need(2);
        const uint16_t v = static_cast<uint16_t>(static_cast<uint8_t>(bytes_[pos_])) |
                           static_cast<uint16_t>(static_cast<uint8_t>(bytes_[pos_ + 1]) << 8);
        pos_ += 2;
        return v;
    }
    uint32_t u32() {
        need(4);
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            v |= static_cast<uint32_t>(static_cast<uint8_t>(bytes_[pos_ + i])) << (i * 8);
        }
        pos_ += 4;
        return v;
    }
    int32_t i32() { return static_cast<int32_t>(u32()); }
    uint64_t u64() {
        need(8);
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= static_cast<uint64_t>(static_cast<uint8_t>(bytes_[pos_ + i])) << (i * 8);
        }
        pos_ += 8;
        return v;
    }
    float f32() {
        const uint32_t bits = u32();
        float v = 0.0f;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
    std::string string() {
        const uint16_t n = u16();
        need(n);
        std::string s(bytes_.substr(pos_, n));
        pos_ += n;
        return s;
    }
    std::string_view bytes(size_t n) {
        need(n);
        const std::string_view s = bytes_.substr(pos_, n);
        pos_ += n;
        return s;
    }
    void skip(size_t n) {
        need(n);
        pos_ += n;
    }
    [[nodiscard]] size_t remaining() const { return bytes_.size() - pos_; }

private:
    void need(size_t n) const {
        if (n > bytes_.size() - pos_) {
            throw AppError(ErrorCode::animationCorrupt, "truncated animation payload");
        }
    }
    std::string_view bytes_;
    size_t pos_ = 0;
};

uint64_t hashPayload(std::string_view bytes) {
    return hashBytes(kFnvOffset, bytes.data(), bytes.size());
}

void addVertex(std::vector<SkinnedVertex>& vertices, const Vec3& p, const Vec3& n,
               const Vec2& uv, const std::array<uint8_t, 4>& bones,
               const std::array<float, 4>& weights) {
    SkinnedVertex v;
    v.px = p.x; v.py = p.y; v.pz = p.z;
    v.nx = n.x; v.ny = n.y; v.nz = n.z;
    v.u = uv.x; v.v = uv.y;
    for (int i = 0; i < 4; ++i) {
        v.bone[i] = bones[static_cast<size_t>(i)];
        v.weight[i] = weights[static_cast<size_t>(i)];
    }
    vertices.push_back(v);
}

void appendBox(std::vector<SkinnedVertex>& vertices, std::vector<uint32_t>& indices,
               const Vec3& minCorner, const Vec3& maxCorner, uint8_t bone) {
    const Vec3 c = (minCorner + maxCorner) * 0.5f;
    const Vec3 e = (maxCorner - minCorner) * 0.5f;
    const std::array<uint8_t, 4> bones{bone, 0, 0, 0};
    const std::array<float, 4> weights{1.0f, 0.0f, 0.0f, 0.0f};
    struct Face {
        Vec3 normal;
        Vec3 corners[4];
    };
    const Face faces[6] = {
        {Vec3{-1, 0, 0}, {Vec3{-e.x, -e.y, -e.z}, Vec3{-e.x, -e.y, e.z},
                          Vec3{-e.x, e.y, e.z}, Vec3{-e.x, e.y, -e.z}}},
        {Vec3{1, 0, 0}, {Vec3{e.x, -e.y, e.z}, Vec3{e.x, -e.y, -e.z},
                         Vec3{e.x, e.y, -e.z}, Vec3{e.x, e.y, e.z}}},
        {Vec3{0, -1, 0}, {Vec3{-e.x, -e.y, e.z}, Vec3{e.x, -e.y, e.z},
                          Vec3{e.x, -e.y, -e.z}, Vec3{-e.x, -e.y, -e.z}}},
        {Vec3{0, 1, 0}, {Vec3{-e.x, e.y, -e.z}, Vec3{e.x, e.y, -e.z},
                         Vec3{e.x, e.y, e.z}, Vec3{-e.x, e.y, e.z}}},
        {Vec3{0, 0, -1}, {Vec3{e.x, -e.y, -e.z}, Vec3{-e.x, -e.y, -e.z},
                          Vec3{-e.x, e.y, -e.z}, Vec3{e.x, e.y, -e.z}}},
        {Vec3{0, 0, 1}, {Vec3{-e.x, -e.y, e.z}, Vec3{e.x, -e.y, e.z},
                         Vec3{e.x, e.y, e.z}, Vec3{-e.x, e.y, e.z}}},
    };
    for (const Face& face : faces) {
        const uint32_t base = static_cast<uint32_t>(vertices.size());
        for (int corner = 0; corner < 4; ++corner) {
            addVertex(vertices, c + face.corners[corner], face.normal,
                      {static_cast<float>(corner & 1),
                       static_cast<float>((corner >> 1) & 1)},
                      bones, weights);
        }
        indices.insert(indices.end(), {
            base + 0, base + 1, base + 2, base + 0, base + 2, base + 3,
        });
    }
}

void appendCapsuleBody(std::vector<SkinnedVertex>& vertices, std::vector<uint32_t>& indices,
                       const std::vector<Bone>& bones, const std::vector<uint32_t>& spineBones) {
    constexpr int radial = 14;
    constexpr int rings = 17;
    const float y0 = 0.78f;
    const float y1 = 2.12f;
    const uint32_t base = static_cast<uint32_t>(vertices.size());
    for (int ring = 0; ring < rings; ++ring) {
        const float t = static_cast<float>(ring) / static_cast<float>(rings - 1);
        const float y = y0 + (y1 - y0) * t;
        const float radius = 0.32f + 0.16f * std::sin(t * kPi);

        uint32_t b0 = spineBones.front();
        uint32_t b1 = spineBones.front();
        float best = std::numeric_limits<float>::max();
        float second = std::numeric_limits<float>::max();
        for (uint32_t boneIndex : spineBones) {
            const float d = std::abs(y - bones[boneIndex].bindWorld.y);
            if (d < best) {
                second = best;
                b1 = b0;
                best = d;
                b0 = boneIndex;
            } else if (d < second) {
                second = d;
                b1 = boneIndex;
            }
        }
        float w1 = second > 1.0e-5f ? std::clamp(best / (best + second), 0.0f, 1.0f) : 0.0f;
        w1 = std::min(w1, 0.9f);
        const float w0 = 1.0f - w1;
        if (b1 == b0) w1 = 0.0f;
        const std::array<uint8_t, 4> bones4{
            static_cast<uint8_t>(b0), static_cast<uint8_t>(b1), 0, 0};
        const std::array<float, 4> weights4{w0, w1, 0.0f, 0.0f};
        for (int r = 0; r < radial; ++r) {
            const float a = 2.0f * kPi * static_cast<float>(r) / static_cast<float>(radial);
            const Vec3 p{std::cos(a) * radius, y, std::sin(a) * radius};
            const Vec3 n = normalize({std::cos(a), 0.12f, std::sin(a)});
            addVertex(vertices, p, n,
                      {static_cast<float>(r) / static_cast<float>(radial), t},
                      bones4, weights4);
        }
    }
    for (int ring = 0; ring < rings - 1; ++ring) {
        for (int r = 0; r < radial; ++r) {
            const uint32_t i0 = base + static_cast<uint32_t>(ring * radial + r);
            const uint32_t i1 = base + static_cast<uint32_t>(ring * radial + (r + 1) % radial);
            const uint32_t i2 = base + static_cast<uint32_t>((ring + 1) * radial + r);
            const uint32_t i3 = base + static_cast<uint32_t>((ring + 1) * radial + (r + 1) % radial);
            indices.insert(indices.end(), {i0, i2, i1, i1, i2, i3});
        }
    }
}

void appendHead(std::vector<SkinnedVertex>& vertices, std::vector<uint32_t>& indices,
                uint8_t bone) {
    constexpr int slices = 14;
    constexpr int stacks = 8;
    const Vec3 center{0.0f, 2.35f, 0.0f};
    const uint32_t base = static_cast<uint32_t>(vertices.size());
    const std::array<uint8_t, 4> bones{bone, 0, 0, 0};
    const std::array<float, 4> weights{1, 0, 0, 0};
    for (int stack = 0; stack <= stacks; ++stack) {
        const float v = static_cast<float>(stack) / static_cast<float>(stacks);
        const float phi = v * kPi;
        for (int slice = 0; slice <= slices; ++slice) {
            const float u = static_cast<float>(slice) / static_cast<float>(slices);
            const float theta = u * 2.0f * kPi;
            const Vec3 n{std::sin(phi) * std::cos(theta), std::cos(phi),
                         std::sin(phi) * std::sin(theta)};
            const Vec3 p{n.x * 0.28f, n.y * 0.34f, n.z * 0.28f};
            addVertex(vertices, center + p, n, {u, v}, bones, weights);
        }
    }
    for (int stack = 0; stack < stacks; ++stack) {
        for (int slice = 0; slice < slices; ++slice) {
            const uint32_t i0 = base + static_cast<uint32_t>(stack * (slices + 1) + slice);
            const uint32_t i1 = i0 + 1;
            const uint32_t i2 = i0 + static_cast<uint32_t>(slices + 1);
            const uint32_t i3 = i2 + 1;
            indices.insert(indices.end(), {i0, i2, i1, i1, i2, i3});
        }
    }
}

} // namespace

CharacterAsset generateCharacterAsset() {
    CharacterAsset asset;
    asset.bones = {
        {"root", -1, Mat4::identity(), {0.0f, 0.0f, 0.0f}},
        {"hips", 0, Mat4::identity(), {0.0f, 0.88f, 0.0f}},
        {"spine", 1, Mat4::identity(), {0.0f, 1.20f, 0.0f}},
        {"chest", 2, Mat4::identity(), {0.0f, 1.56f, 0.0f}},
        {"neck", 3, Mat4::identity(), {0.0f, 1.94f, 0.0f}},
        {"head", 4, Mat4::identity(), {0.0f, 2.16f, 0.0f}},
        {"arm_l", 3, Mat4::identity(), {0.38f, 1.70f, 0.0f}},
        {"arm_r", 3, Mat4::identity(), {-0.38f, 1.70f, 0.0f}},
        {"leg_l", 1, Mat4::identity(), {0.16f, 0.84f, 0.0f}},
        {"leg_r", 1, Mat4::identity(), {-0.16f, 0.84f, 0.0f}},
    };
    for (Bone& bone : asset.bones) {
        bone.inverseBind = Mat4::translation(bone.bindWorld * -1.0f);
    }

    const std::vector<uint32_t> spineBones{1, 2, 3, 4, 5};
    appendCapsuleBody(asset.vertices, asset.indices, asset.bones, spineBones);
    appendHead(asset.vertices, asset.indices, 5);
    appendBox(asset.vertices, asset.indices, {0.28f, 0.88f, -0.10f},
              {0.46f, 1.72f, 0.10f}, 6);
    appendBox(asset.vertices, asset.indices, {-0.46f, 0.88f, -0.10f},
              {-0.28f, 1.72f, 0.10f}, 7);
    appendBox(asset.vertices, asset.indices, {0.06f, 0.05f, -0.12f},
              {0.26f, 0.88f, 0.12f}, 8);
    appendBox(asset.vertices, asset.indices, {-0.26f, 0.05f, -0.12f},
              {-0.06f, 0.88f, 0.12f}, 9);

    asset.bindBounds = {};
    for (const SkinnedVertex& v : asset.vertices) {
        asset.bindBounds.expand(Vec3{v.px, v.py, v.pz});
    }

    constexpr int keyCount = 49;
    asset.duration = 2.0f;
    asset.tracks.reserve(asset.bones.size());
    for (uint32_t boneIndex = 0; boneIndex < asset.bones.size(); ++boneIndex) {
        BoneTrack track;
        track.boneIndex = boneIndex;
        track.keys.reserve(keyCount);
        const Vec3 localBind = (boneIndex == 0 ? Vec3{} :
            asset.bones[boneIndex].bindWorld -
            asset.bones[static_cast<size_t>(asset.bones[boneIndex].parent)].bindWorld);
        for (int k = 0; k < keyCount; ++k) {
            const float time = asset.duration * static_cast<float>(k) /
                               static_cast<float>(keyCount - 1);
            const float phase = 2.0f * kPi * time / asset.duration;
            Vec3 translation = localBind;
            Quat rotation{};
            switch (boneIndex) {
            case 0:
                translation.y += 0.035f * std::sin(phase * 2.0f);
                rotation = normalizeQuat({0.0f, std::sin(phase * 0.5f) * 0.08f, 0.0f,
                                          std::cos(phase * 0.5f)});
                break;
            case 1:
                rotation = normalizeQuat({0.0f, std::sin(phase) * 0.10f,
                                          std::sin(phase) * 0.06f, std::cos(phase)});
                break;
            case 2:
                rotation = normalizeQuat({std::sin(phase) * 0.055f, 0.0f, 0.0f,
                                          std::cos(phase)});
                break;
            case 3:
                rotation = normalizeQuat({-std::sin(phase) * 0.045f,
                                          std::sin(phase) * 0.05f, 0.0f, std::cos(phase)});
                break;
            case 4:
            case 5:
                rotation = normalizeQuat({std::sin(phase) * 0.07f, 0.0f, 0.0f,
                                          std::cos(phase)});
                break;
            case 6:
            case 7: {
                const float side = boneIndex == 6 ? 1.0f : -1.0f;
                rotation = normalizeQuat({side * std::sin(phase) * 0.60f, 0.0f,
                                          side * 0.08f, std::cos(phase)});
                break;
            }
            case 8:
            case 9: {
                const float side = boneIndex == 8 ? 1.0f : -1.0f;
                rotation = normalizeQuat({-side * std::sin(phase) * 0.45f, 0.0f, 0.0f,
                                          std::cos(phase)});
                break;
            }
            default: break;
            }
            track.keys.push_back({time, translation, rotation});
        }
        asset.tracks.push_back(std::move(track));
    }
    return asset;
}

void writeCharacterBinary(const CharacterAsset& asset, const std::filesystem::path& path) {
    LittleWriter payload;
    for (const SkinnedVertex& v : asset.vertices) {
        payload.bytes(&v, sizeof(v));
    }
    for (uint32_t index : asset.indices) payload.u32(index);
    for (const Bone& bone : asset.bones) {
        payload.string(bone.name);
        payload.i32(bone.parent);
        for (float value : bone.inverseBind.m) payload.f32(value);
        payload.f32(bone.bindWorld.x);
        payload.f32(bone.bindWorld.y);
        payload.f32(bone.bindWorld.z);
    }
    for (const BoneTrack& track : asset.tracks) {
        payload.u32(track.boneIndex);
        payload.u32(static_cast<uint32_t>(track.keys.size()));
        for (const Keyframe& key : track.keys) {
            payload.f32(key.time);
            payload.f32(key.translation.x);
            payload.f32(key.translation.y);
            payload.f32(key.translation.z);
            payload.f32(key.rotation.x);
            payload.f32(key.rotation.y);
            payload.f32(key.rotation.z);
            payload.f32(key.rotation.w);
        }
    }
    payload.f32(asset.bindBounds.mn.x);
    payload.f32(asset.bindBounds.mn.y);
    payload.f32(asset.bindBounds.mn.z);
    payload.f32(asset.bindBounds.mx.x);
    payload.f32(asset.bindBounds.mx.y);
    payload.f32(asset.bindBounds.mx.z);

    LittleWriter file;
    file.bytes(kMagic, sizeof(kMagic));
    file.u32(kVersion);
    file.u32(kHeaderSize);
    file.u64(payload.data().size());
    file.u64(hashPayload(std::string_view(
        reinterpret_cast<const char*>(payload.data().data()), payload.data().size())));
    file.u32(static_cast<uint32_t>(asset.bones.size()));
    file.u32(static_cast<uint32_t>(asset.vertices.size()));
    file.u32(static_cast<uint32_t>(asset.indices.size()));
    file.u32(static_cast<uint32_t>(asset.tracks.size()));
    file.u32(static_cast<uint32_t>(
        std::accumulate(asset.tracks.begin(), asset.tracks.end(), size_t{0},
                        [](size_t n, const BoneTrack& t) { return n + t.keys.size(); })));
    file.f32(asset.duration);
    file.u32(0);
    file.u32(0);
    file.bytes(payload.data().data(), payload.data().size());
    const std::string_view bytes(reinterpret_cast<const char*>(file.data().data()),
                                 file.data().size());
    writeTextFile(path, bytes);
}

CharacterAsset loadCharacterBinary(const std::filesystem::path& path) {
    const std::string bytes = readTextFile(path, ErrorCode::animationMissing,
                                           ErrorCode::animationRead, kMaxPayload + kHeaderSize);
    if (bytes.size() < kHeaderSize) {
        throw AppError(ErrorCode::animationCorrupt, "animation file is shorter than header");
    }
    LittleReader header(std::string_view(bytes.data(), kHeaderSize));
    const std::string_view magic = header.bytes(8);
    if (std::memcmp(magic.data(), kMagic, sizeof(kMagic)) != 0) {
        throw AppError(ErrorCode::animationCorrupt, "bad animation magic");
    }
    const uint32_t version = header.u32();
    if (version != kVersion) {
        throw AppError(ErrorCode::animationCorrupt, "unsupported animation version");
    }
    const uint32_t headerSize = header.u32();
    if (headerSize != kHeaderSize) {
        throw AppError(ErrorCode::animationCorrupt, "unsupported animation header size");
    }
    const uint64_t payloadSize = header.u64();
    const uint64_t storedHash = header.u64();
    const uint32_t boneCount = header.u32();
    const uint32_t vertexCount = header.u32();
    const uint32_t indexCount = header.u32();
    const uint32_t trackCount = header.u32();
    const uint32_t keyframeCount = header.u32();
    const float duration = header.f32();
    header.skip(8);

    if (payloadSize > kMaxPayload || payloadSize != bytes.size() - kHeaderSize) {
        throw AppError(ErrorCode::animationCorrupt, "invalid animation payload size");
    }
    if (boneCount == 0 || boneCount > kMaxBones || vertexCount == 0 ||
        vertexCount > kMaxVertices || indexCount == 0 || indexCount > kMaxIndices ||
        indexCount % 3 != 0 || trackCount == 0 || trackCount > boneCount ||
        keyframeCount == 0 || keyframeCount > kMaxKeys) {
        throw AppError(ErrorCode::animationCorrupt, "animation counts are out of range");
    }
    if (!std::isfinite(duration) || duration <= 0.0f || duration > 3600.0f) {
        throw AppError(ErrorCode::animationCorrupt, "invalid animation duration");
    }
    const std::string_view payloadView(bytes.data() + kHeaderSize,
                                       static_cast<size_t>(payloadSize));
    if (hashPayload(payloadView) != storedHash) {
        throw AppError(ErrorCode::animationCorrupt, "animation checksum mismatch");
    }

    CharacterAsset asset;
    asset.duration = duration;
    asset.payloadHash = storedHash;
    LittleReader reader(payloadView);
    asset.vertices.resize(vertexCount);
    for (SkinnedVertex& v : asset.vertices) {
        std::memcpy(&v, reader.bytes(sizeof(v)).data(), sizeof(v));
        if (!finite(v.px) || !finite(v.py) || !finite(v.pz) || !finite(v.nx) ||
            !finite(v.ny) || !finite(v.nz) || !finite(v.u) || !finite(v.v)) {
            throw AppError(ErrorCode::animationCorrupt, "non-finite vertex data");
        }
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            if (v.bone[i] >= boneCount) {
                throw AppError(ErrorCode::animationCorrupt, "vertex bone index out of range");
            }
            if (!finite(v.weight[i]) || v.weight[i] < 0.0f) {
                throw AppError(ErrorCode::animationCorrupt, "invalid vertex weight");
            }
            sum += v.weight[i];
        }
        if (sum < 0.98f || sum > 1.02f) {
            throw AppError(ErrorCode::animationCorrupt, "vertex weights are not normalized");
        }
    }
    asset.indices.resize(indexCount);
    for (uint32_t& index : asset.indices) {
        index = reader.u32();
        if (index >= vertexCount) {
            throw AppError(ErrorCode::animationCorrupt, "index out of range");
        }
    }
    asset.bones.resize(boneCount);
    for (uint32_t i = 0; i < boneCount; ++i) {
        Bone& bone = asset.bones[i];
        bone.name = reader.string();
        bone.parent = reader.i32();
        if (bone.parent < -1 || bone.parent >= static_cast<int32_t>(boneCount) ||
            bone.parent == static_cast<int32_t>(i)) {
            throw AppError(ErrorCode::animationCorrupt, "invalid bone parent");
        }
        for (float& value : bone.inverseBind.m) {
            value = reader.f32();
            if (!finite(value)) throw AppError(ErrorCode::animationCorrupt, "non-finite bone matrix");
        }
        bone.bindWorld = {reader.f32(), reader.f32(), reader.f32()};
    }
    asset.tracks.resize(trackCount);
    std::unordered_set<uint32_t> trackBones;
    for (BoneTrack& track : asset.tracks) {
        track.boneIndex = reader.u32();
        const uint32_t keys = reader.u32();
        if (track.boneIndex >= boneCount || keys == 0 || keys > kMaxKeys ||
            !trackBones.insert(track.boneIndex).second) {
            throw AppError(ErrorCode::animationCorrupt, "invalid animation track");
        }
        track.keys.resize(keys);
        float previousTime = -1.0f;
        for (Keyframe& key : track.keys) {
            key.time = reader.f32();
            key.translation = {reader.f32(), reader.f32(), reader.f32()};
            key.rotation = {reader.f32(), reader.f32(), reader.f32(), reader.f32()};
            if (!finite(key.time) || key.time < previousTime || key.time < 0.0f ||
                key.time > duration + 0.001f || !finite(key.translation.x) ||
                !finite(key.translation.y) || !finite(key.translation.z) ||
                !finite(key.rotation.x) || !finite(key.rotation.y) ||
                !finite(key.rotation.z) || !finite(key.rotation.w)) {
                throw AppError(ErrorCode::animationCorrupt, "invalid keyframe data");
            }
            previousTime = key.time;
        }
    }
    asset.bindBounds.mn = {reader.f32(), reader.f32(), reader.f32()};
    asset.bindBounds.mx = {reader.f32(), reader.f32(), reader.f32()};
    if (!asset.bindBounds.valid() || reader.remaining() != 0) {
        throw AppError(ErrorCode::animationCorrupt, "invalid animation bounds or trailing payload");
    }
    for (uint32_t i = 0; i < boneCount; ++i) {
        const int32_t parent = asset.bones[i].parent;
        if (parent >= 0 && static_cast<uint32_t>(parent) >= i) {
            throw AppError(ErrorCode::animationCorrupt, "bone hierarchy is not topologically ordered");
        }
    }
    return asset;
}

std::vector<Mat4> evaluatePose(const CharacterAsset& asset, float timeSeconds) {
    const float time = std::clamp(timeSeconds, 0.0f, asset.duration);
    std::vector<Mat4> local(asset.bones.size(), Mat4::identity());
    for (const BoneTrack& track : asset.tracks) {
        if (track.boneIndex >= local.size() || track.keys.empty()) continue;
        const auto& keys = track.keys;
        Vec3 translation = keys.front().translation;
        Quat rotation = keys.front().rotation;
        if (time >= keys.back().time) {
            translation = keys.back().translation;
            rotation = keys.back().rotation;
        } else {
            for (size_t i = 1; i < keys.size(); ++i) {
                if (time <= keys[i].time) {
                    const Keyframe& a = keys[i - 1];
                    const Keyframe& b = keys[i];
                    const float span = std::max(1.0e-6f, b.time - a.time);
                    const float t = (time - a.time) / span;
                    translation = lerp(a.translation, b.translation, t);
                    rotation = slerp(a.rotation, b.rotation, t);
                    break;
                }
            }
        }
        local[track.boneIndex] = Mat4::rotationQuat(rotation) * Mat4::translation(translation);
    }
    std::vector<Mat4> world(asset.bones.size(), Mat4::identity());
    std::vector<Mat4> palette(asset.bones.size(), Mat4::identity());
    for (size_t i = 0; i < asset.bones.size(); ++i) {
        const int32_t parent = asset.bones[i].parent;
        world[i] = parent < 0 ? local[i] : local[i] * world[static_cast<size_t>(parent)];
        palette[i] = asset.bones[i].inverseBind * world[i];
    }
    return palette;
}

AABB transformedCharacterBounds(const AABB& bindBounds, const std::vector<Mat4>& palette,
                                const Mat4& world) {
    if (palette.empty()) return transformedAABB(bindBounds, world);
    AABB result;
    for (int i = 0; i < 8; ++i) {
        const Vec3 p = bindBounds.corner(i);
        Vec3 skinned{};
        for (size_t bone = 0; bone < palette.size(); ++bone) {
            skinned += transformPoint(palette[bone], p) * (1.0f / static_cast<float>(palette.size()));
        }
        result.expand(transformPoint(world, skinned));
    }
    return result;
}

} // namespace uid166

