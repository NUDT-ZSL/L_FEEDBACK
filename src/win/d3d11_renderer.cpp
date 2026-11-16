#ifdef _WIN32

#include "win/d3d11_renderer.hpp"

#include "core/adaptive.hpp"
#include "core/bvh.hpp"
#include "core/command_recorder.hpp"
#include "core/culling.hpp"
#include "core/error.hpp"
#include "core/json.hpp"
#include "core/shadows.hpp"
#include "win/shaders_hlsl.hpp"

#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace uid166 {

namespace {

template <typename T>
class Com {
public:
    Com() = default;
    ~Com() { reset(); }
    Com(const Com&) = delete;
    Com& operator=(const Com&) = delete;
    Com(Com&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    Com& operator=(Com&& other) noexcept {
        if (this != &other) {
            reset(other.ptr_);
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T* get() const { return ptr_; }
    T* operator->() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    T** put() {
        reset();
        return &ptr_;
    }
    T* const* address() const { return &ptr_; }
    void reset(T* ptr = nullptr) {
        if (ptr_) ptr_->Release();
        ptr_ = ptr;
    }
    T* detach() {
        T* value = ptr_;
        ptr_ = nullptr;
        return value;
    }

private:
    T* ptr_ = nullptr;
};

std::string hresultText(HRESULT hr) {
    std::ostringstream out;
    out << "HRESULT=0x" << std::hex << static_cast<unsigned long>(hr);
    return out.str();
}

void checkHr(HRESULT hr, ErrorCode code, const char* what) {
    if (FAILED(hr)) throw AppError(code, std::string(what) + " failed, " + hresultText(hr));
}

constexpr uint32_t kStaticStride = 32;
constexpr uint32_t kCharacterStride = 52;
constexpr uint32_t kCascadeCount = 3;
constexpr uint32_t kLocalLightCount = 4;
constexpr uint32_t kCubeFaceCount = 6;
constexpr uint32_t kMaxBones = 64;
constexpr uint32_t kDefaultCascadeSize = 2048;
constexpr uint32_t kDefaultLocalSize = 1024;
constexpr uint32_t kOcclusionWidth = 480;
constexpr uint32_t kOcclusionHeight = 270;
constexpr uint32_t kMaxHardwareQueriesPerFrame = 1024;

struct StaticVertex {
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
};

struct InstanceGpu {
    Mat4 world;
    uint32_t material = 0;
    uint32_t objectId = 0;
    float padding[2]{0.0f, 0.0f};
};

static_assert(sizeof(InstanceGpu) == 80);

struct FrameConstants {
    Mat4 viewProj;
    Mat4 cascadeMatrices[kCascadeCount];
    Mat4 spotMatrices[kLocalLightCount];
    float cascadeSplits[4]{};
    Vec4 cameraPos{};
    Vec4 cameraForward{};
    Vec4 directionalDirection{};
    Vec4 directionalColor{};
    Vec4 localPosRange[kLocalLightCount]{};
    Vec4 localColorIntensity[kLocalLightCount]{};
    Vec4 localDirType[kLocalLightCount]{};
    Vec4 localAndSlices[kLocalLightCount]{};
    Vec4 materialColors[8]{};
};

struct ObjectConstants {
    Mat4 world;
    Vec4 parameters;
};

struct BoneConstants {
    Mat4 bones[kMaxBones];
};

struct ShadowConstants {
    Mat4 viewProj;
};

static_assert(sizeof(FrameConstants) == 976);
static_assert(sizeof(ObjectConstants) == 80);
static_assert(sizeof(ShadowConstants) == 64);

struct PendingQuery {
    Com<ID3D11Query> query;
    uint32_t objectId = 0;
    uint64_t issuedFrame = 0;
};

struct HardwareState {
    bool occluded = false;
    uint64_t frame = 0;
};

struct ShadowResources {
    uint32_t cascadeSize = 0;
    uint32_t localSize = 0;
    Com<ID3D11Texture2D> cascadeTexture;
    Com<ID3D11ShaderResourceView> cascadeSrv;
    std::array<Com<ID3D11DepthStencilView>, kCascadeCount> cascadeDsv;

    Com<ID3D11Texture2D> pointTexture;
    Com<ID3D11ShaderResourceView> pointSrv;
    std::array<Com<ID3D11DepthStencilView>, kCubeFaceCount * kLocalLightCount> pointDsv;

    Com<ID3D11Texture2D> spotTexture;
    Com<ID3D11ShaderResourceView> spotSrv;
    std::array<Com<ID3D11DepthStencilView>, kLocalLightCount> spotDsv;

    bool valid() const { return cascadeSize != 0 && localSize != 0; }
};

void buildCubeMesh(std::vector<StaticVertex>& vertices, std::vector<uint32_t>& indices) {
    const Vec3 normals[6] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
    };
    const Vec3 corners[6][4] = {
        {{-0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, -0.5f}},
        {{0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, 0.5f}},
        {{-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}},
        {{-0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}},
        {{0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}},
        {{-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}},
    };
    for (int face = 0; face < 6; ++face) {
        const uint32_t base = static_cast<uint32_t>(vertices.size());
        for (int corner = 0; corner < 4; ++corner) {
            const Vec3& p = corners[face][corner];
            vertices.push_back({p.x, p.y, p.z, normals[face].x, normals[face].y,
                                normals[face].z, static_cast<float>(corner & 1),
                                static_cast<float>((corner >> 1) & 1)});
        }
        indices.insert(indices.end(), {
            base + 0, base + 1, base + 2, base + 0, base + 2, base + 3,
        });
    }
}

AABB skinnedBounds(const CharacterAsset& asset, const std::vector<Mat4>& palette,
                   const Mat4& world) {
    AABB bounds;
    for (const SkinnedVertex& vertex : asset.vertices) {
        Vec3 p{};
        for (int i = 0; i < 4; ++i) {
            if (vertex.weight[i] <= 0.0f) continue;
            const uint32_t bone = vertex.bone[i];
            if (bone >= palette.size()) continue;
            p += transformPoint(palette[bone], {vertex.px, vertex.py, vertex.pz}) *
                 vertex.weight[i];
        }
        bounds.expand(transformPoint(world, p));
    }
    return bounds;
}

void setViewport(ID3D11DeviceContext* context, uint32_t width, uint32_t height) {
    D3D11_VIEWPORT viewport{};
    viewport.Width = static_cast<float>(width);
    viewport.Height = static_cast<float>(height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    context->RSSetViewports(1, &viewport);
}

} // namespace

struct D3D11Renderer::Impl {
    Impl(HWND window, uint32_t width, uint32_t height, SceneDesc scene,
         CharacterAsset character)
        : window_(window), clientWidth_(std::max(1u, width)), clientHeight_(std::max(1u, height)),
          scene_(std::move(scene)), character_(std::move(character)) {
        initialize();
    }

    ~Impl() {
        if (context_) context_->ClearState();
    }

    void initialize();
    void createSwapChain();
    void createShaders();
    void createStaticGeometry();
    void createCharacterGeometry();
    void createStaticBuffers();
    void createStates();
    void ensureSceneTargets(uint32_t width, uint32_t height);
    void ensureOcclusionTarget();
    void createShadowResources(uint32_t cascadeSize, uint32_t localSize);
    void releaseShadowResources();
    void recreateBackBuffer();
    void uploadFrameConstants(const Camera& camera, const CsmFrame& csm,
                              const LocalShadowFrame& localShadows);
    void uploadObjectConstants(const Mat4& world, uint32_t material);
    void uploadBones(const std::vector<Mat4>& palette);
    uint32_t uploadInstances(const std::vector<InstanceGpu>& instances);
    std::vector<InstanceGpu> instancesForIds(const std::vector<uint32_t>& ids) const;
    void bindMainPipeline();
    void drawStaticShadow(const std::vector<uint32_t>& ids, const Mat4& shadowViewProj);
    void drawCharacterShadow(const Mat4& shadowViewProj, const Mat4& world);
    void renderCascades(const CsmFrame& csm);
    void renderLocalShadows(const LocalShadowFrame& localShadows, const Mat4& characterWorld);
    void renderMainScene(const Camera& camera, const CsmFrame& csm,
                         const LocalShadowFrame& localShadows,
                         const RecordingResult& recording, const Mat4& characterWorld);
    void issueHardwareQueries(const CullResult& cull, const Camera& camera);
    void pollHardwareQueries();
    void updateHardwareOcclusionSet();
    void beginGpuTimer();
    void endGpuTimer();
    void pollGpuTimer();
    void present();
    [[nodiscard]] std::vector<uint32_t> chooseOccluded(const CullResult& cull) const;
    [[nodiscard]] LiveFrameStats render(uint64_t frame, float deltaSeconds, const Camera& camera);
    [[nodiscard]] uint32_t cascadeResolution(const QualityState& quality) const;
    [[nodiscard]] uint32_t localResolution(const QualityState& quality) const;

    HWND window_ = nullptr;
    uint32_t clientWidth_ = 1;
    uint32_t clientHeight_ = 1;
    bool deviceRemoved_ = false;

    Com<ID3D11Device> device_;
    Com<ID3D11DeviceContext> context_;
    Com<IDXGISwapChain> swapChain_;
    Com<ID3D11Texture2D> backBuffer_;
    Com<ID3D11RenderTargetView> backRtv_;

    Com<ID3D11Texture2D> sceneTexture_;
    Com<ID3D11RenderTargetView> sceneRtv_;
    Com<ID3D11ShaderResourceView> sceneSrv_;
    Com<ID3D11Texture2D> sceneDepthTexture_;
    Com<ID3D11DepthStencilView> sceneDepthDsv_;
    uint32_t sceneWidth_ = 0;
    uint32_t sceneHeight_ = 0;

    Com<ID3D11Texture2D> occlusionTexture_;
    Com<ID3D11DepthStencilView> occlusionDsv_;

    Com<ID3D11VertexShader> sceneVs_;
    Com<ID3D11VertexShader> skinnedVs_;
    Com<ID3D11PixelShader> scenePs_;
    Com<ID3D11VertexShader> staticShadowVs_;
    Com<ID3D11VertexShader> skinnedShadowVs_;
    Com<ID3D11VertexShader> fullscreenVs_;
    Com<ID3D11PixelShader> fullscreenPs_;
    Com<ID3D11InputLayout> staticLayout_;
    Com<ID3D11InputLayout> skinnedLayout_;

    Com<ID3D11Buffer> staticVertexBuffer_;
    Com<ID3D11Buffer> staticIndexBuffer_;
    Com<ID3D11Buffer> characterVertexBuffer_;
    Com<ID3D11Buffer> characterIndexBuffer_;
    Com<ID3D11Buffer> instanceBuffer_;
    Com<ID3D11ShaderResourceView> instanceSrv_;
    uint32_t instanceCapacity_ = 0;

    Com<ID3D11Buffer> frameCb_;
    Com<ID3D11Buffer> objectCb_;
    Com<ID3D11Buffer> boneCb_;
    Com<ID3D11Buffer> shadowCb_;
    Com<ID3D11RasterizerState> rasterState_;
    Com<ID3D11DepthStencilState> depthState_;
    Com<ID3D11DepthStencilState> queryDepthState_;
    Com<ID3D11SamplerState> shadowSampler_;
    Com<ID3D11SamplerState> linearSampler_;

    ShadowResources shadows_;
    SceneDesc scene_;
    CharacterAsset character_;
    Bvh bvh_;
    OcclusionCuller occlusion_;
    CsmUpdatePolicy csmPolicy_;
    LocalShadowPolicy localShadowPolicy_;
    AdaptiveQuality adaptive_;
    CommandRecorder recorder_;
    Camera previousCamera_;
    bool firstFrame_ = true;
    float simulationTime_ = 0.0f;
    std::vector<Mat4> bonePalette_;
    std::vector<uint32_t> lastVisible_;
    std::unordered_map<uint32_t, HardwareState> hardwareState_;
    std::vector<PendingQuery> pendingQueries_;
    uint32_t hardwareResultsThisFrame_ = 0;
    uint32_t hardwareQueriesThisFrame_ = 0;
    uint32_t hardwareOccludedThisFrame_ = 0;
    bool cameraFast_ = false;

    Com<ID3D11Query> disjointQuery_;
    Com<ID3D11Query> startQuery_;
    Com<ID3D11Query> endQuery_;
    bool gpuTimerPending_ = false;
    bool gpuTimerActive_ = false;
    float lastGpuFrameMs_ = 1000.0f / 60.0f;
    QualityState quality_{};
    uint64_t currentFrame_ = 0;
    size_t directionalLightIndex_ = 0;
};

void D3D11Renderer::Impl::initialize() {
    UINT createFlags = 0;
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_0,
    };
    D3D_FEATURE_LEVEL featureLevel{};
    checkHr(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createFlags,
                              featureLevels, 1, D3D11_SDK_VERSION, device_.put(),
                              &featureLevel, context_.put()),
            ErrorCode::deviceInit, "D3D11CreateDevice");
    createSwapChain();
    createShaders();
    createStaticGeometry();
    createCharacterGeometry();
    createStaticBuffers();
    createStates();
    std::vector<AABB> objectBounds;
    objectBounds.reserve(scene_.objects.size());
    for (const SceneObject& object : scene_.objects) objectBounds.push_back(object.bounds);
    bvh_.build(objectBounds);
    bool foundDirectional = false;
    for (size_t i = 0; i < scene_.lights.size(); ++i) {
        if (scene_.lights[i].type == LightType::directional) {
            directionalLightIndex_ = i;
            foundDirectional = true;
            break;
        }
    }
    if (!foundDirectional) {
        throw AppError(ErrorCode::sceneSchema, "no directional light in renderer scene");
    }
    ensureSceneTargets(clientWidth_, clientHeight_);
    ensureOcclusionTarget();
    uploadBones(std::vector<Mat4>(character_.bones.size(), Mat4::identity()));

    const size_t objectCount = scene_.objects.size();
    D3D11_BUFFER_DESC instanceDesc{};
    instanceDesc.ByteWidth = static_cast<UINT>(std::max<size_t>(1, objectCount) * sizeof(InstanceGpu));
    instanceDesc.Usage = D3D11_USAGE_DYNAMIC;
    instanceDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    instanceDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    instanceDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    instanceDesc.StructureByteStride = sizeof(InstanceGpu);
    checkHr(device_->CreateBuffer(&instanceDesc, nullptr, instanceBuffer_.put()),
            ErrorCode::resourceCreate, "Create instance buffer");
    instanceCapacity_ = static_cast<uint32_t>(objectCount);
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = std::max(1u, instanceCapacity_);
    checkHr(device_->CreateShaderResourceView(instanceBuffer_.get(), &srvDesc,
                                              instanceSrv_.put()),
            ErrorCode::resourceCreate, "Create instance SRV");

    releaseShadowResources();
    createShadowResources(kDefaultCascadeSize, kDefaultLocalSize);
    occlusion_.reset(scene_.objects.size());
    localShadowPolicy_.reset(scene_.lights.size());
    adaptive_ = AdaptiveQuality(AdaptiveConfig{});
    quality_ = adaptive_.state();
}

void D3D11Renderer::Impl::createSwapChain() {
    DXGI_SWAP_CHAIN_DESC desc{};
    desc.BufferDesc.Width = clientWidth_;
    desc.BufferDesc.Height = clientHeight_;
    desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.BufferDesc.RefreshRate.Numerator = 60;
    desc.BufferDesc.RefreshRate.Denominator = 1;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 2;
    desc.OutputWindow = window_;
    desc.Windowed = TRUE;
    desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    desc.Flags = 0;

    Com<IDXGIDevice> dxgiDevice;
    checkHr(device_->QueryInterface(__uuidof(IDXGIDevice),
                                    reinterpret_cast<void**>(dxgiDevice.put())),
            ErrorCode::swapChainInit, "Query IDXGIDevice");
    Com<IDXGIAdapter> adapter;
    checkHr(dxgiDevice->GetAdapter(adapter.put()), ErrorCode::swapChainInit,
            "Get DXGI adapter");
    Com<IDXGIFactory> factory;
    checkHr(adapter->GetParent(__uuidof(IDXGIFactory),
                               reinterpret_cast<void**>(factory.put())),
            ErrorCode::swapChainInit, "Get DXGI factory");
    checkHr(factory->CreateSwapChain(device_.get(), &desc, swapChain_.put()),
            ErrorCode::swapChainInit, "CreateSwapChain");
    recreateBackBuffer();
}

void D3D11Renderer::Impl::createShaders() {
    auto compile = [](const char* source, const char* entry, const char* target) {
        Com<ID3DBlob> blob;
        Com<ID3DBlob> errors;
        const HRESULT hr = D3DCompile(source, std::strlen(source), nullptr, nullptr, nullptr,
                                      entry, target, D3DCOMPILE_ENABLE_STRICTNESS, 0,
                                      blob.put(), errors.put());
        if (FAILED(hr)) {
            std::string message = "shader compile failed";
            if (errors) {
                message += ": ";
                message.append(static_cast<const char*>(errors->GetBufferPointer()),
                               errors->GetBufferSize());
            } else {
                message += ", " + hresultText(hr);
            }
            throw AppError(ErrorCode::shaderCompile, message);
        }
        return blob;
    };

    Com<ID3DBlob> blob = compile(shaders::scene, "SceneVS", "vs_5_0");
    checkHr(device_->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                                        nullptr, sceneVs_.put()),
            ErrorCode::shaderCompile, "Create SceneVS");
    const D3D11_INPUT_ELEMENT_DESC staticElements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    checkHr(device_->CreateInputLayout(staticElements, 3, blob->GetBufferPointer(),
                                       blob->GetBufferSize(), staticLayout_.put()),
            ErrorCode::shaderCompile, "Create static input layout");

    blob = compile(shaders::scene, "SkinnedVS", "vs_5_0");
    checkHr(device_->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                                        nullptr, skinnedVs_.put()),
            ErrorCode::shaderCompile, "Create SkinnedVS");
    const D3D11_INPUT_ELEMENT_DESC skinnedElements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"BLENDINDICES", 0, DXGI_FORMAT_R8G8B8A8_UINT, 0, 32, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"BLENDWEIGHT", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 36,
         D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    checkHr(device_->CreateInputLayout(skinnedElements, 5, blob->GetBufferPointer(),
                                       blob->GetBufferSize(), skinnedLayout_.put()),
            ErrorCode::shaderCompile, "Create skinned input layout");

    blob = compile(shaders::scene, "ScenePS", "ps_5_0");
    checkHr(device_->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                                       nullptr, scenePs_.put()),
            ErrorCode::shaderCompile, "Create ScenePS");

    blob = compile(shaders::shadow, "StaticShadowVS", "vs_5_0");
    checkHr(device_->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                                        nullptr, staticShadowVs_.put()),
            ErrorCode::shaderCompile, "Create StaticShadowVS");
    blob = compile(shaders::shadow, "SkinnedShadowVS", "vs_5_0");
    checkHr(device_->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                                        nullptr, skinnedShadowVs_.put()),
            ErrorCode::shaderCompile, "Create SkinnedShadowVS");

    blob = compile(shaders::fullscreen, "FullscreenVS", "vs_5_0");
    checkHr(device_->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                                        nullptr, fullscreenVs_.put()),
            ErrorCode::shaderCompile, "Create FullscreenVS");
    blob = compile(shaders::fullscreen, "FullscreenPS", "ps_5_0");
    checkHr(device_->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                                       nullptr, fullscreenPs_.put()),
            ErrorCode::shaderCompile, "Create FullscreenPS");
}

void D3D11Renderer::Impl::createStaticGeometry() {
    std::vector<StaticVertex> vertices;
    std::vector<uint32_t> indices;
    buildCubeMesh(vertices, indices);

    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(vertices.size() * sizeof(StaticVertex));
    desc.Usage = D3D11_USAGE_IMMUTABLE;
    desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA data{vertices.data(), 0, 0};
    checkHr(device_->CreateBuffer(&desc, &data, staticVertexBuffer_.put()),
            ErrorCode::resourceCreate, "Create static vertex buffer");
    desc.ByteWidth = static_cast<UINT>(indices.size() * sizeof(uint32_t));
    desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    data.pSysMem = indices.data();
    checkHr(device_->CreateBuffer(&desc, &data, staticIndexBuffer_.put()),
            ErrorCode::resourceCreate, "Create static index buffer");
}

void D3D11Renderer::Impl::createCharacterGeometry() {
    if (character_.vertices.empty() || character_.indices.empty()) {
        throw AppError(ErrorCode::resourceCreate, "character mesh is empty");
    }
    if (character_.bones.size() > kMaxBones) {
        throw AppError(ErrorCode::resourceCreate, "character skeleton exceeds 64 bones");
    }
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(character_.vertices.size() * sizeof(SkinnedVertex));
    desc.Usage = D3D11_USAGE_IMMUTABLE;
    desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA data{character_.vertices.data(), 0, 0};
    checkHr(device_->CreateBuffer(&desc, &data, characterVertexBuffer_.put()),
            ErrorCode::resourceCreate, "Create character vertex buffer");
    desc.ByteWidth = static_cast<UINT>(character_.indices.size() * sizeof(uint32_t));
    desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    data.pSysMem = character_.indices.data();
    checkHr(device_->CreateBuffer(&desc, &data, characterIndexBuffer_.put()),
            ErrorCode::resourceCreate, "Create character index buffer");
}

void D3D11Renderer::Impl::createStaticBuffers() {
    D3D11_BUFFER_DESC desc{};
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    desc.ByteWidth = sizeof(FrameConstants);
    checkHr(device_->CreateBuffer(&desc, nullptr, frameCb_.put()),
            ErrorCode::resourceCreate, "Create frame CB");
    desc.ByteWidth = sizeof(ObjectConstants);
    checkHr(device_->CreateBuffer(&desc, nullptr, objectCb_.put()),
            ErrorCode::resourceCreate, "Create object CB");
    desc.ByteWidth = sizeof(BoneConstants);
    checkHr(device_->CreateBuffer(&desc, nullptr, boneCb_.put()),
            ErrorCode::resourceCreate, "Create bone CB");
    desc.ByteWidth = sizeof(ShadowConstants);
    checkHr(device_->CreateBuffer(&desc, nullptr, shadowCb_.put()),
            ErrorCode::resourceCreate, "Create shadow CB");
}

void D3D11Renderer::Impl::createStates() {
    D3D11_RASTERIZER_DESC raster{};
    raster.FillMode = D3D11_FILL_SOLID;
    raster.CullMode = D3D11_CULL_NONE;
    raster.FrontCounterClockwise = FALSE;
    raster.DepthBias = 0;
    raster.DepthClipEnable = TRUE;
    raster.ScissorEnable = FALSE;
    raster.MultisampleEnable = FALSE;
    raster.AntialiasedLineEnable = FALSE;
    checkHr(device_->CreateRasterizerState(&raster, rasterState_.put()),
            ErrorCode::resourceCreate, "Create rasterizer state");

    D3D11_DEPTH_STENCIL_DESC depth{};
    depth.DepthEnable = TRUE;
    depth.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    depth.DepthFunc = D3D11_COMPARISON_LESS;
    depth.StencilEnable = FALSE;
    checkHr(device_->CreateDepthStencilState(&depth, depthState_.put()),
            ErrorCode::resourceCreate, "Create depth state");
    depth.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    depth.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
    checkHr(device_->CreateDepthStencilState(&depth, queryDepthState_.put()),
            ErrorCode::resourceCreate, "Create query depth state");

    D3D11_SAMPLER_DESC sampler{};
    sampler.Filter = D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
    sampler.AddressU = D3D11_TEXTURE_ADDRESS_BORDER;
    sampler.AddressV = D3D11_TEXTURE_ADDRESS_BORDER;
    sampler.AddressW = D3D11_TEXTURE_ADDRESS_BORDER;
    sampler.ComparisonFunc = D3D11_COMPARISON_LESS_EQUAL;
    sampler.BorderColor[0] = 1.0f;
    sampler.BorderColor[1] = 1.0f;
    sampler.BorderColor[2] = 1.0f;
    sampler.BorderColor[3] = 1.0f;
    sampler.MinLOD = 0.0f;
    sampler.MaxLOD = 0.0f;
    checkHr(device_->CreateSamplerState(&sampler, shadowSampler_.put()),
            ErrorCode::resourceCreate, "Create shadow sampler");
    sampler.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler.BorderColor[0] = 0.0f;
    sampler.BorderColor[1] = 0.0f;
    sampler.BorderColor[2] = 0.0f;
    sampler.BorderColor[3] = 0.0f;
    checkHr(device_->CreateSamplerState(&sampler, linearSampler_.put()),
            ErrorCode::resourceCreate, "Create linear sampler");
}

void D3D11Renderer::Impl::ensureSceneTargets(uint32_t width, uint32_t height) {
    width = std::max(1u, width);
    height = std::max(1u, height);
    if (sceneWidth_ == width && sceneHeight_ == height && sceneTexture_) return;
    sceneSrv_.reset();
    sceneRtv_.reset();
    sceneDepthDsv_.reset();
    sceneDepthTexture_.reset();
    sceneTexture_.reset();

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    checkHr(device_->CreateTexture2D(&desc, nullptr, sceneTexture_.put()),
            ErrorCode::resourceCreate, "Create scene color");
    checkHr(device_->CreateRenderTargetView(sceneTexture_.get(), nullptr, sceneRtv_.put()),
            ErrorCode::resourceCreate, "Create scene RTV");
    checkHr(device_->CreateShaderResourceView(sceneTexture_.get(), nullptr, sceneSrv_.put()),
            ErrorCode::resourceCreate, "Create scene SRV");

    desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    checkHr(device_->CreateTexture2D(&desc, nullptr, sceneDepthTexture_.put()),
            ErrorCode::resourceCreate, "Create scene depth");
    checkHr(device_->CreateDepthStencilView(sceneDepthTexture_.get(), nullptr,
                                            sceneDepthDsv_.put()),
            ErrorCode::resourceCreate, "Create scene DSV");
    sceneWidth_ = width;
    sceneHeight_ = height;
}

void D3D11Renderer::Impl::ensureOcclusionTarget() {
    if (occlusionTexture_) return;
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = kOcclusionWidth;
    desc.Height = kOcclusionHeight;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    checkHr(device_->CreateTexture2D(&desc, nullptr, occlusionTexture_.put()),
            ErrorCode::resourceCreate, "Create occlusion depth");
    checkHr(device_->CreateDepthStencilView(occlusionTexture_.get(), nullptr,
                                            occlusionDsv_.put()),
            ErrorCode::resourceCreate, "Create occlusion DSV");
}

void D3D11Renderer::Impl::releaseShadowResources() {
    for (auto& dsv : shadows_.cascadeDsv) dsv.reset();
    for (auto& dsv : shadows_.pointDsv) dsv.reset();
    for (auto& dsv : shadows_.spotDsv) dsv.reset();
    shadows_.cascadeSrv.reset();
    shadows_.cascadeTexture.reset();
    shadows_.pointSrv.reset();
    shadows_.pointTexture.reset();
    shadows_.spotSrv.reset();
    shadows_.spotTexture.reset();
    shadows_.cascadeSize = 0;
    shadows_.localSize = 0;
}

void D3D11Renderer::Impl::createShadowResources(uint32_t cascadeSize,
                                                uint32_t localSize) {
    cascadeSize = std::max(64u, cascadeSize);
    localSize = std::max(64u, localSize);
    if (shadows_.valid() && shadows_.cascadeSize == cascadeSize &&
        shadows_.localSize == localSize) {
        return;
    }
    releaseShadowResources();
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = cascadeSize;
    desc.Height = cascadeSize;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R32_TYPELESS;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;

    desc.ArraySize = kCascadeCount;
    checkHr(device_->CreateTexture2D(&desc, nullptr, shadows_.cascadeTexture.put()),
            ErrorCode::resourceCreate, "Create cascade shadow texture");
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
    srvDesc.Texture2DArray.MostDetailedMip = 0;
    srvDesc.Texture2DArray.MipLevels = 1;
    srvDesc.Texture2DArray.FirstArraySlice = 0;
    srvDesc.Texture2DArray.ArraySize = kCascadeCount;
    checkHr(device_->CreateShaderResourceView(shadows_.cascadeTexture.get(), &srvDesc,
                                              shadows_.cascadeSrv.put()),
            ErrorCode::resourceCreate, "Create cascade shadow SRV");
    for (uint32_t face = 0; face < kCascadeCount; ++face) {
        D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
        dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
        dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
        dsvDesc.Texture2DArray.MipSlice = 0;
        dsvDesc.Texture2DArray.FirstArraySlice = face;
        dsvDesc.Texture2DArray.ArraySize = 1;
        checkHr(device_->CreateDepthStencilView(shadows_.cascadeTexture.get(), &dsvDesc,
                                                shadows_.cascadeDsv[face].put()),
                ErrorCode::resourceCreate, "Create cascade DSV");
        context_->ClearDepthStencilView(shadows_.cascadeDsv[face].get(),
                                        D3D11_CLEAR_DEPTH, 1.0f, 0);
    }

    desc.Width = localSize;
    desc.Height = localSize;
    desc.ArraySize = kCubeFaceCount * kLocalLightCount;
    desc.MiscFlags = D3D11_RESOURCE_MISC_TEXTURECUBE;
    checkHr(device_->CreateTexture2D(&desc, nullptr, shadows_.pointTexture.put()),
            ErrorCode::resourceCreate, "Create point shadow cube array");
    D3D11_SHADER_RESOURCE_VIEW_DESC cubeSrv{};
    cubeSrv.Format = DXGI_FORMAT_R32_FLOAT;
    cubeSrv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBEARRAY;
    cubeSrv.TextureCubeArray.MostDetailedMip = 0;
    cubeSrv.TextureCubeArray.MipLevels = 1;
    cubeSrv.TextureCubeArray.First2DArrayFace = 0;
    cubeSrv.TextureCubeArray.NumCubes = kLocalLightCount;
    checkHr(device_->CreateShaderResourceView(shadows_.pointTexture.get(), &cubeSrv,
                                              shadows_.pointSrv.put()),
            ErrorCode::resourceCreate, "Create point shadow SRV");
    for (uint32_t face = 0; face < kCubeFaceCount * kLocalLightCount; ++face) {
        D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
        dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
        dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
        dsvDesc.Texture2DArray.FirstArraySlice = face;
        dsvDesc.Texture2DArray.ArraySize = 1;
        checkHr(device_->CreateDepthStencilView(shadows_.pointTexture.get(), &dsvDesc,
                                                shadows_.pointDsv[face].put()),
                ErrorCode::resourceCreate, "Create point shadow DSV");
        context_->ClearDepthStencilView(shadows_.pointDsv[face].get(),
                                        D3D11_CLEAR_DEPTH, 1.0f, 0);
    }

    desc.MiscFlags = 0;
    desc.ArraySize = kLocalLightCount;
    checkHr(device_->CreateTexture2D(&desc, nullptr, shadows_.spotTexture.put()),
            ErrorCode::resourceCreate, "Create spot shadow array");
    srvDesc.Texture2DArray.ArraySize = kLocalLightCount;
    checkHr(device_->CreateShaderResourceView(shadows_.spotTexture.get(), &srvDesc,
                                              shadows_.spotSrv.put()),
            ErrorCode::resourceCreate, "Create spot shadow SRV");
    for (uint32_t slice = 0; slice < kLocalLightCount; ++slice) {
        D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
        dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
        dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
        dsvDesc.Texture2DArray.FirstArraySlice = slice;
        dsvDesc.Texture2DArray.ArraySize = 1;
        checkHr(device_->CreateDepthStencilView(shadows_.spotTexture.get(), &dsvDesc,
                                                shadows_.spotDsv[slice].put()),
                ErrorCode::resourceCreate, "Create spot shadow DSV");
        context_->ClearDepthStencilView(shadows_.spotDsv[slice].get(),
                                        D3D11_CLEAR_DEPTH, 1.0f, 0);
    }
    shadows_.cascadeSize = cascadeSize;
    shadows_.localSize = localSize;
}

void D3D11Renderer::Impl::recreateBackBuffer() {
    backRtv_.reset();
    backBuffer_.reset();
    checkHr(swapChain_->GetBuffer(0, __uuidof(ID3D11Texture2D),
                                  reinterpret_cast<void**>(backBuffer_.put())),
            ErrorCode::swapChainInit, "Get swap chain back buffer");
    checkHr(device_->CreateRenderTargetView(backBuffer_.get(), nullptr, backRtv_.put()),
            ErrorCode::swapChainInit, "Create back buffer RTV");
}

void D3D11Renderer::Impl::uploadFrameConstants(const Camera& camera, const CsmFrame& csm,
                                               const LocalShadowFrame& localShadows) {
    FrameConstants constants{};
    constants.viewProj = camera.viewProjection();
    for (uint32_t i = 0; i < std::min<uint32_t>(kCascadeCount, csm.cascades.size()); ++i) {
        constants.cascadeMatrices[i] = csm.cascades[i].matrix;
    }
    constants.cascadeSplits[0] = csm.cascades.size() > 0 ? csm.cascades[0].splitFar : camera.farZ;
    constants.cascadeSplits[1] = csm.cascades.size() > 1 ? csm.cascades[1].splitFar : camera.farZ;
    constants.cascadeSplits[2] = csm.cascades.size() > 2 ? csm.cascades[2].splitFar : camera.farZ;
    constants.cascadeSplits[3] = camera.nearZ;
    constants.cameraPos = {camera.eye.x, camera.eye.y, camera.eye.z, 1.0f};
    const Vec3 forward = normalize(camera.target - camera.eye);
    constants.cameraForward = {forward.x, forward.y, forward.z, 0.0f};

    for (const Light& light : scene_.lights) {
        if (light.type != LightType::directional) continue;
        constants.directionalDirection = {light.direction.x, light.direction.y,
                                           light.direction.z, 1.0f};
        constants.directionalColor = {light.color.x, light.color.y, light.color.z,
                                      light.intensity};
        break;
    }

    uint32_t localIndex = 0;
    for (const LocalShadowDecision& decision : localShadows.lights) {
        if (localIndex >= kLocalLightCount) break;
        const Light* light = nullptr;
        for (const Light& candidate : scene_.lights) {
            if (candidate.id == decision.lightId) {
                light = &candidate;
                break;
            }
        }
        if (!light) {
            ++localIndex;
            continue;
        }
        constants.localPosRange[localIndex] = {light->position.x, light->position.y,
                                               light->position.z, light->range};
        constants.localColorIntensity[localIndex] = {light->color.x, light->color.y,
                                                     light->color.z, light->intensity};
        constants.localDirType[localIndex] = {
            light->direction.x, light->direction.y, light->direction.z,
            light->type == LightType::point ? 1.0f : 2.0f,
        };
        constants.localAndSlices[localIndex] = {
            light->innerCos, light->outerCos, static_cast<float>(localIndex),
            static_cast<float>(localIndex),
        };
        if (light->type == LightType::spot) {
            for (const ShadowMatrixInfo& info : decision.matrices) {
                if (info.face == "spot") {
                    constants.spotMatrices[localIndex] = info.matrix;
                    break;
                }
            }
        }
        ++localIndex;
    }
    constants.materialColors[0] = {0.46f, 0.52f, 0.61f, 1.0f};
    constants.materialColors[1] = {0.68f, 0.58f, 0.43f, 1.0f};
    constants.materialColors[2] = {0.38f, 0.51f, 0.44f, 1.0f};
    constants.materialColors[3] = {0.58f, 0.43f, 0.40f, 1.0f};
    constants.materialColors[4] = {0.42f, 0.46f, 0.58f, 1.0f};
    constants.materialColors[5] = {0.60f, 0.62f, 0.66f, 1.0f};
    constants.materialColors[6] = {0.72f, 0.64f, 0.50f, 1.0f};
    constants.materialColors[7] = {0.34f, 0.36f, 0.40f, 1.0f};

    D3D11_MAPPED_SUBRESOURCE mapped{};
    checkHr(context_->Map(frameCb_.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped),
            ErrorCode::resourceCreate, "Map frame CB");
    std::memcpy(mapped.pData, &constants, sizeof(constants));
    context_->Unmap(frameCb_.get(), 0);
}

void D3D11Renderer::Impl::uploadObjectConstants(const Mat4& world, uint32_t material) {
    ObjectConstants constants;
    constants.world = world;
    constants.parameters = {static_cast<float>(material), 0.0f, 0.0f, 0.0f};
    D3D11_MAPPED_SUBRESOURCE mapped{};
    checkHr(context_->Map(objectCb_.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped),
            ErrorCode::resourceCreate, "Map object CB");
    std::memcpy(mapped.pData, &constants, sizeof(constants));
    context_->Unmap(objectCb_.get(), 0);
}

void D3D11Renderer::Impl::uploadBones(const std::vector<Mat4>& palette) {
    BoneConstants constants;
    for (uint32_t i = 0; i < kMaxBones; ++i) {
        constants.bones[i] = i < palette.size() ? palette[i] : Mat4::identity();
    }
    D3D11_MAPPED_SUBRESOURCE mapped{};
    checkHr(context_->Map(boneCb_.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped),
            ErrorCode::resourceCreate, "Map bone CB");
    std::memcpy(mapped.pData, &constants, sizeof(constants));
    context_->Unmap(boneCb_.get(), 0);
}

uint32_t D3D11Renderer::Impl::uploadInstances(const std::vector<InstanceGpu>& instances) {
    if (instances.size() > instanceCapacity_) {
        throw AppError(ErrorCode::resourceCreate, "instance upload exceeds buffer capacity");
    }
    if (instances.empty()) return 0;
    D3D11_MAPPED_SUBRESOURCE mapped{};
    checkHr(context_->Map(instanceBuffer_.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped),
            ErrorCode::resourceCreate, "Map instance buffer");
    std::memcpy(mapped.pData, instances.data(), instances.size() * sizeof(InstanceGpu));
    context_->Unmap(instanceBuffer_.get(), 0);
    return static_cast<uint32_t>(instances.size());
}

std::vector<InstanceGpu> D3D11Renderer::Impl::instancesForIds(
    const std::vector<uint32_t>& ids) const {
    std::vector<InstanceGpu> instances;
    instances.reserve(ids.size());
    for (uint32_t id : ids) {
        if (id >= scene_.objects.size()) continue;
        InstanceGpu instance;
        instance.world = objectWorldMatrix(scene_.objects[id]);
        instance.material = scene_.objects[id].material;
        instance.objectId = id;
        instances.push_back(instance);
    }
    return instances;
}

void D3D11Renderer::Impl::bindMainPipeline() {
    context_->IASetInputLayout(staticLayout_.get());
    context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context_->VSSetShader(sceneVs_.get(), nullptr, 0);
    context_->PSSetShader(scenePs_.get(), nullptr, 0);
    ID3D11Buffer* frameBuffer = frameCb_.get();
    context_->VSSetConstantBuffers(0, 1, &frameBuffer);
    context_->PSSetConstantBuffers(0, 1, &frameBuffer);
    ID3D11ShaderResourceView* instanceView = instanceSrv_.get();
    context_->VSSetShaderResources(0, 1, &instanceView);
    ID3D11ShaderResourceView* shadowViews[3] = {
        shadows_.cascadeSrv.get(), shadows_.pointSrv.get(), shadows_.spotSrv.get(),
    };
    context_->PSSetShaderResources(1, 3, shadowViews);
    ID3D11SamplerState* sampler = shadowSampler_.get();
    context_->PSSetSamplers(0, 1, &sampler);
    context_->RSSetState(rasterState_.get());
}

void D3D11Renderer::Impl::drawStaticShadow(const std::vector<uint32_t>& ids,
                                           const Mat4& shadowViewProj) {
    if (ids.empty()) return;
    std::vector<InstanceGpu> instances = instancesForIds(ids);
    const uint32_t count = uploadInstances(instances);
    ShadowConstants constants{shadowViewProj};
    D3D11_MAPPED_SUBRESOURCE mapped{};
    checkHr(context_->Map(shadowCb_.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped),
            ErrorCode::resourceCreate, "Map shadow CB");
    std::memcpy(mapped.pData, &constants, sizeof(constants));
    context_->Unmap(shadowCb_.get(), 0);
    ID3D11Buffer* shadowBuffer = shadowCb_.get();
    context_->VSSetConstantBuffers(3, 1, &shadowBuffer);
    context_->IASetInputLayout(staticLayout_.get());
    context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context_->VSSetShader(staticShadowVs_.get(), nullptr, 0);
    ID3D11ShaderResourceView* instanceView = instanceSrv_.get();
    context_->VSSetShaderResources(0, 1, &instanceView);
    ID3D11Buffer* indexBuffer = staticIndexBuffer_.get();
    context_->IASetIndexBuffer(indexBuffer, DXGI_FORMAT_R32_UINT, 0);
    const UINT stride = kStaticStride;
    const UINT offset = 0;
    ID3D11Buffer* vertexBuffer = staticVertexBuffer_.get();
    context_->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
    context_->DrawIndexedInstanced(36, count, 0, 0, 0);
}

void D3D11Renderer::Impl::drawCharacterShadow(const Mat4& shadowViewProj,
                                               const Mat4& world) {
    ShadowConstants constants{shadowViewProj};
    D3D11_MAPPED_SUBRESOURCE mapped{};
    checkHr(context_->Map(shadowCb_.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped),
            ErrorCode::resourceCreate, "Map shadow CB");
    std::memcpy(mapped.pData, &constants, sizeof(constants));
    context_->Unmap(shadowCb_.get(), 0);
    uploadObjectConstants(world, 6);
    context_->IASetInputLayout(skinnedLayout_.get());
    context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context_->VSSetShader(skinnedShadowVs_.get(), nullptr, 0);
    ID3D11Buffer* objectBuffer = objectCb_.get();
    context_->VSSetConstantBuffers(1, 1, &objectBuffer);
    ID3D11Buffer* boneBuffer = boneCb_.get();
    context_->VSSetConstantBuffers(2, 1, &boneBuffer);
    ID3D11Buffer* shadowBuffer = shadowCb_.get();
    context_->VSSetConstantBuffers(3, 1, &shadowBuffer);
    ID3D11Buffer* indexBuffer = characterIndexBuffer_.get();
    context_->IASetIndexBuffer(indexBuffer, DXGI_FORMAT_R32_UINT, 0);
    const UINT stride = kCharacterStride;
    const UINT offset = 0;
    ID3D11Buffer* vertexBuffer = characterVertexBuffer_.get();
    context_->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
    context_->DrawIndexed(static_cast<UINT>(character_.indices.size()), 0, 0);
}

void D3D11Renderer::Impl::renderCascades(const CsmFrame& csm) {
    context_->OMSetRenderTargets(0, nullptr, nullptr);
    context_->PSSetShader(nullptr, nullptr, 0);
    context_->IASetInputLayout(staticLayout_.get());
    context_->VSSetShader(staticShadowVs_.get(), nullptr, 0);
    context_->OMSetDepthStencilState(depthState_.get(), 0);
    context_->RSSetState(rasterState_.get());
    for (const Cascade& cascade : csm.cascades) {
        if (!cascade.updated || cascade.index >= kCascadeCount) continue;
        ID3D11DepthStencilView* dsv = shadows_.cascadeDsv[cascade.index].get();
        context_->OMSetRenderTargets(0, nullptr, dsv);
        context_->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);
        setViewport(context_.get(), shadows_.cascadeSize, shadows_.cascadeSize);
        std::vector<uint32_t> casters;
        bvh_.queryFrustum(cascade.frustum, casters);
        drawStaticShadow(casters, cascade.matrix);
        drawCharacterShadow(cascade.matrix, Mat4::identity());
    }
}

void D3D11Renderer::Impl::renderLocalShadows(const LocalShadowFrame& localShadows,
                                             const Mat4& characterWorld) {
    context_->OMSetRenderTargets(0, nullptr, nullptr);
    context_->PSSetShader(nullptr, nullptr, 0);
    context_->IASetInputLayout(staticLayout_.get());
    context_->VSSetShader(staticShadowVs_.get(), nullptr, 0);
    context_->OMSetDepthStencilState(depthState_.get(), 0);
    context_->RSSetState(rasterState_.get());
    uint32_t localIndex = 0;
    for (const LocalShadowDecision& light : localShadows.lights) {
        if (localIndex >= kLocalLightCount) break;
        if (light.updated) {
            if (light.type == LightType::point) {
                for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
                    const uint32_t slice = localIndex * kCubeFaceCount + face;
                    ID3D11DepthStencilView* dsv = shadows_.pointDsv[slice].get();
                    context_->OMSetRenderTargets(0, nullptr, dsv);
                    context_->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);
                    setViewport(context_.get(), shadows_.localSize, shadows_.localSize);
                    if (face < light.matrices.size()) {
                        const Mat4& matrix = light.matrices[face].matrix;
                        const Frustum frustum = frustumFromViewProj(matrix);
                        std::vector<uint32_t> casters;
                        bvh_.queryFrustum(frustum, casters);
                        drawStaticShadow(casters, matrix);
                        drawCharacterShadow(matrix, characterWorld);
                    }
                }
            } else if (light.type == LightType::spot) {
                ID3D11DepthStencilView* dsv = shadows_.spotDsv[localIndex].get();
                context_->OMSetRenderTargets(0, nullptr, dsv);
                context_->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);
                setViewport(context_.get(), shadows_.localSize, shadows_.localSize);
                for (const ShadowMatrixInfo& info : light.matrices) {
                    if (info.face != "spot") continue;
                    const Frustum frustum = frustumFromViewProj(info.matrix);
                    std::vector<uint32_t> casters;
                    bvh_.queryFrustum(frustum, casters);
                    drawStaticShadow(casters, info.matrix);
                    drawCharacterShadow(info.matrix, characterWorld);
                }
            }
        }
        ++localIndex;
    }
}

void D3D11Renderer::Impl::renderMainScene(const Camera& camera, const CsmFrame& csm,
                                          const LocalShadowFrame& localShadows,
                                          const RecordingResult& recording,
                                          const Mat4& characterWorld) {
    (void)camera;
    ID3D11RenderTargetView* rtv = sceneRtv_.get();
    context_->OMSetRenderTargets(1, &rtv, sceneDepthDsv_.get());
    setViewport(context_.get(), sceneWidth_, sceneHeight_);
    const float clearColor[4] = {0.018f, 0.024f, 0.040f, 1.0f};
    context_->ClearRenderTargetView(sceneRtv_.get(), clearColor);
    context_->ClearDepthStencilView(sceneDepthDsv_.get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
    context_->OMSetDepthStencilState(depthState_.get(), 0);
    uploadFrameConstants(camera, csm, localShadows);
    bindMainPipeline();

    std::vector<InstanceGpu> allInstances;
    allInstances.reserve(recording.commands.size());
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    ranges.reserve(recording.batches.size());
    for (const RenderBatch& batch : recording.batches) {
        const uint32_t first = static_cast<uint32_t>(allInstances.size());
        for (const RenderCommand& command : batch.commands) {
            InstanceGpu instance;
            instance.world = command.world;
            instance.material = command.material;
            instance.objectId = command.objectId;
            allInstances.push_back(instance);
        }
        ranges.emplace_back(first, static_cast<uint32_t>(batch.commands.size()));
    }
    uploadInstances(allInstances);
    ID3D11Buffer* indexBuffer = staticIndexBuffer_.get();
    context_->IASetIndexBuffer(indexBuffer, DXGI_FORMAT_R32_UINT, 0);
    const UINT stride = kStaticStride;
    const UINT offset = 0;
    ID3D11Buffer* vertexBuffer = staticVertexBuffer_.get();
    context_->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
    for (const auto& [first, count] : ranges) {
        if (count != 0) context_->DrawIndexedInstanced(36, count, 0, first, 0);
    }

    context_->IASetInputLayout(skinnedLayout_.get());
    context_->VSSetShader(skinnedVs_.get(), nullptr, 0);
    ID3D11Buffer* objectBuffer = objectCb_.get();
    context_->VSSetConstantBuffers(1, 1, &objectBuffer);
    ID3D11Buffer* boneBuffer = boneCb_.get();
    context_->VSSetConstantBuffers(2, 1, &boneBuffer);
    uploadObjectConstants(characterWorld, 6);
    indexBuffer = characterIndexBuffer_.get();
    context_->IASetIndexBuffer(indexBuffer, DXGI_FORMAT_R32_UINT, 0);
    const UINT charStride = kCharacterStride;
    ID3D11Buffer* charVertexBuffer = characterVertexBuffer_.get();
    context_->IASetVertexBuffers(0, 1, &charVertexBuffer, &charStride, &offset);
    context_->DrawIndexed(static_cast<UINT>(character_.indices.size()), 0, 0);

    ID3D11RenderTargetView* back = backRtv_.get();
    context_->OMSetRenderTargets(1, &back, nullptr);
    setViewport(context_.get(), clientWidth_, clientHeight_);
    context_->IASetInputLayout(nullptr);
    context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context_->VSSetShader(fullscreenVs_.get(), nullptr, 0);
    context_->PSSetShader(fullscreenPs_.get(), nullptr, 0);
    ID3D11ShaderResourceView* sceneView = sceneSrv_.get();
    context_->PSSetShaderResources(0, 1, &sceneView);
    ID3D11SamplerState* linear = linearSampler_.get();
    context_->PSSetSamplers(0, 1, &linear);
    context_->Draw(3, 0);
}

void D3D11Renderer::Impl::pollHardwareQueries() {
    hardwareResultsThisFrame_ = 0;
    auto it = pendingQueries_.begin();
    while (it != pendingQueries_.end()) {
        BOOL visible = FALSE;
        const HRESULT hr = context_->GetData(it->query.get(), &visible, sizeof(visible),
                                             D3D11_ASYNC_GETDATA_DONOTFLUSH);
        if (hr == S_OK) {
            hardwareState_[it->objectId] = {visible == FALSE, currentFrame_};
            ++hardwareResultsThisFrame_;
            it = pendingQueries_.erase(it);
        } else {
            ++it;
        }
    }
}

void D3D11Renderer::Impl::updateHardwareOcclusionSet() {
    hardwareOccludedThisFrame_ = 0;
    for (const auto& [id, state] : hardwareState_) {
        (void)id;
        if (state.frame + 2 >= currentFrame_ && state.occluded) {
            ++hardwareOccludedThisFrame_;
        }
    }
}

std::vector<uint32_t> D3D11Renderer::Impl::chooseOccluded(const CullResult& cull) const {
    if (cull.cameraFast || hardwareState_.empty()) return cull.occluded;
    std::vector<uint32_t> result = cull.occluded;
    std::unordered_set<uint32_t> known(result.begin(), result.end());
    for (uint32_t id : cull.frustumVisible) {
        if (id >= scene_.objects.size() || scene_.objects[id].motion.enabled) continue;
        const auto found = hardwareState_.find(id);
        if (found != hardwareState_.end() && found->second.occluded &&
            found->second.frame + 2 >= currentFrame_ && known.insert(id).second) {
            result.push_back(id);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

void D3D11Renderer::Impl::issueHardwareQueries(const CullResult& cull, const Camera& camera) {
    hardwareQueriesThisFrame_ = 0;
    if (pendingQueries_.size() > 8192) return;
    ensureOcclusionTarget();
    std::vector<uint32_t> candidates;
    candidates.reserve(cull.frustumVisible.size());
    for (uint32_t id : cull.frustumVisible) {
        const bool prior = std::binary_search(lastVisible_.begin(), lastVisible_.end(), id);
        if (!prior && id < scene_.objects.size()) candidates.push_back(id);
    }
    if (candidates.size() > kMaxHardwareQueriesPerFrame) {
        candidates.resize(kMaxHardwareQueriesPerFrame);
    }

    context_->OMSetRenderTargets(0, nullptr, occlusionDsv_.get());
    context_->ClearDepthStencilView(occlusionDsv_.get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
    setViewport(context_.get(), kOcclusionWidth, kOcclusionHeight);
    context_->OMSetDepthStencilState(depthState_.get(), 0);
    context_->PSSetShader(nullptr, nullptr, 0);
    context_->IASetInputLayout(staticLayout_.get());
    context_->VSSetShader(staticShadowVs_.get(), nullptr, 0);
    context_->RSSetState(rasterState_.get());
    ShadowConstants shadow{camera.viewProjection()};
    D3D11_MAPPED_SUBRESOURCE mapped{};
    checkHr(context_->Map(shadowCb_.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped),
            ErrorCode::resourceCreate, "Map shadow CB");
    std::memcpy(mapped.pData, &shadow, sizeof(shadow));
    context_->Unmap(shadowCb_.get(), 0);
    ID3D11Buffer* shadowBuffer = shadowCb_.get();
    context_->VSSetConstantBuffers(3, 1, &shadowBuffer);
    ID3D11ShaderResourceView* instanceView = instanceSrv_.get();
    context_->VSSetShaderResources(0, 1, &instanceView);
    ID3D11Buffer* indexBuffer = staticIndexBuffer_.get();
    context_->IASetIndexBuffer(indexBuffer, DXGI_FORMAT_R32_UINT, 0);
    const UINT stride = kStaticStride;
    const UINT offset = 0;
    ID3D11Buffer* vertexBuffer = staticVertexBuffer_.get();
    context_->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);

    std::vector<InstanceGpu> occluders;
    occluders.reserve(lastVisible_.size());
    for (uint32_t id : lastVisible_) {
        if (id >= scene_.objects.size()) continue;
        InstanceGpu instance;
        instance.world = objectWorldMatrix(scene_.objects[id]);
        instance.objectId = id;
        occluders.push_back(instance);
    }
    if (!occluders.empty()) {
        uploadInstances(occluders);
        context_->DrawIndexedInstanced(36, static_cast<UINT>(occluders.size()), 0, 0, 0);
    }

    std::vector<InstanceGpu> proxies;
    proxies.reserve(candidates.size());
    for (uint32_t id : candidates) {
        InstanceGpu proxy;
        proxy.world = objectWorldMatrix(scene_.objects[id]);
        proxy.objectId = id;
        proxies.push_back(proxy);
    }
    if (!proxies.empty()) {
        uploadInstances(proxies);
        context_->OMSetDepthStencilState(queryDepthState_.get(), 0);
        for (uint32_t i = 0; i < proxies.size(); ++i) {
            PendingQuery pending;
            D3D11_QUERY_DESC queryDesc{};
            queryDesc.Query = D3D11_QUERY_OCCLUSION_PREDICATE;
            checkHr(device_->CreateQuery(&queryDesc, pending.query.put()),
                    ErrorCode::resourceCreate, "Create occlusion query");
            pending.objectId = proxies[i].objectId;
            pending.issuedFrame = currentFrame_;
            context_->Begin(pending.query.get());
            context_->DrawIndexedInstanced(36, 1, 0, i, 0);
            context_->End(pending.query.get());
            pendingQueries_.push_back(std::move(pending));
            ++hardwareQueriesThisFrame_;
        }
        context_->OMSetDepthStencilState(depthState_.get(), 0);
    }
}

void D3D11Renderer::Impl::beginGpuTimer() {
    if (!disjointQuery_) {
        D3D11_QUERY_DESC desc{};
        desc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
        checkHr(device_->CreateQuery(&desc, disjointQuery_.put()),
                ErrorCode::resourceCreate, "Create disjoint query");
        desc.Query = D3D11_QUERY_TIMESTAMP;
        checkHr(device_->CreateQuery(&desc, startQuery_.put()),
                ErrorCode::resourceCreate, "Create start timestamp");
        checkHr(device_->CreateQuery(&desc, endQuery_.put()),
                ErrorCode::resourceCreate, "Create end timestamp");
    }
    if (gpuTimerPending_) return;
    context_->Begin(disjointQuery_.get());
    context_->End(startQuery_.get());
    gpuTimerActive_ = true;
}

void D3D11Renderer::Impl::endGpuTimer() {
    if (!gpuTimerActive_) return;
    context_->End(endQuery_.get());
    context_->End(disjointQuery_.get());
    gpuTimerPending_ = true;
    gpuTimerActive_ = false;
}

void D3D11Renderer::Impl::pollGpuTimer() {
    if (!gpuTimerPending_) return;
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint{};
    UINT64 start = 0;
    UINT64 end = 0;
    const HRESULT h0 = context_->GetData(disjointQuery_.get(), &disjoint, sizeof(disjoint),
                                         D3D11_ASYNC_GETDATA_DONOTFLUSH);
    const HRESULT h1 = context_->GetData(startQuery_.get(), &start, sizeof(start),
                                         D3D11_ASYNC_GETDATA_DONOTFLUSH);
    const HRESULT h2 = context_->GetData(endQuery_.get(), &end, sizeof(end),
                                         D3D11_ASYNC_GETDATA_DONOTFLUSH);
    if (h0 == S_OK && h1 == S_OK && h2 == S_OK) {
        if (!disjoint.Disjoint && disjoint.Frequency != 0 && end > start) {
            lastGpuFrameMs_ = static_cast<float>(
                static_cast<double>(end - start) * 1000.0 /
                static_cast<double>(disjoint.Frequency));
        }
        gpuTimerPending_ = false;
    }
}

void D3D11Renderer::Impl::present() {
    checkHr(swapChain_->Present(1, 0), ErrorCode::swapChainInit, "Present");
}

uint32_t D3D11Renderer::Impl::cascadeResolution(const QualityState& quality) const {
    const uint32_t value = static_cast<uint32_t>(
        static_cast<float>(kDefaultCascadeSize) * quality.shadowScale);
    return std::max(256u, (value / 64u) * 64u);
}

uint32_t D3D11Renderer::Impl::localResolution(const QualityState& quality) const {
    const uint32_t value = static_cast<uint32_t>(
        static_cast<float>(kDefaultLocalSize) * quality.shadowScale);
    return std::max(256u, (value / 64u) * 64u);
}

LiveFrameStats D3D11Renderer::Impl::render(uint64_t frame, float deltaSeconds,
                                           const Camera& camera) {
    currentFrame_ = frame;
    if (device_->GetDeviceRemovedReason() != S_OK) {
        deviceRemoved_ = true;
        throw AppError(ErrorCode::deviceInit, "D3D11 device removed");
    }
    pollGpuTimer();
    quality_ = adaptive_.update(frame, lastGpuFrameMs_);
    const uint32_t renderWidth = std::max(1u, static_cast<uint32_t>(
        static_cast<float>(clientWidth_) * quality_.renderScale));
    const uint32_t renderHeight = std::max(1u, static_cast<uint32_t>(
        static_cast<float>(clientHeight_) * quality_.renderScale));
    ensureSceneTargets(renderWidth, renderHeight);
    const uint32_t cascadeSize = cascadeResolution(quality_);
    const uint32_t localSize = localResolution(quality_);
    createShadowResources(cascadeSize, localSize);

    simulationTime_ += std::max(0.0f, deltaSeconds);
    updateMovingObjects(scene_, simulationTime_);
    updateMovingLights(scene_, simulationTime_);
    for (const SceneObject& object : scene_.objects) {
        if (object.motion.enabled) bvh_.updateObject(object.id, object.bounds);
    }

    pollHardwareQueries();
    const CullResult cull = occlusion_.cull(scene_.objects, bvh_, camera, previousCamera_,
                                            CullOptions{});
    cameraFast_ = cull.cameraFast;
    const std::vector<uint32_t> occluded = chooseOccluded(cull);
    std::vector<uint32_t> visible;
    visible.reserve(cull.frustumVisible.size());
    size_t occludedIndex = 0;
    for (uint32_t id : cull.frustumVisible) {
        while (occludedIndex < occluded.size() && occluded[occludedIndex] < id) ++occludedIndex;
        if (occludedIndex < occluded.size() && occluded[occludedIndex] == id) continue;
        visible.push_back(id);
    }

    const Mat4 characterWorld = Mat4::identity();
    const float animationTime = std::fmod(simulationTime_, character_.duration);
    bonePalette_ = evaluatePose(character_, animationTime);
    uploadBones(bonePalette_);
    const AABB characterBounds = skinnedBounds(character_, bonePalette_, characterWorld);
    const bool characterVisible = camera.frustum().intersects(characterBounds);

    CascadeConfig cascadeConfig;
    cascadeConfig.shadowSize = cascadeSize;
    cascadeConfig.stabilizationSize = kDefaultCascadeSize;
    const CsmFrame csm = csmPolicy_.update(camera, scene_.lights[directionalLightIndex_],
                                           bvh_, cascadeConfig, true, firstFrame_);
    LocalShadowConfig localConfig;
    localConfig.faceResolution = static_cast<float>(localSize);
    const LocalShadowFrame local = localShadowPolicy_.update(
        camera, scene_.lights, visible, scene_.objects, characterBounds, characterVisible,
        true, firstFrame_, localConfig);
    const RecordingResult recording = recorder_.record(visible, scene_.objects,
                                                       RecordingConfig{});

    beginGpuTimer();
    renderCascades(csm);
    renderLocalShadows(local, characterWorld);
    renderMainScene(camera, csm, local, recording, characterWorld);
    lastVisible_ = visible;
    issueHardwareQueries(cull, camera);
    updateHardwareOcclusionSet();
    endGpuTimer();
    present();

    LiveFrameStats stats;
    stats.frame = frame;
    stats.frustumVisible = static_cast<uint32_t>(cull.frustumVisible.size());
    stats.occluded = static_cast<uint32_t>(occluded.size());
    stats.visible = static_cast<uint32_t>(visible.size());
    stats.csmUpdated = csm.updatedCascades;
    stats.localShadowUpdated = local.updatedCount;
    stats.hardwareQueries = hardwareQueriesThisFrame_;
    stats.hardwareResults = hardwareResultsThisFrame_;
    stats.hardwareOccluded = hardwareOccludedThisFrame_;
    stats.recordedCommands = static_cast<uint32_t>(recording.commands.size());
    stats.staticBatches = recording.mergedDrawCalls;
    stats.occlusionQueryDraws = hardwareQueriesThisFrame_;
    uint32_t shadowDraws = csm.updatedCascades * 2u;
    for (const LocalShadowDecision& light : local.lights) {
        if (!light.updated) continue;
        shadowDraws += light.type == LightType::point ? 12u : 2u;
    }
    stats.totalDrawCalls = recording.mergedDrawCalls + 1u + shadowDraws +
                           hardwareQueriesThisFrame_;
    stats.recordingMicros = recording.elapsedMicros;
    stats.recordingBudgetExceeded = recording.budgetExceeded;
    stats.cameraFast = cull.cameraFast;
    stats.gpuFrameMs = lastGpuFrameMs_;
    stats.qualityLevel = quality_.level;
    stats.renderScale = quality_.renderScale;
    stats.shadowScale = quality_.shadowScale;
    stats.renderWidth = sceneWidth_;
    stats.renderHeight = sceneHeight_;
    stats.shadowResolution = shadows_.cascadeSize;

    previousCamera_ = camera;
    firstFrame_ = false;
    return stats;
}

D3D11Renderer::D3D11Renderer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
D3D11Renderer::~D3D11Renderer() = default;

std::unique_ptr<D3D11Renderer> D3D11Renderer::create(HWND window, uint32_t width,
                                                     uint32_t height, SceneDesc scene,
                                                     CharacterAsset character) {
    auto impl = std::make_unique<Impl>(window, width, height, std::move(scene),
                                       std::move(character));
    return std::unique_ptr<D3D11Renderer>(new D3D11Renderer(std::move(impl)));
}

void D3D11Renderer::resize(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) return;
    if (width == impl_->clientWidth_ && height == impl_->clientHeight_) return;
    impl_->context_->OMSetRenderTargets(0, nullptr, nullptr);
    impl_->backRtv_.reset();
    impl_->backBuffer_.reset();
    checkHr(impl_->swapChain_->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0),
            ErrorCode::swapChainInit, "ResizeBuffers");
    impl_->clientWidth_ = width;
    impl_->clientHeight_ = height;
    impl_->recreateBackBuffer();
}

LiveFrameStats D3D11Renderer::renderFrame(uint64_t frame, float deltaSeconds,
                                          const Camera& camera) {
    return impl_->render(frame, deltaSeconds, camera);
}

bool D3D11Renderer::deviceRemoved() const {
    return impl_->deviceRemoved_;
}

} // namespace uid166

#endif

