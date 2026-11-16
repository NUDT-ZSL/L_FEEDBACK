#include "Renderer.h"

#include <cfloat>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>

using namespace DirectX;

namespace {

constexpr DXGI_FORMAT kColorFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
constexpr DXGI_FORMAT kMainDepthFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
constexpr DXGI_FORMAT kShadowDepthFormat = DXGI_FORMAT_D32_FLOAT;
constexpr float kLightIntensity = 1.38f;

std::string CompileShader(const std::string& source, const char* entryPoint,
                          const char* profile, const std::string& sourceName) {
    D3D_SHADER_MACRO macros[] = {
        {"CSM_COMPILER", "1"},
        {nullptr, nullptr},
    };
    ComPtr<ID3DBlob> byteCode;
    ComPtr<ID3DBlob> errors;
    const HRESULT result = D3DCompile(
        source.data(), source.size(), sourceName.c_str(), macros, nullptr, entryPoint,
        profile, D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
        &byteCode, &errors);
    if (FAILED(result)) {
        std::string message = "shader compilation failed";
        if (errors != nullptr) {
            const char* text = static_cast<const char*>(errors->GetBufferPointer());
            const size_t length = errors->GetBufferSize();
            if (text != nullptr && length != 0) {
                message += ":\n";
                message.append(text, length);
            }
        } else {
            message += ": " + HResultText(result);
        }
        throw std::runtime_error(message);
    }

    const char* data = static_cast<const char*>(byteCode->GetBufferPointer());
    const size_t size = byteCode->GetBufferSize();
    return std::string(data, size);
}

void CreateShaderAndLayout(ID3D11Device* device, const std::string& byteCode,
                           ComPtr<ID3D11VertexShader>& vertexShader,
                           ComPtr<ID3D11InputLayout>& inputLayout) {
    const D3D11_INPUT_ELEMENT_DESC elements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,
         static_cast<UINT>(offsetof(Vertex, position)), D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,
         static_cast<UINT>(offsetof(Vertex, normal)), D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,
         static_cast<UINT>(offsetof(Vertex, color)), D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0,
         static_cast<UINT>(offsetof(Vertex, animation)), D3D11_INPUT_PER_VERTEX_DATA, 0},
    };

    ThrowIfFailed(device->CreateVertexShader(byteCode.data(), byteCode.size(), nullptr,
                                             &vertexShader),
                  "CreateVertexShader");
    ThrowIfFailed(
        device->CreateInputLayout(elements, static_cast<UINT>(std::size(elements)),
                                  byteCode.data(), byteCode.size(), &inputLayout),
        "CreateInputLayout");
}

std::string LoadFileText(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open shader file: " + path.string());
    }
    std::string contents;
    contents.assign(std::istreambuf_iterator<char>(input),
                    std::istreambuf_iterator<char>());
    if (contents.empty()) {
        throw std::runtime_error("shader file is empty: " + path.string());
    }
    return contents;
}

}  // namespace

Renderer::Renderer(const SceneData& scene, const Options& options)
    : scene_(scene), options_(options) {}

Renderer::~Renderer() {
    ReleaseDeviceResources();
}

void Renderer::Initialize() {
    LoadShaderSource();
    CreateDevice();
    CreateDeviceResources();
}

void Renderer::LoadShaderSource() {
    const std::filesystem::path executableDirectory = ExecutableDirectory();
    const std::filesystem::path candidates[] = {
        executableDirectory / L"shaders" / L"scene.hlsl",
        std::filesystem::current_path() / L"shaders" / L"scene.hlsl",
        executableDirectory / L".." / L".." / L"shaders" / L"scene.hlsl",
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            shaderSource_ = LoadFileText(candidate);
            return;
        }
    }
    throw std::runtime_error(
        "could not find shaders/scene.hlsl next to the executable or in the "
        "current directory");
}

void Renderer::CreateDevice() {
    const D3D_FEATURE_LEVEL requestedLevels[] = {D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL acquiredLevel = D3D_FEATURE_LEVEL_11_0;
    const UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    HRESULT result = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, requestedLevels,
        static_cast<UINT>(std::size(requestedLevels)), D3D11_SDK_VERSION, &device_,
        &acquiredLevel, &context_);
    if (result == E_INVALIDARG) {
        throw std::runtime_error(
            "Direct3D 11 feature level 11_0 is not supported by the selected adapter");
    }
    ThrowIfFailed(result, "D3D11CreateDevice");
    if (acquiredLevel < D3D_FEATURE_LEVEL_11_0) {
        throw std::runtime_error("Direct3D feature level 11_0 is required");
    }
}

void Renderer::CreateDeviceResources() {
    // Rebuild in the dependency order required by the device-loss contract:
    // offscreen swapchain substitute -> render target -> depth buffer ->
    // shadow maps -> vertex/index buffers.
    CreateMainTargets();
    CreateShadowTargets();
    CreateSceneBuffers();
    CreatePipelineObjects();
    CalculateCascades();
}

void Renderer::CreateMainTargets() {
    D3D11_TEXTURE2D_DESC colorDescription{};
    colorDescription.Width = kWidth;
    colorDescription.Height = kHeight;
    colorDescription.MipLevels = 1;
    colorDescription.ArraySize = 1;
    colorDescription.Format = kColorFormat;
    colorDescription.SampleDesc.Count = 1;
    colorDescription.Usage = D3D11_USAGE_DEFAULT;
    colorDescription.BindFlags = D3D11_BIND_RENDER_TARGET;
    ThrowIfFailed(device_->CreateTexture2D(&colorDescription, nullptr, &colorTexture_),
                  "Create color texture");
    ThrowIfFailed(
        device_->CreateRenderTargetView(colorTexture_.Get(), nullptr,
                                        &colorRenderTargetView_),
        "Create color render target view");

    D3D11_TEXTURE2D_DESC depthDescription = colorDescription;
    depthDescription.Format = kMainDepthFormat;
    depthDescription.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    ThrowIfFailed(device_->CreateTexture2D(&depthDescription, nullptr, &depthTexture_),
                  "Create main depth texture");
    ThrowIfFailed(device_->CreateDepthStencilView(depthTexture_.Get(), nullptr,
                                                  &depthStencilView_),
                  "Create main depth stencil view");

    D3D11_TEXTURE2D_DESC readbackDescription = colorDescription;
    readbackDescription.BindFlags = 0;
    readbackDescription.Usage = D3D11_USAGE_STAGING;
    readbackDescription.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ThrowIfFailed(
        device_->CreateTexture2D(&readbackDescription, nullptr, &readbackTexture_),
        "Create readback texture");
}

void Renderer::CreateShadowTargets() {
    D3D11_TEXTURE2D_DESC description{};
    description.Width = kShadowMapSize;
    description.Height = kShadowMapSize;
    description.MipLevels = 1;
    description.ArraySize = kCascadeCount;
    description.Format = DXGI_FORMAT_R32_TYPELESS;
    description.SampleDesc.Count = 1;
    description.Usage = D3D11_USAGE_DEFAULT;
    description.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
    ThrowIfFailed(device_->CreateTexture2D(&description, nullptr, &shadowTexture_),
                  "Create shadow texture array");

    for (uint32_t cascade = 0; cascade < kCascadeCount; ++cascade) {
        D3D11_DEPTH_STENCIL_VIEW_DESC viewDescription{};
        viewDescription.Format = kShadowDepthFormat;
        viewDescription.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
        viewDescription.Texture2DArray.MipSlice = 0;
        viewDescription.Texture2DArray.FirstArraySlice = cascade;
        viewDescription.Texture2DArray.ArraySize = 1;
        ThrowIfFailed(device_->CreateDepthStencilView(
                          shadowTexture_.Get(), &viewDescription,
                          &shadowDepthViews_[cascade]),
                      "Create cascade depth stencil view");
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC resourceDescription{};
    resourceDescription.Format = DXGI_FORMAT_R32_FLOAT;
    resourceDescription.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
    resourceDescription.Texture2DArray.MostDetailedMip = 0;
    resourceDescription.Texture2DArray.MipLevels = 1;
    resourceDescription.Texture2DArray.FirstArraySlice = 0;
    resourceDescription.Texture2DArray.ArraySize = kCascadeCount;
    ThrowIfFailed(device_->CreateShaderResourceView(
                      shadowTexture_.Get(), &resourceDescription,
                      &shadowShaderResourceView_),
                  "Create shadow shader resource view");
}

void Renderer::CreateSceneBuffers() {
    D3D11_BUFFER_DESC description{};
    description.Usage = D3D11_USAGE_DEFAULT;
    description.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    description.ByteWidth = static_cast<UINT>(scene_.vertices.size() * sizeof(Vertex));
    D3D11_SUBRESOURCE_DATA initialData{};
    initialData.pSysMem = scene_.vertices.data();
    ThrowIfFailed(device_->CreateBuffer(&description, &initialData, &vertexBuffer_),
                  "Create vertex buffer");

    D3D11_BUFFER_DESC indexDescription{};
    indexDescription.Usage = D3D11_USAGE_DEFAULT;
    indexDescription.BindFlags = D3D11_BIND_INDEX_BUFFER;
    indexDescription.ByteWidth =
        static_cast<UINT>(scene_.indices.size() * sizeof(uint32_t));
    D3D11_SUBRESOURCE_DATA indexData{};
    indexData.pSysMem = scene_.indices.data();
    ThrowIfFailed(device_->CreateBuffer(&indexDescription, &indexData, &indexBuffer_),
                  "Create index buffer");

    D3D11_BUFFER_DESC constantDescription{};
    constantDescription.Usage = D3D11_USAGE_DYNAMIC;
    constantDescription.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    constantDescription.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    constantDescription.ByteWidth = sizeof(SceneConstantsCpu);
    ThrowIfFailed(
        device_->CreateBuffer(&constantDescription, nullptr, &sceneConstantBuffer_),
        "Create scene constant buffer");
}

void Renderer::CreatePipelineObjects() {
    const std::string sourceName = "scene.hlsl";
    const std::string mainVsByteCode =
        CompileShader(shaderSource_, "MainVS", "vs_5_0", sourceName);
    const std::string mainPsByteCode =
        CompileShader(shaderSource_, "MainPS", "ps_5_0", sourceName);
    const std::string shadowVsByteCode =
        CompileShader(shaderSource_, "ShadowVS", "vs_5_0", sourceName);
    const std::string shadowPsByteCode =
        CompileShader(shaderSource_, "ShadowPS", "ps_5_0", sourceName);

    CreateShaderAndLayout(device_.Get(), mainVsByteCode, mainVertexShader_,
                          inputLayout_);
    ThrowIfFailed(device_->CreatePixelShader(mainPsByteCode.data(),
                                             mainPsByteCode.size(), nullptr,
                                             &mainPixelShader_),
                  "Create main pixel shader");
    ThrowIfFailed(device_->CreateVertexShader(shadowVsByteCode.data(),
                                              shadowVsByteCode.size(), nullptr,
                                              &shadowVertexShader_),
                  "Create shadow vertex shader");
    ThrowIfFailed(device_->CreatePixelShader(shadowPsByteCode.data(),
                                             shadowPsByteCode.size(), nullptr,
                                             &shadowPixelShader_),
                  "Create shadow pixel shader");

    D3D11_RASTERIZER_DESC rasterizerDescription{};
    rasterizerDescription.FillMode = D3D11_FILL_SOLID;
    rasterizerDescription.CullMode = D3D11_CULL_NONE;
    rasterizerDescription.FrontCounterClockwise = false;
    rasterizerDescription.DepthClipEnable = true;
    ThrowIfFailed(
        device_->CreateRasterizerState(&rasterizerDescription, &rasterizerState_),
        "Create rasterizer state");

    D3D11_DEPTH_STENCIL_DESC depthDescription{};
    depthDescription.DepthEnable = true;
    depthDescription.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    depthDescription.DepthFunc = D3D11_COMPARISON_LESS;
    depthDescription.StencilEnable = false;
    ThrowIfFailed(device_->CreateDepthStencilState(&depthDescription,
                                                   &depthStencilState_),
                  "Create depth stencil state");

    D3D11_SAMPLER_DESC samplerDescription{};
    samplerDescription.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    samplerDescription.AddressU = D3D11_TEXTURE_ADDRESS_BORDER;
    samplerDescription.AddressV = D3D11_TEXTURE_ADDRESS_BORDER;
    samplerDescription.AddressW = D3D11_TEXTURE_ADDRESS_BORDER;
    samplerDescription.BorderColor[0] = 1.0f;
    samplerDescription.BorderColor[1] = 1.0f;
    samplerDescription.BorderColor[2] = 1.0f;
    samplerDescription.BorderColor[3] = 1.0f;
    samplerDescription.ComparisonFunc = D3D11_COMPARISON_NEVER;
    samplerDescription.MaxLOD = D3D11_FLOAT32_MAX;
    ThrowIfFailed(device_->CreateSamplerState(&samplerDescription, &shadowSampler_),
                  "Create shadow sampler");
}

void Renderer::CalculateCascades() {
    const XMVECTOR eye = XMLoadFloat3(&scene_.camera.position);
    const XMVECTOR target = XMLoadFloat3(&scene_.camera.target);
    const XMVECTOR up = XMLoadFloat3(&scene_.camera.up);
    const XMMATRIX view = XMMatrixLookAtLH(eye, target, up);
    const XMMATRIX projection = XMMatrixPerspectiveFovLH(
        scene_.camera.verticalFovRadians, scene_.camera.aspectRatio,
        scene_.camera.nearPlane, scene_.camera.farPlane);
    const XMMATRIX viewProjection = XMMatrixMultiply(view, projection);
    XMStoreFloat4x4(&mainView_, view);
    XMStoreFloat4x4(&mainViewProjection_, viewProjection);

    const float nearPlane = scene_.camera.nearPlane;
    const float farPlane = scene_.camera.farPlane;
    splitDistances_[0] = nearPlane;
    splitDistances_[1] = nearPlane + options_.cascadeSplits[0] * (farPlane - nearPlane);
    splitDistances_[2] = nearPlane + options_.cascadeSplits[1] * (farPlane - nearPlane);
    splitDistances_[3] = farPlane;

    const XMMATRIX inverseView = XMMatrixInverse(nullptr, view);
    const float tanHalfFov = std::tan(scene_.camera.verticalFovRadians * 0.5f);
    const XMVECTOR lightDirection = XMVector3Normalize(
        XMLoadFloat3(&scene_.lightDirection));
    const XMVECTOR fallbackUp = XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f);
    const XMVECTOR lightUp =
        std::abs(XMVectorGetY(lightDirection)) > 0.94f ? fallbackUp : up;

    for (uint32_t cascade = 0; cascade < kCascadeCount; ++cascade) {
        const float cascadeNear = static_cast<float>(splitDistances_[cascade]);
        const float cascadeFar = static_cast<float>(splitDistances_[cascade + 1]);
        const float halfHeightNear = tanHalfFov * cascadeNear;
        const float halfWidthNear = halfHeightNear * scene_.camera.aspectRatio;
        const float halfHeightFar = tanHalfFov * cascadeFar;
        const float halfWidthFar = halfHeightFar * scene_.camera.aspectRatio;

        std::array<XMVECTOR, 8> corners = {
            XMVector3TransformCoord(
                XMVectorSet(-halfWidthNear, -halfHeightNear, cascadeNear, 1.0f),
                inverseView),
            XMVector3TransformCoord(
                XMVectorSet(halfWidthNear, -halfHeightNear, cascadeNear, 1.0f),
                inverseView),
            XMVector3TransformCoord(
                XMVectorSet(-halfWidthNear, halfHeightNear, cascadeNear, 1.0f),
                inverseView),
            XMVector3TransformCoord(
                XMVectorSet(halfWidthNear, halfHeightNear, cascadeNear, 1.0f),
                inverseView),
            XMVector3TransformCoord(
                XMVectorSet(-halfWidthFar, -halfHeightFar, cascadeFar, 1.0f),
                inverseView),
            XMVector3TransformCoord(
                XMVectorSet(halfWidthFar, -halfHeightFar, cascadeFar, 1.0f),
                inverseView),
            XMVector3TransformCoord(
                XMVectorSet(-halfWidthFar, halfHeightFar, cascadeFar, 1.0f),
                inverseView),
            XMVector3TransformCoord(
                XMVectorSet(halfWidthFar, halfHeightFar, cascadeFar, 1.0f),
                inverseView),
        };

        XMVECTOR center = XMVectorZero();
        for (const XMVECTOR corner : corners) {
            center = XMVectorAdd(center, corner);
        }
        center = XMVectorScale(center, 1.0f / 8.0f);

        const XMVECTOR lightEye =
            XMVectorSubtract(center, XMVectorScale(lightDirection, 95.0f));
        const XMMATRIX lightView = XMMatrixLookAtLH(lightEye, center, lightUp);

        XMFLOAT3 transformed{};
        float minimumX = FLT_MAX;
        float maximumX = -FLT_MAX;
        float minimumY = FLT_MAX;
        float maximumY = -FLT_MAX;
        float minimumZ = FLT_MAX;
        float maximumZ = -FLT_MAX;
        for (const XMVECTOR corner : corners) {
            XMStoreFloat3(&transformed, XMVector3TransformCoord(corner, lightView));
            minimumX = std::min(minimumX, transformed.x);
            maximumX = std::max(maximumX, transformed.x);
            minimumY = std::min(minimumY, transformed.y);
            maximumY = std::max(maximumY, transformed.y);
            minimumZ = std::min(minimumZ, transformed.z);
            maximumZ = std::max(maximumZ, transformed.z);
        }
        const float horizontalPadding =
            2.5f + (cascadeFar - cascadeNear) * 0.055f;
        const float lightNear = std::max(0.01f, minimumZ - 70.0f);
        const float lightFar = maximumZ + 12.0f;
        const XMMATRIX lightProjection = XMMatrixOrthographicOffCenterLH(
            minimumX - horizontalPadding, maximumX + horizontalPadding,
            minimumY - horizontalPadding, maximumY + horizontalPadding,
            lightNear, lightFar);
        const XMMATRIX lightViewProjection = XMMatrixMultiply(lightView, lightProjection);

        cascades_[cascade].ratio = options_.cascadeSplits[cascade];
        cascades_[cascade].nearDistance = cascadeNear;
        cascades_[cascade].farDistance = cascadeFar;
        XMStoreFloat4x4(&cascades_[cascade].lightViewProjection,
                        lightViewProjection);
    }
}

void Renderer::UpdateConstants(uint32_t activeCascade, double animationTime) {
    SceneConstantsCpu constants;
    constants.mainViewProjection = mainViewProjection_;
    constants.mainView = mainView_;
    for (uint32_t cascade = 0; cascade < kCascadeCount; ++cascade) {
        constants.lightViewProjection[cascade] =
            cascades_[cascade].lightViewProjection;
    }
    constants.cameraPositionTime = {
        scene_.camera.position.x, scene_.camera.position.y,
        scene_.camera.position.z, static_cast<float>(animationTime)};
    constants.lightDirectionIntensity = {
        scene_.lightDirection.x, scene_.lightDirection.y,
        scene_.lightDirection.z, kLightIntensity};
    constants.cascadeDepths = {
        static_cast<float>(splitDistances_[0]),
        static_cast<float>(splitDistances_[1]),
        static_cast<float>(splitDistances_[2]),
        static_cast<float>(splitDistances_[3])};
    constants.shadowBiasAndMapSize = {
        static_cast<float>(options_.slopeOffset),
        static_cast<float>(options_.constOffset),
        static_cast<float>(kShadowMapSize), 0.0f};

    const double firstWidth = splitDistances_[1] - splitDistances_[0];
    const double secondWidth = splitDistances_[2] - splitDistances_[1];
    constants.cascadeTransitions = {
        static_cast<float>(std::max(0.3, std::min(3.0, firstWidth * 0.18))),
        static_cast<float>(std::max(0.3, std::min(3.0, secondWidth * 0.18))),
        0.0f, 0.0f};
    constants.activeCascade = activeCascade;

    D3D11_MAPPED_SUBRESOURCE mapped{};
    ThrowIfFailed(context_->Map(sceneConstantBuffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD,
                                0, &mapped),
                  "Map scene constant buffer");
    std::memcpy(mapped.pData, &constants, sizeof(constants));
    context_->Unmap(sceneConstantBuffer_.Get(), 0);
}

void Renderer::RenderFrame(double animationTime) {
    if (device_ == nullptr || context_ == nullptr) {
        throw std::runtime_error("renderer is not initialized");
    }

    const uint32_t zero = 0;
    const UINT stride = sizeof(Vertex);
    context_->IASetVertexBuffers(0, 1, vertexBuffer_.GetAddressOf(), &stride, &zero);
    context_->IASetIndexBuffer(indexBuffer_.Get(), DXGI_FORMAT_R32_UINT, 0);
    context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context_->IASetInputLayout(inputLayout_.Get());
    context_->RSSetState(rasterizerState_.Get());
    context_->OMSetDepthStencilState(depthStencilState_.Get(), 0);
    context_->VSSetConstantBuffers(0, 1, sceneConstantBuffer_.GetAddressOf());
    context_->PSSetConstantBuffers(0, 1, sceneConstantBuffer_.GetAddressOf());

    context_->VSSetShader(shadowVertexShader_.Get(), nullptr, 0);
    context_->PSSetShader(shadowPixelShader_.Get(), nullptr, 0);
    ID3D11ShaderResourceView* nullResourceViews[1] = {nullptr};
    ID3D11SamplerState* nullSamplers[1] = {nullptr};
    context_->PSSetShaderResources(0, 1, nullResourceViews);
    context_->PSSetSamplers(0, 1, nullSamplers);

    D3D11_VIEWPORT shadowViewport{};
    shadowViewport.Width = static_cast<float>(kShadowMapSize);
    shadowViewport.Height = static_cast<float>(kShadowMapSize);
    shadowViewport.MinDepth = 0.0f;
    shadowViewport.MaxDepth = 1.0f;
    context_->RSSetViewports(1, &shadowViewport);

    const float clearDepth = 1.0f;
    for (uint32_t cascade = 0; cascade < kCascadeCount; ++cascade) {
        context_->OMSetRenderTargets(0, nullptr, shadowDepthViews_[cascade].Get());
        context_->ClearDepthStencilView(shadowDepthViews_[cascade].Get(),
                                        D3D11_CLEAR_DEPTH, clearDepth, 0);
        UpdateConstants(cascade, animationTime);
        context_->DrawIndexed(static_cast<UINT>(scene_.indices.size()), 0, 0);
        CheckDeviceLost();
    }
    context_->OMSetRenderTargets(0, nullptr, nullptr);

    D3D11_VIEWPORT mainViewport{};
    mainViewport.Width = static_cast<float>(kWidth);
    mainViewport.Height = static_cast<float>(kHeight);
    mainViewport.MinDepth = 0.0f;
    mainViewport.MaxDepth = 1.0f;
    context_->RSSetViewports(1, &mainViewport);

    ID3D11RenderTargetView* colorTargets[] = {colorRenderTargetView_.Get()};
    context_->OMSetRenderTargets(1, colorTargets, depthStencilView_.Get());
    const float clearColor[4] = {0.025f, 0.030f, 0.041f, 1.0f};
    context_->ClearRenderTargetView(colorRenderTargetView_.Get(), clearColor);
    context_->ClearDepthStencilView(depthStencilView_.Get(), D3D11_CLEAR_DEPTH,
                                     clearDepth, 0);

    context_->VSSetShader(mainVertexShader_.Get(), nullptr, 0);
    context_->PSSetShader(mainPixelShader_.Get(), nullptr, 0);
    context_->PSSetShaderResources(0, 1, shadowShaderResourceView_.GetAddressOf());
    context_->PSSetSamplers(0, 1, shadowSampler_.GetAddressOf());
    UpdateConstants(0, animationTime);
    context_->DrawIndexed(static_cast<UINT>(scene_.indices.size()), 0, 0);
    CheckDeviceLost();
}

void Renderer::CheckDeviceLost() const {
    if (device_ == nullptr) {
        throw DeviceLostError(DXGI_ERROR_DEVICE_REMOVED);
    }
    const HRESULT reason = device_->GetDeviceRemovedReason();
    if (FAILED(reason)) {
        throw DeviceLostError(reason);
    }
}

std::vector<uint8_t> Renderer::ReadColorBuffer() {
    CheckDeviceLost();
    context_->CopyResource(readbackTexture_.Get(), colorTexture_.Get());
    CheckDeviceLost();

    D3D11_MAPPED_SUBRESOURCE mapped{};
    ThrowIfFailed(context_->Map(readbackTexture_.Get(), 0, D3D11_MAP_READ, 0, &mapped),
                  "Map readback texture");

    std::vector<uint8_t> pixels(static_cast<size_t>(kWidth) * kHeight * 4);
    for (uint32_t row = 0; row < kHeight; ++row) {
        const uint8_t* source =
            static_cast<const uint8_t*>(mapped.pData) + row * mapped.RowPitch;
        std::memcpy(pixels.data() + static_cast<size_t>(row) * kWidth * 4, source,
                    static_cast<size_t>(kWidth) * 4);
    }
    context_->Unmap(readbackTexture_.Get(), 0);
    CheckDeviceLost();
    return pixels;
}

void Renderer::RecoverFromDeviceLoss() {
    constexpr uint32_t kRecoveryAttempts = 5;
    for (uint32_t attempt = 0; attempt < kRecoveryAttempts; ++attempt) {
        ReleaseDeviceResources();
        try {
            CreateDevice();
            CreateDeviceResources();
            ++deviceLossRecoveryCount_;
            return;
        } catch (const DeviceLostError&) {
            if (attempt + 1 == kRecoveryAttempts) {
                throw;
            }
            Sleep(100u * (attempt + 1));
        }
    }
}

void Renderer::SimulateDeviceLossAndRecover() {
    if (context_ != nullptr) {
        context_->ClearState();
        context_->Flush();
    }
    RecoverFromDeviceLoss();
}

void Renderer::ReleaseDeviceResources() {
    if (context_ != nullptr) {
        context_->ClearState();
        context_->Flush();
    }

    // Release in reverse dependency order so no child view outlives its parent.
    shadowSampler_.Reset();
    depthStencilState_.Reset();
    rasterizerState_.Reset();
    shadowShaderResourceView_.Reset();
    for (auto& view : shadowDepthViews_) {
        view.Reset();
    }
    shadowTexture_.Reset();
    sceneConstantBuffer_.Reset();
    indexBuffer_.Reset();
    vertexBuffer_.Reset();
    inputLayout_.Reset();
    shadowPixelShader_.Reset();
    shadowVertexShader_.Reset();
    mainPixelShader_.Reset();
    mainVertexShader_.Reset();
    readbackTexture_.Reset();
    depthStencilView_.Reset();
    depthTexture_.Reset();
    colorRenderTargetView_.Reset();
    colorTexture_.Reset();
    context_.Reset();
    device_.Reset();
}

