#pragma once

#include "Options.h"
#include "Scene.h"

struct CascadeData {
    double ratio = 0.0;
    float nearDistance = 0.0f;
    float farDistance = 0.0f;
    DirectX::XMFLOAT4X4 lightViewProjection{};
};

class Renderer {
public:
    static constexpr uint32_t kWidth = 1280;
    static constexpr uint32_t kHeight = 720;
    static constexpr uint32_t kCascadeCount = 3;
    static constexpr uint32_t kShadowMapSize = 1024;
    static constexpr uint32_t kDrawCallsPerFrame = kCascadeCount + 1;

    Renderer(const SceneData& scene, const Options& options);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void Initialize();
    void RenderFrame(double animationTime);
    std::vector<uint8_t> ReadColorBuffer();

    // Simulates a device-loss event at the application boundary by destroying
    // and rebuilding every device-dependent object in real dependency order.
    // The exact same recovery routine handles actual DXGI errors.
    void SimulateDeviceLossAndRecover();
    void RecoverFromDeviceLoss();

    uint64_t DeviceLossRecoveryCount() const { return deviceLossRecoveryCount_; }
    const std::array<CascadeData, kCascadeCount>& Cascades() const {
        return cascades_;
    }
    const std::array<double, 4>& SplitDistances() const { return splitDistances_; }
    const std::array<double, 3>& CascadeSplitRatios() const {
        return options_.cascadeSplits;
    }
    double SlopeOffset() const { return options_.slopeOffset; }
    double ConstOffset() const { return options_.constOffset; }

private:
    struct SceneConstantsCpu {
        DirectX::XMFLOAT4X4 mainViewProjection{};
        DirectX::XMFLOAT4X4 mainView{};
        DirectX::XMFLOAT4X4 lightViewProjection[kCascadeCount]{};
        DirectX::XMFLOAT4 cameraPositionTime{};
        DirectX::XMFLOAT4 lightDirectionIntensity{};
        DirectX::XMFLOAT4 cascadeDepths{};
        DirectX::XMFLOAT4 shadowBiasAndMapSize{};
        DirectX::XMFLOAT4 cascadeTransitions{};
        UINT activeCascade = 0;
        DirectX::XMFLOAT3 padding{};
    };
    static_assert(sizeof(SceneConstantsCpu) == 416,
                  "SceneConstantsCpu must match the HLSL cbuffer layout");

    void CreateDevice();
    void CreateDeviceResources();
    void CreatePipelineObjects();
    void CreateSceneBuffers();
    void CreateMainTargets();
    void CreateShadowTargets();
    void ReleaseDeviceResources();
    void CalculateCascades();
    void LoadShaderSource();
    void UpdateConstants(uint32_t activeCascade, double animationTime);
    void CheckDeviceLost() const;

    ComPtr<ID3D11VertexShader> mainVertexShader_;
    ComPtr<ID3D11PixelShader> mainPixelShader_;
    ComPtr<ID3D11VertexShader> shadowVertexShader_;
    ComPtr<ID3D11PixelShader> shadowPixelShader_;
    ComPtr<ID3D11InputLayout> inputLayout_;
    ComPtr<ID3D11Buffer> vertexBuffer_;
    ComPtr<ID3D11Buffer> indexBuffer_;
    ComPtr<ID3D11Buffer> sceneConstantBuffer_;
    ComPtr<ID3D11Texture2D> colorTexture_;
    ComPtr<ID3D11RenderTargetView> colorRenderTargetView_;
    ComPtr<ID3D11Texture2D> depthTexture_;
    ComPtr<ID3D11DepthStencilView> depthStencilView_;
    ComPtr<ID3D11Texture2D> readbackTexture_;
    ComPtr<ID3D11Texture2D> shadowTexture_;
    std::array<ComPtr<ID3D11DepthStencilView>, kCascadeCount> shadowDepthViews_;
    ComPtr<ID3D11ShaderResourceView> shadowShaderResourceView_;
    ComPtr<ID3D11RasterizerState> rasterizerState_;
    ComPtr<ID3D11DepthStencilState> depthStencilState_;
    ComPtr<ID3D11SamplerState> shadowSampler_;

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;

    const SceneData& scene_;
    const Options& options_;
    std::array<CascadeData, kCascadeCount> cascades_{};
    std::array<double, 4> splitDistances_{};
    DirectX::XMFLOAT4X4 mainView_{};
    DirectX::XMFLOAT4X4 mainViewProjection_{};
    std::string shaderSource_;
    uint64_t deviceLossRecoveryCount_ = 0;
};

