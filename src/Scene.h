#pragma once

#include "Common.h"

struct Camera {
    DirectX::XMFLOAT3 position = {0.0f, 17.5f, 29.0f};
    DirectX::XMFLOAT3 target = {0.0f, 3.2f, -3.5f};
    DirectX::XMFLOAT3 up = {0.0f, 1.0f, 0.0f};
    float verticalFovRadians = DirectX::XMConvertToRadians(52.0f);
    float nearPlane = 0.5f;
    float farPlane = 140.0f;
    float aspectRatio = 1280.0f / 720.0f;
};

struct Vertex {
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT3 normal;
    DirectX::XMFLOAT3 color;
    DirectX::XMFLOAT4 animation;
};

struct SceneData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    Camera camera;
    DirectX::XMFLOAT3 lightDirection = {-0.48f, -0.78f, -0.40f};
};

SceneData GenerateDeterministicScene();

