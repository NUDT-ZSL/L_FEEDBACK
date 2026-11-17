#pragma once

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <DirectXMath.h>
#endif

namespace raster::platform {

inline bool verifyDirectXMathHeader() {
#if defined(_WIN32)
    const DirectX::XMFLOAT3 input{1.0f, 2.0f, 3.0f};
    const DirectX::XMVECTOR loaded = DirectX::XMLoadFloat3(&input);
    return DirectX::XMVectorGetX(loaded) == 1.0f &&
           DirectX::XMVectorGetY(loaded) == 2.0f &&
           DirectX::XMVectorGetZ(loaded) == 3.0f;
#else
    return true;
#endif
}

}  // namespace raster::platform

