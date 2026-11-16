#include "WicPng.h"

#include <shlwapi.h>

WicPngWriter::WicPngWriter() {
    ThrowIfFailed(
        CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                         IID_PPV_ARGS(&factory_)),
        "Create WIC imaging factory");
}

void WicPngWriter::Write(const std::filesystem::path& path, uint32_t width,
                         uint32_t height,
                         const std::vector<uint8_t>& bgraPixels) {
    const size_t requiredBytes = static_cast<size_t>(width) * height * 4;
    if (bgraPixels.size() != requiredBytes) {
        throw std::invalid_argument("PNG pixel buffer has an unexpected size");
    }

    try {
        ComPtr<IStream> fileStream;
        ThrowIfFailed(
            SHCreateStreamOnFileEx(
                path.c_str(),
                STGM_CREATE | STGM_READWRITE | STGM_SHARE_DENY_NONE,
                FILE_ATTRIBUTE_NORMAL, TRUE, nullptr, &fileStream),
            "Create PNG file stream");

        ComPtr<IWICStream> stream;
        ThrowIfFailed(factory_->CreateStream(&stream), "Create WIC stream");
        ThrowIfFailed(stream->InitializeFromIStream(fileStream.Get()),
                      "Initialize WIC stream");

        ComPtr<IWICBitmapEncoder> encoder;
        ThrowIfFailed(
            factory_->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder),
            "Create PNG encoder");
        ThrowIfFailed(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache),
                      "Initialize PNG encoder");

        ComPtr<IWICBitmapFrameEncode> frame;
        ComPtr<IPropertyBag2> properties;
        ThrowIfFailed(encoder->CreateNewFrame(&frame, &properties),
                      "Create PNG frame");
        ThrowIfFailed(frame->Initialize(properties.Get()), "Initialize PNG frame");
        ThrowIfFailed(frame->SetSize(width, height), "Set PNG size");
        ThrowIfFailed(frame->SetResolution(96.0, 96.0), "Set PNG resolution");

        WICPixelFormatGUID pixelFormat = GUID_WICPixelFormat32bppBGRA;
        ThrowIfFailed(frame->SetPixelFormat(&pixelFormat), "Set PNG pixel format");
        if (!IsEqualGUID(pixelFormat, GUID_WICPixelFormat32bppBGRA)) {
            throw std::runtime_error("WIC PNG encoder rejected 32bpp BGRA input");
        }

        const UINT rowStride = width * 4;
        ThrowIfFailed(
            frame->WritePixels(height, rowStride,
                               static_cast<UINT>(bgraPixels.size()),
                               const_cast<BYTE*>(bgraPixels.data())),
            "Write PNG pixels");
        ThrowIfFailed(frame->Commit(), "Commit PNG frame");
        ThrowIfFailed(encoder->Commit(), "Commit PNG encoder");
        stream.Reset();
        fileStream.Reset();
    } catch (...) {
        throw;
    }
}

