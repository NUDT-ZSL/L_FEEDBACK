#pragma once

#include "asset.hpp"
#include "error.hpp"
#include "offline_renderer.hpp"
#include "scene.hpp"

#include <cstdint>
#include <string>

namespace skel {

struct PreviewResult {
    bool windowOpened = false;
    bool completed = false;
    std::string diagnostic;
    Error error;
};

PreviewResult runPreview(const Asset& asset, const RenderSettings& settings,
                         std::uint32_t initialFrame, bool forceOffline);

} // namespace skel

