#pragma once

#include "lightcluster/deterministic_engine.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace lightcluster {

struct DebugFrameStatus {
    bool checked = false;
    bool ok = true;
    std::uint32_t readbackLights = 0;
    std::uint32_t readbackPoint = 0;
    std::uint32_t readbackDirectional = 0;
};

[[nodiscard]] std::string formatEventJson(const RunEvent& event);
[[nodiscard]] std::string formatFrameJson(
    const FrameTick& tick,
    const RunConfig& config,
    const DebugFrameStatus& debug);
[[nodiscard]] std::string formatBufferInitJson(
    const SyncOutcome& sync,
    const SceneSnapshot& snapshot,
    const RunConfig& config);
[[nodiscard]] std::string formatDebugErrorJson(
    std::uint32_t frame,
    const BufferValidation& validation);
[[nodiscard]] std::string formatFatalJson(
    std::string_view businessCode,
    std::string_view message);

[[nodiscard]] std::string_view toString(RunEventKind kind) noexcept;

}  // namespace lightcluster
