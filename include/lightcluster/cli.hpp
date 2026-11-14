#pragma once

#include "lightcluster/deterministic_engine.hpp"

#include <string>
#include <string_view>

namespace lightcluster {

enum class CliParseResult {
    Ok,
    Help,
    Error,
};

struct CliOptions {
    RunConfig run{};
};

[[nodiscard]] CliParseResult parseCommandLine(
    int argc,
    char** argv,
    CliOptions& options,
    std::string& error);

[[nodiscard]] std::string_view usageText() noexcept;

}  // namespace lightcluster
