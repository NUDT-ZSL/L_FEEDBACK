#pragma once

#include "skinbake/state_machine.hpp"

#include <string>

namespace skinbake {

struct Scenario {
    TimelineDefinition definition;
    std::uint32_t defaultFrames = 180;
    std::uint32_t fps = 60;
    std::string description;
};

Scenario make_scenario(const std::string& name, bool variant = false);

const char* const* error_scenario_names(std::size_t& count);

}

