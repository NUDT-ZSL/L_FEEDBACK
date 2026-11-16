#pragma once

#include "types.hpp"

namespace mv {

CliResult parseCommandLine(int argc, char** argv);
std::string helpText();

}  // namespace mv
