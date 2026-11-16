#include "cli.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace orv {

namespace {

std::string requiredValue(int argc, char** argv, int& index, const std::string& argument) {
  if (index + 1 >= argc) {
    throw cliError("CLI_ERR_MISSING_ARGUMENT", "missing value for " + argument);
  }
  ++index;
  return argv[index];
}

bool parseInteger(const std::string& text, int& value) {
  if (text.empty()) {
    return false;
  }
  int parsed = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
    return false;
  }
  value = parsed;
  return true;
}

bool parseFloat(const std::string& text, float& value) {
  if (text.empty()) {
    return false;
  }
  char* end = nullptr;
  const float parsed = std::strtof(text.c_str(), &end);
  if (end == text.c_str() || *end != '\0' || !std::isfinite(parsed)) {
    return false;
  }
  value = parsed;
  return true;
}

}  // namespace

std::string renderPassName(RenderPass pass) {
  switch (pass) {
    case RenderPass::Geometry:
      return "geometry";
    case RenderPass::Lighting:
      return "lighting";
    case RenderPass::Full:
      return "full";
  }
  return "unknown";
}

std::string usageText() {
  return
      "offline_render_verify - deterministic headless OpenGL deferred renderer\n"
      "\n"
      "Usage:\n"
      "  offline_render_verify --scene <scene.json> [options]\n"
      "\n"
      "Options:\n"
      "  --scene <path>       Scene JSON input (required)\n"
      "  --output <prefix>    Output prefix (default: render_output)\n"
      "  --pass <name>        geometry | lighting | full (default: full)\n"
      "  --width <pixels>     Output width, 1..8192 (default: 640)\n"
      "  --height <pixels>    Output height, 1..8192 (default: 360)\n"
      "  --shadow-size <n>    Power of two in [512, 4096] (default: 1024)\n"
      "  --shadow-bias <f>    Shadow comparison bias in [0,1] (default: 0.002)\n"
      "  --help               Show this text\n";
}

bool validateShadowSize(int size) {
  return size >= 512 && size <= 4096 && (size & (size - 1)) == 0;
}

bool parseCli(int argc, char** argv, CliOptions& options, AppError& error) {
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--help" || argument == "-h") {
        options.showHelp = true;
        continue;
      }
      if (argument == "--scene") {
        options.scenePath = requiredValue(argc, argv, i, argument);
        continue;
      }
      if (argument == "--output") {
        options.outputPrefix = requiredValue(argc, argv, i, argument);
        continue;
      }
      if (argument == "--pass") {
        const std::string value = requiredValue(argc, argv, i, argument);
        if (value == "geometry") {
          options.pass = RenderPass::Geometry;
        } else if (value == "lighting") {
          options.pass = RenderPass::Lighting;
        } else if (value == "full") {
          options.pass = RenderPass::Full;
        } else {
          throw cliError("CLI_ERR_INVALID_PASS",
                         "invalid --pass value '" + value +
                             "'; expected geometry, lighting, or full");
        }
        continue;
      }
      if (argument == "--width" || argument == "--height") {
        const std::string value = requiredValue(argc, argv, i, argument);
        int parsed = 0;
        if (!parseInteger(value, parsed) || parsed < 1 || parsed > 8192) {
          throw cliError("CLI_ERR_INVALID_ARGUMENT",
                         argument + " must be an integer in [1, 8192]");
        }
        if (argument == "--width") {
          options.width = parsed;
        } else {
          options.height = parsed;
        }
        continue;
      }
      if (argument == "--shadow-size") {
        const std::string value = requiredValue(argc, argv, i, argument);
        int parsed = 0;
        if (!parseInteger(value, parsed) || !validateShadowSize(parsed)) {
          throw renderError("RENDER_ERR_INVALID_SHADOW_SIZE", 4,
                            "--shadow-size must be a power of two in [512, 4096]");
        }
        options.shadowSize = parsed;
        continue;
      }
      if (argument == "--shadow-bias") {
        const std::string value = requiredValue(argc, argv, i, argument);
        float parsed = 0.0f;
        if (!parseFloat(value, parsed) || parsed < 0.0f || parsed > 1.0f) {
          throw cliError("CLI_ERR_INVALID_ARGUMENT",
                         "--shadow-bias must be a finite number in [0, 1]");
        }
        options.shadowBias = parsed;
        continue;
      }
      throw cliError("CLI_ERR_UNKNOWN_ARGUMENT", "unknown argument: " + argument);
    }

    if (options.showHelp) {
      return true;
    }
    if (options.scenePath.empty()) {
      throw cliError("CLI_ERR_MISSING_ARGUMENT", "--scene is required");
    }
    if (options.outputPrefix.empty()) {
      options.outputPrefix = "render_output";
    }
    return true;
  } catch (const AppError& caught) {
    error = caught;
    return false;
  }
}

}  // namespace orv
