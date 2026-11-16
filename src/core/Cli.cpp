#include "core/Cli.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "core/Error.hpp"

namespace lcr {
namespace {

std::uint64_t parseU64(const std::string& text, const char* flag) {
  try {
    std::size_t used = 0;
    const std::uint64_t value = std::stoull(text, &used, 10);
    if (used != text.size()) {
      throw std::invalid_argument("trailing");
    }
    return value;
  } catch (...) {
    fail(
        ErrorCode::InvalidArgument,
        "cli.parse",
        std::string("invalid unsigned integer for ") + flag + ": " + text);
  }
}

std::uint32_t parseU32(const std::string& text, const char* flag) {
  const std::uint64_t value = parseU64(text, flag);
  if (value > 0xFFFFFFFFULL) {
    fail(
        ErrorCode::InvalidArgument,
        "cli.parse",
        std::string("value is too large for ") + flag + ": " + text);
  }
  return static_cast<std::uint32_t>(value);
}

int parseInt(const std::string& text, const char* flag) {
  try {
    std::size_t used = 0;
    const long value = std::stol(text, &used, 10);
    if (used != text.size() || value < -2147483647L || value > 2147483647L) {
      throw std::invalid_argument("bad range");
    }
    return static_cast<int>(value);
  } catch (...) {
    fail(
        ErrorCode::InvalidArgument,
        "cli.parse",
        std::string("invalid integer for ") + flag + ": " + text);
  }
}

double parseDouble(const std::string& text, const char* flag) {
  try {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value)) {
      throw std::invalid_argument("bad double");
    }
    return value;
  } catch (...) {
    fail(
        ErrorCode::InvalidArgument,
        "cli.parse",
        std::string("invalid number for ") + flag + ": " + text);
  }
}

std::vector<double> parseSwitchTimes(const std::string& text) {
  std::vector<double> result;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item.empty()) {
      fail(ErrorCode::InvalidArgument, "cli.parse", "empty switch time");
    }
    const double value = parseDouble(item, "--switch-time");
    if (value < 0.0) {
      fail(ErrorCode::InvalidArgument, "cli.parse", "switch time cannot be negative");
    }
    result.push_back(value);
  }
  return result;
}

std::string requireValue(const int argc, char** argv, int& index, const char* flag) {
  if (index + 1 >= argc) {
    fail(
        ErrorCode::InvalidArgument,
        "cli.parse",
        std::string("missing value for ") + flag);
  }
  ++index;
  return argv[index];
}

}  // namespace

CliOptions parseCommandLine(const int argc, char** argv) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--headless") {
      options.headless = true;
    } else if (arg == "--help" || arg == "-h") {
      options.help = true;
    } else if (arg == "--seed") {
      options.seed = parseU64(requireValue(argc, argv, i, "--seed"), "--seed");
    } else if (arg == "--logic-frames") {
      options.logicFrames =
          parseU32(requireValue(argc, argv, i, "--logic-frames"), "--logic-frames");
      options.logicFramesSpecified = true;
    } else if (arg == "--max-render-frames") {
      options.maxRenderFrames =
          parseU32(requireValue(argc, argv, i, "--max-render-frames"), "--max-render-frames");
    } else if (arg == "--render-fps") {
      options.renderFps =
          parseDouble(requireValue(argc, argv, i, "--render-fps"), "--render-fps");
    } else if (arg == "--texture-size") {
      options.textureSize =
          parseU32(requireValue(argc, argv, i, "--texture-size"), "--texture-size");
    } else if (arg == "--equipment") {
      options.equipment = requireValue(argc, argv, i, "--equipment");
    } else if (arg == "--switch-time") {
      const std::vector<double> parsed =
          parseSwitchTimes(requireValue(argc, argv, i, "--switch-time"));
      options.switchTimes.insert(
          options.switchTimes.end(), parsed.begin(), parsed.end());
    } else if (arg == "--report") {
      options.reportPath = requireValue(argc, argv, i, "--report");
    } else if (arg == "--snapshot") {
      options.snapshotPath = requireValue(argc, argv, i, "--snapshot");
    } else if (arg == "--fault") {
      options.faultName = requireValue(argc, argv, i, "--fault");
    } else if (arg == "--width") {
      options.windowWidth = parseInt(requireValue(argc, argv, i, "--width"), "--width");
    } else if (arg == "--height") {
      options.windowHeight = parseInt(requireValue(argc, argv, i, "--height"), "--height");
    } else {
      fail(
          ErrorCode::InvalidArgument,
          "cli.parse",
          "unknown argument: " + arg);
    }
  }

  if (options.logicFrames == 0U) {
    fail(ErrorCode::InvalidArgument, "cli.parse", "--logic-frames must be greater than zero");
  }
  if (options.renderFps <= 0.0 || options.renderFps > 3000.0) {
    fail(ErrorCode::InvalidArgument, "cli.parse", "--render-fps must be in (0, 3000]");
  }
  if (options.textureSize < 16U || options.textureSize > 1024U ||
      (options.textureSize & (options.textureSize - 1U)) != 0U) {
    fail(
        ErrorCode::InvalidArgument,
        "cli.parse",
        "--texture-size must be a power of two in [16, 1024]");
  }
  if (options.equipment != "iron" && options.equipment != "ember") {
    fail(
        ErrorCode::InvalidArgument,
        "cli.parse",
        "--equipment must be 'iron' or 'ember'");
  }
  if (options.windowWidth < 480 || options.windowHeight < 320) {
    fail(ErrorCode::InvalidArgument, "cli.parse", "window size is too small");
  }

  if (options.faultName == "none") {
    options.fault = FaultKind::None;
  } else if (options.faultName == "texture") {
    options.fault = FaultKind::Texture;
  } else if (options.faultName == "shader") {
    options.fault = FaultKind::Shader;
  } else if (options.faultName == "bone") {
    options.fault = FaultKind::Bone;
  } else {
    fail(
        ErrorCode::InvalidArgument,
        "cli.parse",
        "--fault must be none, texture, shader or bone");
  }
  return options;
}

void printUsage(const char* executableName) {
  std::cout
      << "Usage: " << executableName << " [options]\n"
      << "  --headless                 Disable window creation (headless build always does).\n"
      << "  --seed N                   Deterministic seed (default 1337).\n"
      << "  --logic-frames N           Fixed 60 Hz logic steps to simulate (default 240).\n"
      << "  --render-fps F             Variable render rate used for statistics (default 120).\n"
      << "  --max-render-frames N      Stop after N rendered frames; 0 means full duration.\n"
      << "  --switch-time T[,T...]     Toggle equipment at each fixed-step time in seconds.\n"
      << "  --equipment iron|ember     Initial equipment set (default iron).\n"
      << "  --texture-size N           Power-of-two procedural texture size (default 128).\n"
      << "  --report PATH              Validation report output.\n"
      << "  --snapshot PATH            Final state snapshot output.\n"
      << "  --fault KIND               none|texture|shader|bone (deterministic failure test).\n"
      << "  --width N --height N       Window dimensions for character_demo.\n"
      << "  --help                     Show this help.\n";
}

}  // namespace lcr
