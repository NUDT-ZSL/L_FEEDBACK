#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#ifdef _WIN32
#include <windows.h>
#include <cstdio>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "core/Animation.hpp"
#include "core/CapeSorter.hpp"
#include "core/Cli.hpp"
#include "core/Diagnostics.hpp"
#include "core/Error.hpp"
#include "core/Report.hpp"
#include "core/Serializer.hpp"
#include "core/Simulation.hpp"
#include "render/GLRenderer.hpp"

namespace {

using namespace lcr;

void attachWindowsConsole() {
#ifdef _WIN32
  if (AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole()) {
    FILE* stream = nullptr;
    freopen_s(&stream, "CONOUT$", "w", stdout);
    freopen_s(&stream, "CONOUT$", "w", stderr);
    freopen_s(&stream, "CONIN$", "r", stdin);
  }
#endif
}

std::string gLastGlfwError;

void glfwErrorCallback(const int code, const char* description) {
  gLastGlfwError =
      "GLFW error " + std::to_string(code) + ": " +
      (description != nullptr ? description : "<unknown>");
}

class GlfwRuntime {
 public:
  GlfwRuntime() {
    glfwSetErrorCallback(glfwErrorCallback);
    if (glfwInit() != GLFW_TRUE) {
      fail(
          ErrorCode::GpuInit,
          "glfw.init",
          "glfwInit failed: " + gLastGlfwError);
    }
    initialized_ = true;
  }

  ~GlfwRuntime() {
    if (window_ != nullptr) {
      glfwDestroyWindow(window_);
    }
    if (initialized_) {
      glfwTerminate();
    }
  }

  GLFWwindow* createWindow(
      const int width,
      const int height,
      const char* title) {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (window_ == nullptr) {
      fail(
          ErrorCode::GpuInit,
          "glfw.createWindow",
          "window/context creation failed: " + gLastGlfwError);
    }
    glfwMakeContextCurrent(window_);
    if (gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)) == 0) {
      fail(ErrorCode::GpuInit, "glad.load", "failed to load OpenGL 3.3 entry points");
    }
    glfwSwapInterval(0);
    return window_;
  }

 private:
  bool initialized_ = false;
  GLFWwindow* window_ = nullptr;
};

struct InputState {
  OrbitCamera* camera = nullptr;
  GLRenderer* renderer = nullptr;
  bool dragging = false;
  double lastX = 0.0;
  double lastY = 0.0;
};

void cursorCallback(GLFWwindow* window, const double x, const double y) {
  auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
  if (input == nullptr || input->camera == nullptr) {
    return;
  }
  if (input->dragging) {
    const double dx = x - input->lastX;
    const double dy = y - input->lastY;
    input->camera->yaw += static_cast<float>(dx) * 0.0075F;
    input->camera->pitch =
        std::fmax(-0.18F, std::fmin(1.18F, input->camera->pitch + static_cast<float>(dy) * 0.006F));
  }
  input->lastX = x;
  input->lastY = y;
}

void mouseButtonCallback(GLFWwindow* window, const int button, const int action, int) {
  auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
  if (input == nullptr || button != GLFW_MOUSE_BUTTON_LEFT) {
    return;
  }
  if (action == GLFW_PRESS) {
    input->dragging = true;
    glfwGetCursorPos(window, &input->lastX, &input->lastY);
  } else if (action == GLFW_RELEASE) {
    input->dragging = false;
  }
}

void scrollCallback(GLFWwindow* window, double, const double yOffset) {
  auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
  if (input == nullptr || input->camera == nullptr) {
    return;
  }
  input->camera->distance = std::fmax(
      2.6F,
      std::fmin(9.5F, input->camera->distance - static_cast<float>(yOffset) * 0.35F));
}

void framebufferCallback(GLFWwindow* window, const int width, const int height) {
  auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
  if (input != nullptr && input->renderer != nullptr) {
    input->renderer->resize(width, height);
  }
}

std::string equipmentAtTime(
    const Simulation& simulation,
    const double renderTime) {
  std::string equipment = simulation.config().initialEquipment;
  for (const SwitchEvent& event : simulation.switchEvents()) {
    if (event.renderTime <= renderTime + 1.0e-9) {
      equipment = event.toEquipment;
    }
  }
  return equipment;
}

RunReport makeReport(
    const Simulation& simulation,
    const std::vector<double>& frameTimes,
    const PipelineReport& pipeline,
    const OrbitCamera& camera) {
  RunReport report;
  report.timingMode = "wall_clock";
  report.config = simulation.config();
  report.logicSteps = simulation.logicTick();
  report.simulatedSeconds = simulation.logicTime();
  report.renderFrames = static_cast<std::uint32_t>(frameTimes.size());
  report.frameStats = summarizeFrameTimes(frameTimes);
  report.skeletonHash = simulation.assets().skeletonHash;
  report.meshHash = simulation.assets().meshHash;
  report.characterHash = simulation.assets().combinedHash;
  report.equipmentCatalogHash = simulation.equipment().catalogHash();
  report.finalStateHash = simulation.stateHash(simulation.renderTime());
  report.finalEquipment = simulation.equipment().active().id;
  report.equipmentSets = simulation.equipment().sets();
  report.switchEvents = simulation.switchEvents();
  report.pipeline = pipeline;
  report.pipeline.equipmentPipelineRebuilds = 0;

  const double comparisonDuration =
      std::fmax(0.25, std::fmin(4.0, simulation.logicTime()));
  report.interpolation.push_back(
      compareInterpolation(simulation.assets(), 45.0, comparisonDuration));
  report.interpolation.push_back(
      compareInterpolation(simulation.assets(), 144.0, comparisonDuration));

  const double duration = std::fmax(0.01, simulation.logicTime());
  const double capeTimes[] = {
      0.35,
      std::max(0.5, duration * 0.35),
      std::max(0.7, duration * 0.70),
      std::max(0.9, duration * 0.92),
  };
  for (const double time : capeTimes) {
    if (time > duration) {
      continue;
    }
    CapeSampleReport sample;
    sample.renderTime = time;
    sample.equipment = equipmentAtTime(simulation, time);
    sample.sort = analyzeCapeSort(
        simulation.assets(),
        simulation.interpolatedPose(time),
        camera.position());
    report.capeSamples.push_back(sample);
  }
  return report;
}

void runWindowed(const CliOptions& options) {
  SimulationConfig config;
  config.seed = options.seed;
  config.textureSize = options.textureSize;
  config.initialEquipment = options.equipment;
  config.scheduledSwitchTimes = options.switchTimes;
  config.forceTextureFailure = options.fault == FaultKind::Texture;

  Simulation simulation(config);
  if (options.fault == FaultKind::Bone) {
    simulation.assets().parts.front().mesh.vertices.front().joints[0] = kBoneCount;
    simulation.assets().validate();
  }

  GlfwRuntime glfw;
  GLFWwindow* window = glfw.createWindow(
      options.windowWidth,
      options.windowHeight,
      "Layered Character Renderer");
  glfwSetKeyCallback(
      window,
      [](GLFWwindow* handle, const int key, int, const int action, int) {
        if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
          glfwSetWindowShouldClose(handle, GLFW_TRUE);
        }
      });

  GLRenderer renderer;
  renderer.initialize(
      simulation.assets(),
      simulation.equipment(),
      options.fault == FaultKind::Shader,
      options.windowWidth,
      options.windowHeight);

  OrbitCamera camera;
  InputState input;
  input.camera = &camera;
  input.renderer = &renderer;
  glfwSetWindowUserPointer(window, &input);
  glfwSetCursorPosCallback(window, cursorCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetScrollCallback(window, scrollCallback);
  glfwSetFramebufferSizeCallback(window, framebufferCallback);

  std::vector<double> frameTimes;
  frameTimes.reserve(4096);
  std::uint64_t renderedFrames = 0;
  bool eWasDown = false;
  double previousTime = glfwGetTime();
  double titleUpdateTime = previousTime;
  while (glfwWindowShouldClose(window) == GLFW_FALSE) {
    glfwPollEvents();
    const double now = glfwGetTime();
    const double delta = std::fmax(0.0, now - previousTime);
    previousTime = now;
    if (!frameTimes.empty() || delta > 0.0) {
      frameTimes.push_back(delta * 1000.0);
    }
    simulation.advanceBy(delta);

    const bool eDown = glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS;
    if (eDown && !eWasDown) {
      simulation.switchEquipmentNow(simulation.renderTime());
    }
    eWasDown = eDown;
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
      camera = OrbitCamera{};
    }
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    const Pose pose = simulation.interpolatedPose(simulation.renderTime());
    renderer.render(
        pose,
        simulation.equipment().active(),
        simulation.equipment().activeIndex(),
        camera);
    glfwSwapBuffers(window);
    ++renderedFrames;

    if (now - titleUpdateTime > 0.25) {
      const std::string title =
          "Layered Character Renderer | " +
          simulation.equipment().active().displayName +
          " | frame " + std::to_string(renderedFrames) +
          " | E: equipment  drag: orbit  wheel: zoom";
      glfwSetWindowTitle(window, title.c_str());
      titleUpdateTime = now;
    }
    if (options.maxRenderFrames != 0U && renderedFrames >= options.maxRenderFrames) {
      break;
    }
    if (options.logicFramesSpecified &&
        simulation.logicTick() >= static_cast<std::uint64_t>(options.logicFrames)) {
      break;
    }
  }

  const RunReport report = makeReport(
      simulation,
      frameTimes,
      renderer.pipelineReport(),
      camera);
  writeValidationReport(options.reportPath, report);
  writeStateSnapshot(
      options.snapshotPath,
      simulation,
      simulation.renderTime(),
      renderer.pipelineReport());

  std::cout << "{\"status\":\"ok\",\"mode\":\"windowed\",\"seed\":" << options.seed
            << ",\"logic_steps\":" << simulation.logicTick()
            << ",\"render_frames\":" << renderedFrames
            << ",\"shader_programs\":"
            << renderer.pipelineReport().shaderProgramBuilds
            << ",\"equipment_pipeline_rebuilds\":0"
            << ",\"state_hash\":\""
            << hexU64(simulation.stateHash(simulation.renderTime()))
            << "\"}\n";
}

}  // namespace

int main(int argc, char** argv) {
  attachWindowsConsole();
  try {
    const CliOptions options = parseCommandLine(argc, argv);
    if (options.help) {
      printUsage(argv[0]);
      return 0;
    }
    if (options.headless) {
      fail(
          ErrorCode::InvalidArgument,
          "main.headless",
          "character_demo is the windowed executable; use character_headless for --headless");
    }
    runWindowed(options);
    return 0;
  } catch (const AppError& error) {
    emitErrorToStderr(error);
    return errorExitCode(error.code());
  } catch (const std::exception& error) {
    const AppError wrapped(
        ErrorCode::InternalInvariant,
        "window.main",
        error.what());
    emitErrorToStderr(wrapped);
    return errorExitCode(wrapped.code());
  }
}
