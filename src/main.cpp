#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

#include "cli.hpp"
#include "context.hpp"
#include "json.hpp"
#include "output.hpp"
#include "recovery.hpp"
#include "renderer.hpp"
#include "scene.hpp"

namespace {

volatile std::sig_atomic_t g_signalNumber = 0;

void signalHandler(int signalNumber) { g_signalNumber = signalNumber; }

#ifdef _WIN32
BOOL WINAPI consoleControlHandler(DWORD event) {
  if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT || event == CTRL_CLOSE_EVENT ||
      event == CTRL_SHUTDOWN_EVENT) {
    g_signalNumber = 2;
    return TRUE;
  }
  return FALSE;
}
#endif

void installSignalHandlers() {
#ifdef _WIN32
  SetConsoleCtrlHandler(consoleControlHandler, TRUE);
#endif
  std::signal(SIGINT, signalHandler);
#ifndef _WIN32
  std::signal(SIGTERM, signalHandler);
#endif
}

bool interrupted() { return g_signalNumber != 0; }

int interruptionExitCode() {
#ifdef _WIN32
  return 130;
#else
  return 128 + static_cast<int>(g_signalNumber);
#endif
}

void throwIfInterrupted() {
  if (interrupted()) {
    throw orv::renderError("RENDER_INTERRUPTED", interruptionExitCode(),
                           "rendering interrupted by a handled signal");
  }
}

void emitInterruptedState() {
  std::cout << orv::makeFinalStateJson(false, "interrupted", "RENDER_INTERRUPTED",
                                      interruptionExitCode(), "", {}, {})
            << "\n";
  std::cout.flush();
}

void testPauseAfterMarker() {
  if (const char* pauseValue =
          std::getenv("OFFLINE_RENDER_TEST_PAUSE_AFTER_MARKER_MS")) {
    char* end = nullptr;
    const long milliseconds = std::strtol(pauseValue, &end, 10);
    if (end != pauseValue && *end == '\0' && milliseconds > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    }
  }
  if (std::getenv("OFFLINE_RENDER_TEST_RAISE_SIGINT") != nullptr) {
    std::raise(SIGINT);
  }
}

int run(int argc, char** argv) {
  orv::CliOptions options;
  orv::AppError error;
  if (!orv::parseCli(argc, argv, options, error)) {
    orv::emitErrorJson(error);
    return error.exit_code;
  }
  if (options.showHelp) {
    std::cout << orv::usageText();
    return 0;
  }

  std::filesystem::path prefix;
  try {
    prefix = std::filesystem::absolute(options.outputPrefix);
  } catch (const std::filesystem::filesystem_error& caught) {
    orv::emitErrorJson(orv::ioError("IO_ERR_OUTPUT_PATH", 6,
                                    "unable to resolve output prefix: " +
                                        std::string(caught.what())));
    return 6;
  }

  orv::RecoveryReport recovery;
  orv::AppError recoveryError;
  recovery = orv::recoverIncompleteSession(prefix, recoveryError);
  if (recovery.foundMarker) {
    std::cerr << orv::makeRecoveryJson(recovery.removedTempFiles.size()) << "\n";
    std::cerr.flush();
  }
  if (!recoveryError.code.empty()) {
    orv::emitErrorJson(recoveryError);
  }

  throwIfInterrupted();

  orv::Scene scene;
  if (!orv::loadScene(options.scenePath, scene, error)) {
    orv::emitErrorJson(error);
    return error.exit_code;
  }
  throwIfInterrupted();

  orv::SessionMarker marker(prefix);
  if (!marker.begin(error)) {
    orv::emitErrorJson(error);
    return error.exit_code;
  }

  testPauseAfterMarker();
  throwIfInterrupted();

  orv::GlContext context;
  if (!context.initialize(error)) {
    orv::emitErrorJson(error);
    return error.exit_code;
  }

  orv::RenderOptions renderOptions;
  renderOptions.width = options.width;
  renderOptions.height = options.height;
  renderOptions.shadowSize = options.shadowSize;
  renderOptions.shadowBias = options.shadowBias;
  renderOptions.pass = options.pass;

  orv::Renderer renderer(scene, renderOptions, interrupted);
  const orv::RenderOutput output = renderer.render();
  throwIfInterrupted();

  const std::string passName = orv::renderPassName(options.pass);
  const auto ppmPath = orv::makePpmPath(prefix, passName);
  const auto statsPath = orv::makeStatsPath(prefix);
  if (!orv::writePpmAtomically(ppmPath, output.image, error)) {
    orv::emitErrorJson(error);
    return error.exit_code;
  }
  throwIfInterrupted();
  if (!orv::writeStatsAtomically(statsPath, output.stats, error)) {
    orv::emitErrorJson(error);
    return error.exit_code;
  }
  throwIfInterrupted();

  orv::AppError markerError;
  if (!marker.complete(markerError)) {
    orv::emitErrorJson(markerError);
  }
  throwIfInterrupted();

  std::cout << orv::makeFinalStateJson(true, "completed", "RENDER_OK", 0, passName,
                                      ppmPath, statsPath)
            << "\n";
  std::cout.flush();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  installSignalHandlers();
  try {
    return run(argc, argv);
  } catch (const orv::AppError& caught) {
    if (caught.code == "RENDER_INTERRUPTED") {
      emitInterruptedState();
    } else {
      orv::emitErrorJson(caught);
    }
    return caught.exit_code;
  } catch (const std::exception& caught) {
    orv::emitErrorJson(orv::AppError{"INTERNAL_ERR", caught.what(), 1, {}, {}});
    return 1;
  } catch (...) {
    orv::emitErrorJson(
        orv::AppError{"INTERNAL_ERR", "unknown internal failure", 1, {}, {}});
    return 1;
  }
}
