#include "lightcluster/buffer_sync.hpp"
#include "lightcluster/cli.hpp"
#include "lightcluster/deterministic_engine.hpp"
#include "lightcluster/exit_codes.hpp"
#include "lightcluster/json_log.hpp"
#include "lightcluster/renderer.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

void emit(const std::string& line) {
    std::cout << line << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace lightcluster;

    CliOptions options;
    std::string parseMessage;
    const CliParseResult parsed =
        parseCommandLine(argc, argv, options, parseMessage);
    if (parsed == CliParseResult::Help) {
        if (!parseMessage.empty()) {
            std::cout << parseMessage << '\n';
        } else {
            std::cout << usageText();
        }
        return kExitSuccess;
    }
    if (parsed == CliParseResult::Error) {
        emit(formatFatalJson("CLI_INVALID_ARGUMENT", parseMessage));
        std::cerr << parseMessage << '\n' << usageText();
        return kExitUsage;
    }

    DeterministicEngine* enginePointer = nullptr;
    Renderer* rendererPointer = nullptr;
    std::unique_ptr<DeterministicEngine> engine;
    std::unique_ptr<Renderer> renderer;

    try {
        engine = std::make_unique<DeterministicEngine>(options.run);
        enginePointer = engine.get();
    } catch (const std::exception& exception) {
        emit(formatFatalJson("ENGINE_INIT_FAILED", exception.what()));
        return kExitUsage;
    }

    try {
        renderer = std::make_unique<Renderer>();
        renderer->initialize(options.run.headless);
        rendererPointer = renderer.get();
        renderer->upload(
            engine->synchronizer().residentImage(),
            engine->initialSync().grew);
    } catch (const std::exception& exception) {
        emit(formatFatalJson("GPU_INIT_FAILED", exception.what()));
        return kExitGpuInit;
    }

    const SceneSnapshot initial = enginePointer->scene().snapshot();
    emit(formatBufferInitJson(
        enginePointer->initialSync(), initial, options.run));

    bool faultInjected = false;
    try {
        for (std::uint32_t frame = 0; frame < options.run.frames; ++frame) {
            const FrameTick tick = enginePointer->tick(frame);
            const GpuBufferImage& resident =
                enginePointer->synchronizer().residentImage();

            if (tick.sync.uploaded) {
                rendererPointer->upload(resident, tick.sync.grew);
            }

            DebugFrameStatus debug;
            BufferValidation validation;
            if (options.run.debug) {
                if (options.run.injectDebugFault && !faultInjected) {
                    rendererPointer->corruptCountForDiagnostic();
                    faultInjected = true;
                }
                const std::vector<std::uint8_t> actual =
                    rendererPointer->readback(resident.usedBytes);
                validation = validateReadback(actual, resident);
                debug.checked = true;
                debug.ok = validation.ok;
                debug.readbackLights = validation.actualLights;
                debug.readbackPoint = validation.actualPoint;
                debug.readbackDirectional = validation.actualDirectional;
            }

            if (options.run.debug && !validation.ok) {
                for (const RunEvent& event : tick.events) {
                    emit(formatEventJson(event));
                }
                emit(formatFrameJson(tick, options.run, debug));
                emit(formatDebugErrorJson(frame, validation));
                return kExitBufferMismatch;
            }

            rendererPointer->render(resident);
            rendererPointer->swapBuffers(options.run.headless);

            for (const RunEvent& event : tick.events) {
                emit(formatEventJson(event));
            }
            emit(formatFrameJson(tick, options.run, debug));

            rendererPointer->pollEvents();
            if (rendererPointer->shouldClose()) {
                emit(formatFatalJson(
                    "WINDOW_CLOSED",
                    "window closed before the requested frame count"));
                return kExitUsage;
            }
        }
    } catch (const std::exception& exception) {
        emit(formatFatalJson("RENDER_LOOP_FAILED", exception.what()));
        return kExitGpuUpload;
    }

    return kExitSuccess;
}
