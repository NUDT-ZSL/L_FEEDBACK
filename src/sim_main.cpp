#include "lightcluster/buffer_sync.hpp"
#include "lightcluster/cli.hpp"
#include "lightcluster/deterministic_engine.hpp"
#include "lightcluster/exit_codes.hpp"
#include "lightcluster/json_log.hpp"

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
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

    try {
        DeterministicEngine engine(options.run);
        const SceneSnapshot initial = engine.scene().snapshot();
        emit(formatBufferInitJson(engine.initialSync(), initial, options.run));

        std::vector<std::uint8_t> fakeGpu = engine.initialSync().usedBytes == 0
            ? std::vector<std::uint8_t>{}
            : engine.synchronizer().residentImage().bytes;
        bool faultInjected = false;

        for (std::uint32_t frame = 0; frame < options.run.frames; ++frame) {
            const FrameTick tick = engine.tick(frame);
            if (tick.sync.uploaded) {
                fakeGpu = engine.synchronizer().residentImage().bytes;
            }

            for (const RunEvent& event : tick.events) {
                emit(formatEventJson(event));
            }

            DebugFrameStatus debug;
            BufferValidation validation;
            if (options.run.debug) {
                if (options.run.injectDebugFault && !faultInjected &&
                    fakeGpu.size() >= sizeof(std::uint32_t)) {
                    std::uint32_t corruptedCount = 0;
                    std::memcpy(
                        &corruptedCount, fakeGpu.data(),
                        sizeof(corruptedCount));
                    ++corruptedCount;
                    std::memcpy(
                        fakeGpu.data(), &corruptedCount,
                        sizeof(corruptedCount));
                    faultInjected = true;
                }

                validation = validateReadback(
                    fakeGpu, engine.synchronizer().residentImage());
                debug.checked = true;
                debug.ok = validation.ok;
                debug.readbackLights = validation.actualLights;
                debug.readbackPoint = validation.actualPoint;
                debug.readbackDirectional = validation.actualDirectional;
            }

            emit(formatFrameJson(tick, options.run, debug));
            if (options.run.debug && !validation.ok) {
                emit(formatDebugErrorJson(frame, validation));
                return kExitBufferMismatch;
            }
        }
    } catch (const std::exception& exception) {
        emit(formatFatalJson("ENGINE_EXCEPTION", exception.what()));
        return kExitUsage;
    }

    return kExitSuccess;
}
