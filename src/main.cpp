#include "skinbake/bake.hpp"
#include "skinbake/error.hpp"
#include "skinbake/json.hpp"
#include "skinbake/scenarios.hpp"
#include "skinbake/snapshot.hpp"
#include "skinbake/verify.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using namespace skinbake;

class Arguments {
public:
    Arguments(int argc, char** argv, int start) {
        static const std::unordered_set<std::string> flags{
            "--variant", "--no-resume", "--help"};
        for (int i = start; i < argc; ++i) {
            std::string token = argv[i];
            if (token == "--help" || token == "-h") {
                flags_["--help"] = "";
                continue;
            }
            if (token.rfind("--", 0) != 0) {
                throw_error(ErrorCode::InvalidArgument,
                            "unexpected positional argument: " + token,
                            {{"argument", token}});
            }
            if (flags.count(token) != 0) {
                if (values_.count(token) != 0) {
                    throw_error(ErrorCode::InvalidArgument,
                                "duplicate option: " + token,
                                {{"option", token}});
                }
                values_[token] = "";
                flags_[token] = "";
                continue;
            }
            if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0) {
                throw_error(ErrorCode::InvalidArgument,
                            "missing value for option: " + token,
                            {{"option", token}});
            }
            if (values_.count(token) != 0) {
                throw_error(ErrorCode::InvalidArgument,
                            "duplicate option: " + token,
                            {{"option", token}});
            }
            values_[token] = argv[++i];
        }
    }

    bool has(const std::string& name) const {
        return values_.count(name) != 0;
    }
    bool flag(const std::string& name) const {
        return flags_.count(name) != 0;
    }
    std::string value(const std::string& name,
                      const std::string& fallback = "") const {
        auto it = values_.find(name);
        return it == values_.end() ? fallback : it->second;
    }
    std::string required(const std::string& name) const {
        auto it = values_.find(name);
        if (it == values_.end() || it->second.empty()) {
            throw_error(ErrorCode::InvalidArgument,
                        "required option is missing: " + name,
                        {{"option", name}});
        }
        return it->second;
    }
    std::uint32_t u32(const std::string& name,
                      std::uint32_t fallback) const {
        if (!has(name)) {
            return fallback;
        }
        return parse_u32(value(name), name);
    }
    std::uint32_t required_u32(const std::string& name) const {
        return parse_u32(required(name), name);
    }
    double number(const std::string& name, double fallback) const {
        if (!has(name)) {
            return fallback;
        }
        return parse_double(value(name), name);
    }

private:
    static std::uint32_t parse_u32(const std::string& text,
                                   const std::string& option) {
        std::size_t consumed = 0;
        unsigned long long value = 0;
        try {
            value = std::stoull(text, &consumed, 10);
        } catch (const std::exception&) {
            throw_error(ErrorCode::InvalidArgument,
                        "invalid unsigned integer for " + option + ": " + text,
                        {{"option", option}, {"value", text}});
        }
        if (consumed != text.size() ||
            value > std::numeric_limits<std::uint32_t>::max()) {
            throw_error(ErrorCode::InvalidArgument,
                        "unsigned integer is out of range for " + option +
                            ": " + text,
                        {{"option", option}, {"value", text}});
        }
        return static_cast<std::uint32_t>(value);
    }

    static double parse_double(const std::string& text,
                               const std::string& option) {
        std::size_t consumed = 0;
        double value = 0.0;
        try {
            value = std::stod(text, &consumed);
        } catch (const std::exception&) {
            throw_error(ErrorCode::InvalidArgument,
                        "invalid number for " + option + ": " + text,
                        {{"option", option}, {"value", text}});
        }
        if (consumed != text.size() || !std::isfinite(value)) {
            throw_error(ErrorCode::InvalidArgument,
                        "invalid number for " + option + ": " + text,
                        {{"option", option}, {"value", text}});
        }
        return value;
    }

    std::unordered_map<std::string, std::string> values_;
    std::unordered_map<std::string, std::string> flags_;
};

void print_usage(std::ostream& out) {
    out <<
R"(skinbake - deterministic offline skeletal animation snapshot baker

Usage:
  skinbake bake --scenario NAME --output FILE [options]
      --scenario idle|parallel|transition|checksum|version|long-loop
      --frames N --fps N --version N --workers N --block-frames N
      --variant --no-resume --event-log FILE
      --crash-after-commit-frame N

  skinbake verify --scenario NAME --file SNAPSHOT [options]
      --fps N --expect-version N --tolerance 1e-5

  skinbake load --file SNAPSHOT [--expect-version N]
  skinbake inspect --file SNAPSHOT
  skinbake error --scenario NAME
      NAME: curve-time|bezier-denominator|weight-zero|weight-not-one|
            skeleton-cycle|curve-empty
  skinbake timeline-check [--frames 10000] [--fps 60]

All failure output is a stable JSON object on stderr.  Snapshot magic is
SKNBKE01; the snapshot version is checked before checksum validation.
)";
}

std::string mismatch_json(const std::vector<std::uint32_t>& frames) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < frames.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << frames[i];
    }
    out << ']';
    return out.str();
}

Scenario scenario_from_arguments(const Arguments& args) {
    const std::string name = args.value("--scenario", "idle");
    Scenario scenario = make_scenario(name, args.flag("--variant"));
    if (args.has("--fps")) {
        scenario.fps = args.u32("--fps", scenario.fps);
        if (scenario.fps == 0) {
            throw_error(ErrorCode::InvalidArgument, "fps must be positive");
        }
        scenario.definition.scene.fps = scenario.fps;
    }
    return scenario;
}

int command_bake(const Arguments& args) {
    Scenario scenario = scenario_from_arguments(args);
    const auto timeline = prepare_timeline(scenario.definition);
    BakeOptions options;
    options.outputPath = args.required("--output");
    options.version = args.u32("--version", 1);
    if (options.version == 0) {
        throw_error(ErrorCode::InvalidArgument,
                    "snapshot version must be positive");
    }
    options.totalFrames = args.u32("--frames", scenario.defaultFrames);
    if (options.totalFrames == 0) {
        throw_error(ErrorCode::InvalidArgument,
                    "frame count must be positive");
    }
    options.requestedWorkers = args.u32("--workers", 1);
    options.blockFrames = args.u32("--block-frames", 8);
    options.resume = !args.flag("--no-resume");
    options.eventLogPath = args.value("--event-log");
    if (args.has("--crash-after-commit-frame")) {
        options.crashAfterCommitFrame =
            args.u32("--crash-after-commit-frame", 0);
    }

    const BakeReport report = bake(*timeline, options);
    std::cout << json_object({
                     {"ok", json_bool(true)},
                     {"mode", json_string("bake")},
                     {"version", json_u64(report.version)},
                     {"clip", json_string(report.clip)},
                     {"frames", json_u64(report.frames)},
                     {"fps", json_u64(report.fps)},
                     {"fingerprint", json_string(std::to_string(report.fingerprint))},
                     {"resumed", json_bool(report.resumed)},
                     {"discardedProgress", json_bool(report.discardedProgress)},
                     {"discardReason", json_string(report.discardReason)},
                     {"resumeFrame", json_u64(report.resumeFrame)},
                     {"firstNewFrame", json_u64(report.firstNewFrame)},
                     {"newFramesCommitted",
                      json_u64(report.newFramesCommitted)},
                     {"requestedWorkers",
                      json_u64(report.requestedWorkers)},
                     {"usedWorkers", json_u64(report.usedWorkers)},
                     {"logicalCores", json_u64(report.logicalCores)},
                     {"output", json_string(report.outputPath)},
                 })
              << '\n';
    return 0;
}

int command_verify(const Arguments& args) {
    Scenario scenario = scenario_from_arguments(args);
    const auto timeline = prepare_timeline(scenario.definition);
    std::optional<std::uint32_t> expectedVersion;
    if (args.has("--expect-version")) {
        expectedVersion = args.required_u32("--expect-version");
    }
    const double tolerance = args.number("--tolerance", 1e-5);
    const VerificationReport report = verify_snapshot(
        *timeline, args.required("--file"), tolerance, expectedVersion);
    std::cout << json_object({
                     {"ok", json_bool(true)},
                     {"mode", json_string("verify")},
                     {"passed", json_bool(report.passed)},
                     {"frames", json_u64(report.frames)},
                     {"tolerance", json_double(report.tolerance)},
                     {"maxAbsError", json_double(report.maxAbsError)},
                     {"maxErrorFrame", json_u64(report.maxErrorFrame)},
                     {"byteIdentical", json_bool(report.byteIdentical)},
                     {"mismatchFrames", mismatch_json(report.mismatchFrames)},
                 })
              << '\n';
    return report.passed ? 0 : 3;
}

int command_load(const Arguments& args) {
    std::optional<std::uint32_t> expectedVersion;
    if (args.has("--expect-version")) {
        expectedVersion = args.required_u32("--expect-version");
    }
    const std::string path = args.required("--file");
    const Snapshot snapshot = load_snapshot(path, expectedVersion);
    std::cout << json_object({
                     {"ok", json_bool(true)},
                     {"mode", json_string("load")},
                     {"file", json_string(path)},
                     {"version", json_u64(snapshot.version)},
                     {"fingerprint", json_string(std::to_string(snapshot.fingerprint))},
                     {"frames", json_u64(snapshot.frames.size())},
                     {"totalFrames", json_u64(snapshot.totalFrames)},
                     {"bones", json_u64(snapshot.boneCount)},
                     {"vertices", json_u64(snapshot.vertexCount)},
                     {"complete", json_bool(snapshot.complete)},
                     {"checksum", json_string("ok")},
                 })
              << '\n';
    return 0;
}

int command_inspect(const Arguments& args) {
    const std::string path = args.required("--file");
    const Snapshot snapshot = load_snapshot(path);
    std::cout << json_object({
                     {"ok", json_bool(true)},
                     {"mode", json_string("inspect")},
                     {"file", json_string(path)},
                     {"version", json_u64(snapshot.version)},
                     {"fingerprint", json_string(std::to_string(snapshot.fingerprint))},
                     {"frames", json_u64(snapshot.frames.size())},
                     {"totalFrames", json_u64(snapshot.totalFrames)},
                     {"bones", json_u64(snapshot.boneCount)},
                     {"vertices", json_u64(snapshot.vertexCount)},
                     {"complete", json_bool(snapshot.complete)},
                 })
              << '\n';
    return 0;
}

int command_error(const Arguments& args) {
    Scenario scenario = scenario_from_arguments(args);
    (void)prepare_timeline(scenario.definition);
    throw_error(ErrorCode::InvalidArgument,
                "selected scenario did not produce its expected error");
}

int command_timeline_check(const Arguments& args) {
    const std::uint32_t frames = args.u32("--frames", 10000);
    const std::uint32_t fps = args.u32("--fps", 60);
    if (frames == 0 || fps == 0) {
        throw_error(ErrorCode::InvalidArgument,
                    "timeline-check frames and fps must be positive");
    }
    Scenario scenario = make_scenario("idle");
    scenario.fps = fps;
    scenario.definition.scene.fps = fps;
    const auto prepared = prepare_scene(scenario.definition.scene);
    StateMachine machine(prepared, "idle");
    const Clip& clip = prepared->scene.clips[0];
    std::int64_t previousTime = 0;
    std::int64_t maximumDeviation = 0;
    for (std::uint32_t frame = 0; frame < frames; ++frame) {
        const std::int64_t frameTime = rounded_frame_time_ms(frame, fps);
        if (frame != 0) {
            machine.advance(frameTime - previousTime);
        }
        previousTime = frameTime;
        const std::int64_t expected =
            wrap_playback_time(frameTime, clip.durationMs, clip.loop);
        const std::int64_t playbackDeviation = static_cast<std::int64_t>(
            std::llabs(machine.playback_time_ms() - expected));
        maximumDeviation = std::max(
            maximumDeviation, playbackDeviation);
        if (machine.propagation_time_ms() != frameTime) {
            const std::int64_t propagationDeviation =
                static_cast<std::int64_t>(std::llabs(
                    machine.propagation_time_ms() - frameTime));
            maximumDeviation = std::max(
                maximumDeviation, propagationDeviation);
        }
    }
    const bool passed = maximumDeviation == 0;
    std::cout << json_object({
                     {"ok", json_bool(true)},
                     {"mode", json_string("timeline-check")},
                     {"passed", json_bool(passed)},
                     {"frames", json_u64(frames)},
                     {"fps", json_u64(fps)},
                     {"maxDeviationMs", json_i64(maximumDeviation)},
                     {"finalPlaybackTimeMs",
                      json_i64(machine.playback_time_ms())},
                     {"finalPropagationTimeMs",
                      json_i64(machine.propagation_time_ms())},
                 })
              << '\n';
    return passed ? 0 : 3;
}

int run(int argc, char** argv) {
    if (argc < 2) {
        print_usage(std::cout);
        return 0;
    }
    const std::string command = argv[1];
    if (command == "--help" || command == "-h" || command == "help") {
        print_usage(std::cout);
        return 0;
    }
    Arguments args(argc, argv, 2);
    if (args.flag("--help")) {
        print_usage(std::cout);
        return 0;
    }
    if (command == "bake") {
        return command_bake(args);
    }
    if (command == "verify") {
        return command_verify(args);
    }
    if (command == "load") {
        return command_load(args);
    }
    if (command == "inspect") {
        return command_inspect(args);
    }
    if (command == "error") {
        return command_error(args);
    }
    if (command == "timeline-check") {
        return command_timeline_check(args);
    }
    throw_error(ErrorCode::InvalidArgument,
                "unknown command: " + command, {{"command", command}});
}

}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const skinbake::BusinessError& error) {
        std::cerr << error.to_json() << '\n';
        return skinbake::exit_code_for(error.code());
    } catch (const std::exception& error) {
        skinbake::BusinessError wrapped(
            skinbake::ErrorCode::InternalError,
            std::string("unhandled failure: ") + error.what());
        std::cerr << wrapped.to_json() << '\n';
        return skinbake::exit_code_for(wrapped.code());
    }
}

