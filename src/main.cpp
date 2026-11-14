#include "portaudio_output.h"
#include "scenario.h"
#include "spatial_audio/audio_engine.h"
#include "wav_io.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

using spatial_audio::AudioEngine;
using spatial_audio::ErrorCode;
using spatial_audio::MovingScenario;
using spatial_audio::SourceId;
using spatial_audio::SourceStats;

struct Options {
    bool deterministic = false;
    bool realtime = false;
    bool probeWav = false;
    bool help = false;
    double duration = 4.0;
    std::uint64_t seed = 0x5eed1234ULL;
    std::string outputPath = "artifacts/deterministic_mix.wav";
    std::string sampleDirectory = "artifacts/generated_samples";
    std::string probePath;
};

void printUsage() {
    std::printf(
        "Spatial Audio Mixer\n"
        "\n"
        "Usage:\n"
        "  spatial_audio_mixer --deterministic [options]\n"
        "  spatial_audio_mixer --realtime [options]\n"
        "  spatial_audio_mixer --probe-wav PATH\n"
        "\n"
        "Modes:\n"
        "  --deterministic   Render a repeatable scenario to a stereo WAV.\n"
        "  --realtime        Play the moving scenario through PortAudio.\n"
        "  --probe-wav PATH  Validate the strict WAV decoder and exit.\n"
        "\n"
        "Options:\n"
        "  --duration SEC    Scenario duration (default: 4.0)\n"
        "  --seed N          Deterministic 64-bit PRNG seed\n"
        "  --out PATH        Output WAV for deterministic mode\n"
        "  --sample-dir DIR  Directory for generated mono input WAVs\n"
        "  --help            Show this text\n");
}

bool parseDouble(const char* text, double& value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char* end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

bool parseUnsigned(const char* text, std::uint64_t& value) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        return false;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 0);
    if (end == text || *end != '\0') {
        return false;
    }
    value = static_cast<std::uint64_t>(parsed);
    return true;
}

bool parseOptions(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--deterministic") {
            options.deterministic = true;
        } else if (argument == "--realtime") {
            options.realtime = true;
        } else if (argument == "--probe-wav") {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --probe-wav path\n");
                return false;
            }
            options.probeWav = true;
            options.probePath = argv[i];
        } else if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else if (argument == "--duration") {
            if (++i >= argc || !parseDouble(argv[i], options.duration)) {
                std::fprintf(stderr, "invalid --duration value\n");
                return false;
            }
        } else if (argument == "--seed") {
            if (++i >= argc || !parseUnsigned(argv[i], options.seed)) {
                std::fprintf(stderr, "invalid --seed value\n");
                return false;
            }
        } else if (argument == "--out") {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --out path\n");
                return false;
            }
            options.outputPath = argv[i];
        } else if (argument == "--sample-dir") {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --sample-dir path\n");
                return false;
            }
            options.sampleDirectory = argv[i];
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argument.c_str());
            return false;
        }
    }

    if (options.duration < 0.1 || options.duration > 300.0) {
        std::fprintf(stderr, "duration must be in [0.1, 300] seconds\n");
        return false;
    }
    const int modeCount = static_cast<int>(options.deterministic) +
                          static_cast<int>(options.realtime) +
                          static_cast<int>(options.probeWav);
    if (modeCount > 1) {
        std::fprintf(stderr,
                     "--deterministic, --realtime, and --probe-wav are "
                     "mutually exclusive\n");
        return false;
    }
    return true;
}

void printStats(const char* label, const SourceStats& stats) {
    std::printf(
        "%-14s blocks=%llu avg_gain=%.4f avg_pan=%+.4f "
        "L=%.4f R=%.4f occluded=%.1f%% avg_cutoff=%.1fHz\n",
        label, static_cast<unsigned long long>(stats.processedBlocks),
        stats.averageDistanceGain, stats.averagePan, stats.averageLeftGain,
        stats.averageRightGain, stats.occlusionFraction * 100.0f,
        stats.averageCutoffHz);
}

std::uint64_t wavFnv1a(const std::vector<std::int16_t>& samples) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::int16_t sample : samples) {
        const std::uint16_t bits =
            static_cast<std::uint16_t>(sample);
        hash ^= static_cast<std::uint64_t>(bits & 0xffu);
        hash *= 1099511628211ULL;
        hash ^= static_cast<std::uint64_t>(bits >> 8u);
        hash *= 1099511628211ULL;
    }
    return hash;
}

int runDeterministic(const Options& options) {
    std::error_code filesystemError;
    std::filesystem::create_directories(options.sampleDirectory,
                                        filesystemError);
    if (filesystemError) {
        std::fprintf(stderr, "cannot create sample directory '%s': %s\n",
                     options.sampleDirectory.c_str(),
                     filesystemError.message().c_str());
        return 3;
    }
    const std::filesystem::path output(options.outputPath);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path(),
                                            filesystemError);
        if (filesystemError) {
            std::fprintf(stderr, "cannot create output directory: %s\n",
                         filesystemError.message().c_str());
            return 3;
        }
    }

    AudioEngine engine;
    MovingScenario scenario(engine, options.seed);
    std::string scenarioError;
    if (!scenario.prepare(options.sampleDirectory, scenarioError)) {
        std::fprintf(stderr, "scenario preparation failed: %s\n",
                     scenarioError.c_str());
        return 4;
    }

    const std::uint64_t totalFrames = static_cast<std::uint64_t>(
        std::llround(options.duration * spatial_audio::kSampleRate));
    std::vector<std::int16_t> mix;
    mix.reserve(static_cast<std::size_t>(totalFrames) * 2);

    std::uint64_t renderedFrames = 0;
    std::uint64_t blockCount = 0;
    std::int32_t peakAbs = 0;
    while (renderedFrames < totalFrames) {
        const std::size_t frames = static_cast<std::size_t>(
            std::min<std::uint64_t>(spatial_audio::kBlockFrames,
                                    totalFrames - renderedFrames));
        const double time =
            static_cast<double>(renderedFrames) / spatial_audio::kSampleRate;
        scenario.update(time);

        const std::size_t oldSize = mix.size();
        mix.resize(oldSize + frames * 2, 0);
        engine.processBlock(mix.data() + oldSize, frames);

        for (std::size_t i = 0; i < frames * 2; ++i) {
            const std::int32_t magnitude =
                std::abs(static_cast<std::int32_t>(mix[oldSize + i]));
            peakAbs = std::max(peakAbs, magnitude);
        }
        renderedFrames += frames;
        ++blockCount;
    }

    std::string writeError;
    if (!spatial_audio::detail::writeStereoWav16(options.outputPath, mix,
                                                 writeError)) {
        std::fprintf(stderr, "failed to write output WAV '%s': %s\n",
                     options.outputPath.c_str(), writeError.c_str());
        return 5;
    }
    if (!spatial_audio::detail::inspectStereoWav16(
            options.outputPath, renderedFrames, writeError)) {
        std::fprintf(stderr, "output WAV validation failed for '%s': %s\n",
                     options.outputPath.c_str(), writeError.c_str());
        return 7;
    }

    std::printf(
        "mode=deterministic seed=0x%llx samples=%zu frame_rate=%d "
        "block_frames=%zu rendered_frames=%llu blocks=%llu output=%s\n",
        static_cast<unsigned long long>(scenario.seed()),
        engine.loadedSampleCount(), spatial_audio::kSampleRate,
        spatial_audio::kBlockFrames,
        static_cast<unsigned long long>(renderedFrames),
        static_cast<unsigned long long>(blockCount),
        options.outputPath.c_str());

    bool validationOk = true;
    if (peakAbs <= 0) {
        std::fprintf(stderr, "validation: output is silent\n");
        validationOk = false;
    }
    if (peakAbs > 32112) {
        std::fprintf(stderr,
                     "validation: peak %d exceeded soft-limiter ceiling\n",
                     peakAbs);
        validationOk = false;
    }

    const auto& permanent = scenario.permanentSources();
    bool sawOcclusion = false;
    bool sawLeftPan = false;
    bool sawRightPan = false;
    for (std::size_t i = 0; i < permanent.size(); ++i) {
        SourceStats stats{};
        if (!engine.sourceStats(permanent[i].id, stats)) {
            std::fprintf(stderr, "validation: missing stats for source %zu\n", i);
            validationOk = false;
            continue;
        }
        const std::string label =
            "source[" + std::to_string(i) + "]/" +
            std::to_string(permanent[i].toneHz) + "Hz";
        printStats(label.c_str(), stats);
        if (stats.processedBlocks != blockCount) {
            std::fprintf(stderr,
                         "validation: source %zu processed %llu of %llu blocks\n",
                         i,
                         static_cast<unsigned long long>(stats.processedBlocks),
                         static_cast<unsigned long long>(blockCount));
            validationOk = false;
        }
        if (stats.averageDistanceGain < 0.0f ||
            stats.averageDistanceGain > 1.0f ||
            stats.averagePan < -1.0f || stats.averagePan > 1.0f ||
            stats.occlusionFraction < 0.0f ||
            stats.occlusionFraction > 1.0f) {
            std::fprintf(stderr,
                         "validation: source %zu has out-of-range statistics\n",
                         i);
            validationOk = false;
        }
        sawOcclusion = sawOcclusion || stats.occlusionFraction > 0.0f;
        sawLeftPan = sawLeftPan || stats.averagePan < -0.05f;
        sawRightPan = sawRightPan || stats.averagePan > 0.05f;
    }
    if (!sawOcclusion) {
        std::fprintf(stderr,
                     "validation: no source was occluded by the test wall\n");
        validationOk = false;
    }
    if (!sawLeftPan || !sawRightPan) {
        std::fprintf(stderr,
                     "validation: trajectory did not exercise both pan "
                     "directions\n");
        validationOk = false;
    }

    if (scenario.transientWasAdded()) {
        SourceStats stats{};
        if (!engine.sourceStats(scenario.transientSource().id, stats)) {
            std::fprintf(stderr,
                         "validation: missing transient source statistics\n");
            validationOk = false;
        } else {
            printStats("transient/494Hz", stats);
            if (stats.processedBlocks == 0 || stats.processedBlocks >= blockCount) {
                std::fprintf(stderr,
                             "validation: transient lifecycle statistics "
                             "look incorrect\n");
                validationOk = false;
            }
        }
    }

    const std::size_t expectedActive =
        scenario.transientWasAdded() && !scenario.transientWasRemoved()
            ? 9
            : 8;
    const std::size_t active = engine.activeSourceCount();
    if (active != expectedActive) {
        std::fprintf(stderr,
                     "validation: final active source count is %zu, expected %zu\n",
                     active, expectedActive);
        validationOk = false;
    }
    if (options.duration >= 2.6 &&
        (!scenario.transientWasAdded() || !scenario.transientWasRemoved())) {
        std::fprintf(stderr,
                     "validation: transient add/remove lifecycle did not complete\n");
        validationOk = false;
    }

    std::printf(
        "peak_abs=%d fnv1a_wav16=0x%016llx validation=%s\n", peakAbs,
        static_cast<unsigned long long>(wavFnv1a(mix)),
        validationOk ? "PASS" : "FAIL");
    return validationOk ? 0 : 6;
}

int runRealtime(const Options& options) {
    std::error_code filesystemError;
    std::filesystem::create_directories(options.sampleDirectory,
                                        filesystemError);
    if (filesystemError) {
        std::fprintf(stderr, "cannot create sample directory '%s': %s\n",
                     options.sampleDirectory.c_str(),
                     filesystemError.message().c_str());
        return 3;
    }

    AudioEngine engine;
    MovingScenario scenario(engine, options.seed);
    std::string scenarioError;
    if (!scenario.prepare(options.sampleDirectory, scenarioError)) {
        std::fprintf(stderr, "scenario preparation failed: %s\n",
                     scenarioError.c_str());
        return 4;
    }

    spatial_audio::PortAudioOutput output(engine);
    std::string audioError;
    const ErrorCode startResult = output.start(audioError);
    if (startResult != ErrorCode::Ok) {
        std::fprintf(stderr, "PortAudio error: %s\n", audioError.c_str());
        return static_cast<int>(startResult);
    }

    std::printf("mode=realtime seed=0x%llx duration=%.2fs output=PortAudio "
                "44.1kHz stereo 16-bit\n",
                static_cast<unsigned long long>(scenario.seed()),
                options.duration);

    const auto start = std::chrono::steady_clock::now();
    while (true) {
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          start)
                .count();
        if (elapsed >= options.duration) {
            break;
        }
        scenario.update(elapsed);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    output.stop();
    std::printf("realtime_stream_closed active_sources=%zu\n",
                engine.activeSourceCount());
    return 0;
}

int runProbeWav(const Options& options) {
    AudioEngine engine;
    spatial_audio::SampleId sampleId = 0;
    const ErrorCode result = engine.loadWav(options.probePath, sampleId);
    if (result != ErrorCode::Ok) {
        std::fprintf(stderr,
                     "probe failed: %s (exit code %d), file='%s'\n",
                     errorCodeName(result), static_cast<int>(result),
                     options.probePath.c_str());
        return static_cast<int>(result);
    }
    std::printf("probe ok: file='%s' sample_id=%u\n",
                options.probePath.c_str(), sampleId);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options options{};
    if (!parseOptions(argc, argv, options)) {
        printUsage();
        return 2;
    }
    if (options.help) {
        printUsage();
        return 0;
    }
    if (!options.deterministic && !options.realtime && !options.probeWav) {
        std::fprintf(stderr,
                     "select --deterministic, --realtime, or --probe-wav\n\n");
        printUsage();
        return 2;
    }
    if (options.deterministic) {
        return runDeterministic(options);
    }
    if (options.realtime) {
        return runRealtime(options);
    }
    return runProbeWav(options);
}
