#include "spatial_audio/audio_engine.h"
#include "wav_io.h"

#include <sndfile.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace {

std::atomic<std::size_t> g_allocations{0};

int failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__,    \
                         __LINE__, #condition);                              \
            ++failures;                                                      \
        }                                                                    \
    } while (false)

bool near(float actual, float expected, float tolerance) {
    return std::fabs(actual - expected) <= tolerance;
}

constexpr float kPi = 3.14159265358979323846f;

std::vector<std::int16_t> constantPcm(std::size_t frames, float amplitude) {
    std::vector<std::int16_t> pcm(frames);
    for (std::size_t i = 0; i < frames; ++i) {
        pcm[i] = static_cast<std::int16_t>(
            std::lrintf(amplitude * 32767.0f));
    }
    return pcm;
}

std::vector<std::int16_t> sinePcm(std::size_t frames,
                                  int frequency,
                                  float amplitude) {
    std::vector<std::int16_t> pcm(frames);
    const float twoPi = 2.0f * kPi;
    for (std::size_t i = 0; i < frames; ++i) {
        const float time =
            static_cast<float>(i) /
            static_cast<float>(spatial_audio::kSampleRate);
        pcm[i] = static_cast<std::int16_t>(
            std::lrintf(amplitude * std::sin(twoPi * frequency * time) *
                        32767.0f));
    }
    return pcm;
}

bool writeCustomWav(const std::string& path,
                    int sampleRate,
                    int channels,
                    int format,
                    const std::vector<float>& samples) {
    SF_INFO info{};
    info.samplerate = sampleRate;
    info.channels = channels;
    info.format = format;
    SNDFILE* file = sf_open(path.c_str(), SFM_WRITE, &info);
    if (file == nullptr) {
        return false;
    }
    const sf_count_t frames =
        static_cast<sf_count_t>(samples.size() / channels);
    const sf_count_t written = sf_writef_float(file, samples.data(), frames);
    const int closeResult = sf_close(file);
    return written == frames && closeResult == 0;
}

float outputRms(const std::vector<std::int16_t>& samples) {
    double energy = 0.0;
    for (std::int16_t sample : samples) {
        const double value = static_cast<double>(sample) / 32767.0;
        energy += value * value;
    }
    return static_cast<float>(std::sqrt(
        energy /
        static_cast<double>(std::max<std::size_t>(1, samples.size()))));
}

void testWavErrors(const std::filesystem::path& root) {
    spatial_audio::AudioEngine engine;
    spatial_audio::SampleId sampleId = 0;
    const auto missing =
        engine.loadWav((root / "missing.wav").string(), sampleId);
    CHECK(missing == spatial_audio::ErrorCode::WavOpenFailed);
    CHECK(engine.loadedSampleCount() == 0);

    const auto stereoPath = root / "stereo.wav";
    std::vector<std::int16_t> stereo(1000, 0);
    std::string error;
    CHECK(spatial_audio::detail::writeStereoWav16(stereoPath.string(), stereo,
                                                  error));
    CHECK(engine.loadWav(stereoPath.string(), sampleId) ==
          spatial_audio::ErrorCode::WavUnsupportedFormat);
    CHECK(engine.loadedSampleCount() == 0);

    const auto ratePath = root / "rate_48000.wav";
    std::vector<float> floatSamples(4800, 0.0f);
    CHECK(writeCustomWav(ratePath.string(), 48000, 1,
                         SF_FORMAT_WAV | SF_FORMAT_PCM_16, floatSamples));
    CHECK(engine.loadWav(ratePath.string(), sampleId) ==
          spatial_audio::ErrorCode::WavUnsupportedFormat);

    const auto floatPath = root / "float.wav";
    CHECK(writeCustomWav(floatPath.string(), 44100, 1,
                         SF_FORMAT_WAV | SF_FORMAT_FLOAT, floatSamples));
    CHECK(engine.loadWav(floatPath.string(), sampleId) ==
          spatial_audio::ErrorCode::WavUnsupportedFormat);
    CHECK(engine.loadedSampleCount() == 0);
}

std::unique_ptr<spatial_audio::AudioEngine> makeSourceEngine(
    const std::string& samplePath,
    spatial_audio::SourceConfig config,
    spatial_audio::SourceId& sourceId) {
    auto engine = std::make_unique<spatial_audio::AudioEngine>();
    spatial_audio::SampleId sampleId = 0;
    CHECK(engine->loadWav(samplePath, sampleId) ==
          spatial_audio::ErrorCode::Ok);
    config.sample = sampleId;
    CHECK(engine->addSource(config, sourceId) ==
          spatial_audio::ErrorCode::Ok);

    spatial_audio::ListenerState listener{};
    listener.position = {0.0f, 0.0f, 0.0f};
    listener.forward = {0.0f, 0.0f, 1.0f};
    listener.up = {0.0f, 1.0f, 0.0f};
    CHECK(engine->setListener(listener));
    return engine;
}

void testAttenuationAndPanning(const std::filesystem::path& root) {
    const auto samplePath = root / "constant.wav";
    std::string error;
    CHECK(spatial_audio::detail::writeMonoWav16(
        samplePath.string(), constantPcm(4410, 0.5f), error));

    std::vector<std::int16_t> block(spatial_audio::kBlockFrames * 2);

    spatial_audio::SourceConfig originConfig{};
    originConfig.loop = true;
    originConfig.initialPosition = {0.0f, 0.0f, 0.0f};
    originConfig.baseGain = 1.0f;
    originConfig.maxAudibleDistance = 10.0f;
    spatial_audio::SourceId originId = spatial_audio::kInvalidSourceId;
    auto origin = makeSourceEngine(samplePath.string(), originConfig, originId);
    origin->processBlock(block.data(), spatial_audio::kBlockFrames);
    spatial_audio::SourceStats originStats{};
    CHECK(origin->sourceStats(originId, originStats));
    CHECK(near(originStats.averageDistanceGain, 1.0f, 1.0e-5f));
    CHECK(near(originStats.averagePan, 0.0f, 1.0e-5f));
    CHECK(near(originStats.averageLeftGain, 0.70710678f, 1.0e-4f));
    CHECK(near(originStats.averageRightGain, 0.70710678f, 1.0e-4f));

    spatial_audio::SourceConfig distanceConfig = originConfig;
    distanceConfig.initialPosition = {5.0f, 0.0f, 0.0f};
    spatial_audio::SourceId distanceId = spatial_audio::kInvalidSourceId;
    auto distance =
        makeSourceEngine(samplePath.string(), distanceConfig, distanceId);
    distance->processBlock(block.data(), spatial_audio::kBlockFrames);
    spatial_audio::SourceStats distanceStats{};
    CHECK(distance->sourceStats(distanceId, distanceStats));
    CHECK(near(distanceStats.averageDistanceGain, 0.5f, 1.0e-5f));
    CHECK(near(distanceStats.averagePan, 1.0f, 1.0e-5f));

    spatial_audio::SourceConfig edgeConfig = originConfig;
    edgeConfig.initialPosition = {10.0f, 0.0f, 0.0f};
    spatial_audio::SourceId edgeId = spatial_audio::kInvalidSourceId;
    auto edge = makeSourceEngine(samplePath.string(), edgeConfig, edgeId);
    edge->processBlock(block.data(), spatial_audio::kBlockFrames);
    spatial_audio::SourceStats edgeStats{};
    CHECK(edge->sourceStats(edgeId, edgeStats));
    CHECK(near(edgeStats.averageDistanceGain, 0.0f, 1.0e-5f));
    CHECK(std::all_of(block.begin(), block.end(),
                      [](std::int16_t sample) { return sample == 0; }));

    std::printf("attenuation/equal-power panning: distance 0 -> 1.0, "
                "5/10 -> 0.5, 10/10 -> 0.0; center pan is equal power\n");
}

void testParameterSmoothing(const std::filesystem::path& root) {
    const auto samplePath = root / "smooth_constant.wav";
    std::string error;
    CHECK(spatial_audio::detail::writeMonoWav16(
        samplePath.string(), constantPcm(4410, 0.2f), error));

    spatial_audio::SourceConfig config{};
    config.initialPosition = {3.0f, 0.0f, 0.0f};
    config.baseGain = 1.0f;
    config.maxAudibleDistance = 10.0f;
    spatial_audio::SourceId id = spatial_audio::kInvalidSourceId;
    auto engine = makeSourceEngine(samplePath.string(), config, id);
    std::vector<std::int16_t> block(spatial_audio::kBlockFrames * 2);

    for (int i = 0; i < 3; ++i) {
        engine->processBlock(block.data(), spatial_audio::kBlockFrames);
    }
    const int previousRight = block[1];
    CHECK(previousRight > 4000);
    CHECK(block[0] <= 1);

    CHECK(engine->setSourcePosition(id, {-3.0f, 0.0f, 0.0f}));
    engine->processBlock(block.data(), spatial_audio::kBlockFrames);
    const int firstLeft = block[0];
    const int lastLeft = block[(spatial_audio::kBlockFrames - 1) * 2];
    const int firstRight = block[1];
    CHECK(firstLeft > 0);
    CHECK(firstLeft < lastLeft / 5);
    CHECK(lastLeft > 4000);
    CHECK(std::abs(firstRight - previousRight) < 300);
    CHECK(std::abs(firstLeft) <= 1200);

    std::printf("parameter smoothing: pan crosses left/right over one 256-frame "
                "block (%d -> %d left samples)\n",
                firstLeft, lastLeft);
}

void testOcclusionFilter(const std::filesystem::path& root) {
    const auto samplePath = root / "tone_8000.wav";
    std::string error;
    CHECK(spatial_audio::detail::writeMonoWav16(
        samplePath.string(), sinePcm(44100, 8000, 0.30f), error));

    spatial_audio::SourceConfig config{};
    config.initialPosition = {0.0f, 0.0f, 4.0f};
    config.baseGain = 1.0f;
    config.maxAudibleDistance = 10.0f;
    config.occludedCutoffHz = 500.0f;

    spatial_audio::SourceId clearId = spatial_audio::kInvalidSourceId;
    auto clear =
        makeSourceEngine(samplePath.string(), config, clearId);
    spatial_audio::SourceId blockedId = spatial_audio::kInvalidSourceId;
    auto blocked =
        makeSourceEngine(samplePath.string(), config, blockedId);
    std::vector<spatial_audio::Occluder> occluders(1);
    occluders[0].center = {0.0f, 0.0f, 2.0f};
    occluders[0].halfExtents = {4.0f, 3.0f, 0.10f};
    occluders[0].enabled = true;
    CHECK(blocked->setOccluders(occluders));

    std::vector<std::int16_t> block(spatial_audio::kBlockFrames * 2);
    std::vector<std::int16_t> clearTail;
    std::vector<std::int16_t> blockedTail;
    for (int i = 0; i < 12; ++i) {
        clear->processBlock(block.data(), spatial_audio::kBlockFrames);
        blocked->processBlock(block.data(), spatial_audio::kBlockFrames);
    }

    for (int i = 0; i < 2; ++i) {
        clear->processBlock(block.data(), spatial_audio::kBlockFrames);
        clearTail.insert(clearTail.end(), block.begin(), block.end());
        blocked->processBlock(block.data(), spatial_audio::kBlockFrames);
        blockedTail.insert(blockedTail.end(), block.begin(), block.end());
    }

    spatial_audio::SourceStats blockedStats{};
    CHECK(blocked->sourceStats(blockedId, blockedStats));
    CHECK(near(blockedStats.occlusionFraction, 1.0f, 1.0e-5f));
    CHECK(near(blockedStats.averageCutoffHz, 500.0f, 1.0e-3f));
    spatial_audio::SourceStats clearStats{};
    CHECK(clear->sourceStats(clearId, clearStats));
    CHECK(near(clearStats.occlusionFraction, 0.0f, 1.0e-5f));
    CHECK(clearStats.averageCutoffHz > 19000.0f);

    const float clearRms = outputRms(clearTail);
    const float blockedRms = outputRms(blockedTail);
    CHECK(clearRms > 0.04f);
    CHECK(blockedRms < clearRms * 0.25f);
    std::printf("occlusion filter: 8 kHz RMS %.4f (clear) -> %.4f "
                "(500 Hz low-pass), cutoff=500 Hz\n",
                clearRms, blockedRms);
}

void testManySourcesAndLimiter(const std::filesystem::path& root) {
    const auto samplePath = root / "loud_constant.wav";
    std::string error;
    CHECK(spatial_audio::detail::writeMonoWav16(
        samplePath.string(), constantPcm(4410, 0.95f), error));

    spatial_audio::AudioEngine engine;
    spatial_audio::SampleId sampleId = 0;
    CHECK(engine.loadWav(samplePath.string(), sampleId) ==
          spatial_audio::ErrorCode::Ok);
    spatial_audio::ListenerState listener{};
    listener.forward = {0.0f, 0.0f, 1.0f};
    listener.up = {0.0f, 1.0f, 0.0f};
    CHECK(engine.setListener(listener));

    std::vector<spatial_audio::SourceId> ids;
    for (std::size_t i = 0; i < spatial_audio::kMaxSources; ++i) {
        spatial_audio::SourceConfig config{};
        config.sample = sampleId;
        config.initialPosition = {0.0f, 0.0f, 0.0f};
        config.baseGain = 1.0f;
        config.maxAudibleDistance = 20.0f;
        spatial_audio::SourceId id = spatial_audio::kInvalidSourceId;
        CHECK(engine.addSource(config, id) == spatial_audio::ErrorCode::Ok);
        ids.push_back(id);
    }
    CHECK(ids.size() == spatial_audio::kMaxSources);
    spatial_audio::SourceConfig overflowConfig{};
    overflowConfig.sample = sampleId;
    overflowConfig.maxAudibleDistance = 20.0f;
    spatial_audio::SourceId overflowId = spatial_audio::kInvalidSourceId;
    CHECK(engine.addSource(overflowConfig, overflowId) ==
          spatial_audio::ErrorCode::NoFreeSourceSlot);

    std::vector<std::int16_t> block(spatial_audio::kBlockFrames * 2);
    std::int32_t peak = 0;
    for (int i = 0; i < 4; ++i) {
        engine.processBlock(block.data(), spatial_audio::kBlockFrames);
        for (std::int16_t sample : block) {
            peak = std::max(peak, std::abs(static_cast<std::int32_t>(sample)));
        }
    }
    CHECK(peak > 30000);
    CHECK(peak <= 32112);
    CHECK(std::all_of(block.begin(), block.end(), [](std::int16_t sample) {
        return sample >= -32112 && sample <= 32112;
    }));
    CHECK(engine.activeSourceCount() == spatial_audio::kMaxSources);

    std::printf("32-source mix: peak=%d (soft-limiter ceiling=32112, "
                "no integer hard clip)\n",
                peak);
}

void testLifecycleAndCallbackAllocations(const std::filesystem::path& root) {
    const auto samplePath = root / "lifecycle.wav";
    std::string error;
    CHECK(spatial_audio::detail::writeMonoWav16(
        samplePath.string(), constantPcm(4410, 0.2f), error));

    spatial_audio::AudioEngine engine;
    spatial_audio::SampleId sampleId = 0;
    CHECK(engine.loadWav(samplePath.string(), sampleId) ==
          spatial_audio::ErrorCode::Ok);
    std::vector<spatial_audio::SourceId> ids;
    for (int i = 0; i < 8; ++i) {
        spatial_audio::SourceConfig config{};
        config.sample = sampleId;
        config.initialPosition = {static_cast<float>(i), 0.0f, 2.0f};
        config.baseGain = 0.05f;
        config.maxAudibleDistance = 20.0f;
        spatial_audio::SourceId id = spatial_audio::kInvalidSourceId;
        CHECK(engine.addSource(config, id) == spatial_audio::ErrorCode::Ok);
        ids.push_back(id);
    }

    std::vector<std::int16_t> block(spatial_audio::kBlockFrames * 2);
    engine.processBlock(block.data(), spatial_audio::kBlockFrames);
    CHECK(engine.activeSourceCount() == 8);

    CHECK(engine.removeSource(ids[0]));
    CHECK(engine.activeSourceCount() == 8);
    engine.processBlock(block.data(), spatial_audio::kBlockFrames);
    CHECK(engine.activeSourceCount() == 7);
    CHECK(!engine.setSourcePosition(ids[0], {0.0f, 0.0f, 0.0f}));

    spatial_audio::SourceConfig replacement{};
    replacement.sample = sampleId;
    replacement.initialPosition = {0.0f, 0.0f, 1.0f};
    replacement.baseGain = 0.05f;
    replacement.maxAudibleDistance = 20.0f;
    spatial_audio::SourceId newId = spatial_audio::kInvalidSourceId;
    CHECK(engine.addSource(replacement, newId) ==
          spatial_audio::ErrorCode::Ok);
    CHECK(newId != ids[0]);
    engine.processBlock(block.data(), spatial_audio::kBlockFrames);
    CHECK(engine.activeSourceCount() == 8);

    spatial_audio::SourceId cancelled = spatial_audio::kInvalidSourceId;
    CHECK(engine.addSource(replacement, cancelled) ==
          spatial_audio::ErrorCode::Ok);
    CHECK(engine.removeSource(cancelled));
    engine.processBlock(block.data(), spatial_audio::kBlockFrames);
    CHECK(engine.activeSourceCount() == 8);
    spatial_audio::SourceStats cancelledStats{};
    CHECK(engine.sourceStats(cancelled, cancelledStats));
    CHECK(cancelledStats.processedBlocks == 1);

    const std::size_t before = g_allocations.load(std::memory_order_relaxed);
    engine.processBlock(block.data(), spatial_audio::kBlockFrames);
    const std::size_t after = g_allocations.load(std::memory_order_relaxed);
    CHECK(after == before);

    std::printf("lifecycle: deferred remove at block boundary, slot reuse safe, "
                "audio callback allocations=%zu\n",
                after - before);
}

void testConcurrentControlAndAudio(const std::filesystem::path& root) {
    const auto samplePath = root / "threaded.wav";
    std::string error;
    CHECK(spatial_audio::detail::writeMonoWav16(
        samplePath.string(), constantPcm(4410, 0.1f), error));

    spatial_audio::AudioEngine engine;
    spatial_audio::SampleId sampleId = 0;
    CHECK(engine.loadWav(samplePath.string(), sampleId) ==
          spatial_audio::ErrorCode::Ok);
    spatial_audio::ListenerState listener{};
    listener.forward = {0.0f, 0.0f, 1.0f};
    listener.up = {0.0f, 1.0f, 0.0f};
    CHECK(engine.setListener(listener));

    std::vector<spatial_audio::SourceId> ids;
    for (int i = 0; i < 8; ++i) {
        spatial_audio::SourceConfig config{};
        config.sample = sampleId;
        config.initialPosition = {static_cast<float>(i), 0.0f, 2.0f};
        config.baseGain = 0.04f;
        config.maxAudibleDistance = 20.0f;
        spatial_audio::SourceId id = spatial_audio::kInvalidSourceId;
        CHECK(engine.addSource(config, id) == spatial_audio::ErrorCode::Ok);
        ids.push_back(id);
    }

    std::atomic<bool> stop{false};
    std::thread audioThread([&]() {
        std::vector<std::int16_t> block(spatial_audio::kBlockFrames * 2);
        while (!stop.load(std::memory_order_acquire)) {
            engine.processBlock(block.data(), spatial_audio::kBlockFrames);
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    for (int swap = 0; swap < 12; ++swap) {
        spatial_audio::SourceId removedId = ids.front();
        CHECK(engine.removeSource(removedId));
        for (int wait = 0; wait < 200 && engine.activeSourceCount() != 7; ++wait) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CHECK(engine.activeSourceCount() == 7);

        spatial_audio::SourceConfig config{};
        config.sample = sampleId;
        config.initialPosition = {static_cast<float>(swap) * 0.4f, 0.0f, 3.0f};
        config.baseGain = 0.04f;
        config.maxAudibleDistance = 20.0f;
        spatial_audio::SourceId replacement = spatial_audio::kInvalidSourceId;
        CHECK(engine.addSource(config, replacement) ==
              spatial_audio::ErrorCode::Ok);
        CHECK(engine.setSourcePosition(replacement,
                                       {-1.0f, 0.0f, 2.0f}));
        ids.front() = replacement;
        for (int wait = 0; wait < 200 && engine.activeSourceCount() != 8; ++wait) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CHECK(engine.activeSourceCount() == 8);

        listener.position = {
            static_cast<float>((swap % 3) - 1),
            0.0f,
            0.2f * static_cast<float>(swap % 2),
        };
        listener.forward = {
            std::sin(0.3f * static_cast<float>(swap)), 0.0f,
            std::cos(0.3f * static_cast<float>(swap)),
        };
        CHECK(engine.setListener(listener));
        for (size_t i = 1; i < ids.size(); ++i) {
            CHECK(engine.setSourcePosition(
                ids[i], {static_cast<float>(i) + 0.1f * static_cast<float>(swap),
                         0.0f, 2.0f}));
        }
    }

    const std::size_t before = g_allocations.load(std::memory_order_relaxed);
    stop.store(true, std::memory_order_release);
    audioThread.join();
    const std::size_t after = g_allocations.load(std::memory_order_relaxed);
    CHECK(after == before);
    CHECK(engine.activeSourceCount() == 8);
    std::printf("threaded stress: 12 add/remove swaps while audio callback ran\n");
}

}  // namespace

void* operator new(std::size_t size) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* pointer = std::malloc(size)) {
        return pointer;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* pointer = std::malloc(size)) {
        return pointer;
    }
    throw std::bad_alloc();
}

void* operator new(std::size_t size, std::align_val_t alignment) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    void* pointer = nullptr;
#if defined(_MSC_VER)
    pointer = _aligned_malloc(size, static_cast<std::size_t>(alignment));
#else
    if (posix_memalign(&pointer, static_cast<std::size_t>(alignment), size) !=
        0) {
        pointer = nullptr;
    }
#endif
    if (pointer != nullptr) {
        return pointer;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
    return operator new(size, alignment);
}

void operator delete(void* pointer) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

void operator delete(void* pointer, std::align_val_t) noexcept {
#if defined(_MSC_VER)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}

void operator delete[](void* pointer, std::align_val_t alignment) noexcept {
    operator delete(pointer, alignment);
}

void operator delete(void* pointer, std::size_t, std::align_val_t alignment) noexcept {
    operator delete(pointer, alignment);
}

void operator delete[](void* pointer,
                       std::size_t,
                       std::align_val_t alignment) noexcept {
    operator delete(pointer, alignment);
}

int main() {
    const std::filesystem::path root =
        std::filesystem::current_path() / "audio_engine_test_artifacts";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directories(root, error);
    if (error) {
        std::fprintf(stderr, "cannot create test directory: %s\n",
                     error.message().c_str());
        return 1;
    }

    testWavErrors(root);
    testAttenuationAndPanning(root);
    testParameterSmoothing(root);
    testOcclusionFilter(root);
    testManySourcesAndLimiter(root);
    testLifecycleAndCallbackAllocations(root);
    testConcurrentControlAndAudio(root);

    if (failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "%d TEST(S) FAILED\n", failures);
    return 1;
}
