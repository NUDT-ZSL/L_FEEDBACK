#include "lightcluster/animation.hpp"
#include "lightcluster/buffer_sync.hpp"
#include "lightcluster/deterministic_engine.hpp"
#include "lightcluster/json_log.hpp"
#include "lightcluster/scene.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void check(bool condition, const char* expression, int line) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "CHECK FAILED line " << line << ": " << expression << '\n';
    }
}

template <typename A, typename B>
void checkEqual(
    const A& actual,
    const B& expected,
    const char* expression,
    int line) {
    ++checks;
    if (!(actual == expected)) {
        ++failures;
        std::cerr << "CHECK FAILED line " << line << ": " << expression
                  << " (actual=" << actual << ", expected=" << expected
                  << ")\n";
    }
}

void checkNear(
    float actual,
    float expected,
    float tolerance,
    const char* expression,
    int line) {
    ++checks;
    if (std::fabs(actual - expected) > tolerance) {
        ++failures;
        std::cerr << "CHECK FAILED line " << line << ": " << expression
                  << " (actual=" << actual << ", expected=" << expected
                  << ")\n";
    }
}

#define CHECK(condition) check((condition), #condition, __LINE__)
#define CHECK_EQ(actual, expected) \
    checkEqual((actual), (expected), #actual " == " #expected, __LINE__)
#define CHECK_NEAR(actual, expected, tolerance) \
    checkNear((actual), (expected), (tolerance), #actual " ~= " #expected, \
              __LINE__)

using namespace lightcluster;

Light pointLight(std::uint32_t id = 0) {
    Light light;
    light.id = id;
    light.type = LightType::Point;
    light.position = {1.0F, 2.0F, 3.0F};
    light.color = {0.4F, 0.6F, 0.8F};
    light.intensity = 1.5F;
    light.range = 8.0F;
    return light;
}

Light directionalLight(std::uint32_t id = 0) {
    Light light;
    light.id = id;
    light.type = LightType::Directional;
    light.direction = {0.0F, -1.0F, 0.0F};
    light.color = {1.0F, 0.9F, 0.8F};
    light.intensity = 0.2F;
    return light;
}

std::vector<Light> threeLights() {
    return {pointLight(), pointLight(), directionalLight()};
}

std::uint32_t readU32(const std::vector<std::uint8_t>& bytes, std::size_t at) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes.data() + at, sizeof(value));
    return value;
}

void writeU32(
    std::vector<std::uint8_t>& bytes,
    std::size_t at,
    std::uint32_t value) {
    std::memcpy(bytes.data() + at, &value, sizeof(value));
}

void testSceneMutationsAndSnapshots() {
    SceneManager scene(threeLights());
    CHECK_EQ(scene.version(), 1ULL);
    CHECK_EQ(scene.snapshot().lights.size(), std::size_t{3});
    const std::uint32_t id = scene.addLight(pointLight());
    CHECK_EQ(id, 4U);
    CHECK_EQ(scene.version(), 2ULL);
    CHECK(scene.updateLight(id, [](Light& light) { light.intensity = 9.0F; }));
    CHECK_EQ(scene.version(), 3ULL);
    CHECK(scene.removeLight(id));
    CHECK_EQ(scene.version(), 4ULL);

    const SceneSnapshot saved = scene.saveSnapshot();
    CHECK_EQ(saved.version, 4ULL);
    CHECK_EQ(saved.lights.size(), std::size_t{3});
    CHECK(scene.addLight(pointLight()) != 0);
    CHECK_EQ(scene.version(), 5ULL);
    CHECK_EQ(saved.lights.size(), std::size_t{3});

    const std::uint64_t beforeRestore = scene.version();
    const std::uint64_t restoredVersion = scene.restore(saved);
    CHECK(restoredVersion > beforeRestore);
    CHECK(restoredVersion > saved.version);
    CHECK_EQ(scene.version(), restoredVersion);
    CHECK_EQ(scene.snapshot().lights.size(), saved.lights.size());
    CHECK(scene.updateLight(1, [](Light& light) { light.range = 4.0F; }));
    CHECK_EQ(scene.version(), restoredVersion + 1);

    SceneSnapshot futureVersion = saved;
    futureVersion.version = 500;
    const std::uint64_t afterFutureRestore = scene.restore(futureVersion);
    CHECK_EQ(afterFutureRestore, 501ULL);
    CHECK(afterFutureRestore > futureVersion.version);

    LightBufferSynchronizer sync(SyncMode::Immediate, 0);
    const SceneSnapshot beforeUpload = scene.snapshot();
    (void)sync.initialize(beforeUpload, 0);
    SceneSnapshot restoredAgain = beforeUpload;
    restoredAgain.version = 900;
    const std::uint64_t restoredLiveVersion = scene.restore(restoredAgain);
    const SyncOutcome afterRestoreSync = sync.sync(scene.snapshot(), 1);
    CHECK(afterRestoreSync.uploaded);
    CHECK_EQ(afterRestoreSync.gpuVersion, restoredLiveVersion);
}

void testConcurrentSnapshotConsistency() {
    SceneManager scene({pointLight(1)});
    std::atomic<bool> running{true};
    std::atomic<int> observed{0};
    std::atomic<bool> failed{false};

    std::thread reader([&] {
        while (running.load(std::memory_order_acquire)) {
            const SceneSnapshot snapshot = scene.snapshot();
            if (snapshot.lights.size() != 1) {
                failed.store(true, std::memory_order_relaxed);
                break;
            }
            const float expectedCounter =
                static_cast<float>(snapshot.version - 1);
            if (std::fabs(
                    snapshot.lights[0].position.x - expectedCounter) >
                1.0e-4F) {
                failed.store(true, std::memory_order_relaxed);
                break;
            }
            observed.fetch_add(1, std::memory_order_relaxed);
        }
    });

    for (std::uint32_t i = 1; i <= 2000; ++i) {
        (void)scene.updateLight(1, [i](Light& light) {
            light.position.x = static_cast<float>(i);
        });
    }
    running.store(false, std::memory_order_release);
    reader.join();
    CHECK(!failed.load());
    CHECK(observed.load() > 0);
    CHECK_EQ(scene.version(), 2001ULL);
}

void testAnimationInterpolationAndPoolFull() {
    SceneManager scene({pointLight(1)});
    AnimationPool pool(1);

    AnimationRequest request;
    request.lightId = 1;
    request.durationFrames = 4;
    request.targetPosition = {4.0F, 6.0F, 8.0F};
    request.targetIntensity = 2.0F;
    request.animatePosition = true;
    request.animateIntensity = true;
    const AnimationQueueResult first = pool.queue(scene, request, 10);
    CHECK(first.accepted);
    CHECK_EQ(first.endFrame, 14U);
    CHECK_EQ(scene.version(), 1ULL);

    const AnimationQueueResult rejected = pool.queue(scene, request, 10);
    CHECK(!rejected.accepted);
    CHECK(rejected.code == AnimationRejectCode::PoolFull);
    CHECK_EQ(
        std::string(toString(rejected.code)),
        std::string("ANIM_POOL_FULL"));

    AnimationFrameResult frame = pool.applyFrame(scene, 11);
    CHECK_EQ(frame.applied, 1U);
    CHECK_EQ(frame.active, 1U);
    auto light = scene.findLight(1);
    CHECK(light.has_value());
    CHECK_NEAR(light->position.x, 1.75F, 1.0e-6F);
    CHECK_NEAR(light->intensity, 1.625F, 1.0e-6F);

    frame = pool.applyFrame(scene, 12);
    light = scene.findLight(1);
    CHECK_NEAR(light->position.x, 2.5F, 1.0e-6F);
    CHECK_NEAR(light->intensity, 1.75F, 1.0e-6F);

    frame = pool.applyFrame(scene, 14);
    CHECK_EQ(frame.completed, 1U);
    CHECK_EQ(frame.active, 0U);
    light = scene.findLight(1);
    CHECK_NEAR(light->position.x, 4.0F, 1.0e-6F);
    CHECK_NEAR(light->intensity, 2.0F, 1.0e-6F);
}

void testAnimationPositionOnly() {
    SceneManager scene({pointLight(1)});
    AnimationPool pool(1);
    AnimationRequest request;
    request.lightId = 1;
    request.durationFrames = 2;
    request.targetPosition = {5.0F, 5.0F, 5.0F};
    request.targetIntensity = 99.0F;
    request.animatePosition = true;
    request.animateIntensity = false;
    CHECK(pool.queue(scene, request, 0).accepted);
    (void)pool.applyFrame(scene, 2);
    const auto light = scene.findLight(1);
    CHECK(light.has_value());
    CHECK_NEAR(light->position.x, 5.0F, 1.0e-6F);
    CHECK_NEAR(light->intensity, 1.5F, 1.0e-6F);
}

void testImmediateBufferReuseAndGrowth() {
    SceneManager scene(threeLights());
    LightBufferSynchronizer sync(SyncMode::Immediate, 0);
    const SceneSnapshot initial = scene.snapshot();
    const SyncOutcome first = sync.initialize(initial, 0);
    CHECK(first.uploaded);
    CHECK_EQ(first.capacityLights, kMinimumBufferCapacity);
    CHECK_EQ(sync.residentImage().lightCount, 3U);
    CHECK_EQ(sync.residentImage().pointCount, 2U);
    CHECK_EQ(sync.residentImage().directionalCount, 1U);
    CHECK_EQ(sync.residentImage().usedBytes, 32U + 3U * 48U);

    const SyncOutcome unchanged = sync.sync(scene.snapshot(), 1);
    CHECK(!unchanged.uploaded);
    CHECK_EQ(std::string(unchanged.reason), std::string("version_unchanged"));

    (void)scene.updateLight(
        1, [](Light& light) { light.intensity = 3.0F; });
    const SyncOutcome updated = sync.sync(scene.snapshot(), 2);
    CHECK(updated.uploaded);
    CHECK(!updated.grew);
    CHECK_EQ(updated.gpuVersion, scene.version());

    while (scene.lightCount() <= kMinimumBufferCapacity) {
        (void)scene.addLight(pointLight());
    }
    const SyncOutcome grown = sync.sync(scene.snapshot(), 3);
    CHECK(grown.uploaded);
    CHECK(grown.grew);
    CHECK_EQ(grown.capacityLights, kMinimumBufferCapacity * 2);
    CHECK_EQ(grown.gpuLightCount, scene.lightCount());
}

void testDeferredConvergence() {
    SceneSnapshot first;
    first.version = 1;
    first.lights = threeLights();
    LightBufferSynchronizer sync(SyncMode::Deferred, 3);
    (void)sync.initialize(first, 0);

    SceneSnapshot second = first;
    second.version = 2;
    second.lights[0].intensity = 2.0F;
    const SyncOutcome atZero = sync.sync(second, 0);
    CHECK(!atZero.uploaded);
    CHECK_EQ(atZero.pendingSnapshots, 1U);

    const SyncOutcome atTwo = sync.sync(second, 2);
    CHECK(!atTwo.uploaded);
    CHECK_EQ(atTwo.pendingSnapshots, 1U);

    const SyncOutcome atThree = sync.sync(second, 3);
    CHECK(atThree.uploaded);
    CHECK_EQ(atThree.gpuVersion, 2ULL);
    CHECK_EQ(atThree.sourceFrame, 0U);
    CHECK_EQ(atThree.snapshotAgeFrames, 3U);

    const SyncOutcome atFour = sync.sync(second, 4);
    CHECK(!atFour.uploaded);
    CHECK_EQ(atFour.gpuVersion, 2ULL);
    CHECK_EQ(atFour.frontendVersion, 2ULL);
}

void testDeferredLatestOfMultipleSnapshots() {
    SceneSnapshot one;
    one.version = 1;
    one.lights = threeLights();
    LightBufferSynchronizer sync(SyncMode::Deferred, 3);
    (void)sync.initialize(one, 0);
    SceneSnapshot two = one;
    two.version = 2;
    SceneSnapshot three = one;
    three.version = 3;
    (void)sync.sync(two, 0);
    (void)sync.sync(three, 1);
    const SyncOutcome dueTwo = sync.sync(three, 3);
    CHECK(dueTwo.uploaded);
    CHECK_EQ(dueTwo.gpuVersion, 2ULL);
    CHECK_EQ(dueTwo.sourceFrame, 0U);
    const SyncOutcome dueThree = sync.sync(three, 4);
    CHECK(dueThree.uploaded);
    CHECK_EQ(dueThree.gpuVersion, 3ULL);
    CHECK_EQ(dueThree.sourceFrame, 1U);
}

void testDebugReadbackValidation() {
    SceneManager scene(threeLights());
    LightBufferSynchronizer sync(SyncMode::Immediate, 0);
    (void)sync.initialize(scene.snapshot(), 0);
    const GpuBufferImage& expected = sync.residentImage();

    BufferValidation valid = validateReadback(expected.bytes, expected);
    CHECK(valid.ok);
    CHECK_EQ(std::string(valid.code), std::string("OK"));

    auto corrupt = expected.bytes;
    writeU32(corrupt, 0, readU32(corrupt, 0) + 1);
    BufferValidation count = validateReadback(corrupt, expected);
    CHECK(!count.ok);
    CHECK_EQ(std::string(count.code), std::string("GPU_COUNT_MISMATCH"));

    corrupt = expected.bytes;
    const std::uint32_t point = readU32(corrupt, 4);
    writeU32(corrupt, 4, point + 1);
    BufferValidation types = validateReadback(corrupt, expected);
    CHECK(!types.ok);
    CHECK_EQ(
        std::string(types.code),
        std::string("GPU_TYPE_COUNT_MISMATCH"));

    corrupt = expected.bytes;
    writeU32(corrupt, 20, readU32(corrupt, 20) + 1);
    BufferValidation version = validateReadback(corrupt, expected);
    CHECK(!version.ok);
    CHECK_EQ(
        std::string(version.code),
        std::string("GPU_VERSION_MISMATCH"));

    corrupt = expected.bytes;
    corrupt[kGpuHeaderBytes + 3] ^= 0x01U;
    BufferValidation payload = validateReadback(corrupt, expected);
    CHECK(!payload.ok);
    CHECK_EQ(
        std::string(payload.code),
        std::string("GPU_LIGHT_DATA_MISMATCH"));
}

RunConfig deterministicConfig(SyncMode mode, std::uint32_t delay) {
    RunConfig config;
    config.frames = 48;
    config.syncMode = mode;
    config.delayFrames = delay;
    config.animationPoolCapacity = 4;
    config.seed = 42;
    return config;
}

void testDeterministicSequenceInvariants() {
    RunConfig config = deterministicConfig(SyncMode::Deferred, 4);
    DeterministicEngine engine(config);
    bool rejectedAny = false;
    FrameTick lastTick;
    for (std::uint32_t frame = 0; frame < config.frames; ++frame) {
        const FrameTick tick = engine.tick(frame);
        const LightCounts counts = tick.frontend.counts();
        CHECK(counts.total >= 64);
        CHECK(counts.point >= 64);
        CHECK(counts.directional >= 4);
        CHECK_EQ(counts.total, counts.point + counts.directional);
        CHECK(tick.frontend.version >= 1);
        CHECK(!tick.frontend.lights.empty());
        for (const RunEvent& event : tick.events) {
            if (event.kind == RunEventKind::AnimationRejected &&
                event.businessCode == "ANIM_POOL_FULL") {
                rejectedAny = true;
            }
        }
        if (frame >= config.delayFrames) {
            CHECK(tick.sync.snapshotAgeFrames <= config.delayFrames);
        }
        lastTick = tick;
    }
    CHECK(rejectedAny);
    CHECK_EQ(lastTick.sync.gpuVersion, lastTick.frontend.version);
}

void testDeterminismAndImmediateSkip() {
    RunConfig config = deterministicConfig(SyncMode::Immediate, 0);
    DeterministicEngine firstEngine(config);
    DeterministicEngine secondEngine(config);
    for (std::uint32_t frame = 0; frame < config.frames; ++frame) {
        const FrameTick a = firstEngine.tick(frame);
        const FrameTick b = secondEngine.tick(frame);
        CHECK_EQ(a.frontend.version, b.frontend.version);
        CHECK_EQ(a.frontend.lights.size(), b.frontend.lights.size());
        CHECK_EQ(a.sync.uploaded, b.sync.uploaded);
        CHECK_EQ(a.animations.applied, b.animations.applied);
        CHECK_EQ(a.events.size(), b.events.size());
        if (frame == 0) {
            CHECK(!a.sync.uploaded);
            CHECK_EQ(a.frontend.version, 1ULL);
        }
    }
}

void testSnapshotSaveRestoreInEngine() {
    RunConfig config = deterministicConfig(SyncMode::Immediate, 0);
    config.saveSnapshotFrame = 8;
    config.restoreSnapshotFrame = 20;
    config.frames = 48;
    DeterministicEngine engine(config);
    bool saved = false;
    bool restored = false;
    for (std::uint32_t frame = 0; frame < config.frames; ++frame) {
        const FrameTick tick = engine.tick(frame);
        for (const RunEvent& event : tick.events) {
            if (event.kind == RunEventKind::SnapshotSaved) {
                saved = true;
                CHECK_EQ(event.versionAfter, event.restoredSnapshotVersion);
            }
            if (event.kind == RunEventKind::SnapshotRestored) {
                restored = true;
                CHECK(event.versionAfter > event.versionBefore);
                CHECK(event.versionAfter > event.restoredSnapshotVersion);
                CHECK(event.activeLights >= 64);
            }
        }
    }
    CHECK(saved);
    CHECK(restored);
}

void testJsonFields() {
    SceneManager scene(threeLights());
    LightBufferSynchronizer sync(SyncMode::Immediate, 0);
    (void)sync.initialize(scene.snapshot(), 0);
    FrameTick tick;
    tick.frame = 0;
    tick.frontend = scene.snapshot();
    tick.sync = sync.sync(tick.frontend, 0);
    DebugFrameStatus debug;
    debug.checked = true;
    debug.readbackLights = 3;
    debug.readbackPoint = 2;
    debug.readbackDirectional = 1;
    const std::string line = formatFrameJson(tick, RunConfig{}, debug);
    CHECK(line.find("\"event\":\"frame\"") != std::string::npos);
    CHECK(line.find("\"scene_version\":1") != std::string::npos);
    CHECK(line.find("\"uploaded\":false") != std::string::npos);
    CHECK(line.find("\"debug_ok\":true") != std::string::npos);
}

}  // namespace

int main() {
    testSceneMutationsAndSnapshots();
    testConcurrentSnapshotConsistency();
    testAnimationInterpolationAndPoolFull();
    testAnimationPositionOnly();
    testImmediateBufferReuseAndGrowth();
    testDeferredConvergence();
    testDeferredLatestOfMultipleSnapshots();
    testDebugReadbackValidation();
    testDeterministicSequenceInvariants();
    testDeterminismAndImmediateSkip();
    testSnapshotSaveRestoreInEngine();
    testJsonFields();

    if (failures != 0) {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "All " << checks << " checks passed\n";
    return 0;
}
