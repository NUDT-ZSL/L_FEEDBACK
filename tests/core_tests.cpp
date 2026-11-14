#include "indoor/RenderTypes.hpp"
#include "indoor/ResourceManager.hpp"
#include "indoor/Scene.hpp"
#include "indoor/ShadowSystems.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

int failures = 0;

void require(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void section(const char* name) {
    std::cout << "[core] " << name << '\n';
}

void testDeterminism() {
    section("deterministic scene construction");
    const indoor::Scene a = indoor::buildDeterministicScene(1337, 300);
    const indoor::Scene b = indoor::buildDeterministicScene(1337, 300);
    const indoor::Scene c = indoor::buildDeterministicScene(42, 300);

    require(indoor::sceneDigest(a) == indoor::sceneDigest(b),
            "same seed/light count must produce identical scene data");
    require(indoor::sceneDigest(a) != indoor::sceneDigest(c),
            "different seed must change scene data");
    require(a.lights.size() == 300 && b.lights.size() == 300,
            "requested light count must be honored");
    require(a.objects.size() == b.objects.size() && a.objects.size() >= 140,
            "indoor scene must contain the deterministic object population");
    require(a.lights[17].position == b.lights[17].position &&
                a.lights[17].color == b.lights[17].color &&
                a.lights[17].intensity == b.lights[17].intensity,
            "representative light fields must match exactly");
    std::cout << "  digest=" << indoor::sceneDigestHex(a)
              << " lights=" << a.lights.size()
              << " objects=" << a.objects.size() << '\n';
}

void testMaterialPresets() {
    section("five material-layer presets");
    indoor::MaterialPresetId preset = indoor::MaterialPresetId::BaseOnly;
    std::vector<std::string> names;
    for (int i = 0; i < 5; ++i) {
        names.emplace_back(indoor::materialPresetName(preset));
        preset = indoor::nextMaterialPreset(preset);
    }
    require(static_cast<int>(preset) == 0,
            "five preset cycle must return to the first preset");
    require(names.front() == "0 BaseOnly" && names.back() == "4 FullLayered",
            "preset labels must expose all layer combinations");
    for (const std::string& name : names) {
        std::cout << "  " << name << '\n';
    }
}

std::unordered_map<std::uint32_t, std::uint64_t> snapshotVersions(
    const indoor::Scene& scene, const indoor::ShadowVersionLedger& ledger) {
    std::unordered_map<std::uint32_t, std::uint64_t> result;
    for (const indoor::Light& light : scene.lights) {
        result[light.id] = ledger.version(light.id);
    }
    return result;
}

void testIncrementalShadowInvalidation() {
    section("incremental shadow invalidation and version continuity");
    indoor::Scene scene = indoor::buildDeterministicScene(1337, 256);
    indoor::ShadowVersionLedger ledger;
    ledger.initialize(scene.lights);
    const auto before = snapshotVersions(scene, ledger);

    const std::vector<indoor::MovedObject> moved =
        indoor::advanceAnimation(scene, 1.0 / 30.0);
    const std::vector<std::uint32_t> dirty =
        indoor::findAffectedLights(scene.lights, moved);
    require(!moved.empty(), "dynamic objects must move on animation update");
    require(!dirty.empty(), "moving objects must invalidate at least one light");
    require(dirty.size() < scene.lights.size(),
            "only a local subset of lights may be invalidated");
    require(dirty.size() <= moved.size() * 20,
            "dirty-light sweep should remain bounded for local movement");

    std::unordered_map<std::uint32_t, std::uint64_t> afterBump;
    std::unordered_map<std::uint32_t, std::uint64_t> bumpValues;
    const std::uint64_t migrationsBefore = ledger.migrations();
    for (std::uint32_t lightId : dirty) {
        bumpValues[lightId] = ledger.bump(lightId);
    }
    for (const indoor::Light& light : scene.lights) {
        const std::uint64_t previous = before.at(light.id);
        const std::uint64_t current = ledger.version(light.id);
        if (bumpValues.find(light.id) != bumpValues.end()) {
            require(current == previous + 1,
                    "dirty light version must advance by exactly one");
        } else {
            require(current == previous,
                    "unaffected light version must remain byte-for-byte unchanged");
        }
    }
    require(ledger.migrations() == migrationsBefore + dirty.size(),
            "migration counter must equal the number of dirty maps");
    std::cout << "  movedObjects=" << moved.size()
              << " dirtyLights=" << dirty.size()
              << " totalLights=" << scene.lights.size()
              << " migrations=" << ledger.migrations() << '\n';

    const std::uint64_t singleVersion = ledger.version(dirty.front());
    ledger.bump(dirty.front());
    require(ledger.version(dirty.front()) == singleVersion + 1,
            "repeated invalidation must continue monotonically");
}

void testLruAndLifecycle() {
    section("shadow cache LRU, duplicate prevention and release");
    indoor::ShadowCachePolicy cache(3);
    auto a = cache.acquire(101);
    auto b = cache.acquire(202);
    auto c = cache.acquire(303);
    require(a.created && b.created && c.created, "first acquisitions allocate");
    require(cache.size() == 3, "cache must reach requested capacity");

    auto duplicate = cache.acquire(101);
    require(!duplicate.created && !duplicate.evictedLightId.has_value(),
            "the same light must never allocate a duplicate shadow slot");

    cache.setVersion(101, 7);
    auto d = cache.acquire(404);
    require(d.created, "fourth distinct light must allocate after eviction");
    require(d.evictedLightId.has_value() && *d.evictedLightId == 202,
            "least-recently-used entry must be evicted");
    require(cache.size() == 3 && !cache.contains(202),
            "cache size must stay at capacity after eviction");
    require(cache.contains(101) && cache.version(101) == 7,
            "recently used entry and its version must survive");

    require(cache.release(303), "resident light must be releasable");
    require(!cache.contains(303) && cache.size() == 2,
            "release must immediately free the cache slot");
    require(!cache.release(999), "release of absent light must be harmless");

    cache.clear();
    require(cache.size() == 0, "scene switch must free all logical slots");
    std::cout << "  capacity=" << cache.capacity()
              << " evicted=202 residentAfterRelease=2\n";
}

void testVersionLedgerLifecycle() {
    section("version ledger and light deletion");
    indoor::Scene scene = indoor::buildDeterministicScene(1337, 64);
    indoor::ShadowVersionLedger ledger;
    ledger.initialize(scene.lights);
    require(ledger.trackedLights() == scene.lights.size(),
            "ledger must track every source light");

    const std::uint32_t id = scene.lights.back().id;
    const std::uint64_t version = ledger.bump(id);
    require(version == 2, "initial version is 1 and increments to 2");
    require(indoor::removeLight(scene, id), "existing light must be removable");
    ledger.remove(id);
    require(ledger.version(id) == 0 && ledger.trackedLights() == 63,
            "deleted light must leave no version or cache identity behind");
    require(ledger.migrations() >= 1,
            "migration history may survive deletion for diagnostics");
}

void testFrameBoundaryResourceQueue() {
    section("thread-safe resource queue frame fence");
    indoor::ResourceManager manager;
    indoor::ResourceRequest request;
    request.kind = indoor::ResourceRequestKind::DeleteLight;
    request.lightId = 9001;
    const std::uint64_t sequence = manager.submit(request);
    require(manager.waitForProcessed(sequence),
            "worker must process a submitted request");

    auto sameFrame = manager.pollAtFrameBoundary(0);
    require(sameFrame.empty(),
            "a command must not become visible in the frame it was submitted");
    auto nextFrame = manager.pollAtFrameBoundary(1);
    require(nextFrame.size() == 1 && nextFrame[0].kind == request.kind &&
                nextFrame[0].lightId == 9001 && nextFrame[0].applyAtFrame == 1,
            "command must become visible exactly at the next frame boundary");

    manager.setCurrentFrame(5);
    manager.submit(request);
    require(manager.waitForProcessed(sequence + 1),
            "second request must be processed");
    require(manager.pollAtFrameBoundary(5).empty(),
            "request at frame 5 cannot migrate during frame 5");
    auto later = manager.pollAtFrameBoundary(6);
    require(later.size() == 1 && later[0].applyAtFrame == 6,
            "request at frame 5 must migrate at frame 6");
    std::cout << "  frame0=0 commands frame1=1 frame5=0 frame6=1\n";
}

void testPixelComparison() {
    section("deterministic pixel digest helper");
    const std::vector<std::uint8_t> a = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<std::uint8_t> b = {1, 2, 3, 4, 9, 9, 9, 9};
    require(indoor::pixelDigest(a) == indoor::pixelDigest(a),
            "identical pixels must hash identically");
    require(indoor::countPixelDifferences(a, a) == 0,
            "same image must report zero pixel differences");
    require(indoor::countPixelDifferences(a, b) == 1,
            "one changed RGBA pixel must be counted exactly once");
}

} // namespace

int main() {
    try {
        testDeterminism();
        testMaterialPresets();
        testIncrementalShadowInvalidation();
        testLruAndLifecycle();
        testVersionLedgerLifecycle();
        testFrameBoundaryResourceQueue();
        testPixelComparison();
    } catch (const std::exception& error) {
        std::cerr << "FATAL core test exception: " << error.what() << '\n';
        return 2;
    }

    if (failures != 0) {
        std::cerr << failures << " core assertion(s) failed\n";
        return 1;
    }
    std::cout << "All core tests passed.\n";
    return 0;
}
