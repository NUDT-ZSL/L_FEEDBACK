#include "core/adaptive.hpp"
#include "core/animation.hpp"
#include "core/benchmark.hpp"
#include "core/bvh.hpp"
#include "core/command_recorder.hpp"
#include "core/culling.hpp"
#include "core/error.hpp"
#include "core/json.hpp"
#include "core/scene.hpp"
#include "core/shadows.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "[FAIL] " << message << '\n';
    }
}

template <typename Fn>
void test(const char* name, Fn&& fn) {
    try {
        fn();
        std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "[FAIL] " << name << " threw: " << error.what() << '\n';
    }
}

std::filesystem::path uniqueTempDirectory() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("uid166_core_tests_" + std::to_string(stamp));
}

std::string fileBytes(const std::filesystem::path& path) {
    return uid166::readTextFile(path, uid166::ErrorCode::sceneMissing,
                                uid166::ErrorCode::sceneRead);
}

uid166::SceneObject makeBox(uint32_t id, const uid166::Vec3& mn, const uid166::Vec3& mx) {
    uid166::SceneObject object;
    object.id = id;
    object.bounds.mn = mn;
    object.bounds.mx = mx;
    return object;
}

std::vector<uid166::AABB> boundsOf(const std::vector<uid166::SceneObject>& objects) {
    std::vector<uid166::AABB> bounds;
    bounds.reserve(objects.size());
    for (const auto& object : objects) bounds.push_back(object.bounds);
    return bounds;
}

std::vector<uint32_t> bruteFrustum(const std::vector<uid166::SceneObject>& objects,
                                   const uid166::Frustum& frustum) {
    std::vector<uint32_t> result;
    for (const auto& object : objects) {
        if (frustum.intersects(object.bounds)) result.push_back(object.id);
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<float> matrixFromJson(const uid166::JsonValue& value) {
    const auto& array = requireArray(value, "matrix", uid166::ErrorCode::internal);
    std::vector<float> result;
    result.reserve(16);
    for (const auto& element : array.array) {
        result.push_back(static_cast<float>(
            requireNumber(element, "matrix element", uid166::ErrorCode::internal)));
    }
    return result;
}

} // namespace

int main() {
    const std::filesystem::path temp = uniqueTempDirectory();
    std::filesystem::create_directories(temp);

    test("scene JSON roundtrip and deterministic hash", [&] {
        uid166::SceneDesc scene = uid166::generateScene({5000, 166});
        check(scene.objects.size() == 5000, "5000 objects generated");
        check(scene.lights.size() == 5, "one directional and four local lights");
        const auto scenePath = temp / "scene.json";
        uid166::writeSceneJson(scene, scenePath);
        const std::string first = fileBytes(scenePath);
        uid166::writeSceneJson(scene, scenePath);
        const std::string second = fileBytes(scenePath);
        check(first == second, "scene serialization is byte deterministic");
        uid166::SceneDesc loaded = uid166::loadSceneJson(scenePath);
        check(loaded.objects.size() == scene.objects.size(), "object count survives load");
        check(loaded.lights.size() == scene.lights.size(), "light count survives load");
        check(loaded.objects.front().id == 0, "object ids remain stable");
        check(std::abs(loaded.objects[17].position.x - scene.objects[17].position.x) < 1.0e-5f,
              "object transform survives load");
    });

    test("animation binary roundtrip and pose evaluation", [&] {
        uid166::CharacterAsset asset = uid166::generateCharacterAsset();
        check(asset.bones.size() == 10, "character has ten bones");
        check(asset.tracks.size() == asset.bones.size(), "every bone has a track");
        check(asset.vertices.size() > 300, "character has skinned geometry");
        const auto path = temp / "character.skan";
        uid166::writeCharacterBinary(asset, path);
        uid166::CharacterAsset loaded = uid166::loadCharacterBinary(path);
        check(loaded.vertices.size() == asset.vertices.size(), "vertex count roundtrip");
        check(loaded.indices.size() == asset.indices.size(), "index count roundtrip");
        check(loaded.bones.size() == asset.bones.size(), "bone count roundtrip");
        uid166::CharacterAsset loadedAgain = uid166::loadCharacterBinary(path);
        check(loaded.payloadHash != 0 && loaded.payloadHash == loadedAgain.payloadHash,
              "payload checksum is stable");
        const auto poseA = uid166::evaluatePose(loaded, 0.4f);
        const auto poseB = uid166::evaluatePose(loaded, 0.4f);
        check(poseA.size() == loaded.bones.size(), "pose contains the full palette");
        for (size_t i = 0; i < poseA.size(); ++i) {
            check(uid166::matrixExactlyEqual(poseA[i], poseB[i]),
                  "pose evaluation is deterministic");
        }
    });

    test("corrupt and missing animation files map to business codes", [&] {
        uid166::CharacterAsset asset = uid166::generateCharacterAsset();
        const auto path = temp / "broken.skan";
        uid166::writeCharacterBinary(asset, path);
        const std::string bytes = fileBytes(path);
        uid166::writeTextFile(path, std::string_view(bytes.data(), bytes.size() - 7));
        bool corrupt = false;
        try {
            (void)uid166::loadCharacterBinary(path);
        } catch (const uid166::AppError& error) {
            corrupt = error.code() == uid166::ErrorCode::animationCorrupt;
        }
        check(corrupt, "truncated animation reports animationCorrupt");
        bool missing = false;
        try {
            (void)uid166::loadCharacterBinary(temp / "does_not_exist.skan");
        } catch (const uid166::AppError& error) {
            missing = error.code() == uid166::ErrorCode::animationMissing;
        }
        check(missing, "missing animation reports animationMissing");
    });

    test("scene parse/schema/missing errors are distinct", [&] {
        const auto badJson = temp / "invalid.json";
        uid166::writeTextFile(badJson, "{\"version\": 1, \"objects\": [}");
        bool parse = false;
        try {
            (void)uid166::loadSceneJson(badJson);
        } catch (const uid166::AppError& error) {
            parse = error.code() == uid166::ErrorCode::sceneParse;
        }
        check(parse, "invalid JSON reports sceneParse");

        const auto badSchema = temp / "bad_schema.json";
        uid166::writeTextFile(badSchema, "{\"version\":1}");
        bool schema = false;
        try {
            (void)uid166::loadSceneJson(badSchema);
        } catch (const uid166::AppError& error) {
            schema = error.code() == uid166::ErrorCode::sceneSchema;
        }
        check(schema, "missing required fields report sceneSchema");
        bool missing = false;
        try {
            (void)uid166::loadSceneJson(temp / "missing_scene.json");
        } catch (const uid166::AppError& error) {
            missing = error.code() == uid166::ErrorCode::sceneMissing;
        }
        check(missing, "missing scene reports sceneMissing");
    });

    test("BVH frustum query matches brute force and supports refit", [&] {
        uid166::SceneDesc scene = uid166::generateScene({800, 42});
        uid166::Bvh bvh;
        bvh.build(boundsOf(scene.objects));
        uid166::Camera camera;
        camera.eye = {0.0f, 12.0f, -30.0f};
        camera.target = {0.0f, 2.0f, 0.0f};
        camera.aspect = 16.0f / 9.0f;
        camera.farZ = 260.0f;
        std::vector<uint32_t> query;
        bvh.queryFrustum(camera.frustum(), query);
        check(query == bruteFrustum(scene.objects, camera.frustum()),
              "BVH query equals brute-force frustum test");

        const uint32_t movedId = 3;
        scene.objects[movedId].bounds.mn = {-2000.0f, -2000.0f, -2000.0f};
        scene.objects[movedId].bounds.mx = {-1990.0f, -1990.0f, -1990.0f};
        bvh.updateObject(movedId, scene.objects[movedId].bounds);
        bvh.queryFrustum(camera.frustum(), query);
        check(std::find(query.begin(), query.end(), movedId) == query.end(),
              "refit removes moved object from previous region");
        check(query == bruteFrustum(scene.objects, camera.frustum()),
              "BVH refit remains consistent with brute force");
    });

    test("CSM has three stable cascades for fixed camera and light", [&] {
        uid166::SceneDesc scene = uid166::generateScene({256, 7});
        uid166::Bvh bvh;
        bvh.build(boundsOf(scene.objects));
        uid166::Camera camera = uid166::cameraAtFrame(scene.camera, 45, 1.0f / 60.0f);
        const uid166::Light& sun = scene.lights[0];
        uid166::CsmUpdatePolicy policy;
        uid166::CascadeConfig config;
        const uid166::CsmFrame first = policy.update(camera, sun, bvh, config, true, true);
        check(first.cascades.size() == 3, "three directional cascades");
        check(first.updatedCascades == 3, "first frame updates all cascades");
        check(!first.matricesStable, "first frame has no previous matrices");
        const uid166::CsmFrame second = policy.update(camera, sun, bvh, config, true, false);
        check(second.updatedCascades == 3, "animation dirtiness updates cascade contents");
        check(second.matricesStable, "matrices remain stable across animated content");
        for (size_t i = 0; i < first.cascades.size(); ++i) {
            check(uid166::matrixExactlyEqual(first.cascades[i].matrix,
                                             second.cascades[i].matrix),
                  "cascade matrix is element-wise identical");
            check(second.cascades[i].splitFar > second.cascades[i].splitNear,
                  "cascade split interval is increasing");
        }
        check(uid166::selectCascade(second, second.cascades[0].splitFar * 0.5f) == 0,
              "near depth selects cascade zero");
        check(uid166::selectCascade(second, camera.farZ * 1.1f) == 2,
              "far depth clamps to last cascade");
    });

    test("local shadow policy uses dirty/visibility and a per-frame budget", [&] {
        uid166::SceneDesc scene = uid166::generateScene({512, 9});
        uid166::Camera camera;
        camera.eye = {0.0f, 8.0f, -28.0f};
        camera.target = {0.0f, 2.0f, 0.0f};
        std::vector<uint32_t> visible;
        for (uint32_t i = 0; i < 64; ++i) visible.push_back(i);
        const uid166::AABB characterBounds{{-1.0f, 0.0f, -1.0f}, {1.0f, 2.6f, 1.0f}};
        uid166::LocalShadowPolicy policy;
        policy.reset(scene.lights.size());
        uid166::LocalShadowConfig config;
        config.maxUpdatesPerFrame = 2;
        bool sawUpdate = false;
        for (int frame = 0; frame < 8; ++frame) {
            const uid166::LocalShadowFrame result = policy.update(
                camera, scene.lights, visible, scene.objects, characterBounds, true, true,
                frame == 0, config);
            check(result.updatedCount <= 2, "local shadow updates respect per-frame budget");
            sawUpdate = sawUpdate || result.updatedCount != 0;
            for (const auto& light : result.lights) {
                const size_t expected = light.type == uid166::LightType::point ? 6u : 1u;
                check(light.matrices.size() == expected,
                      "point lights expose six cube matrices, spots expose one");
            }
        }
        check(sawUpdate, "visible dirty lights are updated over time");
    });

    test("occlusion results are delayed one frame and fast-camera path activates", [&] {
        std::vector<uid166::SceneObject> objects;
        objects.push_back(makeBox(0, {-10.0f, -10.0f, -1.0f}, {10.0f, 10.0f, 1.0f}));
        objects.push_back(makeBox(1, {-1.0f, -1.0f, 8.0f}, {1.0f, 1.0f, 10.0f}));
        uid166::Bvh bvh;
        bvh.build(boundsOf(objects));
        uid166::Camera camera;
        camera.eye = {0.0f, 0.0f, -20.0f};
        camera.target = {0.0f, 0.0f, 10.0f};
        camera.farZ = 100.0f;
        uid166::OcclusionCuller culler;
        culler.reset(objects.size());
        uid166::CullOptions options;
        const uid166::CullResult frame0 = culler.cull(objects, bvh, camera, camera, options);
        check(frame0.occluded.empty(), "first frame has no delayed occlusion data");
        const uid166::CullResult frame1 = culler.cull(objects, bvh, camera, camera, options);
        check(std::find(frame1.occluded.begin(), frame1.occluded.end(), 1u) !=
                  frame1.occluded.end(),
              "object behind occluder is removed on the next frame");
        check(frame1.occluded.size() + frame1.visible.size() ==
                  frame1.frustumVisible.size(),
              "culling partition covers every frustum candidate");
        uid166::Camera fast = camera;
        fast.eye.x += 4.0f;
        fast.target.x += 4.0f;
        const uid166::CullResult fastFrame = culler.cull(objects, bvh, fast, camera, options);
        check(fastFrame.cameraFast, "large camera movement selects conservative fast path");
    });

    test("adaptive quality uses hysteresis and restores default", [&] {
        uid166::AdaptiveQuality adaptive;
        for (uint64_t frame = 0; frame < 40; ++frame) {
            (void)adaptive.update(frame, 28.0f);
        }
        check(adaptive.state().level > 0, "sustained high GPU time degrades quality");
        bool sawDegrade = false;
        for (const auto& entry : adaptive.history()) {
            sawDegrade = sawDegrade || entry.action == uid166::QualityAction::degrade;
        }
        const uint32_t degradedLevel = adaptive.state().level;
        for (uint64_t frame = 40; frame < 220; ++frame) {
            (void)adaptive.update(frame, 6.0f);
        }
        check(adaptive.state().level == 0, "sustained low GPU time restores default quality");
        check(degradedLevel > 0, "degrade level was observed");
        check(adaptive.history().size() <= 120, "adaptive history is capped at 120 entries");
        bool sawRestore = false;
        for (const auto& entry : adaptive.history()) {
            sawRestore = sawRestore || entry.action == uid166::QualityAction::restore;
        }
        check(sawDegrade && sawRestore, "history records both degrade and restore actions");
    });

    test("parallel command recording is deterministic and budget-safe", [&] {
        uid166::SceneDesc scene = uid166::generateScene({2000, 11});
        std::vector<uint32_t> visible;
        for (uint32_t i = 0; i < scene.objects.size(); ++i) visible.push_back(i);
        uid166::CommandRecorder recorder;
        uid166::RecordingConfig config;
        config.workerCount = 4;
        config.budgetMicros = 100000;
        const auto first = recorder.record(visible, scene.objects, config);
        const auto second = recorder.record(visible, scene.objects, config);
        check(first.commands.size() == visible.size(), "all commands are recorded");
        check(first.batches.size() <= 8, "commands merge into material batches");
        uint64_t total = 0;
        for (const auto& worker : first.workers) total += worker.commands;
        check(total + first.fallbackSerialCommands == visible.size(),
              "worker and fallback counts cover the visible set");
        check(first.commands.front().objectId == visible.front(), "command order is stable");
        for (size_t i = 0; i < first.commands.size(); i += 137) {
            check(first.commands[i].objectId == second.commands[i].objectId,
                  "parallel recording preserves object order");
        }
        uid166::RecordingConfig tight = config;
        tight.budgetMicros = 1;
        const auto cut = recorder.record(visible, scene.objects, tight);
        check(cut.commands.size() == visible.size(),
              "budget exhaustion still produces a complete correct command list");
        check(cut.budgetHit || cut.budgetExceeded, "tight budget is reported");
    });

    test("120-frame 5000-object benchmark integration", [&] {
        uid166::BenchmarkOptions options;
        options.data.dataDir = temp / "benchmark_data";
        options.outputPath = temp / "benchmark.json";
        options.frames = 120;
        options.data.objectCount = 5000;
        options.data.seed = 166;
        auto data = uid166::prepareSceneData(options.data);
        const uid166::BenchmarkSummary summary = uid166::runBenchmark(options, std::move(data));
        check(summary.validationPassed, "benchmark per-frame invariants pass");
        check(summary.longestStableMatrixRun >= 40, "stable shadow matrix window is present");
        check(summary.defaultQualityRestored, "final quality returns to default");
        check(std::filesystem::exists(options.outputPath), "benchmark JSON exists");
        const uid166::JsonValue root = uid166::parseJson(
            fileBytes(options.outputPath), uid166::ErrorCode::internal,
            uid166::ErrorCode::internal);
        const auto& frames = requireArray(requireMember(root, "frames",
                                                        uid166::ErrorCode::internal),
                                          "frames", uid166::ErrorCode::internal).array;
        check(frames.size() == 120, "benchmark contains 120 frames");
        bool sawFast = false;
        bool sawFrozen = false;
        bool stableMatrices = true;
        std::vector<float> previousCascade;
        bool previousFrozen = false;
        for (size_t i = 0; i < frames.size(); ++i) {
            const auto& frame = frames[i];
            const auto& culling = requireObject(requireMember(frame, "culling",
                                                              uid166::ErrorCode::internal),
                                                "culling", uid166::ErrorCode::internal);
            const int64_t visible = requireInteger(requireMember(culling, "visible",
                                                                  uid166::ErrorCode::internal),
                                                   "visible", uid166::ErrorCode::internal);
            const int64_t occluded = requireInteger(requireMember(culling, "occluded",
                                                                   uid166::ErrorCode::internal),
                                                    "occluded", uid166::ErrorCode::internal);
            const int64_t frustumVisible = requireInteger(
                requireMember(culling, "frustumVisible", uid166::ErrorCode::internal),
                "frustumVisible", uid166::ErrorCode::internal);
            check(visible + occluded == frustumVisible,
                  "visible + occluded equals frustumVisible in JSON");
            const auto& visibleIds = requireArray(
                requireMember(culling, "visibleObjectIds", uid166::ErrorCode::internal),
                "visibleObjectIds", uid166::ErrorCode::internal).array;
            const auto& occludedIds = requireArray(
                requireMember(culling, "occludedObjectIds", uid166::ErrorCode::internal),
                "occludedObjectIds", uid166::ErrorCode::internal).array;
            check(static_cast<int64_t>(visibleIds.size()) == visible,
                  "visible id set size matches count");
            check(static_cast<int64_t>(occludedIds.size()) == occluded,
                  "occluded id set size matches count");
            const auto& camera = requireObject(requireMember(frame, "camera",
                                                             uid166::ErrorCode::internal),
                                               "camera", uid166::ErrorCode::internal);
            const bool frozen = requireMember(camera, "frozenFromPrevious",
                                              uid166::ErrorCode::internal).boolean;
            sawFrozen = sawFrozen || frozen;
            sawFast = sawFast || requireMember(culling, "cameraFast",
                                               uid166::ErrorCode::internal).boolean;
            const auto& shadows = requireObject(requireMember(frame, "shadows",
                                                              uid166::ErrorCode::internal),
                                                "shadows", uid166::ErrorCode::internal);
            const auto& directional = requireObject(
                requireMember(shadows, "directional", uid166::ErrorCode::internal),
                "directional", uid166::ErrorCode::internal);
            const auto& cascades = requireArray(
                requireMember(directional, "cascades", uid166::ErrorCode::internal),
                "cascades", uid166::ErrorCode::internal).array;
            check(cascades.size() == 3, "JSON reports three cascades");
            std::vector<float> flattened;
            int64_t cascadeSelected = 0;
            for (const auto& cascade : cascades) {
                const auto matrix = matrixFromJson(requireMember(
                    cascade, "matrix", uid166::ErrorCode::internal));
                flattened.insert(flattened.end(), matrix.begin(), matrix.end());
                cascadeSelected += requireInteger(
                    requireMember(cascade, "selectedVisibleObjects",
                                  uid166::ErrorCode::internal),
                    "selectedVisibleObjects", uid166::ErrorCode::internal);
            }
            check(cascadeSelected == visible,
                  "cascade selection partitions all visible objects");
            if (frozen && previousFrozen && !previousCascade.empty()) {
                stableMatrices = stableMatrices && flattened == previousCascade;
            }
            previousCascade = std::move(flattened);
            previousFrozen = frozen;

            const auto& local = requireArray(requireMember(shadows, "local",
                                                           uid166::ErrorCode::internal),
                                             "local", uid166::ErrorCode::internal).array;
            check(local.size() == 4, "four local shadow decisions are reported");
            int64_t updatedLocal = 0;
            for (const auto& light : local) {
                const std::string type = requireString(requireMember(
                    light, "type", uid166::ErrorCode::internal), "type",
                    uid166::ErrorCode::internal);
                const auto& matrices = requireArray(requireMember(
                    light, "matrices", uid166::ErrorCode::internal), "matrices",
                    uid166::ErrorCode::internal).array;
                check(matrices.size() == (type == "point" ? 6u : 1u),
                      "local shadow matrix count matches light type");
                if (requireMember(light, "updated", uid166::ErrorCode::internal).boolean) {
                    ++updatedLocal;
                }
            }
            check(updatedLocal <= 2, "local shadow update budget is enforced in JSON");
        }
        check(stableMatrices, "frozen camera frames have identical cascade matrices");
        check(sawFast, "fast-camera conservative path is exercised");
        check(sawFrozen, "camera hold window is exercised");
        const auto& history = requireArray(requireMember(root, "adaptiveHistory",
                                                         uid166::ErrorCode::internal),
                                           "adaptiveHistory", uid166::ErrorCode::internal).array;
        check(history.size() == 120, "JSON contains the last 120 adaptive history frames");

        uid166::BenchmarkOptions secondOptions = options;
        secondOptions.data.reuseExisting = true;
        secondOptions.outputPath = temp / "benchmark_second.json";
        auto secondData = uid166::prepareSceneData(secondOptions.data);
        const uid166::BenchmarkSummary secondSummary =
            uid166::runBenchmark(secondOptions, std::move(secondData));
        check(summary.frameStatsHash == secondSummary.frameStatsHash,
              "frame statistics hash is deterministic across runs");
    });

    std::cout << "checks=" << checks << " failures=" << failures << '\n';
    if (failures == 0) {
        std::error_code ec;
        std::filesystem::remove_all(temp, ec);
    }
    return failures == 0 ? 0 : 1;
}
