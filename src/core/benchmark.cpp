#include "core/benchmark.hpp"

#include "core/bvh.hpp"
#include "core/error.hpp"
#include "core/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string_view>

namespace uid166 {

namespace {

uint64_t fileHash(const std::filesystem::path& path, ErrorCode missingCode,
                  ErrorCode readCode) {
    const std::string bytes = readTextFile(path, missingCode, readCode,
                                           256u * 1024u * 1024u);
    return hashBytes(kFnvOffset, bytes.data(), bytes.size());
}

void writeMatrix(JsonWriter& w, const Mat4& matrix) {
    w.beginArray();
    for (float value : matrix.m) w.number(value);
    w.endArray();
}

void writeIds(JsonWriter& w, const std::vector<uint32_t>& ids) {
    w.beginArray();
    for (uint32_t id : ids) w.integer(id);
    w.endArray();
}

void hashU64(uint64_t& hash, uint64_t value) {
    hash = hashBytes(hash, &value, sizeof(value));
}

void hashMat4(uint64_t& hash, const Mat4& matrix) {
    for (float value : matrix.m) {
        const int64_t quantized = static_cast<int64_t>(
            std::llround(static_cast<double>(value) * 10000.0));
        hash = hashBytes(hash, &quantized, sizeof(quantized));
    }
}

AABB skinnedBounds(const CharacterAsset& asset, const std::vector<Mat4>& palette,
                   const Mat4& world) {
    AABB bounds;
    for (const SkinnedVertex& vertex : asset.vertices) {
        Vec3 p{};
        for (int i = 0; i < 4; ++i) {
            if (vertex.weight[i] <= 0.0f) continue;
            const uint32_t bone = vertex.bone[i];
            if (bone >= palette.size()) continue;
            p += transformPoint(palette[bone], {vertex.px, vertex.py, vertex.pz}) *
                 vertex.weight[i];
        }
        bounds.expand(transformPoint(world, p));
    }
    return bounds;
}

void writeCascade(JsonWriter& w, const Cascade& cascade, uint32_t selectedObjects) {
    w.beginObject();
    w.key("index"); w.integer(cascade.index);
    w.key("splitNear"); w.number(cascade.splitNear);
    w.key("splitFar"); w.number(cascade.splitFar);
    w.key("matrix"); writeMatrix(w, cascade.matrix);
    w.key("updated"); w.boolean(cascade.updated);
    w.key("visibleCasters"); w.integer(cascade.visibleCasters);
    w.key("selectedVisibleObjects"); w.integer(selectedObjects);
    w.endObject();
}

void writeLocalShadow(JsonWriter& w, const LocalShadowDecision& light) {
    w.beginObject();
    w.key("lightId"); w.integer(light.lightId);
    w.key("type"); w.string(light.type == LightType::point ? "point" : "spot");
    w.key("visible"); w.boolean(light.visible);
    w.key("dirty"); w.boolean(light.dirty);
    w.key("updated"); w.boolean(light.updated);
    w.key("deferredByBudget"); w.boolean(light.deferredByBudget);
    w.key("sourceMask"); w.integer(light.sourceMask);
    w.key("sourceNames"); w.string(shadowSourceNames(light.sourceMask));
    w.key("matrices");
    w.beginArray();
    for (const ShadowMatrixInfo& info : light.matrices) {
        w.beginObject();
        w.key("face"); w.string(info.face);
        w.key("matrix"); writeMatrix(w, info.matrix);
        w.endObject();
    }
    w.endArray();
    w.endObject();
}

void writeQualityHistory(JsonWriter& w, const std::vector<AdaptiveHistoryEntry>& history) {
    w.beginArray();
    for (const AdaptiveHistoryEntry& entry : history) {
        w.beginObject();
        w.key("frame"); w.integer(static_cast<int64_t>(entry.frame));
        w.key("gpuFrameMs"); w.number(entry.gpuFrameMs);
        w.key("smoothedFrameMs"); w.number(entry.smoothedFrameMs);
        w.key("level"); w.integer(entry.level);
        w.key("renderScale"); w.number(entry.renderScale);
        w.key("shadowScale"); w.number(entry.shadowScale);
        w.key("action"); w.string(qualityActionName(entry.action));
        w.key("reason"); w.string(entry.reason);
        w.endObject();
    }
    w.endArray();
}

} // namespace

PreparedData loadSceneData(const DataOptions& options) {
    PreparedData data;
    data.paths.scene = options.dataDir / "uid166_scene.json";
    data.paths.animation = options.dataDir / "uid166_character.skan";
    data.scene = loadSceneJson(data.paths.scene);
    data.character = loadCharacterBinary(data.paths.animation);
    data.sceneHash = fileHash(data.paths.scene, ErrorCode::sceneMissing,
                              ErrorCode::sceneRead);
    data.animationHash = fileHash(data.paths.animation, ErrorCode::animationMissing,
                                  ErrorCode::animationRead);
    return data;
}

PreparedData prepareSceneData(const DataOptions& options) {
    if (options.reuseExisting) return loadSceneData(options);
    std::error_code ec;
    std::filesystem::create_directories(options.dataDir, ec);
    if (ec) {
        throw AppError(ErrorCode::outputWrite,
                       "cannot create data directory: " + options.dataDir.string());
    }
    SceneDesc generated = generateScene({options.objectCount, options.seed});
    CharacterAsset character = generateCharacterAsset();
    writeSceneJson(generated, options.dataDir / "uid166_scene.json");
    writeCharacterBinary(character, options.dataDir / "uid166_character.skan");
    return loadSceneData(options);
}

BenchmarkSummary runBenchmark(const BenchmarkOptions& options, PreparedData data) {
    if (options.frames == 0 || options.frames > 100000) {
        throw AppError(ErrorCode::usage, "frame count must be in [1, 100000]");
    }
    if (options.frameSeconds <= 0.0f || options.frameSeconds > 1.0f) {
        throw AppError(ErrorCode::usage, "frame step must be in (0, 1]");
    }
    SceneDesc& scene = data.scene;
    std::vector<AABB> bounds;
    bounds.reserve(scene.objects.size());
    for (const SceneObject& object : scene.objects) bounds.push_back(object.bounds);
    Bvh bvh;
    bvh.build(bounds);

    const Light* sun = nullptr;
    for (const Light& light : scene.lights) {
        if (light.type == LightType::directional) {
            sun = &light;
            break;
        }
    }
    if (!sun) throw AppError(ErrorCode::sceneSchema, "directional light not found");

    OcclusionCuller occlusion;
    occlusion.reset(scene.objects.size());
    CsmUpdatePolicy csmPolicy;
    LocalShadowPolicy localShadowPolicy;
    localShadowPolicy.reset(scene.lights.size());
    AdaptiveQuality adaptive(options.adaptive);
    CommandRecorder recorder;

    BenchmarkSummary summary;
    JsonWriter json(true);
    json.beginObject();
    json.key("schema"); json.string("uid166.benchmark.v1");
    json.key("mode"); json.string("headless-deterministic");
    json.key("gpuTimeSource"); json.string("deterministic-model");
    json.key("config");
    json.beginObject();
    json.key("frames"); json.integer(options.frames);
    json.key("frameSeconds"); json.number(options.frameSeconds);
    json.key("objectCount"); json.integer(scene.objects.size());
    json.key("seed"); json.integer(static_cast<int64_t>(scene.seed));
    json.key("recordingBudgetUs"); json.integer(options.recording.budgetMicros);
    json.key("recordingWorkers"); json.integer(options.recording.workerCount);
    json.key("cascadeCount"); json.integer(options.cascades.count);
    json.key("cascadeShadowSize"); json.integer(options.cascades.shadowSize);
    json.key("cascadeStabilizationSize"); json.integer(options.cascades.stabilizationSize);
    json.key("maxLocalShadowUpdates"); json.integer(options.localShadows.maxUpdatesPerFrame);
    json.key("simulateGpuLoad"); json.boolean(options.simulateGpuLoad);
    json.key("writeCullingSets"); json.boolean(options.writeCullingSets);
    json.key("sceneFile"); json.string(data.paths.scene.string());
    json.key("animationFile"); json.string(data.paths.animation.string());
    json.endObject();
    json.key("integrity");
    json.beginObject();
    json.key("sceneHash"); json.integer(static_cast<int64_t>(data.sceneHash));
    json.key("animationHash"); json.integer(static_cast<int64_t>(data.animationHash));
    json.key("animationPayloadHash");
    json.integer(static_cast<int64_t>(data.character.payloadHash));
    json.endObject();
    json.key("frames");
    json.beginArray();

    Camera previousCamera = cameraAtFrame(scene.camera, 0, options.frameSeconds);
    uint64_t previousFrameHash = kFnvOffset;
    uint32_t stableRun = 0;
    uint32_t currentStableStart = 0;
    bool previousCameraFrozen = false;
    bool previousMatricesStable = false;
    uint32_t qualityLevelsSeen = 0;
    std::array<uint32_t, 8> levelCounts{};

    for (uint32_t frameIndex = 0; frameIndex < options.frames; ++frameIndex) {
        const float timeSeconds = static_cast<float>(frameIndex) * options.frameSeconds;
        updateMovingObjects(scene, timeSeconds);
        updateMovingLights(scene, timeSeconds);
        for (const SceneObject& object : scene.objects) {
            if (object.motion.enabled) bvh.updateObject(object.id, object.bounds);
        }
        const Camera camera = cameraAtFrame(scene.camera, frameIndex, options.frameSeconds);
        const CullResult cull = occlusion.cull(scene.objects, bvh, camera, previousCamera,
                                               options.culling);

        const Mat4 characterWorld = Mat4::identity();
        const float animationTime = std::fmod(timeSeconds, data.character.duration);
        const std::vector<Mat4> palette = evaluatePose(data.character, animationTime);
        const AABB characterBounds = skinnedBounds(data.character, palette, characterWorld);
        const bool characterVisible = camera.frustum().intersects(characterBounds);
        const bool animationDirty = true;

        CascadeConfig cascadeConfig = options.cascades;
        const uint32_t shadowResolution = std::max<uint32_t>(
            256u, static_cast<uint32_t>(static_cast<float>(options.cascades.shadowSize) *
                                        adaptive.state().shadowScale));
        cascadeConfig.shadowSize = shadowResolution;
        CsmFrame csm = csmPolicy.update(camera, *sun, bvh, cascadeConfig,
                                        animationDirty, frameIndex == 0);

        const LocalShadowFrame local = localShadowPolicy.update(
            camera, scene.lights, cull.visible, scene.objects, characterBounds,
            characterVisible, animationDirty, frameIndex == 0, options.localShadows);
        std::array<uint32_t, 3> cascadeSelection{};
        const Vec3 cameraForward = normalize(camera.target - camera.eye);
        for (uint32_t id : cull.visible) {
            if (id >= scene.objects.size()) continue;
            const float viewDepth = dot(scene.objects[id].bounds.center() - camera.eye,
                                        cameraForward);
            ++cascadeSelection[selectCascade(csm, viewDepth)];
        }

        const float workload = 2.4f + static_cast<float>(cull.visible.size()) * 0.0027f +
                               static_cast<float>(local.updatedCount) * 0.55f +
                               static_cast<float>(csm.updatedCascades) * 0.35f;
        const float qualityCost = 0.48f + 0.52f * adaptive.state().renderScale;
        float simulatedLoad = 0.0f;
        if (options.simulateGpuLoad && frameIndex >= options.simulatedLoadStart &&
            frameIndex < options.simulatedLoadEnd) {
            simulatedLoad = 13.5f;
        }
        const float gpuFrameMs = workload * qualityCost + simulatedLoad +
                                 (1.0f - adaptive.state().shadowScale) * 0.15f;
        const QualityState quality = adaptive.update(frameIndex, gpuFrameMs);
        if (quality.level >= levelCounts.size()) {
            throw AppError(ErrorCode::internal, "quality level exceeds statistics table");
        }
        ++levelCounts[quality.level];
        qualityLevelsSeen = std::max(qualityLevelsSeen, quality.level + 1u);

        const RecordingResult recording = recorder.record(cull.visible, scene.objects,
                                                          options.recording);
        uint32_t shadowDrawCalls = csm.updatedCascades * 2u;
        for (const LocalShadowDecision& light : local.lights) {
            if (!light.updated) continue;
            shadowDrawCalls += light.type == LightType::point ? 12u : 2u;
        }
        const uint32_t mainDrawCalls = recording.mergedDrawCalls + 1u;
        const uint32_t totalDrawCalls = mainDrawCalls + shadowDrawCalls;

        const bool cameraFrozen = matrixExactlyEqual(camera.view(), previousCamera.view()) &&
                                  matrixExactlyEqual(camera.projection(),
                                                     previousCamera.projection());
        if (frameIndex > 0 && cameraFrozen && csm.matricesStable) {
            if (!previousCameraFrozen || !previousMatricesStable) {
                stableRun = 1;
                currentStableStart = frameIndex;
            } else {
                ++stableRun;
            }
            if (stableRun > summary.longestStableMatrixRun) {
                summary.longestStableMatrixRun = stableRun;
                summary.stableWindowStart = currentStableStart;
                summary.stableWindowEnd = frameIndex;
            }
        } else {
            stableRun = 0;
        }

        bool frameValid = cull.visible.size() + cull.occluded.size() ==
                          cull.frustumVisible.size();
        frameValid = frameValid &&
                     cull.frustumRejected + cull.frustumVisible.size() == cull.totalObjects;
        frameValid = frameValid && csm.cascades.size() == options.cascades.count;
        for (const Cascade& cascade : csm.cascades) frameValid = frameValid && cascade.matrix.m[15] != 0.0f;
        frameValid = frameValid && recording.commands.size() == cull.visible.size();
        summary.validationPassed = summary.validationPassed && frameValid;

        uint64_t frameHash = kFnvOffset;
        hashU64(frameHash, frameIndex);
        hashU64(frameHash, cull.visible.size());
        hashU64(frameHash, cull.occluded.size());
        for (uint32_t id : cull.visible) hashU64(frameHash, id);
        for (uint32_t id : cull.occluded) hashU64(frameHash, id);
        for (const Cascade& cascade : csm.cascades) {
            hashMat4(frameHash, cascade.matrix);
            hashU64(frameHash, cascade.updated);
        }
        for (const LocalShadowDecision& light : local.lights) {
            hashU64(frameHash, light.sourceMask);
            hashU64(frameHash, light.updated);
            for (const ShadowMatrixInfo& info : light.matrices) hashMat4(frameHash, info.matrix);
        }
        for (const RenderCommand& command : recording.commands) {
            hashU64(frameHash, command.objectId);
            hashMat4(frameHash, command.world);
        }
        hashU64(frameHash, quality.level);
        summary.frameStatsHash = hashBytes(summary.frameStatsHash, &frameHash, sizeof(frameHash));

        json.beginObject();
        json.key("frame"); json.integer(frameIndex);
        json.key("timeSeconds"); json.number(timeSeconds);
        json.key("camera");
        json.beginObject();
        json.key("eye"); json.beginArray(); json.number(camera.eye.x); json.number(camera.eye.y);
        json.number(camera.eye.z); json.endArray();
        json.key("target"); json.beginArray(); json.number(camera.target.x);
        json.number(camera.target.y); json.number(camera.target.z); json.endArray();
        json.key("frozenFromPrevious"); json.boolean(cameraFrozen);
        json.endObject();

        json.key("animation");
        json.beginObject();
        json.key("timeSeconds"); json.number(animationTime);
        json.key("characterVisible"); json.boolean(characterVisible);
        json.key("characterBoundsMin"); json.beginArray();
        json.number(characterBounds.mn.x); json.number(characterBounds.mn.y);
        json.number(characterBounds.mn.z); json.endArray();
        json.key("characterBoundsMax"); json.beginArray();
        json.number(characterBounds.mx.x); json.number(characterBounds.mx.y);
        json.number(characterBounds.mx.z); json.endArray();
        json.endObject();

        json.key("culling");
        json.beginObject();
        json.key("totalObjects"); json.integer(cull.totalObjects);
        json.key("frustumRejected"); json.integer(cull.frustumRejected);
        json.key("frustumVisible"); json.integer(cull.frustumVisible.size());
        json.key("visible"); json.integer(cull.visible.size());
        json.key("occluded"); json.integer(cull.occluded.size());
        json.key("cameraFast"); json.boolean(cull.cameraFast);
        json.key("delayedQueryResults"); json.integer(cull.delayedQueryResults);
        json.key("hierarchyTests"); json.integer(cull.hierarchyTests);
        json.key("bvhNodesVisited"); json.integer(cull.bvhStats.nodesVisited);
        if (options.writeCullingSets) {
            json.key("visibleObjectIds"); writeIds(json, cull.visible);
            json.key("occludedObjectIds"); writeIds(json, cull.occluded);
        }
        json.endObject();

        json.key("shadows");
        json.beginObject();
        json.key("directional");
        json.beginObject();
        json.key("lightId"); json.integer(sun->id);
        json.key("updatedCascades"); json.integer(csm.updatedCascades);
        json.key("matricesStable"); json.boolean(csm.matricesStable);
        json.key("splitDistances");
        json.beginArray();
        for (float split : csm.splitDistances) json.number(split);
        json.endArray();
        json.key("cascades");
        json.beginArray();
        for (const Cascade& cascade : csm.cascades) {
            writeCascade(json, cascade, cascadeSelection[cascade.index]);
        }
        json.endArray();
        json.key("cascadeSelection");
        json.beginArray();
        for (uint32_t count : cascadeSelection) json.integer(count);
        json.endArray();
        json.endObject();
        json.key("local");
        json.beginArray();
        for (const LocalShadowDecision& light : local.lights) writeLocalShadow(json, light);
        json.endArray();
        json.key("updatedShadowCount");
        json.integer(csm.updatedCascades + local.updatedCount);
        json.key("updatedLightShadowCount"); json.integer(local.updatedCount);
        json.key("dirtyLightCount"); json.integer(local.dirtyCount);
        json.key("visibleLightCount"); json.integer(local.visibleCount);
        json.key("shadowResolution"); json.integer(shadowResolution);
        json.endObject();

        json.key("draw");
        json.beginObject();
        json.key("visibleObjects"); json.integer(cull.visible.size());
        json.key("recordedCommands"); json.integer(recording.commands.size());
        json.key("staticBatches"); json.integer(recording.mergedDrawCalls);
        json.key("characterDrawCalls"); json.integer(1);
        json.key("shadowDrawCalls"); json.integer(shadowDrawCalls);
        json.key("occlusionQueryDraws"); json.integer(0);
        json.key("mainDrawCalls"); json.integer(mainDrawCalls);
        json.key("totalDrawCalls"); json.integer(totalDrawCalls);
        json.key("submittedInstances");
        json.integer(recording.commands.size() + csm.updatedCascades +
                     local.updatedCount);
        json.endObject();

        json.key("recording");
        json.beginObject();
        json.key("elapsedUs"); json.integer(static_cast<int64_t>(recording.elapsedMicros));
        json.key("parallelStageUs");
        json.integer(static_cast<int64_t>(recording.parallelStageMicros));
        json.key("budgetUs"); json.integer(options.recording.budgetMicros);
        json.key("budgetHit"); json.boolean(recording.budgetHit);
        json.key("budgetExceeded"); json.boolean(recording.budgetExceeded);
        json.key("fallbackSerialCommands");
        json.integer(recording.fallbackSerialCommands);
        json.key("workers");
        json.beginArray();
        for (const WorkerLoad& worker : recording.workers) {
            json.beginObject();
            json.key("worker"); json.integer(worker.workerId);
            json.key("commands"); json.integer(worker.commands);
            json.key("elapsedUs"); json.integer(static_cast<int64_t>(worker.elapsedMicros));
            json.endObject();
        }
        json.endArray();
        json.endObject();

        json.key("adaptive");
        json.beginObject();
        json.key("gpuFrameMs"); json.number(gpuFrameMs);
        json.key("smoothedFrameMs"); json.number(quality.smoothedFrameMs);
        json.key("level"); json.integer(quality.level);
        json.key("renderScale"); json.number(quality.renderScale);
        json.key("shadowScale"); json.number(quality.shadowScale);
        json.key("action"); json.string(qualityActionName(quality.action));
        json.key("historyLength"); json.integer(adaptive.history().size());
        json.endObject();
        json.key("frameHash"); json.integer(static_cast<int64_t>(frameHash));
        json.endObject();

        summary.visibleSum += cull.visible.size();
        summary.occludedSum += cull.occluded.size();
        summary.shadowUpdates += csm.updatedCascades + local.updatedCount;
        summary.localShadowUpdates += local.updatedCount;
        summary.csmUpdates += csm.updatedCascades;
        summary.recordingSumUs += recording.elapsedMicros;
        summary.recordingMaxUs = std::max(summary.recordingMaxUs, recording.elapsedMicros);
        if (recording.budgetExceeded) ++summary.budgetHitFrames;
        summary.minVisible = std::min(summary.minVisible,
                                      static_cast<uint32_t>(cull.visible.size()));
        summary.maxVisible = std::max(summary.maxVisible,
                                      static_cast<uint32_t>(cull.visible.size()));
        summary.minQualityLevel = std::min(summary.minQualityLevel, quality.level);
        summary.maxQualityLevel = std::max(summary.maxQualityLevel, quality.level);

        previousCameraFrozen = cameraFrozen;
        previousMatricesStable = csm.matricesStable;
        previousCamera = camera;
        previousFrameHash = frameHash;
    }
    json.endArray();

    const QualityState& finalQuality = adaptive.state();
    summary.defaultQualityRestored = finalQuality.level == 0;
    json.key("adaptiveHistory");
    writeQualityHistory(json, adaptive.history());
    json.key("summary");
    json.beginObject();
    json.key("frameStatsHash");
    json.integer(static_cast<int64_t>(summary.frameStatsHash));
    json.key("lastFrameHash"); json.integer(static_cast<int64_t>(previousFrameHash));
    json.key("validationPassed"); json.boolean(summary.validationPassed);
    json.key("minVisible"); json.integer(summary.minVisible);
    json.key("maxVisible"); json.integer(summary.maxVisible);
    json.key("averageVisible");
    json.number(static_cast<double>(summary.visibleSum) / static_cast<double>(options.frames));
    json.key("totalOccluded"); json.integer(static_cast<int64_t>(summary.occludedSum));
    json.key("totalCsmUpdates"); json.integer(static_cast<int64_t>(summary.csmUpdates));
    json.key("totalLocalShadowUpdates");
    json.integer(static_cast<int64_t>(summary.localShadowUpdates));
    json.key("totalShadowUpdates"); json.integer(static_cast<int64_t>(summary.shadowUpdates));
    json.key("averageRecordingUs");
    json.number(static_cast<double>(summary.recordingSumUs) /
                static_cast<double>(options.frames));
    json.key("maxRecordingUs"); json.integer(static_cast<int64_t>(summary.recordingMaxUs));
    json.key("budgetHitFrames"); json.integer(summary.budgetHitFrames);
    json.key("longestStableMatrixRun"); json.integer(summary.longestStableMatrixRun);
    json.key("stableWindowStart"); json.integer(summary.stableWindowStart);
    json.key("stableWindowEnd"); json.integer(summary.stableWindowEnd);
    json.key("qualityLevelsSeen"); json.integer(qualityLevelsSeen);
    json.key("levelCounts");
    json.beginArray();
    for (uint32_t count : levelCounts) json.integer(count);
    json.endArray();
    json.key("finalQualityLevel"); json.integer(finalQuality.level);
    json.key("finalRenderScale"); json.number(finalQuality.renderScale);
    json.key("finalShadowScale"); json.number(finalQuality.shadowScale);
    json.key("defaultQualityRestored"); json.boolean(summary.defaultQualityRestored);
    json.key("adaptiveHistoryEntries"); json.integer(adaptive.history().size());
    json.endObject();
    json.endObject();
    writeTextFile(options.outputPath, json.str());
    return summary;
}

} // namespace uid166

