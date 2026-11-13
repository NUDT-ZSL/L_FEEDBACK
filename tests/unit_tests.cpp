#include "skinbake/bake.hpp"
#include "skinbake/error.hpp"
#include "skinbake/fs.hpp"
#include "skinbake/scenarios.hpp"
#include "skinbake/snapshot.hpp"
#include "skinbake/state_machine.hpp"
#include "skinbake/verify.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace skinbake;

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++g_failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

template <typename Fn>
void expect_error(ErrorCode expected, Fn&& fn, const std::string& label) {
    try {
        fn();
        check(false, label + " did not throw");
    } catch (const BusinessError& error) {
        check(error.code() == expected,
              label + " returned " + error_code_name(error.code()) +
                  " instead of " + error_code_name(expected));
    }
}

std::uint32_t float_bits(float value) {
    std::uint32_t bitsValue = 0;
    std::memcpy(&bitsValue, &value, sizeof(bitsValue));
    return bitsValue;
}

Curve constant_curve(double value) {
    Keyframe key;
    key.timeMs = 0;
    key.value = value;
    return Curve({key});
}

Clip constant_clip(std::size_t boneCount, std::int64_t duration) {
    Clip clip;
    clip.name = "root";
    clip.durationMs = duration;
    clip.loop = true;
    clip.bones.resize(boneCount);
    for (std::size_t bone = 0; bone < boneCount; ++bone) {
        for (Curve& curve : clip.bones[bone].channels) {
            curve = constant_curve(0.0);
        }
        clip.bones[bone].channels[ScaleX] = constant_curve(1.0);
        clip.bones[bone].channels[ScaleY] = constant_curve(1.0);
    }
    return clip;
}

void test_curve_evaluator() {
    std::vector<Keyframe> keys(3);
    keys[0].timeMs = 0;
    keys[0].value = 2.0;
    keys[1].timeMs = 300;
    keys[1].value = 5.0;
    keys[2].timeMs = 900;
    keys[2].value = -1.0;
    keys[0].outTangent = Tangent{100.0, 2.0};
    keys[1].inTangent = Tangent{100.0, 1.0};
    keys[1].outTangent = Tangent{200.0, -1.5};
    keys[2].inTangent = Tangent{200.0, 2.5};
    Curve curve(keys);
    curve.validate(CurveContext{"curve", 0, 0});
    curve.prepare();

    check(float_bits(curve.evaluate(-50.0)) ==
              float_bits(2.0),
          "curve must constantly extend before the first key");
    check(float_bits(curve.evaluate(2000.0)) ==
              float_bits(-1.0),
          "curve must constantly extend after the last key");
    check(float_bits(curve.evaluate(0.0)) == float_bits(2.0),
          "curve first key exact value");
    check(float_bits(curve.evaluate(300.0)) == float_bits(5.0),
          "curve interior key exact value");
    const float first = curve.evaluate_float(177.0);
    for (int i = 0; i < 1000; ++i) {
        check(float_bits(curve.evaluate_float(177.0)) ==
                  float_bits(first),
              "repeated curve evaluation must be bit-identical");
    }
    check(first > 2.0f && first < 5.0f, "bezier interpolation stays in range");
}

void test_curve_errors() {
    Curve duplicate({
        Keyframe{0, 0.0, Tangent{}, Tangent{100.0, 0.0}},
        Keyframe{0, 1.0, Tangent{100.0, 0.0}, Tangent{}},
    });
    expect_error(ErrorCode::CurveTimeNotIncreasing,
                 [&]() { duplicate.validate(CurveContext{"bad", 0, 0}); },
                 "non-increasing curve time");

    Curve denominator({
        Keyframe{0, 0.0, Tangent{}, Tangent{0.0, 0.4}},
        Keyframe{300, 1.0, Tangent{100.0, 0.3}, Tangent{}},
    });
    expect_error(
        ErrorCode::BezierDenominatorZero,
        [&]() { denominator.validate(CurveContext{"bad", 0, 0}); },
        "bezier denominator");
}

void test_time_rounding_and_loop() {
    check(rounded_frame_time_ms(1, 80) == 13,
          "12.5 ms must round half away from zero to 13");
    check(rounded_frame_time_ms(3, 80) == 38,
          "37.5 ms must round half away from zero to 38");
    check(wrap_playback_time(1000, 1000, true) == 0,
          "loop boundary must wrap to zero");
    check(wrap_playback_time(-1, 1000, true) == 999,
          "negative loop time must use positive modulo");
    for (std::uint32_t frame = 0; frame < 10000; ++frame) {
        const std::int64_t time = rounded_frame_time_ms(frame, 60);
        check(wrap_playback_time(time, 1000, true) == time % 1000,
              "10000-frame playback must not accumulate drift");
    }
}

void test_skinning_and_weights() {
    Scene scene;
    scene.fps = 60;
    scene.defaultClip = "root";
    scene.bones.resize(2);
    scene.bones[0].name = "root";
    scene.bones[0].parent = -1;
    scene.bones[1].name = "child";
    scene.bones[1].parent = 0;
    Clip clip = constant_clip(2, 1000);
    clip.bones[0].channels[TranslationX] = constant_curve(10.0);
    clip.bones[1].channels[TranslationX] = constant_curve(5.0);
    scene.clips.push_back(clip);
    Vertex vertex;
    vertex.position = Vec2{1.0, 2.0};
    vertex.influenceCount = 2;
    vertex.boneIndices = {{0, 1, -1, -1}};
    vertex.weights = {{0.25, 0.75, 0.0, 0.0}};
    scene.vertices.push_back(vertex);

    const auto prepared = prepare_scene(scene);
    const Pose pose = evaluate_clip_pose(*prepared, 0, 0);
    check(std::fabs(pose.vertices[0].x - 14.75) < 1e-12,
          "hierarchical skinning x position");
    check(std::fabs(pose.vertices[0].y - 2.0) < 1e-12,
          "hierarchical skinning y position");
    check(std::fabs(pose.boneWorld[1].at(0, 2) - 15.0) < 1e-12,
          "child world translation accumulates parent");

    Scene bad = scene;
    bad.vertices[0].weights = {{0.2, 0.6, 0.0, 0.0}};
    expect_error(ErrorCode::WeightSumNotOne,
                 [&]() { validate_scene(bad); },
                 "weight sum not one");
    bad.vertices[0].weights = {{0.0, 0.0, 0.0, 0.0}};
    expect_error(ErrorCode::WeightSumZero,
                 [&]() { validate_scene(bad); }, "weight sum zero");
}

void test_scenario_errors() {
    expect_error(
        ErrorCode::CurveTimeNotIncreasing,
        [&]() { prepare_timeline(make_scenario("curve-time").definition); },
        "curve-time scenario");
    expect_error(
        ErrorCode::BezierDenominatorZero,
        [&]() {
            prepare_timeline(make_scenario("bezier-denominator").definition);
        },
        "bezier-denominator scenario");
    expect_error(
        ErrorCode::WeightSumZero,
        [&]() { prepare_timeline(make_scenario("weight-zero").definition); },
        "weight-zero scenario");
    expect_error(
        ErrorCode::WeightSumNotOne,
        [&]() { prepare_timeline(make_scenario("weight-not-one").definition); },
        "weight-not-one scenario");
    expect_error(
        ErrorCode::SkeletonCycle,
        [&]() {
            prepare_timeline(make_scenario("skeleton-cycle").definition);
        },
        "skeleton-cycle scenario");
}

void test_snapshot_roundtrip_and_errors(const std::filesystem::path& work) {
    Snapshot snapshot;
    snapshot.version = 7;
    snapshot.fps = 60;
    snapshot.fingerprint = 0x123456789abcdef0ull;
    snapshot.totalFrames = 2;
    snapshot.boneCount = 1;
    snapshot.vertexCount = 2;
    snapshot.complete = true;
    FrameRecord frame0;
    frame0.frameIndex = 0;
    frame0.timeMs = 0;
    frame0.boneMatrices = {1, 0, 2, 0, 0, 1, 3, 0, 0, 0, 1, 0};
    frame0.vertices = {4, 5, 6, 7};
    FrameRecord frame1 = frame0;
    frame1.frameIndex = 1;
    frame1.timeMs = 17;
    frame1.vertices = {8, 9, 10, 11};
    snapshot.frames = {frame0, frame1};

    const std::string good = (work / "roundtrip.snapshot").string();
    write_snapshot_atomic(good, snapshot);
    const Snapshot loaded = load_snapshot(good, 7, snapshot.fingerprint);
    check(loaded.frames.size() == 2, "snapshot frame count roundtrip");
    check(float_bits(loaded.frames[1].vertices[0]) == float_bits(8.0f),
          "snapshot vertex payload roundtrip");

    expect_error(
        ErrorCode::SnapshotVersionMismatch,
        [&]() { (void)load_snapshot(good, 9); }, "snapshot version mismatch");

    std::vector<std::uint8_t> bytes = read_file_bytes(good);
    bytes.back() ^= 0x5a;
    const std::string corrupt = (work / "checksum.snapshot").string();
    atomic_write_bytes(corrupt, bytes);
    expect_error(
        ErrorCode::SnapshotChecksumMismatch,
        [&]() { (void)load_snapshot(corrupt); },
        "snapshot checksum mismatch");

    bytes = read_file_bytes(good);
    bytes[0] = 'X';
    const std::string badMagic = (work / "magic.snapshot").string();
    atomic_write_bytes(badMagic, bytes);
    expect_error(ErrorCode::SnapshotMagicMismatch,
                 [&]() { (void)load_snapshot(badMagic); },
                 "snapshot magic mismatch");

    check(load_snapshot(good).frames.size() == 2,
          "failed loads must not damage a valid snapshot");
}

void test_blend_and_snapshot_restore() {
    Scenario scenario = make_scenario("transition");
    const auto timeline = prepare_timeline(scenario.definition);
    const TimelineFrame frame12 = timeline->evaluate(12);
    const TimelineFrame frame13 = timeline->evaluate(13);
    check(frame12.blendWeight == 0.0, "switch frame starts at blend weight zero");
    check(std::fabs(frame13.blendWeight - 17.0 / 200.0) < 1e-15,
          "blend weight follows rounded playback time");

    StateMachine machine(timeline->prepared_shared(), "idle");
    machine.advance(200);
    machine.switch_clip("action", 200);
    const TimelineFrame switched = machine.sample();
    check(switched.blendWeight == 0.0,
          "pose snapshot keeps switch frame visually continuous");
    for (std::size_t i = 0; i < switched.pose.vertices.size(); ++i) {
        check(std::fabs(switched.pose.vertices[i].x -
                        frame12.pose.vertices[i].x) < 1e-12 &&
                  std::fabs(switched.pose.vertices[i].y -
                            frame12.pose.vertices[i].y) < 1e-12,
              "switch pose equals timeline old snapshot");
    }
    machine.advance(17);
    const TimelineFrame advanced = machine.sample();
    check(std::fabs(advanced.blendWeight - 17.0 / 200.0) < 1e-15,
          "state machine blend weight advances by playback time");
    check(std::fabs(advanced.pose.vertices[8].x -
                    frame13.pose.vertices[8].x) < 1e-12,
          "state machine and timeline blend agree");
}

BakeOptions options_for(const std::string& path, std::uint32_t workers,
                        std::uint32_t frames) {
    BakeOptions options;
    options.outputPath = path;
    options.version = 1;
    options.totalFrames = frames;
    options.requestedWorkers = workers;
    options.blockFrames = 8;
    return options;
}

void test_parallel_determinism_and_resume(
    const std::filesystem::path& work) {
    Scenario scenario = make_scenario("idle");
    const auto timeline = prepare_timeline(scenario.definition);
    const std::string serial = (work / "serial.snapshot").string();
    const std::string parallel = (work / "parallel.snapshot").string();
    const BakeReport first =
        bake(*timeline, options_for(serial, 1, 64));
    const BakeReport second =
        bake(*timeline, options_for(parallel, 32, 64));
    check(first.newFramesCommitted == 64 &&
              second.newFramesCommitted == 64,
          "parallel and serial bakes commit every frame");
    check(read_file_bytes(serial) == read_file_bytes(parallel),
          "parallel output must be byte-identical to serial output");

    const VerificationReport verify =
        verify_snapshot(*timeline, serial, 1e-5, 1);
    check(verify.passed && verify.byteIdentical,
          "baked snapshot matches from-scratch reference");

    Snapshot complete = load_snapshot(serial);
    Snapshot prefix = complete;
    prefix.frames.resize(7);
    prefix.complete = false;
    write_snapshot_atomic(progress_path_for(serial), prefix);
    const BakeReport resumed =
        bake(*timeline, options_for(serial, 8, 64));
    check(resumed.resumed && resumed.resumeFrame == 7 &&
              resumed.newFramesCommitted == 57,
          "resume starts at the first missing frame");
    check(read_file_bytes(serial) == read_file_bytes(parallel),
          "resumed output must remain byte-identical");
}

double reference_blend_at(const PreparedScene& prepared,
                          const TransitionSpec& transition,
                          std::uint32_t frame,
                          std::uint32_t fps,
                          std::size_t vertex) {
    const std::int64_t switchTime =
        rounded_frame_time_ms(transition.switchFrame, fps);
    const std::int64_t frameTime = rounded_frame_time_ms(frame, fps);
    const std::size_t idle = clip_index(prepared, "idle");
    const std::size_t action = clip_index(prepared, "action");
    const Pose oldPose = evaluate_clip_pose(prepared, idle, switchTime);
    const Pose newPose =
        evaluate_clip_pose(prepared, action, frameTime - switchTime);
    const double weight = std::min(
        1.0, static_cast<double>(frameTime - switchTime) /
                 static_cast<double>(transition.blendWindowMs));
    return oldPose.vertices[vertex].x +
           (newPose.vertices[vertex].x - oldPose.vertices[vertex].x) * weight;
}

void test_linear_vertex_blend() {
    Scenario scenario = make_scenario("transition");
    const auto timeline = prepare_timeline(scenario.definition);
    const TimelineFrame frame = timeline->evaluate(20);
    const double expected =
        reference_blend_at(*timeline->prepared_shared(),
                           scenario.definition.transitions[0], 20, 60, 4);
    check(std::fabs(frame.pose.vertices[4].x - expected) < 1e-12,
          "blend window linearly interpolates vertex positions");
}

}

int main() {
    try {
        const std::filesystem::path work =
            std::filesystem::current_path() / "unit-test-work";
        std::filesystem::remove_all(work);
        std::filesystem::create_directories(work);

        test_curve_evaluator();
        test_curve_errors();
        test_time_rounding_and_loop();
        test_skinning_and_weights();
        test_scenario_errors();
        test_snapshot_roundtrip_and_errors(work);
        test_blend_and_snapshot_restore();
        test_parallel_determinism_and_resume(work);
        test_linear_vertex_blend();
    } catch (const std::exception& error) {
        std::cerr << "UNCAUGHT: " << error.what() << '\n';
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " unit checks failed\n";
        return 1;
    }
    std::cout << "PASS all skinbake unit checks\n";
    return 0;
}

