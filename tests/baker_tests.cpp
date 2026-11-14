#include <bit>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "baker/atomic_io.h"
#include "baker/byte_io.h"
#include "baker/interrupt.h"
#include "baker/model.h"
#include "baker/render.h"
#include "baker/scene.h"
#include "baker/scene_generator.h"
#include "baker/snapshot.h"

namespace {

int failures = 0;

void check(bool condition, const char* expression, int line) {
    if (!condition) {
        std::cerr << "FAIL line " << line << ": " << expression << '\n';
        ++failures;
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

template <typename Callable>
void expect_error(baker::ErrorCode expected, Callable&& callable, int line) {
    try {
        callable();
        std::cerr << "FAIL line " << line << ": expected " << baker::error_name(expected)
                  << '\n';
        ++failures;
    } catch (const baker::BakerError& error) {
        if (error.code() != expected) {
            std::cerr << "FAIL line " << line << ": got " << baker::error_name(error.code())
                      << ", wanted " << baker::error_name(expected) << '\n';
            ++failures;
        }
    }
}

std::vector<std::uint8_t> generated_scene() {
    baker::GeneratorOptions options;
    options.width = 96;
    options.height = 72;
    options.layers = 4;
    options.tex_size = 16;
    options.seed = 1234;
    return baker::generate_scene_bytes(options);
}

void test_generator_and_loader() {
    const std::string fnv_probe = "a";
    CHECK(baker::fnv1a64({}) == 14695981039346656037ull);
    CHECK(baker::fnv1a64(std::span<const std::uint8_t>(
              reinterpret_cast<const std::uint8_t*>(fnv_probe.data()),
              fnv_probe.size())) == 0xaf63dc4c8601ec8cull);

    const std::vector<std::uint8_t> first = generated_scene();
    const std::vector<std::uint8_t> second = generated_scene();
    CHECK(first == second);

    baker::Scene scene = baker::parse_scene(first);
    scene.input_hash = baker::fnv1a64(first);
    CHECK(scene.width == 96);
    CHECK(scene.height == 72);
    CHECK(scene.layers.size() == 4);
    CHECK(scene.texels.size() == 16u * 16u * 4u);
    CHECK(scene.layers.front().playback == baker::Playback::kClamp);
    CHECK(scene.layers.back().vertex_keys.size() == 3);
}

void test_business_errors() {
    auto bytes = generated_scene();

    auto bad_magic = bytes;
    bad_magic[0] = static_cast<std::uint8_t>('X');
    expect_error(
        baker::ErrorCode::kMagic,
        [&] { (void)baker::parse_scene(bad_magic); }, __LINE__);

    auto bad_version = bytes;
    bad_version[4] = 99;
    expect_error(
        baker::ErrorCode::kVersion,
        [&] { (void)baker::parse_scene(bad_version); }, __LINE__);

    auto bad_value = bytes;
    bad_value[8] = 0;
    bad_value[9] = 0;
    bad_value[10] = 0;
    bad_value[11] = 0;
    expect_error(
        baker::ErrorCode::kValue,
        [&] { (void)baker::parse_scene(bad_value); }, __LINE__);

    std::vector<std::uint8_t> truncated(bytes.begin(), bytes.begin() + 100);
    expect_error(
        baker::ErrorCode::kLength,
        [&] { (void)baker::parse_scene(truncated); }, __LINE__);

    auto trailing = bytes;
    trailing.push_back(0);
    expect_error(
        baker::ErrorCode::kLength,
        [&] { (void)baker::parse_scene(trailing); }, __LINE__);

    auto bad_texture = bytes;
    const std::uint32_t nan_bits = 0x7fc00000u;
    std::memcpy(bad_texture.data() + 24, &nan_bits, sizeof(nan_bits));
    expect_error(
        baker::ErrorCode::kValue,
        [&] { (void)baker::parse_scene(bad_texture); }, __LINE__);

    auto bad_playback = bytes;
    const std::size_t first_layer_offset = 24u + 16u * 16u * 4u * 4u;
    const std::size_t playback_offset = first_layer_offset + 4u + 36u + 4u + 8u;
    const std::uint32_t invalid_playback = 99u;
    std::memcpy(bad_playback.data() + playback_offset, &invalid_playback,
                sizeof(invalid_playback));
    expect_error(
        baker::ErrorCode::kValue,
        [&] { (void)baker::parse_scene(bad_playback); }, __LINE__);
}

void test_deterministic_render() {
    const std::vector<std::uint8_t> bytes = generated_scene();
    baker::Scene scene = baker::parse_scene(bytes);
    scene.input_hash = baker::fnv1a64(bytes);

    auto render_once = [&] {
        baker::PixelBuffer state = baker::make_empty_buffer(scene.width, scene.height);
        baker::Stats stats;
        stats.layer_pixels.assign(scene.layers.size(), 0);
        baker::render_scene(scene, baker::RenderOptions{0.25f}, 0, state, stats, {});
        return std::pair{state, stats};
    };

    const auto first = render_once();
    const auto second = render_once();
    CHECK(first.first.rgba == second.first.rgba);
    CHECK(first.second.layer_pixels == second.second.layer_pixels);
    CHECK(first.second.degenerate_layers == 1);
    CHECK(first.second.layer_pixels.back() == 0);

    bool all_finite = true;
    for (const float value : first.first.rgba) {
        all_finite = all_finite && std::isfinite(value) && value >= 0.0f && value <= 1.0f;
    }
    CHECK(all_finite);
}

void test_animation_modes() {
    baker::Layer layer;
    layer.playback = baker::Playback::kLoop;
    layer.vertex_keys = {
        {0.0f, {}},
        {1.0f, {{{2.0f, 0.0f}, {2.0f, 0.0f}, {2.0f, 0.0f}, {2.0f, 0.0f}}}},
    };
    layer.opacity_keys = {{0.0f, 0.2f}, {1.0f, 0.8f}};
    layer.base = {{{-2.0f, -2.0f, 2.0f},
                   {2.0f, -2.0f, 2.0f},
                   {2.0f, 2.0f, 2.0f},
                   {-2.0f, 2.0f, 2.0f}}};
    layer.uv = {{{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}}};
    layer.albedo = {1.0f, 1.0f, 1.0f};
    layer.roughness = 1.0f;
    layer.specular = 0.0f;

    baker::Scene scene;
    scene.width = 16;
    scene.height = 16;
    scene.tex_size = 1;
    scene.texels = {1.0f, 1.0f, 1.0f, 1.0f};
    scene.layers = {layer};
    scene.input_hash = 1;

    const auto render_at = [&](float time, baker::Playback playback) {
        scene.layers[0].playback = playback;
        baker::PixelBuffer state = baker::make_empty_buffer(16, 16);
        baker::Stats stats;
        stats.layer_pixels.assign(1, 0);
        baker::render_scene(scene, baker::RenderOptions{time}, 0, state, stats, {});
        return state;
    };

    const auto loop_start = render_at(0.0f, baker::Playback::kLoop);
    const auto loop_wrapped = render_at(1.0f, baker::Playback::kLoop);
    const auto clamped_end = render_at(1.0f, baker::Playback::kClamp);
    CHECK(loop_start.rgba == loop_wrapped.rgba);
    CHECK(loop_start.rgba != clamped_end.rgba);

    scene.layers[0].vertex_keys.resize(1);
    scene.layers[0].opacity_keys = {{0.0f, 0.4f}};
    const auto single_a = render_at(0.0f, baker::Playback::kLoop);
    const auto single_b = render_at(123.5f, baker::Playback::kClamp);
    CHECK(single_a.rgba == single_b.rgba);
}

void test_output_channels() {
    baker::PixelBuffer pixels;
    pixels.width = 1;
    pixels.height = 1;
    pixels.rgba = {0.1f, 0.2f, 0.3f, 0.4f};
    const std::vector<std::uint8_t> encoded = baker::encode_output(pixels);
    CHECK(encoded.size() == 20);
    CHECK(encoded[0] == 'Q');
    CHECK(encoded[1] == 'R');
    CHECK(encoded[2] == 'B');
    CHECK(encoded[3] == 'A');
    CHECK(encoded[16] == 25);
    CHECK(encoded[17] == 51);
    CHECK(encoded[18] == 76);
    CHECK(encoded[19] == 102);
}

void test_snapshot_roundtrip() {
    const std::vector<std::uint8_t> bytes = generated_scene();
    baker::Scene scene = baker::parse_scene(bytes);
    scene.input_hash = baker::fnv1a64(bytes);
    baker::PixelBuffer pixels = baker::make_empty_buffer(scene.width, scene.height);
    pixels.rgba[0] = 0.25f;
    pixels.rgba[1] = 0.5f;
    pixels.rgba[2] = 0.75f;
    pixels.rgba[3] = 1.0f;
    baker::Stats stats;
    stats.layer_pixels = {123, 0, 456, 789};
    stats.degenerate_layers = 1;

    baker::Snapshot snapshot =
        baker::make_snapshot(scene, 3, 0.25f, stats, pixels);
    const std::vector<std::uint8_t> encoded = baker::encode_snapshot(snapshot);
    const std::filesystem::path path = "unit-snapshot.tmp";
    baker::write_atomic_file(path, encoded, {});
    const baker::SnapshotReadResult read = baker::read_snapshot_file(path);
    std::filesystem::remove(path);

    CHECK(read.status == baker::SnapshotStatus::kLoaded);
    CHECK(read.snapshot.version == baker::kSnapshotVersion);
    CHECK(read.snapshot.input_hash == scene.input_hash);
    CHECK(read.snapshot.completed_layers == 3);
    CHECK(read.snapshot.time_bits == std::bit_cast<std::uint32_t>(0.25f));
    CHECK(read.snapshot.degenerate_layers == 1);
    CHECK(read.snapshot.layer_pixels == stats.layer_pixels);
    CHECK(read.snapshot.pixels == pixels.rgba);

    baker::Snapshot legacy = snapshot;
    legacy.version = baker::kLegacySnapshotVersion;
    const std::filesystem::path legacy_path = "unit-legacy-snapshot.tmp";
    baker::write_atomic_file(legacy_path, baker::encode_snapshot(legacy), {});
    const baker::SnapshotReadResult legacy_read =
        baker::read_snapshot_file(legacy_path);
    std::filesystem::remove(legacy_path);
    CHECK(legacy_read.status == baker::SnapshotStatus::kLoaded);
    CHECK(legacy_read.snapshot.legacy_stats);
    CHECK(legacy_read.snapshot.time_bits == 0);
}

void test_atomic_interrupt_cleanup() {
    const std::filesystem::path target = "unit-atomic-target.tmp";
    std::filesystem::remove(target);
    baker::request_interrupt_for_tests(SIGINT);
    try {
        const std::vector<std::uint8_t> bytes(1024, 42);
        baker::write_atomic_file(target, bytes, {});
        CHECK(false);
    } catch (const baker::Interrupted&) {
        CHECK(true);
    }
    baker::g_signal_number.store(0);
    CHECK(!std::filesystem::exists(target));

    std::size_t temp_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(".")) {
        if (entry.path().filename().string().starts_with("unit-atomic-target.tmp.tmp.")) {
            ++temp_count;
        }
    }
    CHECK(temp_count == 0);
}

}  // namespace

int main() {
    test_generator_and_loader();
    test_business_errors();
    test_deterministic_render();
    test_animation_modes();
    test_output_channels();
    test_snapshot_roundtrip();
    test_atomic_interrupt_cleanup();

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "all focused tests passed\n";
    return 0;
}
