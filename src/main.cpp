#include <algorithm>
#include <cerrno>
#include <bit>
#include <cfenv>
#include <cmath>
#include <clocale>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <locale>
#include <optional>
#include <sstream>
#include <string>

#include "baker/async_writer.h"
#include "baker/error.h"
#include "baker/interrupt.h"
#include "baker/render.h"
#include "baker/scene.h"
#include "baker/snapshot.h"

namespace {

struct Arguments {
    std::filesystem::path input;
    std::filesystem::path output;
    std::filesystem::path snapshot;
    float time = 0.0f;
    bool sync_io = false;
};

void usage(std::ostream& out) {
    out << "usage: baker --input SCENE --output OUT --snapshot SNAP "
           "[--time T] [--sync-io]\n";
}

[[noreturn]] void argument_error(const std::string& message) {
    throw baker::BakerError(baker::ErrorCode::kArgs, message);
}

bool parse_float(const char* text, float& value) {
    char* end = nullptr;
    errno = 0;
    value = std::strtof(text, &end);
    return errno == 0 && end != text && *end == '\0' && std::isfinite(value);
}

bool environment_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        const auto value_after = [&](const char* name) -> const char* {
            if (index + 1 >= argc) {
                argument_error(std::string("missing value for ") + name);
            }
            return argv[++index];
        };

        if (option == "--input") {
            arguments.input = value_after("--input");
        } else if (option == "--output") {
            arguments.output = value_after("--output");
        } else if (option == "--snapshot") {
            arguments.snapshot = value_after("--snapshot");
        } else if (option == "--time") {
            if (!parse_float(value_after("--time"), arguments.time)) {
                argument_error("--time requires a finite number");
            }
        } else if (option == "--sync-io") {
            arguments.sync_io = true;
        } else if (option == "--help" || option == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            argument_error("unknown option '" + option + "'");
        }
    }

    if (arguments.input.empty()) {
        argument_error("--input is required");
    }
    if (arguments.output.empty()) {
        argument_error("--output is required");
    }
    if (arguments.snapshot.empty()) {
        argument_error("--snapshot is required");
    }
    return arguments;
}

void print_layer_pixels(const baker::Stats& stats) {
    std::cout << "[baker] layer_pixels=";
    for (std::size_t index = 0; index < stats.layer_pixels.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << stats.layer_pixels[index];
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    (void)std::setlocale(LC_ALL, "C");
    std::locale::global(std::locale::classic());
    std::cout.imbue(std::locale::classic());
    std::cerr.imbue(std::locale::classic());
    if (std::fesetround(FE_TONEAREST) != 0) {
        std::cerr << "ERR_IO: cannot establish round-to-nearest mode\n";
        return static_cast<int>(baker::ErrorCode::kIo);
    }
    baker::install_signal_handlers();

    try {
        const Arguments arguments = parse_arguments(argc, argv);
        const baker::Scene scene = baker::load_scene(arguments.input);
        const std::uint32_t time_bits =
            std::bit_cast<std::uint32_t>(arguments.time);

        std::cout << "[baker] hash=" << std::hex << std::setw(16)
                  << std::setfill('0') << scene.input_hash << std::dec
                  << std::setfill(' ') << " res=" << scene.width << 'x'
                  << scene.height << " layers=" << scene.layers.size() << '\n';

        baker::SnapshotReadResult snapshot_result =
            baker::read_snapshot_file(arguments.snapshot);
        std::uint32_t first_layer = 0;
        baker::PixelBuffer state =
            baker::make_empty_buffer(scene.width, scene.height);
        baker::Stats stats;
        stats.layer_pixels.assign(scene.layers.size(), 0);

        bool snapshot_hit = false;
        if (snapshot_result.status == baker::SnapshotStatus::kLoaded) {
            baker::Snapshot& snapshot = snapshot_result.snapshot;
            const char* reason = nullptr;
            if (snapshot.input_hash != scene.input_hash) {
                reason = "hash_mismatch";
            } else if (snapshot.width != scene.width ||
                       snapshot.height != scene.height) {
                reason = "resolution_mismatch";
            } else if (snapshot.layer_count != scene.layers.size()) {
                reason = "layer_mismatch";
            } else if (snapshot.time_bits != time_bits) {
                reason = "time_mismatch";
            }

            if (reason != nullptr) {
                std::cout << "[baker] snapshot=discard reason=" << reason << '\n';
            } else {
                snapshot_hit = true;
                first_layer = snapshot.completed_layers;
                state.rgba = std::move(snapshot.pixels);
                if (first_layer == 0) {
                    std::fill(state.rgba.begin(), state.rgba.end(), 0.0f);
                }
                stats.layer_pixels = snapshot.layer_pixels;
                if (snapshot.encoded_degenerate_count) {
                    stats.degenerate_layers = snapshot.degenerate_layers;
                } else {
                    for (std::uint32_t index = 0; index < first_layer; ++index) {
                        if (stats.layer_pixels[index] == 0) {
                            ++stats.degenerate_layers;
                        }
                    }
                }
                std::cout << "[baker] snapshot=hit resume_from=" << first_layer
                          << '\n';
            }
        } else {
            std::cout << "[baker] snapshot=discard reason="
                      << baker::snapshot_status_name(snapshot_result.status)
                      << '\n';
        }

        const bool sync_io =
            arguments.sync_io || environment_flag_enabled("BAKER_SYNC_IO");
        std::size_t queue_capacity = baker::queue_capacity_from_env();
        if (std::getenv("BAKER_QUEUE_CAPACITY") == nullptr) {
            const std::size_t scene_jobs =
                std::min<std::size_t>(scene.layers.size() + 1u, 64u);
            queue_capacity = std::max(queue_capacity, scene_jobs);
        }
        baker::AsyncWriter writer(queue_capacity, !sync_io,
                                  baker::io_delay_ms_from_env());
        writer.start();

        try {
            baker::render_scene(
                scene, baker::RenderOptions{arguments.time}, first_layer, state, stats,
                [&](std::uint32_t completed, const baker::PixelBuffer& pixels,
                    const baker::Stats& current_stats) {
                    const baker::Snapshot snapshot =
                        baker::make_snapshot(scene, completed, arguments.time,
                                             current_stats, pixels);
                    writer.push({arguments.snapshot,
                                 baker::encode_snapshot(snapshot), "snapshot"});
                });

            std::vector<std::uint8_t> output = baker::encode_output(state);
            writer.push({arguments.output, std::move(output), "output"});
            writer.finish();
        } catch (...) {
            writer.abort();
            throw;
        }

        std::cout << "[baker] degenerate_layers=" << stats.degenerate_layers << '\n';
        print_layer_pixels(stats);
        std::cout << "[baker] output=" << arguments.output.string() << '\n';
        return static_cast<int>(baker::ErrorCode::kOk);
    } catch (const baker::Interrupted& error) {
        std::cerr << "ERR_INTERRUPTED: " << error.what() << '\n';
        return 128 + error.signal_number();
    } catch (const baker::BakerError& error) {
        std::cerr << baker::error_name(error.code()) << ": " << error.what() << '\n';
        return static_cast<int>(error.code());
    } catch (const std::exception& error) {
        std::cerr << "ERR_IO: " << error.what() << '\n';
        return static_cast<int>(baker::ErrorCode::kIo);
    }
}
