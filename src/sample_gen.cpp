#include "sample_gen.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "error.h"
#include "wav.h"

namespace fs = std::filesystem;

namespace vr {

namespace {

struct RawSpec {
    const char* file;
    std::uint64_t frames;
    unsigned frequency;
    std::uint32_t seed;
    std::uint32_t amplitude;
};

constexpr std::array<RawSpec, 4> kRawSpecs = {{
    {"caster_alpha.raw", 180000, 311, 0xA11CEu, 12000},
    {"caster_beta.raw", 150000, 523, 0xB0B5u, 28000},
    {"caster_gamma.raw", 120000, 197, 0xC0FFEEu, 14000},
    {"caster_silent.raw", 0, 1, 0xDEADu, 0},
}};

std::uint32_t next_random(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

std::int16_t make_sample(std::uint64_t index, std::uint64_t frames,
                         unsigned frequency, std::uint32_t seed,
                         std::uint32_t amplitude) {
    if (frames == 0) {
        return 0;
    }
    const std::uint64_t phase =
        (index * static_cast<std::uint64_t>(frequency)) % 48000u;
    std::int64_t triangle = 0;
    if (phase < 24000u) {
        triangle = static_cast<std::int64_t>(phase) * 2 - 24000;
    } else {
        triangle = 24000 - static_cast<std::int64_t>(phase - 24000u) * 2;
    }
    std::uint32_t state = seed ^ static_cast<std::uint32_t>(index * 2654435761u);
    const std::int64_t noise =
        static_cast<std::int64_t>(next_random(state) >> 16) - 32768;
    const std::int64_t mixed = triangle * 3 / 4 + noise / 8;
    std::int64_t value = mixed * static_cast<std::int64_t>(amplitude) / 24000;

    const std::uint64_t fade = std::min<std::uint64_t>(2048, frames / 4);
    if (fade > 0) {
        std::uint64_t envelope = 32768;
        if (index < fade) {
            envelope = index * 32768u / fade;
        }
        const std::uint64_t tail = frames > index ? frames - index : 0;
        if (tail < fade) {
            envelope = std::min<std::uint64_t>(envelope, tail * 32768u / fade);
        }
        value = value * static_cast<std::int64_t>(envelope) / 32768;
    }
    value = std::clamp<std::int64_t>(value, -32768, 32767);
    return static_cast<std::int16_t>(value);
}

void write_raw(const fs::path& path, const RawSpec& spec) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw AppError(ErrorCode::Io, "cannot create sample raw file: " + path.string());
    }
    for (std::uint64_t i = 0; i < spec.frames; ++i) {
        const std::int16_t sample = make_sample(
            i, spec.frames, spec.frequency, spec.seed, spec.amplitude);
        const std::uint16_t u = static_cast<std::uint16_t>(sample);
        const char bytes[2] = {
            static_cast<char>(u & 0xFFu),
            static_cast<char>((u >> 8) & 0xFFu),
        };
        out.write(bytes, 2);
    }
    if (!out) {
        throw AppError(ErrorCode::Io, "cannot write sample raw file: " + path.string());
    }
}

void write_text(const fs::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw AppError(ErrorCode::Io, "cannot create sample file: " + path.string());
    }
    out << text;
    if (!out) {
        throw AppError(ErrorCode::Io, "cannot write sample file: " + path.string());
    }
}

}  // namespace

void generate_samples(const fs::path& output_dir) {
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        throw AppError(ErrorCode::OutputUnwritable,
                       "cannot create sample directory: " + output_dir.string());
    }

    for (const auto& spec : kRawSpecs) {
        write_raw(output_dir / spec.file, spec);
    }

    write_text(output_dir / "events.json",
        "{\n"
        "  \"sample_rate\": 48000,\n"
        "  \"duration_ms\": 8100,\n"
        "  \"events\": [\n"
        "    {\"type\": \"kill\", \"match_time_ms\": 1000},\n"
        "    {\"type\": \"tower\", \"match_time_ms\": 3000},\n"
        "    {\"type\": \"kill\", \"match_time_ms\": 6000},\n"
        "    {\"type\": \"tower\", \"match_time_ms\": 7000}\n"
        "  ]\n"
        "}\n");

    write_text(output_dir / "caster_alpha.json",
        "{\n"
        "  \"commentary_id\": \"caster_alpha\",\n"
        "  \"raw_file\": \"caster_alpha.raw\",\n"
        "  \"sample_rate\": 48000,\n"
        "  \"start_frame\": 0,\n"
        "  \"end_frame\": 220000,\n"
        "  \"gain_db\": -6.0,\n"
        "  \"events\": [\n"
        "    {\"type\": \"kill\", \"match_time_ms\": 1000, \"local_frame\": 0},\n"
        "    {\"type\": \"tower\", \"match_time_ms\": 3000, \"local_frame\": 96000}\n"
        "  ]\n"
        "}\n");

    write_text(output_dir / "caster_beta.json",
        "{\n"
        "  \"commentary_id\": \"caster_beta\",\n"
        "  \"raw_file\": \"caster_beta.raw\",\n"
        "  \"sample_rate\": 48000,\n"
        "  \"start_frame\": 4096,\n"
        "  \"end_frame\": 160000,\n"
        "  \"gain_db\": 6.0,\n"
        "  \"events\": [\n"
        "    {\"type\": \"tower\", \"match_time_ms\": 3000, \"local_frame\": 4096}\n"
        "  ]\n"
        "}\n");

    write_text(output_dir / "caster_gamma.json",
        "{\n"
        "  \"commentary_id\": \"caster_gamma\",\n"
        "  \"raw_file\": \"caster_gamma.raw\",\n"
        "  \"sample_rate\": 48000,\n"
        "  \"start_frame\": 8192,\n"
        "  \"end_frame\": 110000,\n"
        "  \"gain_db\": -2.0,\n"
        "  \"events\": [\n"
        "    {\"type\": \"kill\", \"match_time_ms\": 6000, \"local_frame\": 8192}\n"
        "  ]\n"
        "}\n");

    write_text(output_dir / "caster_silent.json",
        "{\n"
        "  \"commentary_id\": \"caster_silent\",\n"
        "  \"raw_file\": \"caster_silent.raw\",\n"
        "  \"sample_rate\": 48000,\n"
        "  \"start_frame\": 0,\n"
        "  \"end_frame\": 1000,\n"
        "  \"gain_db\": 0.0,\n"
        "  \"events\": [\n"
        "    {\"type\": \"tower\", \"match_time_ms\": 7000, \"local_frame\": 0}\n"
        "  ]\n"
        "}\n");
}

}  // namespace vr

