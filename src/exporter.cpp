#include "exporter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "error.h"
#include "json.h"
#include "sha256.h"
#include "wav.h"

namespace fs = std::filesystem;

namespace vr {

namespace {

constexpr std::uint64_t kBlockFrames = 1024;
constexpr const char* kStateFile = ".voice-replay-state.json";
constexpr const char* kManifestFile = "manifest.json";
constexpr const char* kStateTempFile = ".voice-replay-state.json.tmp";
constexpr const char* kManifestTempFile = "manifest.json.tmp";
constexpr const char* kManifestFormat = "voice-replay-manifest-v1";

struct MatchEvent {
    std::string type;
    double time_ms = 0.0;
    std::int64_t sample = 0;
};

struct MarkerEvent {
    std::string type;
    double time_ms = 0.0;
    std::int64_t local_frame = 0;
};

struct TrackJob {
    fs::path meta_path;
    fs::path raw_path;
    std::string meta_name;
    std::string raw_logical_name;
    std::string id;
    std::string output_rel;
    std::uint64_t raw_frames = 0;
    std::int64_t start_frame = 0;
    std::int64_t end_frame = 0;
    double gain_db = 0.0;
    double gain_linear = 1.0;
    std::vector<MarkerEvent> events;
    std::int64_t alignment_offset = 0;
    std::uint64_t used_frames = 0;
    std::uint64_t requested_frames = 0;
    bool truncated_to_raw = false;
    std::string anchor_type;
    double anchor_time_ms = 0.0;
};

struct Job {
    fs::path input_dir;
    fs::path output_dir;
    fs::path events_path;
    std::vector<MatchEvent> match_events;
    std::vector<TrackJob> tracks;
    double duration_ms = 0.0;
    std::uint64_t total_frames = 0;
    std::uint64_t total_blocks = 0;
    std::string input_sha256;
    std::string job_sha256;
};

struct TrackRuntime {
    std::uint64_t clipped_samples = 0;
};

struct ResumeState {
    std::uint64_t completed_blocks = 0;
    std::uint64_t master_clipped_samples = 0;
    std::string master_prefix_sha256;
    std::vector<std::string> track_prefix_sha256;
    std::vector<std::uint64_t> track_clipped_samples;
    std::string input_sha256;
    std::string job_sha256;
};

std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw AppError(ErrorCode::Io, "cannot open file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw AppError(ErrorCode::Io, "cannot read file: " + path.string());
    }
    return buffer.str();
}

Json load_json(const fs::path& path, ErrorCode parse_code) {
    std::string text = read_file(path);
    if (text.size() >= 3 &&
        static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB &&
        static_cast<unsigned char>(text[2]) == 0xBF) {
        text.erase(0, 3);
    }
    Json value;
    std::string error;
    if (!parse_json(text, value, error)) {
        throw AppError(parse_code,
                       "invalid JSON in " + path.string() + ": " + error);
    }
    return value;
}

const Json& require_object(const Json& value, const std::string& where) {
    if (!value.is_object()) {
        throw AppError(ErrorCode::MetaInvalid, where + " must be a JSON object");
    }
    return value;
}

const Json* find_field(const Json& object,
                       std::initializer_list<std::string_view> names) {
    return object.find_any(names);
}

const Json& require_field(const Json& object,
                          std::initializer_list<std::string_view> names,
                          const std::string& where) {
    const Json* value = find_field(object, names);
    if (value == nullptr) {
        std::string joined;
        for (const auto name : names) {
            if (!joined.empty()) {
                joined += "/";
            }
            joined.append(name);
        }
        throw AppError(ErrorCode::MetaInvalid, where + " is missing field " + joined);
    }
    return *value;
}

std::string require_string(const Json& object,
                           std::initializer_list<std::string_view> names,
                           const std::string& where) {
    const Json& value = require_field(object, names, where);
    if (!value.is_string() || value.string.empty()) {
        throw AppError(ErrorCode::MetaInvalid, where + " has an invalid string field");
    }
    return value.string;
}

double require_finite_number(const Json& object,
                             std::initializer_list<std::string_view> names,
                             const std::string& where) {
    const Json& value = require_field(object, names, where);
    if (!value.is_number() || !std::isfinite(value.number)) {
        throw AppError(ErrorCode::MetaInvalid,
                       where + " has a non-finite or non-numeric value");
    }
    return value.number;
}

std::int64_t require_integer(const Json& object,
                             std::initializer_list<std::string_view> names,
                             const std::string& where) {
    const double value = require_finite_number(object, names, where);
    if (std::floor(value) != value ||
        value < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
        value > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
        throw AppError(ErrorCode::MetaInvalid, where + " must be an integer frame value");
    }
    return static_cast<std::int64_t>(value);
}

std::uint32_t get_sample_rate(const Json& object, const std::string& where,
                              bool required) {
    const Json* rate = find_field(object, {"sample_rate", "sampleRate"});
    if (rate == nullptr) {
        if (required) {
            throw AppError(ErrorCode::MetaInvalid, where + " is missing sample_rate");
        }
        return kSampleRate;
    }
    if (!rate->is_number() || !std::isfinite(rate->number) ||
        std::floor(rate->number) != rate->number ||
        rate->number <= 0.0 || rate->number > 4294967295.0) {
        throw AppError(ErrorCode::MetaInvalid, where + " has an invalid sample_rate");
    }
    const auto value = static_cast<std::uint32_t>(rate->number);
    if (value != kSampleRate) {
        std::ostringstream message;
        message << where << " sample rate " << value << " Hz does not match 48000 Hz";
        throw AppError(ErrorCode::RateMismatch, message.str());
    }
    return value;
}

std::string generic_string(const fs::path& path) {
    return path.generic_string();
}

std::string sanitize_id(std::string_view id) {
    std::string out;
    for (const char c : id) {
        const unsigned char u = static_cast<unsigned char>(c);
        if ((u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') ||
            (u >= '0' && u <= '9') || c == '-' || c == '_') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
        if (out.size() >= 48) {
            break;
        }
    }
    if (out.empty()) {
        out = "track";
    }
    return out;
}

std::string padded_index(std::size_t index) {
    std::ostringstream out;
    out << std::setw(4) << std::setfill('0') << index;
    return out.str();
}

bool path_is_safe_relative(const fs::path& path) {
    if (path.empty() || path.is_absolute()) {
        return false;
    }
    for (const auto& part : path) {
        if (part == "..") {
            return false;
        }
    }
    return true;
}

void append_hash_file(Sha256& hash, const std::string& logical_name,
                      const fs::path& path) {
    std::error_code ec;
    const std::uint64_t size = fs::file_size(path, ec);
    if (ec) {
        throw AppError(ErrorCode::Io, "cannot stat input file: " + path.string());
    }
    hash.update(logical_name);
    hash.update("\0", 1);
    const std::uint64_t little = size;
    for (int i = 0; i < 8; ++i) {
        const char byte = static_cast<char>((little >> (8 * i)) & 0xFFu);
        hash.update(&byte, 1);
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw AppError(ErrorCode::Io, "cannot open input file: " + path.string());
    }
    std::array<char, 1024 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            hash.update(buffer.data(), static_cast<std::size_t>(count));
        }
    }
    if (!input.eof()) {
        throw AppError(ErrorCode::Io, "cannot read input file: " + path.string());
    }
}

std::pair<const MatchEvent*, double> find_anchor(
    const std::vector<MatchEvent>& table, const MarkerEvent& marker) {
    const MatchEvent* best = nullptr;
    double best_distance = std::numeric_limits<double>::infinity();
    for (const auto& event : table) {
        if (event.type != marker.type) {
            continue;
        }
        const double distance = std::abs(event.time_ms - marker.time_ms);
        if (distance < best_distance) {
            best = &event;
            best_distance = distance;
        }
    }
    if (best != nullptr) {
        return {best, best_distance};
    }
    for (const auto& event : table) {
        const double distance = std::abs(event.time_ms - marker.time_ms);
        if (distance < best_distance) {
            best = &event;
            best_distance = distance;
        }
    }
    if (best != nullptr) {
        return {best, best_distance};
    }
    return {nullptr, best_distance};
}

std::int64_t saturating_add(std::int64_t left, std::int64_t right) {
    if (right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) {
        return std::numeric_limits<std::int64_t>::max();
    }
    if (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right) {
        return std::numeric_limits<std::int64_t>::min();
    }
    return left + right;
}

std::int64_t saturating_sub(std::int64_t left, std::int64_t right) {
    if (right < 0 && left > std::numeric_limits<std::int64_t>::max() + right) {
        return std::numeric_limits<std::int64_t>::max();
    }
    if (right > 0 && left < std::numeric_limits<std::int64_t>::min() + right) {
        return std::numeric_limits<std::int64_t>::min();
    }
    return left - right;
}

std::uint64_t timeline_frames(std::int64_t start_frame,
                              std::uint64_t raw_frames,
                              std::int64_t end_frame,
                              std::int64_t offset,
                              std::uint64_t total_frames) {
    if (start_frame < 0 || end_frame <= start_frame) {
        return 0;
    }
    const std::uint64_t start = static_cast<std::uint64_t>(start_frame);
    if (start >= raw_frames) {
        return 0;
    }
    const std::uint64_t available_end =
        std::min<std::uint64_t>(raw_frames, static_cast<std::uint64_t>(end_frame));
    if (available_end <= start) {
        return 0;
    }
    const std::int64_t global_begin = saturating_add(start_frame, offset);
    const std::int64_t global_end = saturating_add(
        static_cast<std::int64_t>(available_end), offset);
    const std::int64_t clipped_begin = std::max<std::int64_t>(0, global_begin);
    const std::int64_t clipped_end =
        std::min<std::int64_t>(static_cast<std::int64_t>(total_frames), global_end);
    if (clipped_end <= clipped_begin) {
        return 0;
    }
    return static_cast<std::uint64_t>(clipped_end - clipped_begin);
}

Job validate_and_build_job(const fs::path& input_dir, const fs::path& output_dir) {
    std::error_code ec;
    if (!fs::exists(input_dir, ec) || !fs::is_directory(input_dir, ec)) {
        throw AppError(ErrorCode::Io, "input directory does not exist: " + input_dir.string());
    }
    if (fs::exists(output_dir, ec) && !fs::is_directory(output_dir, ec)) {
        throw AppError(ErrorCode::OutputUnwritable,
                       "output path is not a directory: " + output_dir.string());
    }

    Job job;
    job.input_dir = input_dir;
    job.output_dir = output_dir;
    job.events_path = input_dir / "events.json";
    if (!fs::exists(job.events_path, ec)) {
        throw AppError(ErrorCode::MetaInvalid,
                       "events.json is missing from the input directory");
    }

    const Json events_root = require_object(
        load_json(job.events_path, ErrorCode::MetaInvalid), "events.json");
    get_sample_rate(events_root, "events.json", true);

    job.duration_ms = require_finite_number(
        events_root,
        {"duration_ms", "match_duration_ms", "total_duration_ms",
         "durationMs", "matchDurationMs", "totalDurationMs"},
        "events.json");
    if (job.duration_ms <= 0.0 ||
        job.duration_ms > 1.0e12) {
        throw AppError(ErrorCode::MetaInvalid,
                       "events.json duration_ms must be positive and finite");
    }
    const double total_frames_d =
        job.duration_ms * static_cast<double>(kSampleRate) / 1000.0;
    if (!std::isfinite(total_frames_d) || total_frames_d > 9.0e15) {
        throw AppError(ErrorCode::MetaInvalid,
                       "events.json duration is too large");
    }
    job.total_frames = static_cast<std::uint64_t>(std::llround(total_frames_d));
    if (job.total_frames == 0) {
        throw AppError(ErrorCode::MetaInvalid,
                       "events.json duration is shorter than one sample");
    }
    const std::uint64_t max_stereo_frames =
        (0xFFFFFFFFull - 36ull) / (2ull * 2ull);
    if (job.total_frames > max_stereo_frames) {
        throw AppError(ErrorCode::MetaInvalid,
                       "events.json duration exceeds the RIFF WAV size limit");
    }
    job.total_blocks =
        (job.total_frames + kBlockFrames - 1u) / kBlockFrames;

    const Json* events_array = events_root.find("events");
    if (events_array == nullptr || !events_array->is_array()) {
        throw AppError(ErrorCode::MetaInvalid,
                       "events.json must contain an events array");
    }
    for (std::size_t i = 0; i < events_array->array.size(); ++i) {
        const std::string where =
            "events.json events[" + std::to_string(i) + "]";
        const Json& value = require_object(events_array->array[i], where);
        MatchEvent event;
        event.type = require_string(value, {"type", "event_type", "kind"}, where);
        event.time_ms = require_finite_number(
            value, {"match_time_ms", "time_ms", "matchTimeMs", "timestamp_ms",
                    "event_time_ms"}, where);
        if (event.time_ms < 0.0 || event.time_ms > job.duration_ms) {
            throw AppError(ErrorCode::MetaInvalid,
                           where + " time is outside the match duration");
        }
        event.sample = static_cast<std::int64_t>(
            std::llround(event.time_ms * static_cast<double>(kSampleRate) / 1000.0));
        job.match_events.push_back(std::move(event));
    }

    std::vector<fs::path> meta_paths;
    for (const auto& entry : fs::directory_iterator(input_dir, ec)) {
        if (ec) {
            throw AppError(ErrorCode::Io,
                           "cannot enumerate input directory: " + input_dir.string());
        }
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const fs::path path = entry.path();
        const std::string name = path.filename().string();
        if (path.extension() == ".json" &&
            name != "events.json" &&
            name != "manifest.json" &&
            name != kStateFile &&
            name != kStateTempFile &&
            name != kManifestTempFile) {
            meta_paths.push_back(path);
        }
    }
    std::sort(meta_paths.begin(), meta_paths.end(),
              [](const fs::path& a, const fs::path& b) {
                  return a.filename().generic_string() < b.filename().generic_string();
              });
    if (meta_paths.empty()) {
        throw AppError(ErrorCode::MetaInvalid,
                       "input directory contains no commentary metadata JSON files");
    }

    std::set<std::string> ids;
    for (const auto& meta_path : meta_paths) {
        const std::string where = meta_path.filename().string();
        Json root = load_json(meta_path, ErrorCode::MetaInvalid);
        require_object(root, where);
        get_sample_rate(root, where, false);

        TrackJob track;
        track.meta_path = meta_path;
        track.meta_name = meta_path.filename().generic_string();
        track.id = require_string(
            root, {"commentary_id", "id", "caster_id", "commentator_id",
                   "commentaryId"}, where);
        if (!ids.insert(track.id).second) {
            throw AppError(ErrorCode::MetaInvalid,
                           "duplicate commentary_id: " + track.id);
        }
        track.start_frame = require_integer(
            root, {"start_frame", "startFrame", "speech_start_frame", "speechStartFrame"},
            where);
        track.end_frame = require_integer(
            root, {"end_frame", "endFrame", "speech_end_frame", "speechEndFrame"}, where);
        if (track.start_frame < 0 || track.end_frame < 0) {
            throw AppError(ErrorCode::MetaInvalid,
                           where + " contains a negative frame offset");
        }
        if (track.end_frame < track.start_frame) {
            throw AppError(ErrorCode::MetaInvalid,
                           where + " end_frame must not be before start_frame");
        }
        track.gain_db = require_finite_number(
            root, {"gain_db", "gainDb", "gain"}, where);
        track.gain_linear = std::pow(10.0, track.gain_db / 20.0);
        if (!std::isfinite(track.gain_linear)) {
            throw AppError(ErrorCode::MetaInvalid,
                           where + " gain produces a non-finite linear scale");
        }

        const Json* raw_field = find_field(
            root, {"raw_file", "file", "audio_file", "rawFile", "audioFile"});
        std::string raw_name;
        if (raw_field != nullptr) {
            if (!raw_field->is_string() || raw_field->string.empty()) {
                throw AppError(ErrorCode::MetaInvalid,
                               where + " has an invalid raw_file field");
            }
            raw_name = raw_field->string;
        } else {
            raw_name = meta_path.stem().string() + ".raw";
        }
        const fs::path raw_path(raw_name);
        track.raw_path = raw_path.is_absolute() ? raw_path : input_dir / raw_path;
        track.raw_logical_name = generic_string(
            raw_path.is_absolute() ? raw_path : raw_path.lexically_normal());
        if (!fs::exists(track.raw_path, ec) || !fs::is_regular_file(track.raw_path, ec)) {
            throw AppError(ErrorCode::MetaInvalid,
                           where + " references a missing raw file: " + raw_name);
        }
        const std::uint64_t raw_bytes = fs::file_size(track.raw_path, ec);
        if (ec) {
            throw AppError(ErrorCode::Io,
                           "cannot stat raw file: " + track.raw_path.string());
        }
        if (raw_bytes % 2u != 0) {
            throw AppError(ErrorCode::MetaInvalid,
                           where + " raw file length is not an even 16-bit sample count");
        }
        track.raw_frames = raw_bytes / 2u;

        const Json* event_array = root.find("events");
        if (event_array == nullptr || !event_array->is_array() ||
            event_array->array.empty()) {
            throw AppError(ErrorCode::MetaInvalid,
                           where + " must contain a non-empty events array for alignment");
        }
        for (std::size_t i = 0; i < event_array->array.size(); ++i) {
            const std::string event_where =
                where + " events[" + std::to_string(i) + "]";
            const Json& event_json =
                require_object(event_array->array[i], event_where);
            MarkerEvent marker;
            marker.type = require_string(
                event_json, {"type", "event_type", "kind"}, event_where);
            marker.time_ms = require_finite_number(
                event_json,
                {"match_time_ms", "time_ms", "matchTimeMs", "timestamp_ms",
                 "event_time_ms"},
                event_where);
            if (marker.time_ms < 0.0 || marker.time_ms > job.duration_ms) {
                throw AppError(ErrorCode::MetaInvalid,
                               event_where + " time is outside the match duration");
            }
            const Json* local = find_field(
                event_json, {"local_frame", "localFrame", "frame", "frame_offset",
                             "frameOffset"});
            if (local == nullptr) {
                marker.local_frame = track.start_frame;
            } else {
                if (!local->is_number() || !std::isfinite(local->number) ||
                    std::floor(local->number) != local->number ||
                    local->number < static_cast<double>(
                        std::numeric_limits<std::int64_t>::min()) ||
                    local->number > static_cast<double>(
                        std::numeric_limits<std::int64_t>::max())) {
                    throw AppError(ErrorCode::MetaInvalid,
                                   event_where + " has an invalid local frame");
                }
                marker.local_frame = static_cast<std::int64_t>(local->number);
                if (marker.local_frame < 0) {
                    throw AppError(ErrorCode::MetaInvalid,
                                   event_where + " has a negative local frame");
                }
            }
            track.events.push_back(std::move(marker));
        }

        std::sort(track.events.begin(), track.events.end(),
                  [](const MarkerEvent& a, const MarkerEvent& b) {
                      if (a.time_ms != b.time_ms) {
                          return a.time_ms < b.time_ms;
                      }
                      if (a.type != b.type) {
                          return a.type < b.type;
                      }
                      return a.local_frame < b.local_frame;
                  });
        const MarkerEvent& anchor_marker = track.events.front();
        const auto anchor = find_anchor(job.match_events, anchor_marker);
        const double anchor_time =
            anchor.first != nullptr ? anchor.first->time_ms : anchor_marker.time_ms;
        track.anchor_type = anchor_marker.type;
        track.anchor_time_ms = anchor_time;
        const std::int64_t anchor_sample = static_cast<std::int64_t>(
            std::llround(anchor_time * static_cast<double>(kSampleRate) / 1000.0));
        track.alignment_offset = anchor_sample - anchor_marker.local_frame;
        track.requested_frames =
            static_cast<std::uint64_t>(track.end_frame - track.start_frame);
        track.truncated_to_raw =
            track.end_frame > static_cast<std::int64_t>(track.raw_frames) ||
            track.start_frame >= static_cast<std::int64_t>(track.raw_frames);
        track.used_frames = timeline_frames(
            track.start_frame, track.raw_frames, track.end_frame,
            track.alignment_offset, job.total_frames);
        job.tracks.push_back(std::move(track));
    }

    std::sort(job.tracks.begin(), job.tracks.end(),
              [](const TrackJob& a, const TrackJob& b) {
                  return a.id < b.id;
              });
    for (std::size_t i = 0; i < job.tracks.size(); ++i) {
        job.tracks[i].output_rel =
            "tracks/track_" + padded_index(i + 1) + "_" +
            sanitize_id(job.tracks[i].id) + ".wav";
    }

    std::map<std::string, fs::path> input_files;
    input_files["events.json"] = job.events_path;
    for (const auto& track : job.tracks) {
        input_files[track.meta_name] = track.meta_path;
        input_files[track.raw_logical_name] = track.raw_path;
    }
    Sha256 input_hash;
    for (const auto& item : input_files) {
        append_hash_file(input_hash, item.first, item.second);
    }
    job.input_sha256 = input_hash.digest_hex();

    std::ostringstream canonical;
    canonical << "voice-replay-job-v1\n"
              << "rate=" << kSampleRate << "\n"
              << "total_frames=" << job.total_frames << "\n"
              << "block_frames=" << kBlockFrames << "\n"
              << "input=" << job.input_sha256 << "\n";
    for (const auto& track : job.tracks) {
        canonical << "track=" << track.id
                  << "|start=" << track.start_frame
                  << "|end=" << track.end_frame
                  << "|offset=" << track.alignment_offset
                  << "|gain=" << json_number(track.gain_db)
                  << "|out=" << track.output_rel << "\n";
    }
    job.job_sha256 = sha256_bytes(canonical.str());
    return job;
}

void ensure_output_writable(const fs::path& output_dir) {
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        throw AppError(ErrorCode::OutputUnwritable,
                       "cannot create output directory: " + output_dir.string());
    }
    const fs::path probe = output_dir / ".voice-replay-write-probe";
    {
        std::ofstream out(probe, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw AppError(ErrorCode::OutputUnwritable,
                           "output directory is not writable: " + output_dir.string());
        }
        out << "probe";
        if (!out) {
            throw AppError(ErrorCode::OutputUnwritable,
                           "output directory is not writable: " + output_dir.string());
        }
    }
    fs::remove(probe, ec);
}

void fsync_path(const fs::path& path) {
#ifdef _WIN32
    HANDLE handle = CreateFileW(path.wstring().c_str(), GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle != INVALID_HANDLE_VALUE) {
        FlushFileBuffers(handle);
        CloseHandle(handle);
    }
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
#endif
}

void atomic_write(const fs::path& target, std::string_view text,
                  ErrorCode failure_code) {
    fs::path temp = target;
    temp += ".tmp";
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw AppError(failure_code,
                           "cannot create temporary file: " + temp.string());
        }
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        out.flush();
        if (!out) {
            throw AppError(failure_code,
                           "cannot write temporary file: " + temp.string());
        }
    }
    fsync_path(temp);
#ifdef _WIN32
    if (!MoveFileExW(temp.wstring().c_str(), target.wstring().c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw AppError(failure_code,
                       "cannot atomically replace file: " + target.string());
    }
#else
    std::error_code ec;
    fs::rename(temp, target, ec);
    if (ec) {
        throw AppError(failure_code,
                       "cannot atomically replace file: " + target.string());
    }
#endif
    fsync_path(target);
}

std::string state_json(const Job& job, std::uint64_t completed_blocks,
                       std::uint64_t master_clip,
                       const std::string& master_prefix,
                       const std::vector<TrackRuntime>& runtimes,
                       const std::vector<WavWriter>& track_writers) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"state_version\": 1,\n";
    out << "  \"input_sha256\": " << json_quote(job.input_sha256) << ",\n";
    out << "  \"job_sha256\": " << json_quote(job.job_sha256) << ",\n";
    out << "  \"sample_rate\": " << kSampleRate << ",\n";
    out << "  \"block_frames\": " << kBlockFrames << ",\n";
    out << "  \"total_frames\": " << job.total_frames << ",\n";
    out << "  \"total_blocks\": " << job.total_blocks << ",\n";
    out << "  \"completed_blocks\": " << completed_blocks << ",\n";
    out << "  \"master\": {\n";
    out << "    \"prefix_sha256\": " << json_quote(master_prefix) << ",\n";
    out << "    \"clipped_samples\": " << master_clip << "\n";
    out << "  },\n";
    out << "  \"tracks\": [";
    for (std::size_t i = 0; i < job.tracks.size(); ++i) {
        out << (i == 0 ? "\n" : ",\n");
        out << "    {\n";
        out << "      \"commentary_id\": " << json_quote(job.tracks[i].id) << ",\n";
        out << "      \"prefix_sha256\": "
            << json_quote(track_writers[i].prefix_sha256()) << ",\n";
        out << "      \"clipped_samples\": " << runtimes[i].clipped_samples << "\n";
        out << "    }";
    }
    if (!job.tracks.empty()) {
        out << "\n  ";
    }
    out << "]\n";
    out << "}\n";
    return out.str();
}

std::uint64_t require_state_u64(const Json& object, std::string_view key,
                                const std::string& where) {
    const Json* value = object.find(key);
    if (value == nullptr || !value->is_number() || !std::isfinite(value->number) ||
        std::floor(value->number) != value->number || value->number < 0.0 ||
        value->number > static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
        throw AppError(ErrorCode::StateCorrupt,
                       "resume state has invalid " + where + "." + std::string(key));
    }
    return static_cast<std::uint64_t>(value->number);
}

std::string require_state_string(const Json& object, std::string_view key,
                                 const std::string& where) {
    const Json* value = object.find(key);
    if (value == nullptr || !value->is_string() || value->string.size() != 64) {
        throw AppError(ErrorCode::StateCorrupt,
                       "resume state has invalid " + where + "." + std::string(key));
    }
    for (const char c : value->string) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            throw AppError(ErrorCode::StateCorrupt,
                           "resume state has a non-lowercase SHA-256 value");
        }
    }
    return value->string;
}

ResumeState load_state(const Job& job) {
    const fs::path path = job.output_dir / kStateFile;
    if (!fs::exists(path)) {
        throw AppError(ErrorCode::StateMissing,
                       "resume requested but state file is missing: " + path.string());
    }
    Json root;
    std::string error;
    try {
        root = load_json(path, ErrorCode::StateCorrupt);
    } catch (const AppError&) {
        throw;
    }
    if (!root.is_object()) {
        throw AppError(ErrorCode::StateCorrupt, "resume state root is not an object");
    }
    if (require_state_u64(root, "state_version", "root") != 1) {
        throw AppError(ErrorCode::StateCorrupt, "unsupported resume state version");
    }
    const std::string input_sha =
        require_state_string(root, "input_sha256", "root");
    const std::string job_sha =
        require_state_string(root, "job_sha256", "root");
    if (input_sha != job.input_sha256) {
        throw AppError(ErrorCode::StateCorrupt,
                       "resume state input digest does not match current inputs");
    }
    if (job_sha != job.job_sha256) {
        throw AppError(ErrorCode::StateCorrupt,
                       "resume state job digest does not match current export settings");
    }
    if (require_state_u64(root, "sample_rate", "root") != kSampleRate ||
        require_state_u64(root, "block_frames", "root") != kBlockFrames ||
        require_state_u64(root, "total_frames", "root") != job.total_frames ||
        require_state_u64(root, "total_blocks", "root") != job.total_blocks) {
        throw AppError(ErrorCode::StateCorrupt,
                       "resume state timeline does not match current inputs");
    }

    ResumeState state;
    state.completed_blocks = require_state_u64(root, "completed_blocks", "root");
    if (state.completed_blocks > job.total_blocks) {
        throw AppError(ErrorCode::StateCorrupt,
                       "resume state completed_blocks exceeds total_blocks");
    }
    state.input_sha256 = input_sha;
    state.job_sha256 = job_sha;

    const Json* master = root.find("master");
    if (master == nullptr || !master->is_object()) {
        throw AppError(ErrorCode::StateCorrupt, "resume state master is missing");
    }
    state.master_prefix_sha256 =
        require_state_string(*master, "prefix_sha256", "master");
    state.master_clipped_samples =
        require_state_u64(*master, "clipped_samples", "master");
    const Json* tracks = root.find("tracks");
    if (tracks == nullptr || !tracks->is_array() ||
        tracks->array.size() != job.tracks.size()) {
        throw AppError(ErrorCode::StateCorrupt,
                       "resume state track list does not match current inputs");
    }
    for (std::size_t i = 0; i < job.tracks.size(); ++i) {
        const Json& item = tracks->array[i];
        if (!item.is_object()) {
            throw AppError(ErrorCode::StateCorrupt,
                           "resume state track entry is not an object");
        }
        const Json* id = item.find("commentary_id");
        if (id == nullptr || !id->is_string() || id->string != job.tracks[i].id) {
            throw AppError(ErrorCode::StateCorrupt,
                           "resume state track order or commentary_id differs");
        }
        state.track_prefix_sha256.push_back(
            require_state_string(item, "prefix_sha256", "track"));
        state.track_clipped_samples.push_back(
            require_state_u64(item, "clipped_samples", "track"));
    }
    return state;
}

std::string manifest_json(const Job& job,
                          const std::string& master_sha,
                          std::uint64_t master_clip,
                          const std::vector<std::string>& track_sha,
                          const std::vector<TrackRuntime>& runtimes) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"format\": " << json_quote(kManifestFormat) << ",\n";
    out << "  \"sample_rate\": " << kSampleRate << ",\n";
    out << "  \"bits_per_sample\": " << kBitsPerSample << ",\n";
    out << "  \"total_frames\": " << job.total_frames << ",\n";
    out << "  \"duration_ms\": " << json_number(job.duration_ms) << ",\n";
    out << "  \"input_sha256\": " << json_quote(job.input_sha256) << ",\n";
    out << "  \"master\": {\n";
    out << "    \"file\": \"master.wav\",\n";
    out << "    \"channels\": 2,\n";
    out << "    \"frames\": " << job.total_frames << ",\n";
    out << "    \"clipped_samples\": " << master_clip << ",\n";
    out << "    \"sha256\": " << json_quote(master_sha) << "\n";
    out << "  },\n";
    out << "  \"tracks\": [";
    for (std::size_t i = 0; i < job.tracks.size(); ++i) {
        const auto& track = job.tracks[i];
        out << (i == 0 ? "\n" : ",\n");
        out << "    {\n";
        out << "      \"commentary_id\": " << json_quote(track.id) << ",\n";
        out << "      \"file\": " << json_quote(track.output_rel) << ",\n";
        out << "      \"channels\": 1,\n";
        out << "      \"frames\": " << job.total_frames << ",\n";
        out << "      \"raw_frames\": " << track.raw_frames << ",\n";
        out << "      \"requested_frames\": " << track.requested_frames << ",\n";
        out << "      \"used_frames\": " << track.used_frames << ",\n";
        out << "      \"truncated_to_raw\": "
            << (track.truncated_to_raw ? "true" : "false") << ",\n";
        out << "      \"alignment_offset_frames\": "
            << track.alignment_offset << ",\n";
        out << "      \"gain_db\": " << json_number(track.gain_db) << ",\n";
        out << "      \"clipped_samples\": "
            << runtimes[i].clipped_samples << ",\n";
        out << "      \"anchor_event_type\": " << json_quote(track.anchor_type) << ",\n";
        out << "      \"anchor_event_time_ms\": "
            << json_number(track.anchor_time_ms) << ",\n";
        out << "      \"sha256\": " << json_quote(track_sha[i]) << "\n";
        out << "    }";
    }
    if (!job.tracks.empty()) {
        out << "\n  ";
    }
    out << "]\n";
    out << "}\n";
    return out.str();
}

void print_digest_line(const std::string& relative, std::uint64_t frames,
                       const std::string& sha) {
    std::cerr << "voice-replay: SHA256 " << relative
              << " frames=" << frames << " sha256=" << sha << "\n";
}

void print_manifest_digest(const fs::path& manifest_path) {
    std::cerr << "voice-replay: MANIFEST_SHA256 " << manifest_path.string()
              << " sha256=" << sha256_file(manifest_path) << "\n";
}

bool manifest_matches(const Job& job) {
    const fs::path manifest_path = job.output_dir / kManifestFile;
    std::error_code ec;
    if (!fs::exists(manifest_path, ec)) {
        return false;
    }
    Json root;
    std::string error;
    try {
        root = load_json(manifest_path, ErrorCode::MetaInvalid);
    } catch (const AppError&) {
        return false;
    }
    if (!root.is_object()) {
        return false;
    }
    const Json* format = root.find("format");
    const Json* input_sha = root.find("input_sha256");
    const Json* total_frames = root.find("total_frames");
    if (format == nullptr || !format->is_string() ||
        format->string != kManifestFormat ||
        input_sha == nullptr || !input_sha->is_string() ||
        input_sha->string != job.input_sha256 ||
        total_frames == nullptr || !total_frames->is_number() ||
        total_frames->number != static_cast<double>(job.total_frames)) {
        return false;
    }
    const Json* master = root.find("master");
    if (master == nullptr || !master->is_object()) {
        return false;
    }
    const Json* master_file = master->find("file");
    const Json* master_frames = master->find("frames");
    const Json* master_sha = master->find("sha256");
    if (master_file == nullptr || !master_file->is_string() ||
        master_file->string != "master.wav" ||
        master_frames == nullptr || !master_frames->is_number() ||
        master_frames->number != static_cast<double>(job.total_frames) ||
        master_sha == nullptr || !master_sha->is_string()) {
        return false;
    }
    const fs::path master_path = job.output_dir / "master.wav";
    if (!fs::exists(master_path, ec) ||
        fs::file_size(master_path, ec) !=
            wav_size_for_frames(job.total_frames, 2) ||
        sha256_file(master_path) != master_sha->string) {
        return false;
    }

    const Json* tracks = root.find("tracks");
    if (tracks == nullptr || !tracks->is_array() ||
        tracks->array.size() != job.tracks.size()) {
        return false;
    }
    for (std::size_t i = 0; i < job.tracks.size(); ++i) {
        const Json& item = tracks->array[i];
        if (!item.is_object()) {
            return false;
        }
        const Json* id = item.find("commentary_id");
        const Json* file = item.find("file");
        const Json* frames = item.find("frames");
        const Json* sha = item.find("sha256");
        if (id == nullptr || !id->is_string() || id->string != job.tracks[i].id ||
            file == nullptr || !file->is_string() ||
            file->string != job.tracks[i].output_rel ||
            frames == nullptr || !frames->is_number() ||
            frames->number != static_cast<double>(job.total_frames) ||
            sha == nullptr || !sha->is_string()) {
            return false;
        }
        const fs::path path = job.output_dir / fs::path(file->string);
        if (!path_is_safe_relative(fs::path(file->string)) ||
            !fs::exists(path, ec) ||
            fs::file_size(path, ec) !=
                wav_size_for_frames(job.total_frames, 1) ||
            sha256_file(path) != sha->string) {
            return false;
        }
    }

    std::cerr << "voice-replay: REUSE existing byte-identical outputs from "
              << manifest_path.string() << "\n";
    print_digest_line("master.wav", job.total_frames, master_sha->string);
    for (const auto& item : tracks->array) {
        const Json* file = item.find("file");
        const Json* sha = item.find("sha256");
        print_digest_line(file->string, job.total_frames, sha->string);
    }
    print_manifest_digest(manifest_path);
    return true;
}

std::int16_t quantize_sample(double normalized, bool& clipped) {
    const double scaled = normalized * 32768.0;
    if (!std::isfinite(scaled) || scaled >= 32767.5) {
        clipped = true;
        return normalized < 0.0 ? std::numeric_limits<std::int16_t>::min()
                                : std::numeric_limits<std::int16_t>::max();
    }
    if (scaled <= -32768.5) {
        clipped = true;
        return std::numeric_limits<std::int16_t>::min();
    }
    const long long rounded = std::llround(scaled);
    if (rounded > std::numeric_limits<std::int16_t>::max()) {
        clipped = true;
        return std::numeric_limits<std::int16_t>::max();
    }
    if (rounded < std::numeric_limits<std::int16_t>::min()) {
        clipped = true;
        return std::numeric_limits<std::int16_t>::min();
    }
    return static_cast<std::int16_t>(rounded);
}

void append_i16_le(std::vector<char>& out, std::int16_t value) {
    const std::uint16_t u = static_cast<std::uint16_t>(value);
    out.push_back(static_cast<char>(u & 0xFFu));
    out.push_back(static_cast<char>((u >> 8) & 0xFFu));
}

void append_master_samples(std::vector<char>& out,
                           const std::vector<double>& mix,
                           std::uint64_t frames,
                           std::uint64_t& master_clip) {
    out.clear();
    out.reserve(static_cast<std::size_t>(frames * 4u));
    for (std::uint64_t i = 0; i < frames; ++i) {
        bool clipped_left = false;
        bool clipped_right = false;
        const std::int16_t left = quantize_sample(mix[i * 2u], clipped_left);
        const std::int16_t right = quantize_sample(mix[i * 2u + 1u], clipped_right);
        if (clipped_left || clipped_right) {
            ++master_clip;
        }
        append_i16_le(out, left);
        append_i16_le(out, right);
    }
}

void append_stem_samples(std::vector<char>& out,
                         const std::vector<double>& track_mix,
                         std::uint64_t frames,
                         TrackRuntime& runtime) {
    out.clear();
    out.reserve(static_cast<std::size_t>(frames * 2u));
    for (std::uint64_t i = 0; i < frames; ++i) {
        bool clipped = false;
        const std::int16_t value = quantize_sample(track_mix[i], clipped);
        if (clipped) {
            ++runtime.clipped_samples;
        }
        append_i16_le(out, value);
    }
}

int parm_abort_after_blocks() {
    const char* text = std::getenv("VOICE_REPLAY_ABORT_AFTER_BLOCKS");
    if (text == nullptr || *text == '\0') {
        return -1;
    }
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == nullptr || *end != '\0' || value < 0 || value > 1000000000L) {
        return -1;
    }
    return static_cast<int>(value);
}

void write_state(const Job& job, std::uint64_t completed_blocks,
                 std::uint64_t master_clip, const std::string& master_prefix,
                 const std::vector<TrackRuntime>& runtimes,
                 const std::vector<WavWriter>& track_writers,
                 ErrorCode failure_code) {
    atomic_write(job.output_dir / kStateFile,
                 state_json(job, completed_blocks, master_clip, master_prefix,
                            runtimes, track_writers),
                 failure_code);
}

void clear_state_files(const fs::path& output_dir) {
    std::error_code ec;
    fs::remove(output_dir / kStateFile, ec);
    fs::remove(output_dir / kStateTempFile, ec);
}

void export_blocks(const Job& job,
                   std::vector<std::ifstream>& raws,
                   WavWriter& master,
                   std::vector<WavWriter>& track_writers,
                   std::vector<TrackRuntime>& runtimes,
                   std::uint64_t& master_clip,
                   std::uint64_t start_block,
                   int abort_after_blocks);

void export_job(Job& job, bool resume, int abort_after_blocks) {
    if (resume && fs::exists(job.output_dir / kStateFile)) {
        load_state(job);
    }
    if (manifest_matches(job)) {
        clear_state_files(job.output_dir);
        return;
    }
    if (resume) {
        const ResumeState state = load_state(job);
        const std::uint64_t prefix_frames =
            std::min(job.total_frames, state.completed_blocks * kBlockFrames);

        WavWriter master;
        master.resume(job.output_dir / "master.wav", job.total_frames, 2,
                      prefix_frames, state.master_prefix_sha256);
        std::vector<WavWriter> track_writers(job.tracks.size());
        for (std::size_t i = 0; i < job.tracks.size(); ++i) {
            track_writers[i].resume(
                job.output_dir / job.tracks[i].output_rel, job.total_frames, 1,
                prefix_frames, state.track_prefix_sha256[i]);
        }

        std::uint64_t master_clip = state.master_clipped_samples;
        std::vector<TrackRuntime> runtimes(job.tracks.size());
        for (std::size_t i = 0; i < runtimes.size(); ++i) {
            runtimes[i].clipped_samples = state.track_clipped_samples[i];
        }
        std::vector<std::ifstream> raws(job.tracks.size());
        for (std::size_t i = 0; i < job.tracks.size(); ++i) {
            raws[i].open(job.tracks[i].raw_path, std::ios::binary);
            if (!raws[i]) {
                throw AppError(ErrorCode::Io,
                               "cannot open raw file: " +
                                   job.tracks[i].raw_path.string());
            }
        }

        export_blocks(job, raws, master, track_writers, runtimes, master_clip,
                      state.completed_blocks, abort_after_blocks);
    } else {
        clear_state_files(job.output_dir);
        ensure_output_writable(job.output_dir);
        WavWriter master;
        master.create(job.output_dir / "master.wav", job.total_frames, 2);
        std::vector<WavWriter> track_writers(job.tracks.size());
        for (std::size_t i = 0; i < job.tracks.size(); ++i) {
            track_writers[i].create(job.output_dir / job.tracks[i].output_rel,
                                    job.total_frames, 1);
        }
        std::vector<TrackRuntime> runtimes(job.tracks.size());
        std::uint64_t master_clip = 0;
        write_state(job, 0, master_clip, master.prefix_sha256(),
                    runtimes, track_writers, ErrorCode::OutputUnwritable);

        std::vector<std::ifstream> raws(job.tracks.size());
        for (std::size_t i = 0; i < job.tracks.size(); ++i) {
            raws[i].open(job.tracks[i].raw_path, std::ios::binary);
            if (!raws[i]) {
                throw AppError(ErrorCode::Io,
                               "cannot open raw file: " +
                                   job.tracks[i].raw_path.string());
            }
        }

        export_blocks(job, raws, master, track_writers, runtimes, master_clip,
                      0, abort_after_blocks);
    }
}

void export_blocks(const Job& job,
                   std::vector<std::ifstream>& raws,
                   WavWriter& master,
                   std::vector<WavWriter>& track_writers,
                   std::vector<TrackRuntime>& runtimes,
                   std::uint64_t& master_clip,
                   std::uint64_t start_block,
                   int abort_after_blocks) {
    const int env_abort = parm_abort_after_blocks();
    const int effective_abort =
        abort_after_blocks >= 0 ? abort_after_blocks : env_abort;
    if (effective_abort == 0 && start_block == 0) {
        std::cerr << "voice-replay: TEST_ABORT at block boundary after 0 blocks\n";
        std::_Exit(75);
    }
    std::vector<double> master_mix(static_cast<std::size_t>(kBlockFrames * 2u));
    std::vector<char> master_bytes;
    std::vector<double> track_mix(static_cast<std::size_t>(kBlockFrames));
    std::vector<char> stem_bytes;
    std::vector<std::uint8_t> raw_bytes;

    for (std::uint64_t block = start_block; block < job.total_blocks; ++block) {
        const std::uint64_t frame_begin = block * kBlockFrames;
        const std::uint64_t frames =
            std::min(kBlockFrames, job.total_frames - frame_begin);
        const std::uint64_t frame_end = frame_begin + frames;
        std::fill(master_mix.begin(),
                  master_mix.begin() + static_cast<std::ptrdiff_t>(frames * 2u), 0.0);

        for (std::size_t t = 0; t < job.tracks.size(); ++t) {
            const TrackJob& track = job.tracks[t];
            std::fill(track_mix.begin(),
                      track_mix.begin() + static_cast<std::ptrdiff_t>(frames), 0.0);

            const std::int64_t speech_begin = track.start_frame;
            const std::uint64_t available_end =
                std::min<std::uint64_t>(
                    track.raw_frames,
                    static_cast<std::uint64_t>(track.end_frame));
            const std::int64_t global_begin =
                saturating_add(track.start_frame, track.alignment_offset);
            const std::int64_t global_end = saturating_add(
                static_cast<std::int64_t>(available_end),
                track.alignment_offset);
            const std::int64_t intersection_begin =
                std::max<std::int64_t>(
                    static_cast<std::int64_t>(frame_begin),
                    std::max<std::int64_t>(0, global_begin));
            const std::int64_t intersection_end =
                std::min<std::int64_t>(
                    static_cast<std::int64_t>(frame_end),
                    std::min<std::int64_t>(
                        static_cast<std::int64_t>(job.total_frames), global_end));

            if (intersection_end > intersection_begin) {
                const std::int64_t source_begin = saturating_add(
                    speech_begin,
                    saturating_sub(intersection_begin, global_begin));
                const std::uint64_t count = static_cast<std::uint64_t>(
                    intersection_end - intersection_begin);
                if (source_begin < 0 ||
                    static_cast<std::uint64_t>(source_begin) + count >
                        track.raw_frames) {
                    throw AppError(ErrorCode::Internal,
                                   "computed raw source range is outside the file");
                }
                const std::uint64_t raw_byte_offset =
                    static_cast<std::uint64_t>(source_begin) * 2u;
                if (raw_byte_offset > static_cast<std::uint64_t>(
                        std::numeric_limits<std::streamoff>::max()) ||
                    count > static_cast<std::uint64_t>(
                        std::numeric_limits<std::streamsize>::max()) / 2u) {
                    throw AppError(ErrorCode::Internal, "raw read is too large");
                }
                raw_bytes.resize(static_cast<std::size_t>(count * 2u));
                raws[t].clear();
                raws[t].seekg(static_cast<std::streamoff>(raw_byte_offset));
                raws[t].read(reinterpret_cast<char*>(raw_bytes.data()),
                             static_cast<std::streamsize>(count * 2u));
                if (raws[t].gcount() != static_cast<std::streamsize>(count * 2u)) {
                    throw AppError(ErrorCode::Io,
                                   "raw file ended before its validated frame count: " +
                                       track.raw_path.string());
                }
                const std::uint64_t output_offset = static_cast<std::uint64_t>(
                    intersection_begin - static_cast<std::int64_t>(frame_begin));
                for (std::uint64_t i = 0; i < count; ++i) {
                    const std::uint16_t u =
                        static_cast<std::uint16_t>(raw_bytes[i * 2u]) |
                        static_cast<std::uint16_t>(
                            static_cast<std::uint16_t>(raw_bytes[i * 2u + 1u]) << 8);
                    const double source =
                        static_cast<double>(static_cast<std::int16_t>(u)) / 32768.0;
                    const double contribution = source * track.gain_linear;
                    const std::uint64_t out_index = output_offset + i;
                    track_mix[static_cast<std::size_t>(out_index)] += contribution;
                    master_mix[static_cast<std::size_t>(out_index * 2u)] += contribution;
                    master_mix[static_cast<std::size_t>(out_index * 2u + 1u)] +=
                        contribution;
                }
            }

            append_stem_samples(stem_bytes, track_mix, frames, runtimes[t]);
            track_writers[t].write_frames(
                reinterpret_cast<const std::int16_t*>(stem_bytes.data()), frames);
        }

        append_master_samples(master_bytes, master_mix, frames, master_clip);
        master.write_frames(
            reinterpret_cast<const std::int16_t*>(master_bytes.data()), frames);

        master.flush();
        for (auto& writer : track_writers) {
            writer.flush();
        }
        write_state(job, block + 1, master_clip, master.prefix_sha256(),
                    runtimes, track_writers, ErrorCode::Io);

        if (effective_abort >= 0 &&
            block + 1 >= static_cast<std::uint64_t>(effective_abort)) {
            std::cerr << "voice-replay: TEST_ABORT at block boundary after "
                      << (block + 1) << " blocks\n";
            std::_Exit(75);
        }
    }

    master.close();
    for (auto& writer : track_writers) {
        writer.close();
    }

    std::vector<std::string> track_sha(job.tracks.size());
    for (std::size_t i = 0; i < job.tracks.size(); ++i) {
        track_sha[i] = sha256_file(job.output_dir / job.tracks[i].output_rel);
    }
    const std::string master_sha = sha256_file(job.output_dir / "master.wav");
    const std::string manifest = manifest_json(
        job, master_sha, master_clip, track_sha, runtimes);
    const fs::path manifest_path = job.output_dir / kManifestFile;
    std::error_code ec;
    bool same_manifest = false;
    if (fs::exists(manifest_path, ec)) {
        try {
            same_manifest = read_file(manifest_path) == manifest;
        } catch (const AppError&) {
            same_manifest = false;
        }
    }
    if (!same_manifest) {
        atomic_write(manifest_path, manifest, ErrorCode::OutputUnwritable);
    }
    clear_state_files(job.output_dir);

    print_digest_line("master.wav", job.total_frames, master_sha);
    for (std::size_t i = 0; i < job.tracks.size(); ++i) {
        print_digest_line(job.tracks[i].output_rel, job.total_frames, track_sha[i]);
    }
    print_manifest_digest(manifest_path);
}

}  // namespace

void run_export(const fs::path& input_dir, const fs::path& output_dir,
                bool resume, int abort_after_blocks) {
    Job job = validate_and_build_job(input_dir, output_dir);
    export_job(job, resume, abort_after_blocks);
}

}  // namespace vr

