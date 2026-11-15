#include "automix/engine.h"

#include "automix/error.h"
#include "automix/json.h"
#include "automix/resampler.h"
#include "automix/wav_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

namespace automix {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

StereoBuffer toStereoTrack(const AudioData& audio) {
    StereoBuffer track(48000, static_cast<std::size_t>(audio.frames));
    if (audio.channels == 1) {
        constexpr double kEqualPower = 0.70710678118654752440;
        for (std::size_t frame = 0; frame < track.frames; ++frame) {
            const double value =
                static_cast<double>(audio.samples[frame]) * kEqualPower;
            track.left(frame) = static_cast<float>(value);
            track.right(frame) = static_cast<float>(value);
        }
    } else {
        for (std::size_t frame = 0; frame < track.frames; ++frame) {
            track.left(frame) = audio.samples[frame * 2];
            track.right(frame) = audio.samples[frame * 2 + 1];
        }
    }
    return track;
}

void applyGain(StereoBuffer& buffer, double gainDb) {
    const double gain = dbToLinear(gainDb);
    for (float& sample : buffer.interleaved) {
        sample = static_cast<float>(static_cast<double>(sample) * gain);
    }
}

void addInto(StereoBuffer& destination, const StereoBuffer& source) {
    if (source.frames == 0) {
        return;
    }
    if (destination.frames < source.frames) {
        const std::size_t oldFrames = destination.frames;
        destination.frames = source.frames;
        destination.interleaved.resize(source.frames * 2, 0.0f);
        (void)oldFrames;
    }
    for (std::size_t frame = 0; frame < source.frames; ++frame) {
        destination.left(frame) =
            static_cast<float>(destination.left(frame) + source.left(frame));
        destination.right(frame) =
            static_cast<float>(destination.right(frame) + source.right(frame));
    }
}

std::string jsonNumber(double value, int precision = 10) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(precision) << value;
    return out.str();
}

std::string jsonBool(bool value) {
    return value ? "true" : "false";
}

void appendDrcObject(std::ostringstream& out, const DrcConfig& config,
                     const DrcStats& stats, const std::string& indent) {
    out << indent << "\"enabled\": " << jsonBool(config.enabled) << ",\n";
    out << indent << "\"threshold_db\": " << jsonNumber(config.thresholdDb) << ",\n";
    out << indent << "\"ratio\": " << jsonNumber(config.ratio) << ",\n";
    out << indent << "\"attack_ms\": " << jsonNumber(config.attackMs) << ",\n";
    out << indent << "\"release_ms\": " << jsonNumber(config.releaseMs) << ",\n";
    out << indent << "\"knee_db\": " << jsonNumber(config.kneeDb) << ",\n";
    out << indent << "\"max_reduction_db\": " << jsonNumber(stats.maxReductionDb) << ",\n";
    out << indent << "\"mean_reduction_db\": " << jsonNumber(stats.meanReductionDb) << ",\n";
    out << indent << "\"active_samples\": " << stats.activeSamples;
}

}  // namespace

RenderResult renderMix(const AppConfig& config, Logger& logger) {
    const auto renderStart = Clock::now();
    RenderResult result;
    result.masterDrc.enabled = false;
    const bool anySolo =
        std::any_of(config.tracks.begin(), config.tracks.end(),
                    [](const TrackConfig& track) { return track.solo; });

    for (const TrackConfig& trackConfig : config.tracks) {
        const bool audible = !trackConfig.mute && (!anySolo || trackConfig.solo);

        const auto readStart = Clock::now();
        AudioData source = readAudioFile(trackConfig.input);
        const auto readEnd = Clock::now();

        const auto resampleStart = Clock::now();
        BandlimitedResampler resampler;
        AudioData converted = resampler.convert(source, 48000);
        const auto resampleEnd = Clock::now();

        const auto gainStart = Clock::now();
        StereoBuffer trackBuffer = toStereoTrack(converted);
        if (audible && std::abs(trackConfig.gainDb) > 1e-12) {
            applyGain(trackBuffer, trackConfig.gainDb);
        }
        if (!audible) {
            std::fill(trackBuffer.interleaved.begin(), trackBuffer.interleaved.end(), 0.0f);
        }
        const auto gainEnd = Clock::now();

        const auto drcStart = Clock::now();
        DrcStats drcStats;
        const bool drcEnabled =
            audible && trackConfig.drc.has_value() && trackConfig.drc->enabled;
        if (drcEnabled) {
            drcStats = processDrc(trackBuffer, *trackConfig.drc);
        }
        const auto drcEnd = Clock::now();
        const auto mixStart = Clock::now();
        addInto(result.audio, trackBuffer);
        result.mixMs += elapsedMs(mixStart, Clock::now());

        TrackReport report;
        report.name = trackConfig.name;
        report.sourcePath = pathToUtf8(trackConfig.input);
        report.sourceFormat = source.formatName;
        report.sourceSampleRate = source.sampleRate;
        report.sourceChannels = source.channels;
        report.sourceFrames = source.frames;
        report.outputFrames = converted.frames;
        report.configuredGainDb = trackConfig.gainDb;
        report.effectiveGainDb = audible ? trackConfig.gainDb : -120.0;
        report.muted = trackConfig.mute;
        report.soloSuppressed = anySolo && !trackConfig.solo;
        report.drcEnabled = drcEnabled;
        report.drc.enabled = false;
        if (trackConfig.drc.has_value()) {
            report.drc = *trackConfig.drc;
        }
        report.drcStats = drcStats;
        report.postDrcLevels = measureLevels(trackBuffer);
        report.readMs = elapsedMs(readStart, readEnd);
        report.resampleMs = elapsedMs(resampleStart, resampleEnd);
        report.gainPanMs = elapsedMs(gainStart, gainEnd);
        report.drcMs = elapsedMs(drcStart, drcEnd);
        result.tracks.push_back(report);

        std::ostringstream summary;
        summary << "track " << trackConfig.name << " source=" << report.sourcePath
                << " format=" << report.sourceFormat
                << " in_rate=" << report.sourceSampleRate
                << " channels=" << report.sourceChannels
                << " frames=" << report.sourceFrames
                << " audible=" << (audible ? "yes" : "no")
                << " gain_db=" << trackConfig.gainDb
                << " drc_max_db=" << report.drcStats.maxReductionDb
                << " read_ms=" << report.readMs
                << " resample_ms=" << report.resampleMs
                << " drc_ms=" << report.drcMs;
        logger.info(summary.str());
    }

    result.mixLevels = measureLevels(result.audio);

    if (std::abs(config.masterGainDb) > 1e-12) {
        applyGain(result.audio, config.masterGainDb);
    }

    const auto masterDrcStart = Clock::now();
    if (config.masterDrc.has_value() && config.masterDrc->enabled) {
        result.masterDrcEnabled = true;
        result.masterDrc = *config.masterDrc;
        result.masterDrcStats = processDrc(result.audio, *config.masterDrc);
    } else if (config.masterDrc.has_value()) {
        result.masterDrc = *config.masterDrc;
    }
    const auto masterDrcEnd = Clock::now();
    result.masterDrcMs = elapsedMs(masterDrcStart, masterDrcEnd);
    result.preLimiterLevels = measureLevels(result.audio);

    const auto limiterStart = Clock::now();
    result.limiterStats = processLimiter(result.audio, config.limiter);
    const auto limiterEnd = Clock::now();
    result.limiterMs = elapsedMs(limiterStart, limiterEnd);

    const auto loudnessStart = Clock::now();
    result.loudness = measureBs1770(result.audio);
    const auto loudnessEnd = Clock::now();
    result.loudnessMs = elapsedMs(loudnessStart, loudnessEnd);

    result.totalMs = elapsedMs(renderStart, Clock::now());

    std::ostringstream completion;
    completion << "mix complete frames=" << result.audio.frames
               << " pre_limiter_peak_dbfs=" << result.preLimiterLevels.peakDbfs
               << " limiter_max_attenuation_db="
               << result.limiterStats.maxAttenuationDb
               << " integrated_lufs="
               << (result.loudness.hasIntegratedValue ? result.loudness.integratedLufs
                                                      : -120.0)
               << " true_peak_dbtp=" << result.loudness.truePeakDbtp
               << " total_ms=" << result.totalMs;
    logger.info(completion.str());
    return result;
}

std::string buildReportJson(const AppConfig& config,
                            const RenderResult& result,
                            double writeMs) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(10);
    out << "{\n";
    out << "  \"tool\": {\"name\": \"automix\", \"version\": \"" AUTOMIX_VERSION "\"},\n";
    out << "  \"config\": " << jsonQuote(pathToUtf8(config.configPath)) << ",\n";
    out << "  \"output\": " << jsonQuote(pathToUtf8(config.output)) << ",\n";
    out << "  \"sample_rate\": 48000,\n";
    out << "  \"channels\": 2,\n";
    out << "  \"format\": \"WAV/FLOAT_32\",\n";
    out << "  \"duration_seconds\": "
        << jsonNumber(static_cast<double>(result.audio.frames) / 48000.0) << ",\n";
    out << "  \"master_gain_db\": " << jsonNumber(config.masterGainDb) << ",\n";
    out << "  \"tracks\": [\n";
    for (std::size_t index = 0; index < result.tracks.size(); ++index) {
        const TrackReport& track = result.tracks[index];
        out << "    {\n";
        out << "      \"name\": " << jsonQuote(track.name) << ",\n";
        out << "      \"source\": " << jsonQuote(track.sourcePath) << ",\n";
        out << "      \"source_format\": " << jsonQuote(track.sourceFormat) << ",\n";
        out << "      \"source_sample_rate\": " << track.sourceSampleRate << ",\n";
        out << "      \"source_channels\": " << track.sourceChannels << ",\n";
        out << "      \"source_frames\": " << track.sourceFrames << ",\n";
        out << "      \"output_frames\": " << track.outputFrames << ",\n";
        out << "      \"configured_gain_db\": " << jsonNumber(track.configuredGainDb)
            << ",\n";
        out << "      \"effective_gain_db\": " << jsonNumber(track.effectiveGainDb)
            << ",\n";
        out << "      \"muted\": " << jsonBool(track.muted) << ",\n";
        out << "      \"solo_suppressed\": " << jsonBool(track.soloSuppressed) << ",\n";
        out << "      \"post_processing_peak_dbfs\": "
            << jsonNumber(track.postDrcLevels.peakDbfs) << ",\n";
        out << "      \"post_processing_rms_dbfs\": "
            << jsonNumber(track.postDrcLevels.rmsDbfs) << ",\n";
        out << "      \"drc\": {\n";
        appendDrcObject(out, track.drc, track.drcStats, "        ");
        out << "\n      },\n";
        out << "      \"timing_ms\": {\"read\": " << jsonNumber(track.readMs)
            << ", \"resample\": " << jsonNumber(track.resampleMs)
            << ", \"gain_pan\": " << jsonNumber(track.gainPanMs)
            << ", \"drc\": " << jsonNumber(track.drcMs) << "}\n";
        out << "    }" << (index + 1 == result.tracks.size() ? "" : ",") << "\n";
    }
    out << "  ],\n";
    out << "  \"mix\": {\n";
    out << "    \"pre_master_peak_dbfs\": " << jsonNumber(result.mixLevels.peakDbfs)
        << ",\n";
    out << "    \"pre_master_rms_dbfs\": " << jsonNumber(result.mixLevels.rmsDbfs)
        << ",\n";
    out << "    \"pre_limiter_peak_dbfs\": "
        << jsonNumber(result.preLimiterLevels.peakDbfs) << ",\n";
    out << "    \"pre_limiter_rms_dbfs\": "
        << jsonNumber(result.preLimiterLevels.rmsDbfs) << "\n";
    out << "  },\n";
    out << "  \"master_drc\": {\n";
    appendDrcObject(out, result.masterDrc, result.masterDrcStats, "    ");
    out << "\n  },\n";
    out << "  \"limiter\": {\n";
    out << "    \"enabled\": " << jsonBool(config.limiter.enabled) << ",\n";
    out << "    \"threshold_dbfs\": " << jsonNumber(config.limiter.thresholdDbfs)
        << ",\n";
    out << "    \"lookahead_ms\": " << jsonNumber(config.limiter.lookaheadMs) << ",\n";
    out << "    \"lookahead_samples\": " << result.limiterStats.lookaheadSamples
        << ",\n";
    out << "    \"release_ms\": " << jsonNumber(config.limiter.releaseMs) << ",\n";
    out << "    \"triggered_samples\": " << result.limiterStats.triggeredSamples
        << ",\n";
    out << "    \"affected_samples\": " << result.limiterStats.affectedSamples
        << ",\n";
    out << "    \"max_attenuation_db\": "
        << jsonNumber(result.limiterStats.maxAttenuationDb) << ",\n";
    out << "    \"output_peak_dbfs\": "
        << jsonNumber(result.limiterStats.outputPeakDbfs) << "\n";
    out << "  },\n";
    out << "  \"loudness\": {\n";
    out << "    \"integrated_lufs\": "
        << (result.loudness.hasIntegratedValue
                ? jsonNumber(result.loudness.integratedLufs)
                : "null")
        << ",\n";
    out << "    \"true_peak_dbtp\": "
        << (std::isfinite(result.loudness.truePeakDbtp)
                ? jsonNumber(result.loudness.truePeakDbtp)
                : "null")
        << ",\n";
    out << "    \"true_peak_linear\": "
        << jsonNumber(result.loudness.truePeakLinear) << ",\n";
    out << "    \"relative_gate_lufs\": "
        << (std::isfinite(result.loudness.relativeGateLufs)
                ? jsonNumber(result.loudness.relativeGateLufs)
                : "null")
        << ",\n";
    out << "    \"total_blocks\": " << result.loudness.totalBlocks << ",\n";
    out << "    \"gated_blocks\": " << result.loudness.gatedBlocks << "\n";
    out << "  },\n";
    out << "  \"timing_ms\": {\n";
    out << "    \"render\": " << jsonNumber(result.totalMs) << ",\n";
    double trackProcessingMs = 0.0;
    for (const TrackReport& track : result.tracks) {
        trackProcessingMs += track.readMs + track.resampleMs + track.gainPanMs +
                             track.drcMs;
    }
    out << "    \"track_processing\": " << jsonNumber(trackProcessingMs) << ",\n";
    out << "    \"mix_accumulate\": " << jsonNumber(result.mixMs) << ",\n";
    out << "    \"track_processing_and_mix\": " << jsonNumber(result.mixMs) << ",\n";
    out << "    \"master_drc\": " << jsonNumber(result.masterDrcMs) << ",\n";
    out << "    \"limiter\": " << jsonNumber(result.limiterMs) << ",\n";
    out << "    \"loudness\": " << jsonNumber(result.loudnessMs) << ",\n";
    out << "    \"write\": " << jsonNumber(writeMs) << ",\n";
    out << "    \"total\": " << jsonNumber(result.totalMs + writeMs) << "\n";
    out << "  }\n";
    out << "}\n";
    return out.str();
}

}  // namespace automix
