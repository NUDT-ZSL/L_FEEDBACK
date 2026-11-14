using System.Globalization;
using ReplayForge.Contracts;
using ReplayForge.Media;
using ReplayForge.Utilities;

namespace ReplayForge.Pipeline;

public sealed class NormalizedAudio
{
    public required string Path { get; init; }

    public required AdtsAudioInfo Info { get; init; }

    public required LoudnessResult Measured { get; init; }

    public required double AppliedGainDb { get; init; }

    public required AudioReport Report { get; init; }
}

public static class AudioNormalizer
{
    public static async Task<NormalizedAudio> NormalizeAsync(
        Manifest manifest,
        string manifestDirectory,
        Ffmpeg ffmpeg,
        string workDirectory,
        string inputHash,
        CancellationToken cancellationToken)
    {
        var audioPath = PathUtil.Resolve(manifestDirectory, manifest.Audio.Path);
        var sourceInfo = AdtsParser.Parse(
            audioPath,
            manifest.Audio.SampleRate,
            manifest.Audio.Channels);
        if (sourceInfo.SampleRate != 48000)
        {
            throw new ReplayException("E_INVALID_STREAM",
                "Commentary audio must be 48 kHz.",
                new { path = audioPath, sampleRate = sourceInfo.SampleRate });
        }

        var cacheDirectory = Path.Combine(workDirectory, "audio");
        Directory.CreateDirectory(cacheDirectory);
        var normalizedPath = Path.Combine(cacheDirectory, "commentary-normalized.aac");
        var metadataPath = Path.Combine(cacheDirectory, "normalization.json");

        LoudnessResult measured;
        double gainDb;
        AudioNormalizationCache? cache = null;
        if (File.Exists(normalizedPath) && File.Exists(metadataPath))
        {
            try
            {
                cache = JsonIO.DeserializeFile<AudioNormalizationCache>(metadataPath);
            }
            catch (ReplayException)
            {
                cache = null;
            }
        }

        if (cache is not null
            && cache.InputSha256 == inputHash
            && Math.Abs(cache.TargetLoudnessLufs - manifest.Timeline.TargetLoudnessLufs) < 1e-9
            && Math.Abs(cache.DurationSeconds - manifest.Timeline.DurationSeconds) < 1e-9
            && Math.Abs(cache.TimelineStartSeconds - manifest.Audio.TimelineStartSeconds) < 1e-9
            && cache.Channels == manifest.Audio.Channels)
        {
            measured = new LoudnessResult
            {
                IntegratedLufs = cache.MeasuredLufs,
                GainDb = 0,
                GatingBlocks = cache.GatingBlocks,
                PeakSample = cache.PeakSample
            };
            gainDb = cache.GainDb;
        }
        else
        {
            measured = await AudioLoudness.MeasureAsync(ffmpeg, audioPath, cancellationToken).ConfigureAwait(false);
            gainDb = AudioLoudness.GainForTarget(measured, manifest.Timeline.TargetLoudnessLufs);

            var temporary = normalizedPath + ".tmp-" + Environment.ProcessId;
            JsonIO.TryDelete(temporary);
            var duration = manifest.Timeline.DurationSeconds;
            var fadeOutStart = Math.Max(0, duration - 0.005);
            var delay = manifest.Audio.TimelineStartSeconds > 0
                ? $"adelay={F(manifest.Audio.TimelineStartSeconds * 1000)}:all=1,"
                : "";
            var filter =
                $"asetpts=PTS-STARTPTS,{delay}volume={F(gainDb)}dB,aresample=48000," +
                $"afade=t=in:st=0:d=0.005,afade=t=out:st={F(fadeOutStart)}:d=0.005," +
                $"apad=pad_dur={F(duration)},atrim=0:{F(duration)}";
            var arguments = new List<string>
            {
                "-i", audioPath,
                "-vn",
                "-map_metadata", "-1",
                "-fflags", "+bitexact",
                "-af", filter,
                "-ac", manifest.Audio.Channels.ToString(CultureInfo.InvariantCulture),
                "-ar", "48000",
                "-c:a", "aac",
                "-b:a", "160k",
                "-f", "adts",
                temporary
            };
            await ffmpeg.RunCheckedAsync(arguments, "normalize commentary audio", workDirectory, cancellationToken)
                .ConfigureAwait(false);
            File.Move(temporary, normalizedPath, overwrite: true);

            JsonIO.WriteAtomic(metadataPath, new AudioNormalizationCache
            {
                InputSha256 = inputHash,
                TargetLoudnessLufs = manifest.Timeline.TargetLoudnessLufs,
                DurationSeconds = duration,
                TimelineStartSeconds = manifest.Audio.TimelineStartSeconds,
                Channels = manifest.Audio.Channels,
                MeasuredLufs = measured.IntegratedLufs,
                GainDb = gainDb,
                GatingBlocks = measured.GatingBlocks,
                PeakSample = measured.PeakSample
            });
        }

        var outputInfo = AdtsParser.Parse(normalizedPath, 48000, manifest.Audio.Channels);
        return new NormalizedAudio
        {
            Path = normalizedPath,
            Info = outputInfo,
            Measured = measured,
            AppliedGainDb = gainDb,
            Report = new AudioReport
            {
                SourceSampleRate = sourceInfo.SampleRate,
                OutputSampleRate = outputInfo.SampleRate,
                Channels = outputInfo.Channels,
                MeasuredLoudnessLufs = measured.IntegratedLufs,
                TargetLoudnessLufs = manifest.Timeline.TargetLoudnessLufs,
                AppliedGainDb = gainDb,
                SampleFrames = outputInfo.Frames.Count
            }
        };
    }

    private static string F(double value) => value.ToString("0.######", CultureInfo.InvariantCulture);
}

public sealed class AudioNormalizationCache
{
    public string InputSha256 { get; set; } = "";

    public double TargetLoudnessLufs { get; set; }

    public double DurationSeconds { get; set; }

    public double TimelineStartSeconds { get; set; }

    public int Channels { get; set; }

    public double MeasuredLufs { get; set; }

    public double GainDb { get; set; }

    public int GatingBlocks { get; set; }

    public double PeakSample { get; set; }
}
