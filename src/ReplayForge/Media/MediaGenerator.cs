using System.Globalization;
using ReplayForge.Contracts;
using ReplayForge.Utilities;

namespace ReplayForge.Media;

public sealed class MediaGenerator
{
    private readonly Ffmpeg _ffmpeg;
    private readonly string _manifestDirectory;

    public MediaGenerator(Ffmpeg ffmpeg, string manifestDirectory)
    {
        _ffmpeg = ffmpeg;
        _manifestDirectory = manifestDirectory;
    }

    public async Task<List<string>> GenerateAsync(
        Manifest manifest,
        bool overwrite,
        CancellationToken cancellationToken)
    {
        var generator = manifest.Generator
                        ?? throw new ReplayException("E_MANIFEST_INVALID",
                            "Manifest has no generator section.");
        ValidateGenerator(generator);

        var generated = new List<string>();
        var mediaDirectory = Path.GetDirectoryName(PathUtil.Resolve(_manifestDirectory, manifest.Videos[0].Path))
                             ?? _manifestDirectory;
        Directory.CreateDirectory(mediaDirectory);
        var audioDirectory = Path.GetDirectoryName(PathUtil.Resolve(_manifestDirectory, manifest.Audio.Path))
                             ?? _manifestDirectory;
        Directory.CreateDirectory(audioDirectory);

        for (var index = 0; index < generator.Videos.Count; index++)
        {
            var video = generator.Videos[index];
            var layer = manifest.Videos.FirstOrDefault(item => item.Id == video.Id)
                        ?? throw new ReplayException("E_MANIFEST_INVALID",
                            $"Generator video '{video.Id}' has no matching manifest layer.");
            var path = PathUtil.Resolve(_manifestDirectory, layer.Path);
            if (File.Exists(path) && !overwrite)
            {
                generated.Add(path);
                continue;
            }

            var temporary = path + ".generating-" + Environment.ProcessId;
            JsonIO.TryDelete(temporary);
            var arguments = BuildVideoArguments(video, generator.DurationSeconds, temporary);
            await _ffmpeg.RunCheckedAsync(arguments, $"generate video '{video.Id}'", _manifestDirectory, cancellationToken)
                .ConfigureAwait(false);
            File.Move(temporary, path, overwrite: true);
            generated.Add(path);
        }

        var audioPath = PathUtil.Resolve(_manifestDirectory, manifest.Audio.Path);
        if (!File.Exists(audioPath) || overwrite)
        {
            var temporary = audioPath + ".generating-" + Environment.ProcessId;
            JsonIO.TryDelete(temporary);
            var arguments = BuildAudioArguments(generator.Audio, generator.DurationSeconds, temporary);
            await _ffmpeg.RunCheckedAsync(arguments, "generate commentary audio", _manifestDirectory, cancellationToken)
                .ConfigureAwait(false);
            File.Move(temporary, audioPath, overwrite: true);
            generated.Add(audioPath);
        }
        else
        {
            generated.Add(audioPath);
        }

        return generated;
    }

    private static List<string> BuildVideoArguments(
        GeneratorVideoConfig video,
        double durationSeconds,
        string outputPath)
    {
        if (video.Width <= 0 || video.Height <= 0 || video.Width % 2 != 0 || video.Height % 2 != 0)
        {
            throw new ReplayException("E_MANIFEST_INVALID",
                $"Generator video '{video.Id}' dimensions must be positive even numbers.");
        }

        var pattern = video.Pattern switch
        {
            "testsrc2" => $"testsrc2=size={video.Width}x{video.Height}:rate={video.Fps}:duration={F(durationSeconds)}",
            "smptebars" => $"smptebars=size={video.Width}x{video.Height}:rate={video.Fps}:duration={F(durationSeconds)}",
            _ => throw new ReplayException("E_MANIFEST_INVALID",
                $"Unknown generator video pattern '{video.Pattern}'.")
        };

        var profile = video.Profile.ToLowerInvariant() switch
        {
            "baseline" => "baseline",
            "main" => "main",
            "high" => "high",
            _ => throw new ReplayException("E_MANIFEST_INVALID",
                $"Unknown generator H.264 profile '{video.Profile}'.")
        };

        var videoFilter = video.HueDegrees == 0 ? null : $"hue=h={video.HueDegrees}";
        var gop = Math.Max(1, video.Fps * 2);
        var arguments = new List<string>
        {
            "-f", "lavfi",
            "-i", pattern
        };
        if (videoFilter is not null)
        {
            arguments.AddRange(new[] { "-vf", videoFilter });
        }

        arguments.AddRange(new[]
        {
            "-map_metadata", "-1",
            "-fflags", "+bitexact",
            "-flags:v", "+bitexact",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-profile:v", profile,
            "-pix_fmt", "yuv420p",
            "-g", gop.ToString(CultureInfo.InvariantCulture),
            "-keyint_min", gop.ToString(CultureInfo.InvariantCulture),
            "-sc_threshold", "0",
            "-bf", "0",
            "-threads", "1",
            "-x264-params", $"repeat-headers=1:aud=1:keyint={gop}:min-keyint={gop}:scenecut=0:open-gop=0",
            "-an",
            "-movflags", "+faststart",
            "-output_ts_offset", F(video.StartPtsSeconds),
            "-f", "mp4",
            outputPath
        });
        return arguments;
    }

    private static List<string> BuildAudioArguments(
        GeneratorAudioConfig audio,
        double durationSeconds,
        string outputPath)
    {
        var source = string.Create(
            CultureInfo.InvariantCulture,
            $"sine=frequency={audio.ToneHz}:sample_rate={audio.SampleRate}:duration={durationSeconds}");
        return new List<string>
        {
            "-f", "lavfi",
            "-i", source,
            "-map_metadata", "-1",
            "-fflags", "+bitexact",
            "-af", "volume=0.18,aresample=48000",
            "-ac", audio.Channels.ToString(CultureInfo.InvariantCulture),
            "-ar", audio.SampleRate.ToString(CultureInfo.InvariantCulture),
            "-c:a", "aac",
            "-b:a", "128k",
            "-f", "adts",
            outputPath
        };
    }

    private static void ValidateGenerator(GeneratorConfig generator)
    {
        if (generator.DurationSeconds <= 0)
        {
            throw new ReplayException("E_MANIFEST_INVALID", "Generator duration must be positive.");
        }

        if (generator.Videos.Count == 0)
        {
            throw new ReplayException("E_MANIFEST_INVALID", "Generator must define at least one video.");
        }

        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var video in generator.Videos)
        {
            if (string.IsNullOrWhiteSpace(video.Id) || !ids.Add(video.Id))
            {
                throw new ReplayException("E_MANIFEST_INVALID", "Generator video ids must be unique and non-empty.");
            }

            if (video.Fps <= 0 || video.StartPtsSeconds < 0)
            {
                throw new ReplayException("E_MANIFEST_INVALID",
                    $"Generator video '{video.Id}' has invalid fps or start PTS.");
            }
        }

        if (generator.Audio.SampleRate <= 0 || generator.Audio.Channels is < 1 or > 2)
        {
            throw new ReplayException("E_MANIFEST_INVALID", "Generator audio format is invalid.");
        }
    }

    private static string F(double value) => value.ToString("0.######", CultureInfo.InvariantCulture);
}
