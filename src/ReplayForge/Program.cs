using System.Text.Json;
using ReplayForge.Contracts;
using ReplayForge.Media;
using ReplayForge.Pipeline;
using ReplayForge.Utilities;
using ReplayForge.Validation;

namespace ReplayForge;

public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        Manifest? manifest = null;
        string? manifestPath = null;
        CommandReport report;
        try
        {
            if (args.Length == 0 || args[0] is "help" or "--help" or "-h")
            {
                PrintHelp();
                return 0;
            }

            var command = args[0];
            var options = ParseOptions(args.Skip(1).ToArray());
            switch (command)
            {
                case "generate":
                {
                    var directory = Required(options, "dir");
                    var duration = DoubleOption(options, "duration", 20.0);
                    var seed = IntOption(options, "seed", 1337);
                    var overwrite = options.ContainsKey("overwrite");
                    var ffmpegPath = options.GetValueOrDefault("ffmpeg");
                    manifest = DefaultScenario.Create(duration, seed, ffmpegPath);
                    manifestPath = Path.Combine(directory, "manifest.json");
                    var fullDirectory = Path.GetFullPath(directory);
                    Directory.CreateDirectory(fullDirectory);
                    var ffmpeg = Ffmpeg.Locate(ffmpegPath, fullDirectory);
                    var generated = await new MediaGenerator(ffmpeg, fullDirectory)
                        .GenerateAsync(manifest, overwrite, CancellationToken.None);
                    JsonIO.WriteAtomic(manifestPath, manifest);
                    report = new CommandReport
                    {
                        Ok = true,
                        Command = "generate",
                        ManifestPath = Path.GetFullPath(manifestPath),
                        OutputPath = PathUtil.Resolve(fullDirectory, manifest.OutputPath),
                        GeneratedFiles = generated
                    };
                    break;
                }
                case "run":
                {
                    manifestPath = Path.GetFullPath(Required(options, "manifest"));
                    manifest = JsonIO.DeserializeFile<Manifest>(manifestPath);
                    var manifestDirectory = Path.GetDirectoryName(manifestPath)!;
                    var ffmpeg = Ffmpeg.Locate(
                        options.GetValueOrDefault("ffmpeg") ?? manifest.FfmpegPath,
                        manifestDirectory);
                    report = await new RenderPipeline(ffmpeg).RunAsync(
                        manifest,
                        manifestPath,
                        options.ContainsKey("force-restart"),
                        CancellationToken.None);
                    break;
                }
                case "demo":
                {
                    var directory = Required(options, "dir");
                    var duration = DoubleOption(options, "duration", 20.0);
                    var seed = IntOption(options, "seed", 1337);
                    var overwrite = options.ContainsKey("overwrite");
                    var ffmpegPath = options.GetValueOrDefault("ffmpeg");
                    manifest = DefaultScenario.Create(duration, seed, ffmpegPath);
                    manifestPath = Path.Combine(directory, "manifest.json");
                    var fullDirectory = Path.GetFullPath(directory);
                    Directory.CreateDirectory(fullDirectory);
                    var ffmpeg = Ffmpeg.Locate(ffmpegPath, fullDirectory);
                    await new MediaGenerator(ffmpeg, fullDirectory)
                        .GenerateAsync(manifest, overwrite, CancellationToken.None);
                    JsonIO.WriteAtomic(manifestPath, manifest);
                    report = await new RenderPipeline(ffmpeg).RunAsync(
                        manifest,
                        Path.GetFullPath(manifestPath),
                        options.ContainsKey("force-restart"),
                        CancellationToken.None);
                    report.Command = "demo";
                    report.ManifestPath = Path.GetFullPath(manifestPath);
                    break;
                }
                case "validate":
                {
                    var file = Path.GetFullPath(Required(options, "file"));
                    var info = Mp4Parser.Parse(file);
                    var video = info.VideoTrack
                                ?? throw new ReplayException("E_INVALID_STREAM", "No video track.", new { path = file });
                    var audio = info.AudioTrack
                                ?? throw new ReplayException("E_INVALID_STREAM", "No audio track.", new { path = file });
                    report = new CommandReport
                    {
                        Ok = true,
                        Command = "validate",
                        OutputPath = file,
                        TotalFrames = video.Samples.Count,
                        TotalDurationSeconds = video.DurationSeconds,
                        Validation = new ValidationReport
                        {
                            Ok = true,
                            FileLength = info.FileLength,
                            MoovOffset = info.MoovOffset,
                            MdatDataLength = info.MdatDataLength,
                            VideoTrackDurationSeconds = video.DurationSeconds,
                            AudioTrackDurationSeconds = audio.DurationSeconds,
                            VideoSamples = video.Samples.Count,
                            AudioSamples = audio.Samples.Count,
                            Keyframes = video.Samples.Count(sample => sample.IsKeyframe),
                            Checks = new List<string> { "parsed moov sample tables", "parsed H.264 avcC" }
                        }
                    };
                    break;
                }
                default:
                    throw new ReplayException("E_USAGE", $"Unknown command '{command}'.");
            }
        }
        catch (ReplayException ex)
        {
            report = ErrorReportFor("error", ex);
            await WriteSummaryIfPossibleAsync(manifest, manifestPath, report).ConfigureAwait(false);
            Console.WriteLine(JsonIO.Serialize(report));
            return 0;
        }
        catch (Exception ex)
        {
            var replay = new ReplayException("E_INTERNAL", ex.Message, new { type = ex.GetType().FullName });
            report = ErrorReportFor("error", replay);
            await WriteSummaryIfPossibleAsync(manifest, manifestPath, report).ConfigureAwait(false);
            Console.WriteLine(JsonIO.Serialize(report));
            return 0;
        }

        await WriteSummaryIfPossibleAsync(manifest, manifestPath, report).ConfigureAwait(false);
        Console.WriteLine(JsonIO.Serialize(report));
        return 0;
    }

    private static async Task WriteSummaryIfPossibleAsync(
        Manifest? manifest,
        string? manifestPath,
        CommandReport report)
    {
        if (manifest is null || manifestPath is null || string.IsNullOrWhiteSpace(manifest.SummaryPath))
        {
            return;
        }

        try
        {
            var directory = Path.GetDirectoryName(Path.GetFullPath(manifestPath))!;
            var summaryPath = PathUtil.Resolve(directory, manifest.SummaryPath);
            JsonIO.WriteAtomic(summaryPath, report);
            await Task.CompletedTask;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"warning: cannot write summary file: {ex.Message}");
        }
    }

    private static CommandReport ErrorReportFor(string command, ReplayException exception)
    {
        var error = new ErrorReport
        {
            Code = exception.Code,
            Message = exception.Message
        };
        if (exception.Details is not null)
        {
            var element = JsonSerializer.SerializeToElement(exception.Details, JsonIO.Options);
            error.Details = element;
            error.Layer = FindString(element, "layer");
            error.Path = FindString(element, "path");
            error.ActualStartPtsSeconds = FindDouble(element, "actualStartPtsSeconds");
            error.ExpectedStartPtsSeconds = FindDouble(element, "expectedStartPtsSeconds");
            error.ToleranceSeconds = FindDouble(element, "toleranceSeconds");
        }

        return new CommandReport
        {
            Ok = false,
            Command = command,
            Error = error
        };
    }

    private static string? FindString(JsonElement element, string name)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (var property in element.EnumerateObject())
            {
                if (string.Equals(property.Name, name, StringComparison.OrdinalIgnoreCase)
                    && property.Value.ValueKind == JsonValueKind.String)
                {
                    return property.Value.GetString();
                }

                var nested = FindString(property.Value, name);
                if (nested is not null)
                {
                    return nested;
                }
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in element.EnumerateArray())
            {
                var nested = FindString(item, name);
                if (nested is not null)
                {
                    return nested;
                }
            }
        }

        return null;
    }

    private static double? FindDouble(JsonElement element, string name)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (var property in element.EnumerateObject())
            {
                if (string.Equals(property.Name, name, StringComparison.OrdinalIgnoreCase)
                    && property.Value.TryGetDouble(out var value))
                {
                    return value;
                }

                var nested = FindDouble(property.Value, name);
                if (nested is not null)
                {
                    return nested;
                }
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in element.EnumerateArray())
            {
                var nested = FindDouble(item, name);
                if (nested is not null)
                {
                    return nested;
                }
            }
        }

        return null;
    }

    private static Dictionary<string, string> ParseOptions(string[] args)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            if (arg.StartsWith("--", StringComparison.Ordinal))
            {
                var name = arg[2..];
                if (name.Length == 0)
                {
                    throw new ReplayException("E_USAGE", "Empty option name.");
                }

                if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
                {
                    result[name] = args[++i];
                }
                else
                {
                    result[name] = "true";
                }
            }
            else
            {
                throw new ReplayException("E_USAGE", $"Unexpected positional argument '{arg}'.");
            }
        }

        return result;
    }

    private static string Required(Dictionary<string, string> options, string name)
    {
        if (!options.TryGetValue(name, out var value) || string.IsNullOrWhiteSpace(value))
        {
            throw new ReplayException("E_USAGE", $"Missing required option --{name}.");
        }

        return value;
    }

    private static double DoubleOption(Dictionary<string, string> options, string name, double fallback)
    {
        if (!options.TryGetValue(name, out var value))
        {
            return fallback;
        }

        if (!double.TryParse(value, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var result))
        {
            throw new ReplayException("E_USAGE", $"Option --{name} must be a number.");
        }

        return result;
    }

    private static int IntOption(Dictionary<string, string> options, string name, int fallback)
    {
        if (!options.TryGetValue(name, out var value))
        {
            return fallback;
        }

        if (!int.TryParse(value, System.Globalization.NumberStyles.Integer,
                System.Globalization.CultureInfo.InvariantCulture, out var result))
        {
            throw new ReplayException("E_USAGE", $"Option --{name} must be an integer.");
        }

        return result;
    }

    private static void PrintHelp()
    {
        Console.WriteLine(
            """
            replayforge — deterministic esports replay renderer

              replayforge generate --dir <dir> [--duration 20] [--seed 1337] [--overwrite] [--ffmpeg <path>]
              replayforge run --manifest <manifest.json> [--force-restart] [--ffmpeg <path>]
              replayforge demo --dir <dir> [--duration 20] [--seed 1337] [--overwrite] [--ffmpeg <path>]
              replayforge validate --file <output.mp4>

            Business errors are returned as JSON and the process exit code remains 0.
            """);
    }
}
