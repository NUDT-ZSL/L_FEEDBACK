using System.Text.Json;
using AudioPipeline;

var jsonOptions = new JsonSerializerOptions
{
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    WriteIndented = true
};

if (args.Length == 0 || IsHelp(args[0]))
{
    PrintUsage();
    return 0;
}

try
{
    return args[0].ToLowerInvariant() switch
    {
        "fixture" or "generate" => RunFixture(args[1..], jsonOptions),
        "validate" => RunValidate(args[1..], jsonOptions),
        "render" or "resume" => RunRender(args[1..], jsonOptions),
        "inspect" => RunInspect(args[1..], jsonOptions),
        "manifest" => RunManifest(args[1..], jsonOptions),
        "help" or "--help" or "-h" => PrintHelpAndReturn(),
        _ => UnknownCommand(args[0])
    };
}
catch (PipelineException ex)
{
    Console.Error.WriteLine($"{ex.StableCode}: {ex.Message}");
    Console.WriteLine(JsonSerializer.Serialize(ex.ToEnvelope(), jsonOptions));
    return 1;
}
catch (Exception ex)
{
    var envelope = new ErrorEnvelope
    {
        Code = (int)PipelineErrorCode.RenderIo,
        Name = PipelineErrorNames.RenderIo,
        Message = ex.Message
    };
    Console.Error.WriteLine($"{envelope.Name}: {envelope.Message}");
    Console.WriteLine(JsonSerializer.Serialize(envelope, jsonOptions));
    return 1;
}

static int RunFixture(string[] args, JsonSerializerOptions jsonOptions)
{
    var options = ParseOptions(args);
    var output = RequireOption(options, "--output", "-o");
    var seed = ParseIntOption(options, "--seed", FixtureGenerator.DefaultSeed);
    var result = FixtureGenerator.Generate(output, seed);
    Console.WriteLine(JsonSerializer.Serialize(result, jsonOptions));
    return 0;
}

static int RunValidate(string[] args, JsonSerializerOptions jsonOptions)
{
    var options = ParseOptions(args);
    var scenePath = RequireOption(options, "--scene", "-s");
    var scene = SceneLoader.LoadAndValidate(scenePath);
    var result = new
    {
        scenePath = scene.ScenePath,
        scene.Definition.Name,
        eventCount = scene.Events.Count,
        scene.TotalFrames,
        durationSeconds = scene.TotalFrames / (double)AudioConstants.SampleRate,
        eventSummaryHash = scene.EventSummaryHash,
        events = scene.Events.Select(item => new
        {
            item.Definition.Id,
            item.StartFrame,
            item.DurationFrames,
            item.EndFrame,
            sourceSha256 = item.SourceSha256
        })
    };
    Console.WriteLine(JsonSerializer.Serialize(result, jsonOptions));
    return 0;
}

static int RunRender(string[] args, JsonSerializerOptions jsonOptions)
{
    var options = ParseOptions(args);
    var scenePath = RequireOption(options, "--scene", "-s");
    var outputPath = RequireOption(options, "--output", "-o");
    var renderOptions = new RenderOptions
    {
        MaxBlocksThisRun = ParseIntOption(options, "--stop-after-blocks", int.MaxValue),
        MaxBlockBufferBytes = ParseIntOption(
            options,
            "--max-block-buffer-bytes",
            AudioConstants.DefaultMaxBlockBufferBytes)
    };
    if (renderOptions.MaxBlocksThisRun < 0 || renderOptions.MaxBlockBufferBytes <= 0)
    {
        throw new PipelineException(
            PipelineErrorCode.SceneValidation,
            "Render limits must be positive (--stop-after-blocks may be zero).");
    }

    using var cancellation = new CancellationTokenSource();
    ConsoleCancelEventHandler handler = (_, eventArgs) =>
    {
        eventArgs.Cancel = true;
        cancellation.Cancel();
        Console.Error.WriteLine("Ctrl+C received; the active block will finish before stopping.");
    };
    Console.CancelKeyPress += handler;
    try
    {
        var result = OfflineRenderer.Render(
            scenePath,
            outputPath,
            renderOptions,
            cancellation.Token,
            progress => Console.Error.WriteLine(
                $"block {progress.Index:D6} {(progress.Reused ? "reused" : "rendered")} " +
                progress.Sha256[..12]));
        Console.WriteLine(JsonSerializer.Serialize(result, jsonOptions));
        if (result.Cancelled)
        {
            return 130;
        }

        return result.Completed ? 0 : 3;
    }
    finally
    {
        Console.CancelKeyPress -= handler;
    }
}

static int RunInspect(string[] args, JsonSerializerOptions jsonOptions)
{
    var options = ParseOptions(args);
    var wavPath = RequireOption(options, "--wav", "-w");
    var info = WavIO.ReadInfo(wavPath);
    var result = new
    {
        wavPath = Path.GetFullPath(wavPath),
        info.FormatTag,
        info.Channels,
        info.SampleRate,
        info.BitsPerSample,
        info.BlockAlign,
        info.FrameCount,
        info.DataByteLength,
        info.FileLength
    };
    Console.WriteLine(JsonSerializer.Serialize(result, jsonOptions));
    return 0;
}

static int RunManifest(string[] args, JsonSerializerOptions jsonOptions)
{
    var options = ParseOptions(args);
    var outputPath = RequireOption(options, "--output", "-o");
    var store = new ManifestStore(outputPath);
    var manifest = store.TryLoadManifest() ??
                   throw new PipelineException(
                       PipelineErrorCode.ManifestInvalid,
                       $"No readable manifest exists at '{store.ManifestPath}'.");
    Console.WriteLine(JsonSerializer.Serialize(manifest, jsonOptions));
    return 0;
}

static Dictionary<string, string> ParseOptions(string[] args)
{
    var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    for (var index = 0; index < args.Length; index++)
    {
        var key = args[index];
        if (!key.StartsWith("-", StringComparison.Ordinal))
        {
            throw new PipelineException(
                PipelineErrorCode.SceneValidation,
                $"Unexpected argument '{key}'.");
        }

        if (index + 1 >= args.Length)
        {
            throw new PipelineException(
                PipelineErrorCode.SceneValidation,
                $"Missing value for option '{key}'.");
        }

        result[key] = args[++index];
    }

    return result;
}

static string RequireOption(
    IReadOnlyDictionary<string, string> options,
    params string[] names)
{
    foreach (var name in names)
    {
        if (options.TryGetValue(name, out var value) && !string.IsNullOrWhiteSpace(value))
        {
            return value;
        }
    }

    throw new PipelineException(
        PipelineErrorCode.SceneValidation,
        $"Required option '{names[0]}' was not supplied.");
}

static int ParseIntOption(
    IReadOnlyDictionary<string, string> options,
    string name,
    int defaultValue)
{
    if (!options.TryGetValue(name, out var text))
    {
        return defaultValue;
    }

    if (!int.TryParse(
            text,
            System.Globalization.NumberStyles.Integer,
            System.Globalization.CultureInfo.InvariantCulture,
            out var value))
    {
        throw new PipelineException(
            PipelineErrorCode.SceneValidation,
            $"Option '{name}' must be a base-10 integer.");
    }

    return value;
}

static int UnknownCommand(string command)
{
    Console.Error.WriteLine($"Unknown command '{command}'.");
    PrintUsage();
    return 2;
}

static int PrintHelpAndReturn()
{
    PrintUsage();
    return 0;
}

static bool IsHelp(string value) =>
    value is "--help" or "-h" or "help";

static void PrintUsage()
{
    Console.WriteLine(
        """
        Spatial audio offline renderer

        Commands:
          fixture  --output DIR [--seed N]
          validate --scene FILE
          render   --scene FILE --output FILE [--stop-after-blocks N]
                   [--max-block-buffer-bytes N]
          inspect  --wav FILE
          manifest --output FILE

        The renderer always writes stereo 48 kHz 16-bit PCM. Block state is stored
        under <output>.state and an exclusive .audio-pipeline.lock is held in the
        output directory while a writer is active.
        """);
}
