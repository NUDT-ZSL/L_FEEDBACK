using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace AudioPipeline;

public static class SceneLoader
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        AllowTrailingCommas = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true
    };

    public static ValidatedScene LoadAndValidate(string scenePath)
    {
        string fullScenePath;
        try
        {
            fullScenePath = Path.GetFullPath(scenePath);
        }
        catch (Exception ex) when (
            ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            throw Validation($"Invalid scene path '{scenePath}': {ex.Message}");
        }

        SceneDefinition? scene;
        try
        {
            var json = File.ReadAllText(fullScenePath, Encoding.UTF8);
            scene = JsonSerializer.Deserialize<SceneDefinition>(json, JsonOptions);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new PipelineException(
                PipelineErrorCode.SceneIo,
                $"Unable to read scene '{fullScenePath}': {ex.Message}",
                ex);
        }
        catch (JsonException ex)
        {
            throw new PipelineException(
                PipelineErrorCode.SceneJson,
                $"Invalid scene JSON: {ex.Message}",
                ex);
        }

        if (scene is null)
        {
            throw Validation("Scene JSON produced a null scene.");
        }

        if (scene.Version != 1)
        {
            throw new PipelineException(
                PipelineErrorCode.SceneVersion,
                $"Unsupported scene version {scene.Version}; expected version 1.");
        }

        if (scene.Events is null || scene.Events.Count == 0)
        {
            throw Validation("The scene must contain at least one event.");
        }

        if (scene.Events.Count > AudioConstants.MaxEvents)
        {
            throw new PipelineException(
                PipelineErrorCode.EventLimit,
                $"The scene contains {scene.Events.Count} events; the limit is {AudioConstants.MaxEvents}.");
        }

        if (!scene.Listener.IsFinite)
        {
            throw Validation("Listener coordinates must be finite.");
        }

        var sceneDirectory = Path.GetDirectoryName(fullScenePath) ?? Directory.GetCurrentDirectory();
        var seenIds = new HashSet<string>(StringComparer.Ordinal);
        var sourceCache = new Dictionary<string, Pcm16MonoSource>(StringComparer.OrdinalIgnoreCase);
        var events = new List<ValidatedEvent>(scene.Events.Count);
        var totalFrames = 0L;

        for (var index = 0; index < scene.Events.Count; index++)
        {
            var definition = scene.Events[index] ??
                throw Validation($"Event #{index} is null.");
            ValidateEvent(definition, index);

            if (!seenIds.Add(definition.Id))
            {
                throw Validation($"Duplicate event id '{definition.Id}'.");
            }

            var durationFrames = FrameMath.DurationFrameCount(definition.DurationSeconds);
            if (durationFrames < 1)
            {
                throw Validation($"Event '{definition.Id}' duration rounds to zero frames.");
            }

            var startFrame = FrameMath.StartFrame(definition.DelaySeconds);
            var endFrame = checked(startFrame + durationFrames);
            if (endFrame > AudioConstants.MaxFrames)
            {
                throw new PipelineException(
                    PipelineErrorCode.DurationLimit,
                    $"Event '{definition.Id}' ends at frame {endFrame}; the limit is {AudioConstants.MaxFrames} " +
                    $"({AudioConstants.MaxTimelineSeconds:0} seconds).");
            }

            string sourcePath;
            try
            {
                sourcePath = Path.GetFullPath(Path.Combine(sceneDirectory, definition.SourcePath));
            }
            catch (Exception ex) when (
                ex is ArgumentException or NotSupportedException or PathTooLongException)
            {
                throw Validation(
                    $"Event '{definition.Id}' has an invalid sourcePath: {ex.Message}");
            }

            if (!File.Exists(sourcePath))
            {
                throw new PipelineException(
                    PipelineErrorCode.SourceMissing,
                    $"Source for event '{definition.Id}' was not found: '{sourcePath}'.");
            }

            if (!sourceCache.TryGetValue(sourcePath, out var source))
            {
                source = WavIO.ReadMonoPcm16(sourcePath, AudioConstants.MaxFrames);
                sourceCache.Add(sourcePath, source);
            }

            var sourceHash = WavIO.Sha256Hex(sourcePath);
            var occluders = definition.Occluders ?? [];
            events.Add(new ValidatedEvent
            {
                Index = index,
                Definition = definition,
                AbsoluteSourcePath = sourcePath,
                SourceSha256 = sourceHash,
                Source = source,
                StartFrame = startFrame,
                DurationFrames = durationFrames,
                EndFrame = endFrame,
                DurationSeconds = durationFrames / (double)AudioConstants.SampleRate,
                Occluders = occluders.ToArray()
            });
            totalFrames = Math.Max(totalFrames, endFrame);
        }

        if (totalFrames <= 0)
        {
            throw Validation("The scene produces no output frames.");
        }

        var eventSummaryHash = ComputeEventSummaryHash(scene, events);
        return new ValidatedScene
        {
            ScenePath = fullScenePath,
            Definition = scene,
            Events = events,
            TotalFrames = totalFrames,
            EventSummaryHash = eventSummaryHash
        };
    }

    public static string ComputeEventSummaryHash(
        SceneDefinition scene,
        IReadOnlyList<ValidatedEvent> events)
    {
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = false }))
        {
            writer.WriteStartObject();
            writer.WriteNumber("sceneVersion", scene.Version);
            writer.WriteNumber("seed", scene.Seed);
            writer.WriteNumber("sampleRate", AudioConstants.SampleRate);
            writer.WriteNumber("blockFrames", AudioConstants.BlockFrames);
            writer.WriteString("algorithmVersion", AudioConstants.AlgorithmVersion);
            writer.WriteStartObject("listener");
            WriteVec3(writer, scene.Listener);
            writer.WriteEndObject();
            writer.WriteStartArray("events");
            foreach (var item in events)
            {
                var definition = item.Definition;
                writer.WriteStartObject();
                writer.WriteString("id", definition.Id);
                writer.WriteStartObject("start");
                WriteVec3(writer, definition.Start);
                writer.WriteEndObject();
                writer.WriteStartObject("end");
                WriteVec3(writer, definition.End);
                writer.WriteEndObject();
                writer.WriteNumber("durationSeconds", definition.DurationSeconds);
                writer.WriteNumber("delaySeconds", definition.DelaySeconds);
                writer.WriteNumber("gain", definition.Gain);
                writer.WriteString("sourceSha256", item.SourceSha256);
                writer.WriteNumber("minDistance", definition.MinDistance);
                writer.WriteNumber("maxDistance", definition.MaxDistance);
                writer.WriteBoolean("doppler", definition.Doppler);
                writer.WriteStartArray("occluders");
                foreach (var occluder in item.Occluders)
                {
                    writer.WriteStartObject();
                    writer.WriteStartObject("min");
                    WriteVec3(writer, occluder.Min);
                    writer.WriteEndObject();
                    writer.WriteStartObject("max");
                    WriteVec3(writer, occluder.Max);
                    writer.WriteEndObject();
                    writer.WriteEndObject();
                }

                writer.WriteEndArray();
                writer.WriteEndObject();
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        return Convert.ToHexString(SHA256.HashData(stream.ToArray()));
    }

    private static void ValidateEvent(EventDefinition definition, int index)
    {
        var label = string.IsNullOrWhiteSpace(definition.Id)
            ? $"Event #{index}"
            : $"Event '{definition.Id}'";

        if (string.IsNullOrWhiteSpace(definition.Id))
        {
            throw Validation($"{label} must have a non-empty id.");
        }

        if (!definition.Start.IsFinite || !definition.End.IsFinite)
        {
            throw Validation($"{label} coordinates must be finite.");
        }

        if (!double.IsFinite(definition.DurationSeconds) ||
            definition.DurationSeconds <= 0 ||
            definition.DurationSeconds > AudioConstants.MaxTimelineSeconds)
        {
            throw new PipelineException(
                PipelineErrorCode.DurationLimit,
                $"{label} duration must be finite, positive, and at most " +
                $"{AudioConstants.MaxTimelineSeconds:0} seconds.");
        }

        if (!double.IsFinite(definition.DelaySeconds) ||
            definition.DelaySeconds < 0 ||
            definition.DelaySeconds > AudioConstants.MaxTimelineSeconds)
        {
            throw new PipelineException(
                PipelineErrorCode.DurationLimit,
                $"{label} delay must be finite, non-negative, and at most " +
                $"{AudioConstants.MaxTimelineSeconds:0} seconds.");
        }

        if (!double.IsFinite(definition.Gain) ||
            Math.Abs(definition.Gain) > 1_000_000)
        {
            throw Validation(
                $"{label} gain must be finite and have an absolute value no greater than 1,000,000.");
        }

        if (!double.IsFinite(definition.MinDistance) ||
            !double.IsFinite(definition.MaxDistance) ||
            definition.MinDistance < 0 ||
            definition.MaxDistance <= definition.MinDistance)
        {
            throw Validation(
                $"{label} minDistance/maxDistance must be finite and satisfy " +
                "0 <= minDistance < maxDistance.");
        }

        if (string.IsNullOrWhiteSpace(definition.SourcePath))
        {
            throw Validation($"{label} must specify sourcePath.");
        }

        if (definition.Occluders is not null)
        {
            foreach (var occluder in definition.Occluders)
            {
                if (!occluder.IsValid)
                {
                    throw Validation($"{label} contains an invalid AABB occluder.");
                }
            }
        }
    }

    private static void WriteVec3(Utf8JsonWriter writer, Vec3 value)
    {
        writer.WriteNumber("x", value.X);
        writer.WriteNumber("y", value.Y);
        writer.WriteNumber("z", value.Z);
    }

    private static PipelineException Validation(string message) =>
        new(PipelineErrorCode.SceneValidation, message);
}
