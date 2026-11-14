using System.Text.Json.Serialization;

namespace AudioPipeline;

public sealed class SceneDefinition
{
    [JsonPropertyName("version")]
    public int Version { get; set; }

    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("seed")]
    public int Seed { get; set; }

    [JsonPropertyName("listener")]
    public Vec3 Listener { get; set; }

    [JsonPropertyName("events")]
    public List<EventDefinition>? Events { get; set; }
}

public sealed class EventDefinition
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    [JsonPropertyName("start")]
    public Vec3 Start { get; set; }

    [JsonPropertyName("end")]
    public Vec3 End { get; set; }

    [JsonPropertyName("durationSeconds")]
    public double DurationSeconds { get; set; }

    [JsonPropertyName("delaySeconds")]
    public double DelaySeconds { get; set; }

    [JsonPropertyName("gain")]
    public double Gain { get; set; }

    [JsonPropertyName("sourcePath")]
    public string SourcePath { get; set; } = string.Empty;

    [JsonPropertyName("minDistance")]
    public double MinDistance { get; set; }

    [JsonPropertyName("maxDistance")]
    public double MaxDistance { get; set; }

    [JsonPropertyName("doppler")]
    public bool Doppler { get; set; }

    [JsonPropertyName("occluders")]
    public List<Aabb>? Occluders { get; set; }
}

public sealed class ValidatedEvent
{
    public required int Index { get; init; }

    public required EventDefinition Definition { get; init; }

    public required string AbsoluteSourcePath { get; init; }

    public required string SourceSha256 { get; init; }

    public required Pcm16MonoSource Source { get; init; }

    public required long StartFrame { get; init; }

    public required long DurationFrames { get; init; }

    public required long EndFrame { get; init; }

    public required double DurationSeconds { get; init; }

    public required IReadOnlyList<Aabb> Occluders { get; init; }
}

public sealed class ValidatedScene
{
    public required string ScenePath { get; init; }

    public required SceneDefinition Definition { get; init; }

    public required IReadOnlyList<ValidatedEvent> Events { get; init; }

    public required long TotalFrames { get; init; }

    public required string EventSummaryHash { get; init; }
}
