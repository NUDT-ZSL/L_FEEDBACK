using System.Text.Json.Serialization;

namespace ReplayForge.State;

public sealed class ResumeSnapshot
{
    [JsonPropertyName("schemaVersion")]
    public int SchemaVersion { get; set; } = 1;

    [JsonPropertyName("manifestSha256")]
    public string ManifestSha256 { get; set; } = "";

    [JsonPropertyName("inputVersions")]
    public Dictionary<string, string> InputVersions { get; set; } = new(StringComparer.Ordinal);

    [JsonPropertyName("totalSegments")]
    public int TotalSegments { get; set; }

    [JsonPropertyName("completedSegments")]
    public int CompletedSegments { get; set; }

    [JsonPropertyName("safeOutputLength")]
    public long SafeOutputLength { get; set; }

    [JsonPropertyName("nextAudioFrame")]
    public int NextAudioFrame { get; set; }

    [JsonPropertyName("videoSampleCount")]
    public int VideoSampleCount { get; set; }

    [JsonPropertyName("audioSampleCount")]
    public int AudioSampleCount { get; set; }

    [JsonPropertyName("normalizedAudioSha256")]
    public string? NormalizedAudioSha256 { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = "rendering";

    [JsonPropertyName("cleanShutdown")]
    public bool CleanShutdown { get; set; }

    [JsonPropertyName("updatedAtUtc")]
    public DateTimeOffset UpdatedAtUtc { get; set; } = DateTimeOffset.UtcNow;

    [JsonPropertyName("lastMessage")]
    public string? LastMessage { get; set; }
}

public sealed class SegmentJournal
{
    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("firstFrame")]
    public int FirstFrame { get; set; }

    [JsonPropertyName("frameCount")]
    public int FrameCount { get; set; }

    [JsonPropertyName("startSeconds")]
    public double StartSeconds { get; set; }

    [JsonPropertyName("endSeconds")]
    public double EndSeconds { get; set; }

    [JsonPropertyName("outputStart")]
    public long OutputStart { get; set; }

    [JsonPropertyName("outputEnd")]
    public long OutputEnd { get; set; }

    [JsonPropertyName("videoSamples")]
    public List<JournalSample> VideoSamples { get; set; } = new();

    [JsonPropertyName("audioSamples")]
    public List<JournalSample> AudioSamples { get; set; } = new();

    [JsonPropertyName("codec")]
    public CodecJournal? Codec { get; set; }
}

public sealed class JournalSample
{
    [JsonPropertyName("offset")]
    public long Offset { get; set; }

    [JsonPropertyName("size")]
    public int Size { get; set; }

    [JsonPropertyName("keyframe")]
    public bool Keyframe { get; set; }
}

public sealed class CodecJournal
{
    [JsonPropertyName("avcCBase64")]
    public string AvcCBase64 { get; set; } = "";

    [JsonPropertyName("width")]
    public int Width { get; set; }

    [JsonPropertyName("height")]
    public int Height { get; set; }

    [JsonPropertyName("audioSpecificConfigBase64")]
    public string AudioSpecificConfigBase64 { get; set; } = "";

    [JsonPropertyName("audioSampleRate")]
    public int AudioSampleRate { get; set; }

    [JsonPropertyName("audioChannels")]
    public int AudioChannels { get; set; }
}
