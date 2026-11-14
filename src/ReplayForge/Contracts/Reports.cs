using System.Text.Json.Serialization;

namespace ReplayForge.Contracts;

public sealed class CommandReport
{
    [JsonPropertyName("ok")]
    public bool Ok { get; set; }

    [JsonPropertyName("command")]
    public string Command { get; set; } = "";

    [JsonPropertyName("error")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ErrorReport? Error { get; set; }

    [JsonPropertyName("outputPath")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? OutputPath { get; set; }

    [JsonPropertyName("totalFrames")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? TotalFrames { get; set; }

    [JsonPropertyName("totalDurationSeconds")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? TotalDurationSeconds { get; set; }

    [JsonPropertyName("manifestPath")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ManifestPath { get; set; }

    [JsonPropertyName("generatedFiles")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<string>? GeneratedFiles { get; set; }

    [JsonPropertyName("alignment")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public AlignmentReport? Alignment { get; set; }

    [JsonPropertyName("resampling")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ResamplingReport? Resampling { get; set; }

    [JsonPropertyName("audio")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public AudioReport? Audio { get; set; }

    [JsonPropertyName("recovery")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public RecoveryReport? Recovery { get; set; }

    [JsonPropertyName("validation")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ValidationReport? Validation { get; set; }

    [JsonPropertyName("warnings")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<string>? Warnings { get; set; }
}

public sealed class ErrorReport
{
    [JsonPropertyName("code")]
    public string Code { get; set; } = "E_INTERNAL";

    [JsonPropertyName("message")]
    public string Message { get; set; } = "";

    [JsonPropertyName("layer")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Layer { get; set; }

    [JsonPropertyName("path")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Path { get; set; }

    [JsonPropertyName("actualStartPtsSeconds")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? ActualStartPtsSeconds { get; set; }

    [JsonPropertyName("expectedStartPtsSeconds")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? ExpectedStartPtsSeconds { get; set; }

    [JsonPropertyName("toleranceSeconds")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? ToleranceSeconds { get; set; }

    [JsonPropertyName("details")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public object? Details { get; set; }
}

public sealed class AlignmentReport
{
    [JsonPropertyName("masterStartPtsSeconds")]
    public double MasterStartPtsSeconds { get; set; }

    [JsonPropertyName("toleranceSeconds")]
    public double ToleranceSeconds { get; set; }

    [JsonPropertyName("layers")]
    public List<LayerAlignmentReport> Layers { get; set; } = new();
}

public sealed class LayerAlignmentReport
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("path")]
    public string Path { get; set; } = "";

    [JsonPropertyName("sourceFirstPtsSeconds")]
    public double SourceFirstPtsSeconds { get; set; }

    [JsonPropertyName("expectedFirstPtsSeconds")]
    public double ExpectedFirstPtsSeconds { get; set; }

    [JsonPropertyName("firstPtsErrorSeconds")]
    public double FirstPtsErrorSeconds { get; set; }

    [JsonPropertyName("mappedFirstFrameTimelineSeconds")]
    public double MappedFirstFrameTimelineSeconds { get; set; }

    [JsonPropertyName("sourceFps")]
    public double SourceFps { get; set; }

    [JsonPropertyName("sourceFrameCount")]
    public int SourceFrameCount { get; set; }

    [JsonPropertyName("snappedToFrame")]
    public bool SnappedToFrame { get; set; }
}

public sealed class ResamplingReport
{
    [JsonPropertyName("outputFps")]
    public int OutputFps { get; set; }

    [JsonPropertyName("cumulativeOutputFrames")]
    public int CumulativeOutputFrames { get; set; }

    [JsonPropertyName("layers")]
    public List<LayerResamplingReport> Layers { get; set; } = new();
}

public sealed class LayerResamplingReport
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("sourceFramesInWindow")]
    public int SourceFramesInWindow { get; set; }

    [JsonPropertyName("emittedFrames")]
    public int EmittedFrames { get; set; }

    [JsonPropertyName("droppedFrames")]
    public int DroppedFrames { get; set; }

    [JsonPropertyName("interpolatedFrames")]
    public int InterpolatedFrames { get; set; }
}

public sealed class AudioReport
{
    [JsonPropertyName("sourceSampleRate")]
    public int SourceSampleRate { get; set; }

    [JsonPropertyName("outputSampleRate")]
    public int OutputSampleRate { get; set; }

    [JsonPropertyName("channels")]
    public int Channels { get; set; }

    [JsonPropertyName("measuredLoudnessLufs")]
    public double MeasuredLoudnessLufs { get; set; }

    [JsonPropertyName("targetLoudnessLufs")]
    public double TargetLoudnessLufs { get; set; }

    [JsonPropertyName("appliedGainDb")]
    public double AppliedGainDb { get; set; }

    [JsonPropertyName("sampleFrames")]
    public int SampleFrames { get; set; }
}

public sealed class RecoveryReport
{
    [JsonPropertyName("mode")]
    public string Mode { get; set; } = "fresh";

    [JsonPropertyName("previousCleanShutdown")]
    public bool? PreviousCleanShutdown { get; set; }

    [JsonPropertyName("resumedFromSegment")]
    public int ResumedFromSegment { get; set; }

    [JsonPropertyName("rolledBackBytes")]
    public long RolledBackBytes { get; set; }

    [JsonPropertyName("discardedPartialSegments")]
    public int DiscardedPartialSegments { get; set; }

    [JsonPropertyName("logs")]
    public List<string> Logs { get; set; } = new();
}

public sealed class ValidationReport
{
    [JsonPropertyName("ok")]
    public bool Ok { get; set; }

    [JsonPropertyName("fileLength")]
    public long FileLength { get; set; }

    [JsonPropertyName("moovOffset")]
    public long MoovOffset { get; set; }

    [JsonPropertyName("mdatDataLength")]
    public long MdatDataLength { get; set; }

    [JsonPropertyName("videoTrackDurationSeconds")]
    public double VideoTrackDurationSeconds { get; set; }

    [JsonPropertyName("audioTrackDurationSeconds")]
    public double AudioTrackDurationSeconds { get; set; }

    [JsonPropertyName("videoSamples")]
    public int VideoSamples { get; set; }

    [JsonPropertyName("audioSamples")]
    public int AudioSamples { get; set; }

    [JsonPropertyName("keyframes")]
    public int Keyframes { get; set; }

    [JsonPropertyName("checks")]
    public List<string> Checks { get; set; } = new();
}
