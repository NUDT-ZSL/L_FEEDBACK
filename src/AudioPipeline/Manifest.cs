using System.Text.Json.Serialization;

namespace AudioPipeline;

public sealed class BlockManifest
{
    [JsonPropertyName("manifestVersion")]
    public int ManifestVersion { get; set; }

    [JsonPropertyName("algorithmVersion")]
    public string AlgorithmVersion { get; set; } = string.Empty;

    [JsonPropertyName("sampleRate")]
    public int SampleRate { get; set; }

    [JsonPropertyName("channels")]
    public int Channels { get; set; }

    [JsonPropertyName("bitsPerSample")]
    public int BitsPerSample { get; set; }

    [JsonPropertyName("blockFrames")]
    public int BlockFrames { get; set; }

    [JsonPropertyName("totalFrames")]
    public long TotalFrames { get; set; }

    [JsonPropertyName("totalBlocks")]
    public int TotalBlocks { get; set; }

    [JsonPropertyName("eventSummaryHash")]
    public string EventSummaryHash { get; set; } = string.Empty;

    [JsonPropertyName("eventSummaryAlgorithm")]
    public string EventSummaryAlgorithm { get; set; } = AudioConstants.EventSummaryAlgorithm;

    [JsonPropertyName("blocks")]
    public List<BlockRecord> Blocks { get; set; } = [];
}

public sealed class BlockRecord
{
    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = BlockStatus.Pending;

    [JsonPropertyName("frameCount")]
    public int FrameCount { get; set; }

    [JsonPropertyName("byteLength")]
    public long ByteLength { get; set; }

    [JsonPropertyName("sha256")]
    public string Sha256 { get; set; } = string.Empty;

    [JsonPropertyName("exitStates")]
    public List<EventExitState> ExitStates { get; set; } = [];
}

public sealed class EventExitState
{
    [JsonPropertyName("eventId")]
    public string EventId { get; set; } = string.Empty;

    [JsonPropertyName("nextLocalFrame")]
    public long NextLocalFrame { get; set; }

    [JsonPropertyName("lowPassState")]
    public double LowPassState { get; set; }

    [JsonPropertyName("dopplerReadPosition")]
    public double DopplerReadPosition { get; set; }

    [JsonPropertyName("dopplerDelayPhase")]
    public double DopplerDelayPhase { get; set; }
}

public static class BlockStatus
{
    public const string Pending = "pending";
    public const string Complete = "complete";
}
