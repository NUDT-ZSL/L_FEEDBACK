using System.Text.Json.Serialization;

namespace ReplayForge.Contracts;

public sealed class Manifest
{
    [JsonPropertyName("schemaVersion")]
    public int SchemaVersion { get; set; } = 1;

    [JsonPropertyName("generator")]
    public GeneratorConfig? Generator { get; set; }

    [JsonPropertyName("timeline")]
    public TimelineConfig Timeline { get; set; } = new();

    [JsonPropertyName("videos")]
    public List<VideoLayerConfig> Videos { get; set; } = new();

    [JsonPropertyName("audio")]
    public AudioConfig Audio { get; set; } = new();

    [JsonPropertyName("outputPath")]
    public string OutputPath { get; set; } = "output/final.mp4";

    [JsonPropertyName("workDirectory")]
    public string WorkDirectory { get; set; } = "work";

    [JsonPropertyName("summaryPath")]
    public string? SummaryPath { get; set; }

    [JsonPropertyName("ffmpegPath")]
    public string? FfmpegPath { get; set; }
}

public sealed class GeneratorConfig
{
    [JsonPropertyName("seed")]
    public int Seed { get; set; } = 1337;

    [JsonPropertyName("durationSeconds")]
    public double DurationSeconds { get; set; } = 20.0;

    [JsonPropertyName("videos")]
    public List<GeneratorVideoConfig> Videos { get; set; } = new();

    [JsonPropertyName("audio")]
    public GeneratorAudioConfig Audio { get; set; } = new();
}

public sealed class GeneratorVideoConfig
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("fileName")]
    public string FileName { get; set; } = "";

    [JsonPropertyName("pattern")]
    public string Pattern { get; set; } = "testsrc2";

    [JsonPropertyName("width")]
    public int Width { get; set; } = 320;

    [JsonPropertyName("height")]
    public int Height { get; set; } = 180;

    [JsonPropertyName("fps")]
    public int Fps { get; set; } = 30;

    [JsonPropertyName("profile")]
    public string Profile { get; set; } = "High";

    [JsonPropertyName("startPtsSeconds")]
    public double StartPtsSeconds { get; set; }

    [JsonPropertyName("hueDegrees")]
    public int HueDegrees { get; set; }
}

public sealed class GeneratorAudioConfig
{
    [JsonPropertyName("fileName")]
    public string FileName { get; set; } = "commentary.aac";

    [JsonPropertyName("sampleRate")]
    public int SampleRate { get; set; } = 48000;

    [JsonPropertyName("channels")]
    public int Channels { get; set; } = 2;

    [JsonPropertyName("toneHz")]
    public double ToneHz { get; set; } = 523.25;
}

public sealed class TimelineConfig
{
    [JsonPropertyName("masterStartPtsSeconds")]
    public double MasterStartPtsSeconds { get; set; } = 10.0;

    [JsonPropertyName("durationSeconds")]
    public double DurationSeconds { get; set; } = 20.0;

    [JsonPropertyName("outputFps")]
    public int OutputFps { get; set; } = 30;

    [JsonPropertyName("width")]
    public int Width { get; set; } = 640;

    [JsonPropertyName("height")]
    public int Height { get; set; } = 360;

    [JsonPropertyName("syncToleranceSeconds")]
    public double SyncToleranceSeconds { get; set; } = 0.020;

    [JsonPropertyName("targetLoudnessLufs")]
    public double TargetLoudnessLufs { get; set; } = -16.0;

    [JsonPropertyName("segmentDurationSeconds")]
    public double SegmentDurationSeconds { get; set; } = 1.0;
}

public sealed class VideoLayerConfig
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("role")]
    public string Role { get; set; } = "main";

    [JsonPropertyName("path")]
    public string Path { get; set; } = "";

    [JsonPropertyName("timelineStartSeconds")]
    public double TimelineStartSeconds { get; set; }

    [JsonPropertyName("layout")]
    public LayoutConfig Layout { get; set; } = new();
}

public sealed class LayoutConfig
{
    [JsonPropertyName("x")]
    public int X { get; set; }

    [JsonPropertyName("y")]
    public int Y { get; set; }

    [JsonPropertyName("width")]
    public int Width { get; set; }

    [JsonPropertyName("height")]
    public int Height { get; set; }

    [JsonPropertyName("opacity")]
    public double Opacity { get; set; } = 1.0;
}

public sealed class AudioConfig
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "commentary";

    [JsonPropertyName("path")]
    public string Path { get; set; } = "";

    [JsonPropertyName("timelineStartSeconds")]
    public double TimelineStartSeconds { get; set; }

    [JsonPropertyName("sampleRate")]
    public int SampleRate { get; set; } = 48000;

    [JsonPropertyName("channels")]
    public int Channels { get; set; } = 2;
}
