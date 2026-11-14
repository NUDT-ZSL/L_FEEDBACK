using System.Text.Json.Serialization;

namespace AudioPipeline;

public enum PipelineErrorCode
{
    Ok = 0,
    SceneIo = 1001,
    SceneJson = 1002,
    SceneVersion = 1003,
    SceneValidation = 1004,
    SourceMissing = 1005,
    SourceFormat = 1006,
    DurationLimit = 1007,
    EventLimit = 1008,
    BlockBufferLimit = 1009,
    OutputLocked = 2001,
    ManifestInvalid = 2002,
    RenderIo = 3001,
    Cancelled = 3002
}

public static class PipelineErrorNames
{
    public const string SceneIo = "SCENE_IO";
    public const string SceneJson = "SCENE_JSON";
    public const string SceneVersion = "SCENE_VERSION";
    public const string SceneValidation = "SCENE_VALIDATION";
    public const string SourceMissing = "SOURCE_MISSING";
    public const string SourceFormat = "SOURCE_FORMAT";
    public const string DurationLimit = "DURATION_LIMIT";
    public const string EventLimit = "EVENT_LIMIT";
    public const string BlockBufferLimit = "BLOCK_BUFFER_LIMIT";
    public const string OutputLocked = "OUTPUT_LOCKED";
    public const string ManifestInvalid = "MANIFEST_INVALID";
    public const string RenderIo = "RENDER_IO";
    public const string Cancelled = "CANCELLED";

    public static string For(PipelineErrorCode code) => code switch
    {
        PipelineErrorCode.SceneIo => SceneIo,
        PipelineErrorCode.SceneJson => SceneJson,
        PipelineErrorCode.SceneVersion => SceneVersion,
        PipelineErrorCode.SceneValidation => SceneValidation,
        PipelineErrorCode.SourceMissing => SourceMissing,
        PipelineErrorCode.SourceFormat => SourceFormat,
        PipelineErrorCode.DurationLimit => DurationLimit,
        PipelineErrorCode.EventLimit => EventLimit,
        PipelineErrorCode.BlockBufferLimit => BlockBufferLimit,
        PipelineErrorCode.OutputLocked => OutputLocked,
        PipelineErrorCode.ManifestInvalid => ManifestInvalid,
        PipelineErrorCode.RenderIo => RenderIo,
        PipelineErrorCode.Cancelled => Cancelled,
        _ => "OK"
    };
}

public sealed class PipelineException : Exception
{
    public PipelineException(PipelineErrorCode code, string message, Exception? inner = null)
        : base(message, inner)
    {
        Code = code;
    }

    public PipelineErrorCode Code { get; }

    public int NumericCode => (int)Code;

    public string StableCode => PipelineErrorNames.For(Code);

    public ErrorEnvelope ToEnvelope() => new()
    {
        Code = NumericCode,
        Name = StableCode,
        Message = Message
    };
}

public sealed class ErrorEnvelope
{
    [JsonPropertyName("code")]
    public int Code { get; init; }

    [JsonPropertyName("name")]
    public string Name { get; init; } = string.Empty;

    [JsonPropertyName("message")]
    public string Message { get; init; } = string.Empty;
}
