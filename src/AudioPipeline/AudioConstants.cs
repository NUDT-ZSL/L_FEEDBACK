namespace AudioPipeline;

public static class AudioConstants
{
    public const int SampleRate = 48_000;
    public const int Channels = 2;
    public const int BitsPerSample = 16;
    public const int BytesPerSample = BitsPerSample / 8;
    public const int BlockFrames = 48_000;
    public const int BlockBytes = BlockFrames * Channels * BytesPerSample;
    public const double SpeedOfSoundMetersPerSecond = 343.0;
    public const double LowPassCoefficient = 0.6;
    public const int MaxEvents = 128;
    public const double MaxTimelineSeconds = 300.0;
    public const int DefaultMaxBlockBufferBytes = 1_048_576;
    public const string AlgorithmVersion = "spatial-master-v1";
    public const int ManifestVersion = 1;
    public const string LockFileName = ".audio-pipeline.lock";
    public const string EventSummaryAlgorithm = "SHA256-canonical-scene-v1";

    public static long MaxFrames => (long)Math.Ceiling(MaxTimelineSeconds * SampleRate);
}
