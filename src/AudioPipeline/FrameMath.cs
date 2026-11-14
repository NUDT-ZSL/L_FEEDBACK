namespace AudioPipeline;

public static class FrameMath
{
    public static long StartFrame(double delaySeconds) =>
        checked((long)Math.Round(
            delaySeconds * AudioConstants.SampleRate,
            MidpointRounding.AwayFromZero));

    public static long DurationFrameCount(double durationSeconds)
    {
        var exactFrames = durationSeconds * AudioConstants.SampleRate;
        var normalized = Math.Round(exactFrames, 9, MidpointRounding.AwayFromZero);
        return checked((long)Math.Ceiling(normalized));
    }

    public static long EndFrame(double delaySeconds, double durationSeconds) =>
        checked(StartFrame(delaySeconds) + DurationFrameCount(durationSeconds));
}
