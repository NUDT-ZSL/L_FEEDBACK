namespace AudioPipeline;

public sealed class SpatialFrame
{
    public required Vec3 SourcePosition { get; init; }

    public required Vec3 SourceVelocity { get; init; }

    public required double Distance { get; init; }

    public required double DistanceGain { get; init; }

    public required double NormalizedPan { get; init; }

    public required double LeftGain { get; init; }

    public required double RightGain { get; init; }

    public required double DopplerFactor { get; init; }

    public required bool Occluded { get; init; }
}

public static class SpatialDsp
{
    public static SpatialFrame EvaluateFrame(
        EventDefinition definition,
        Vec3 listener,
        long localFrame,
        long durationFrames,
        IReadOnlyList<Aabb> occluders)
    {
        var amount = durationFrames <= 1
            ? 0.0
            : Math.Clamp(localFrame / (double)(durationFrames - 1), 0.0, 1.0);
        var position = Geometry.Lerp(definition.Start, definition.End, amount);
        var playbackSeconds = durationFrames / (double)AudioConstants.SampleRate;
        var velocity = playbackSeconds <= 0
            ? default
            : (definition.End - definition.Start) * (1.0 / playbackSeconds);

        var toListener = listener - position;
        var distance = toListener.Length;
        var distanceGain = LinearDistanceGain(distance, definition.MinDistance, definition.MaxDistance);

        var normalizedPan = 0.0;
        if (Math.Abs(toListener.X) > 1e-12 || Math.Abs(toListener.Z) > 1e-12)
        {
            var azimuth = Math.Atan2(toListener.X, toListener.Z);
            normalizedPan = Math.Clamp(azimuth / (Math.PI / 2.0), -1.0, 1.0);
        }

        var leftGain = (1.0 - normalizedPan) * 0.5;
        var rightGain = (1.0 + normalizedPan) * 0.5;

        var dopplerFactor = 1.0;
        if (definition.Doppler && distance > 1e-12)
        {
            var radialSpeed = Vec3.Dot(velocity, toListener * (1.0 / distance));
            var rawFactor = AudioConstants.SpeedOfSoundMetersPerSecond /
                            (AudioConstants.SpeedOfSoundMetersPerSecond - radialSpeed);
            dopplerFactor = double.IsFinite(rawFactor) ? Math.Clamp(rawFactor, 0.5, 2.0) : 1.0;
        }

        var occluded = false;
        foreach (var box in occluders)
        {
            if (Geometry.SegmentIntersectsAabb(listener, position, box))
            {
                occluded = true;
                break;
            }
        }

        return new SpatialFrame
        {
            SourcePosition = position,
            SourceVelocity = velocity,
            Distance = distance,
            DistanceGain = distanceGain,
            NormalizedPan = normalizedPan,
            LeftGain = leftGain,
            RightGain = rightGain,
            DopplerFactor = dopplerFactor,
            Occluded = occluded
        };
    }

    public static double LinearDistanceGain(double distance, double minDistance, double maxDistance)
    {
        if (distance <= minDistance)
        {
            return 1.0;
        }

        if (distance >= maxDistance)
        {
            return 0.0;
        }

        return (maxDistance - distance) / (maxDistance - minDistance);
    }
}

public static class AudioDsp
{
    public static double ProcessLowPass(double input, ref double state) =>
        (AudioConstants.LowPassCoefficient * input) +
        ((1.0 - AudioConstants.LowPassCoefficient) * state);

    public static double ReadInterpolated(short[] samples, double readPosition)
    {
        if (readPosition < 0 || samples.Length == 0)
        {
            return 0.0;
        }

        var low = (int)Math.Floor(readPosition);
        var fraction = readPosition - low;
        if (low >= samples.Length)
        {
            return 0.0;
        }

        var first = samples[low] / 32768.0;
        var second = low + 1 < samples.Length ? samples[low + 1] / 32768.0 : 0.0;
        return first + ((second - first) * fraction);
    }

    public static double ReadIntegral(short[] samples, long sampleIndex)
    {
        if (sampleIndex < 0 || sampleIndex >= samples.Length)
        {
            return 0.0;
        }

        return samples[sampleIndex] / 32768.0;
    }
}

public sealed class FixedPointMixBuffer
{
    public const int FixedScale = 65536;
    private const int Scale = FixedScale;
    private readonly int[] _accumulator;

    public FixedPointMixBuffer(int frameCount)
    {
        if (frameCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(frameCount));
        }

        FrameCount = frameCount;
        _accumulator = new int[checked(frameCount * AudioConstants.Channels)];
    }

    public int FrameCount { get; }

    public int ByteLength => checked(_accumulator.Length * sizeof(int));

    public void AddStereo(int frame, double left, double right)
    {
        if ((uint)frame >= (uint)FrameCount)
        {
            throw new ArgumentOutOfRangeException(nameof(frame));
        }

        var offset = frame * AudioConstants.Channels;
        _accumulator[offset] = SaturatingAdd(_accumulator[offset], ToFixed(left));
        _accumulator[offset + 1] = SaturatingAdd(_accumulator[offset + 1], ToFixed(right));
    }

    public void ReadStereo(int frame, out short left, out short right)
    {
        if ((uint)frame >= (uint)FrameCount)
        {
            throw new ArgumentOutOfRangeException(nameof(frame));
        }

        var offset = frame * AudioConstants.Channels;
        left = ToPcm16(_accumulator[offset]);
        right = ToPcm16(_accumulator[offset + 1]);
    }

    public int GetFixedLeft(int frame) => _accumulator[frame * AudioConstants.Channels];

    public int GetFixedRight(int frame) => _accumulator[(frame * AudioConstants.Channels) + 1];

    public static int ToFixed(double normalizedSample)
    {
        if (!double.IsFinite(normalizedSample))
        {
            throw new ArgumentOutOfRangeException(
                nameof(normalizedSample),
                "Mix sample must be finite.");
        }

        var scaled = normalizedSample * Scale;
        if (scaled >= int.MaxValue)
        {
            return int.MaxValue;
        }

        if (scaled <= int.MinValue)
        {
            return int.MinValue;
        }

        return (int)Math.Round(scaled, MidpointRounding.AwayFromZero);
    }

    public static int SaturatingAdd(int left, int right)
    {
        var sum = (long)left + right;
        return sum switch
        {
            > int.MaxValue => int.MaxValue,
            < int.MinValue => int.MinValue,
            _ => (int)sum
        };
    }

    public static short ToPcm16(int fixedSample)
    {
        var normalized = fixedSample / (double)Scale;
        var value = Math.Round(normalized * 32768.0, MidpointRounding.AwayFromZero);
        return value switch
        {
            > short.MaxValue => short.MaxValue,
            < short.MinValue => short.MinValue,
            _ => (short)value
        };
    }
}
