using System.Buffers.Binary;
using System.Text.Json;

namespace AudioPipeline;

public sealed class FixtureResult
{
    public required string Directory { get; init; }

    public required string ScenePath { get; init; }

    public required int Seed { get; init; }

    public required long ExpectedFrames { get; init; }
}

public static class FixtureGenerator
{
    public const int DefaultSeed = 1_592_598_566;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true
    };

    public static FixtureResult Generate(string outputDirectory, int seed = DefaultSeed)
    {
        var root = Path.GetFullPath(outputDirectory);
        var sourceDirectory = Path.Combine(root, "sources");
        Directory.CreateDirectory(sourceDirectory);

        var sourceA = Path.Combine(sourceDirectory, "ambient_a.wav");
        var sourceB = Path.Combine(sourceDirectory, "ambient_b.wav");
        GenerateAmbient(sourceA, seed, 2.4, 173.0, 1.0);
        GenerateAmbient(sourceB, seed ^ 0x5F3759DF, 2.0, 241.0, 1.7);

        var scene = CreateScene(seed);
        var scenePath = Path.Combine(root, "scene.json");
        File.WriteAllText(scenePath, JsonSerializer.Serialize(scene, JsonOptions));

        var totalFrames = scene.Events!
            .Select(item => FrameMath.EndFrame(item.DelaySeconds, item.DurationSeconds))
            .Max();

        return new FixtureResult
        {
            Directory = root,
            ScenePath = scenePath,
            Seed = seed,
            ExpectedFrames = totalFrames
        };
    }

    public static SceneDefinition CreateScene(int seed = DefaultSeed) => new()
    {
        Version = 1,
        Name = "deterministic-spatial-fixture",
        Seed = seed,
        Listener = new Vec3(0, 1.6, 0),
        Events =
        [
            new EventDefinition
            {
                Id = "doppler-pass",
                Start = new Vec3(-10, 1, 8),
                End = new Vec3(10, 1, 8),
                DurationSeconds = 2.2,
                DelaySeconds = 0,
                Gain = 0.65,
                SourcePath = "sources/ambient_a.wav",
                MinDistance = 1,
                MaxDistance = 24,
                Doppler = true,
                Occluders =
                [
                    new Aabb(new Vec3(-1.2, -2, 5), new Vec3(1.2, 2, 6))
                ]
            },
            new EventDefinition
            {
                Id = "occluded-bed",
                Start = new Vec3(0, 0.4, 9),
                End = new Vec3(0, 0.4, 9),
                DurationSeconds = 1.35,
                DelaySeconds = 0.1,
                Gain = 0.45,
                SourcePath = "sources/ambient_b.wav",
                MinDistance = 1,
                MaxDistance = 30,
                Doppler = false,
                Occluders =
                [
                    new Aabb(new Vec3(-1, -1, 5), new Vec3(1, 1, 6))
                ]
            },
            new EventDefinition
            {
                Id = "front-pulse",
                Start = new Vec3(1.5, 1.5, 2),
                End = new Vec3(-1.5, 1.5, 2),
                DurationSeconds = 0.7,
                DelaySeconds = 1.05,
                Gain = 0.5,
                SourcePath = "sources/ambient_b.wav",
                MinDistance = 0.5,
                MaxDistance = 12,
                Doppler = false,
                Occluders = []
            }
        ]
    };

    private static void GenerateAmbient(
        string path,
        int seed,
        double seconds,
        double baseFrequency,
        double modulationRate)
    {
        var frameCount = checked((int)Math.Round(seconds * AudioConstants.SampleRate));
        var rng = new DeterministicRng(unchecked((ulong)(uint)seed));
        var lowNoise = 0.0;
        var dataBytes = checked(frameCount * AudioConstants.BytesPerSample);

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        WavIO.WriteMonoPcm16Header(stream, frameCount);
        var buffer = new byte[dataBytes];
        for (var i = 0; i < frameCount; i++)
        {
            var t = i / (double)AudioConstants.SampleRate;
            lowNoise = (lowNoise * 0.92) + (rng.NextSigned() * 0.08);
            var sweep = baseFrequency + (28.0 * Math.Sin(2 * Math.PI * modulationRate * t));
            var tonal =
                (0.42 * Math.Sin(2 * Math.PI * sweep * t)) +
                (0.18 * Math.Sin(2 * Math.PI * (sweep * 2.01) * t + 0.7)) +
                (0.11 * Math.Sin(2 * Math.PI * 31.0 * t));
            var envelope = Math.Min(1.0, t / 0.015) * Math.Min(1.0, (seconds - t) / 0.03);
            var sample = Math.Clamp((tonal + (lowNoise * 2.2)) * envelope, -1.0, 1.0);
            BinaryPrimitives.WriteInt16LittleEndian(
                buffer.AsSpan(i * AudioConstants.BytesPerSample, AudioConstants.BytesPerSample),
                (short)Math.Round(sample * short.MaxValue, MidpointRounding.AwayFromZero));
        }

        stream.Write(buffer);
        stream.Flush(true);
    }
}
