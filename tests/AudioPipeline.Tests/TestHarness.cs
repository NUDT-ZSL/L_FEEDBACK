using System.Diagnostics;
using System.Text.Json;
using AudioPipeline;

namespace AudioPipeline.Tests;

internal static class TestHarness
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true
    };

    public static readonly string RepositoryRoot = FindRepositoryRoot();

    public static readonly string RunRoot = Path.Combine(
        RepositoryRoot,
        "artifacts",
        "test-runs",
        $"{DateTime.Now:yyyyMMdd-HHmmss}-{Guid.NewGuid().ToString("N")[..8]}");

    private static int _passed;
    private static int _failed;

    public static int Run(string name, Action test)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            test();
            stopwatch.Stop();
            _passed++;
            Console.WriteLine($"PASS {name} ({stopwatch.ElapsedMilliseconds} ms)");
            return 0;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            _failed++;
            Console.WriteLine($"FAIL {name} ({stopwatch.ElapsedMilliseconds} ms)");
            Console.WriteLine(ex);
            return 1;
        }
    }

    public static int Finish()
    {
        Console.WriteLine();
        Console.WriteLine($"{_passed} passed, {_failed} failed");
        Console.WriteLine($"Run artifacts: {RunRoot}");
        return _failed == 0 ? 0 : 1;
    }

    public static string NewDirectory(string name)
    {
        var path = Path.Combine(RunRoot, name);
        Directory.CreateDirectory(path);
        return path;
    }

    public static void Assert(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    public static void AssertEqual<T>(T expected, T actual, string message)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
        {
            throw new InvalidOperationException($"{message}: expected {expected}, actual {actual}");
        }
    }

    public static PipelineErrorCode ExpectCode(Action action)
    {
        try
        {
            action();
        }
        catch (PipelineException ex)
        {
            return ex.Code;
        }

        throw new InvalidOperationException("Expected a PipelineException, but no exception was thrown.");
    }

    public static FixtureResult GenerateFixture(string name, int seed = FixtureGenerator.DefaultSeed)
    {
        var directory = NewDirectory(name);
        return FixtureGenerator.Generate(directory, seed);
    }

    public static RenderResult Render(
        string scenePath,
        string outputPath,
        int maxBlocks = int.MaxValue)
    {
        return OfflineRenderer.Render(
            scenePath,
            outputPath,
            new RenderOptions { MaxBlocksThisRun = maxBlocks });
    }

    public static SceneDefinition ReadScene(string scenePath)
    {
        var json = File.ReadAllText(scenePath);
        return JsonSerializer.Deserialize<SceneDefinition>(json, JsonOptions) ??
               throw new InvalidOperationException("Unable to deserialize test scene.");
    }

    public static string WriteScene(
        string directory,
        SceneDefinition scene,
        string fileName = "scene.json")
    {
        var path = Path.Combine(directory, fileName);
        File.WriteAllText(path, JsonSerializer.Serialize(scene, JsonOptions));
        return path;
    }

    public static BlockManifest ReadManifest(string outputPath)
    {
        var store = new ManifestStore(outputPath);
        return store.TryLoadManifest() ??
               throw new InvalidOperationException($"Manifest missing at {store.ManifestPath}");
    }

    public static string MasterHash(string path) => WavIO.Sha256Hex(path);

    public static byte[] ReadWavData(string path)
    {
        var info = WavIO.ReadInfo(path);
        var bytes = new byte[checked((int)info.DataByteLength)];
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        stream.Position = info.DataOffset;
        var read = stream.Read(bytes);
        Assert(read == bytes.Length, "WAV data chunk was not fully read.");
        return bytes;
    }

    public static void WriteMonoPcm16(string path, short[] samples)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        WavIO.WriteMonoPcm16Header(stream, samples.Length);
        var bytes = new byte[samples.Length * 2];
        for (var index = 0; index < samples.Length; index++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteInt16LittleEndian(
                bytes.AsSpan(index * 2, 2),
                samples[index]);
        }

        stream.Write(bytes);
    }

    public static string FindCliDll()
    {
        var localPath = Path.Combine(AppContext.BaseDirectory, "AudioPipeline.Cli.dll");
        if (File.Exists(localPath))
        {
            return localPath;
        }

        foreach (var configuration in new[] { "Release", "Debug" })
        {
            var path = Path.Combine(
                RepositoryRoot,
                "src",
                "AudioPipeline.Cli",
                "bin",
                configuration,
                "net8.0",
                "AudioPipeline.Cli.dll");
            if (File.Exists(path))
            {
                return path;
            }
        }

        throw new InvalidOperationException("The CLI assembly is not built.");
    }

    public static string FindDotnetHost()
    {
        var processPath = Environment.ProcessPath;
        if (processPath is not null &&
            (Path.GetFileName(processPath).Equals("dotnet", StringComparison.OrdinalIgnoreCase) ||
             Path.GetFileName(processPath).Equals("dotnet.exe", StringComparison.OrdinalIgnoreCase)))
        {
            return processPath;
        }

        var configuredHost = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH");
        if (!string.IsNullOrWhiteSpace(configuredHost) && File.Exists(configuredHost))
        {
            return configuredHost;
        }

        var runtimeDirectory = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();
        var dotnetRoot = Path.GetFullPath(
            Path.Combine(runtimeDirectory, "..", "..", "..", ".."));
        var hostName = OperatingSystem.IsWindows() ? "dotnet.exe" : "dotnet";
        var hostPath = Path.Combine(dotnetRoot, hostName);
        return File.Exists(hostPath) ? hostPath : "dotnet";
    }

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory.Parent is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "SpatialAudioPipeline.sln")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new InvalidOperationException("Unable to locate the repository root.");
    }
}
