using System.Text.Json;

namespace AudioPipeline;

public sealed class ManifestStore
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        WriteIndented = true
    };

    public ManifestStore(string outputPath)
    {
        OutputPath = Path.GetFullPath(outputPath);
        StateDirectory = OutputPath + ".state";
        BlockDirectory = Path.Combine(StateDirectory, "blocks");
        ManifestPath = Path.Combine(StateDirectory, "manifest.json");
    }

    public string OutputPath { get; }

    public string StateDirectory { get; }

    public string BlockDirectory { get; }

    public string ManifestPath { get; }

    public void EnsureDirectories()
    {
        Directory.CreateDirectory(StateDirectory);
        Directory.CreateDirectory(BlockDirectory);
    }

    public void CleanTemporaryFiles()
    {
        if (Directory.Exists(BlockDirectory))
        {
            foreach (var path in Directory.EnumerateFiles(BlockDirectory, "*.tmp-*"))
            {
                TryDelete(path);
            }
        }

        if (Directory.Exists(StateDirectory))
        {
            foreach (var path in Directory.EnumerateFiles(StateDirectory, "manifest.json.tmp-*"))
            {
                TryDelete(path);
            }
        }
    }

    public BlockManifest? TryLoadManifest()
    {
        if (!File.Exists(ManifestPath))
        {
            return null;
        }

        try
        {
            var json = File.ReadAllText(ManifestPath);
            return JsonSerializer.Deserialize<BlockManifest>(json, JsonOptions);
        }
        catch (Exception ex) when (
            ex is IOException or UnauthorizedAccessException or JsonException)
        {
            return null;
        }
    }

    public void SaveManifestAtomic(BlockManifest manifest)
    {
        var tempPath = ManifestPath + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            var json = JsonSerializer.Serialize(manifest, JsonOptions);
            using (var stream = new FileStream(
                       tempPath,
                       FileMode.CreateNew,
                       FileAccess.Write,
                       FileShare.None,
                       bufferSize: 16 * 1024,
                       FileOptions.WriteThrough))
            using (var writer = new StreamWriter(stream))
            {
                writer.Write(json);
                writer.Flush();
                stream.Flush(true);
            }

            File.Move(tempPath, ManifestPath, overwrite: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            TryDelete(tempPath);
            throw new PipelineException(
                PipelineErrorCode.RenderIo,
                $"Unable to persist render manifest: {ex.Message}",
                ex);
        }
    }

    public void WriteBlockAtomic(int blockIndex, ReadOnlySpan<byte> pcmBytes)
    {
        var finalPath = GetBlockPath(blockIndex);
        var tempPath = finalPath + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            using (var stream = new FileStream(
                       tempPath,
                       FileMode.CreateNew,
                       FileAccess.Write,
                       FileShare.None,
                       bufferSize: 64 * 1024,
                       FileOptions.WriteThrough))
            {
                stream.Write(pcmBytes);
                stream.Flush(true);
            }

            File.Move(tempPath, finalPath, overwrite: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            TryDelete(tempPath);
            throw new PipelineException(
                PipelineErrorCode.RenderIo,
                $"Unable to persist block {blockIndex}: {ex.Message}",
                ex);
        }
    }

    public string GetBlockPath(int blockIndex) =>
        Path.Combine(BlockDirectory, $"block-{blockIndex:D6}.pcm");

    public void ClearBlocks()
    {
        if (!Directory.Exists(BlockDirectory))
        {
            return;
        }

        foreach (var path in Directory.EnumerateFiles(BlockDirectory, "block-*.pcm"))
        {
            TryDelete(path);
        }

        foreach (var path in Directory.EnumerateFiles(BlockDirectory, "*.tmp-*"))
        {
            TryDelete(path);
        }
    }

    public static bool MatchesScene(BlockManifest manifest, ValidatedScene scene, int totalBlocks)
    {
        return manifest.ManifestVersion == AudioConstants.ManifestVersion &&
               string.Equals(
                   manifest.AlgorithmVersion,
                   AudioConstants.AlgorithmVersion,
                   StringComparison.Ordinal) &&
               manifest.SampleRate == AudioConstants.SampleRate &&
               manifest.Channels == AudioConstants.Channels &&
               manifest.BitsPerSample == AudioConstants.BitsPerSample &&
               manifest.BlockFrames == AudioConstants.BlockFrames &&
               manifest.TotalFrames == scene.TotalFrames &&
               manifest.TotalBlocks == totalBlocks &&
               string.Equals(
                   manifest.EventSummaryAlgorithm,
                   AudioConstants.EventSummaryAlgorithm,
                   StringComparison.Ordinal) &&
               string.Equals(
                   manifest.EventSummaryHash,
                   scene.EventSummaryHash,
                   StringComparison.OrdinalIgnoreCase) &&
               manifest.Blocks is not null &&
               manifest.Blocks.Count == totalBlocks;
    }

    public bool ValidateBlock(BlockRecord record, int expectedIndex, BlockManifest manifest)
    {
        if (record is null ||
            record.Index != expectedIndex ||
            !string.Equals(record.Status, BlockStatus.Complete, StringComparison.Ordinal) ||
            record.FrameCount != AudioConstants.BlockFrames ||
            record.ByteLength != AudioConstants.BlockBytes ||
            record.Sha256 is null ||
            record.Sha256.Length != 64 ||
            record.ExitStates is null)
        {
            return false;
        }

        var path = GetBlockPath(expectedIndex);
        if (!File.Exists(path))
        {
            return false;
        }

        try
        {
            var info = new FileInfo(path);
            if (info.Length != record.ByteLength)
            {
                return false;
            }

            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            var actualHash = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream));
            return string.Equals(actualHash, record.Sha256, StringComparison.OrdinalIgnoreCase);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    public static BlockRecord CreatePendingBlock(int index) => new()
    {
        Index = index,
        Status = BlockStatus.Pending,
        FrameCount = AudioConstants.BlockFrames,
        ByteLength = AudioConstants.BlockBytes,
        Sha256 = string.Empty,
        ExitStates = []
    };

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }
}
