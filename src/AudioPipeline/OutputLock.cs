using System.Text;

namespace AudioPipeline;

public sealed class OutputLock : IDisposable
{
    private readonly FileStream _stream;

    private OutputLock(FileStream stream, string path)
    {
        _stream = stream;
        Path = path;
    }

    public string Path { get; }

    public static OutputLock Acquire(string outputDirectory)
    {
        var directory = global::System.IO.Path.GetFullPath(outputDirectory);
        try
        {
            Directory.CreateDirectory(directory);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new PipelineException(
                PipelineErrorCode.RenderIo,
                $"Unable to create output directory '{directory}': {ex.Message}",
                ex);
        }

        var lockPath = global::System.IO.Path.Combine(directory, AudioConstants.LockFileName);
        try
        {
            var stream = new FileStream(
                lockPath,
                FileMode.OpenOrCreate,
                FileAccess.ReadWrite,
                FileShare.None,
                bufferSize: 128,
                FileOptions.WriteThrough);
            stream.SetLength(0);
            var metadata = Encoding.UTF8.GetBytes(
                $"pid={Environment.ProcessId}\nutc={DateTimeOffset.UtcNow:O}\n");
            stream.Write(metadata);
            stream.Flush(true);
            return new OutputLock(stream, lockPath);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new PipelineException(
                PipelineErrorCode.OutputLocked,
                $"Output directory is already owned by another writer: '{directory}'.",
                ex);
        }
    }

    public void Dispose() => _stream.Dispose();
}
