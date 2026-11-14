namespace AudioPipeline;

public static class WavAssembler
{
    public static string Assemble(
        ManifestStore store,
        string outputPath,
        long totalFrames,
        int totalBlocks)
    {
        var fullOutputPath = Path.GetFullPath(outputPath);
        var outputDirectory = Path.GetDirectoryName(fullOutputPath) ??
                              throw new PipelineException(
                                  PipelineErrorCode.RenderIo,
                                  "Output path must have a parent directory.");
        Directory.CreateDirectory(outputDirectory);
        var tempPath = fullOutputPath + ".tmp-" + Guid.NewGuid().ToString("N");

        try
        {
            using (var output = new FileStream(
                       tempPath,
                       FileMode.CreateNew,
                       FileAccess.Write,
                       FileShare.None,
                       bufferSize: 128 * 1024,
                       FileOptions.WriteThrough))
            {
                WavIO.WriteHeader(output, totalFrames);
                var remainingFrames = totalFrames;
                var blockBytes = new byte[AudioConstants.BlockBytes];
                for (var blockIndex = 0; blockIndex < totalBlocks; blockIndex++)
                {
                    var blockPath = store.GetBlockPath(blockIndex);
                    using var block = new FileStream(
                        blockPath,
                        FileMode.Open,
                        FileAccess.Read,
                        FileShare.Read,
                        bufferSize: 64 * 1024);
                    if (block.Length != AudioConstants.BlockBytes)
                    {
                        throw new PipelineException(
                            PipelineErrorCode.RenderIo,
                            $"Block {blockIndex} has length {block.Length}; expected {AudioConstants.BlockBytes}.");
                    }

                    ReadExactly(block, blockBytes);
                    var framesToCopy = (int)Math.Min(AudioConstants.BlockFrames, remainingFrames);
                    var bytesToCopy = checked(framesToCopy * AudioConstants.Channels * AudioConstants.BytesPerSample);
                    output.Write(blockBytes.AsSpan(0, bytesToCopy));
                    remainingFrames -= framesToCopy;
                }

                if (remainingFrames != 0)
                {
                    throw new PipelineException(
                        PipelineErrorCode.RenderIo,
                        $"Block set is too short by {remainingFrames} frames.");
                }

                output.Flush(true);
            }

            File.Move(tempPath, fullOutputPath, overwrite: true);
            return WavIO.Sha256Hex(fullOutputPath);
        }
        catch (PipelineException)
        {
            TryDelete(tempPath);
            throw;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            TryDelete(tempPath);
            throw new PipelineException(
                PipelineErrorCode.RenderIo,
                $"Unable to assemble master WAV: {ex.Message}",
                ex);
        }
    }

    private static void ReadExactly(Stream stream, Span<byte> destination)
    {
        var offset = 0;
        while (offset < destination.Length)
        {
            var read = stream.Read(destination[offset..]);
            if (read == 0)
            {
                throw new EndOfStreamException("Unexpected end of block file.");
            }

            offset += read;
        }
    }

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
