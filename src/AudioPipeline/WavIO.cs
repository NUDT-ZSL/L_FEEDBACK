using System.Buffers.Binary;
using System.Security.Cryptography;

namespace AudioPipeline;

public sealed class Pcm16MonoSource
{
    public required string Path { get; init; }

    public required int SampleRate { get; init; }

    public required short[] Samples { get; init; }

    public int FrameCount => Samples.Length;
}

public sealed class WavInfo
{
    public required int FormatTag { get; init; }

    public required int Channels { get; init; }

    public required int SampleRate { get; init; }

    public required int BitsPerSample { get; init; }

    public required int BlockAlign { get; init; }

    public required long DataOffset { get; init; }

    public required long DataByteLength { get; init; }

    public required long FileLength { get; init; }

    public long FrameCount => BlockAlign == 0 ? 0 : DataByteLength / BlockAlign;
}

public static class WavIO
{
    private const ushort PcmFormatTag = 1;

    public static void WriteMonoPcm16Header(Stream output, long frameCount)
    {
        if (!output.CanWrite || !output.CanSeek)
        {
            throw new ArgumentException("WAV output stream must be writable and seekable.", nameof(output));
        }

        var dataBytes = checked(frameCount * AudioConstants.BytesPerSample);
        var riffSize = checked(36L + dataBytes);
        if (riffSize > uint.MaxValue)
        {
            throw new PipelineException(
                PipelineErrorCode.DurationLimit,
                "The source WAV exceeds the RIFF 4 GiB limit.");
        }

        Span<byte> header = stackalloc byte[44];
        "RIFF"u8.CopyTo(header[0..4]);
        BinaryPrimitives.WriteUInt32LittleEndian(header[4..8], (uint)riffSize);
        "WAVE"u8.CopyTo(header[8..12]);
        "fmt "u8.CopyTo(header[12..16]);
        BinaryPrimitives.WriteUInt32LittleEndian(header[16..20], 16);
        BinaryPrimitives.WriteUInt16LittleEndian(header[20..22], PcmFormatTag);
        BinaryPrimitives.WriteUInt16LittleEndian(header[22..24], 1);
        BinaryPrimitives.WriteInt32LittleEndian(header[24..28], AudioConstants.SampleRate);
        BinaryPrimitives.WriteInt32LittleEndian(
            header[28..32],
            AudioConstants.SampleRate * AudioConstants.BytesPerSample);
        BinaryPrimitives.WriteUInt16LittleEndian(header[32..34], AudioConstants.BytesPerSample);
        BinaryPrimitives.WriteUInt16LittleEndian(header[34..36], AudioConstants.BitsPerSample);
        "data"u8.CopyTo(header[36..40]);
        BinaryPrimitives.WriteUInt32LittleEndian(header[40..44], (uint)dataBytes);
        output.Write(header);
    }

    public static WavInfo ReadInfo(string path)
    {
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            return ReadInfo(stream);
        }
        catch (PipelineException)
        {
            throw;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new PipelineException(
                PipelineErrorCode.SourceFormat,
                $"Unable to read WAV '{path}': {ex.Message}",
                ex);
        }
    }

    public static WavInfo ReadInfo(Stream stream)
    {
        try
        {
            return ReadInfoCore(stream);
        }
        catch (PipelineException)
        {
            throw;
        }
        catch (EndOfStreamException ex)
        {
            throw new PipelineException(
                PipelineErrorCode.SourceFormat,
                $"WAV is truncated: {ex.Message}",
                ex);
        }
    }

    private static WavInfo ReadInfoCore(Stream stream)
    {
        if (!stream.CanRead || !stream.CanSeek)
        {
            throw InvalidWav("WAV stream must be readable and seekable.");
        }

        var fileLength = stream.Length;
        if (fileLength < 12)
        {
            throw InvalidWav("File is too short to contain a RIFF/WAVE header.");
        }

        Span<byte> header = stackalloc byte[12];
        ReadExactly(stream, header);
        if (!header[..4].SequenceEqual("RIFF"u8) || !header[8..12].SequenceEqual("WAVE"u8))
        {
            throw InvalidWav("Expected RIFF/WAVE file.");
        }

        var riffSize = BinaryPrimitives.ReadUInt32LittleEndian(header[4..8]);
        var riffEnd = Math.Min(fileLength, 8L + riffSize);
        if (riffEnd < 12)
        {
            throw InvalidWav("RIFF size is invalid.");
        }

        var position = 12L;
        WavInfo? fmt = null;
        WavInfo? data = null;
        Span<byte> chunkHeader = stackalloc byte[8];
        Span<byte> fmtBytes = stackalloc byte[16];

        while (position + 8 <= riffEnd)
        {
            stream.Position = position;
            ReadExactly(stream, chunkHeader);
            var chunkSize = BinaryPrimitives.ReadUInt32LittleEndian(chunkHeader[4..8]);
            var payloadOffset = position + 8;
            var payloadEnd = payloadOffset + chunkSize;
            if (payloadEnd < payloadOffset || payloadEnd > riffEnd)
            {
                throw InvalidWav("A WAV chunk exceeds the RIFF boundary.");
            }

            if (chunkHeader[..4].SequenceEqual("fmt "u8))
            {
                if (chunkSize < 16)
                {
                    throw InvalidWav("The fmt chunk is shorter than 16 bytes.");
                }

                stream.Position = payloadOffset;
                ReadExactly(stream, fmtBytes);
                var formatTag = BinaryPrimitives.ReadUInt16LittleEndian(fmtBytes[0..2]);
                var channels = BinaryPrimitives.ReadUInt16LittleEndian(fmtBytes[2..4]);
                var sampleRate = BinaryPrimitives.ReadInt32LittleEndian(fmtBytes[4..8]);
                var blockAlign = BinaryPrimitives.ReadUInt16LittleEndian(fmtBytes[12..14]);
                var bitsPerSample = BinaryPrimitives.ReadUInt16LittleEndian(fmtBytes[14..16]);
                fmt = new WavInfo
                {
                    FormatTag = formatTag,
                    Channels = channels,
                    SampleRate = sampleRate,
                    BitsPerSample = bitsPerSample,
                    BlockAlign = blockAlign,
                    DataOffset = 0,
                    DataByteLength = 0,
                    FileLength = fileLength
                };
            }
            else if (chunkHeader[..4].SequenceEqual("data"u8))
            {
                data = new WavInfo
                {
                    FormatTag = 0,
                    Channels = 0,
                    SampleRate = 0,
                    BitsPerSample = 0,
                    BlockAlign = 0,
                    DataOffset = payloadOffset,
                    DataByteLength = chunkSize,
                    FileLength = fileLength
                };
            }

            position = payloadEnd + (chunkSize & 1u);
        }

        if (fmt is null)
        {
            throw InvalidWav("The fmt chunk is missing.");
        }

        if (data is null)
        {
            throw InvalidWav("The data chunk is missing.");
        }

        return new WavInfo
        {
            FormatTag = fmt.FormatTag,
            Channels = fmt.Channels,
            SampleRate = fmt.SampleRate,
            BitsPerSample = fmt.BitsPerSample,
            BlockAlign = fmt.BlockAlign,
            DataOffset = data.DataOffset,
            DataByteLength = data.DataByteLength,
            FileLength = fileLength
        };
    }

    public static Pcm16MonoSource ReadMonoPcm16(string path, long maxFrames)
    {
        WavInfo info;
        try
        {
            info = ReadInfo(path);
        }
        catch (PipelineException)
        {
            throw;
        }
        catch (EndOfStreamException ex)
        {
            throw new PipelineException(
                PipelineErrorCode.SourceFormat,
                $"Source WAV '{path}' is truncated: {ex.Message}",
                ex);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new PipelineException(
                PipelineErrorCode.SourceFormat,
                $"Unable to read WAV source '{path}': {ex.Message}",
                ex);
        }

        if (info.FormatTag != PcmFormatTag ||
            info.Channels != 1 ||
            info.SampleRate != AudioConstants.SampleRate ||
            info.BitsPerSample != 16 ||
            info.BlockAlign != AudioConstants.BytesPerSample)
        {
            throw new PipelineException(
                PipelineErrorCode.SourceFormat,
                $"Source '{path}' must be mono 48 kHz 16-bit PCM WAV " +
                $"(format={info.FormatTag}, channels={info.Channels}, rate={info.SampleRate}, bits={info.BitsPerSample}).");
        }

        if (info.DataByteLength % AudioConstants.BytesPerSample != 0)
        {
            throw new PipelineException(
                PipelineErrorCode.SourceFormat,
                $"Source '{path}' has a data chunk that is not aligned to 16-bit samples.");
        }

        var frameCount = checked((int)(info.DataByteLength / AudioConstants.BytesPerSample));
        if (frameCount > maxFrames)
        {
            throw new PipelineException(
                PipelineErrorCode.DurationLimit,
                $"Source '{path}' exceeds the {AudioConstants.MaxTimelineSeconds:0} second limit.");
        }

        var bytes = new byte[frameCount * AudioConstants.BytesPerSample];
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            stream.Position = info.DataOffset;
            ReadExactly(stream, bytes);
        }
        catch (Exception ex) when (ex is IOException or EndOfStreamException)
        {
            throw new PipelineException(
                PipelineErrorCode.SourceFormat,
                $"Unable to read source WAV data '{path}': {ex.Message}",
                ex);
        }

        var samples = new short[frameCount];
        for (var i = 0; i < frameCount; i++)
        {
            samples[i] = BinaryPrimitives.ReadInt16LittleEndian(
                bytes.AsSpan(i * AudioConstants.BytesPerSample, AudioConstants.BytesPerSample));
        }

        return new Pcm16MonoSource
        {
            Path = path,
            SampleRate = info.SampleRate,
            Samples = samples
        };
    }

    public static void WriteHeader(Stream output, long frameCount)
    {
        if (!output.CanWrite || !output.CanSeek)
        {
            throw new ArgumentException("WAV output stream must be writable and seekable.", nameof(output));
        }

        var dataBytes = checked(frameCount * AudioConstants.Channels * AudioConstants.BytesPerSample);
        var riffSize = checked(36L + dataBytes);
        if (riffSize > uint.MaxValue)
        {
            throw new PipelineException(
                PipelineErrorCode.DurationLimit,
                "The output WAV exceeds the RIFF 4 GiB limit.");
        }

        Span<byte> header = stackalloc byte[44];
        "RIFF"u8.CopyTo(header[0..4]);
        BinaryPrimitives.WriteUInt32LittleEndian(header[4..8], (uint)riffSize);
        "WAVE"u8.CopyTo(header[8..12]);
        "fmt "u8.CopyTo(header[12..16]);
        BinaryPrimitives.WriteUInt32LittleEndian(header[16..20], 16);
        BinaryPrimitives.WriteUInt16LittleEndian(header[20..22], PcmFormatTag);
        BinaryPrimitives.WriteUInt16LittleEndian(header[22..24], AudioConstants.Channels);
        BinaryPrimitives.WriteInt32LittleEndian(header[24..28], AudioConstants.SampleRate);
        BinaryPrimitives.WriteInt32LittleEndian(
            header[28..32],
            AudioConstants.SampleRate * AudioConstants.Channels * AudioConstants.BytesPerSample);
        BinaryPrimitives.WriteUInt16LittleEndian(
            header[32..34],
            AudioConstants.Channels * AudioConstants.BytesPerSample);
        BinaryPrimitives.WriteUInt16LittleEndian(header[34..36], AudioConstants.BitsPerSample);
        "data"u8.CopyTo(header[36..40]);
        BinaryPrimitives.WriteUInt32LittleEndian(header[40..44], (uint)dataBytes);
        output.Write(header);
    }

    public static string Sha256Hex(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream));
    }

    public static string Sha256Hex(ReadOnlySpan<byte> bytes) =>
        Convert.ToHexString(SHA256.HashData(bytes));

    private static PipelineException InvalidWav(string message) =>
        new(PipelineErrorCode.SourceFormat, message);

    private static void ReadExactly(Stream stream, Span<byte> destination)
    {
        var read = 0;
        while (read < destination.Length)
        {
            var count = stream.Read(destination[read..]);
            if (count == 0)
            {
                throw new EndOfStreamException("Unexpected end of file.");
            }

            read += count;
        }
    }
}
