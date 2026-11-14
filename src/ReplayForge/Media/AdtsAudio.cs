using ReplayForge.Utilities;

namespace ReplayForge.Media;

public sealed class AdtsFrame
{
    public long Offset { get; init; }

    public int TotalLength { get; init; }

    public int HeaderLength { get; init; }

    public int SampleCount { get; init; }

    public int RawLength => TotalLength - HeaderLength;
}

public sealed class AdtsAudioInfo
{
    public required string Path { get; init; }

    public required int ObjectType { get; init; }

    public required int SampleRate { get; init; }

    public required int SampleRateIndex { get; init; }

    public required int Channels { get; init; }

    public required int ChannelConfiguration { get; init; }

    public required bool CrcPresent { get; init; }

    public required List<AdtsFrame> Frames { get; init; }

    public int TotalSamples => Frames.Sum(frame => frame.SampleCount);

    public double DurationSeconds => (double)TotalSamples / SampleRate;

    public byte[] AudioSpecificConfig
    {
        get
        {
            var value = (ObjectType << 11) | (SampleRateIndex << 7) | (ChannelConfiguration << 3);
            return new[] { (byte)(value >> 8), (byte)value };
        }
    }
}

public static class AdtsParser
{
    private static readonly int[] SampleRates =
    {
        96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050,
        16000, 12000, 11025, 8000, 7350
    };

    public static AdtsAudioInfo Parse(string path, int? expectedSampleRate = null, int? expectedChannels = null)
    {
        if (!File.Exists(path))
        {
            throw new ReplayException("E_INVALID_STREAM", "AAC file does not exist.", new { path });
        }

        var data = File.ReadAllBytes(path);
        var frames = new List<AdtsFrame>();
        var offset = 0;
        int? objectType = null;
        int? sampleRateIndex = null;
        int? sampleRate = null;
        int? channelConfig = null;
        bool? crcPresent = null;

        if (data.Length >= 10 && data[0] == (byte)'I' && data[1] == (byte)'D' && data[2] == (byte)'3')
        {
            var id3Size = (data[6] & 0x7F) << 21 | (data[7] & 0x7F) << 14 | (data[8] & 0x7F) << 7 | (data[9] & 0x7F);
            offset = 10 + id3Size;
        }

        while (offset < data.Length)
        {
            if (offset + 7 > data.Length)
            {
                throw Invalid(path, offset, "Truncated ADTS header.");
            }

            if (data[offset] != 0xFF || (data[offset + 1] & 0xF6) != 0xF0)
            {
                throw Invalid(path, offset, "Invalid ADTS sync word.");
            }

            var layer = (data[offset + 1] >> 1) & 0x03;
            if (layer != 0)
            {
                throw Invalid(path, offset, "ADTS layer must be zero.");
            }

            var protectionAbsent = (data[offset + 1] & 0x01) != 0;
            var frameObjectType = ((data[offset + 2] >> 6) & 0x03) + 1;
            var frameSampleRateIndex = (data[offset + 2] >> 2) & 0x0F;
            var frameChannelConfig = ((data[offset + 2] & 0x01) << 2) | ((data[offset + 3] >> 6) & 0x03);
            var frameLength = ((data[offset + 3] & 0x03) << 11)
                              | (data[offset + 4] << 3)
                              | ((data[offset + 5] >> 5) & 0x07);
            var blocks = data[offset + 6] & 0x03;
            var headerLength = protectionAbsent ? 7 : 9;

            if (frameSampleRateIndex >= SampleRates.Length)
            {
                throw Invalid(path, offset, $"Unsupported ADTS sample-rate index {frameSampleRateIndex}.");
            }

            if (frameLength < headerLength || offset + frameLength > data.Length)
            {
                throw Invalid(path, offset, "ADTS frame length is outside the file.");
            }

            objectType ??= frameObjectType;
            sampleRateIndex ??= frameSampleRateIndex;
            sampleRate ??= SampleRates[frameSampleRateIndex];
            channelConfig ??= frameChannelConfig;
            crcPresent ??= !protectionAbsent;

            if (frameObjectType != objectType)
            {
                throw Invalid(path, offset, "Inconsistent ADTS object type.");
            }

            if (frameSampleRateIndex != sampleRateIndex)
            {
                throw Invalid(path, offset, "Inconsistent ADTS sample rate.");
            }

            if (frameChannelConfig != channelConfig)
            {
                throw Invalid(path, offset, "Inconsistent ADTS channel configuration.");
            }

            if (!protectionAbsent != crcPresent)
            {
                throw Invalid(path, offset, "Inconsistent ADTS CRC mode.");
            }

            frames.Add(new AdtsFrame
            {
                Offset = offset,
                TotalLength = frameLength,
                HeaderLength = headerLength,
                SampleCount = 1024 * (blocks + 1)
            });
            offset += frameLength;
        }

        if (frames.Count == 0)
        {
            throw Invalid(path, 0, "No ADTS frames were found.");
        }

        if (objectType != 2)
        {
            throw new ReplayException("E_INVALID_STREAM",
                "Only AAC-LC (ADTS object type 2) commentary audio is supported.",
                new { path, objectType });
        }

        if (channelConfig is < 1 or > 7)
        {
            throw new ReplayException("E_INVALID_STREAM", "Unsupported AAC channel configuration.",
                new { path, channelConfiguration = channelConfig });
        }

        var channels = channelConfig == 7 ? 8 : channelConfig!.Value;
        if (expectedSampleRate is not null && sampleRate != expectedSampleRate)
        {
            throw new ReplayException("E_INVALID_STREAM",
                $"AAC sample rate {sampleRate} Hz does not match manifest value {expectedSampleRate} Hz.",
                new { path, expected = expectedSampleRate, actual = sampleRate });
        }

        if (expectedChannels is not null && channels != expectedChannels)
        {
            throw new ReplayException("E_INVALID_STREAM",
                $"AAC channel count {channels} does not match manifest value {expectedChannels}.",
                new { path, expected = expectedChannels, actual = channels });
        }

        return new AdtsAudioInfo
        {
            Path = path,
            ObjectType = objectType.Value,
            SampleRate = sampleRate!.Value,
            SampleRateIndex = sampleRateIndex!.Value,
            Channels = channels,
            ChannelConfiguration = channelConfig!.Value,
            CrcPresent = crcPresent!.Value,
            Frames = frames
        };
    }

    public static AdtsAudioInfo ParseBytes(byte[] data, string path, int? expectedSampleRate = null, int? expectedChannels = null)
    {
        var temporary = Path.Combine(Path.GetTempPath(), "replayforge-adts-" + Guid.NewGuid().ToString("N") + ".aac");
        try
        {
            File.WriteAllBytes(temporary, data);
            return Parse(temporary, expectedSampleRate, expectedChannels);
        }
        finally
        {
            JsonIO.TryDelete(temporary);
        }
    }

    private static ReplayException Invalid(string path, int offset, string message)
        => new("E_INVALID_STREAM", message, new { path, byteOffset = offset });
}
