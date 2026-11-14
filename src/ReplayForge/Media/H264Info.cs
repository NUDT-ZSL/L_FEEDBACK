using ReplayForge.Utilities;

namespace ReplayForge.Media;

public sealed class H264Info
{
    public required byte[] AvcC { get; init; }

    public required IReadOnlyList<byte[]> Sps { get; init; }

    public required IReadOnlyList<byte[]> Pps { get; init; }

    public required int LengthSize { get; init; }

    public required int ProfileIdc { get; init; }

    public required int ProfileCompatibility { get; init; }

    public required int LevelIdc { get; init; }

    public required string ProfileName { get; init; }

    public bool IsSupported => ProfileIdc is 66 or 77 or 100;
}

public static class H264Inspector
{
    public static H264Info ParseAvcC(byte[] avcC, string path)
    {
        if (avcC.Length < 7 || avcC[0] != 1)
        {
            throw new ReplayException("E_INVALID_STREAM", "Invalid AVCDecoderConfigurationRecord.", new { path });
        }

        var lengthSize = (avcC[4] & 0x03) + 1;
        var offset = 5;
        var sps = new List<byte[]>();
        var pps = new List<byte[]>();

        if (offset >= avcC.Length)
        {
            throw new ReplayException("E_INVALID_STREAM", "Missing H.264 SPS in avcC.", new { path });
        }

        var spsCount = avcC[offset++] & 0x1F;
        if (spsCount == 0)
        {
            throw new ReplayException("E_INVALID_STREAM", "H.264 stream has no SPS.", new { path });
        }

        for (var i = 0; i < spsCount; i++)
        {
            if (offset + 2 > avcC.Length)
            {
                throw new ReplayException("E_INVALID_STREAM", "Truncated SPS length in avcC.", new { path });
            }

            var length = Bits.U16(avcC, offset);
            offset += 2;
            if (length == 0 || offset + length > avcC.Length)
            {
                throw new ReplayException("E_INVALID_STREAM", "Truncated SPS payload in avcC.", new { path });
            }

            sps.Add(avcC.AsSpan(offset, length).ToArray());
            offset += length;
        }

        if (offset >= avcC.Length)
        {
            throw new ReplayException("E_INVALID_STREAM", "Missing H.264 PPS in avcC.", new { path });
        }

        var ppsCount = avcC[offset++];
        if (ppsCount == 0)
        {
            throw new ReplayException("E_INVALID_STREAM", "H.264 stream has no PPS.", new { path });
        }

        for (var i = 0; i < ppsCount; i++)
        {
            if (offset + 2 > avcC.Length)
            {
                throw new ReplayException("E_INVALID_STREAM", "Truncated PPS length in avcC.", new { path });
            }

            var length = Bits.U16(avcC, offset);
            offset += 2;
            if (length == 0 || offset + length > avcC.Length)
            {
                throw new ReplayException("E_INVALID_STREAM", "Truncated PPS payload in avcC.", new { path });
            }

            pps.Add(avcC.AsSpan(offset, length).ToArray());
            offset += length;
        }

        var distinctSps = sps.Distinct(ByteArrayComparer.Instance).ToList();
        if (distinctSps.Count != 1)
        {
            throw new ReplayException("E_INVALID_STREAM", "Inconsistent H.264 SPS parameter sets in avcC.",
                new { path, spsCount = distinctSps.Count });
        }

        var firstSps = sps[0];
        if (firstSps.Length < 4 || (firstSps[0] & 0x1F) != 7)
        {
            throw new ReplayException("E_INVALID_STREAM", "Invalid H.264 SPS NAL unit.", new { path });
        }

        var profileIdc = firstSps[1];
        var compatibility = firstSps[2];
        var levelIdc = firstSps[3];
        var profileName = profileIdc switch
        {
            66 => "Baseline",
            77 => "Main",
            100 => "High",
            88 => "Extended",
            110 => "High 10",
            122 => "High 4:2:2",
            244 => "High 4:4:4 Predictive",
            _ => $"Unknown({profileIdc})"
        };

        var info = new H264Info
        {
            AvcC = avcC,
            Sps = sps,
            Pps = pps,
            LengthSize = lengthSize,
            ProfileIdc = profileIdc,
            ProfileCompatibility = compatibility,
            LevelIdc = levelIdc,
            ProfileName = profileName
        };

        if (!info.IsSupported)
        {
            throw new ReplayException(
                "E_INVALID_STREAM",
                $"Unsupported H.264 profile {profileName} (profile_idc={profileIdc}). Supported profiles are Baseline, Main, and High.",
                new { path, profileIdc, profileName, levelIdc });
        }

        return info;
    }

    public static IReadOnlyList<byte[]> SplitLengthPrefixed(byte[] sample, int lengthSize)
    {
        var result = new List<byte[]>();
        var offset = 0;
        while (offset < sample.Length)
        {
            if (offset + lengthSize > sample.Length)
            {
                throw new ReplayException("E_INVALID_STREAM", "Truncated H.264 NAL length.");
            }

            long length = 0;
            for (var i = 0; i < lengthSize; i++)
            {
                length = (length << 8) | sample[offset + i];
            }

            offset += lengthSize;
            if (length <= 0 || length > sample.Length - offset)
            {
                throw new ReplayException("E_INVALID_STREAM", "Invalid H.264 NAL length.");
            }

            result.Add(sample.AsSpan(offset, (int)length).ToArray());
            offset += (int)length;
        }

        return result;
    }

    public static IReadOnlyList<byte[]> SplitAnnexB(byte[] data)
    {
        var starts = new List<int>();
        for (var i = 0; i + 3 <= data.Length;)
        {
            if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1)
            {
                starts.Add(i);
                i += 3;
            }
            else if (i + 4 <= data.Length && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1)
            {
                starts.Add(i);
                i += 4;
            }
            else
            {
                i++;
            }
        }

        var nals = new List<byte[]>();
        for (var i = 0; i < starts.Count; i++)
        {
            var start = starts[i];
            var prefix = start + 2 < data.Length && data[start + 2] == 1 ? 3 : 4;
            var end = i + 1 < starts.Count ? starts[i + 1] : data.Length;
            var payloadStart = start + prefix;
            while (end > payloadStart && data[end - 1] == 0)
            {
                end--;
            }

            if (end > payloadStart)
            {
                nals.Add(data.AsSpan(payloadStart, end - payloadStart).ToArray());
            }
        }

        if (starts.Count == 0)
        {
            throw new ReplayException("E_INVALID_STREAM", "No Annex B NAL units were found.");
        }

        return nals;
    }

    public static bool IsSameParameterSet(byte[] left, byte[] right)
        => left.AsSpan().SequenceEqual(right);
}

internal sealed class ByteArrayComparer : IEqualityComparer<byte[]>
{
    public static readonly ByteArrayComparer Instance = new();

    public bool Equals(byte[]? x, byte[]? y)
        => x is not null && y is not null && x.AsSpan().SequenceEqual(y);

    public int GetHashCode(byte[] obj)
    {
        var hash = new HashCode();
        foreach (var value in obj)
        {
            hash.Add(value);
        }

        return hash.ToHashCode();
    }
}
