using ReplayForge.Utilities;

namespace ReplayForge.Media;

public sealed class EncodedAccessUnit
{
    public required byte[] Data { get; init; }

    public required bool Keyframe { get; init; }
}

public sealed class EncodedSegment
{
    public required List<EncodedAccessUnit> AccessUnits { get; init; }

    public required IReadOnlyList<byte[]> Sps { get; init; }

    public required IReadOnlyList<byte[]> Pps { get; init; }

    public required byte[] AvcC { get; init; }
}

public static class H264SegmentReader
{
    public static EncodedSegment Read(string path, int expectedFrameCount)
    {
        if (!File.Exists(path))
        {
            throw new ReplayException("E_FFMPEG_FAILED", "Encoded H.264 segment is missing.", new { path });
        }

        var data = File.ReadAllBytes(path);
        var nals = H264Inspector.SplitAnnexB(data);
        var sps = new List<byte[]>();
        var pps = new List<byte[]>();
        var accessUnits = new List<EncodedAccessUnit>();
        var current = new List<byte[]>();
        var currentKeyframe = false;

        void Flush()
        {
            if (current.Count == 0)
            {
                return;
            }

            accessUnits.Add(new EncodedAccessUnit
            {
                Data = LengthPrefix(current, 4),
                Keyframe = currentKeyframe
            });
            current = new List<byte[]>();
            currentKeyframe = false;
        }

        foreach (var nal in nals)
        {
            if (nal.Length == 0)
            {
                continue;
            }

            var type = nal[0] & 0x1F;
            switch (type)
            {
                case 7:
                    if (!sps.Any(existing => H264Inspector.IsSameParameterSet(existing, nal)))
                    {
                        sps.Add(nal);
                    }

                    break;
                case 8:
                    if (!pps.Any(existing => H264Inspector.IsSameParameterSet(existing, nal)))
                    {
                        pps.Add(nal);
                    }

                    break;
                case 9:
                    Flush();
                    break;
                case 6:
                case 5:
                case 1:
                case 2:
                case 3:
                case 4:
                    current.Add(nal);
                    currentKeyframe |= type == 5;
                    break;
            }
        }

        Flush();
        if (accessUnits.Count != expectedFrameCount)
        {
            throw new ReplayException("E_FFMPEG_FAILED",
                $"Encoded segment contains {accessUnits.Count} access units; expected {expectedFrameCount}.",
                new { path, expected = expectedFrameCount, actual = accessUnits.Count });
        }

        if (!accessUnits[0].Keyframe)
        {
            throw new ReplayException("E_FFMPEG_FAILED", "First access unit of a segment is not a keyframe.",
                new { path });
        }

        if (sps.Count == 0 || pps.Count == 0)
        {
            throw new ReplayException("E_INVALID_STREAM", "Encoded segment has no SPS/PPS parameter sets.",
                new { path });
        }

        var avcC = BuildAvcC(sps, pps);
        H264Inspector.ParseAvcC(avcC, path);
        return new EncodedSegment
        {
            AccessUnits = accessUnits,
            Sps = sps,
            Pps = pps,
            AvcC = avcC
        };
    }

    public static bool SameParameterSets(
        IReadOnlyList<byte[]> leftSps,
        IReadOnlyList<byte[]> leftPps,
        IReadOnlyList<byte[]> rightSps,
        IReadOnlyList<byte[]> rightPps)
    {
        if (leftSps.Count != rightSps.Count || leftPps.Count != rightPps.Count)
        {
            return false;
        }

        return leftSps.Zip(rightSps).All(pair => H264Inspector.IsSameParameterSet(pair.First, pair.Second))
               && leftPps.Zip(rightPps).All(pair => H264Inspector.IsSameParameterSet(pair.First, pair.Second));
    }

    public static byte[] BuildAvcC(IReadOnlyList<byte[]> sps, IReadOnlyList<byte[]> pps)
    {
        if (sps.Count == 0 || pps.Count == 0 || sps[0].Length < 4)
        {
            throw new ReplayException("E_INVALID_STREAM", "Cannot build avcC without SPS and PPS.");
        }

        using var stream = new MemoryStream();
        stream.WriteByte(1);
        stream.WriteByte(sps[0][1]);
        stream.WriteByte(sps[0][2]);
        stream.WriteByte(sps[0][3]);
        stream.WriteByte(0xFF);
        stream.WriteByte((byte)(0xE0 | sps.Count));
        foreach (var item in sps)
        {
            WriteU16(stream, item.Length);
            stream.Write(item);
        }

        stream.WriteByte((byte)pps.Count);
        foreach (var item in pps)
        {
            WriteU16(stream, item.Length);
            stream.Write(item);
        }

        return stream.ToArray();
    }

    private static byte[] LengthPrefix(IReadOnlyList<byte[]> nals, int lengthSize)
    {
        using var stream = new MemoryStream();
        foreach (var nal in nals)
        {
            if (lengthSize == 4)
            {
                WriteU32(stream, nal.Length);
            }
            else if (lengthSize == 2)
            {
                WriteU16(stream, nal.Length);
            }
            else
            {
                throw new ReplayException("E_INVALID_STREAM", "Unsupported AVC NAL length size.");
            }

            stream.Write(nal);
        }

        return stream.ToArray();
    }

    private static void WriteU16(Stream stream, int value)
    {
        stream.WriteByte((byte)(value >> 8));
        stream.WriteByte((byte)value);
    }

    private static void WriteU32(Stream stream, int value)
    {
        stream.WriteByte((byte)(value >> 24));
        stream.WriteByte((byte)(value >> 16));
        stream.WriteByte((byte)(value >> 8));
        stream.WriteByte((byte)value);
    }
}
