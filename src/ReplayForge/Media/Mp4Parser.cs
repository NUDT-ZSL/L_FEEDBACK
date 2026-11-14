using System.Buffers.Binary;
using ReplayForge.Utilities;

namespace ReplayForge.Media;

public sealed record Mp4Box(long Offset, long Size, string Type, int HeaderSize)
{
    public long PayloadOffset => Offset + HeaderSize;

    public long PayloadLength => Size - HeaderSize;
}

public sealed class Mp4Sample
{
    public int Index { get; init; }

    public long Offset { get; init; }

    public int Size { get; init; }

    public long DtsTicks { get; init; }

    public long PtsTicks { get; init; }

    public int DurationTicks { get; init; }

    public bool IsKeyframe { get; init; }
}

public sealed class Mp4Track
{
    public int TrackId { get; set; }

    public string Handler { get; set; } = "";

    public string Codec { get; set; } = "";

    public uint Timescale { get; set; }

    public ulong DurationTicks { get; set; }

    public int Width { get; set; }

    public int Height { get; set; }

    public double PresentationShiftSeconds { get; set; }

    public double? PresentationDurationSeconds { get; set; }

    public H264Info? H264 { get; set; }

    public byte[]? AudioSpecificConfig { get; set; }

    public int AudioSampleRate { get; set; }

    public int AudioChannels { get; set; }

    public List<Mp4Sample> Samples { get; } = new();

    public double DurationSeconds => Timescale == 0 ? 0 : DurationTicks / (double)Timescale;

    public double FirstPresentationPtsSeconds
    {
        get
        {
            if (Samples.Count == 0 || Timescale == 0)
            {
                return 0;
            }

            return Samples.Min(sample => sample.PtsTicks / (double)Timescale) + PresentationShiftSeconds;
        }
    }

    public double AverageFrameRate
    {
        get
        {
            if (Samples.Count < 2 || Timescale == 0)
            {
                return 0;
            }

            var first = Samples[0].DtsTicks;
            var last = Samples[^1].DtsTicks;
            return last == first ? 0 : (Samples.Count - 1) * Timescale / (double)(last - first);
        }
    }
}

public sealed class Mp4FileInfo
{
    public required string Path { get; init; }

    public long FileLength { get; init; }

    public uint MovieTimescale { get; set; }

    public ulong MovieDurationTicks { get; set; }

    public long MoovOffset { get; set; }

    public long MoovSize { get; set; }

    public long MdatOffset { get; set; } = -1;

    public long MdatDataOffset { get; set; } = -1;

    public long MdatDataLength { get; set; }

    public long MdatSize { get; set; }

    public List<Mp4Track> Tracks { get; } = new();

    public Mp4Track? VideoTrack => Tracks.FirstOrDefault(track => track.Handler == "vide");

    public Mp4Track? AudioTrack => Tracks.FirstOrDefault(track => track.Handler == "soun");
}

public static class Mp4Parser
{
    public static Mp4FileInfo Parse(string path)
    {
        if (!File.Exists(path))
        {
            throw new ReplayException("E_INVALID_STREAM", "Media file does not exist.", new { path });
        }

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        var result = new Mp4FileInfo { Path = path, FileLength = stream.Length };
        var topLevel = ReadBoxes(stream, 0, stream.Length);
        if (!topLevel.Any(box => box.Type == "ftyp"))
        {
            throw new ReplayException("E_INVALID_STREAM", "MP4 ftyp box is missing.", new { path });
        }

        var moov = topLevel.FirstOrDefault(box => box.Type == "moov")
                   ?? throw new ReplayException("E_INVALID_STREAM", "MP4 moov box is missing.", new { path });
        result.MoovOffset = moov.Offset;
        result.MoovSize = moov.Size;

        var mdat = topLevel.FirstOrDefault(box => box.Type == "mdat");
        if (mdat is not null)
        {
            result.MdatOffset = mdat.Offset;
            result.MdatDataOffset = mdat.PayloadOffset;
            result.MdatDataLength = mdat.PayloadLength;
            result.MdatSize = mdat.Size;
        }

        var moovBytes = ReadRange(stream, moov.PayloadOffset, checked((int)moov.PayloadLength));
        ParseMoov(moovBytes, result, stream);
        return result;
    }

    private static void ParseMoov(byte[] moov, Mp4FileInfo result, FileStream file)
    {
        var children = ReadBoxes(moov, 0, moov.Length);
        var mvhd = children.FirstOrDefault(box => box.Type == "mvhd")
                   ?? throw new ReplayException("E_INVALID_STREAM", "mvhd box is missing.", new { path = result.Path });
        ParseMvhd(moov, mvhd, result);

        foreach (var trak in children.Where(box => box.Type == "trak"))
        {
            result.Tracks.Add(ParseTrack(moov, trak, result.Path, file, result.MovieTimescale));
        }
    }

    private static void ParseMvhd(byte[] moov, Mp4Box mvhd, Mp4FileInfo result)
    {
        var payload = Slice(moov, mvhd);
        var version = payload[0];
        if (version == 1)
        {
            result.MovieTimescale = Bits.U32(payload, 20);
            result.MovieDurationTicks = Bits.U64(payload, 24);
        }
        else
        {
            result.MovieTimescale = Bits.U32(payload, 12);
            result.MovieDurationTicks = Bits.U32(payload, 16);
        }
    }

    private static Mp4Track ParseTrack(byte[] moov, Mp4Box trak, string path, FileStream file, uint movieTimescale)
    {
        var track = new Mp4Track();
        var trakChildren = ReadBoxes(moov, trak.PayloadOffset, trak.Offset + trak.Size);
        var tkhd = trakChildren.FirstOrDefault(box => box.Type == "tkhd")
                   ?? throw new ReplayException("E_INVALID_STREAM", "tkhd box is missing.", new { path });
        ParseTkhd(moov, tkhd, track);

        var edts = trakChildren.FirstOrDefault(box => box.Type == "edts");
        var edit = ParseEditList(moov, edts, path, movieTimescale);

        var mdia = trakChildren.FirstOrDefault(box => box.Type == "mdia")
                   ?? throw new ReplayException("E_INVALID_STREAM", "mdia box is missing.", new { path });
        var mdiaChildren = ReadBoxes(moov, mdia.PayloadOffset, mdia.Offset + mdia.Size);
        var mdhd = mdiaChildren.FirstOrDefault(box => box.Type == "mdhd")
                   ?? throw new ReplayException("E_INVALID_STREAM", "mdhd box is missing.", new { path });
        ParseMdhd(moov, mdhd, track);

        var hdlr = mdiaChildren.FirstOrDefault(box => box.Type == "hdlr")
                   ?? throw new ReplayException("E_INVALID_STREAM", "hdlr box is missing.", new { path });
        var hdlrPayload = Slice(moov, hdlr);
        track.Handler = Bits.FourCc(hdlrPayload, 8);

        var minf = mdiaChildren.FirstOrDefault(box => box.Type == "minf")
                   ?? throw new ReplayException("E_INVALID_STREAM", "minf box is missing.", new { path });
        var minfChildren = ReadBoxes(moov, minf.PayloadOffset, minf.Offset + minf.Size);
        var stbl = minfChildren.FirstOrDefault(box => box.Type == "stbl")
                   ?? throw new ReplayException("E_INVALID_STREAM", "stbl box is missing.", new { path });
        var stblChildren = ReadBoxes(moov, stbl.PayloadOffset, stbl.Offset + stbl.Size);

        ParseSampleDescription(moov, stblChildren, track, path);
        var (sttsDurations, sttsStart) = ParseStts(moov, stblChildren.FirstOrDefault(box => box.Type == "stts"), path);
        var cttsOffsets = ParseCtts(moov, stblChildren.FirstOrDefault(box => box.Type == "ctts"), sttsDurations.Count, path);
        var sizes = ParseStsz(moov, stblChildren.FirstOrDefault(box => box.Type == "stsz"), path);
        var chunkOffsets = ParseChunkOffsets(moov, stblChildren, path);
        var stsc = ParseStsc(moov, stblChildren.FirstOrDefault(box => box.Type == "stsc"), path);
        var keyframes = ParseStss(moov, stblChildren.FirstOrDefault(box => box.Type == "stss"), sizes.Count);

        if (sizes.Count != sttsDurations.Count)
        {
            throw new ReplayException("E_INVALID_STREAM",
                $"Sample-size table count {sizes.Count} does not match timing count {sttsDurations.Count}.",
                new { path, track.TrackId });
        }

        var sampleIndex = 0;
        for (var chunkIndex = 0; chunkIndex < chunkOffsets.Count && sampleIndex < sizes.Count; chunkIndex++)
        {
            var entry = stsc.Last(entry => entry.FirstChunk <= chunkIndex + 1);
            var offset = chunkOffsets[chunkIndex];
            for (var inChunk = 0; inChunk < entry.SamplesPerChunk && sampleIndex < sizes.Count; inChunk++)
            {
                var size = sizes[sampleIndex];
                track.Samples.Add(new Mp4Sample
                {
                    Index = sampleIndex,
                    Offset = offset,
                    Size = size,
                    DtsTicks = sttsStart[sampleIndex],
                    PtsTicks = sttsStart[sampleIndex] + cttsOffsets[sampleIndex],
                    DurationTicks = sttsDurations[sampleIndex],
                    IsKeyframe = keyframes is null || keyframes.Contains(sampleIndex + 1)
                });
                offset += size;
                sampleIndex++;
            }
        }

        if (sampleIndex != sizes.Count)
        {
            throw new ReplayException("E_INVALID_STREAM",
                $"Chunk table only maps {sampleIndex} of {sizes.Count} samples.",
                new { path, track.TrackId });
        }

        track.PresentationShiftSeconds = ComputePresentationShift(track, edit);
        track.PresentationDurationSeconds = edit.PresentationDurationSeconds ?? track.DurationSeconds;

        if (track.Handler == "vide" && track.H264 is not null)
        {
            ValidateEmbeddedParameterSets(track, file, path);
        }

        return track;
    }

    private static void ParseTkhd(byte[] moov, Mp4Box tkhd, Mp4Track track)
    {
        var payload = Slice(moov, tkhd);
        var version = payload[0];
        track.TrackId = version == 1 ? checked((int)Bits.U32(payload, 20)) : checked((int)Bits.U32(payload, 12));
    }

    private static void ParseMdhd(byte[] moov, Mp4Box mdhd, Mp4Track track)
    {
        var payload = Slice(moov, mdhd);
        var version = payload[0];
        if (version == 1)
        {
            track.Timescale = Bits.U32(payload, 20);
            track.DurationTicks = Bits.U64(payload, 24);
        }
        else
        {
            track.Timescale = Bits.U32(payload, 12);
            track.DurationTicks = Bits.U32(payload, 16);
        }
    }

    private static (double EmptyBeforeSeconds, long MediaTimeTicks, double? PresentationDurationSeconds) ParseEditList(
        byte[] moov,
        Mp4Box? edts,
        string path,
        uint movieTimescale)
    {
        if (edts is null)
        {
            return (0, 0, null);
        }

        var edtsChildren = ReadBoxes(moov, edts.PayloadOffset, edts.Offset + edts.Size);
        var elst = edtsChildren.FirstOrDefault(box => box.Type == "elst");
        if (elst is null)
        {
            return (0, 0, null);
        }

        var payload = Slice(moov, elst);
        var version = payload[0];
        var count = checked((int)Bits.U32(payload, 4));
        var offset = 8;
        long emptyBeforeTicks = 0;
        long mediaTime = 0;
        double? presentationDuration = null;
        for (var i = 0; i < count; i++)
        {
            long duration;
            long media;
            if (version == 1)
            {
                duration = checked((long)Bits.U64(payload, offset));
                media = unchecked((long)Bits.U64(payload, offset + 8));
                offset += 20;
            }
            else
            {
                duration = Bits.U32(payload, offset);
                media = unchecked((int)Bits.U32(payload, offset + 4));
                offset += 12;
            }

            if (media == -1)
            {
                emptyBeforeTicks += duration;
                continue;
            }

            mediaTime = media;
            presentationDuration = movieTimescale == 0 ? null : duration / (double)movieTimescale;
            break;
        }

        if (payload.Length < offset)
        {
            throw new ReplayException("E_INVALID_STREAM", "Truncated edit-list box.", new { path });
        }

        var emptySeconds = movieTimescale == 0 ? 0 : emptyBeforeTicks / (double)movieTimescale;
        return (emptySeconds, mediaTime, presentationDuration);
    }

    private static double ComputePresentationShift(
        Mp4Track track,
        (double EmptyBeforeSeconds, long MediaTimeTicks, double? PresentationDurationSeconds) edit)
    {
        if (track.Timescale == 0)
        {
            return 0;
        }

        if (edit.EmptyBeforeSeconds == 0 && edit.MediaTimeTicks == 0)
        {
            return 0;
        }

        var mediaSeconds = edit.MediaTimeTicks / (double)track.Timescale;
        return edit.EmptyBeforeSeconds - mediaSeconds;
    }

    private static void ParseSampleDescription(byte[] moov, List<Mp4Box> stblChildren, Mp4Track track, string path)
    {
        var stsd = stblChildren.FirstOrDefault(box => box.Type == "stsd")
                   ?? throw new ReplayException("E_INVALID_STREAM", "stsd box is missing.", new { path });
        var payload = Slice(moov, stsd).ToArray();
        var entryCount = Bits.U32(payload, 4);
        if (entryCount == 0)
        {
            throw new ReplayException("E_INVALID_STREAM", "Sample description has no entries.", new { path });
        }

        var entries = ReadBoxes(payload, 8, payload.Length);
        if (entries.Count == 0)
        {
            throw new ReplayException("E_INVALID_STREAM", "Sample entry is missing.", new { path });
        }

        var entry = entries[0];
        track.Codec = entry.Type;
        if (track.Handler == "vide")
        {
            if (entry.Type is not ("avc1" or "avc3"))
            {
                throw new ReplayException("E_INVALID_STREAM",
                    $"Unsupported video codec '{entry.Type}'. H.264 is required.",
                    new { path, codec = entry.Type });
            }

            var entryPayload = Slice(payload, entry);
            if (entryPayload.Length < 78)
            {
                throw new ReplayException("E_INVALID_STREAM", "Truncated visual sample entry.", new { path });
            }

            track.Width = Bits.U16(entryPayload, 24);
            track.Height = Bits.U16(entryPayload, 26);
            var children = ReadBoxes(payload, entry.PayloadOffset + 78, entry.Offset + entry.Size);
            var avcC = children.FirstOrDefault(box => box.Type == "avcC")
                       ?? throw new ReplayException("E_INVALID_STREAM", "H.264 avcC box is missing.",
                           new { path, layerPath = path });
            var avcCBytes = Slice(payload, avcC).ToArray();
            track.H264 = H264Inspector.ParseAvcC(avcCBytes, path);
        }
        else if (track.Handler == "soun")
        {
            if (entry.Type != "mp4a")
            {
                throw new ReplayException("E_INVALID_STREAM",
                    $"Unsupported audio codec '{entry.Type}'. AAC is required.",
                    new { path, codec = entry.Type });
            }

            var entryPayload = Slice(payload, entry);
            if (entryPayload.Length < 28)
            {
                throw new ReplayException("E_INVALID_STREAM", "Truncated audio sample entry.", new { path });
            }

            track.AudioChannels = Bits.U16(entryPayload, 16);
            track.AudioSampleRate = (int)(Bits.U32(entryPayload, 24) >> 16);
            var children = ReadBoxes(payload, entry.PayloadOffset + 28, entry.Offset + entry.Size);
            var esds = children.FirstOrDefault(box => box.Type == "esds");
            if (esds is not null)
            {
                track.AudioSpecificConfig = TryFindDecoderSpecificInfo(Slice(payload, esds));
            }
        }
    }

    private static byte[]? TryFindDecoderSpecificInfo(ReadOnlySpan<byte> esds)
    {
        for (var i = 4; i + 2 <= esds.Length; i++)
        {
            if (esds[i] == 0x05)
            {
                var length = 0;
                var j = i + 1;
                for (var k = 0; k < 4 && j < esds.Length; k++)
                {
                    var value = esds[j++];
                    length = (length << 7) | (value & 0x7F);
                    if ((value & 0x80) == 0)
                    {
                        break;
                    }
                }

                if (length is 2 or 5 && j + length <= esds.Length)
                {
                    return esds.Slice(j, length).ToArray();
                }
            }
        }

        return null;
    }

    private static (List<int> Durations, List<long> Starts) ParseStts(byte[] moov, Mp4Box? stts, string path)
    {
        var durations = new List<int>();
        var starts = new List<long>();
        if (stts is null)
        {
            return (durations, starts);
        }

        var payload = Slice(moov, stts);
        var count = checked((int)Bits.U32(payload, 4));
        var offset = 8;
        long start = 0;
        for (var i = 0; i < count; i++)
        {
            var sampleCount = checked((int)Bits.U32(payload, offset));
            var delta = checked((int)Bits.U32(payload, offset + 4));
            offset += 8;
            for (var sample = 0; sample < sampleCount; sample++)
            {
                durations.Add(delta);
                starts.Add(start);
                start += delta;
            }
        }

        if (offset > payload.Length)
        {
            throw new ReplayException("E_INVALID_STREAM", "Truncated stts box.", new { path });
        }

        return (durations, starts);
    }

    private static List<long> ParseCtts(byte[] moov, Mp4Box? ctts, int sampleCount, string path)
    {
        var offsets = Enumerable.Repeat(0L, sampleCount).ToList();
        if (ctts is null)
        {
            return offsets;
        }

        var payload = Slice(moov, ctts);
        var version = payload[0];
        var count = checked((int)Bits.U32(payload, 4));
        var offset = 8;
        var sampleIndex = 0;
        for (var i = 0; i < count; i++)
        {
            var run = checked((int)Bits.U32(payload, offset));
            var delta = version == 1
                ? unchecked((int)Bits.U32(payload, offset + 4))
                : checked((int)Bits.U32(payload, offset + 4));
            offset += 8;
            for (var j = 0; j < run && sampleIndex < sampleCount; j++, sampleIndex++)
            {
                offsets[sampleIndex] = delta;
            }
        }

        if (sampleIndex != sampleCount)
        {
            throw new ReplayException("E_INVALID_STREAM", "ctts sample count is inconsistent.", new { path });
        }

        return offsets;
    }

    private static List<int> ParseStsz(byte[] moov, Mp4Box? stsz, string path)
    {
        if (stsz is null)
        {
            throw new ReplayException("E_INVALID_STREAM", "stsz box is missing.", new { path });
        }

        var payload = Slice(moov, stsz);
        var fixedSize = Bits.U32(payload, 4);
        var count = checked((int)Bits.U32(payload, 8));
        if (fixedSize != 0)
        {
            return Enumerable.Repeat(checked((int)fixedSize), count).ToList();
        }

        if (12 + count * 4 > payload.Length)
        {
            throw new ReplayException("E_INVALID_STREAM", "Truncated stsz box.", new { path });
        }

        var sizes = new List<int>(count);
        for (var i = 0; i < count; i++)
        {
            sizes.Add(checked((int)Bits.U32(payload, 12 + i * 4)));
        }

        return sizes;
    }

    private static List<long> ParseChunkOffsets(byte[] moov, List<Mp4Box> stbl, string path)
    {
        var stco = stbl.FirstOrDefault(box => box.Type == "stco");
        var co64 = stbl.FirstOrDefault(box => box.Type == "co64");
        if (stco is null && co64 is null)
        {
            throw new ReplayException("E_INVALID_STREAM", "stco/co64 box is missing.", new { path });
        }

        var box = stco ?? co64!;
        var payload = Slice(moov, box);
        var count = checked((int)Bits.U32(payload, 4));
        var offsets = new List<long>(count);
        var width = box.Type == "co64" ? 8 : 4;
        if (8 + count * width > payload.Length)
        {
            throw new ReplayException("E_INVALID_STREAM", "Truncated chunk-offset table.", new { path });
        }

        for (var i = 0; i < count; i++)
        {
            offsets.Add(width == 8
                ? checked((long)Bits.U64(payload, 8 + i * 8))
                : Bits.U32(payload, 8 + i * 4));
        }

        return offsets;
    }

    private static List<StscEntry> ParseStsc(byte[] moov, Mp4Box? stsc, string path)
    {
        if (stsc is null)
        {
            throw new ReplayException("E_INVALID_STREAM", "stsc box is missing.", new { path });
        }

        var payload = Slice(moov, stsc);
        var count = checked((int)Bits.U32(payload, 4));
        if (count == 0 || 8 + count * 12 > payload.Length)
        {
            throw new ReplayException("E_INVALID_STREAM", "Invalid stsc table.", new { path });
        }

        var entries = new List<StscEntry>(count);
        for (var i = 0; i < count; i++)
        {
            entries.Add(new StscEntry(
                Bits.U32(payload, 8 + i * 12),
                checked((int)Bits.U32(payload, 12 + i * 12))));
        }

        return entries;
    }

    private static HashSet<int>? ParseStss(byte[] moov, Mp4Box? stss, int sampleCount)
    {
        if (stss is null)
        {
            return null;
        }

        var payload = Slice(moov, stss);
        var count = checked((int)Bits.U32(payload, 4));
        var result = new HashSet<int>();
        for (var i = 0; i < count; i++)
        {
            var sample = checked((int)Bits.U32(payload, 8 + i * 4));
            if (sample >= 1 && sample <= sampleCount)
            {
                result.Add(sample);
            }
        }

        return result;
    }

    private static void ValidateEmbeddedParameterSets(Mp4Track track, FileStream file, string path)
    {
        if (track.H264 is null || track.Timescale == 0)
        {
            return;
        }

        foreach (var sample in track.Samples.Where(sample => sample.IsKeyframe))
        {
            if (sample.Size > 16 * 1024 * 1024)
            {
                throw new ReplayException("E_INVALID_STREAM", "H.264 sample is unreasonably large.", new { path });
            }

            var bytes = ReadRange(file, sample.Offset, sample.Size);
            var nals = H264Inspector.SplitLengthPrefixed(bytes, track.H264.LengthSize);
            foreach (var nal in nals)
            {
                var type = nal[0] & 0x1F;
                if (type == 7)
                {
                    if (!H264Inspector.IsSameParameterSet(nal, track.H264.Sps[0]))
                    {
                        throw new ReplayException("E_INVALID_STREAM",
                            "Embedded H.264 SPS differs from the avcC parameter set.",
                            new { path, trackId = track.TrackId, sample = sample.Index });
                    }
                }
                else if (type == 8)
                {
                    if (!track.H264.Pps.Any(pps => H264Inspector.IsSameParameterSet(nal, pps)))
                    {
                        throw new ReplayException("E_INVALID_STREAM",
                            "Embedded H.264 PPS differs from the avcC parameter set.",
                            new { path, trackId = track.TrackId, sample = sample.Index });
                    }
                }
            }
        }
    }

    private static List<Mp4Box> ReadBoxes(Stream stream, long start, long end)
    {
        var boxes = new List<Mp4Box>();
        var offset = start;
        Span<byte> header = stackalloc byte[16];
        while (offset + 8 <= end)
        {
            stream.Position = offset;
            stream.ReadExactly(header[..8]);
            var size = BinaryPrimitives.ReadUInt32BigEndian(header);
            var type = System.Text.Encoding.ASCII.GetString(header.Slice(4, 4));
            var headerSize = 8;
            long actualSize = size;
            if (size == 1)
            {
                if (offset + 16 > end)
                {
                    throw new ReplayException("E_INVALID_STREAM", "Truncated extended MP4 box header.");
                }

                stream.ReadExactly(header.Slice(8, 8));
                actualSize = checked((long)BinaryPrimitives.ReadUInt64BigEndian(header.Slice(8, 8)));
                headerSize = 16;
            }
            else if (size == 0)
            {
                actualSize = end - offset;
            }

            if (actualSize < headerSize || offset + actualSize > end)
            {
                throw new ReplayException("E_INVALID_STREAM",
                    $"Invalid MP4 box size at offset {offset}: {actualSize} ({type}).");
            }

            boxes.Add(new Mp4Box(offset, actualSize, type, headerSize));
            offset += actualSize;
        }

        return boxes;
    }

    private static List<Mp4Box> ReadBoxes(byte[] data, int start, int end)
        => ReadBoxes(data, (long)start, end);

    private static List<Mp4Box> ReadBoxes(byte[] data, long start, long end)
    {
        var boxes = new List<Mp4Box>();
        var offset = checked((int)start);
        var limit = checked((int)end);
        while (offset + 8 <= limit)
        {
            var size = (long)Bits.U32(data, offset);
            var type = Bits.FourCc(data, offset + 4);
            var headerSize = 8;
            if (size == 1)
            {
                size = checked((long)Bits.U64(data, offset + 8));
                headerSize = 16;
            }
            else if (size == 0)
            {
                size = limit - offset;
            }

            if (size < headerSize || offset + size > limit)
            {
                throw new ReplayException("E_INVALID_STREAM", $"Invalid MP4 box size at offset {offset} ({type}).");
            }

            boxes.Add(new Mp4Box(offset, size, type, headerSize));
            offset += checked((int)size);
        }

        return boxes;
    }

    private static ReadOnlySpan<byte> Slice(byte[] data, Mp4Box box)
        => data.AsSpan(checked((int)box.PayloadOffset), checked((int)box.PayloadLength));

    private static byte[] ReadRange(Stream stream, long offset, int count)
    {
        var bytes = new byte[count];
        stream.Position = offset;
        stream.ReadExactly(bytes);
        return bytes;
    }

    private sealed record StscEntry(uint FirstChunk, int SamplesPerChunk);
}
