using System.Buffers.Binary;
using System.Text;
using ReplayForge.State;
using ReplayForge.Utilities;

namespace ReplayForge.Media;

public static class Mp4Muxer
{
    public const int InitialMediaDataOffset = 48;

    public static void InitializeOutput(string path)
    {
        var directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        WriteFtyp(stream);
        WriteU32(stream, 1);
        WriteFourCc(stream, "mdat");
        WriteU64(stream, 16);
        stream.Flush(flushToDisk: true);
    }

    public static void FinalizeOutput(
        string outputPath,
        int outputWidth,
        int outputHeight,
        int outputFps,
        byte[] avcC,
        int audioSampleRate,
        int audioChannels,
        byte[] audioSpecificConfig,
        IReadOnlyList<JournalSample> videoSamples,
        IReadOnlyList<JournalSample> audioSamples)
    {
        using var stream = new FileStream(outputPath, FileMode.Open, FileAccess.ReadWrite, FileShare.Read);
        var moovOffset = stream.Length;
        if (moovOffset < InitialMediaDataOffset)
        {
            throw new ReplayException("E_OUTPUT_CORRUPT", "Output data area is truncated.", new { outputPath });
        }

        stream.Position = 40;
        WriteU64(stream, checked((ulong)(moovOffset - 32)));
        stream.Position = moovOffset;

        var videoDurationSeconds = videoSamples.Count / (double)outputFps;
        var audioDurationSeconds = audioSamples.Count * 1024.0 / audioSampleRate;
        _ = audioDurationSeconds;
        var movieDurationSeconds = videoDurationSeconds;
        var moov = BuildMoov(
            outputWidth,
            outputHeight,
            outputFps,
            avcC,
            audioSampleRate,
            audioChannels,
            audioSpecificConfig,
            videoSamples,
            audioSamples,
            movieDurationSeconds);
        stream.Write(moov);
        stream.Flush(flushToDisk: true);
    }

    public static byte[] BuildMoov(
        int width,
        int height,
        int fps,
        byte[] avcC,
        int audioSampleRate,
        int audioChannels,
        byte[] audioSpecificConfig,
        IReadOnlyList<JournalSample> videoSamples,
        IReadOnlyList<JournalSample> audioSamples,
        double movieDurationSeconds)
    {
        var movieTimescale = 1000u;
        var movieDuration = ToU32(Math.Round(movieDurationSeconds * movieTimescale));
        using var moov = new MemoryStream();
        WriteBox(moov, "moov", body =>
        {
            WriteBox(body, "mvhd", mvhd =>
            {
                WriteU32(mvhd, 0);
                WriteU32(mvhd, 0);
                WriteU32(mvhd, 0);
                WriteU32(mvhd, movieTimescale);
                WriteU32(mvhd, movieDuration);
                WriteU32(mvhd, 0x00010000);
                WriteU16(mvhd, 0x0100);
                WriteU16(mvhd, 0);
                WriteZeros(mvhd, 8);
                WriteMatrix(mvhd);
                WriteZeros(mvhd, 24);
                WriteU32(mvhd, 3);
            });

            WriteBox(body, "trak", trak =>
                BuildVideoTrak(trak, width, height, fps, avcC, videoSamples, movieTimescale));
            WriteBox(body, "trak", trak =>
                BuildAudioTrak(
                    trak,
                    audioSampleRate,
                    audioChannels,
                    audioSpecificConfig,
                    audioSamples,
                    movieTimescale,
                    movieDurationSeconds));
        });
        return moov.ToArray();
    }

    private static void BuildVideoTrak(
        Stream trak,
        int width,
        int height,
        int fps,
        byte[] avcC,
        IReadOnlyList<JournalSample> samples,
        uint movieTimescale)
    {
        var durationSeconds = samples.Count / (double)fps;
        WriteBox(trak, "tkhd", tkhd =>
        {
            WriteU32(tkhd, 0x00000003);
            WriteU32(tkhd, 0);
            WriteU32(tkhd, 0);
            WriteU32(tkhd, 1);
            WriteU32(tkhd, 0);
            WriteU32(tkhd, ToU32(Math.Round(durationSeconds * movieTimescale)));
            WriteZeros(tkhd, 8);
            WriteU16(tkhd, 0);
            WriteU16(tkhd, 0);
            WriteU16(tkhd, 0);
            WriteU16(tkhd, 0);
            WriteMatrix(tkhd);
            WriteU32(tkhd, (uint)width << 16);
            WriteU32(tkhd, (uint)height << 16);
        });

        WriteBox(trak, "mdia", mdia =>
        {
            WriteBox(mdia, "mdhd", mdhd =>
            {
                WriteU32(mdhd, 0);
                WriteU32(mdhd, 0);
                WriteU32(mdhd, 0);
                WriteU32(mdhd, checked((uint)fps));
                WriteU32(mdhd, checked((uint)samples.Count));
                WriteU16(mdhd, 0x55C4);
                WriteU16(mdhd, 0);
            });
            WriteHdlr(mdia, "vide", "VideoHandler");
            WriteBox(mdia, "minf", minf =>
            {
                WriteBox(minf, "vmhd", vmhd =>
                {
                    WriteU32(vmhd, 1);
                    WriteU16(vmhd, 0);
                    WriteU16(vmhd, 0);
                    WriteU16(vmhd, 0);
                    WriteU16(vmhd, 0);
                });
                WriteDinf(minf);
                WriteBox(minf, "stbl", stbl =>
                {
                    WriteBox(stbl, "stsd", stsd =>
                    {
                        WriteU32(stsd, 0);
                        WriteU32(stsd, 1);
                        WriteAvc1(stsd, width, height, avcC);
                    });
                    WriteStts(stbl, samples.Count, 1);
                    WriteStss(stbl, samples);
                    WriteStsc(stbl);
                    WriteStsz(stbl, samples);
                    WriteCo64(stbl, samples);
                });
            });
        });
    }

    private static void BuildAudioTrak(
        Stream trak,
        int sampleRate,
        int channels,
        byte[] audioSpecificConfig,
        IReadOnlyList<JournalSample> samples,
        uint movieTimescale,
        double presentationDurationSeconds)
    {
        var mediaDuration = checked((uint)(samples.Count * 1024L));
        WriteBox(trak, "tkhd", tkhd =>
        {
            WriteU32(tkhd, 0x00000003);
            WriteU32(tkhd, 0);
            WriteU32(tkhd, 0);
            WriteU32(tkhd, 2);
            WriteU32(tkhd, 0);
            WriteU32(tkhd, ToU32(Math.Round(presentationDurationSeconds * movieTimescale)));
            WriteZeros(tkhd, 8);
            WriteU16(tkhd, 0);
            WriteU16(tkhd, 0);
            WriteU16(tkhd, 0x0100);
            WriteU16(tkhd, 0);
            WriteMatrix(tkhd);
            WriteU32(tkhd, 0);
            WriteU32(tkhd, 0);
        });

        WriteBox(trak, "edts", edts =>
        {
            WriteBox(edts, "elst", elst =>
            {
                WriteU32(elst, 0);
                WriteU32(elst, 1);
                WriteU32(elst, ToU32(Math.Round(presentationDurationSeconds * movieTimescale)));
                WriteU32(elst, 0);
                WriteU32(elst, 0x00010000);
            });
        });

        WriteBox(trak, "mdia", mdia =>
        {
            WriteBox(mdia, "mdhd", mdhd =>
            {
                WriteU32(mdhd, 0);
                WriteU32(mdhd, 0);
                WriteU32(mdhd, 0);
                WriteU32(mdhd, checked((uint)sampleRate));
                WriteU32(mdhd, mediaDuration);
                WriteU16(mdhd, 0x55C4);
                WriteU16(mdhd, 0);
            });
            WriteHdlr(mdia, "soun", "SoundHandler");
            WriteBox(mdia, "minf", minf =>
            {
                WriteBox(minf, "smhd", smhd =>
                {
                    WriteU32(smhd, 0);
                    WriteU16(smhd, 0);
                    WriteU16(smhd, 0);
                });
                WriteDinf(minf);
                WriteBox(minf, "stbl", stbl =>
                {
                    WriteBox(stbl, "stsd", stsd =>
                    {
                        WriteU32(stsd, 0);
                        WriteU32(stsd, 1);
                        WriteMp4a(stsd, sampleRate, channels, audioSpecificConfig);
                    });
                    WriteStts(stbl, samples.Count, 1024);
                    WriteStsc(stbl);
                    WriteStsz(stbl, samples);
                    WriteCo64(stbl, samples);
                });
            });
        });
    }

    private static void WriteAvc1(Stream stream, int width, int height, byte[] avcC)
    {
        WriteBox(stream, "avc1", entry =>
        {
            WriteZeros(entry, 6);
            WriteU16(entry, 1);
            WriteU16(entry, 0);
            WriteU16(entry, 0);
            WriteZeros(entry, 12);
            WriteU16(entry, checked((ushort)width));
            WriteU16(entry, checked((ushort)height));
            WriteU32(entry, 0x00480000);
            WriteU32(entry, 0x00480000);
            WriteU32(entry, 0);
            WriteU16(entry, 1);
            entry.WriteByte(0);
            WriteZeros(entry, 31);
            WriteU16(entry, 0x0018);
            WriteU16(entry, 0xFFFF);
            WriteBox(entry, "avcC", child => child.Write(avcC));
        });
    }

    private static void WriteMp4a(Stream stream, int sampleRate, int channels, byte[] audioSpecificConfig)
    {
        WriteBox(stream, "mp4a", entry =>
        {
            WriteZeros(entry, 6);
            WriteU16(entry, 1);
            WriteU16(entry, 0);
            WriteU16(entry, 0);
            WriteU32(entry, 0);
            WriteU16(entry, checked((ushort)channels));
            WriteU16(entry, 16);
            WriteU16(entry, 0);
            WriteU16(entry, 0);
            WriteU32(entry, checked((uint)sampleRate) << 16);
            WriteEsds(entry, sampleRate, channels, audioSpecificConfig);
        });
    }

    private static void WriteEsds(Stream stream, int sampleRate, int channels, byte[] asc)
    {
        _ = sampleRate;
        _ = channels;
        var bitrate = 160_000u;
        var decoderSpecific = Descriptor(0x05, asc);
        var decoderConfigBody = new MemoryStream();
        decoderConfigBody.WriteByte(0x40);
        decoderConfigBody.WriteByte(0x15);
        WriteZeros(decoderConfigBody, 3);
        WriteU32(decoderConfigBody, bitrate);
        WriteU32(decoderConfigBody, bitrate);
        decoderConfigBody.Write(decoderSpecific);
        var decoderConfig = Descriptor(0x04, decoderConfigBody.ToArray());
        var esBody = new MemoryStream();
        WriteU16(esBody, 0);
        esBody.WriteByte(0);
        esBody.Write(decoderConfig);
        esBody.Write(Descriptor(0x06, new byte[] { 0x02 }));

        WriteBox(stream, "esds", esds =>
        {
            WriteU32(esds, 0);
            esds.Write(Descriptor(0x03, esBody.ToArray()));
        });
    }

    private static byte[] Descriptor(byte tag, byte[] payload)
    {
        using var stream = new MemoryStream();
        stream.WriteByte(tag);
        WriteDescriptorLength(stream, payload.Length);
        stream.Write(payload);
        return stream.ToArray();
    }

    private static void WriteDescriptorLength(Stream stream, int length)
    {
        if (length < 0x80)
        {
            stream.WriteByte((byte)length);
            return;
        }

        Span<byte> bytes = stackalloc byte[4];
        var count = 0;
        var value = length;
        while (value > 0)
        {
            bytes[count++] = (byte)(value & 0x7F);
            value >>= 7;
        }

        for (var i = count - 1; i >= 0; i--)
        {
            var current = bytes[i];
            if (i != 0)
            {
                current |= 0x80;
            }

            stream.WriteByte(current);
        }
    }

    private static void WriteHdlr(Stream parent, string type, string name)
    {
        WriteBox(parent, "hdlr", hdlr =>
        {
            WriteU32(hdlr, 0);
            WriteU32(hdlr, 0);
            WriteFourCc(hdlr, type);
            WriteZeros(hdlr, 12);
            var bytes = Encoding.UTF8.GetBytes(name);
            hdlr.Write(bytes);
            hdlr.WriteByte(0);
        });
    }

    private static void WriteDinf(Stream parent)
    {
        WriteBox(parent, "dinf", dinf =>
        {
            WriteBox(dinf, "dref", dref =>
            {
                WriteU32(dref, 0);
                WriteU32(dref, 1);
                WriteBox(dref, "url ", url =>
                {
                    WriteU32(url, 1);
                });
            });
        });
    }

    private static void WriteStts(Stream parent, int sampleCount, int delta)
    {
        WriteBox(parent, "stts", stts =>
        {
            WriteU32(stts, 0);
            WriteU32(stts, sampleCount == 0 ? 0u : 1u);
            if (sampleCount > 0)
            {
                WriteU32(stts, checked((uint)sampleCount));
                WriteU32(stts, checked((uint)delta));
            }
        });
    }

    private static void WriteStss(Stream parent, IReadOnlyList<JournalSample> samples)
    {
        var keyframes = samples
            .Select((sample, index) => (sample, index))
            .Where(item => item.sample.Keyframe)
            .Select(item => item.index + 1)
            .ToList();
        WriteBox(parent, "stss", stss =>
        {
            WriteU32(stss, 0);
            WriteU32(stss, checked((uint)keyframes.Count));
            foreach (var keyframe in keyframes)
            {
                WriteU32(stss, checked((uint)keyframe));
            }
        });
    }

    private static void WriteStsc(Stream parent)
    {
        WriteBox(parent, "stsc", stsc =>
        {
            WriteU32(stsc, 0);
            WriteU32(stsc, 1);
            WriteU32(stsc, 1);
            WriteU32(stsc, 1);
            WriteU32(stsc, 1);
        });
    }

    private static void WriteStsz(Stream parent, IReadOnlyList<JournalSample> samples)
    {
        WriteBox(parent, "stsz", stsz =>
        {
            WriteU32(stsz, 0);
            WriteU32(stsz, 0);
            WriteU32(stsz, checked((uint)samples.Count));
            foreach (var sample in samples)
            {
                WriteU32(stsz, checked((uint)sample.Size));
            }
        });
    }

    private static void WriteCo64(Stream parent, IReadOnlyList<JournalSample> samples)
    {
        WriteBox(parent, "co64", co64 =>
        {
            WriteU32(co64, 0);
            WriteU32(co64, checked((uint)samples.Count));
            foreach (var sample in samples)
            {
                WriteU64(co64, checked((ulong)sample.Offset));
            }
        });
    }

    private static void WriteFtyp(Stream stream)
    {
        WriteBox(stream, "ftyp", ftyp =>
        {
            WriteFourCc(ftyp, "isom");
            WriteU32(ftyp, 0x200);
            WriteFourCc(ftyp, "isom");
            WriteFourCc(ftyp, "iso2");
            WriteFourCc(ftyp, "avc1");
            WriteFourCc(ftyp, "mp41");
        });
    }

    private static void WriteMatrix(Stream stream)
    {
        WriteU32(stream, 0x00010000);
        WriteU32(stream, 0);
        WriteU32(stream, 0);
        WriteU32(stream, 0);
        WriteU32(stream, 0x00010000);
        WriteU32(stream, 0);
        WriteU32(stream, 0);
        WriteU32(stream, 0);
        WriteU32(stream, 0x40000000);
    }

    private static void WriteBox(Stream stream, string type, Action<Stream> body)
    {
        var start = stream.Position;
        WriteU32(stream, 0);
        WriteFourCc(stream, type);
        body(stream);
        var end = stream.Position;
        var size = end - start;
        if (size > uint.MaxValue)
        {
            throw new ReplayException("E_OUTPUT_CORRUPT", $"MP4 box '{type}' exceeds the 32-bit box limit.");
        }

        stream.Position = start;
        WriteU32(stream, (uint)size);
        stream.Position = end;
    }

    private static void WriteZeros(Stream stream, int count)
    {
        Span<byte> zeros = stackalloc byte[32];
        while (count > 0)
        {
            var write = Math.Min(count, zeros.Length);
            stream.Write(zeros[..write]);
            count -= write;
        }
    }

    private static void WriteFourCc(Stream stream, string value)
    {
        if (value.Length != 4)
        {
            throw new ArgumentException("FourCC must contain four characters.", nameof(value));
        }

        stream.Write(Encoding.ASCII.GetBytes(value));
    }

    private static void WriteU16(Stream stream, ushort value)
    {
        Span<byte> bytes = stackalloc byte[2];
        BinaryPrimitives.WriteUInt16BigEndian(bytes, value);
        stream.Write(bytes);
    }

    private static void WriteU32(Stream stream, uint value)
    {
        Span<byte> bytes = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(bytes, value);
        stream.Write(bytes);
    }

    private static void WriteU64(Stream stream, ulong value)
    {
        Span<byte> bytes = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64BigEndian(bytes, value);
        stream.Write(bytes);
    }

    private static uint ToU32(double value)
    {
        if (value < 0 || value > uint.MaxValue)
        {
            throw new ReplayException("E_OUTPUT_CORRUPT", "MP4 duration exceeds the 32-bit limit.");
        }

        return (uint)Math.Round(value);
    }
}
