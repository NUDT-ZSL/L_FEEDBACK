using ReplayForge.Contracts;
using ReplayForge.Media;
using ReplayForge.Utilities;

namespace ReplayForge.Validation;

public static class OutputValidator
{
    public static ValidationReport Validate(
        string outputPath,
        int expectedWidth,
        int expectedHeight,
        int expectedFps,
        int expectedVideoSamples,
        int expectedAudioSamples,
        int expectedAudioSampleRate)
    {
        Mp4FileInfo info;
        try
        {
            info = Mp4Parser.Parse(outputPath);
        }
        catch (ReplayException ex)
        {
            throw Corrupt(outputPath, "MP4 parser rejected the file.", new { parserError = ex.Message, details = ex.Details });
        }

        var checks = new List<string>();
        if (info.MdatOffset < 0)
        {
            throw Corrupt(outputPath, "mdat box is missing.");
        }

        if (info.MoovOffset < 0)
        {
            throw Corrupt(outputPath, "moov box is missing.");
        }

        if (info.MdatDataOffset != Mp4Muxer.InitialMediaDataOffset)
        {
            throw Corrupt(outputPath, "mdat data offset does not match the incremental layout.",
                new { actual = info.MdatDataOffset, expected = Mp4Muxer.InitialMediaDataOffset });
        }

        if (info.MdatOffset + info.MdatSize != info.MoovOffset)
        {
            throw Corrupt(outputPath, "mdat declared size does not end at the moov box.",
                new { mdatOffset = info.MdatOffset, mdatSize = info.MdatSize, moovOffset = info.MoovOffset });
        }

        if (info.MoovOffset + info.MoovSize != info.FileLength)
        {
            throw Corrupt(outputPath, "moov does not end at the physical EOF.",
                new { moovEnd = info.MoovOffset + info.MoovSize, fileLength = info.FileLength });
        }

        if (info.MdatDataOffset + info.MdatDataLength != info.MoovOffset)
        {
            throw Corrupt(outputPath, "mdat data length is inconsistent with the moov offset.");
        }

        checks.Add("ftyp/mdat/moov top-level layout is consistent");
        checks.Add("mdat large-size and physical EOF are consistent");

        var videoTracks = info.Tracks.Where(track => track.Handler == "vide").ToList();
        var audioTracks = info.Tracks.Where(track => track.Handler == "soun").ToList();
        if (videoTracks.Count != 1 || audioTracks.Count != 1)
        {
            throw Corrupt(outputPath, "Output must contain exactly one video track and one audio track.",
                new { videoTracks = videoTracks.Count, audioTracks = audioTracks.Count });
        }

        var video = videoTracks[0];
        var audio = audioTracks[0];
        if (video.H264 is null || video.Codec is not ("avc1" or "avc3"))
        {
            throw Corrupt(outputPath, "Video track is not a valid H.264 track.");
        }

        if (video.Width != expectedWidth || video.Height != expectedHeight)
        {
            throw Corrupt(outputPath, "Video dimensions do not match the manifest.",
                new { actual = $"{video.Width}x{video.Height}", expected = $"{expectedWidth}x{expectedHeight}" });
        }

        if (video.Samples.Count != expectedVideoSamples)
        {
            throw Corrupt(outputPath, "Video sample count does not match the completed segments.",
                new { actual = video.Samples.Count, expected = expectedVideoSamples });
        }

        if (audio.Samples.Count != expectedAudioSamples)
        {
            throw Corrupt(outputPath, "Audio sample count does not match the completed segments.",
                new { actual = audio.Samples.Count, expected = expectedAudioSamples });
        }

        var expectedVideoDuration = expectedVideoSamples / (double)expectedFps;
        if (info.MovieTimescale == 0
            || Math.Abs(info.MovieDurationTicks / (double)info.MovieTimescale - expectedVideoDuration) > 1e-3)
        {
            throw Corrupt(outputPath, "Movie-header duration does not match the video timeline.",
                new
                {
                    info.MovieTimescale,
                    info.MovieDurationTicks,
                    expectedVideoDuration
                });
        }

        if (Math.Abs(video.DurationSeconds - expectedVideoDuration) > 1.0 / expectedFps + 1e-6)
        {
            throw Corrupt(outputPath, "Video track duration does not match its sample table.",
                new { actual = video.DurationSeconds, expected = expectedVideoDuration });
        }

        if (video.Timescale != (uint)expectedFps || video.DurationTicks != (ulong)expectedVideoSamples)
        {
            throw Corrupt(outputPath, "Video timescale or duration is inconsistent.",
                new { video.Timescale, video.DurationTicks, expectedFps, expectedVideoSamples });
        }

        if (audio.Timescale != (uint)expectedAudioSampleRate)
        {
            throw Corrupt(outputPath, "Audio timescale does not match the output sample rate.",
                new { audio.Timescale, expectedAudioSampleRate });
        }

        if (audio.Codec != "mp4a" || audio.AudioSpecificConfig is null
            || audio.AudioChannels <= 0 || audio.AudioSampleRate != expectedAudioSampleRate)
        {
            throw Corrupt(outputPath, "Audio track does not contain a valid AAC sample entry.",
                new
                {
                    audio.Codec,
                    audio.AudioSampleRate,
                    audio.AudioChannels,
                    hasAudioSpecificConfig = audio.AudioSpecificConfig is not null
                });
        }

        var expectedAudioDuration = expectedAudioSamples * 1024.0 / expectedAudioSampleRate;
        if (Math.Abs(audio.DurationSeconds - expectedAudioDuration) > 1e-6)
        {
            throw Corrupt(outputPath, "Audio track duration does not match its sample table.",
                new { actual = audio.DurationSeconds, expected = expectedAudioDuration });
        }

        if (audio.PresentationDurationSeconds is null
            || Math.Abs(audio.PresentationDurationSeconds.Value - expectedVideoDuration) > 1e-3)
        {
            throw Corrupt(outputPath, "Audio edit list does not present commentary exactly to video duration.",
                new
                {
                    presentationDuration = audio.PresentationDurationSeconds,
                    expectedVideoDuration
                });
        }

        checks.Add("track durations match sample tables");
        checks.Add("movie-header duration matches the exact video timeline");
        checks.Add("video fps, sample count, and dimensions match manifest");
        checks.Add("audio sample rate and sample count match output state");
        checks.Add("audio sample entry carries an AAC AudioSpecificConfig");
        checks.Add("audio edit list presents commentary exactly to video duration");

        ValidateOffsets(outputPath, info, video, "video");
        ValidateOffsets(outputPath, info, audio, "audio");
        checks.Add("all chunk offsets and sizes are inside mdat");

        if (video.Samples.Count == 0 || !video.Samples[0].IsKeyframe)
        {
            throw Corrupt(outputPath, "Video track does not start with a keyframe.");
        }

        ValidateKeyframes(outputPath, info, video);
        checks.Add("keyframe sample offsets point to valid IDR access units");

        return new ValidationReport
        {
            Ok = true,
            FileLength = info.FileLength,
            MoovOffset = info.MoovOffset,
            MdatDataLength = info.MdatDataLength,
            VideoTrackDurationSeconds = video.DurationSeconds,
            AudioTrackDurationSeconds = audio.PresentationDurationSeconds ?? audio.DurationSeconds,
            VideoSamples = video.Samples.Count,
            AudioSamples = audio.Samples.Count,
            Keyframes = video.Samples.Count(sample => sample.IsKeyframe),
            Checks = checks
        };
    }

    private static void ValidateOffsets(string outputPath, Mp4FileInfo info, Mp4Track track, string name)
    {
        var dataEnd = info.MdatDataOffset + info.MdatDataLength;
        foreach (var sample in track.Samples)
        {
            if (sample.Size <= 0 || sample.Offset < info.MdatDataOffset
                || sample.Offset + sample.Size > dataEnd)
            {
                throw Corrupt(outputPath, $"{name} sample lies outside mdat.",
                    new { sample = sample.Index, sample.Offset, sample.Size, mdatStart = info.MdatDataOffset, dataEnd });
            }
        }
    }

    private static void ValidateKeyframes(string outputPath, Mp4FileInfo info, Mp4Track video)
    {
        using var stream = new FileStream(outputPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        var keyframes = video.Samples.Where(sample => sample.IsKeyframe).ToList();
        if (keyframes.Count == 0)
        {
            throw Corrupt(outputPath, "Video track has no keyframes.");
        }

        foreach (var sample in keyframes)
        {
            var bytes = new byte[sample.Size];
            stream.Position = sample.Offset;
            stream.ReadExactly(bytes);
            var nals = H264Inspector.SplitLengthPrefixed(bytes, video.H264!.LengthSize);
            if (!nals.Any(nal => (nal[0] & 0x1F) == 5))
            {
                throw Corrupt(outputPath, "A sync sample does not contain an IDR NAL.",
                    new { sample = sample.Index, sample.Offset });
            }
        }
    }

    private static ReplayException Corrupt(string path, string message, object? details = null)
        => new("E_OUTPUT_CORRUPT", message, new { path, details });
}
