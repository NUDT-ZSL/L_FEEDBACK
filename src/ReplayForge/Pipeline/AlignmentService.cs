using ReplayForge.Contracts;
using ReplayForge.Media;
using ReplayForge.Utilities;

namespace ReplayForge.Pipeline;

public sealed class ValidatedVideoLayer
{
    public required VideoLayerConfig Config { get; init; }

    public required string ResolutionPath { get; init; }

    public required Mp4Track Track { get; init; }

    public required double SourceFirstPtsSeconds { get; init; }

    public required double ExpectedFirstPtsSeconds { get; init; }

    public required double FirstPtsErrorSeconds { get; init; }

    public required double MappedStartSeconds { get; init; }

    public required double SourceFps { get; init; }

    public required int SourceFrameCountInWindow { get; init; }

    public required int EmittedFrames { get; init; }

    public required int DroppedFrames { get; init; }

    public required int InterpolatedFrames { get; init; }
}

public sealed class AlignmentResult
{
    public required int TotalOutputFrames { get; init; }

    public required double ExactDurationSeconds { get; init; }

    public required List<ValidatedVideoLayer> Layers { get; init; }

    public required AlignmentReport Report { get; init; }

    public required ResamplingReport Resampling { get; init; }
}

public static class AlignmentService
{
    public static AlignmentResult Align(Manifest manifest, string manifestDirectory)
    {
        ValidateManifest(manifest, manifestDirectory);
        var timeline = manifest.Timeline;
        var totalFrames = checked((int)Math.Round(
            timeline.DurationSeconds * timeline.OutputFps,
            MidpointRounding.AwayFromZero));
        if (totalFrames <= 0)
        {
            throw new ReplayException("E_MANIFEST_INVALID", "Timeline produces zero output frames.");
        }

        var exactDuration = totalFrames / (double)timeline.OutputFps;
        var layers = new List<ValidatedVideoLayer>();
        var alignmentLayers = new List<LayerAlignmentReport>();
        var resamplingLayers = new List<LayerResamplingReport>();

        foreach (var layer in manifest.Videos)
        {
            var fullPath = PathUtil.Resolve(manifestDirectory, layer.Path);
            Mp4FileInfo mp4;
            try
            {
                mp4 = Mp4Parser.Parse(fullPath);
            }
            catch (ReplayException ex)
            {
                throw AddLayer(ex, layer.Id, fullPath);
            }

            var track = mp4.VideoTrack;
            if (track is null)
            {
                throw new ReplayException("E_INVALID_STREAM",
                    "No MP4 video track was found.",
                    new { layer = layer.Id, path = fullPath });
            }

            if (track.Samples.Count == 0)
            {
                throw new ReplayException("E_INVALID_STREAM",
                    "MP4 video track contains no samples.",
                    new { layer = layer.Id, path = fullPath });
            }

            if (track.H264 is null)
            {
                throw new ReplayException("E_INVALID_STREAM",
                    "Video track is not a supported H.264 stream.",
                    new { layer = layer.Id, path = fullPath, codec = track.Codec });
            }

            var firstMediaPts = track.Samples.Min(sample => sample.PtsTicks / (double)track.Timescale);
            var sourceFirst = firstMediaPts + track.PresentationShiftSeconds;
            var expectedFirst = timeline.MasterStartPtsSeconds + layer.TimelineStartSeconds;
            var firstError = sourceFirst - expectedFirst;
            if (Math.Abs(firstError) > timeline.SyncToleranceSeconds + 1e-9)
            {
                throw new ReplayException("E_SYNC_FAILED",
                    $"Layer '{layer.Id}' first PTS is outside the synchronization tolerance.",
                    new
                    {
                        layer = layer.Id,
                        path = fullPath,
                        actualStartPtsSeconds = sourceFirst,
                        expectedStartPtsSeconds = expectedFirst,
                        toleranceSeconds = timeline.SyncToleranceSeconds,
                        errorSeconds = firstError
                    });
            }

            var mappedStart = Math.Round(
                layer.TimelineStartSeconds * timeline.OutputFps,
                MidpointRounding.AwayFromZero) / timeline.OutputFps;
            var sourceLast = track.Samples.Max(sample =>
                (sample.PtsTicks + sample.DurationTicks) / (double)track.Timescale)
                + track.PresentationShiftSeconds;
            var mappedLast = mappedStart + (sourceLast - sourceFirst);
            if (mappedLast + 1e-6 < exactDuration)
            {
                throw new ReplayException("E_SYNC_FAILED",
                    $"Layer '{layer.Id}' ends before the requested timeline duration.",
                    new
                    {
                        layer = layer.Id,
                        path = fullPath,
                        sourceFirstPtsSeconds = sourceFirst,
                        sourceLastPtsSeconds = sourceLast,
                        mappedLastTimelineSeconds = mappedLast,
                        requiredTimelineSeconds = exactDuration
                    });
            }

            var sourceSamplesInWindow = track.Samples.Count(sample =>
            {
                var mapped = mappedStart
                             + (sample.PtsTicks / (double)track.Timescale - firstMediaPts);
                return mapped >= mappedStart - 1e-7 && mapped < exactDuration - 1e-7;
            });
            var emittedFrames = CountFramesInWindow(mappedStart, exactDuration, timeline.OutputFps);
            var dropped = Math.Max(0, sourceSamplesInWindow - emittedFrames);
            var interpolated = Math.Max(0, emittedFrames - sourceSamplesInWindow);
            var sourceFps = track.AverageFrameRate;

            layers.Add(new ValidatedVideoLayer
            {
                Config = layer,
                ResolutionPath = fullPath,
                Track = track,
                SourceFirstPtsSeconds = sourceFirst,
                ExpectedFirstPtsSeconds = expectedFirst,
                FirstPtsErrorSeconds = firstError,
                MappedStartSeconds = mappedStart,
                SourceFps = sourceFps,
                SourceFrameCountInWindow = sourceSamplesInWindow,
                EmittedFrames = emittedFrames,
                DroppedFrames = dropped,
                InterpolatedFrames = interpolated
            });

            alignmentLayers.Add(new LayerAlignmentReport
            {
                Id = layer.Id,
                Path = PathUtil.ToDisplay(manifestDirectory, fullPath),
                SourceFirstPtsSeconds = sourceFirst,
                ExpectedFirstPtsSeconds = expectedFirst,
                FirstPtsErrorSeconds = firstError,
                MappedFirstFrameTimelineSeconds = mappedStart,
                SourceFps = sourceFps,
                SourceFrameCount = track.Samples.Count,
                SnappedToFrame = Math.Abs(firstError) > 1e-7
            });

            resamplingLayers.Add(new LayerResamplingReport
            {
                Id = layer.Id,
                SourceFramesInWindow = sourceSamplesInWindow,
                EmittedFrames = emittedFrames,
                DroppedFrames = dropped,
                InterpolatedFrames = interpolated
            });
        }

        return new AlignmentResult
        {
            TotalOutputFrames = totalFrames,
            ExactDurationSeconds = exactDuration,
            Layers = layers,
            Report = new AlignmentReport
            {
                MasterStartPtsSeconds = timeline.MasterStartPtsSeconds,
                ToleranceSeconds = timeline.SyncToleranceSeconds,
                Layers = alignmentLayers
            },
            Resampling = new ResamplingReport
            {
                OutputFps = timeline.OutputFps,
                CumulativeOutputFrames = totalFrames,
                Layers = resamplingLayers
            }
        };
    }

    public static int CountFramesInWindow(double startSeconds, double endSeconds, int fps)
    {
        var first = (int)Math.Ceiling(startSeconds * fps - 1e-9);
        var lastExclusive = (int)Math.Ceiling(endSeconds * fps - 1e-9);
        return Math.Max(0, lastExclusive - first);
    }

    public static void ValidateManifest(Manifest manifest, string manifestDirectory)
    {
        if (manifest.SchemaVersion != 1)
        {
            throw new ReplayException("E_MANIFEST_INVALID",
                $"Unsupported manifest schemaVersion {manifest.SchemaVersion}.");
        }

        var timeline = manifest.Timeline;
        if (timeline.OutputFps <= 0 || timeline.Width <= 0 || timeline.Height <= 0
            || timeline.Width % 2 != 0 || timeline.Height % 2 != 0)
        {
            throw new ReplayException("E_MANIFEST_INVALID",
                "Output fps and dimensions must be positive; dimensions must be even.");
        }

        if (timeline.DurationSeconds <= 0 || timeline.SegmentDurationSeconds <= 0)
        {
            throw new ReplayException("E_MANIFEST_INVALID",
                "Timeline duration and segment duration must be positive.");
        }

        if (timeline.SyncToleranceSeconds < 0)
        {
            throw new ReplayException("E_MANIFEST_INVALID", "Synchronization tolerance cannot be negative.");
        }

        if (manifest.Audio.TimelineStartSeconds < 0)
        {
            throw new ReplayException("E_MANIFEST_INVALID", "Audio timeline start cannot be negative.");
        }

        if (manifest.Videos.Count == 0)
        {
            throw new ReplayException("E_MANIFEST_INVALID", "Manifest must contain at least one video layer.");
        }

        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var layer in manifest.Videos)
        {
            if (string.IsNullOrWhiteSpace(layer.Id) || !ids.Add(layer.Id))
            {
                throw new ReplayException("E_MANIFEST_INVALID", "Video layer ids must be unique and non-empty.");
            }

            if (string.IsNullOrWhiteSpace(layer.Path))
            {
                throw new ReplayException("E_MANIFEST_INVALID",
                    $"Video layer '{layer.Id}' has no path.");
            }

            if (layer.TimelineStartSeconds < 0)
            {
                throw new ReplayException("E_MANIFEST_INVALID",
                    $"Video layer '{layer.Id}' has a negative timeline start.");
            }

            var layout = layer.Layout;
            if (layout.Width <= 0 || layout.Height <= 0
                || layout.X < 0 || layout.Y < 0
                || layout.X + layout.Width > timeline.Width
                || layout.Y + layout.Height > timeline.Height)
            {
                throw new ReplayException("E_MANIFEST_INVALID",
                    $"Video layer '{layer.Id}' layout is outside the output canvas.");
            }

            if (layout.Opacity is < 0 or > 1)
            {
                throw new ReplayException("E_MANIFEST_INVALID",
                    $"Video layer '{layer.Id}' opacity must be between 0 and 1.");
            }

            var path = PathUtil.Resolve(manifestDirectory, layer.Path);
            if (!File.Exists(path))
            {
                throw new ReplayException("E_INVALID_STREAM", "Video layer file does not exist.",
                    new { layer = layer.Id, path });
            }
        }

        if (string.IsNullOrWhiteSpace(manifest.Audio.Path))
        {
            throw new ReplayException("E_MANIFEST_INVALID", "Manifest has no commentary audio path.");
        }

        if (!File.Exists(PathUtil.Resolve(manifestDirectory, manifest.Audio.Path)))
        {
            throw new ReplayException("E_INVALID_STREAM", "Commentary audio file does not exist.",
                new { layer = manifest.Audio.Id, path = PathUtil.Resolve(manifestDirectory, manifest.Audio.Path) });
        }
    }

    private static ReplayException AddLayer(ReplayException exception, string layer, string path)
        => new(exception.Code, exception.Message, new
        {
            layer,
            path,
            details = exception.Details
        });
}
