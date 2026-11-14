using System.Globalization;
using ReplayForge.Contracts;
using ReplayForge.Media;
using ReplayForge.State;
using ReplayForge.Utilities;
using ReplayForge.Validation;

namespace ReplayForge.Pipeline;

public sealed class RenderPipeline
{
    private readonly Ffmpeg _ffmpeg;

    public RenderPipeline(Ffmpeg ffmpeg)
    {
        _ffmpeg = ffmpeg;
    }

    public async Task<CommandReport> RunAsync(
        Manifest manifest,
        string manifestPath,
        bool forceRestart,
        CancellationToken cancellationToken)
    {
        var manifestDirectory = Path.GetDirectoryName(Path.GetFullPath(manifestPath))!;
        var outputPath = PathUtil.Resolve(manifestDirectory, manifest.OutputPath);
        var workDirectory = PathUtil.Resolve(manifestDirectory, manifest.WorkDirectory);
        var alignment = AlignmentService.Align(manifest, manifestDirectory);
        var timeline = manifest.Timeline;
        var segmentFrameCount = Math.Max(
            1,
            (int)Math.Round(timeline.SegmentDurationSeconds * timeline.OutputFps, MidpointRounding.AwayFromZero));
        var totalSegments = (alignment.TotalOutputFrames + segmentFrameCount - 1) / segmentFrameCount;
        var inputVersions = BuildInputVersions(manifest, manifestDirectory);
        var manifestHash = ResumeStore.HashFile(manifestPath);
        var warnings = new List<string>();
        var recoveryLogs = new List<string>();

        using var store = ResumeStore.Open(workDirectory);
        ResumeSnapshot? snapshot = null;
        List<SegmentJournal> journals = new();
        var recoveryMode = "fresh";
        var previousCleanShutdown = (bool?)null;
        var rolledBackBytes = 0L;
        var resumedFromSegment = 0;
        var hadExistingSnapshot = false;

        if (forceRestart)
        {
            JsonIO.TryDelete(outputPath);
            store.DiscardAll();
            recoveryLogs.Add("force-restart: discarded persistent state and output");
            recoveryMode = "force-restart";
        }

        try
        {
            snapshot = store.LoadSnapshot();
        }
        catch (ReplayException ex)
        {
            recoveryLogs.Add($"discarded malformed snapshot: {ex.Message}");
            JsonIO.TryDelete(outputPath);
            store.DiscardAll();
            snapshot = null;
            recoveryMode = "rebuilt-corrupt-state";
        }

        if (snapshot is not null)
        {
            previousCleanShutdown = snapshot.CleanShutdown;
            if (!ManifestMatches(snapshot, manifestHash, inputVersions, totalSegments))
            {
                recoveryLogs.Add("manifest or input hash changed; discarded snapshot and output");
                JsonIO.TryDelete(outputPath);
                store.DiscardAll();
                snapshot = null;
                recoveryMode = "rebuilt-input-changed";
            }
            else
            {
                hadExistingSnapshot = true;
            }
        }

        if (snapshot is null)
        {
            Mp4Muxer.InitializeOutput(outputPath);
            snapshot = CreateSnapshot(manifestHash, inputVersions, totalSegments, alignment.TotalOutputFrames);
            store.WriteSnapshot(snapshot);
            recoveryLogs.Add($"initialized output at {snapshot.SafeOutputLength} bytes");
            if (recoveryMode == "fresh")
            {
                recoveryLogs.Add("fresh run: no resume snapshot existed");
            }
        }

        try
        {
            if (snapshot.Status == "complete" && snapshot.CleanShutdown)
            {
                try
                {
                    var validation = OutputValidator.Validate(
                        outputPath,
                        timeline.Width,
                        timeline.Height,
                        timeline.OutputFps,
                        alignment.TotalOutputFrames,
                        snapshot.AudioSampleCount,
                        manifest.Audio.SampleRate);
                    recoveryLogs.Add("normal exit path: output was already complete and valid");
                    return BuildSuccessReport(
                        alignment,
                        outputPath,
                        validation,
                        null,
                        null,
                        new RecoveryReport
                        {
                            Mode = "normal-already-complete",
                            PreviousCleanShutdown = true,
                            ResumedFromSegment = snapshot.CompletedSegments,
                            Logs = recoveryLogs
                        },
                        warnings);
                }
                catch (ReplayException ex)
                {
                    recoveryLogs.Add("previously complete output failed validation");
                    throw new ReplayException("E_OUTPUT_CORRUPT",
                        "Previously complete output failed integrity validation.",
                        new
                        {
                            path = outputPath,
                            workDirectory,
                            validationError = ex.Message,
                            details = ex.Details,
                            recoveryLogs
                        });
                }
            }

            if (snapshot is null)
            {
                Mp4Muxer.InitializeOutput(outputPath);
                snapshot = CreateSnapshot(manifestHash, inputVersions, totalSegments, alignment.TotalOutputFrames);
                store.WriteSnapshot(snapshot);
            }

            if (snapshot.TotalSegments != totalSegments)
            {
                recoveryLogs.Add("segment count changed; restarting from segment 0");
                JsonIO.TryDelete(outputPath);
                store.DiscardAll();
                Mp4Muxer.InitializeOutput(outputPath);
                snapshot = CreateSnapshot(manifestHash, inputVersions, totalSegments, alignment.TotalOutputFrames);
                store.WriteSnapshot(snapshot);
                recoveryMode = "rebuilt-segment-plan";
            }

            if (!File.Exists(outputPath) || new FileInfo(outputPath).Length < snapshot.SafeOutputLength)
            {
                recoveryLogs.Add("output is shorter than the safe point; restarting from segment 0");
                JsonIO.TryDelete(outputPath);
                store.DiscardAll();
                Mp4Muxer.InitializeOutput(outputPath);
                snapshot = CreateSnapshot(manifestHash, inputVersions, totalSegments, alignment.TotalOutputFrames);
                store.WriteSnapshot(snapshot);
                recoveryMode = "rebuilt-missing-output";
            }

            if (hadExistingSnapshot && !snapshot.CleanShutdown)
            {
                recoveryMode = "forced-termination-recovery";
                recoveryLogs.Add(
                    $"forced termination detected: previous process did not report a clean exit (safe segment {snapshot.CompletedSegments})");
            }
            else if (hadExistingSnapshot && snapshot.CleanShutdown)
            {
                recoveryMode = "normal-resume";
                recoveryLogs.Add($"normal resume from segment {snapshot.CompletedSegments}");
            }

            journals = store.LoadJournals(snapshot.CompletedSegments);
            ValidateJournalBounds(journals, snapshot.SafeOutputLength);
            store.DiscardJournalsFrom(snapshot.CompletedSegments);
            store.ClearChunks();
            var actualLength = new FileInfo(outputPath).Length;
            if (actualLength > snapshot.SafeOutputLength)
            {
                rolledBackBytes = actualLength - snapshot.SafeOutputLength;
                using var output = new FileStream(outputPath, FileMode.Open, FileAccess.Write, FileShare.Read);
                output.SetLength(snapshot.SafeOutputLength);
                output.Flush(flushToDisk: true);
                recoveryLogs.Add(
                    $"rolled back {rolledBackBytes} bytes beyond the last durable snapshot");
            }

            resumedFromSegment = snapshot.CompletedSegments;
            recoveryLogs.Add($"resuming at segment {snapshot.CompletedSegments}/{totalSegments}");
            snapshot.CleanShutdown = false;
            snapshot.Status = "rendering";
            snapshot.LastMessage = null;
            store.WriteSnapshot(snapshot);

            var audio = await AudioNormalizer.NormalizeAsync(
                    manifest,
                    manifestDirectory,
                    _ffmpeg,
                    workDirectory,
                    inputVersions[Path.GetFullPath(PathUtil.Resolve(manifestDirectory, manifest.Audio.Path))],
                    cancellationToken)
                .ConfigureAwait(false);
            var normalizedAudioHash = ResumeStore.HashFile(audio.Path);
            if (snapshot.NextAudioFrame > 0 && snapshot.NormalizedAudioSha256 != normalizedAudioHash)
            {
                recoveryLogs.Add("normalized audio changed; restarting from segment 0");
                JsonIO.TryDelete(outputPath);
                store.DiscardRenderStateKeepingAudio();
                Mp4Muxer.InitializeOutput(outputPath);
                snapshot = CreateSnapshot(manifestHash, inputVersions, totalSegments, alignment.TotalOutputFrames);
                snapshot.NormalizedAudioSha256 = normalizedAudioHash;
                store.WriteSnapshot(snapshot);
                journals = new List<SegmentJournal>();
                resumedFromSegment = 0;
                recoveryMode = "rebuilt-audio-changed";
            }
            else if (snapshot.NormalizedAudioSha256 is null)
            {
                snapshot.NormalizedAudioSha256 = normalizedAudioHash;
                store.WriteSnapshot(snapshot);
            }

            if (audio.Info.SampleRate != 48000 || audio.Info.Channels != manifest.Audio.Channels)
            {
                throw new ReplayException("E_INVALID_STREAM",
                    "Normalized commentary audio does not preserve the requested 48 kHz channel layout.",
                    new { sampleRate = audio.Info.SampleRate, channels = audio.Info.Channels });
            }

            var audioStarts = BuildAudioStarts(audio.Info);
            if (snapshot.NextAudioFrame > audio.Info.Frames.Count)
            {
                throw new ReplayException("E_STATE_INVALID", "Snapshot audio cursor exceeds audio frame count.");
            }

            CodecJournal? codec = journals.Count > 0 ? journals[0].Codec : null;
            if (codec is null && snapshot.CompletedSegments > 0)
            {
                throw new ReplayException("E_STATE_INVALID", "Resume state has no codec journal.");
            }

            using (var audioStream = new FileStream(
                       audio.Path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024))
            {
                using (var outputStream = new FileStream(
                           outputPath, FileMode.Open, FileAccess.ReadWrite, FileShare.Read, 1024 * 1024))
                {
                    for (var segmentIndex = snapshot.CompletedSegments; segmentIndex < totalSegments; segmentIndex++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var firstFrame = segmentIndex * segmentFrameCount;
                        var frameCount = Math.Min(segmentFrameCount, alignment.TotalOutputFrames - firstFrame);
                        var startSeconds = firstFrame / (double)timeline.OutputFps;
                        var endSeconds = (firstFrame + frameCount) / (double)timeline.OutputFps;
                        var chunkPath = store.ChunkPath(segmentIndex);
                        JsonIO.TryDelete(chunkPath);
                        var arguments = VideoFilterGraph.BuildSegmentCommand(
                            manifest,
                            alignment.Layers,
                            firstFrame,
                            frameCount,
                            chunkPath);
                        await _ffmpeg.RunCheckedAsync(
                                arguments,
                                $"render segment {segmentIndex}",
                                workDirectory,
                                cancellationToken)
                            .ConfigureAwait(false);
                        var encoded = H264SegmentReader.Read(chunkPath, frameCount);
                        if (codec is null)
                        {
                            codec = new CodecJournal
                            {
                                AvcCBase64 = Convert.ToBase64String(encoded.AvcC),
                                Width = timeline.Width,
                                Height = timeline.Height,
                                AudioSpecificConfigBase64 = Convert.ToBase64String(audio.Info.AudioSpecificConfig),
                                AudioSampleRate = audio.Info.SampleRate,
                                AudioChannels = audio.Info.Channels
                            };
                        }
                        else if (Convert.ToBase64String(encoded.AvcC) != codec.AvcCBase64)
                        {
                            throw new ReplayException("E_INVALID_STREAM",
                                "H.264 parameter sets changed between GOP segments.",
                                new { segment = segmentIndex });
                        }

                        outputStream.Position = snapshot.SafeOutputLength;
                        outputStream.SetLength(snapshot.SafeOutputLength);
                        var journal = new SegmentJournal
                        {
                            Index = segmentIndex,
                            FirstFrame = firstFrame,
                            FrameCount = frameCount,
                            StartSeconds = startSeconds,
                            EndSeconds = endSeconds,
                            OutputStart = outputStream.Position,
                            Codec = segmentIndex == 0 ? codec : null
                        };
                        var nextAudioFrame = snapshot.NextAudioFrame;

                        for (var frame = 0; frame < frameCount; frame++)
                        {
                            var videoTime = (firstFrame + frame) / (double)timeline.OutputFps;
                            while (nextAudioFrame < audio.Info.Frames.Count
                                   && audioStarts[nextAudioFrame] < videoTime - 1e-9)
                            {
                                AppendAudioFrame(
                                    audioStream,
                                    outputStream,
                                    audio.Info.Frames[nextAudioFrame],
                                    journal.AudioSamples,
                                    ref nextAudioFrame);
                            }

                            var sampleOffset = outputStream.Position;
                            var sample = encoded.AccessUnits[frame];
                            outputStream.Write(sample.Data);
                            journal.VideoSamples.Add(new JournalSample
                            {
                                Offset = sampleOffset,
                                Size = sample.Data.Length,
                                Keyframe = sample.Keyframe
                            });
                        }

                        while (nextAudioFrame < audio.Info.Frames.Count
                               && audioStarts[nextAudioFrame] < endSeconds - 1e-9)
                        {
                            AppendAudioFrame(
                                audioStream,
                                outputStream,
                                audio.Info.Frames[nextAudioFrame],
                                journal.AudioSamples,
                                ref nextAudioFrame);
                        }

                        outputStream.Flush(flushToDisk: true);
                        journal.OutputEnd = outputStream.Position;
                        store.WriteJournal(journal);
                        snapshot.CompletedSegments = segmentIndex + 1;
                        snapshot.SafeOutputLength = journal.OutputEnd;
                        snapshot.NextAudioFrame = nextAudioFrame;
                        snapshot.VideoSampleCount += journal.VideoSamples.Count;
                        snapshot.AudioSampleCount += journal.AudioSamples.Count;
                        snapshot.NormalizedAudioSha256 = normalizedAudioHash;
                        snapshot.CleanShutdown = false;
                        snapshot.Status = "rendering";
                        snapshot.LastMessage = $"segment {segmentIndex + 1}/{totalSegments} committed";
                        store.WriteSnapshot(snapshot);
                        journals.Add(journal);
                        JsonIO.TryDelete(chunkPath);
                        Console.Error.WriteLine(
                            string.Create(
                                CultureInfo.InvariantCulture,
                                $"[render] segment {segmentIndex + 1}/{totalSegments} committed, safe bytes={snapshot.SafeOutputLength}"));
                    }

                    outputStream.Flush(flushToDisk: true);
                    outputStream.SetLength(snapshot.SafeOutputLength);
                }
            }

            snapshot.Status = "finalizing";
            snapshot.CleanShutdown = false;
            snapshot.LastMessage = "writing moov";
            store.WriteSnapshot(snapshot);

            var allVideoSamples = journals.SelectMany(journal => journal.VideoSamples).ToList();
            var allAudioSamples = journals.SelectMany(journal => journal.AudioSamples).ToList();
            if (allVideoSamples.Count != alignment.TotalOutputFrames
                || allAudioSamples.Count != snapshot.AudioSampleCount
                || journals.Count != totalSegments)
            {
                throw new ReplayException("E_STATE_INVALID",
                    "Completed journals do not add up to the expected output sample tables.",
                    new
                    {
                        videoSamples = allVideoSamples.Count,
                        expectedVideoSamples = alignment.TotalOutputFrames,
                        audioSamples = allAudioSamples.Count,
                        expectedAudioSamples = snapshot.AudioSampleCount,
                        journals = journals.Count,
                        totalSegments
                    });
            }

            var codecState = journals[0].Codec
                             ?? throw new ReplayException("E_STATE_INVALID", "First journal has no codec state.");
            Mp4Muxer.FinalizeOutput(
                outputPath,
                timeline.Width,
                timeline.Height,
                timeline.OutputFps,
                Convert.FromBase64String(codecState.AvcCBase64),
                codecState.AudioSampleRate,
                codecState.AudioChannels,
                Convert.FromBase64String(codecState.AudioSpecificConfigBase64),
                allVideoSamples,
                allAudioSamples);

            var outputValidation = OutputValidator.Validate(
                outputPath,
                timeline.Width,
                timeline.Height,
                timeline.OutputFps,
                alignment.TotalOutputFrames,
                allAudioSamples.Count,
                codecState.AudioSampleRate);

            snapshot.Status = "complete";
            snapshot.CleanShutdown = true;
            snapshot.SafeOutputLength = new FileInfo(outputPath).Length;
            snapshot.LastMessage = "render complete";
            store.WriteSnapshot(snapshot);
            recoveryLogs.Add("normal exit: moov written and integrity validation passed");
            if (rolledBackBytes > 0)
            {
                recoveryLogs.Add($"forced-termination rollback removed {rolledBackBytes} partial bytes");
            }

            if (Math.Abs(audio.AppliedGainDb - AudioLoudness.GainForTarget(audio.Measured, timeline.TargetLoudnessLufs)) > 0.01)
            {
                warnings.Add("Audio gain was peak-limited and may not exactly hit the target loudness.");
            }

            return BuildSuccessReport(
                alignment,
                outputPath,
                outputValidation,
                audio,
                snapshot,
                new RecoveryReport
                {
                    Mode = recoveryMode,
                    PreviousCleanShutdown = previousCleanShutdown,
                    ResumedFromSegment = journals.Count == totalSegments && recoveryMode == "fresh"
                        ? 0
                        : resumedFromSegment,
                    RolledBackBytes = rolledBackBytes,
                    DiscardedPartialSegments = rolledBackBytes > 0 ? 1 : 0,
                    Logs = recoveryLogs
                },
                warnings);
        }
        catch (Exception ex)
        {
            if (snapshot is not null)
            {
                try
                {
                    snapshot.CleanShutdown = true;
                    snapshot.Status = ex is ReplayException replay && replay.Code == "E_OUTPUT_CORRUPT"
                        ? "corrupt"
                        : "paused";
                    snapshot.LastMessage = ex.Message;
                    store.WriteSnapshot(snapshot);
                }
                catch
                {
                    // Preserve the last valid durable snapshot if the error path itself fails.
                }
            }

            throw;
        }
    }

    private static double[] BuildAudioStarts(AdtsAudioInfo audio)
    {
        var starts = new double[audio.Frames.Count];
        long samples = 0;
        for (var i = 0; i < audio.Frames.Count; i++)
        {
            starts[i] = samples / (double)audio.SampleRate;
            samples += audio.Frames[i].SampleCount;
        }

        return starts;
    }

    private static void AppendAudioFrame(
        FileStream audioStream,
        FileStream outputStream,
        AdtsFrame frame,
        List<JournalSample> audioSamples,
        ref int nextAudioFrame)
    {
        var raw = new byte[frame.RawLength];
        audioStream.Position = frame.Offset + frame.HeaderLength;
        audioStream.ReadExactly(raw);
        var offset = outputStream.Position;
        outputStream.Write(raw);
        audioSamples.Add(new JournalSample
        {
            Offset = offset,
            Size = raw.Length,
            Keyframe = false
        });
        nextAudioFrame++;
    }

    private static void ValidateJournalBounds(List<SegmentJournal> journals, long safeLength)
    {
        foreach (var journal in journals)
        {
            if (journal.OutputStart < Mp4Muxer.InitialMediaDataOffset
                || journal.OutputEnd > safeLength
                || journal.OutputStart > journal.OutputEnd)
            {
                throw new ReplayException("E_STATE_INVALID",
                    $"Journal {journal.Index} points outside the durable output range.");
            }

            foreach (var sample in journal.VideoSamples.Concat(journal.AudioSamples))
            {
                if (sample.Offset < Mp4Muxer.InitialMediaDataOffset
                    || sample.Offset + sample.Size > safeLength)
                {
                    throw new ReplayException("E_STATE_INVALID",
                        $"Journal {journal.Index} contains a sample outside the durable output range.");
                }
            }
        }
    }

    private static CommandReport BuildSuccessReport(
        AlignmentResult alignment,
        string outputPath,
        ValidationReport validation,
        NormalizedAudio? audio,
        ResumeSnapshot? snapshot,
        RecoveryReport recovery,
        List<string> warnings)
    {
        return new CommandReport
        {
            Ok = true,
            Command = "run",
            OutputPath = outputPath,
            TotalFrames = alignment.TotalOutputFrames,
            TotalDurationSeconds = alignment.ExactDurationSeconds,
            Alignment = alignment.Report,
            Resampling = alignment.Resampling,
            Audio = audio is null
                ? null
                : new AudioReport
                {
                    SourceSampleRate = audio.Report.SourceSampleRate,
                    OutputSampleRate = audio.Report.OutputSampleRate,
                    Channels = audio.Report.Channels,
                    MeasuredLoudnessLufs = audio.Report.MeasuredLoudnessLufs,
                    TargetLoudnessLufs = audio.Report.TargetLoudnessLufs,
                    AppliedGainDb = audio.Report.AppliedGainDb,
                    SampleFrames = snapshot?.AudioSampleCount ?? audio.Report.SampleFrames
                },
            Recovery = recovery,
            Validation = validation,
            Warnings = warnings.Count == 0 ? null : warnings
        };
    }

    private static ResumeSnapshot CreateSnapshot(
        string manifestHash,
        Dictionary<string, string> inputVersions,
        int totalSegments,
        int totalFrames)
    {
        _ = totalFrames;
        return new ResumeSnapshot
        {
            ManifestSha256 = manifestHash,
            InputVersions = new Dictionary<string, string>(inputVersions, StringComparer.Ordinal),
            TotalSegments = totalSegments,
            CompletedSegments = 0,
            SafeOutputLength = Mp4Muxer.InitialMediaDataOffset,
            NextAudioFrame = 0,
            VideoSampleCount = 0,
            AudioSampleCount = 0,
            Status = "rendering",
            CleanShutdown = false
        };
    }

    private static Dictionary<string, string> BuildInputVersions(Manifest manifest, string manifestDirectory)
    {
        var versions = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var layer in manifest.Videos)
        {
            var path = Path.GetFullPath(PathUtil.Resolve(manifestDirectory, layer.Path));
            versions[path] = ResumeStore.HashFile(path);
        }

        var audioPath = Path.GetFullPath(PathUtil.Resolve(manifestDirectory, manifest.Audio.Path));
        versions[audioPath] = ResumeStore.HashFile(audioPath);
        return versions;
    }

    private static bool ManifestMatches(
        ResumeSnapshot snapshot,
        string manifestHash,
        Dictionary<string, string> inputVersions,
        int totalSegments)
    {
        if (snapshot.ManifestSha256 != manifestHash || snapshot.TotalSegments != totalSegments)
        {
            return false;
        }

        if (snapshot.InputVersions.Count != inputVersions.Count)
        {
            return false;
        }

        return inputVersions.All(pair =>
            snapshot.InputVersions.TryGetValue(pair.Key, out var hash)
            && string.Equals(hash, pair.Value, StringComparison.Ordinal));
    }
}
