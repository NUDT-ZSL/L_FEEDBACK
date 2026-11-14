using System.Buffers.Binary;
using System.Diagnostics;

namespace AudioPipeline;

public sealed class RenderOptions
{
    public int MaxBlocksThisRun { get; init; } = int.MaxValue;

    public int MaxBlockBufferBytes { get; init; } = AudioConstants.DefaultMaxBlockBufferBytes;
}

public sealed class BlockProgress
{
    public required int Index { get; init; }

    public required bool Reused { get; init; }

    public required string Sha256 { get; init; }
}

public sealed class RenderResult
{
    public required string OutputPath { get; init; }

    public required string StateDirectory { get; init; }

    public required long TotalFrames { get; init; }

    public required int TotalBlocks { get; init; }

    public required int RenderedBlocks { get; init; }

    public required int ReusedBlocks { get; init; }

    public required string EventSummaryHash { get; init; }

    public required bool Completed { get; init; }

    public required bool Cancelled { get; init; }

    public string MasterSha256 { get; init; } = string.Empty;

    public TimeSpan Elapsed { get; init; }
}

public static class OfflineRenderer
{
    public static RenderResult Render(
        string scenePath,
        string outputPath,
        RenderOptions? options = null,
        CancellationToken cancellationToken = default,
        Action<BlockProgress>? progress = null)
    {
        var scene = SceneLoader.LoadAndValidate(scenePath);
        return Render(scene, outputPath, options, cancellationToken, progress);
    }

    public static RenderResult Render(
        ValidatedScene scene,
        string outputPath,
        RenderOptions? options = null,
        CancellationToken cancellationToken = default,
        Action<BlockProgress>? progress = null)
    {
        options ??= new RenderOptions();
        ValidateResourceLimits(scene, options);

        var stopwatch = Stopwatch.StartNew();
        var fullOutputPath = Path.GetFullPath(outputPath);
        var outputDirectory = Path.GetDirectoryName(fullOutputPath) ??
                              throw new PipelineException(
                                  PipelineErrorCode.RenderIo,
                                  "Output path must have a parent directory.");

        using var outputLock = OutputLock.Acquire(outputDirectory);
        var store = new ManifestStore(fullOutputPath);
        store.EnsureDirectories();
        store.CleanTemporaryFiles();

        var totalBlocks = checked((int)((scene.TotalFrames + AudioConstants.BlockFrames - 1) /
                                        AudioConstants.BlockFrames));
        var existing = store.TryLoadManifest();
        var canResume = existing is not null && ManifestStore.MatchesScene(existing, scene, totalBlocks);
        BlockManifest manifest;
        var reusedBlocks = 0;
        var firstBlockToRender = 0;

        if (!canResume)
        {
            store.ClearBlocks();
            manifest = CreateFreshManifest(scene, totalBlocks);
            store.SaveManifestAtomic(manifest);
        }
        else
        {
            manifest = existing!;
            firstBlockToRender = FindFirstInvalidBlock(store, manifest, scene, totalBlocks);
            reusedBlocks = firstBlockToRender;
            for (var index = firstBlockToRender; index < totalBlocks; index++)
            {
                manifest.Blocks[index] = ManifestStore.CreatePendingBlock(index);
            }

            if (firstBlockToRender < totalBlocks)
            {
                store.SaveManifestAtomic(manifest);
            }
        }

        var renderedBlocks = 0;
        var cancelled = false;
        if (progress is not null)
        {
            for (var index = 0; index < reusedBlocks; index++)
            {
                progress(new BlockProgress
                {
                    Index = index,
                    Reused = true,
                    Sha256 = manifest.Blocks[index].Sha256
                });
            }
        }

        var runtimes = firstBlockToRender == 0
            ? CreateEmptyRuntimes(scene)
            : RestoreRuntimes(
                scene,
                manifest.Blocks[firstBlockToRender - 1],
                firstBlockToRender * (long)AudioConstants.BlockFrames);

        for (var blockIndex = firstBlockToRender; blockIndex < totalBlocks; blockIndex++)
        {
            if (cancellationToken.IsCancellationRequested ||
                renderedBlocks >= options.MaxBlocksThisRun)
            {
                cancelled = cancellationToken.IsCancellationRequested;
                break;
            }

            var blockStart = blockIndex * (long)AudioConstants.BlockFrames;
            var blockEnd = Math.Min(blockStart + AudioConstants.BlockFrames, scene.TotalFrames);
            var pcmBytes = RenderBlock(scene, blockStart, blockEnd, runtimes);
            var sha256 = WavIO.Sha256Hex(pcmBytes);
            store.WriteBlockAtomic(blockIndex, pcmBytes);

            var boundaryFrame = blockEnd;
            manifest.Blocks[blockIndex] = new BlockRecord
            {
                Index = blockIndex,
                Status = BlockStatus.Complete,
                FrameCount = AudioConstants.BlockFrames,
                ByteLength = pcmBytes.Length,
                Sha256 = sha256,
                ExitStates = CaptureExitStates(scene, runtimes, boundaryFrame)
            };
            store.SaveManifestAtomic(manifest);
            renderedBlocks++;
            progress?.Invoke(new BlockProgress
            {
                Index = blockIndex,
                Reused = false,
                Sha256 = sha256
            });
        }

        var completed = renderedBlocks + reusedBlocks == totalBlocks;
        var masterHash = string.Empty;
        if (completed)
        {
            masterHash = WavAssembler.Assemble(
                store,
                fullOutputPath,
                scene.TotalFrames,
                totalBlocks);
        }

        stopwatch.Stop();
        return new RenderResult
        {
            OutputPath = fullOutputPath,
            StateDirectory = store.StateDirectory,
            TotalFrames = scene.TotalFrames,
            TotalBlocks = totalBlocks,
            RenderedBlocks = renderedBlocks,
            ReusedBlocks = reusedBlocks,
            EventSummaryHash = scene.EventSummaryHash,
            Completed = completed,
            Cancelled = cancelled,
            MasterSha256 = masterHash,
            Elapsed = stopwatch.Elapsed
        };
    }

    private static void ValidateResourceLimits(ValidatedScene scene, RenderOptions options)
    {
        if (scene.Events.Count > AudioConstants.MaxEvents)
        {
            throw new PipelineException(
                PipelineErrorCode.EventLimit,
                $"The scene contains {scene.Events.Count} events; the limit is {AudioConstants.MaxEvents}.");
        }

        if (scene.TotalFrames > AudioConstants.MaxFrames)
        {
            throw new PipelineException(
                PipelineErrorCode.DurationLimit,
                $"The scene produces {scene.TotalFrames} frames; the limit is {AudioConstants.MaxFrames}.");
        }

        var blockBufferBytes = checked(
            AudioConstants.BlockFrames * AudioConstants.Channels * sizeof(int));
        if (blockBufferBytes > options.MaxBlockBufferBytes)
        {
            throw new PipelineException(
                PipelineErrorCode.BlockBufferLimit,
                $"The block accumulator requires {blockBufferBytes} bytes, exceeding " +
                $"the configured limit of {options.MaxBlockBufferBytes} bytes.");
        }
    }

    private static BlockManifest CreateFreshManifest(ValidatedScene scene, int totalBlocks)
    {
        var manifest = new BlockManifest
        {
            ManifestVersion = AudioConstants.ManifestVersion,
            AlgorithmVersion = AudioConstants.AlgorithmVersion,
            SampleRate = AudioConstants.SampleRate,
            Channels = AudioConstants.Channels,
            BitsPerSample = AudioConstants.BitsPerSample,
            BlockFrames = AudioConstants.BlockFrames,
            TotalFrames = scene.TotalFrames,
            TotalBlocks = totalBlocks,
            EventSummaryHash = scene.EventSummaryHash,
            EventSummaryAlgorithm = AudioConstants.EventSummaryAlgorithm,
            Blocks = new List<BlockRecord>(totalBlocks)
        };

        for (var index = 0; index < totalBlocks; index++)
        {
            manifest.Blocks.Add(ManifestStore.CreatePendingBlock(index));
        }

        return manifest;
    }

    private static int FindFirstInvalidBlock(
        ManifestStore store,
        BlockManifest manifest,
        ValidatedScene scene,
        int totalBlocks)
    {
        for (var index = 0; index < totalBlocks; index++)
        {
            var record = manifest.Blocks[index];
            if (record is null ||
                !store.ValidateBlock(record, index, manifest) ||
                !HasCompleteExitStates(scene, record, Math.Min(
                    (index + 1) * (long)AudioConstants.BlockFrames,
                    scene.TotalFrames)))
            {
                return index;
            }
        }

        return totalBlocks;
    }

    private static bool HasCompleteExitStates(
        ValidatedScene scene,
        BlockRecord record,
        long boundaryFrame)
    {
        var eventsById = scene.Events.ToDictionary(
            item => item.Definition.Id,
            item => item,
            StringComparer.Ordinal);
        var states = new Dictionary<string, EventExitState>(StringComparer.Ordinal);
        foreach (var state in record.ExitStates ?? [])
        {
            if (state is null ||
                string.IsNullOrWhiteSpace(state.EventId) ||
                !eventsById.TryGetValue(state.EventId, out var item) ||
                !states.TryAdd(state.EventId, state))
            {
                return false;
            }

            var activeAtBoundary =
                item.StartFrame < boundaryFrame &&
                item.EndFrame > boundaryFrame;
            if (!double.IsFinite(state.LowPassState) ||
                !double.IsFinite(state.DopplerReadPosition) ||
                !double.IsFinite(state.DopplerDelayPhase) ||
                state.DopplerDelayPhase < 0 ||
                state.DopplerDelayPhase >= 1 ||
                state.NextLocalFrame != boundaryFrame - item.StartFrame ||
                !activeAtBoundary)
            {
                return false;
            }
        }

        foreach (var item in scene.Events)
        {
            var activeAtBoundary = item.StartFrame < boundaryFrame && item.EndFrame > boundaryFrame;
            if (!activeAtBoundary)
            {
                continue;
            }

            if (!states.TryGetValue(item.Definition.Id, out var state) ||
                state.NextLocalFrame != boundaryFrame - item.StartFrame)
            {
                return false;
            }
        }

        return true;
    }

    private static EventRuntime[] CreateEmptyRuntimes(ValidatedScene scene) =>
        scene.Events.Select(_ => new EventRuntime()).ToArray();

    private static EventRuntime[] RestoreRuntimes(
        ValidatedScene scene,
        BlockRecord previousBlock,
        long boundaryFrame)
    {
        var states = (previousBlock.ExitStates ?? []).ToDictionary(
            item => item.EventId,
            item => item,
            StringComparer.Ordinal);
        var runtimes = CreateEmptyRuntimes(scene);

        for (var index = 0; index < scene.Events.Count; index++)
        {
            var item = scene.Events[index];
            if (!states.TryGetValue(item.Definition.Id, out var state))
            {
                if (item.StartFrame < boundaryFrame && item.EndFrame > boundaryFrame)
                {
                    throw new PipelineException(
                        PipelineErrorCode.ManifestInvalid,
                        $"Manifest is missing the exit state for active event '{item.Definition.Id}'.");
                }

                continue;
            }

            var expectedFrame = boundaryFrame - item.StartFrame;
            if (state.NextLocalFrame != expectedFrame ||
                expectedFrame < 0 ||
                expectedFrame >= item.DurationFrames)
            {
                throw new PipelineException(
                    PipelineErrorCode.ManifestInvalid,
                    $"Manifest exit state for event '{item.Definition.Id}' has an inconsistent frame offset.");
            }

            runtimes[index].Started = true;
            runtimes[index].NextLocalFrame = state.NextLocalFrame;
            runtimes[index].LowPassState = state.LowPassState;
            runtimes[index].ReadPosition = state.DopplerReadPosition;
        }

        return runtimes;
    }

    private static byte[] RenderBlock(
        ValidatedScene scene,
        long blockStart,
        long blockEnd,
        EventRuntime[] runtimes)
    {
        var mixer = new FixedPointMixBuffer(AudioConstants.BlockFrames);
        for (var eventIndex = 0; eventIndex < scene.Events.Count; eventIndex++)
        {
            var item = scene.Events[eventIndex];
            var runtime = runtimes[eventIndex];
            if (item.EndFrame <= blockStart || item.StartFrame >= blockEnd)
            {
                continue;
            }

            var firstFrame = Math.Max(blockStart, item.StartFrame);
            var lastFrame = Math.Min(blockEnd, item.EndFrame);
            for (var globalFrame = firstFrame; globalFrame < lastFrame; globalFrame++)
            {
                var localFrame = globalFrame - item.StartFrame;
                if (!runtime.Started)
                {
                    if (localFrame != 0)
                    {
                        throw new PipelineException(
                            PipelineErrorCode.ManifestInvalid,
                            $"Event '{item.Definition.Id}' is active before its runtime state was initialized.");
                    }

                    runtime.Started = true;
                    runtime.NextLocalFrame = 0;
                    runtime.LowPassState = 0;
                    runtime.ReadPosition = 0;
                }

                if (runtime.NextLocalFrame != localFrame)
                {
                    throw new PipelineException(
                        PipelineErrorCode.ManifestInvalid,
                        $"Event '{item.Definition.Id}' expected local frame {runtime.NextLocalFrame} " +
                        $"but resumed at {localFrame}.");
                }

                var frame = SpatialDsp.EvaluateFrame(
                    item.Definition,
                    scene.Definition.Listener,
                    localFrame,
                    item.DurationFrames,
                    item.Occluders);

                double sourceSample;
                if (item.Definition.Doppler)
                {
                    sourceSample = AudioDsp.ReadInterpolated(item.Source.Samples, runtime.ReadPosition);
                    runtime.ReadPosition += frame.DopplerFactor;
                }
                else
                {
                    sourceSample = AudioDsp.ReadIntegral(item.Source.Samples, localFrame);
                    runtime.ReadPosition = localFrame + 1;
                }

                var lowPassState = runtime.LowPassState;
                runtime.LowPassState = AudioDsp.ProcessLowPass(sourceSample, ref lowPassState);
                var audibleSample = frame.Occluded ? runtime.LowPassState : sourceSample;
                var amplitude = audibleSample * item.Definition.Gain * frame.DistanceGain;
                var localOutputFrame = (int)(globalFrame - blockStart);
                mixer.AddStereo(
                    localOutputFrame,
                    amplitude * frame.LeftGain,
                    amplitude * frame.RightGain);
                runtime.NextLocalFrame++;
            }
        }

        var pcmBytes = new byte[AudioConstants.BlockBytes];
        for (var frameIndex = 0; frameIndex < AudioConstants.BlockFrames; frameIndex++)
        {
            mixer.ReadStereo(frameIndex, out var left, out var right);
            var offset = frameIndex * AudioConstants.Channels * AudioConstants.BytesPerSample;
            BinaryPrimitives.WriteInt16LittleEndian(pcmBytes.AsSpan(offset, 2), left);
            BinaryPrimitives.WriteInt16LittleEndian(pcmBytes.AsSpan(offset + 2, 2), right);
        }

        return pcmBytes;
    }

    private static List<EventExitState> CaptureExitStates(
        ValidatedScene scene,
        EventRuntime[] runtimes,
        long boundaryFrame)
    {
        var states = new List<EventExitState>();
        for (var index = 0; index < scene.Events.Count; index++)
        {
            var item = scene.Events[index];
            var runtime = runtimes[index];
            if (!runtime.Started ||
                item.StartFrame >= boundaryFrame ||
                item.EndFrame <= boundaryFrame)
            {
                continue;
            }

            var expectedNextFrame = boundaryFrame - item.StartFrame;
            if (runtime.NextLocalFrame != expectedNextFrame)
            {
                throw new PipelineException(
                    PipelineErrorCode.ManifestInvalid,
                    $"Event '{item.Definition.Id}' exited block at local frame " +
                    $"{runtime.NextLocalFrame}; expected {expectedNextFrame}.");
            }

            var phase = item.Definition.Doppler
                ? runtime.ReadPosition - Math.Floor(runtime.ReadPosition)
                : 0.0;
            if (phase < 0 || phase >= 1)
            {
                phase = 0;
            }

            states.Add(new EventExitState
            {
                EventId = item.Definition.Id,
                NextLocalFrame = runtime.NextLocalFrame,
                LowPassState = runtime.LowPassState,
                DopplerReadPosition = runtime.ReadPosition,
                DopplerDelayPhase = phase
            });
        }

        return states;
    }

    private sealed class EventRuntime
    {
        public bool Started { get; set; }

        public long NextLocalFrame { get; set; }

        public double LowPassState { get; set; }

        public double ReadPosition { get; set; }
    }
}
