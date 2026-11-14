using AudioPipeline;
using AudioPipeline.Tests;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Text.Json.Nodes;

Directory.CreateDirectory(TestHarness.RunRoot);

TestHarness.Run("deterministic fixture generation", () =>
{
    var first = TestHarness.GenerateFixture("deterministic-a");
    var second = TestHarness.GenerateFixture("deterministic-b");
    TestHarness.AssertEqual(
        WavIO.Sha256Hex(Path.Combine(first.Directory, "sources", "ambient_a.wav")),
        WavIO.Sha256Hex(Path.Combine(second.Directory, "sources", "ambient_a.wav")),
        "ambient_a is not deterministic");
    TestHarness.AssertEqual(
        WavIO.Sha256Hex(Path.Combine(first.Directory, "sources", "ambient_b.wav")),
        WavIO.Sha256Hex(Path.Combine(second.Directory, "sources", "ambient_b.wav")),
        "ambient_b is not deterministic");
    TestHarness.AssertEqual(
        File.ReadAllText(first.ScenePath),
        File.ReadAllText(second.ScenePath),
        "scene JSON is not deterministic");
    TestHarness.AssertEqual(105_600L, first.ExpectedFrames, "unexpected fixture frame count");
});

TestHarness.Run("scene and source validation with stable error codes", () =>
{
    var fixture = TestHarness.GenerateFixture("scene-validation");
    var validated = SceneLoader.LoadAndValidate(fixture.ScenePath);
    TestHarness.AssertEqual(3, validated.Events.Count, "event count");
    TestHarness.AssertEqual(105_600L, validated.TotalFrames, "total frame count");
    TestHarness.AssertEqual(64, validated.EventSummaryHash.Length, "summary hash length");

    var versionPath = Path.Combine(fixture.Directory, "bad-version.json");
    var versionNode = JsonNode.Parse(File.ReadAllText(fixture.ScenePath))!;
    versionNode["version"] = 2;
    File.WriteAllText(versionPath, versionNode.ToJsonString());
    ExpectCode(
        () => SceneLoader.LoadAndValidate(versionPath),
        PipelineErrorCode.SceneVersion,
        PipelineErrorNames.SceneVersion);

    var malformedPath = Path.Combine(fixture.Directory, "malformed.json");
    File.WriteAllText(malformedPath, "{not-json");
    ExpectCode(
        () => SceneLoader.LoadAndValidate(malformedPath),
        PipelineErrorCode.SceneJson,
        PipelineErrorNames.SceneJson);

    var missingPath = Path.Combine(fixture.Directory, "missing-source.json");
    var missingScene = TestHarness.ReadScene(fixture.ScenePath);
    missingScene.Events![0].SourcePath = "sources/not-there.wav";
    TestHarness.WriteScene(fixture.Directory, missingScene, "missing-source.json");
    ExpectCode(
        () => SceneLoader.LoadAndValidate(missingPath),
        PipelineErrorCode.SourceMissing,
        PipelineErrorNames.SourceMissing);

    var longPath = Path.Combine(fixture.Directory, "long-event.json");
    var longScene = TestHarness.ReadScene(fixture.ScenePath);
    longScene.Events![0].DurationSeconds = 301;
    longScene.Events![0].SourcePath = "sources/not-there.wav";
    TestHarness.WriteScene(fixture.Directory, longScene, "long-event.json");
    ExpectCode(
        () => SceneLoader.LoadAndValidate(longPath),
        PipelineErrorCode.DurationLimit,
        PipelineErrorNames.DurationLimit);

    var delayedPath = Path.Combine(fixture.Directory, "long-delay.json");
    var delayedScene = TestHarness.ReadScene(fixture.ScenePath);
    delayedScene.Events![0].DelaySeconds = 3600;
    delayedScene.Events![0].SourcePath = "sources/not-there.wav";
    TestHarness.WriteScene(fixture.Directory, delayedScene, "long-delay.json");
    ExpectCode(
        () => SceneLoader.LoadAndValidate(delayedPath),
        PipelineErrorCode.DurationLimit,
        PipelineErrorNames.DurationLimit);

    var manyPath = Path.Combine(fixture.Directory, "many-events.json");
    var manyScene = TestHarness.ReadScene(fixture.ScenePath);
    var template = manyScene.Events![0];
    manyScene.Events = [];
    for (var index = 0; index < AudioConstants.MaxEvents + 1; index++)
    {
        manyScene.Events.Add(new EventDefinition
        {
            Id = $"event-{index}",
            Start = template.Start,
            End = template.End,
            DurationSeconds = 0.01,
            DelaySeconds = 0,
            Gain = 0.1,
            SourcePath = template.SourcePath,
            MinDistance = 1,
            MaxDistance = 10,
            Doppler = false,
            Occluders = []
        });
    }

    TestHarness.WriteScene(fixture.Directory, manyScene, "many-events.json");
    ExpectCode(
        () => SceneLoader.LoadAndValidate(manyPath),
        PipelineErrorCode.EventLimit,
        PipelineErrorNames.EventLimit);

    var invalidScene = TestHarness.ReadScene(fixture.ScenePath);
    invalidScene.Events![0].MaxDistance = invalidScene.Events[0].MinDistance;
    var invalidPath = TestHarness.WriteScene(
        fixture.Directory,
        invalidScene,
        "invalid-distances.json");
    ExpectCode(
        () => SceneLoader.LoadAndValidate(invalidPath),
        PipelineErrorCode.SceneValidation,
        PipelineErrorNames.SceneValidation);

    var badSourceDirectory = TestHarness.NewDirectory("source-format-copy");
    CopyDirectory(fixture.Directory, badSourceDirectory);
    var badSourcePath = Path.Combine(badSourceDirectory, "sources", "ambient_a.wav");
    using (var stream = new FileStream(badSourcePath, FileMode.Open, FileAccess.ReadWrite))
    {
        stream.Position = 22;
        stream.WriteByte(2);
        stream.Position = 32;
        stream.WriteByte(4);
    }

    ExpectCode(
        () => SceneLoader.LoadAndValidate(Path.Combine(badSourceDirectory, "scene.json")),
        PipelineErrorCode.SourceFormat,
        PipelineErrorNames.SourceFormat);
});

TestHarness.Run("WAV frame count, RIFF sizes, and final-block padding", () =>
{
    var fixture = TestHarness.GenerateFixture("wav-frame-count");
    var output = Path.Combine(fixture.Directory, "master.wav");
    var result = TestHarness.Render(fixture.ScenePath, output);
    var info = WavIO.ReadInfo(output);
    TestHarness.AssertEqual(1, info.FormatTag, "format tag");
    TestHarness.AssertEqual(2, info.Channels, "channels");
    TestHarness.AssertEqual(48_000, info.SampleRate, "sample rate");
    TestHarness.AssertEqual(16, info.BitsPerSample, "bit depth");
    TestHarness.AssertEqual(105_600L, info.FrameCount, "master frame count");
    TestHarness.AssertEqual(422_400L, info.DataByteLength, "master data byte length");
    TestHarness.AssertEqual(422_444L, info.FileLength, "master file length");
    TestHarness.AssertEqual(105_600L, result.TotalFrames, "renderer total frames");
    TestHarness.AssertEqual(3, result.TotalBlocks, "block count");

    var state = new ManifestStore(output);
    for (var index = 0; index < result.TotalBlocks; index++)
    {
        TestHarness.AssertEqual(
            (long)AudioConstants.BlockBytes,
            new FileInfo(state.GetBlockPath(index)).Length,
            $"block {index} byte length");
    }

    var lastBlock = File.ReadAllBytes(state.GetBlockPath(2));
    var activeBytes = checked(9_600 * 4);
    for (var index = activeBytes; index < lastBlock.Length; index++)
    {
        TestHarness.AssertEqual((byte)0, lastBlock[index], "final block padding must be silent");
    }
});

TestHarness.Run("resume skips completed blocks and is byte-identical", () =>
{
    var fixture = TestHarness.GenerateFixture("resume-skip");
    var reference = Path.Combine(fixture.Directory, "reference.wav");
    TestHarness.Render(fixture.ScenePath, reference);
    var referenceHash = TestHarness.MasterHash(reference);

    var resumed = Path.Combine(fixture.Directory, "resumed.wav");
    var first = TestHarness.Render(fixture.ScenePath, resumed, maxBlocks: 1);
    TestHarness.AssertEqual(1, first.RenderedBlocks, "first partial render count");
    TestHarness.AssertEqual(0, first.ReusedBlocks, "first partial reuse count");
    TestHarness.Assert(!first.Completed, "partial render must not be marked complete");
    TestHarness.AssertEqual(string.Empty, first.MasterSha256, "partial render must not write a master");
    TestHarness.Assert(!File.Exists(resumed), "partial render must not create a master");

    var second = TestHarness.Render(fixture.ScenePath, resumed, maxBlocks: 1);
    TestHarness.AssertEqual(1, second.RenderedBlocks, "second render count");
    TestHarness.AssertEqual(1, second.ReusedBlocks, "second reuse count");
    TestHarness.Assert(!second.Completed, "second render must still be partial");

    var third = TestHarness.Render(fixture.ScenePath, resumed);
    TestHarness.AssertEqual(1, third.RenderedBlocks, "third render count");
    TestHarness.AssertEqual(2, third.ReusedBlocks, "third reuse count");
    TestHarness.Assert(third.Completed, "third render must complete");
    TestHarness.AssertEqual(referenceHash, third.MasterSha256, "resumed master hash");

    var fourth = TestHarness.Render(fixture.ScenePath, resumed);
    TestHarness.AssertEqual(0, fourth.RenderedBlocks, "fully resumed render count");
    TestHarness.AssertEqual(3, fourth.ReusedBlocks, "fully resumed reuse count");
    TestHarness.AssertEqual(referenceHash, fourth.MasterSha256, "fully resumed master hash");
});

TestHarness.Run("missing, size-mismatched, and hash-mismatched blocks recalculate the suffix", () =>
{
    var fixture = TestHarness.GenerateFixture("hash-recovery");
    var reference = Path.Combine(fixture.Directory, "reference.wav");
    TestHarness.Render(fixture.ScenePath, reference);
    var referenceHash = TestHarness.MasterHash(reference);

    RunRecoveryCase("missing-block", blockPath => File.Delete(blockPath));
    RunRecoveryCase("size-mismatch", blockPath =>
    {
        using var stream = new FileStream(blockPath, FileMode.Append, FileAccess.Write);
        stream.WriteByte(0x7f);
        stream.Flush(true);
    });
    RunRecoveryCase("hash-mismatch", blockPath =>
    {
        using var stream = new FileStream(blockPath, FileMode.Open, FileAccess.ReadWrite);
        var lastByte = stream.ReadByte();
        stream.Position = stream.Length - 1;
        stream.WriteByte((byte)(lastByte ^ 0x01));
        stream.Flush(true);
    });
    RunExitStateRecoveryCase();

    void RunRecoveryCase(string name, Action<string> corrupt)
    {
        var caseDirectory = TestHarness.NewDirectory($"hash-recovery-cases/{name}");
        CopyDirectory(fixture.Directory, caseDirectory);
        var scenePath = Path.Combine(caseDirectory, "scene.json");
        var output = Path.Combine(caseDirectory, "master.wav");
        var initial = TestHarness.Render(scenePath, output);
        TestHarness.AssertEqual(3, initial.RenderedBlocks, $"{name} initial render count");

        var state = new ManifestStore(output);
        corrupt(state.GetBlockPath(1));
        var result = TestHarness.Render(scenePath, output);
        TestHarness.AssertEqual(1, result.ReusedBlocks, $"{name} reuse count");
        TestHarness.AssertEqual(2, result.RenderedBlocks, $"{name} recalculated block count");
        TestHarness.Assert(result.Completed, $"{name} completion");
        TestHarness.AssertEqual(referenceHash, result.MasterSha256, $"{name} master hash");

        var manifest = TestHarness.ReadManifest(output);
        TestHarness.AssertEqual(BlockStatus.Complete, manifest.Blocks[1].Status, $"{name} block 1 status");
        TestHarness.AssertEqual(BlockStatus.Complete, manifest.Blocks[2].Status, $"{name} block 2 status");
    }

    void RunExitStateRecoveryCase()
    {
        var caseDirectory = TestHarness.NewDirectory("hash-recovery-cases/exit-state-mismatch");
        CopyDirectory(fixture.Directory, caseDirectory);
        var scenePath = Path.Combine(caseDirectory, "scene.json");
        var output = Path.Combine(caseDirectory, "master.wav");
        TestHarness.Render(scenePath, output);

        var store = new ManifestStore(output);
        var node = JsonNode.Parse(File.ReadAllText(store.ManifestPath))!;
        node["blocks"]![1]!["exitStates"]![0]!["nextLocalFrame"] =
            node["blocks"]![1]!["exitStates"]![0]!["nextLocalFrame"]!.GetValue<long>() + 1;
        File.WriteAllText(store.ManifestPath, node.ToJsonString());

        var result = TestHarness.Render(scenePath, output);
        TestHarness.AssertEqual(1, result.ReusedBlocks, "exit-state mismatch reuse count");
        TestHarness.AssertEqual(2, result.RenderedBlocks, "exit-state mismatch render count");
        TestHarness.AssertEqual(referenceHash, result.MasterSha256, "exit-state mismatch master hash");
    }
});

TestHarness.Run("manifest version change rebuilds every block", () =>
{
    var fixture = TestHarness.GenerateFixture("manifest-version");
    var output = Path.Combine(fixture.Directory, "master.wav");
    TestHarness.Render(fixture.ScenePath, output);
    var referenceHash = TestHarness.MasterHash(output);
    var store = new ManifestStore(output);
    var node = JsonNode.Parse(File.ReadAllText(store.ManifestPath))!;
    node["manifestVersion"] = 999;
    File.WriteAllText(store.ManifestPath, node.ToJsonString());

    var result = TestHarness.Render(fixture.ScenePath, output);
    TestHarness.AssertEqual(0, result.ReusedBlocks, "version rebuild reuse count");
    TestHarness.AssertEqual(3, result.RenderedBlocks, "version rebuild render count");
    TestHarness.Assert(result.Completed, "version rebuild completion");
    TestHarness.AssertEqual(referenceHash, result.MasterSha256, "version rebuild master hash");
    var repaired = TestHarness.ReadManifest(output);
    TestHarness.AssertEqual(
        AudioConstants.ManifestVersion,
        repaired.ManifestVersion,
        "manifest version should be restored");
});

TestHarness.Run("Doppler read phase and occlusion low-pass survive block boundaries", () =>
{
    var fixture = TestHarness.GenerateFixture("cross-block-dsp");
    var reference = Path.Combine(fixture.Directory, "reference.wav");
    TestHarness.Render(fixture.ScenePath, reference);
    var referenceHash = TestHarness.MasterHash(reference);

    var resumed = Path.Combine(fixture.Directory, "resumed.wav");
    TestHarness.Render(fixture.ScenePath, resumed, maxBlocks: 1);
    var firstManifest = TestHarness.ReadManifest(resumed);
    var dopplerState = firstManifest.Blocks[0].ExitStates
        .Single(item => item.EventId == "doppler-pass");
    TestHarness.Assert(
        dopplerState.DopplerReadPosition > 0 &&
        dopplerState.DopplerReadPosition != Math.Floor(dopplerState.DopplerReadPosition),
        "Doppler exit state must retain a fractional read position");
    TestHarness.Assert(
        Math.Abs(dopplerState.DopplerDelayPhase -
                 (dopplerState.DopplerReadPosition - Math.Floor(dopplerState.DopplerReadPosition))) < 1e-12,
        "Doppler delay phase must be exported as the fractional part");
    TestHarness.Assert(
        Math.Abs(dopplerState.LowPassState) > 1e-9,
        "Low-pass state must be persisted at the block boundary");

    TestHarness.Render(fixture.ScenePath, resumed, maxBlocks: 1);
    var secondManifest = TestHarness.ReadManifest(resumed);
    var secondDoppler = secondManifest.Blocks[1].ExitStates
        .Single(item => item.EventId == "doppler-pass");
    TestHarness.Assert(
        secondDoppler.NextLocalFrame > dopplerState.NextLocalFrame,
        "Doppler event state did not advance across the block boundary");
    TestHarness.Assert(
        Math.Abs(secondDoppler.LowPassState - dopplerState.LowPassState) > 1e-9,
        "Low-pass state was reset or frozen at the block boundary");

    var result = TestHarness.Render(fixture.ScenePath, resumed);
    TestHarness.AssertEqual(1, result.RenderedBlocks, "cross-block final render count");
    TestHarness.AssertEqual(2, result.ReusedBlocks, "cross-block final reuse count");
    TestHarness.AssertEqual(referenceHash, result.MasterSha256, "cross-block output hash");

    VerifyOcclusionToggle(fixture.Directory);
});

TestHarness.Run("exclusive output-directory lock rejects a second process", () =>
{
    var fixture = TestHarness.GenerateFixture("lock-contention");
    var output = Path.Combine(fixture.Directory, "master.wav");
    using (OutputLock.Acquire(fixture.Directory))
    {
        TestHarness.AssertEqual(
            PipelineErrorCode.OutputLocked,
            TestHarness.ExpectCode(() => TestHarness.Render(fixture.ScenePath, output)),
            "in-process lock contention code");

        var process = StartCli(
            "render",
            "--scene",
            fixture.ScenePath,
            "--output",
            output);
        var stdoutTask = process.StandardOutput.ReadToEndAsync();
        var stderrTask = process.StandardError.ReadToEndAsync();
        var exited = process.WaitForExit(TimeSpan.FromSeconds(30));
        if (!exited)
        {
            process.Kill(entireProcessTree: true);
        }

        process.WaitForExit();
        var stdout = stdoutTask.GetAwaiter().GetResult();
        var stderr = stderrTask.GetAwaiter().GetResult();
        TestHarness.Assert(exited, "contending CLI process timed out");
        TestHarness.Assert(process.ExitCode != 0, "contending CLI process should fail");
        TestHarness.Assert(
            stdout.Contains(PipelineErrorNames.OutputLocked, StringComparison.Ordinal),
            $"contending CLI output did not expose OUTPUT_LOCKED; stdout={stdout} stderr={stderr}");
    }

    var result = TestHarness.Render(fixture.ScenePath, output);
    TestHarness.Assert(result.Completed, "render should succeed after lock release");
});

TestHarness.Run("fixed-point mixing clips once and leaves no event tail", () =>
{
    var buffer = new FixedPointMixBuffer(2);
    buffer.AddStereo(0, 1.9, 0.0);
    buffer.AddStereo(0, -1.4, 0.0);
    buffer.ReadStereo(0, out var mixedLeft, out _);
    TestHarness.AssertEqual((short)16_384, mixedLeft, "unified cancellation value");

    buffer.AddStereo(1, 2.0, -2.0);
    buffer.ReadStereo(1, out var clippedLeft, out var clippedRight);
    TestHarness.AssertEqual((short)short.MaxValue, clippedLeft, "positive final clip");
    TestHarness.AssertEqual((short)short.MinValue, clippedRight, "negative final clip");

    var directory = TestHarness.NewDirectory("mix-and-tail");
    FixtureGenerator.Generate(directory);
    var scene = FixtureGenerator.CreateScene();
    scene.Events =
    [
        new EventDefinition
        {
            Id = "short",
            Start = new Vec3(0, 0, 1),
            End = new Vec3(0, 0, 1),
            DurationSeconds = 0.05,
            DelaySeconds = 0,
            Gain = 0.8,
            SourcePath = "sources/ambient_a.wav",
            MinDistance = 1,
            MaxDistance = 10,
            Doppler = false,
            Occluders = []
        },
        new EventDefinition
        {
            Id = "late",
            Start = new Vec3(0, 0, 2),
            End = new Vec3(0, 0, 2),
            DurationSeconds = 0.01,
            DelaySeconds = 0.2,
            Gain = 0.4,
            SourcePath = "sources/ambient_b.wav",
            MinDistance = 1,
            MaxDistance = 10,
            Doppler = false,
            Occluders = []
        }
    ];
    var scenePath = TestHarness.WriteScene(directory, scene, "tail-scene.json");
    var output = Path.Combine(directory, "tail.wav");
    var result = TestHarness.Render(scenePath, output);
    TestHarness.AssertEqual(10_080L, result.TotalFrames, "tail scene total frames");
    var data = TestHarness.ReadWavData(output);
    for (var frame = 2_400; frame < 9_600; frame++)
    {
        for (var channel = 0; channel < 2; channel++)
        {
            var sample = BinaryPrimitives.ReadInt16LittleEndian(
                data.AsSpan((frame * 4) + (channel * 2), 2));
            TestHarness.AssertEqual((short)0, sample, $"tail frame {frame} channel {channel}");
        }
    }

    TestHarness.AssertEqual(
        PipelineErrorCode.BlockBufferLimit,
        TestHarness.ExpectCode(() => OfflineRenderer.Render(
            scenePath,
            Path.Combine(directory, "too-small.wav"),
            new RenderOptions { MaxBlockBufferBytes = 1024 })),
        "block buffer limit code");
});

TestHarness.Run("analytical segment/AABB intersection", () =>
{
    var box = new Aabb(new Vec3(-1, -1, 4), new Vec3(1, 1, 6));
    TestHarness.Assert(
        Geometry.SegmentIntersectsAabb(new Vec3(0, 0, 0), new Vec3(0, 0, 10), box),
        "crossing segment should intersect");
    TestHarness.Assert(
        !Geometry.SegmentIntersectsAabb(new Vec3(4, 0, 0), new Vec3(4, 0, 10), box),
        "parallel segment outside the box should miss");
    TestHarness.Assert(
        Geometry.SegmentIntersectsAabb(new Vec3(0, 0, 5), new Vec3(2, 0, 5), box),
        "segment starting inside the box should intersect");
    TestHarness.Assert(
        !Geometry.SegmentIntersectsAabb(new Vec3(0, 0, 0), new Vec3(0, 0, 3), box),
        "segment ending before the box should miss");
    TestHarness.Assert(
        Geometry.SegmentIntersectsAabb(new Vec3(2, 0, 5), new Vec3(-2, 0, 5), box),
        "reversed crossing segment should intersect");
});

Console.WriteLine();
Console.WriteLine("Stable error code sample:");
var sampleError = new PipelineException(PipelineErrorCode.OutputLocked, "contended");
Console.WriteLine($"{sampleError.NumericCode} {sampleError.StableCode}");

Environment.ExitCode = TestHarness.Finish();

static void ExpectCode(Action action, PipelineErrorCode expected, string expectedName)
{
    try
    {
        action();
    }
    catch (PipelineException ex)
    {
        TestHarness.AssertEqual(expected, ex.Code, "business error code");
        TestHarness.AssertEqual((int)expected, ex.NumericCode, "numeric business error code");
        TestHarness.AssertEqual(expectedName, ex.StableCode, "stable business error name");
        return;
    }

    throw new InvalidOperationException($"Expected {expected}, but no exception was thrown.");
}

static void CopyDirectory(string source, string destination)
{
    Directory.CreateDirectory(destination);
    foreach (var path in Directory.EnumerateFiles(source))
    {
        var target = Path.Combine(destination, Path.GetFileName(path));
        if (Path.GetFileName(path) == AudioConstants.LockFileName)
        {
            continue;
        }

        File.Copy(path, target, overwrite: true);
    }

    foreach (var directory in Directory.EnumerateDirectories(source))
    {
        CopyDirectory(directory, Path.Combine(destination, Path.GetFileName(directory)));
    }
}

static Process StartCli(params string[] arguments)
{
    var startInfo = new ProcessStartInfo
    {
        FileName = TestHarness.FindDotnetHost(),
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true
    };
    startInfo.ArgumentList.Add(TestHarness.FindCliDll());
    foreach (var argument in arguments)
    {
        startInfo.ArgumentList.Add(argument);
    }

    var process = new Process { StartInfo = startInfo };
    if (!process.Start())
    {
        throw new InvalidOperationException("Unable to start the CLI lock-contention process.");
    }

    return process;
}

static void VerifyOcclusionToggle(string fixtureDirectory)
{
    var directory = TestHarness.NewDirectory("cross-block-dsp/occlusion-toggle");
    CopyDirectory(Path.Combine(fixtureDirectory, "sources"), Path.Combine(directory, "sources"));
    var scene = FixtureGenerator.CreateScene();
    scene.Events =
    [
        new EventDefinition
        {
            Id = "toggle",
            Start = new Vec3(-4, 0, 8),
            End = new Vec3(4, 0, 8),
            DurationSeconds = 1.4,
            DelaySeconds = 0.5,
            Gain = 0.7,
            SourcePath = "sources/ambient_a.wav",
            MinDistance = 0.5,
            MaxDistance = 20,
            Doppler = true,
            Occluders =
            [
                new Aabb(new Vec3(-1, -2, 5), new Vec3(1, 2, 6))
            ]
        }
    ];
    var scenePath = TestHarness.WriteScene(directory, scene, "toggle-scene.json");
    var reference = Path.Combine(directory, "reference.wav");
    TestHarness.Render(scenePath, reference);
    var referenceHash = TestHarness.MasterHash(reference);

    var resumed = Path.Combine(directory, "resumed.wav");
    TestHarness.Render(scenePath, resumed, maxBlocks: 1);
    var first = TestHarness.ReadManifest(resumed);
    var activeState = first.Blocks[0].ExitStates.Single(item => item.EventId == "toggle");
    TestHarness.Assert(
        Math.Abs(activeState.LowPassState) > 1e-9,
        "occluded event should have a non-zero low-pass state at the boundary");
    var result = TestHarness.Render(scenePath, resumed);
    TestHarness.AssertEqual(1, result.ReusedBlocks, "occlusion toggle reuse count");
    TestHarness.AssertEqual(1, result.RenderedBlocks, "occlusion toggle render count");
    TestHarness.AssertEqual(referenceHash, result.MasterSha256, "occlusion toggle output hash");
}
