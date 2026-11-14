using System.Security.Cryptography;
using ReplayForge.Utilities;

namespace ReplayForge.State;

public sealed class ResumeStore : IDisposable
{
    private readonly FileStream _lockStream;
    private readonly string _runToken;

    private ResumeStore(string workDirectory, FileStream lockStream)
    {
        WorkDirectory = workDirectory;
        _lockStream = lockStream;
        _runToken = $"{Environment.ProcessId}-{Guid.NewGuid():N}";
    }

    public string WorkDirectory { get; }

    public string SnapshotPath => Path.Combine(WorkDirectory, "resume.json");

    public string JournalsDirectory => Path.Combine(WorkDirectory, "journals");

    public string ChunksDirectory => Path.Combine(WorkDirectory, "chunks");

    public static ResumeStore Open(string workDirectory)
    {
        Directory.CreateDirectory(workDirectory);
        var lockPath = Path.Combine(workDirectory, "run.lock");
        try
        {
            var lockStream = new FileStream(
                lockPath,
                FileMode.OpenOrCreate,
                FileAccess.ReadWrite,
                FileShare.None,
                4096,
                FileOptions.None);
            var store = new ResumeStore(Path.GetFullPath(workDirectory), lockStream);
            Directory.CreateDirectory(store.JournalsDirectory);
            Directory.CreateDirectory(store.ChunksDirectory);
            return store;
        }
        catch (IOException ex)
        {
            throw new ReplayException("E_STATE_LOCKED",
                $"Cannot lock work directory '{workDirectory}'. Another render may be running.",
                new { workDirectory, error = ex.Message });
        }
    }

    public ResumeSnapshot? LoadSnapshot()
    {
        if (!File.Exists(SnapshotPath))
        {
            return null;
        }

        var snapshot = JsonIO.DeserializeFile<ResumeSnapshot>(SnapshotPath);
        if (snapshot.SchemaVersion != 1 || snapshot.TotalSegments < 0 || snapshot.CompletedSegments < 0
            || snapshot.CompletedSegments > snapshot.TotalSegments || snapshot.SafeOutputLength < 0)
        {
            throw new ReplayException("E_STATE_INVALID", "Resume snapshot is malformed.", new { path = SnapshotPath });
        }

        return snapshot;
    }

    public void WriteSnapshot(ResumeSnapshot snapshot)
    {
        snapshot.UpdatedAtUtc = DateTimeOffset.UtcNow;
        JsonIO.WriteAtomic(SnapshotPath, snapshot);
    }

    public void WriteJournal(SegmentJournal journal)
    {
        JsonIO.WriteAtomic(JournalPath(journal.Index), journal);
    }

    public List<SegmentJournal> LoadJournals(int completedSegments)
    {
        var journals = new List<SegmentJournal>();
        for (var index = 0; index < completedSegments; index++)
        {
            var path = JournalPath(index);
            if (!File.Exists(path))
            {
                throw new ReplayException("E_STATE_INVALID",
                    $"Completed segment {index} has no durable journal.",
                    new { workDirectory = WorkDirectory, journalPath = path });
            }

            var journal = JsonIO.DeserializeFile<SegmentJournal>(path);
            if (journal.Index != index || journal.FrameCount <= 0
                || journal.VideoSamples.Count != journal.FrameCount
                || journal.OutputEnd < journal.OutputStart)
            {
                throw new ReplayException("E_STATE_INVALID",
                    $"Segment journal {index} is malformed.",
                    new { path });
            }

            journals.Add(journal);
        }

        if (completedSegments > 0 && journals[0].Codec is null)
        {
            throw new ReplayException("E_STATE_INVALID",
                "First segment journal has no codec configuration.",
                new { path = JournalPath(0) });
        }

        return journals;
    }

    public void DiscardJournalsFrom(int firstIndex)
    {
        if (!Directory.Exists(JournalsDirectory))
        {
            return;
        }

        foreach (var file in Directory.EnumerateFiles(JournalsDirectory, "seg-*.json"))
        {
            var name = Path.GetFileNameWithoutExtension(file);
            var parts = name.Split('-');
            if (parts.Length == 2 && int.TryParse(parts[1], out var index) && index >= firstIndex)
            {
                JsonIO.TryDelete(file);
            }
        }
    }

    public void ClearChunks()
    {
        if (!Directory.Exists(ChunksDirectory))
        {
            return;
        }

        foreach (var file in Directory.EnumerateFiles(ChunksDirectory))
        {
            JsonIO.TryDelete(file);
        }
    }

    public void DiscardAll()
    {
        JsonIO.TryDelete(SnapshotPath);
        DiscardJournalsFrom(0);
        ClearChunks();
        var audioDirectory = Path.Combine(WorkDirectory, "audio");
        if (Directory.Exists(audioDirectory))
        {
            foreach (var file in Directory.EnumerateFiles(audioDirectory))
            {
                JsonIO.TryDelete(file);
            }
        }

        var other = Directory.EnumerateFiles(WorkDirectory, "*.tmp-*").ToList();
        foreach (var file in other)
        {
            JsonIO.TryDelete(file);
        }
    }

    public void DiscardRenderStateKeepingAudio()
    {
        JsonIO.TryDelete(SnapshotPath);
        DiscardJournalsFrom(0);
        ClearChunks();
    }

    public string ChunkPath(int segmentIndex)
        => Path.Combine(ChunksDirectory, $"{_runToken}-seg-{segmentIndex:D6}.h264.partial");

    public string JournalPath(int segmentIndex)
        => Path.Combine(JournalsDirectory, $"seg-{segmentIndex:D6}.json");

    public static string HashFile(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024);
        return Convert.ToHexString(SHA256.HashData(stream));
    }

    public static string HashBytes(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes));

    public void Dispose() => _lockStream.Dispose();
}
