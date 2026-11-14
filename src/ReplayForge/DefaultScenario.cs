using ReplayForge.Contracts;

namespace ReplayForge;

public static class DefaultScenario
{
    public static Manifest Create(double durationSeconds, int seed, string? ffmpegPath)
    {
        var manifest = new Manifest
        {
            SchemaVersion = 1,
            Timeline = new TimelineConfig
            {
                MasterStartPtsSeconds = 10.0,
                DurationSeconds = durationSeconds,
                OutputFps = 30,
                Width = 640,
                Height = 360,
                SyncToleranceSeconds = 0.020,
                TargetLoudnessLufs = -16.0,
                SegmentDurationSeconds = 1.0
            },
            OutputPath = "output/final.mp4",
            WorkDirectory = "work",
            SummaryPath = "summary.json",
            FfmpegPath = ffmpegPath
        };

        manifest.Videos.Add(new VideoLayerConfig
        {
            Id = "main",
            Role = "main",
            Path = "media/main-high-24fps.mp4",
            TimelineStartSeconds = 0.0,
            Layout = new LayoutConfig { X = 0, Y = 0, Width = 640, Height = 360, Opacity = 1.0 }
        });
        manifest.Videos.Add(new VideoLayerConfig
        {
            Id = "player_pov",
            Role = "player",
            Path = "media/player-main-30fps.mp4",
            TimelineStartSeconds = 0.5,
            Layout = new LayoutConfig { X = 16, Y = 16, Width = 240, Height = 136, Opacity = 0.92 }
        });
        manifest.Videos.Add(new VideoLayerConfig
        {
            Id = "commentator_cam",
            Role = "commentary_camera",
            Path = "media/commentator-baseline-25fps.mp4",
            TimelineStartSeconds = 0.6,
            Layout = new LayoutConfig { X = 432, Y = 236, Width = 192, Height = 108, Opacity = 0.88 }
        });
        manifest.Audio = new AudioConfig
        {
            Id = "commentary",
            Path = "media/commentary-48k-stereo.aac",
            TimelineStartSeconds = 0.0,
            SampleRate = 48000,
            Channels = 2
        };
        manifest.Generator = new GeneratorConfig
        {
            Seed = seed,
            DurationSeconds = durationSeconds,
            Videos = new List<GeneratorVideoConfig>
            {
                new()
                {
                    Id = "main",
                    FileName = "main-high-24fps.mp4",
                    Pattern = "testsrc2",
                    Width = 320,
                    Height = 180,
                    Fps = 24,
                    Profile = "High",
                    StartPtsSeconds = 10.0,
                    HueDegrees = seed % 31
                },
                new()
                {
                    Id = "player_pov",
                    FileName = "player-main-30fps.mp4",
                    Pattern = "testsrc2",
                    Width = 256,
                    Height = 144,
                    Fps = 30,
                    Profile = "Main",
                    StartPtsSeconds = 10.5,
                    HueDegrees = 20 + seed % 29
                },
                new()
                {
                    Id = "commentator_cam",
                    FileName = "commentator-baseline-25fps.mp4",
                    Pattern = "smptebars",
                    Width = 192,
                    Height = 108,
                    Fps = 25,
                    Profile = "Baseline",
                    StartPtsSeconds = 10.6,
                    HueDegrees = 340 - seed % 17
                }
            },
            Audio = new GeneratorAudioConfig
            {
                FileName = "commentary-48k-stereo.aac",
                SampleRate = 48000,
                Channels = 2,
                ToneHz = 523.25
            }
        };
        return manifest;
    }
}
