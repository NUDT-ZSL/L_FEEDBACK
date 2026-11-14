using System.Globalization;
using System.Text;
using ReplayForge.Contracts;

namespace ReplayForge.Pipeline;

public static class VideoFilterGraph
{
    public static List<string> BuildSegmentCommand(
        Manifest manifest,
        IReadOnlyList<ValidatedVideoLayer> layers,
        int firstFrame,
            int frameCount,
        string outputPath)
    {
        var timeline = manifest.Timeline;
        var fps = timeline.OutputFps;
        var start = firstFrame / (double)fps;
        var end = (firstFrame + frameCount) / (double)fps;
        var duration = frameCount / (double)fps;
        var filter = new StringBuilder();
        filter.Append(
            $"color=c=black:s={timeline.Width}x{timeline.Height}:r={fps}:d={F(duration)},format=rgba[base];");

        var args = new List<string>();
        var activeLayers = new List<(ValidatedVideoLayer Layer, int InputIndex, double LayerStart)>();
        foreach (var layer in layers)
        {
            var layerStart = layer.MappedStartSeconds;
            if (layerStart >= end - 1e-9)
            {
                continue;
            }

            var inputIndex = activeLayers.Count;
            args.Add("-i");
            args.Add(layer.ResolutionPath);
            activeLayers.Add((layer, inputIndex, layerStart));
        }

        for (var i = 0; i < activeLayers.Count; i++)
        {
            var (layer, inputIndex, layerStart) = activeLayers[i];
            var layout = layer.Config.Layout;
            var trimStart = Math.Max(start, layerStart);
            var padStart = Math.Max(0, layerStart - start);
            filter.Append(
                $"[{inputIndex}:v]setpts=PTS-STARTPTS+{F(layerStart)}/TB," +
                $"fps={fps}:start_time={F(start)}," +
                $"trim=start={F(trimStart)}:end={F(end)}," +
                "setpts=PTS-STARTPTS," +
                $"scale={layout.Width}:{layout.Height}:flags=bilinear," +
                "format=rgba," +
                $"colorchannelmixer=rr={F(layout.Opacity)}:gg={F(layout.Opacity)}:" +
                $"bb={F(layout.Opacity)}:aa={F(layout.Opacity)}," +
                $"tpad=start_duration={F(padStart)}:start_mode=add:color=black@0," +
                $"tpad=stop_mode=clone:stop_duration={F(duration)},setsar=1[layer{i}];");
        }

        var current = "base";
        for (var i = 0; i < activeLayers.Count; i++)
        {
            var layout = activeLayers[i].Layer.Config.Layout;
            var next = $"mix{i}";
            filter.Append(
                $"[{current}][layer{i}]overlay=x={layout.X}:y={layout.Y}:format=auto:" +
                $"alpha=premultiplied:eof_action=pass:repeatlast=0:shortest=0[{next}]");
            if (i < activeLayers.Count - 1)
            {
                filter.Append(';');
            }

            current = next;
        }

        filter.Append($";[{current}]format=yuv420p[vout]");
        var gop = Math.Max(1, frameCount);
        args.AddRange(new[]
        {
            "-filter_complex", filter.ToString(),
            "-map", "[vout]",
            "-frames:v", frameCount.ToString(CultureInfo.InvariantCulture),
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            "-g", gop.ToString(CultureInfo.InvariantCulture),
            "-keyint_min", gop.ToString(CultureInfo.InvariantCulture),
            "-sc_threshold", "0",
            "-bf", "0",
            "-threads", "1",
            "-x264-params",
            $"aud=1:repeat-headers=1:keyint={gop}:min-keyint={gop}:scenecut=0:open-gop=0:cabac=1:8x8dct=1",
            "-fflags", "+bitexact",
            "-flags:v", "+bitexact",
            "-f", "h264",
            outputPath
        });
        return args;
    }

    private static string F(double value) => value.ToString("0.######", CultureInfo.InvariantCulture);
}
