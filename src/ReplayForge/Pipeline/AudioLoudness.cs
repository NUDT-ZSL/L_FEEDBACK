using System.Buffers.Binary;
using System.Globalization;
using ReplayForge.Media;
using ReplayForge.Utilities;

namespace ReplayForge.Pipeline;

public sealed class LoudnessResult
{
    public double IntegratedLufs { get; init; }

    public double GainDb { get; init; }

    public int GatingBlocks { get; init; }

    public double PeakSample { get; init; }
}

public static class AudioLoudness
{
    private const int SampleRate = 48000;
    private const int WindowSamples = 19200;
    private const int HopSamples = 4800;

    public static async Task<LoudnessResult> MeasureAsync(
        Ffmpeg ffmpeg,
        string inputPath,
        CancellationToken cancellationToken)
    {
        var analyzer = new Analyzer();
        var arguments = new[]
        {
            "-hide_banner",
            "-nostdin",
            "-loglevel", "error",
            "-i", inputPath,
            "-vn",
            "-ac", "2",
            "-ar", SampleRate.ToString(CultureInfo.InvariantCulture),
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-"
        };

        var result = await ffmpeg.RunAsync(
            arguments,
            null,
            (stream, token) => analyzer.ConsumeAsync(stream, token),
            cancellationToken).ConfigureAwait(false);
        if (result.ExitCode != 0)
        {
            throw new ReplayException("E_FFMPEG_FAILED",
                $"Loudness scan failed with FFmpeg exit code {result.ExitCode}.",
                new { stderr = result.StandardError });
        }

        return analyzer.Complete();
    }

    public static double GainForTarget(LoudnessResult measured, double targetLufs)
    {
        if (double.IsNaN(measured.IntegratedLufs) || measured.IntegratedLufs < -90)
        {
            return 0;
        }

        var gain = targetLufs - measured.IntegratedLufs;
        if (measured.PeakSample > 1e-9)
        {
            var peakLimitedGain = 20 * Math.Log10(0.98 / measured.PeakSample);
            gain = Math.Min(gain, peakLimitedGain);
        }

        return Math.Clamp(gain, -30, 30);
    }

    private sealed class Analyzer
    {
        private readonly Biquad _shelfLeft = new(
            1.53512485958697, -2.69169618940638, 1.19839281085285,
            -1.69065929318241, 0.73248077421585);
        private readonly Biquad _shelfRight = new(
            1.53512485958697, -2.69169618940638, 1.19839281085285,
            -1.69065929318241, 0.73248077421585);
        private readonly Biquad _highPassLeft = new(
            1.0, -2.0, 1.0,
            -1.99004745483398, 0.99007225036621);
        private readonly Biquad _highPassRight = new(
            1.0, -2.0, 1.0,
            -1.99004745483398, 0.99007225036621);
        private readonly double[] _ringLeft = new double[WindowSamples];
        private readonly double[] _ringRight = new double[WindowSamples];
        private readonly List<double> _blockEnergy = new();
        private long _sampleFrames;
        private double _sumLeft;
        private double _sumRight;
        private double _peak;
        private byte[] _remainder = Array.Empty<byte>();

        public async Task ConsumeAsync(Stream stream, CancellationToken cancellationToken)
        {
            var buffer = new byte[32768];
            int read;
            while ((read = await stream.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
            {
                var combined = new byte[_remainder.Length + read];
                _remainder.CopyTo(combined, 0);
                buffer.AsSpan(0, read).CopyTo(combined.AsSpan(_remainder.Length));
                var usable = combined.Length / 8 * 8;
                for (var offset = 0; offset < usable; offset += 8)
                {
                    var left = BinaryPrimitives.ReadSingleLittleEndian(combined.AsSpan(offset, 4));
                    var right = BinaryPrimitives.ReadSingleLittleEndian(combined.AsSpan(offset + 4, 4));
                    AddFrame(left, right);
                }

                _remainder = combined.AsSpan(usable).ToArray();
            }
        }

        public LoudnessResult Complete()
        {
            if (_blockEnergy.Count == 0)
            {
                return new LoudnessResult
                {
                    IntegratedLufs = double.NegativeInfinity,
                    GainDb = 0,
                    PeakSample = _peak
                };
            }

            var absoluteGated = _blockEnergy
                .Where(energy => LoudnessFromEnergy(energy) >= -70.0)
                .ToList();
            if (absoluteGated.Count == 0)
            {
                return new LoudnessResult
                {
                    IntegratedLufs = double.NegativeInfinity,
                    GainDb = 0,
                    GatingBlocks = 0,
                    PeakSample = _peak
                };
            }

            var preliminary = LoudnessFromEnergy(absoluteGated.Average());
            var relativeThreshold = preliminary - 10.0;
            var relativeGated = absoluteGated
                .Where(energy => LoudnessFromEnergy(energy) >= relativeThreshold)
                .ToList();
            var integrated = LoudnessFromEnergy(relativeGated.Average());
            return new LoudnessResult
            {
                IntegratedLufs = integrated,
                GainDb = 0,
                GatingBlocks = relativeGated.Count,
                PeakSample = _peak
            };
        }

        private void AddFrame(float left, float right)
        {
            _peak = Math.Max(_peak, Math.Max(Math.Abs(left), Math.Abs(right)));
            var filteredLeft = _highPassLeft.Process(_shelfLeft.Process(left));
            var filteredRight = _highPassRight.Process(_shelfRight.Process(right));
            var leftSquare = filteredLeft * filteredLeft;
            var rightSquare = filteredRight * filteredRight;
            var ringIndex = (int)(_sampleFrames % WindowSamples);
            if (_sampleFrames >= WindowSamples)
            {
                _sumLeft -= _ringLeft[ringIndex];
                _sumRight -= _ringRight[ringIndex];
            }

            _ringLeft[ringIndex] = leftSquare;
            _ringRight[ringIndex] = rightSquare;
            _sumLeft += leftSquare;
            _sumRight += rightSquare;
            _sampleFrames++;

            if (_sampleFrames >= WindowSamples && (_sampleFrames - WindowSamples) % HopSamples == 0)
            {
                var energy = (_sumLeft + _sumRight) / WindowSamples;
                _blockEnergy.Add(energy);
            }
        }

        private static double LoudnessFromEnergy(double energy)
            => energy <= 1e-20 ? double.NegativeInfinity : -0.691 + 10 * Math.Log10(energy);
    }

    private sealed class Biquad
    {
        private readonly double _b0;
        private readonly double _b1;
        private readonly double _b2;
        private readonly double _a1;
        private readonly double _a2;
        private double _x1;
        private double _x2;
        private double _y1;
        private double _y2;

        public Biquad(double b0, double b1, double b2, double a1, double a2)
        {
            _b0 = b0;
            _b1 = b1;
            _b2 = b2;
            _a1 = a1;
            _a2 = a2;
        }

        public double Process(double input)
        {
            var output = _b0 * input + _b1 * _x1 + _b2 * _x2 - _a1 * _y1 - _a2 * _y2;
            _x2 = _x1;
            _x1 = input;
            _y2 = _y1;
            _y1 = output;
            return output;
        }
    }
}
