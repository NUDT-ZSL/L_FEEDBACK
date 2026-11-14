using System.Diagnostics;
using System.Text;
using ReplayForge.Utilities;

namespace ReplayForge.Media;

public sealed class FfmpegResult
{
    public int ExitCode { get; init; }

    public string StandardOutput { get; init; } = "";

    public string StandardError { get; init; } = "";
}

public sealed class Ffmpeg
{
    public Ffmpeg(string executablePath)
    {
        ExecutablePath = executablePath;
    }

    public string ExecutablePath { get; }

    public static Ffmpeg Locate(string? explicitPath, string? manifestDirectory)
    {
        var candidates = new List<string>();
        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            var candidate = Path.IsPathRooted(explicitPath) || manifestDirectory is null
                ? explicitPath
                : Path.Combine(manifestDirectory, explicitPath);
            candidates.Add(Path.GetFullPath(candidate));
        }

        var environment = Environment.GetEnvironmentVariable("FFMPEG_PATH");
        if (!string.IsNullOrWhiteSpace(environment))
        {
            candidates.Add(environment);
        }

        var executableName = OperatingSystem.IsWindows() ? "ffmpeg.exe" : "ffmpeg";
        candidates.Add(Path.Combine(AppContext.BaseDirectory, executableName));
        candidates.Add(Path.Combine(AppContext.BaseDirectory, "tools", executableName));

        foreach (var candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                return new Ffmpeg(candidate);
            }
        }

        var pathVariable = Environment.GetEnvironmentVariable("PATH") ?? "";
        foreach (var directory in pathVariable.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            var candidate = Path.Combine(directory, executableName);
            if (File.Exists(candidate))
            {
                return new Ffmpeg(candidate);
            }
        }

        throw new ReplayException(
            "E_FFMPEG_NOT_FOUND",
            "FFmpeg was not found. Set ffmpegPath in the manifest, FFMPEG_PATH, or publish the platform runtime package.",
            new { searched = candidates });
    }

    public async Task<FfmpegResult> RunAsync(
        IEnumerable<string> arguments,
        string? workingDirectory = null,
        CancellationToken cancellationToken = default)
        => await RunAsync(arguments, workingDirectory, null, cancellationToken).ConfigureAwait(false);

    public async Task<FfmpegResult> RunAsync(
        IEnumerable<string> arguments,
        string? workingDirectory,
        Func<Stream, CancellationToken, Task>? stdoutConsumer,
        CancellationToken cancellationToken = default)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = ExecutablePath,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        if (workingDirectory is not null)
        {
            startInfo.WorkingDirectory = workingDirectory;
        }

        var executableDirectory = Path.GetDirectoryName(Path.GetFullPath(ExecutablePath));
        if (!string.IsNullOrWhiteSpace(executableDirectory))
        {
            if (OperatingSystem.IsWindows())
            {
                var currentPath = startInfo.Environment.TryGetValue("PATH", out var existingPath)
                    ? existingPath
                    : Environment.GetEnvironmentVariable("PATH") ?? "";
                startInfo.Environment["PATH"] = executableDirectory + Path.PathSeparator + currentPath;
            }
            else
            {
                var currentLibraryPath = startInfo.Environment.TryGetValue("LD_LIBRARY_PATH", out var existingLibraryPath)
                    ? existingLibraryPath
                    : Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? "";
                startInfo.Environment["LD_LIBRARY_PATH"] =
                    executableDirectory + Path.PathSeparator + currentLibraryPath;
            }
        }

        foreach (var argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using var process = new Process { StartInfo = startInfo };
        try
        {
            if (!process.Start())
            {
                throw new ReplayException("E_FFMPEG_FAILED", $"Failed to start FFmpeg: {ExecutablePath}");
            }
        }
        catch (Exception ex) when (ex is not ReplayException)
        {
            throw new ReplayException("E_FFMPEG_FAILED", $"Failed to start FFmpeg: {ex.Message}");
        }

        process.StandardInput.Close();
        var stderrTask = process.StandardError.ReadToEndAsync(cancellationToken);
        Task<string>? stdoutTask = null;
        Task? consumerTask = null;
        if (stdoutConsumer is null)
        {
            stdoutTask = process.StandardOutput.ReadToEndAsync(cancellationToken);
        }
        else
        {
            consumerTask = stdoutConsumer(process.StandardOutput.BaseStream, cancellationToken);
        }
        try
        {
            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch
            {
                // The process may have exited between the checks.
            }

            throw;
        }

        var stdout = stdoutTask is null ? "" : await stdoutTask.ConfigureAwait(false);
        if (consumerTask is not null)
        {
            await consumerTask.ConfigureAwait(false);
        }

        return new FfmpegResult
        {
            ExitCode = process.ExitCode,
            StandardOutput = stdout,
            StandardError = await stderrTask.ConfigureAwait(false)
        };
    }

    public async Task RunCheckedAsync(
        IEnumerable<string> arguments,
        string operation,
        string? workingDirectory = null,
        CancellationToken cancellationToken = default)
    {
        var allArguments = new List<string>
        {
            "-hide_banner",
            "-nostdin",
            "-y",
            "-loglevel",
            "warning"
        };
        allArguments.AddRange(arguments);

        var result = await RunAsync(allArguments, workingDirectory, cancellationToken).ConfigureAwait(false);
        if (result.ExitCode != 0)
        {
            throw new ReplayException(
                "E_FFMPEG_FAILED",
                $"{operation} failed with FFmpeg exit code {result.ExitCode}.",
                new
                {
                    operation,
                    exitCode = result.ExitCode,
                    stderr = Truncate(result.StandardError, 12000),
                    stdout = Truncate(result.StandardOutput, 4000)
                });
        }
    }

    private static string Truncate(string value, int length)
        => value.Length <= length ? value : value[..length] + "...";
}
