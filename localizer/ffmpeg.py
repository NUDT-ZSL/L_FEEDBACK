from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from localizer.errors import EXIT_MUX, LocalizerError, OperationCancelled


VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 100
DURATION_TOLERANCE_MS = 10.0


class FFmpegExecutionError(RuntimeError):
    def __init__(self, command: list[str], returncode: int, stderr: str) -> None:
        super().__init__(f"FFmpeg exited with code {returncode}")
        self.command = command
        self.returncode = returncode
        self.stderr = stderr


class FFmpegRunner:
    def __init__(
        self,
        ffmpeg: str | Path | None,
        ffprobe: str | Path | None,
        cancel: threading.Event,
    ) -> None:
        self.cancel = cancel
        self.ffmpeg = self._resolve_binary(ffmpeg, "ffmpeg")
        if ffprobe is None:
            sibling = self.ffmpeg.with_name(
                self.ffmpeg.name.replace("ffmpeg", "ffprobe", 1)
            )
            ffprobe = sibling if sibling.is_file() else None
        self.ffprobe = self._resolve_binary(ffprobe, "ffprobe")
        self._guard = threading.Lock()
        self._processes: dict[int, subprocess.Popen[bytes]] = {}

    @staticmethod
    def _resolve_binary(value: str | Path | None, name: str) -> Path:
        candidate: Path | None = None
        if value:
            candidate = Path(value).expanduser()
        else:
            found = shutil.which(name)
            if found:
                candidate = Path(found)
        if candidate is None or not candidate.is_file():
            raise LocalizerError(
                "E_MUX",
                f"{name} executable was not found",
                exit_code=EXIT_MUX,
                details={"executable": name},
            )
        return candidate.resolve()

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["LC_ALL"] = "C"
        env["LANG"] = "C"
        env["TZ"] = "UTC"
        env["AV_LOG_FORCE_NOCOLOR"] = "1"
        env.pop("FFREPORT", None)
        return env

    def _run(
        self,
        binary: Path,
        arguments: list[str],
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        if self.cancel.is_set():
            raise OperationCancelled()
        prefix = ["-hide_banner"]
        if binary != self.ffprobe:
            prefix.append("-nostdin")
        command = [os.fspath(binary), *prefix, *arguments]
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt"
            else 0
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            env=self._env(),
            creationflags=creationflags,
        )
        with self._guard:
            self._processes[process.pid] = process
        started = time.monotonic()
        try:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    if self.cancel.is_set():
                        self._stop_process(process)
                        process.communicate()
                        raise OperationCancelled()
                    if (
                        timeout_seconds is not None
                        and time.monotonic() - started > timeout_seconds
                    ):
                        self._stop_process(process)
                        process.communicate()
                        raise FFmpegExecutionError(
                            command,
                            process.returncode or -1,
                            f"process exceeded timeout of {timeout_seconds:.1f}s",
                        )
        finally:
            with self._guard:
                self._processes.pop(process.pid, None)

        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            raise FFmpegExecutionError(command, process.returncode, stderr_text)
        return stdout.decode("utf-8", errors="replace")

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()

    def terminate_all(self) -> None:
        with self._guard:
            processes = list(self._processes.values())
        for process in processes:
            self._stop_process(process)

    def probe(self, path: Path) -> dict[str, Any]:
        output = self._run(
            self.ffprobe,
            [
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                os.fspath(path),
            ],
            timeout_seconds=60,
        )
        return json.loads(output)

    def build_black_video(self, path: Path, total_ms: int) -> int:
        frame_count = max(1, (total_ms + (1000 // VIDEO_FPS) - 1) // (1000 // VIDEO_FPS))
        source = f"color=c=black:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:r={VIDEO_FPS}"
        self._run(
            self.ffmpeg,
            [
                "-y",
                "-f",
                "lavfi",
                "-i",
                source,
                "-frames:v",
                str(frame_count),
                "-an",
                "-c:v",
                "mpeg4",
                "-q:v",
                "2",
                "-g",
                str(VIDEO_FPS),
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(VIDEO_FPS),
                "-video_track_timescale",
                "1000",
                "-fflags",
                "+bitexact",
                "-flags:v",
                "+bitexact",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-metadata",
                "creation_time=1970-01-01T00:00:00Z",
                "-write_tmcd",
                "0",
                "-movflags",
                "+faststart",
                "-threads",
                "1",
                "-f",
                "mp4",
                os.fspath(path),
            ],
            timeout_seconds=600,
        )
        return frame_count

    def mux(
        self,
        video: Path,
        audio_tracks: list[tuple[str, Path]],
        destination: Path,
        *,
        sample_rate: int,
    ) -> None:
        arguments = ["-y", "-i", os.fspath(video)]
        for _, wav_path in audio_tracks:
            arguments.extend(["-i", os.fspath(wav_path)])
        arguments.extend(["-map", "0:v:0"])
        for index in range(len(audio_tracks)):
            arguments.extend(["-map", f"{index + 1}:a:0"])
        arguments.extend(
            [
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                str(sample_rate),
                "-ac",
                "2",
            ]
        )
        for index, (language, _) in enumerate(audio_tracks):
            arguments.extend(
                ["-metadata:s:a:" + str(index), f"language={language}"]
            )
            disposition = "default" if index == 0 else "0"
            arguments.extend(
                ["-disposition:a:" + str(index), disposition]
            )
        arguments.extend(
            [
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-metadata",
                "creation_time=1970-01-01T00:00:00Z",
                "-fflags",
                "+bitexact",
                "-flags:v",
                "+bitexact",
                "-flags:a",
                "+bitexact",
                "-video_track_timescale",
                "1000",
                "-write_tmcd",
                "0",
                "-use_editlist",
                "1",
                "-movflags",
                "+faststart",
                "-threads",
                "1",
                "-f",
                "mp4",
                os.fspath(destination),
            ]
        )
        self._run(self.ffmpeg, arguments, timeout_seconds=900)

    def validate_mux(
        self,
        path: Path,
        expected_languages: list[str],
        *,
        expected_duration_ms: int,
        expected_sample_rate: int,
    ) -> dict[str, Any]:
        metadata = self.probe(path)
        streams = metadata.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        format_duration = _duration_seconds(metadata.get("format", {}))
        video_duration = _duration_seconds(video_streams[0]) if video_streams else None

        problems: list[str] = []
        measurements: list[dict[str, Any]] = []
        if not video_streams:
            problems.append("no video stream")
        if len(audio_streams) != len(expected_languages):
            problems.append(
                f"expected {len(expected_languages)} audio streams, got {len(audio_streams)}"
            )

        for index, audio in enumerate(audio_streams):
            duration = _duration_seconds(audio)
            if duration is None:
                duration = format_duration
            diff_ms = (
                abs(duration - video_duration) * 1000.0
                if duration is not None and video_duration is not None
                else None
            )
            tags = {str(k).lower(): v for k, v in (audio.get("tags") or {}).items()}
            item = {
                "index": index,
                "duration_seconds": duration,
                "video_duration_seconds": video_duration,
                "difference_ms": diff_ms,
                "language": tags.get("language"),
                "channels": audio.get("channels"),
                "sample_rate": audio.get("sample_rate"),
            }
            measurements.append(item)
            if diff_ms is None or diff_ms > DURATION_TOLERANCE_MS + 1e-6:
                problems.append(
                    f"audio stream {index} duration difference is {diff_ms}ms"
                )
            expected_language = (
                expected_languages[index] if index < len(expected_languages) else None
            )
            if item["language"] != expected_language:
                problems.append(
                    f"audio stream {index} language is {item['language']!r}, "
                    f"expected {expected_language!r}"
                )
            if item["channels"] != 2:
                problems.append(f"audio stream {index} is not stereo")
            if str(item["sample_rate"]) != str(expected_sample_rate):
                problems.append(
                    f"audio stream {index} sample rate is {item['sample_rate']}, "
                    f"expected {expected_sample_rate}"
                )

        if format_duration is None:
            problems.append("container duration is unavailable")
        elif abs(format_duration * 1000.0 - expected_duration_ms) > DURATION_TOLERANCE_MS:
            problems.append(
                f"container duration differs from timeline by "
                f"{abs(format_duration * 1000.0 - expected_duration_ms):.3f}ms"
            )

        result = {
            "format_duration_seconds": format_duration,
            "video_duration_seconds": video_duration,
            "audio": measurements,
            "problems": problems,
        }
        if problems:
            raise LocalizerError(
                "E_MUX",
                "muxed MP4 failed duration/language validation",
                exit_code=EXIT_MUX,
                details=result,
            )
        return result


def _duration_seconds(obj: dict[str, Any]) -> float | None:
    raw = obj.get("duration")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    duration_ts = obj.get("duration_ts")
    time_base = obj.get("time_base")
    if duration_ts is not None and isinstance(time_base, str) and "/" in time_base:
        try:
            numerator, denominator = time_base.split("/", 1)
            return float(duration_ts) * float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None
