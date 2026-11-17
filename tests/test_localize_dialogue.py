from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "localize_dialogue.py"
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def write_pcm16_wav(path: Path, sample_rate: int, samples: np.ndarray) -> None:
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    channels = samples.shape[1]
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def make_dialogue(sample_rate: int, duration: float) -> np.ndarray:
    time = np.arange(int(round(sample_rate * duration))) / sample_rate
    return np.stack(
        [
            0.20 * np.sin(2 * np.pi * 440.0 * time),
            0.20 * np.sin(2 * np.pi * 660.0 * time),
        ],
        axis=1,
    )


def make_impulse_effect(
    sample_rate: int,
    duration: float,
    pulse_times: tuple[float, ...],
) -> np.ndarray:
    samples = np.zeros((int(round(sample_rate * duration)), 2), dtype=np.float64)
    for pulse_time in pulse_times:
        start = int(round(pulse_time * sample_rate))
        if start >= samples.shape[0]:
            continue
        stop = min(samples.shape[0], start + 120)
        envelope = np.hanning(stop - start)
        samples[start:stop, 0] += 0.65 * envelope
        samples[start:stop, 1] -= 0.55 * envelope
    return samples


def mux_master(
    path: Path,
    tracks: list[Path],
    codec: str = "pcm_s16le",
) -> None:
    command = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for track in tracks:
        command.extend(["-i", str(track)])
    for ordinal in range(len(tracks)):
        command.extend(["-map", f"{ordinal}:a:0"])
    command.extend(["-c:a", codec, str(path)])
    subprocess.run(command, check=True, capture_output=True, text=True)


def transcode_audio_tracks(source: Path, destination: Path, codec: str) -> None:
    subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-map",
            "0:a:1",
            "-c:a",
            codec,
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def decode_track(
    path: Path,
    ordinal: int,
    sample_rate: int,
    channels: int,
) -> np.ndarray:
    command = [
        FFMPEG,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        f"0:a:{ordinal}",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, channels)


def probe_audio_streams(path: Path) -> list[dict]:
    result = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index,codec_name,channels,sample_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"]


class LocalizeDialogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if FFMPEG is None or FFPROBE is None:
            raise unittest.SkipTest("ffmpeg and ffprobe must be available in PATH")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="uid155-test-")
        self.work = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build_standard_master(
        self,
        *,
        dialogue_duration: float = 2.0,
        effect_duration: float = 2.5,
        effect_sample_rate: int = 48000,
        effect_channels: int = 2,
        extra_track: bool = False,
        container_suffix: str = ".mkv",
    ) -> tuple[Path, Path, Path]:
        dialogue_path = self.work / "original_dialogue.wav"
        effect_path = self.work / "original_effect.wav"
        localized_path = self.work / "localized_dialogue.wav"
        master_path = self.work / f"master{container_suffix}"

        dialogue = make_dialogue(48000, dialogue_duration)
        write_pcm16_wav(dialogue_path, 48000, dialogue)

        effect = make_impulse_effect(
            effect_sample_rate,
            effect_duration,
            (0.50, 1.50, 2.25),
        )
        if effect_channels == 1:
            effect = effect[:, :1]
        write_pcm16_wav(effect_path, effect_sample_rate, effect)

        localized_time = np.arange(int(round(44100 * 1.5))) / 44100
        localized = 0.25 * np.sin(2 * np.pi * 220.0 * localized_time)
        write_pcm16_wav(localized_path, 44100, localized)

        tracks = [dialogue_path, effect_path]
        if extra_track:
            extra_path = self.work / "ignored.wav"
            write_pcm16_wav(extra_path, 48000, make_dialogue(48000, 0.75))
            tracks.append(extra_path)
        mux_master(master_path, tracks)
        return master_path, localized_path, effect_path

    def run_tool(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(TOOL), *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_positive_offset_aligns_and_preserves_effect_samples(self) -> None:
        master, localized, effect_path = self.build_standard_master()
        output = self.work / "localized.mkv"
        report_path = self.work / "report.json"

        result = self.run_tool(
            "--input",
            str(master),
            "--localized-dialogue",
            str(localized),
            "--output",
            str(output),
            "--report",
            str(report_path),
            "--sound-offset-ms",
            "250",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout_report = json.loads(result.stdout)
        file_report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(stdout_report, file_report)
        self.assertEqual(file_report["status"], "success")
        self.assertIsNone(file_report["error_code"])
        self.assertEqual(file_report["ignored_audio_tracks"], 0)

        dialogue_info = file_report["dialogue_track"]
        self.assertEqual(dialogue_info["source_sample_rate"], 44100)
        self.assertEqual(dialogue_info["sample_rate"], 48000)
        self.assertEqual(dialogue_info["channels"], 2)
        self.assertEqual(dialogue_info["sample_count"], 72000)
        self.assertTrue(dialogue_info["resampled"])

        effect_info = file_report["sound_effect_track"]
        self.assertEqual(effect_info["sample_rate"], 48000)
        self.assertEqual(effect_info["channels"], 2)
        self.assertEqual(effect_info["offset_ms"], 250.0)
        self.assertEqual(effect_info["leading_padding_samples"], 12000)
        self.assertEqual(effect_info["truncated_samples"], 60000)
        self.assertEqual(effect_info["retained_samples"], 60000)
        self.assertLessEqual(file_report["alignment"]["alignment_error_ms"], 5.0)
        self.assertTrue(file_report["alignment"]["within_tolerance"])
        self.assertTrue(file_report["output_format"]["pcm"])
        self.assertTrue(file_report["verification"]["passed"])

        timestamp = file_report["processing_timestamp"]
        self.assertTrue(timestamp.endswith("Z"))
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        streams = probe_audio_streams(output)
        self.assertEqual(len(streams), 2)
        self.assertEqual([stream["channels"] for stream in streams], [2, 2])
        self.assertEqual([int(stream["sample_rate"]) for stream in streams], [48000, 48000])

        source_effect = decode_track(effect_path, 0, 48000, 2)
        output_effect = decode_track(output, 1, 48000, 2)
        expected = np.zeros((72000, 2), dtype=np.float32)
        expected[12000:72000] = source_effect[:60000]
        np.testing.assert_array_equal(output_effect[:12000], 0.0)
        np.testing.assert_array_equal(output_effect, expected)

        source_peak = int(np.argmax(np.abs(source_effect[:, 0])))
        output_peak = int(np.argmax(np.abs(output_effect[:, 0])))
        alignment_error_ms = abs((output_peak - source_peak) / 48000 - 0.250) * 1000
        self.assertLessEqual(alignment_error_ms, 5.0)

        output_dialogue = decode_track(output, 0, 48000, 2)
        np.testing.assert_allclose(
            output_dialogue[:, 0],
            output_dialogue[:, 1],
            atol=1e-7,
            rtol=0.0,
        )
        progress_percentages = [
            int(line.split("%", 1)[0].split()[-1])
            for line in result.stderr.splitlines()
            if line.startswith("[progress]")
        ]
        self.assertEqual(progress_percentages, list(range(0, 101, 10)))
        self.assertEqual(list(self.work.glob(".localize_dialogue_*")), [])

    def test_negative_offset_trims_leading_effect_samples(self) -> None:
        master, localized, effect_path = self.build_standard_master(
            effect_duration=2.0
        )
        output = self.work / "negative.mkv"
        report_path = self.work / "negative.json"

        result = self.run_tool(
            str(master),
            str(localized),
            str(output),
            str(report_path),
            "--sound-offset-ms",
            "-300",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        effect_info = report["sound_effect_track"]
        self.assertEqual(effect_info["offset_ms"], -300.0)
        self.assertEqual(effect_info["offset_samples"], -14400)
        self.assertEqual(effect_info["truncated_leading_samples"], 14400)
        self.assertEqual(effect_info["truncated_trailing_samples"], 9600)
        self.assertEqual(effect_info["truncated_samples"], 24000)
        self.assertTrue(report["alignment"]["within_tolerance"])

        source_effect = decode_track(effect_path, 0, 48000, 2)
        output_effect = decode_track(output, 1, 48000, 2)
        expected = source_effect[14400:86400]
        self.assertEqual(output_effect.shape, expected.shape)
        np.testing.assert_array_equal(output_effect, expected)

        source_peak = int(np.argmax(np.abs(source_effect[:, 0])))
        output_peak = int(np.argmax(np.abs(output_effect[:, 0])))
        expected_peak = source_peak - 14400
        self.assertLessEqual(abs(output_peak - expected_peak), 1)
        self.assertLessEqual(abs(output_peak / 48000 - 0.2) * 1000, 5.0)

    def test_extra_track_and_effect_remix_resample(self) -> None:
        master, localized, _ = self.build_standard_master(
            effect_duration=1.2,
            effect_sample_rate=32000,
            effect_channels=1,
            extra_track=True,
        )
        output = self.work / "remixed.mkv"
        report_path = self.work / "remixed.json"

        result = self.run_tool(
            str(master),
            str(localized),
            str(output),
            str(report_path),
            "--sound-offset-ms",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["ignored_audio_tracks"], 1)
        effect_info = report["sound_effect_track"]
        self.assertEqual(effect_info["source_sample_rate"], 32000)
        self.assertEqual(effect_info["source_channels"], 1)
        self.assertTrue(effect_info["resampled"])
        self.assertTrue(effect_info["remixed"])
        self.assertEqual(len(probe_audio_streams(output)), 2)
        self.assertEqual(
            [stream["channels"] for stream in probe_audio_streams(output)],
            [2, 2],
        )

    def test_mp4_output_and_lossy_effect_source_use_pcm_fallback(self) -> None:
        master, localized, _ = self.build_standard_master(
            dialogue_duration=0.5,
            effect_duration=0.7,
            container_suffix=".mp4",
        )
        aac_master = self.work / "aac_master.mp4"
        transcode_audio_tracks(master, aac_master, "aac")
        output = self.work / "result.mp4"
        report_path = self.work / "aac.json"

        result = self.run_tool(
            str(aac_master),
            str(localized),
            str(output),
            str(report_path),
            "--sound-offset-ms",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(report["output_format"]["pcm"])
        self.assertEqual(report["output_format"]["source_effects_codec"], "aac")
        self.assertIsNotNone(report["output_format"]["fallback_reason"])
        self.assertTrue(report["verification"]["passed"])
        streams = probe_audio_streams(output)
        self.assertEqual(len(streams), 2)
        self.assertTrue(
            all(stream["codec_name"].startswith("pcm_") for stream in streams)
        )

    def test_missing_input_writes_file_not_found_report(self) -> None:
        missing = self.work / "missing.mkv"
        localized = self.work / "localized.wav"
        write_pcm16_wav(localized, 48000, np.zeros(4800))
        output = self.work / "output.mkv"
        report_path = self.work / "missing.json"

        result = self.run_tool(
            "-i",
            str(missing),
            "-l",
            str(localized),
            "-o",
            str(output),
            "-r",
            str(report_path),
        )

        self.assertNotEqual(result.returncode, 0)
        stdout_report = json.loads(result.stdout)
        file_report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(stdout_report, file_report)
        self.assertEqual(file_report["status"], "failure")
        self.assertEqual(file_report["error_code"], "FILE_NOT_FOUND")
        self.assertIsNone(file_report["dialogue_track"])
        self.assertFalse(output.exists())

    def test_invalid_offset_writes_invalid_argument_report(self) -> None:
        report_path = self.work / "invalid.json"
        result = self.run_tool(
            "--report",
            str(report_path),
            "--sound-offset-ms",
            "not-a-number",
        )

        self.assertNotEqual(result.returncode, 0)
        report = json.loads(result.stdout)
        self.assertEqual(report["error_code"], "INVALID_ARGUMENT")
        self.assertEqual(
            json.loads(report_path.read_text(encoding="utf-8")),
            report,
        )

    def test_single_track_input_is_unsupported(self) -> None:
        dialogue_path = self.work / "only.wav"
        write_pcm16_wav(dialogue_path, 48000, make_dialogue(48000, 0.4))
        master = self.work / "one_track.mkv"
        mux_master(master, [dialogue_path])
        localized = self.work / "localized.wav"
        write_pcm16_wav(
            localized,
            44100,
            np.zeros(int(44100 * 0.25), dtype=np.float32),
        )
        report_path = self.work / "unsupported.json"

        result = self.run_tool(
            str(master),
            str(localized),
            str(self.work / "output.mkv"),
            str(report_path),
        )

        self.assertNotEqual(result.returncode, 0)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failure")
        self.assertEqual(report["error_code"], "UNSUPPORTED_FORMAT")
        self.assertEqual(json.loads(result.stdout), report)

    def test_unwritable_output_writes_encoding_failed_report(self) -> None:
        master, localized, _ = self.build_standard_master(
            dialogue_duration=0.4,
            effect_duration=0.5,
        )
        report_path = self.work / "encoding.json"

        result = self.run_tool(
            str(master),
            str(localized),
            "/dev/null/impossible-output.mkv",
            str(report_path),
        )

        self.assertNotEqual(result.returncode, 0)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failure")
        self.assertEqual(report["error_code"], "ENCODING_FAILED")
        self.assertEqual(json.loads(result.stdout), report)


if __name__ == "__main__":
    unittest.main()
