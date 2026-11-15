from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import wave
from decimal import Decimal
from pathlib import Path
from unittest import mock

from hls_audio_builder import builder
from hls_audio_builder.cli import main
from hls_audio_builder.media import EncodedProbe, count_adts_frames


ROOT = Path(__file__).resolve().parents[1]


def run_cli(config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hls_audio_builder", "--config", str(config_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.samples = cls.root / "samples"
        cls.output = cls.root / "output"
        generated = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "generate_samples.py"),
                "--output-dir",
                str(cls.samples),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if generated.returncode != 0:
            raise RuntimeError(generated.stderr)

        cls.config = cls.samples / "config.json"
        payload = json.loads(cls.config.read_text(encoding="utf-8"))
        payload["output_dir"] = str(cls.output)
        write_json(cls.config, payload)

        cls.first_build = run_cli(cls.config)
        if cls.first_build.returncode != 0:
            raise RuntimeError(cls.first_build.stderr)
        cls.report_path = cls.output / "build_report.json"
        cls.first_report = cls.report_path.read_bytes()
        cls.first_playlists = {
            path.relative_to(cls.output).as_posix(): path.read_bytes()
            for path in cls.output.rglob("*.m3u8")
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_outputs_playlists_alignment_and_report(self) -> None:
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["segment_count"], 4)
        self.assertEqual(report["sample_rate_hz"], 44100)
        self.assertEqual(
            report["sample_boundaries"], [0, 88200, 176400, 264600, 321930]
        )
        self.assertEqual(
            [variant["bitrate_kbps"] for variant in report["variants"]],
            [64, 128, 192],
        )

        media_playlists = {}
        boundaries_by_bitrate = {}
        uris_by_bitrate = {}
        for bitrate in (64, 128, 192):
            playlist_path = self.output / f"{bitrate}k" / "index.m3u8"
            text = playlist_path.read_text(encoding="utf-8")
            self.assertIn("#EXT-X-TARGETDURATION:2", text)
            self.assertIn("#EXT-X-MEDIA-SEQUENCE:0", text)
            self.assertIn("#EXT-X-PLAYLIST-TYPE:VOD", text)
            self.assertIn("#EXT-X-ENDLIST", text)

            durations = []
            uris = []
            lines = text.splitlines()
            for index, line in enumerate(lines):
                if line.startswith("#EXTINF:"):
                    durations.append(
                        Decimal(line.split(":", 1)[1].split(",", 1)[0])
                    )
                    uris.append(lines[index + 1])
            running = Decimal("0")
            boundaries = [running]
            for duration in durations:
                running += duration
                boundaries.append(running)
            media_playlists[bitrate] = text
            boundaries_by_bitrate[bitrate] = boundaries
            uris_by_bitrate[bitrate] = uris

        self.assertEqual(boundaries_by_bitrate[64], boundaries_by_bitrate[128])
        self.assertEqual(boundaries_by_bitrate[64], boundaries_by_bitrate[192])
        self.assertEqual(
            boundaries_by_bitrate[64],
            [Decimal("0"), Decimal("2"), Decimal("4"), Decimal("6"), Decimal("7.3")],
        )

        frame_counts_by_index = {index: set() for index in range(4)}
        for variant in report["variants"]:
            bitrate = variant["bitrate_kbps"]
            self.assertEqual(variant["segment_count"], 4)
            self.assertEqual(variant["total_duration_seconds"], 7.3)
            self.assertGreater(variant["total_file_size_bytes"], 0)
            self.assertGreater(variant["encoded_average_bitrate_bps"], 0)
            for index, segment in enumerate(report["segments"]):
                encoded = segment["files"][f"{bitrate}k"]
                output_file = self.output / encoded["uri"]
                self.assertTrue(output_file.is_file())
                self.assertEqual(file_hash(output_file), encoded["sha256"])
                self.assertEqual(count_adts_frames(output_file), encoded["aac_frame_count"])
                frame_counts_by_index[index].add(encoded["aac_frame_count"])
                planned = Decimal(str(segment["duration_seconds"]))
                actual = Decimal(str(encoded["encoded_duration_seconds"]))
                self.assertLessEqual(
                    abs(actual - planned), Decimal(1024) / Decimal(44100)
                )

        for counts in frame_counts_by_index.values():
            self.assertEqual(len(counts), 1)

        master = (self.output / "master.m3u8").read_text(encoding="utf-8")
        for bitrate in (64, 128, 192):
            self.assertIn(f"{bitrate}k/index.m3u8", master)
        self.assertEqual(master.count("#EXT-X-STREAM-INF:"), 3)
        self.assertIn("BANDWIDTH=", master)
        self.assertIn("AVERAGE-BANDWIDTH=", master)

        for bitrate in (64, 128, 192):
            for uri in uris_by_bitrate[bitrate]:
                self.assertFalse(uri.startswith("/"))
                self.assertNotIn("..", Path(uri).parts)
                self.assertTrue((self.output / f"{bitrate}k" / uri).is_file())

    def test_repeat_with_overwrite_is_byte_stable(self) -> None:
        payload = json.loads(self.config.read_text(encoding="utf-8"))
        payload["overwrite"] = True
        overwrite_config = self.samples / "config_overwrite.json"
        write_json(overwrite_config, payload)
        result = run_cli(overwrite_config)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.report_path.read_bytes(), self.first_report)

        for relative, content in self.first_playlists.items():
            self.assertEqual((self.output / relative).read_bytes(), content)

        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        for segment in report["segments"]:
            for encoded in segment["files"].values():
                self.assertEqual(
                    file_hash(self.output / encoded["uri"]), encoded["sha256"]
                )

    def test_existing_output_requires_overwrite_exit_4(self) -> None:
        result = run_cli(self.config)
        self.assertEqual(result.returncode, 4)
        self.assertIn("set overwrite to true", result.stderr)

    def test_config_errors_exit_2(self) -> None:
        base = json.loads(self.config.read_text(encoding="utf-8"))

        missing = dict(base)
        missing.pop("variants")
        missing_path = self.root / "missing_variants.json"
        write_json(missing_path, missing)
        self.assertEqual(run_cli(missing_path).returncode, 2)

        unsorted = json.loads(json.dumps(base))
        unsorted["variants"] = list(reversed(unsorted["variants"]))
        unsorted_path = self.root / "unsorted.json"
        write_json(unsorted_path, unsorted)
        self.assertEqual(run_cli(unsorted_path).returncode, 2)

        bad_duration = json.loads(json.dumps(base))
        bad_duration["segment_duration"] = 31
        bad_duration_path = self.root / "bad_duration.json"
        write_json(bad_duration_path, bad_duration)
        self.assertEqual(run_cli(bad_duration_path).returncode, 2)

        mismatched_rate = self.root / "audio_48k.wav"
        with wave.open(str(mismatched_rate), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(48000)
            handle.writeframes(b"\x00\x00" * 48000)
        rate_config = json.loads(json.dumps(base))
        rate_config["output_dir"] = str(self.root / "rate_output")
        rate_config["variants"][1]["input"] = str(mismatched_rate)
        rate_path = self.samples / "rate_mismatch.json"
        write_json(rate_path, rate_config)
        rate_result = run_cli(rate_path)
        self.assertEqual(rate_result.returncode, 2)
        self.assertIn("sample rates must match", rate_result.stderr)

    def test_consistency_failure_exit_3(self) -> None:
        payload = json.loads(self.config.read_text(encoding="utf-8"))
        payload["output_dir"] = str(self.root / "consistency_output")
        payload["overwrite"] = True
        consistency_config = self.samples / "consistency.json"
        write_json(consistency_config, payload)

        original = builder.probe_encoded_segment

        def corrupt_probe(ffprobe: str, source: Path) -> EncodedProbe:
            result = original(ffprobe, source)
            if "128k" in source.parts and source.name == "segment_00001.aac":
                return EncodedProbe(
                    sample_rate=result.sample_rate,
                    channels=result.channels,
                    packet_count=result.packet_count + 1,
                )
            return result

        stderr = io.StringIO()
        with mock.patch.object(builder, "probe_encoded_segment", side_effect=corrupt_probe):
            with contextlib.redirect_stderr(stderr):
                exit_code = main(["--config", str(consistency_config)])
        self.assertEqual(exit_code, 3)
        self.assertIn("128k segment 1", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

