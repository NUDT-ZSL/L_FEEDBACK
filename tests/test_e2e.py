import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from cinema_master.pipeline import run_mastering
from cinema_master.samplegen import generate_sample
from cinema_master.wavio import inspect_wav, write_wav


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _base_config(tracks, target_lufs=-23.0):
    return {
        "schema_version": 1,
        "target_sample_rate": 48000,
        "output_layout": "5.1",
        "target_lufs": target_lufs,
        "true_peak_limit_dbtp": -1.0,
        "max_relative_gain_db": 4.0,
        "loudness": {"tolerance_lu": 0.5, "iteration_limit": 10},
        "compression": {
            "enabled": True,
            "threshold_dbfs": -30.0,
            "ratio": 2.5,
            "attack_ms": 15.0,
            "release_ms": 180.0,
            "knee_db": 6.0,
        },
        "dialogue_detection": {
            "frame_ms": 400.0,
            "overlap": 0.75,
            "threshold_dbfs": -45.0,
            "min_segment_ms": 180.0,
        },
        "limiter": {
            "lookahead_ms": 10.0,
            "release_ms": 120.0,
            "oversample": 4,
            "iteration_limit": 10,
            "safety_margin_db": 0.25,
        },
        "output": {
            "basename": "master",
            "audio_codec": "aac",
            "aac_bitrate": "256k",
            "bits_per_sample": 24,
        },
        "tracks": tracks,
    }


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.input_dir = os.path.join(cls.temp.name, "input")
        cls.output_dir = os.path.join(cls.temp.name, "output")
        generate_sample(cls.input_dir)
        cls.report = run_mastering(cls.input_dir, cls.output_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_resampling_and_channel_mask_records(self):
        tracks = {track["file"]: track for track in self.report["tracks"]}
        self.assertEqual(tracks["dialogue.wav"]["sample_rate_before"], 44100)
        self.assertEqual(tracks["dialogue.wav"]["sample_rate_after"], 48000)
        self.assertTrue(tracks["dialogue.wav"]["resampled"])
        self.assertEqual(tracks["music_51.wav"]["sample_rate_before"], 48000)
        self.assertEqual(tracks["effects_51.wav"]["sample_rate_before"], 32000)
        self.assertEqual(
            tracks["music_51.wav"]["input_channel_mask"], "0x0000003f"
        )
        self.assertEqual(
            tracks["effects_51.wav"]["channel_mask_source"], "implicit_5.1"
        )
        for track in tracks.values():
            self.assertEqual(track["output_timeline_frames"], 432000)

    def test_sample_generator_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            first = os.path.join(temp, "first")
            second = os.path.join(temp, "second")
            generate_sample(first, duration=2.0)
            generate_sample(second, duration=2.0)
            for name in (
                "dialogue.wav",
                "music_51.wav",
                "effects_51.wav",
                "config.json",
            ):
                self.assertEqual(
                    _sha(os.path.join(first, name)),
                    _sha(os.path.join(second, name)),
                    name,
                )

    def test_activity_segments_and_compression_report(self):
        activity = self.report["dialogue_activity"]
        self.assertGreaterEqual(activity["segment_count"], 3)
        self.assertGreater(activity["total_active_sec"], 3.0)
        self.assertEqual(activity["frame_samples"], 19200)
        self.assertEqual(activity["hop_samples"], 4800)
        compression = self.report["compression"]
        self.assertTrue(compression["enabled"])
        self.assertGreater(compression["max_reduction_db"], 0.0)
        self.assertNotEqual(
            compression["pre_compression_program_lufs"],
            compression["post_compression_program_lufs"],
        )

    def test_loudness_true_peak_and_dialogue_priority(self):
        loudness = self.report["loudness"]
        self.assertLessEqual(
            abs(loudness["program_lufs"] - loudness["target_lufs"]), 0.5
        )
        self.assertTrue(loudness["within_tolerance"])
        self.assertLessEqual(self.report["true_peak"]["measured_dbtp"], -1.0)
        for offset in loudness["relative_offsets_db"].values():
            self.assertLessEqual(offset, 0.05)
        self.assertTrue(
            loudness["dialogue_priority"][
                "all_other_groups_at_or_below_dialogue"
            ]
        )

    def test_wav_contract_and_packaging(self):
        wav_path = self.report["output"]["wav"]["path"]
        info = inspect_wav(wav_path)
        self.assertEqual(info["format_name"], "WAVE_FORMAT_EXTENSIBLE")
        self.assertEqual(info["channel_mask"], 0x3F)
        self.assertEqual(info["channels"], 6)
        self.assertEqual(info["data_frames"], 432000)
        container = self.report["output"]["container"]
        self.assertTrue(container["duration_match"])
        self.assertLessEqual(abs(container["duration_delta_sec"]), 0.001)
        self.assertEqual(container["channel_layout"], "5.1")
        self.assertTrue(container["metadata_verified"]["channel_layout"])
        self.assertTrue(container["metadata_verified"]["loudness_metadata"])

    def test_pcm_mp4_option(self):
        with tempfile.TemporaryDirectory() as temp:
            input_dir = os.path.join(temp, "input")
            output_dir = os.path.join(temp, "output")
            generate_sample(input_dir, duration=3.0)
            config_path = os.path.join(input_dir, "config.json")
            config = _read_json(config_path)
            config["output"]["audio_codec"] = "pcm"
            config["output"]["basename"] = "pcm_master"
            _write_json(config_path, config)
            report = run_mastering(input_dir, output_dir)
            self.assertEqual(report["output"]["container"]["codec"], "pcm_s24le")
            self.assertTrue(report["output"]["container"]["duration_match"])

    def test_two_runs_have_identical_wav_and_report(self):
        wav_path = self.report["output"]["wav"]["path"]
        report_path = self.report["output"]["report_path"]
        before_wav = _sha(wav_path)
        before_report = _sha(report_path)
        rerun = run_mastering(self.input_dir, self.output_dir)
        self.assertEqual(before_wav, _sha(wav_path))
        self.assertEqual(before_report, _sha(report_path))
        self.assertEqual(before_wav, rerun["output"]["wav"]["sha256"])
        self.assertFalse(
            [
                name
                for name in os.listdir(self.output_dir)
                if name.startswith(".cinema-master-")
            ]
        )

    def test_invalid_json_writes_error_report_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            input_dir = os.path.join(temp, "input")
            output_dir = os.path.join(temp, "output")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "config.json"), "w") as handle:
                handle.write("{not json")
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cinema_master",
                    input_dir,
                    output_dir,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(process.returncode, 0)
            report = _read_json(os.path.join(output_dir, "report.json"))
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["error"]["code"], "E_CONFIG_INVALID_JSON")

    def test_missing_track_writes_error_report(self):
        with tempfile.TemporaryDirectory() as temp:
            input_dir = os.path.join(temp, "input")
            output_dir = os.path.join(temp, "output")
            os.makedirs(input_dir)
            config = _base_config(
                [{"file": "missing.wav", "group": "dialogue"}]
            )
            _write_json(os.path.join(input_dir, "config.json"), config)
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cinema_master",
                    input_dir,
                    output_dir,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(process.returncode, 0)
            report = _read_json(os.path.join(output_dir, "report.json"))
            self.assertEqual(report["error"]["code"], "E_INPUT_MISSING")

    def test_channel_mismatch_writes_error_report(self):
        with tempfile.TemporaryDirectory() as temp:
            input_dir = os.path.join(temp, "input")
            output_dir = os.path.join(temp, "output")
            os.makedirs(input_dir)
            write_wav(
                os.path.join(input_dir, "stereo.wav"),
                np.zeros((4800, 2)),
                48000,
                0x3,
                24,
                extensible=True,
            )
            config = _base_config(
                [
                    {
                        "file": "stereo.wav",
                        "group": "dialogue",
                        "channel_map": ["FC"],
                    }
                ]
            )
            _write_json(os.path.join(input_dir, "config.json"), config)
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cinema_master",
                    input_dir,
                    output_dir,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(process.returncode, 0)
            report = _read_json(os.path.join(output_dir, "report.json"))
            self.assertEqual(report["error"]["code"], "E_CHANNEL_MISMATCH")

    def test_ffmpeg_nonzero_writes_error_report(self):
        with tempfile.TemporaryDirectory() as temp:
            input_dir = os.path.join(temp, "input")
            output_dir = os.path.join(temp, "output")
            fake_bin = os.path.join(temp, "bin")
            os.makedirs(fake_bin)
            generate_sample(input_dir, duration=3.0)
            fake_ffmpeg = os.path.join(fake_bin, "ffmpeg")
            with open(fake_ffmpeg, "w", encoding="utf-8") as handle:
                handle.write("#!/bin/sh\necho 'forced ffmpeg failure' >&2\nexit 17\n")
            os.chmod(fake_ffmpeg, 0o755)
            environment = os.environ.copy()
            environment["PATH"] = fake_bin + os.pathsep + environment.get("PATH", "")
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cinema_master",
                    input_dir,
                    output_dir,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertNotEqual(process.returncode, 0)
            report = _read_json(os.path.join(output_dir, "report.json"))
            self.assertEqual(report["error"]["code"], "E_FFMPEG_FAILED")
            self.assertFalse(
                [
                    name
                    for name in os.listdir(output_dir)
                    if name.startswith(".cinema-master-")
                ]
            )

    def test_packaged_duration_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            input_dir = os.path.join(temp, "input")
            output_dir = os.path.join(temp, "output")
            fake_bin = os.path.join(temp, "bin")
            os.makedirs(fake_bin)
            generate_sample(input_dir, duration=3.0)
            fake_ffprobe = os.path.join(fake_bin, "ffprobe")
            payload = {
                "streams": [
                    {
                        "codec_name": "aac",
                        "codec_type": "audio",
                        "channels": 6,
                        "channel_layout": "5.1",
                        "sample_rate": "48000",
                        "duration": "0.500000",
                    }
                ],
                "format": {
                    "duration": "0.500000",
                    "format_name": "mov,mp4",
                    "size": "1",
                },
            }
            with open(fake_ffprobe, "w", encoding="utf-8") as handle:
                handle.write("#!/bin/sh\n")
                handle.write("cat <<'EOF'\n")
                handle.write(json.dumps(payload))
                handle.write("\nEOF\n")
            os.chmod(fake_ffprobe, 0o755)
            environment = os.environ.copy()
            environment["PATH"] = fake_bin + os.pathsep + environment.get("PATH", "")
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cinema_master",
                    input_dir,
                    output_dir,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertNotEqual(process.returncode, 0)
            report = _read_json(os.path.join(output_dir, "report.json"))
            self.assertEqual(report["error"]["code"], "E_DURATION_MISMATCH")


if __name__ == "__main__":
    unittest.main()

