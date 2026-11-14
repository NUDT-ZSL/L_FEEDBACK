import io
import json
import os
import tempfile
import unittest

from audio_sync.cli import main
from audio_sync.constants import FRAME_BYTES, MAX_BUFFER_BYTES
from audio_sync.engine import AudioPipeline
from audio_sync.protocol import iter_packet_files
from audio_sync.simulator import (
    META_FILENAME,
    TRUTH_FILENAME,
    ImpairmentConfig,
    simulate,
)
from audio_sync.verifier import verify_run
from audio_sync.wavio import read_wav_duration


SEED = 20261114


def make_dataset(root, **overrides):
    config = ImpairmentConfig(seed=SEED, **overrides)
    return simulate(
        output_dir=root,
        stream_count=overrides.get("stream_count", 3),
        frame_count=overrides.get("frame_count", 120),
        config=config,
    )


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def test_deterministic_byte_identical_reruns(self):
        first = tempfile.mkdtemp()
        second = tempfile.mkdtemp()
        make_dataset(first)
        make_dataset(second)
        for name in ("stream_0.pkt", "stream_1.pkt", TRUTH_FILENAME, META_FILENAME):
            with open(os.path.join(first, name), "rb") as a, open(
                os.path.join(second, name), "rb"
            ) as b:
                self.assertEqual(a.read(), b.read(), name)

    def test_loss_and_fault_injection_reflected(self):
        make_dataset(self.root, loss_rate=0.3, inject_faults=True)
        with open(os.path.join(self.root, META_FILENAME), encoding="utf-8") as h:
            meta = json.load(h)
        self.assertTrue(any(meta["lost_frames"].values()))
        # Fault records make stream_0 larger than a clean multiple of records.
        size = os.path.getsize(os.path.join(self.root, "stream_0.pkt"))
        clean_records = len(meta["schedule"]["0"])
        self.assertGreater(size, clean_records * (8 + FRAME_BYTES))

    def test_ground_truth_origin_fields(self):
        make_dataset(self.root)
        with open(os.path.join(self.root, TRUTH_FILENAME), encoding="utf-8") as h:
            first = json.loads(h.readline())
        stream_truth = first["streams"]["0"]
        self.assertEqual(stream_truth["source"], "synthetic_tone")
        self.assertEqual(stream_truth["original_frame_index"], 0)
        self.assertEqual(stream_truth["sample_offset"], 0)
        self.assertIn("lost", stream_truth)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.out = tempfile.mkdtemp()
        make_dataset(
            self.root,
            loss_rate=0.15,
            reorder_rate=0.1,
            burst_loss=True,
            inject_faults=True,
        )
        self.wav = os.path.join(self.out, "mixed.wav")
        self.log = os.path.join(self.out, "frames.jsonl")

    def _run(self, depth=4):
        with open(os.path.join(self.out, "stderr.log"), "w", encoding="utf-8") as err:
            pipeline = AudioPipeline(
                packet_paths=list(iter_packet_files(self.root)),
                wav_path=self.wav,
                log_path=self.log,
                buffer_depth=depth,
                meta=json.load(
                    open(
                        os.path.join(self.root, META_FILENAME),
                        encoding="utf-8",
                    )
                ),
                stderr_sink=err,
            )
            return pipeline.run()

    def test_pipeline_outputs_and_verification(self):
        summary = self._run()
        self.assertEqual(summary.total_frames, 120)
        self.assertTrue(os.path.exists(self.wav))
        self.assertAlmostEqual(read_wav_duration(self.wav), 120 * 0.02, places=2)

        with open(self.log, encoding="utf-8") as handle:
            logs = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(logs), 120)
        self.assertEqual([row["frame_index"] for row in logs], list(range(120)))
        for row in logs:
            self.assertEqual(len(row["gains"]), 3)
            self.assertLessEqual(row["mix_peak"], 32767)
            for stream in row["streams"].values():
                self.assertLess(stream["buffer_memory_bytes"], MAX_BUFFER_BYTES)

        report = verify_run(
            os.path.join(self.root, TRUTH_FILENAME), self.log, self.wav
        )
        self.assertTrue(report["ok"], report["failures"])
        self.assertGreater(report["plc_frames"], 0)

        with open(os.path.join(self.out, "stderr.log"), encoding="utf-8") as handle:
            stderr_text = handle.read()
        self.assertIn("E_BAD_LENGTH", stderr_text)
        self.assertIn("E_TRUNCATED", stderr_text)

    def test_cli_success_and_bad_depth_exit_codes(self):
        code = main(
            [
                "run",
                "--input-dir",
                self.root,
                "--wav",
                self.wav,
                "--log",
                self.log,
                "--buffer-depth",
                "4",
            ]
        )
        self.assertEqual(code, 0)

        missing_dir = os.path.join(self.out, "does_not_exist")
        code = main(
            [
                "run",
                "--input-dir",
                missing_dir,
                "--wav",
                self.wav,
                "--log",
                self.log,
            ]
        )
        self.assertEqual(code, 3)

    def test_buffer_overflow_backpressure_with_shallow_buffer(self):
        # A depth-2 buffer with a burst of clustered arrivals must apply
        # backpressure: overflow is recorded and the oldest frame is dropped.
        jitter_root = tempfile.mkdtemp()
        config = ImpairmentConfig(
            seed=SEED + 1,
            loss_rate=0.0,
            reorder_rate=0.0,
            jitter_mean_ms=0.0,
            jitter_sigma_ms=0.0,
        )
        metadata = simulate(
            output_dir=jitter_root,
            stream_count=2,
            frame_count=80,
            config=config,
        )
        # Cluster every arrival before the first 20 ms playout tick.
        for entries in metadata["schedule"].values():
            for entry in entries:
                entry["arrival_ms"] = 0.0
        wav = os.path.join(self.out, "shallow.wav")
        log = os.path.join(self.out, "shallow.jsonl")
        with open(os.path.join(self.out, "shallow.err"), "w", encoding="utf-8") as err:
            pipeline = AudioPipeline(
                packet_paths=list(iter_packet_files(jitter_root)),
                wav_path=wav,
                log_path=log,
                buffer_depth=2,
                meta=metadata,
                stderr_sink=err,
            )
            summary = pipeline.run()
        with open(log, encoding="utf-8") as handle:
            events = [
                event
                for row in (json.loads(line) for line in handle if line.strip())
                for event in row["events"]
            ]
        self.assertGreater(summary.backpressure_events, 0)
        self.assertTrue(any("buffer_overflow" in event for event in events))
        self.assertTrue(any("backpressure" in event for event in events))


if __name__ == "__main__":
    unittest.main()
