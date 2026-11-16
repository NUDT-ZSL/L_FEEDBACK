import math
import unittest

import numpy as np

from rhythm_mixer.beatgrid import BeatGrid
from rhythm_mixer.config import load_inputs
from rhythm_mixer.engine import run

from tests._helpers import make_fixture, read_output


class PipelineTests(unittest.TestCase):
    def test_happy_path_outputs_valid_wav_and_report(self):
        input_dir, output_dir = make_fixture()
        wav_path, report_path = run(input_dir, output_dir)
        data, info, report, _, _ = read_output(output_dir)

        self.assertEqual(info.samplerate, 44100)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.subtype, "PCM_16")
        expected = int(round(8.0 * 44100))
        self.assertEqual(info.frames, expected)
        self.assertEqual(report["total_samples"], expected)

        target_peak = 10 ** (-1 / 20)
        self.assertLess(abs(float(np.max(np.abs(data))) - target_peak), 2e-4)

        required = {
            "sample_rate", "bpm", "events", "layer_mixed_samples",
            "beat_period_ms", "master_gain",
        }
        self.assertTrue(required.issubset(report))
        for event in report["events"]:
            self.assertEqual(
                set(event),
                {
                    "original_time_ms", "aligned_time_ms", "beat_index",
                    "snap_deviation_ms", "layer", "action", "gain",
                    "sample_index",
                },
            )
            self.assertAlmostEqual(
                event["snap_deviation_ms"],
                event["aligned_time_ms"] - event["original_time_ms"],
                places=6,
            )

    def test_report_sample_indices_match_beat_grid(self):
        input_dir, output_dir = make_fixture()
        chart = load_inputs(input_dir)
        grid = BeatGrid(chart.bpm, chart.sample_rate)
        run(input_dir, output_dir)
        _, _, report, _, _ = read_output(output_dir)
        for event in report["events"]:
            expected_sample = grid.sample_index_at_ms(event["aligned_time_ms"])
            self.assertEqual(event["sample_index"], expected_sample)
            self.assertAlmostEqual(
                event["aligned_time_ms"],
                grid.beat_time_ms(event["beat_index"]),
                places=6,
            )

    def test_dynamic_enable_disable_changes_mix_content(self):
        input_dir, output_dir = make_fixture()
        run(input_dir, output_dir)
        data, _, report, _, _ = read_output(output_dir)

        def rms(start_s, end_s):
            a = int(start_s * 44100)
            b = int(end_s * 44100)
            return float(np.sqrt(np.mean(data[a:b] ** 2)))

        # Drums enabled 0-4 s, disabled 4-6 s, enabled 6-8 s.
        drums_on_early = rms(0.2, 0.8)
        drums_on_late = rms(6.2, 7.8)
        self.assertGreater(drums_on_early, 0.02)
        self.assertGreater(drums_on_late, 0.02)

        # Melody only exists from beat 4 (2 s) to beat 14 (7 s).
        silence_before = rms(0.1, 0.9)
        melody_window = rms(2.4, 3.6)
        after_melody = rms(7.2, 7.9)
        self.assertGreater(melody_window, silence_before)
        self.assertGreater(melody_window, after_melody)

        # Every layer contributed a non-zero amount of samples.
        for name, count in report["layer_mixed_samples"].items():
            self.assertGreater(count, 0, name)

    def test_layer_muted_when_never_enabled(self):
        input_dir, output_dir = make_fixture()
        chart_path = f"{input_dir}/events.json"
        import json
        with open(chart_path, encoding="utf-8") as handle:
            events_doc = json.load(handle)
        events_doc["events"] = [
            event for event in events_doc["events"]
            if event["layer"] != "melody"
        ]
        with open(chart_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(events_doc, handle)
        run(input_dir, output_dir)
        _, _, report, _, _ = read_output(output_dir)
        self.assertEqual(report["layer_mixed_samples"]["melody"], 0)

    def test_gain_scales_segment_loudness(self):
        input_dir_a, output_a = make_fixture()
        input_dir_b, output_b = make_fixture()
        run(input_dir_a, output_a)
        data_a, _, _, _, _ = read_output(output_a)

        import json
        path = f"{input_dir_b}/events.json"
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        for event in doc["events"]:
            if event["action"] == "enable":
                event["gain"] = event.get("gain", 1.0) * 0.25
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(doc, handle)
        run(input_dir_b, output_b)
        data_b, _, report_b, _, _ = read_output(output_b)

        # Both outputs are peak normalized, but the quieter mix needs a
        # larger master gain to reach -1 dBFS.
        self.assertGreater(report_b["master_gain"], 1.0)
        self.assertEqual(data_a.shape, data_b.shape)

    def test_non_integer_bpm_end_to_end_snapping(self):
        input_dir, output_dir = make_fixture(bpm=121.0)
        # Rewrite fixture events to exact 121 BPM beat boundaries.
        import json
        beat_ms = 60.0 / 121.0 * 1000.0
        events_path = f"{input_dir}/events.json"
        doc = {
            "events": [
                {"time_ms": 0.0, "layer": "drums", "action": "enable"},
                {"time_ms": round(4 * beat_ms, 3), "layer": "bass",
                 "action": "enable", "gain": 0.7},
                {"time_ms": round(8 * beat_ms, 3), "layer": "melody",
                 "action": "enable"},
                {"time_ms": round(12 * beat_ms, 3), "layer": "drums",
                 "action": "disable"},
            ]
        }
        with open(events_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(doc, handle)
        run(input_dir, output_dir)
        _, _, report, _, _ = read_output(output_dir)
        for event in report["events"]:
            self.assertLess(abs(event["snap_deviation_ms"]), 0.5)
        self.assertEqual(
            [event["beat_index"] for event in report["events"]],
            [0, 4, 8, 12],
        )

    def test_repeated_enable_disable_is_idempotent(self):
        input_dir, output_dir = make_fixture()
        from rhythm_mixer.config import load_inputs as _load
        from rhythm_mixer.engine import plan_events
        from rhythm_mixer.mixer import LayerMixer

        chart = _load(input_dir)
        grid = BeatGrid(chart.bpm, chart.sample_rate)
        planned = plan_events(chart, grid)
        mixer = LayerMixer(chart, grid, planned, output_dir + "/x.wav")
        segments = mixer._segments_by_layer()
        # Drums: enable@0, disable@8 (twice), enable@12 (twice identical).
        drums = segments["drums"]
        starts = [start for start, _, _ in drums]
        self.assertEqual(starts, [0, grid.sample_index_at_ms(6000.0)])
        for start, end, gain in drums:
            self.assertGreater(end, start)
            self.assertTrue(math.isfinite(gain))


class BeatGridTests(unittest.TestCase):
    def test_snap_nearest_beat_and_deviation(self):
        grid = BeatGrid(120.0, 44100)
        snapped = grid.snap(2100.0, require_on_beat=False)
        self.assertEqual(snapped.beat_index, 4)
        self.assertAlmostEqual(snapped.aligned_ms, 2000.0)
        self.assertAlmostEqual(snapped.deviation_ms, -100.0)

    def test_half_beat_tie_snaps_forward(self):
        grid = BeatGrid(120.0, 44100)
        snapped = grid.snap(250.0, require_on_beat=False)
        self.assertEqual(snapped.beat_index, 1)
        self.assertEqual(snapped.deviation_ms, 250.0)

    def test_off_beat_beyond_ms_precision_rejected(self):
        grid = BeatGrid(121.0, 44100)
        from rhythm_mixer.errors import NotOnBeatError
        with self.assertRaises(NotOnBeatError):
            grid.snap(500.0)

    def test_non_integer_bpm_accepts_exact_beat_times(self):
        grid = BeatGrid(121.0, 44100)
        beat_s = 60.0 / 121.0
        exact_ms = round(3 * beat_s * 1000.0, 3)
        snapped = grid.snap(exact_ms)
        self.assertEqual(snapped.beat_index, 3)
        self.assertLess(abs(snapped.deviation_ms), 0.5)

    def test_total_samples_within_one_sample(self):
        grid = BeatGrid(121.0, 44100)
        duration_ms = 16 * (60.0 / 121.0) * 1000.0
        samples = grid.total_samples(duration_ms)
        naive = int(duration_ms / 1000.0 * 44100)
        self.assertLessEqual(abs(samples - naive), 1)


if __name__ == "__main__":
    unittest.main()
