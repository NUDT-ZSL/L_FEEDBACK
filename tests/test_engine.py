import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptive_mixer.config import load_config
from adaptive_mixer.engine import MixEngine, RenderInterrupted
from adaptive_mixer.errors import MixerError
from adaptive_mixer.limiter import true_peak_db
from adaptive_mixer.loudness import integrated_loudness
from adaptive_mixer.sample_gen import generate_tracks
from adaptive_mixer.wavio import read_wav, read_wav_chunks


def build_workspace(root: Path, seconds: float = 6.0,
                    frame_rate: int = 100):
    total_frames = int(seconds * frame_rate)
    generate_tracks(root / "tracks",
                    ["strings", "percussion", "ambience"], seconds)
    config = {
        "sample_rate": 48000,
        "frame_rate": frame_rate,
        "total_frames": total_frames,
        "target_lufs": -23.0,
        "crossfade_ms": 120.0,
        "crossfade_type": "s_curve",
        "tracks": [
            {"name": "strings", "path": "tracks/strings.wav"},
            {"name": "percussion", "path": "tracks/percussion.wav"},
            {"name": "ambience", "path": "tracks/ambience.wav"},
        ],
        "scenes": [
            {"scene_id": "explore", "name": "探索",
             "start_frame": 0, "end_frame": total_frames // 3,
             "target_lufs": -26.0,
             "active_tracks": ["strings", "ambience"]},
            {"scene_id": "battle", "name": "战斗",
             "start_frame": total_frames // 3,
             "end_frame": 2 * total_frames // 3,
             "target_lufs": -18.0, "crossfade_type": "equal_power",
             "active_tracks": ["percussion", "strings"]},
            {"scene_id": "cutscene", "name": "过场",
             "start_frame": 2 * total_frames // 3,
             "end_frame": total_frames, "target_lufs": -24.0,
             "crossfade_type": "linear",
             "active_tracks": ["ambience", "strings"]},
        ],
        "output_format": {"sample_rate": 48000, "channels": 1,
                          "subtype": "PCM_24"},
    }
    config_path = root / "scene.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False),
                           encoding="utf-8")
    return config_path


class EngineEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config_path = build_workspace(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_render_meets_spec(self):
        cfg = load_config(self.config_path)
        output = self.root / "out" / "mix.wav"
        engine = MixEngine(cfg, self.root, output,
                           state_file=self.root / "state.json")
        result = engine.render()
        report = result.report

        samples, sr = read_wav(output)
        self.assertEqual(sr, 48000)
        self.assertEqual(len(samples), cfg.total_samples)
        chunks = [name for name, _ in read_wav_chunks(output)["chunks"]]
        self.assertEqual(chunks, ["fmt ", "fact", "data"])

        # True peak ceiling.
        self.assertLessEqual(report["loudness"]["true_peak_db"], -1.0 + 1e-6)
        self.assertLessEqual(true_peak_db(samples), -1.0 + 1e-6)

        # Global loudness within +/-0.5 LU of the duration-weighted target.
        targets = np.array([s.target_lufs for s in cfg.scenes])
        weights = np.array([s.end_frame - s.start_frame
                            for s in cfg.scenes], dtype=float)
        weighted_target = float(np.sum(targets * weights / weights.sum()))
        measured = integrated_loudness(samples, 48000)
        self.assertLessEqual(
            abs(measured - weighted_target), 0.5,
            msg=f"{measured} vs target {weighted_target}")

        # Timeline alignment exact.
        self.assertEqual(report["timeline"]["alignment_error_samples"], 0)
        for scene in report["scenes"]:
            self.assertEqual(scene["alignment_error_samples"], 0)
            self.assertLessEqual(
                abs(scene["actual_lufs"] - scene["target_lufs"]), 0.5)
        self.assertEqual(len(report["crossfades"]), len(cfg.scenes) - 1)
        for event in report["crossfades"]:
            self.assertIn(event["curve_type"],
                          ("s_curve", "equal_power", "linear"))

        # Report file content is serialisable and error code prefix rule.
        report_path = self.root / "out" / "mix.report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False),
                               encoding="utf-8")
        json.loads(report_path.read_text(encoding="utf-8"))

    def test_track_length_mismatch_error(self):
        cfg = load_config(self.config_path)
        short = self.root / "tracks" / "strings.wav"
        data, sr = read_wav(short)
        from adaptive_mixer.wavio import write_wav_24bit
        write_wav_24bit(short, data[:-100], sr)
        with self.assertRaises(MixerError) as ctx:
            MixEngine(cfg, self.root, self.root / "x.wav")
        self.assertEqual(ctx.exception.code, "E_TRACK_LENGTH")


if __name__ == "__main__":
    unittest.main()
