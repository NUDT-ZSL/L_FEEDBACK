import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from voice_mixer.exporter import export_project
from voice_mixer.parser import parse_project
from voice_mixer.timeline import build_timeline


class LargeScaleTest(unittest.TestCase):
    def test_tens_of_thousands_of_events(self):
        tmp = tempfile.mkdtemp(prefix="mixer-large-")
        self.addCleanup(shutil.rmtree, tmp, True)
        speaker_count = 8
        events = []
        tick = 0
        pair_count = 2500
        for pair in range(pair_count):
            for speaker_index in range(speaker_count):
                speaker = f"player_{speaker_index}"
                start_tick = tick
                end_tick = tick + 80
                events.append({"tick": start_tick, "type": "start", "speaker": speaker})
                if pair % 5 == 0:
                    events.append({"tick": start_tick + 10, "type": "ptt", "speaker": speaker})
                events.append({"tick": end_tick, "type": "end", "speaker": speaker})
                tick = end_tick + 20

        project_dict = {
            "sample_rate": 16000,
            "channel_layout": "mono",
            "tick_rate": 1000,
            "channels": [
                {"id": "squad", "gain_db": -1.0, "muted": False},
                {"id": "command", "gain_db": 3.0, "muted": False},
            ],
            "speakers": [
                {
                    "id": f"player_{index}",
                    "seed": 1000 + index,
                    "base_frequency": 100.0 + index * 8.0,
                    "harmonic_ratio": 0.25,
                    "pitch_semitones": (index % 5) - 2,
                    "source_duration_ms": 500.0,
                    "bound_channel": "command" if index == 0 else "squad",
                    "pan": 0.0,
                }
                for index in range(speaker_count)
            ],
            "events": events,
            "export": {
                "output_wav": os.path.join(tmp, "large.wav"),
                "manifest": os.path.join(tmp, "large.json"),
                "work_dir": os.path.join(tmp, "work"),
                "chunk_samples": 16000,
                "intro_ms": 200.0,
                "outro_ms": 300.0,
                "enable_prompts": True,
            },
        }
        project = parse_project(project_dict)
        timeline = build_timeline(project)
        started = time.perf_counter()
        code, manifest = export_project(project, timeline, logger=lambda _: None)
        elapsed = time.perf_counter() - started
        self.assertEqual(code, 0)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(timeline.fragments), speaker_count * pair_count)
        self.assertEqual(len(timeline.ptt_bursts), speaker_count * (pair_count // 5))
        self.assertGreater(manifest["chunk_count"], 1)
        self.assertTrue(os.path.isfile(project.export.output_wav))
        self.assertLess(elapsed, 30.0)


if __name__ == "__main__":
    unittest.main()
