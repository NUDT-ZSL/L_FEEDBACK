import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from audiopack.assets import LoadedClip
from audiopack.config import Config
from audiopack.normalize import normalize_track
from audiopack.renderer import (
    build_language_track,
    render_raw_track,
    total_frames,
    write_pcm16_wav,
)
from audiopack.scheduler import LanguageSchedule, Placement


def make_clip(event_id: str, frames: int, value: float) -> LoadedClip:
    samples = np.full(frames, value, dtype=np.float32)
    return LoadedClip(
        path=f"/tmp/{event_id}.wav",
        samples=samples,
        sample_rate=48000,
        frames=frames,
        content_hash=f"h-{event_id}",
        source_name=f"{event_id}.wav",
    )


def placement(event_id, start, frames, priority=5):
    return Placement(
        event_id=event_id, lang="zh", voice_key=event_id,
        source_name=f"{event_id}.wav", source_path=f"/tmp/{event_id}.wav",
        natural_start=start, start=start, frames=frames, end=start + frames,
        priority=priority, shift_samples=0, content_hash=f"h-{event_id}",
    )


class RendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()

    def test_total_frames_rounded_up_to_whole_ms(self) -> None:
        schedule = LanguageSchedule(
            "zh", [placement("A", 0, 4801)], []
        )
        # 4801 frames -> ceil to next multiple of 48 -> 4848.
        self.assertEqual(total_frames(schedule), 4848)

    def test_silence_gap_is_zero_filled(self) -> None:
        schedule = LanguageSchedule(
            "zh", [placement("A", 0, 48), placement("B", 192, 48)], []
        )
        clips = {"A": make_clip("A", 48, 0.5), "B": make_clip("B", 48, 0.5)}
        track = render_raw_track(schedule, clips, self.config)
        self.assertEqual(track.shape[0], 240)
        np.testing.assert_array_equal(track[48:192], np.zeros(144, np.float32))

    def test_touching_clips_get_equal_power_crossfade(self) -> None:
        frames = 480  # 10ms each, so 5ms fade fits
        schedule = LanguageSchedule(
            "zh", [placement("A", 0, frames), placement("B", frames, frames)], []
        )
        clips = {
            "A": make_clip("A", frames, 0.8),
            "B": make_clip("B", frames, 0.4),
        }
        track = render_raw_track(schedule, clips, self.config)
        fade = self.config.crossfade_frames  # 240 samples
        # Energy in the overlap stays bounded (no doubled-amplitude splat).
        overlap = track[frames - fade: frames]
        self.assertLessEqual(float(np.max(np.abs(overlap))), 1.0001)
        # Outside the crossfade, clips retain their original amplitude.
        self.assertAlmostEqual(float(track[0]), 0.8, places=5)
        self.assertAlmostEqual(float(track[-1]), 0.4, places=5)

    def test_rendering_is_deterministic(self) -> None:
        schedule = LanguageSchedule(
            "zh", [placement("A", 0, 480), placement("B", 480, 480)], []
        )
        clips = {
            "A": make_clip("A", 480, 0.6),
            "B": make_clip("B", 480, 0.3),
        }
        first = render_raw_track(schedule, clips, self.config)
        second = render_raw_track(schedule, clips, self.config)
        np.testing.assert_array_equal(first, second)

    def test_pcm16_wav_header_and_frame_count(self) -> None:
        samples = np.zeros(9600, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.wav")
            info = write_pcm16_wav(path, samples, self.config)
            on_disk = sf.info(path)
            self.assertEqual(on_disk.samplerate, 48000)
            self.assertEqual(on_disk.channels, 1)
            self.assertEqual(on_disk.subtype, "PCM_16")
            self.assertEqual(on_disk.frames, 9600)
            self.assertEqual(info["frames"], 9600)

    def test_build_track_returns_normalized_audio(self) -> None:
        schedule = LanguageSchedule("zh", [placement("A", 0, 4800)], [])
        clips = {"A": make_clip("A", 4800, 0.2)}
        track, norm = build_language_track(schedule, clips, self.config)
        self.assertEqual(track.shape[0], 4800)
        self.assertIsInstance(norm, normalize_track(np.zeros(1, np.float32),
                                                    self.config).__class__)

    def test_empty_schedule_renders_zero_length_track(self) -> None:
        schedule = LanguageSchedule("xx", [], [])
        track = render_raw_track(schedule, {}, self.config)
        self.assertEqual(track.shape[0], 0)
        self.assertEqual(total_frames(schedule), 0)
        with tempfile.TemporaryDirectory() as tmp:
            info = write_pcm16_wav(os.path.join(tmp, "empty.wav"),
                                   track, self.config)
            self.assertEqual(info["frames"], 0)


if __name__ == "__main__":
    unittest.main()
