import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from offline_audio_render import sources as sources_module
from offline_audio_render.renderer import Renderer
from offline_audio_render.scene import load_scene
import yaml


class StreamingMemoryTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="uid49_stream_"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write_long_mono(self, sample_rate=8000, seconds=660):
        # 660 seconds exceeds the 10-minute streaming threshold.
        path = self.dir / "long_mono.wav"
        frames = sample_rate * seconds
        block = 8000
        with sf.SoundFile(str(path), mode="w", samplerate=sample_rate, channels=1, subtype="FLOAT") as handle:
            for start in range(0, frames, block):
                count = min(block, frames - start)
                t = (np.arange(count) + start) / sample_rate
                handle.write((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[:, None])
        return path, frames

    def test_long_source_is_read_in_bounded_blocks(self):
        sample_rate = 8000
        path, frames = self._write_long_mono(sample_rate)
        scene = {
            "sample_rate": sample_rate,
            "layout": "mono",
            "global_seed": 1,
            "block_size": 4096,
            "events": [
                {"id": "long", "time": 0.0, "source": path.name, "position": [1, 0, 0]}
            ],
        }
        scene_path = self.dir / "scene.yaml"
        scene_path.write_text(yaml.safe_dump(scene))

        original_soundfile = sources_module.sf.SoundFile
        peak_bytes = {"value": 0}
        def tracked_trim(instance):
            original_trim(instance)
            working = instance._source.nbytes + instance._ready.nbytes
            peak_bytes["value"] = max(peak_bytes["value"], working)

        original_trim = sources_module.LinearResampler._trim_source
        sources_module.LinearResampler._trim_source = tracked_trim

        class InstrumentedSoundFile(original_soundfile):
            def read(self, frames=-1, dtype=None, always_2d=False, **kwargs):
                data = super().read(
                    frames=frames, dtype=dtype, always_2d=always_2d, **kwargs
                )
                if isinstance(data, np.ndarray):
                    peak_bytes["value"] = max(
                        peak_bytes["value"], data.nbytes
                    )
                return data

        sources_module.sf.SoundFile = InstrumentedSoundFile
        try:
            result = Renderer(load_scene(scene_path), self.dir / "out").render()
        finally:
            sources_module.sf.SoundFile = original_soundfile
            sources_module.LinearResampler._trim_source = original_trim

        source_bytes = frames * 1 * 4
        ratio = peak_bytes["value"] / source_bytes
        self.assertLess(ratio, 0.20)
        data, sr = sf.read(result.output_files["master"], dtype="float32")
        self.assertEqual(sr, sample_rate)
        report = result.report
        self.assertEqual(data.shape[0], report["total_frames"])
        self.assertTrue(np.isfinite(data).all())
        self.assertGreater(data.shape[0], frames * 0.9)

    def test_48k_stereo_long_source_streaming_ratio(self):
        sample_rate = 48000
        seconds = 601
        frames = sample_rate * seconds
        path = self.dir / "long_stereo.wav"
        block = 48000
        with sf.SoundFile(str(path), mode="w", samplerate=sample_rate, channels=2, subtype="FLOAT") as handle:
            for start in range(0, frames, block):
                count = min(block, frames - start)
                t = (np.arange(count) + start) / sample_rate
                left = 0.2 * np.sin(2 * np.pi * 300 * t)
                right = 0.2 * np.sin(2 * np.pi * 305 * t)
                handle.write(np.stack([left, right], axis=1).astype(np.float32))

        original_soundfile = sources_module.sf.SoundFile
        peak_bytes = {"value": 0}
        original_trim = sources_module.LinearResampler._trim_source

        def tracked_trim(instance):
            original_trim(instance)
            working = instance._source.nbytes + instance._ready.nbytes
            peak_bytes["value"] = max(peak_bytes["value"], working)

        sources_module.LinearResampler._trim_source = tracked_trim

        class InstrumentedSoundFile(original_soundfile):
            def read(self, frames=-1, dtype=None, always_2d=False, **kwargs):
                data = super().read(
                    frames=frames, dtype=dtype, always_2d=always_2d, **kwargs
                )
                if isinstance(data, np.ndarray):
                    peak_bytes["value"] = max(peak_bytes["value"], data.nbytes)
                return data

        sources_module.sf.SoundFile = InstrumentedSoundFile
        try:
            scene = {
                "sample_rate": sample_rate,
                "layout": "stereo",
                "global_seed": 2,
                "block_size": 4096,
                "duration": seconds,
                "events": [
                    {"id": "long", "time": 0.0, "source": path.name, "position": [1, 0, 0]}
                ],
            }
            scene_path = self.dir / "scene.yaml"
            scene_path.write_text(yaml.safe_dump(scene))
            result = Renderer(load_scene(scene_path), self.dir / "out").render()
        finally:
            sources_module.sf.SoundFile = original_soundfile
            sources_module.LinearResampler._trim_source = original_trim

        source_bytes = frames * 2 * 4
        self.assertLess(peak_bytes["value"] / source_bytes, 0.20)
        data, _ = sf.read(result.output_files["master"], dtype="float32")
        self.assertEqual(data.shape[1], 2)
        self.assertEqual(data.shape[0], result.report["total_frames"])
        self.assertTrue(np.isfinite(data).all())


if __name__ == "__main__":
    unittest.main()
