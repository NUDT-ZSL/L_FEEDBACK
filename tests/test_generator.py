from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import soundfile as sf

from generate_samples import generate


def _hashes(root: Path) -> dict:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class GeneratorTests(unittest.TestCase):
    def test_generator_is_deterministic_and_varies_formats(self) -> None:
        with tempfile.TemporaryDirectory() as first_temp, tempfile.TemporaryDirectory() as second_temp:
            first = generate(Path(first_temp) / "samples")
            second = generate(Path(second_temp) / "samples")
            self.assertEqual(_hashes(first), _hashes(second))

            formats = {
                (sf.info(str(path)).samplerate, sf.info(str(path)).subtype)
                for path in first.rglob("*.wav")
            }
            self.assertIn((48000, "PCM_16"), formats)
            self.assertIn((44100, "PCM_24"), formats)
            self.assertIn((96000, "PCM_24"), formats)
            self.assertEqual(len(list(first.glob("*/scene.json"))), 3)


if __name__ == "__main__":
    unittest.main()
