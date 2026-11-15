import os
import tempfile
import unittest

import numpy as np

from cinema_master.wavio import (
    inspect_wav,
    read_audio,
    resolve_source_roles,
    write_wav,
)


class WavIoTests(unittest.TestCase):
    def test_extensible_channel_mask_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "extensible.wav")
            data = np.zeros((960, 6), dtype=np.float64)
            data[:, 2] = 0.1
            write_wav(path, data, 48000, 0x3F, 24, extensible=True)
            info = inspect_wav(path)
            self.assertEqual(info["format_name"], "WAVE_FORMAT_EXTENSIBLE")
            self.assertEqual(info["channel_mask"], 0x3F)
            self.assertEqual(info["channels"], 6)
            decoded, _ = read_audio(path)
            self.assertEqual(decoded.shape, (960, 6))
            roles, source = resolve_source_roles(info, None, None)[1:]
            self.assertEqual(source, "wav_extensible_channel_mask")
            self.assertEqual(roles, ["FL", "FR", "FC", "LFE", "BL", "BR"])

    def test_plain_pcm_has_no_mask_and_infers_51(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "plain.wav")
            data = np.zeros((100, 6), dtype=np.float64)
            write_wav(path, data, 32000, 0, 24, extensible=False)
            info = inspect_wav(path)
            self.assertEqual(info["format_name"], "WAVE_FORMAT_PCM")
            self.assertEqual(info["channel_mask"], 0)
            roles, source = resolve_source_roles(info, None, None)[1:]
            self.assertEqual(source, "implicit_5.1")
            self.assertEqual(roles, ["FL", "FR", "FC", "LFE", "BL", "BR"])

    def test_channel_map_length_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "stereo.wav")
            write_wav(path, np.zeros((10, 2)), 48000, 0x3, 24, extensible=True)
            info = inspect_wav(path)
            with self.assertRaises(Exception) as context:
                resolve_source_roles(info, None, ["FC"])
            self.assertEqual(getattr(context.exception, "code", None), "E_CHANNEL_MISMATCH")


if __name__ == "__main__":
    unittest.main()

