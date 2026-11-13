import json
import os
import shutil
import tempfile
import unittest

from voiceforge.generator import SAMPLE_RATES, generate_dataset
from voiceforge.hashing import sha256_file


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="vf-gen-")
        self.first = os.path.join(self.root, "a")
        self.second = os.path.join(self.root, "b")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_fixed_seed_is_byte_reproducible(self):
        manifest_a = generate_dataset(self.first, seed=4242)
        manifest_b = generate_dataset(self.second, seed=4242)
        self.assertEqual(
            json.dumps(manifest_a, sort_keys=True),
            json.dumps(manifest_b, sort_keys=True))
        files = sorted(name for name in os.listdir(self.first)
                       if name.endswith(".wav"))
        self.assertTrue(files)
        for name in files:
            self.assertEqual(
                sha256_file(os.path.join(self.first, name)),
                sha256_file(os.path.join(self.second, name)))

    def test_player_and_segment_counts_and_rates(self):
        manifest = generate_dataset(self.first, seed=7)
        self.assertGreaterEqual(len(manifest["players"]), 3)
        self.assertLessEqual(len(manifest["players"]), 5)
        for player in manifest["players"]:
            self.assertGreaterEqual(len(player["segments"]), 6)
            self.assertLessEqual(len(player["segments"]), 10)
        used_rates = {shard["sample_rate"]
                      for player in manifest["players"]
                      for shard in player["segments"]}
        self.assertTrue(used_rates.issubset(set(SAMPLE_RATES)))

    def test_drift_and_missing_first_packets_are_present(self):
        manifest = generate_dataset(self.first, seed=11)
        drifts = [abs(shard["drift_ms"])
                  for player in manifest["players"]
                  for shard in player["segments"]]
        self.assertTrue(any(drift > 20.0 for drift in drifts))
        self.assertTrue(any(shard["dropped_first_packets_ms"] > 0
                            for player in manifest["players"]
                            for shard in player["segments"]))
        self.assertTrue(manifest["missing_players"])


if __name__ == "__main__":
    unittest.main()

