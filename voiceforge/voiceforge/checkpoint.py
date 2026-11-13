"""Durable, content-addressed checkpoints for resumable synthesis.

Checkpoint layout inside ``<output>/.checkpoint``::

    state.json                 committed last via atomic replace
    accumulator.bin            float64 snapshot of the summed master track
    contrib/<interval>.json    per-interval sidecar (digest, offset, stats)
    contrib/<interval>.bin     float64 contribution of one processed interval

Recovery rules:

* A changed manifest digest invalidates the entire checkpoint.
* A changed shard SHA-256 invalidates that player's work and every downstream
  cumulative state from the earliest affected interval onward; earlier
  unaffected intervals are reused and the accumulator is rebuilt from their
  stored contributions.
* Only ``state.json`` replacement marks a commit, so a hard kill leaves at
  worst an orphaned contribution that is ignored and cleaned on resume.
"""

import array
import json
import os
import shutil

from .hashing import sha256_text
from .lock import atomic_write_text

CHECKPOINT_DIR_NAME = ".checkpoint"
STATE_NAME = "state.json"
ACCUMULATOR_NAME = "accumulator.bin"
MISSING_DIGEST = "missing-source"
FORMAT_VERSION = 1


class CheckpointStore:
    def __init__(self, output_dir: str):
        self.root = os.path.join(output_dir, CHECKPOINT_DIR_NAME)
        self.contrib_dir = os.path.join(self.root, "contrib")
        self.state_path = os.path.join(self.root, STATE_NAME)
        self.accumulator_path = os.path.join(self.root, ACCUMULATOR_NAME)

    def reset(self) -> None:
        if os.path.isdir(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.contrib_dir, exist_ok=True)

    def exists(self) -> bool:
        return os.path.isfile(self.state_path)

    def write_contribution(self, interval_id: str, player_id: str,
                           start_sample: int, shard_digest: str,
                           contribution, loudness: dict) -> None:
        os.makedirs(self.contrib_dir, exist_ok=True)
        bin_path = self._contrib_bin(interval_id)
        sidecar_path = self._contrib_json(interval_id)
        values = array.array("d", contribution)
        self._write_binary_atomic(bin_path, values.tobytes())
        sidecar = {
            "interval_id": interval_id,
            "player_id": player_id,
            "start_sample": start_sample,
            "length": len(contribution),
            "shard_digest": shard_digest,
            "loudness": loudness,
        }
        atomic_write_text(sidecar_path, json.dumps(sidecar, indent=2) + "\n")

    def read_contribution(self, interval_id: str):
        sidecar_path = self._contrib_json(interval_id)
        with open(sidecar_path, "r", encoding="utf-8") as handle:
            sidecar = json.load(handle)
        values = array.array("d")
        if sidecar["length"] > 0:
            with open(self._contrib_bin(interval_id), "rb") as handle:
                values.frombytes(handle.read())
        return sidecar, values

    def write_accumulator(self, accumulator) -> None:
        values = array.array("d", accumulator)
        self._write_binary_atomic(self.accumulator_path, values.tobytes())

    def write_state(self, manifest_digest: str, completed_ids,
                    total_samples: int) -> None:
        state = {
            "format_version": FORMAT_VERSION,
            "manifest_digest": manifest_digest,
            "completed_ids": list(completed_ids),
            "total_samples": total_samples,
        }
        atomic_write_text(self.state_path, json.dumps(state, indent=2) + "\n")

    def read_state(self) -> dict:
        with open(self.state_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_or_reset(self, manifest_text: str, ordered_intervals,
                      digest_for_interval, total_samples: int):
        """Return (retained_id_set, accumulator, clip_mask, loudness_map).

        ``digest_for_interval(interval)`` returns the current shard digest or
        :data:`MISSING_DIGEST` when the player has no shards.
        """
        empty = set(), array.array("d", [0.0] * total_samples), \
            bytearray(total_samples), {}
        if not self.exists():
            os.makedirs(self.contrib_dir, exist_ok=True)
            return empty

        manifest_digest = sha256_text(manifest_text)
        try:
            state = self.read_state()
            if state.get("format_version") != FORMAT_VERSION or \
                    state.get("manifest_digest") != manifest_digest:
                self.reset()
                return empty
            completed_ids = list(state["completed_ids"])
        except (OSError, ValueError, KeyError):
            self.reset()
            return empty

        index_by_id = {interval_id: index
                       for index, interval_id in enumerate(
                           interval["interval_id"] for interval in ordered_intervals)}
        interval_by_id = {interval["interval_id"]: interval
                          for interval in ordered_intervals}
        cut_index = len(ordered_intervals)
        for interval_id in completed_ids:
            interval = interval_by_id.get(interval_id)
            if interval is None:
                cut_index = min(cut_index,
                                index_by_id.get(interval_id, len(ordered_intervals)))
                continue
            current_digest = digest_for_interval(interval)
            sidecar_path = self._contrib_json(interval_id)
            try:
                with open(sidecar_path, "r", encoding="utf-8") as handle:
                    sidecar = json.load(handle)
                stored_digest = sidecar["shard_digest"]
            except (OSError, ValueError, KeyError):
                stored_digest = None
            if stored_digest != current_digest:
                cut_index = min(cut_index, index_by_id[interval_id])

        retained_ids = []
        for interval_id in completed_ids:
            if interval_id in index_by_id and index_by_id[interval_id] < cut_index:
                retained_ids.append(interval_id)

        for interval_id in completed_ids:
            if interval_id not in retained_ids:
                self._discard_contribution(interval_id)

        accumulator = array.array("d", [0.0] * total_samples)
        clip_mask = bytearray(total_samples)
        loudness_map = {}
        from .audio import is_clipped
        for interval_id in retained_ids:
            sidecar, contribution = self.read_contribution(interval_id)
            start = sidecar["start_sample"]
            new_clips = 0
            for offset, value in enumerate(contribution):
                output_index = start + offset
                accumulator[output_index] += value
                if is_clipped(accumulator[output_index]) and not clip_mask[output_index]:
                    clip_mask[output_index] = 1
                    new_clips += 1
            loudness_map[interval_id] = self._compute_loudness(
                contribution, new_clips)
        return set(retained_ids), accumulator, clip_mask, loudness_map

    def _discard_contribution(self, interval_id: str) -> None:
        for path in (self._contrib_bin(interval_id), self._contrib_json(interval_id)):
            if os.path.isfile(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    @staticmethod
    def _compute_loudness(contribution, new_clips: int) -> dict:
        peak = 0.0
        square_sum = 0.0
        for value in contribution:
            magnitude = abs(value)
            if magnitude > peak:
                peak = magnitude
            square_sum += value * value
        rms = (square_sum / len(contribution)) ** 0.5 if contribution else 0.0
        return {
            "peak": peak,
            "rms": rms,
            "new_clipped_samples": new_clips,
        }

    def _contrib_json(self, interval_id: str) -> str:
        return os.path.join(self.contrib_dir, interval_id + ".json")

    def _contrib_bin(self, interval_id: str) -> str:
        return os.path.join(self.contrib_dir, interval_id + ".bin")

    @staticmethod
    def _write_binary_atomic(path: str, data: bytes) -> None:
        import tempfile
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

