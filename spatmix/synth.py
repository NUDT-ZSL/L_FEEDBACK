"""Deterministic dry mono sample synthesis.

Samples are produced lazily in absolute sample-index order so the
resampler can pull at a variable rate. Noise sources draw from a
``numpy.random.Generator`` seeded from a hash of the source id, which
makes output bit-identical across processes for identical parameters.
"""

import hashlib

import numpy as np

CHUNK = 4096


def seed_from_id(source_id: str) -> int:
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


class DrySource:
    """Lazy deterministic mono generator with snapshot-able state."""

    def __init__(self, spec, sample_rate):
        self.spec = spec
        self.sample_rate = sample_rate
        self.total = int(round(spec["duration"] * sample_rate))
        self.start = 0                      # absolute index of buf[0]
        self.buf = np.zeros(0, dtype=np.float64)
        self.rng = None
        if spec["waveform"] == "noise":
            self.rng = np.random.Generator(np.random.PCG64(seed_from_id(spec["id"])))

    # -- generation -------------------------------------------------
    def _generate(self, count):
        spec = self.spec
        begin = self.start + len(self.buf)
        idx = begin + np.arange(count, dtype=np.float64)
        end = self.total
        waveform = spec["waveform"]
        if waveform == "sine":
            out = np.sin(2.0 * np.pi * spec["frequency"] * idx / self.sample_rate)
        elif waveform == "impulse":
            period = max(1, int(round(self.sample_rate / spec["frequency"])))
            out = ((idx.astype(np.int64) % period) == 0).astype(np.float64)
        elif waveform == "noise":
            out = self.rng.standard_normal(count) * 0.5
        else:  # pragma: no cover - validated at parse time
            raise ValueError(waveform)
        if begin + count > end:
            valid = np.maximum(0, end - begin)
            mask = np.arange(count) < valid
            out = np.where(mask, out, 0.0)
        return out * spec["amplitude"]

    def ensure(self, upto_index):
        """Ensure samples with absolute index <= ``upto_index`` exist."""
        have = self.start + len(self.buf)
        if have > upto_index:
            return
        need = upto_index + 1 - have
        count = max(CHUNK, need)
        self.buf = np.concatenate([self.buf, self._generate(count)])

    def discard_before(self, index):
        """Drop buffered samples with absolute index < ``index``."""
        drop = min(max(0, index - self.start), len(self.buf))
        if drop > CHUNK:
            self.buf = self.buf[drop:]
            self.start += drop

    def read(self, positions):
        """Linear-interpolated read at absolute float positions."""
        lo = int(np.floor(positions[0]))
        hi = int(np.floor(positions[-1])) + 1
        self.ensure(hi)
        xp = self.start + np.arange(len(self.buf), dtype=np.float64)
        return np.interp(positions, xp, self.buf)

    # -- snapshot ---------------------------------------------------
    def state_dict(self):
        return {
            "start": self.start,
            "buf": self.buf.tolist(),
            "rng": (self.rng.bit_generator.state if self.rng is not None else None),
        }

    def load_state(self, state):
        self.start = int(state["start"])
        self.buf = np.array(state["buf"], dtype=np.float64)
        if self.rng is not None:
            self.rng.bit_generator.state = state["rng"]
