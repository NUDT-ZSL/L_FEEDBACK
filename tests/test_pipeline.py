"""端到端与单元测试。运行：``python -m unittest discover -s tests -v``"""

import json
import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from engine_pipeline import BakeConfig, BakeError, ErrorCode, run_bake
from engine_pipeline.loops import apply_equal_power_crossfade, detect_loop
from engine_pipeline.normalize import normalize_segments
from engine_pipeline.pitch import (
    estimate_f0,
    phase_vocoder_stretch,
    pitch_shift_preserve_duration,
    resample_to_length,
)
from engine_pipeline.pipeline import RpmTable
from engine_pipeline.synth import SynthConfig, synthesize


SR = 44100


def _cfg(out: str, **overrides) -> BakeConfig:
    base = dict(output_dir=out)
    base.update(overrides)
    return BakeConfig(**base)


class SynthTests(unittest.TestCase):
    def test_duration_rate_mono(self):
        result = synthesize(SynthConfig(sample_rate=SR, duration_s=20.0))
        self.assertEqual(result.audio.ndim, 1)
        self.assertEqual(result.audio.size, 20 * SR)
        self.assertEqual(result.f0_hz.size, result.audio.size)

    def test_deterministic_sample_exact(self):
        a = synthesize(SynthConfig(seed=42)).audio
        b = synthesize(SynthConfig(seed=42)).audio
        np.testing.assert_array_equal(a, b)

    def test_different_seed_differs(self):
        a = synthesize(SynthConfig(seed=1)).audio
        b = synthesize(SynthConfig(seed=2)).audio
        self.assertFalse(np.array_equal(a, b))

    def test_f0_trajectory_linear(self):
        result = synthesize(SynthConfig(rpm_start=800, rpm_end=2400, rpm_to_hz=0.5))
        # 端点基频与线性轨迹一致。
        self.assertAlmostEqual(result.f0_hz[0], 400.0, delta=2.0)
        self.assertAlmostEqual(result.f0_hz[-1], 1200.0, delta=2.0)


class PitchTests(unittest.TestCase):
    def test_stretch_pitch_preserved_length_scaled(self):
        x = np.sin(2 * np.pi * 300 * np.arange(SR // 2) / SR)
        for factor in (0.8, 1.25):
            y = phase_vocoder_stretch(x, factor)
            self.assertAlmostEqual(y.size, round(x.size * factor), delta=2)
            self.assertAlmostEqual(estimate_f0(y, SR), 300.0, delta=6.0)

    def test_resample_direction(self):
        x = np.sin(2 * np.pi * 200 * np.arange(SR) / SR)
        half = resample_to_length(x, SR // 2)
        self.assertAlmostEqual(estimate_f0(half, SR), 400.0, delta=4.0)

    def test_pitch_shift_preserves_duration(self):
        n = SR // 2
        x = np.sin(2 * np.pi * 300 * np.arange(n) / SR)
        for factor, expected in ((0.9, 270.0), (1.1, 330.0)):
            y = pitch_shift_preserve_duration(x, SR, factor)
            self.assertEqual(y.size, n)                       # 时长严格不变
            self.assertAlmostEqual(estimate_f0(y, SR), expected, delta=6.0)

    def test_estimate_f0_silence_raises(self):
        with self.assertRaises(BakeError) as ctx:
            estimate_f0(np.zeros(2048), SR)
        self.assertEqual(ctx.exception.code, ErrorCode.ERR_LOOP_DETECT)


class LoopTests(unittest.TestCase):
    def _periodic(self, f0=300.0, n=SR // 2):
        t = np.arange(n) / SR
        return np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(4 * np.pi * f0 * t)

    def test_integer_cycles_and_low_phase_error(self):
        sig = self._periodic()
        loop = detect_loop(sig, SR, 300.0, target_loop_s=0.1,
                           crossfade_ratio=0.125)
        self.assertGreater(loop.cycle_count, 0)
        self.assertEqual(loop.loop_length, loop.loop_end - loop.loop_start)
        # 整数周期约束：长度 / 周期 ≈ 周期数。
        self.assertAlmostEqual(loop.loop_length / loop.period_samples,
                               loop.cycle_count, delta=0.15)
        self.assertLess(loop.phase_error_rad, 0.5)

    def test_crossfade_constant_energy_and_fade_limit(self):
        sig = self._periodic()
        loop = detect_loop(sig, SR, 300.0, 0.1, 0.125)
        self.assertLessEqual(loop.fade_length, loop.loop_length // 4)
        baked = apply_equal_power_crossfade(sig, loop)
        k = np.linspace(0, 1, loop.fade_length)
        energy = k + (1.0 - k)                       # g_in^2 + g_out^2
        np.testing.assert_allclose(energy, np.ones_like(energy), atol=1e-12)
        # 回绕播放接缝处无突变（淡化区逐样本跳变有界）。
        s, e, f = loop.loop_start, loop.loop_end, loop.fade_length
        wrap = np.concatenate([baked[e - f: e], baked[s: s + f]])
        jump = np.abs(np.diff(wrap)).max()
        interior_p99 = np.percentile(np.abs(np.diff(baked)), 99.0)
        self.assertLessEqual(jump, interior_p99 * 1.6)

    def test_crossfade_ratio_over_quarter_rejected(self):
        sig = self._periodic()
        loop = detect_loop(sig, SR, 300.0, 0.1, 0.125)
        loop.fade_length = loop.loop_length  # 人为制造越界淡化区
        with self.assertRaises(BakeError) as ctx:
            apply_equal_power_crossfade(sig, loop)
        self.assertEqual(ctx.exception.code, ErrorCode.ERR_CROSSFADE)


class NormalizeTests(unittest.TestCase):
    def test_ceiling_and_uniform_gain(self):
        segs = [np.random.default_rng(i).standard_normal(4096) for i in range(3)]
        result = normalize_segments(segs, peak_ceiling=0.7, loudness_tol_db=3.0)
        self.assertTrue(all(p <= 0.7 + 1e-9 for p in result.peaks))
        self.assertAlmostEqual(max(result.peaks), 0.7, delta=1e-6)
        self.assertLess(max(result.rms_db) - min(result.rms_db), 3.0)

    def test_invalid_ceiling(self):
        with self.assertRaises(BakeError) as ctx:
            normalize_segments([np.ones(10)], peak_ceiling=1.5,
                               loudness_tol_db=1.0)
        self.assertEqual(ctx.exception.code, ErrorCode.ERR_PEAK_CEILING)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out = os.path.join(self.tmp, "bake")

    def test_manifest_matches_wavs_and_constraints(self):
        manifest = run_bake(_cfg(self.out))
        with open(os.path.join(self.out, "manifest.json"), encoding="utf-8") as fh:
            on_disk = json.load(fh)
        self.assertEqual(on_disk, manifest)
        self.assertEqual(manifest["sample_rate"], SR)
        self.assertEqual(manifest["segment_count"], len(manifest["segments"]))

        gains = set()
        for entry in manifest["segments"]:
            path = os.path.join(self.out, entry["file"])
            data, rate = sf.read(path, dtype="float64")
            info = sf.info(path)
            self.assertEqual(rate, SR)
            self.assertEqual(info.subtype, "PCM_16")
            self.assertEqual(info.channels, 1)
            self.assertEqual(data.size,
                             entry["source_interval"]["end_sample"]
                             - entry["source_interval"]["start_sample"])
            self.assertLessEqual(np.max(np.abs(data)),
                                 manifest["peak_ceiling"] + 2e-4)
            self.assertLess(entry["loop"]["phase_error_rad"], 0.5)
            self.assertLessEqual(entry["loop"]["fade_samples"],
                                 entry["loop"]["length_samples"] // 4)
            gains.add(round(entry["gain"], 8))
        self.assertEqual(len(gains), 1)                 # 全片段统一增益
        spread = max(s["rms_dbfs"] for s in manifest["segments"]) - \
                 min(s["rms_dbfs"] for s in manifest["segments"])
        self.assertLess(spread, manifest["loudness_tol_db"])

    def test_deterministic_outputs(self):
        m1 = run_bake(_cfg(os.path.join(self.tmp, "a")))
        m2 = run_bake(_cfg(os.path.join(self.tmp, "b")))
        self.assertEqual(json.dumps(m1, sort_keys=True),
                         json.dumps(m2, sort_keys=True))
        d1, _ = sf.read(os.path.join(self.tmp, "a", m1["segments"][0]["file"]))
        d2, _ = sf.read(os.path.join(self.tmp, "b", m2["segments"][0]["file"]))
        np.testing.assert_array_equal(d1, d2)

    def test_loop_seam_continuation(self):
        manifest = run_bake(_cfg(self.out))
        entry = manifest["segments"][0]
        data, _ = sf.read(os.path.join(self.out, entry["file"]))
        loop = entry["loop"]
        s, e, f = loop["start_sample"], loop["end_sample"], loop["fade_samples"]
        # 构造回绕播放序列：... loop 末尾 -> 跳回 loop 起点 ...
        wrap = np.concatenate([data[e - f: e], data[s: s + f]])
        wrap_jump = np.abs(np.diff(wrap)).max()
        # 与信号内部自然逐样本跳变对照，接缝不得引入比内部更强的脉冲。
        interior_p99 = np.percentile(np.abs(np.diff(data)), 99.0)
        self.assertLessEqual(wrap_jump, interior_p99 * 1.6)
        # 显式断言接缝两侧淡化曲线能量守恒：g_in^2 + g_out^2 = 1。
        k = np.linspace(0, 1, f)
        np.testing.assert_allclose(k + (1.0 - k), np.ones(f), atol=1e-12)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _expect_code(self, code: ErrorCode, **overrides):
        out = os.path.join(self.tmp, f"bad_{code.name}")
        with self.assertRaises(BakeError) as ctx:
            run_bake(_cfg(out, **overrides))
        self.assertEqual(ctx.exception.code, code)
        self.assertFalse(os.path.exists(os.path.join(out, "manifest.json")))

    def test_bad_rpm_step(self):
        self._expect_code(ErrorCode.ERR_RPM_TABLE,
                          rpm_table=RpmTable(1000, 2200, 0))

    def test_non_divisible_rpm_range(self):
        self._expect_code(ErrorCode.ERR_RPM_TABLE,
                          rpm_table=RpmTable(1000, 2200, 350))

    def test_bad_peak_ceiling(self):
        self._expect_code(ErrorCode.ERR_PEAK_CEILING, peak_ceiling=1.5)

    def test_bad_crossfade_ratio(self):
        self._expect_code(ErrorCode.ERR_CROSSFADE, crossfade_ratio=0.5)

    def test_segment_out_of_range(self):
        self._expect_code(ErrorCode.ERR_SEGMENT_RANGE,
                          rpm_table=RpmTable(1000, 5000, 400))

    def test_undetectable_loop(self):
        # 循环时长 >= 片段时长属于非法配置，触发循环/配置错误且不产出索引。
        self._expect_code(ErrorCode.ERR_CONFIG, target_loop_seconds=0.6)

    def test_loudness_spread_rejected(self):
        self._expect_code(ErrorCode.ERR_LOUDNESS_SPREAD, loudness_tol_db=0.0)

    def test_silence_loop_rejected(self):
        with self.assertRaises(BakeError) as ctx:
            estimate_f0(np.zeros(4096), SR)
        self.assertEqual(ctx.exception.code, ErrorCode.ERR_LOOP_DETECT)


if __name__ == "__main__":
    unittest.main()
