#!/usr/bin/env python3
"""Acceptance suite for the offline rhythm-game audio engine.

Covers deterministic generation, resampling, quantization, ducking,
looping, layer fades, SFX overlap, end-to-end quality gates, and
error-code/atomicity behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audio_engine import ENGINE_SAMPLE_RATE
from audio_engine.ducking import build_duck_envelope
from audio_engine.layers import layer_gain_envelopes
from audio_engine.loader import load_wav, resample_to_engine
from audio_engine.looping import loop_extend
from audio_engine.quantize import quantize_time
from audio_engine.sfx import schedule_sfx, soft_limit

SR = ENGINE_SAMPLE_RATE
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def run_cli(*args):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "main.py"), *args],
        capture_output=True,
        text=True,
    )


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    tmp = tempfile.mkdtemp(prefix="uid171_accept_")
    try:
        assets_a = os.path.join(tmp, "assets_a")
        assets_b = os.path.join(tmp, "assets_b")

        gen = os.path.join(ROOT, "tools", "generate_samples.py")
        subprocess.run([sys.executable, gen, assets_a], check=True, capture_output=True)
        subprocess.run([sys.executable, gen, assets_b], check=True, capture_output=True)
        names = sorted(os.listdir(assets_a))
        same = names == sorted(os.listdir(assets_b)) and all(
            sha256(os.path.join(assets_a, n)) == sha256(os.path.join(assets_b, n))
            for n in names
        )
        check("deterministic sample generation", same, f"{len(names)} files compared")

        config_path = os.path.join(assets_a, "config.json")
        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)

        data44, sr44 = load_wav(os.path.join(assets_a, "base_layer.wav"))
        check(
            "load preserves sample rate + channels",
            sr44 == 44100 and data44.shape[1] == 2,
            f"sr={sr44}, shape={data44.shape}",
        )
        up = resample_to_engine(data44, 44100, SR)
        dur_ok = abs(up.shape[0] / SR - data44.shape[0] / 44100) < 1e-3
        check(
            "resample to 48k preserves channels/duration",
            up.shape[1] == 2 and dur_ok and np.isfinite(up).all(),
            f"shape={up.shape}",
        )
        t = np.arange(44100) / 44100
        tone = np.sin(2 * np.pi * 5000 * t)[:, None]
        tone_up = resample_to_engine(tone, 44100, SR)
        amp_in = np.sqrt(2 * np.mean(tone**2))
        amp_out = np.sqrt(2 * np.mean(tone_up[480:-480] ** 2))
        check(
            "band-limited resample keeps in-band amplitude",
            abs(amp_out - amp_in) / amp_in < 0.02,
            f"in={amp_in:.4f} out={amp_out:.4f}",
        )
        # imaging: 5 kHz tone upsampled 44.1k->48k must not create the
        # 44.1k-5k=39.1k image (which would fold to 8.9k at 48k)
        spec = np.abs(np.fft.rfft(tone_up[480:-480, 0]))
        freqs = np.fft.rfftfreq(tone_up[480:-480, 0].size, 1 / SR)

        def mag_at(f):
            return spec[np.argmin(np.abs(freqs - f))]

        check(
            "band-limited resample suppresses imaging",
            mag_at(8900) < 0.01 * mag_at(5000),
            f"image/signal={mag_at(8900) / mag_at(5000):.6f}",
        )
        # aliasing on the downsample path: 23 kHz at 48k -> 44.1k must die
        t48 = np.arange(48000) / 48000
        alias = np.sin(2 * np.pi * 23000 * t48)[:, None]
        alias_down = resample_to_engine(alias, 48000, 44100)
        alias_amp = np.sqrt(2 * np.mean(alias_down[480:-480] ** 2))
        check(
            "band-limited resample suppresses aliasing",
            alias_amp < 0.01,
            f"residual={alias_amp:.5f}",
        )

        q, err = quantize_time(1.053, 0.05, 120.0, "1/16")
        check(
            "quantize snaps to nearest 1/16 point",
            abs(q - 1.05) < 1e-9 and abs(err - -3.0) < 1e-6,
            f"q={q}, err={err:.3f} ms",
        )
        q2, _ = quantize_time(0.05 + 4 * 0.5 + 0.004, 0.05, 120.0, "1/4")
        check("quantize 1/4 grid", abs(q2 - 2.05) < 1e-9, f"q={q2}")

        env = build_duck_envelope(2 * SR, SR, [(1.0, 1.5)], duck_gain_db=-6.0)
        duck = 10 ** (-6 / 20)
        s0, s1 = SR, int(1.5 * SR)
        depth_ok = abs(env[s0 + int(0.02 * SR)] - duck) < 1e-3
        hold_ok = np.allclose(env[s0 + int(0.02 * SR) : s1], duck, atol=1e-3)
        rec_ok = abs(env[s1 + int(0.1 * SR) - 1] - 1.0) < 0.02
        cont_ok = np.max(np.abs(np.diff(env))) <= (1 - duck) / int(0.02 * SR) + 1e-9
        check(
            "ducking depth/hold/recover/continuity",
            depth_ok and hold_ok and rec_ok and cont_ok,
            f"min={env.min():.4f}, max_step={np.max(np.abs(np.diff(env))):.6f}",
        )

        bar_len = int(round(4 * 60 / 120 * SR))
        src = np.sin(2 * np.pi * np.arange(2 * bar_len) / bar_len * 4)[:, None]
        target = int(5.3 * bar_len)
        ext, loops = loop_extend(src, target, bar_len, int(0.01 * SR))
        step = np.max(np.abs(np.diff(ext, axis=0)))
        check(
            "loop extension length + loop count",
            ext.shape[0] == target and loops == 2,
            f"len={ext.shape[0]}, loops={loops}",
        )
        check("loop seam sample-level continuity", step < 0.05, f"max step={step:.5f}")

        states = [2] * 30 + [7] * 30 + [13] * 30
        envs = layer_gain_envelopes(states, 30, 3 * SR, SR)
        w = int(0.05 * SR)
        mid = SR
        ramp = envs["percussion"][mid : mid + w]
        fade_ok = ramp[0] < 0.05 and abs(ramp[-1] - 1.0) < 0.05
        mono_ok = np.all(np.diff(ramp) >= -1e-12)
        amb_ok = np.allclose(envs["ambience"][: 2 * SR - w], 0.0) and abs(
            envs["ambience"][2 * SR + w] - 1.0
        ) < 0.05
        check(
            "layer add/remove completes in 50 ms",
            fade_ok and mono_ok and amb_ok,
            f"ramp_end={ramp[-1]:.3f}",
        )

        bus = np.zeros((SR, 2))
        sfx = np.ones((int(0.02 * SR), 1)) * 0.5
        schedule_sfx(bus, sfx, 1000, 1.0, 0.0)
        schedule_sfx(bus, sfx, 1000 + int(0.005 * SR), 1.0, 0.0)
        overlap_val = bus[1000 + int(0.005 * SR) + 10, 0]
        single = 0.5 * (0.5**0.5)
        check(
            "overlapping SFX sum (not overwrite)",
            abs(overlap_val - 2 * single) < 1e-9,
            f"overlap={overlap_val:.4f}, expect={2 * single:.4f}",
        )
        hot = np.full((100, 2), 2.0)
        limited = soft_limit(hot, 10 ** (-1 / 20))
        check(
            "soft limiter caps below -1 dBFS without hard clip",
            np.max(np.abs(limited)) < 10 ** (-1 / 20) and limited[0, 0] > 0.7,
            f"peak={np.max(np.abs(limited)):.4f}",
        )

        out1 = os.path.join(tmp, "run1", "mix.wav")
        rep1 = os.path.join(tmp, "run1", "report.json")
        out2 = os.path.join(tmp, "run2", "mix.wav")
        rep2 = os.path.join(tmp, "run2", "report.json")
        r1 = run_cli("--config", config_path, "--output", out1, "--report", rep1)
        r2 = run_cli("--config", config_path, "--output", out2, "--report", rep2)
        check(
            "engine runs successfully",
            r1.returncode == 0 and r2.returncode == 0,
            (r1.stderr.strip() or r2.stderr.strip()),
        )

        with open(rep1, encoding="utf-8") as fh:
            report = json.load(fh)
        mix, mix_sr = sf.read(out1, dtype="float64", always_2d=True)
        peak_db = 20 * np.log10(np.max(np.abs(mix)))
        check(
            "output peak <= -1 dBFS",
            peak_db <= -1.0 + 1e-9,
            f"peak={peak_db:.2f} dBFS (report {report['mix']['peak_dbfs']})",
        )
        check(
            "report peak matches rendered WAV",
            abs(report["mix"]["peak_dbfs"] - round(peak_db, 2)) < 0.02,
            f"report={report['mix']['peak_dbfs']} wav={peak_db:.2f}",
        )
        max_qerr = max(abs(e["error_ms"]) for e in report["quantization"]["events"])
        check("quantization error <= 5 ms for all events", max_qerr <= 5.0, f"max={max_qerr} ms")
        last_q = max(e["quantized_s"] for e in report["quantization"]["events"])
        check(
            "output covers last trigger + 2 s",
            report["output"]["duration_s"] >= last_q + 2.0 - 0.01,
            f"duration={report['output']['duration_s']} s, last={last_q} s",
        )
        check("loop extension happened", report["loop"]["count"] >= 1, f"loops={report['loop']['count']}")
        check(
            "stereo 48 kHz output",
            mix_sr == SR
            and mix.shape[1] == 2
            and mix.shape[0] == int(round(report["output"]["duration_s"] * SR)),
            f"sr={mix_sr}, shape={mix.shape}",
        )
        check(
            "reproducible report + WAV",
            sha256(rep1) == sha256(rep2) and sha256(out1) == sha256(out2),
            "byte-identical across runs",
        )
        check(
            "ducking slope reported",
            report["ducking"]["max_slope_db_per_s"] > 0,
            f"{report['ducking']['max_slope_db_per_s']} dB/s",
        )

        bad_dir = os.path.join(tmp, "bad")
        os.makedirs(bad_dir, exist_ok=True)

        def write_cfg(tag, cfg):
            # resolve all asset paths to absolute so the bad config is
            # independent of its own directory
            cfg = json.loads(json.dumps(cfg))
            cfg["layers"] = {
                k: os.path.abspath(os.path.join(assets_a, v))
                for k, v in cfg["layers"].items()
            }
            cfg["sfx"] = {
                k: v if os.path.isabs(v) else os.path.abspath(os.path.join(assets_a, v))
                for k, v in cfg["sfx"].items()
            }
            p = os.path.join(bad_dir, f"{tag}_cfg.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            return p

        def expect_failure(tag, cfg, code):
            o = os.path.join(bad_dir, f"{tag}.wav")
            r = os.path.join(bad_dir, f"{tag}.json")
            res = run_cli("--config", cfg, "--output", o, "--report", r)
            leftovers = [p for p in (o, r, o + ".tmp", r + ".tmp") if os.path.exists(p)]
            ok = res.returncode != 0 and code in res.stderr and not leftovers
            check(
                f"error {code}",
                ok,
                f"exit={res.returncode}, stderr={res.stderr.strip()!r}, leftovers={leftovers}",
            )

        cfg_missing = dict(config)
        cfg_missing["layers"] = dict(config["layers"], base="no_such_file.wav")
        expect_failure("missing", write_cfg("missing", cfg_missing), "ERR_MISSING_FILE")

        bad_sr_wav = os.path.join(bad_dir, "bad_sr_input.wav")
        sf.write(bad_sr_wav, np.zeros((2205, 1)), 22050, subtype="PCM_16")
        cfg_bad_sr = dict(config)
        cfg_bad_sr["sfx"] = dict(config["sfx"], hit=os.path.abspath(bad_sr_wav))
        expect_failure("bad_sr", write_cfg("bad_sr", cfg_bad_sr), "ERR_BAD_SAMPLE_RATE")

        cfg_bad_bpm = dict(config, bpm=0)
        expect_failure("bad_bpm", write_cfg("bad_bpm", cfg_bad_bpm), "ERR_INVALID_BPM")

        cfg_neg_bpm = dict(config, bpm=-120)
        expect_failure("neg_bpm", write_cfg("neg_bpm", cfg_neg_bpm), "ERR_INVALID_BPM")

        failed = [n for n, ok, _ in RESULTS if not ok]
        print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
        if failed:
            print("FAILED: " + ", ".join(failed))
            return 1
        print("ALL ACCEPTANCE CHECKS PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
