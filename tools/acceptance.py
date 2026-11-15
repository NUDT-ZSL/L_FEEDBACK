"""End-to-end acceptance checks against locally generated samples.

Covers: output format, loudness tolerance vs ffmpeg ebur128, click-free
joints, report completeness, error isolation, exit codes, concurrency
invariance, byte-level determinism and checkpoint resume.

Usage: python tools/acceptance.py [WORK_DIR]
Requires ffmpeg on PATH (used only as an independent loudness reference).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiopipe import wavio  # noqa: E402
from tools.gen_samples import generate  # noqa: E402

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def run_cli(manifest, out_dir, jobs=4, extra=None):
    cmd = [PY, "-m", "audiopipe", manifest, "--output-dir", out_dir,
           "--jobs", str(jobs)]
    if extra:
        cmd += extra
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def ffmpeg_loudness(path, ss=None, t=None):
    cmd = ["ffmpeg", "-nostats", "-hide_banner"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.3f}"]
    if t is not None:
        cmd += ["-t", f"{t:.3f}"]
    cmd += ["-i", path, "-filter_complex", "ebur128", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stderr.splitlines():
        line = line.strip()
        if line.startswith("I:") and "LUFS" in line:
            try:
                return float(line.split()[1])
            except (ValueError, IndexError):
                pass
    return None


def main():
    work = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "acceptance_out")
    if os.path.isdir(work):
        shutil.rmtree(work)
    samples = os.path.join(work, "samples")
    generate(samples)

    manifest = {
        "crossfade_ms": 20,
        "segments": [
            {"source": "tone_48k_16_stereo.wav", "start_sec": 0.5, "end_sec": 5.5,
             "target_loudness_lkfs": -23.0, "fade_in_ms": 10, "fade_out_ms": 10},
            {"source": "tone_44k1_16_stereo.wav", "start_sec": 0.0, "end_sec": 4.5,
             "target_loudness_lkfs": -23.0, "fade_in_ms": 5, "fade_out_ms": 5},
            {"source": "noise_48k_24_mono.wav", "start_sec": 0.2, "end_sec": 3.8,
             "target_loudness_lkfs": -24.0, "fade_in_ms": 0, "fade_out_ms": 0},
            {"source": "tone_44k1_24_6ch.wav", "start_sec": 0.0, "end_sec": 4.0,
             "target_loudness_lkfs": -23.0, "fade_in_ms": 10, "fade_out_ms": 10},
            {"source": "tone_48k_16_6ch.wav", "start_sec": 0.1, "end_sec": 3.5,
             "target_loudness_lkfs": -23.0, "fade_in_ms": 0, "fade_out_ms": 0},
            {"source": "tone_48k_24_stereo_loud.wav", "start_sec": 0.0, "end_sec": 2.5,
             "target_loudness_lkfs": -23.0, "fade_in_ms": 5, "fade_out_ms": 5},
            {"source": "tone_48k_24_stereo_loud.wav", "start_sec": 0.0, "end_sec": 2.0,
             "target_loudness_lkfs": 2.0, "fade_in_ms": 0, "fade_out_ms": 0},
            {"source": "missing.wav", "start_sec": 0.0, "end_sec": 1.0,
             "target_loudness_lkfs": -23.0},
            {"source": "bad_float32.wav", "start_sec": 0.0, "end_sec": 0.8,
             "target_loudness_lkfs": -23.0},
            {"source": "corrupt.wav", "start_sec": 0.0, "end_sec": 1.0,
             "target_loudness_lkfs": -23.0},
            {"source": "tone_48k_16_stereo.wav", "start_sec": 3.0, "end_sec": 99.0,
             "target_loudness_lkfs": -23.0},
            {"source": "tone_48k_16_stereo.wav", "start_sec": 0.0, "end_sec": 1.0},
        ],
    }
    manifest_path = os.path.join(samples, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    out1 = os.path.join(work, "out1")
    code, log = run_cli(manifest_path, out1, jobs=3)
    print(log.strip())

    check("exit code 1 on partial failure", code == 1, f"got {code}")
    report = json.load(open(os.path.join(out1, "report.json"), encoding="utf-8"))
    segs = {s["index"]: s for s in report["segments"]}
    ok_idx = [i for i, s in segs.items() if s["status"] == "success"]
    bad_idx = [i for i, s in segs.items() if s["status"] == "failed"]
    check("7 segments succeed, 5 isolated failures",
          len(ok_idx) == 7 and len(bad_idx) == 5,
          f"ok={ok_idx} failed={bad_idx}")
    expect_codes = {7: "E_FILE_NOT_FOUND", 8: "E_UNSUPPORTED_FORMAT",
                    9: "E_DECODE_ERROR", 10: "E_INVALID_RANGE",
                    11: "E_MISSING_FIELD"}
    check("business error codes recorded",
          all(segs[i]["error_code"] == c for i, c in expect_codes.items()),
          str({i: segs[i]["error_code"] for i in expect_codes}))

    required = {"source", "start_sec", "end_sec", "integrated_loudness_lkfs",
                "applied_gain_db", "xfade_in_ms", "xfade_out_ms",
                "offset_sec", "status", "error_code"}
    complete = all(required <= set(s) for s in report["segments"])
    check("report entries contain all required fields", complete)
    check("report status completed_with_errors",
          report["status"] == "completed_with_errors", report["status"])
    check("peak limiter warning on hot segment",
          "PEAK_LIMITED" in segs[6]["warnings"], str(segs[6]["warnings"]))

    wav_path = os.path.join(out1, "output.wav")
    audio, rate, ch = wavio.read_wav(wav_path)
    check("output is 48 kHz stereo", rate == 48000 and ch == 2,
          f"{rate} Hz {ch} ch")
    with open(wav_path, "rb") as fh:
        bits = int.from_bytes(fh.read(36)[34:36], "little")
    check("output is 24-bit", bits == 24, f"{bits} bit")
    check("no clipped samples",
          float(np.max(np.abs(audio))) <= 10 ** (-1 / 20) + 1e-4,
          f"peak={float(np.max(np.abs(audio))):.4f}")

    loud_ok, details = True, []
    for i in ok_idx:
        s = segs[i]
        off, dur = s["offset_sec"], s["duration_sec"]
        margin = min(0.6, dur / 4)
        meas = ffmpeg_loudness(wav_path, ss=off + margin, t=dur - 2 * margin)
        src_ch = s["source_channels"]
        tol = 0.5 if src_ch in (1, 2) else 2.0  # 5.1 downmix shifts level
        good = meas is not None and abs(meas - s["target_loudness_lkfs"]) <= tol
        loud_ok &= bool(good)
        details.append(f"seg{i}:{meas}LU vs {s['target_loudness_lkfs']}(tol {tol})")
    check("per-segment output loudness within tolerance", loud_ok,
          "; ".join(details))

    joints_ok, joints_detail = True, []
    diffs = np.abs(np.diff(audio, axis=0)).max(axis=1)
    global_ref = np.percentile(diffs, 99.9)
    for i in ok_idx[1:]:
        center = int(round(segs[i]["offset_sec"] * 48000))
        lo, hi = max(0, center - 240), min(len(diffs), center + 240)
        local = diffs[lo:hi].max() if hi > lo else 0.0
        joints_ok &= local <= max(global_ref * 1.5, 1e-6)
        joints_detail.append(f"@{segs[i]['offset_sec']}s:{local:.4f}")
    check("no clicks at joints", joints_ok,
          f"p99.9={global_ref:.4f} " + " ".join(joints_detail))

    out2 = os.path.join(work, "out2")
    code2, _ = run_cli(manifest_path, out2, jobs=3)
    same = sha256(wav_path) == sha256(os.path.join(out2, "output.wav"))
    check("repeated run is byte-identical", same and code2 == 1)

    out3 = os.path.join(work, "out3")
    code3, _ = run_cli(manifest_path, out3, jobs=1)
    same3 = sha256(wav_path) == sha256(os.path.join(out3, "output.wav"))
    check("jobs=1 vs jobs=3 identical output", same3 and code3 == 1)

    out4 = os.path.join(work, "out4")
    run_cli(manifest_path, out4, jobs=2)
    rep4 = os.path.join(out4, "report.json")
    doc = json.load(open(rep4, encoding="utf-8"))
    doc["status"] = "in_progress"
    with open(rep4, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    state_dir = os.path.join(out4, ".state")
    mtimes = {f: os.path.getmtime(os.path.join(state_dir, f))
              for f in os.listdir(state_dir)}
    victim = sorted(mtimes)[-1]
    os.remove(os.path.join(state_dir, victim))
    os.remove(os.path.join(out4, "output.wav"))
    code4, log4 = run_cli(manifest_path, out4, jobs=2)
    reused = "复用 6" in log4
    rewritten = [f for f in mtimes if f != victim and
                 os.path.getmtime(os.path.join(state_dir, f)) != mtimes[f]]
    same4 = sha256(wav_path) == sha256(os.path.join(out4, "output.wav"))
    check("resume reuses checkpoints and skips succeeded segments",
          code4 == 1 and reused and not rewritten,
          f"reused5={reused} rewritten={rewritten}")
    check("resumed output byte-identical", same4)

    code5, _ = run_cli(os.path.join(work, "no_such_manifest.json"),
                       os.path.join(work, "out5"))
    check("exit code 2 on missing manifest", code5 == 2, f"got {code5}")
    bad_manifest = os.path.join(work, "bad_manifest.json")
    with open(bad_manifest, "w") as fh:
        fh.write("{not json")
    code6, _ = run_cli(bad_manifest, os.path.join(work, "out6"))
    check("exit code 2 on invalid manifest", code6 == 2, f"got {code6}")
    only_bad = dict(manifest)
    only_bad["segments"] = [manifest["segments"][7]]  # missing source file
    only_bad_path = os.path.join(work, "only_bad.json")
    with open(only_bad_path, "w") as fh:
        json.dump(only_bad, fh)
    code7, _ = run_cli(only_bad_path, os.path.join(work, "out7"))
    check("exit code 2 when nothing succeeds", code7 == 2, f"got {code7}")

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
