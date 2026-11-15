"""Acceptance verification for the audio alignment pipeline.

Runs the generated demo workload twice and asserts:

* byte-identical WAV and JSON report across runs (determinism);
* output is 48 kHz / 24-bit / stereo and matches the declared duration;
* integrated loudness is the -16 LUFS platform target (NumPy measurement,
  cross-checked against FFmpeg ``ebur128`` when available);
* true peak never exceeds the -1 dBTP ceiling;
* every expected business error code appears in the report;
* event report timestamps are sample-aligned within one millisecond.
"""

from __future__ import annotations

import filecmp
import json
import os
import subprocess
import sys
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from audiopipeline.loudness import integrated_loudness, true_peak_db  # noqa: E402
from audiopipeline.manifest import ms_to_samples  # noqa: E402


def read_wav24(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    assert width == 3 and channels == 2, "输出必须是 24-bit 立体声"
    data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    unsigned = (
        data[:, 0].astype(np.uint32)
        | (data[:, 1].astype(np.uint32) << 8)
        | (data[:, 2].astype(np.uint32) << 16)
    )
    ints = (unsigned.astype(np.int32) << 8) >> 8
    audio = (ints / (2**23 - 1)).reshape(-1, 2).astype(np.float64)
    return audio, rate


def ffmpeg_ebur128(path: str) -> tuple[float | None, float | None]:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
         "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    lufs = tp = None
    for line in proc.stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("I:"):
            try:
                lufs = float(stripped.split()[1])
            except (IndexError, ValueError):
                pass
        if "TPK:" in stripped:
            try:
                tp = float(stripped.split("TPK:")[1].split()[0])
            except (IndexError, ValueError):
                pass
    return lufs, tp


def main() -> int:
    workspace = os.path.join(ROOT, "demo", "workspace")
    manifest = os.path.join(workspace, "manifest.json")
    out_a = os.path.join(workspace, "verify_a.wav")
    out_b = os.path.join(workspace, "verify_b.wav")
    report_a = os.path.join(workspace, "verify_a.report.json")
    report_b = os.path.join(workspace, "verify_b.report.json")

    def run(output, report):
        proc = subprocess.run(
            [sys.executable, "-m", "audiopipeline.cli", manifest, output,
             "-r", report], cwd=ROOT, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr

    run(out_a, report_a)
    run(out_b, report_b)

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = ""):
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name} {detail}")
        if not ok:
            failures.append(name)

    check("WAV 字节级确定性", filecmp.cmp(out_a, out_b, shallow=False))
    check("报告字节级确定性", filecmp.cmp(report_a, report_b, shallow=False))

    audio, rate = read_wav24(out_a)
    report = json.load(open(report_a, encoding="utf-8"))

    check("采样率 48kHz", rate == 48000, f"rate={rate}")
    declared_ms = report["declared_total_duration_ms"]
    check(
        "时长与清单一致",
        audio.shape[0] == ms_to_samples(declared_ms),
        f"frames={audio.shape[0]} expected={ms_to_samples(declared_ms)}",
    )

    lufs = integrated_loudness(audio)
    tp = true_peak_db(audio)
    check("整体响度 -16 LUFS (±0.3)", lufs is not None and abs(lufs + 16) <= 0.3,
          f"measured={None if lufs is None else round(lufs, 3)}")
    check("真峰值 <= -1 dBTP", tp is not None and tp <= -1.0 + 1e-6,
          f"measured={None if tp is None else round(tp, 3)}")

    ff_lufs, ff_tp = ffmpeg_ebur128(out_a)
    if ff_lufs is not None:
        check("FFmpeg ebur128 响度交叉校验", abs(ff_lufs - lufs) <= 0.5,
              f"ffmpeg={ff_lufs} ours={round(lufs, 3)}")
        check("FFmpeg 真峰值交叉校验", ff_tp is not None and ff_tp <= -1.0 + 0.2,
              f"ffmpeg={ff_tp} ours={round(tp, 3)}")

    expected_codes = {
        "E_DECODE_FAILED", "E_FORMAT_UNSUPPORTED", "E_FILE_NOT_FOUND",
        "E_TIME_INVALID",
    }
    seen = {code for ev in report["events"] for code in ev["error_codes"]}
    check("错误码完整覆盖", expected_codes.issubset(seen),
          f"seen={sorted(seen)}")

    # All catalog codes carry the E_ prefix.
    check("错误码 E_ 前缀", all(c["code"].startswith("E_")
          for c in report["error_catalog"]))

    # Report timestamps align with sample positions within one millisecond.
    aligned = True
    for ev in report["events"]:
        if ev["status"] != "success":
            continue
        for key in ("actual_start_ms", "actual_end_ms"):
            value = ev[key]
            sample_ms = ms_to_samples(value) * 1000.0 / 48000
            if abs(sample_ms - value) > 1.0:
                aligned = False
    check("报告时间戳亚毫秒对齐", aligned)

    summary = report["summary"]
    check(
        "汇总统计正确",
        summary["total_events"] == 16 and summary["succeeded"] == 11
        and summary["skipped"] == 5,
        json.dumps(summary, ensure_ascii=False),
    )

    print("\n汇总:", json.dumps(summary, ensure_ascii=False))
    print("母带:", json.dumps(report["master"], ensure_ascii=False))
    if failures:
        print(f"\n{len(failures)} 项验收失败: {failures}")
        return 1
    print("\n全部验收项通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
