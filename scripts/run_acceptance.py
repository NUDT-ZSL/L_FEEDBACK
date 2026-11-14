"""End-to-end acceptance driver for the adaptive mixing engine.

Generates deterministic 48 kHz/24-bit mono tracks, renders a three-state
game score (explore / battle / cutscene), exercises the thread-pool batch
path and the checkpoint resume path, then writes a machine-readable
acceptance report plus human-readable summary into the output directory.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_mixer import __version__
from adaptive_mixer.batch import BatchJob, process_batch
from adaptive_mixer.config import load_config
from adaptive_mixer.engine import MixEngine, RenderInterrupted
from adaptive_mixer.errors import MixerError
from adaptive_mixer.limiter import true_peak_db
from adaptive_mixer.loudness import integrated_loudness
from adaptive_mixer.sample_gen import SAMPLE_RATE, generate_tracks
from adaptive_mixer.wavio import read_wav, read_wav_chunks


def build_config(root: Path, seconds: float = 9.0,
                 frame_rate: int = 100) -> Path:
    total_frames = int(seconds * frame_rate)
    generate_tracks(root / "tracks",
                    ["strings", "percussion", "ambience"], seconds)
    boundaries = [0, total_frames // 3, 2 * total_frames // 3, total_frames]
    scenes = [
        ("explore", "探索", -26.0, "s_curve",
         ["strings", "ambience"]),
        ("battle", "战斗", -18.0, "equal_power",
         ["percussion", "strings"]),
        ("cutscene", "过场", -24.0, "linear",
         ["ambience", "strings"]),
    ]
    scene_json = []
    for index, (scene_id, name, lufs, curve, tracks) in enumerate(scenes):
        entry = {
            "scene_id": scene_id, "name": name,
            "start_frame": boundaries[index],
            "end_frame": boundaries[index + 1],
            "target_lufs": lufs,
            "crossfade_ms": 150.0,
            "crossfade_type": curve,
            "active_tracks": tracks,
        }
        if index == 0:
            entry.pop("crossfade_ms")
        scene_json.append(entry)
    config = {
        "sample_rate": SAMPLE_RATE,
        "frame_rate": frame_rate,
        "total_frames": total_frames,
        "target_lufs": -23.0,
        "crossfade_ms": 150.0,
        "crossfade_type": "s_curve",
        "tracks": [
            {"name": "strings", "path": "tracks/strings.wav"},
            {"name": "percussion", "path": "tracks/percussion.wav"},
            {"name": "ambience", "path": "tracks/ambience.wav"},
        ],
        "scenes": scene_json,
        "output_format": {"sample_rate": SAMPLE_RATE, "channels": 1,
                          "subtype": "PCM_24"},
    }
    path = root / "scene_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def check(name: str, condition: bool, detail: str = "") -> dict:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
    return {"name": name, "status": status, "detail": detail}


def main() -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "artifacts" / f"acceptance_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    latest = ROOT / "artifacts" / "acceptance"
    latest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checks = []

    # 1. Deterministic single render.
    single_dir = out_root / "single"
    config_path = build_config(single_dir)
    cfg = load_config(config_path)
    output_path = single_dir / "adaptive_mix.wav"
    state_path = single_dir / "adaptive_mix.state.json"
    t0 = time.perf_counter()
    result = MixEngine(cfg, single_dir, output_path,
                       state_file=state_path).render()
    render_seconds = time.perf_counter() - t0
    report = result.report
    (single_dir / "mix_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    samples, sr = read_wav(output_path)
    chunks = read_wav_chunks(output_path)
    chunk_ids = [name for name, _ in chunks["chunks"]]
    measured_lufs = integrated_loudness(samples, sr)
    targets = np.array([s.target_lufs for s in cfg.scenes])
    weights = np.array([s.end_frame - s.start_frame
                        for s in cfg.scenes], dtype=float)
    weighted_target = float(np.sum(targets * weights / weights.sum()))
    tp = true_peak_db(samples)

    checks.append(check("输出采样率 48000 Hz", sr == 48000, f"{sr} Hz"))
    checks.append(check("RIFF 块顺序 fmt/fact/data",
                        chunk_ids == ["fmt ", "fact", "data"],
                        str(chunk_ids)))
    checks.append(check("输出时长等于总帧数",
                        len(samples) == cfg.total_samples,
                        f"{len(samples)}/{cfg.total_samples} samples"))
    checks.append(check("时间轴对齐误差为 0",
                        report["timeline"]["alignment_error_samples"] == 0))
    checks.append(check("真峰值 <= -1 dBTP", tp <= -1.0 + 1e-6,
                        f"{tp:.3f} dBTP"))
    checks.append(check(
        "整体响度偏差 <= 0.5 LU",
        abs(measured_lufs - weighted_target) <= 0.5,
        f"measured={measured_lufs:.3f} target={weighted_target:.3f}"))
    scene_ok = all(abs(s["actual_lufs"] - s["target_lufs"]) <= 0.5
                   for s in report["scenes"])
    checks.append(check("各场景响度偏差 <= 0.5 LU", scene_ok,
                        json.dumps([{s["scene_id"]:
                                     round(s["actual_lufs"]
                                           - s["target_lufs"], 3)}
                                    for s in report["scenes"]],
                                   ensure_ascii=False)))
    checks.append(check("交叉淡化事件数量正确",
                        len(report["crossfades"]) == len(cfg.scenes) - 1))

    # Determinism: rerender and compare bit-for-bit.
    rerender_dir = out_root / "rerender"
    rerender_cfg_path = build_config(rerender_dir)
    rerender = MixEngine(
        load_config(rerender_cfg_path), rerender_dir,
        rerender_dir / "adaptive_mix.wav",
        state_file=rerender_dir / "state.json").render()
    samples_again, _ = read_wav(rerender.output_path)
    checks.append(check("确定性：重复渲染位级一致",
                        np.array_equal(samples, samples_again)))

    # 2. Checkpoint: interrupt after the first scene, then resume.
    resume_dir = out_root / "resume"
    resume_cfg_path = build_config(resume_dir)
    resume_cfg = load_config(resume_cfg_path)
    resume_out = resume_dir / "adaptive_mix.wav"
    resume_state = resume_dir / "state.json"
    engine = MixEngine(resume_cfg, resume_dir, resume_out,
                       state_file=resume_state)
    try:
        engine.render(interrupt_after=lambda sid: sid == "explore")
    except RenderInterrupted:
        pass
    checks.append(check("中断后生成断点文件", resume_state.is_file()))
    resumed = MixEngine(resume_cfg, resume_dir, resume_out,
                        state_file=resume_state).render()
    resume_samples, _ = read_wav(resume_out)
    checks.append(check("断点续混识别为 resumed", resumed.resumed is True))
    checks.append(check("续混输出长度与总帧数一致",
                        len(resume_samples) == resume_cfg.total_samples))
    checks.append(check("续混与一次渲染样本一致",
                        np.allclose(resume_samples, samples, atol=1e-12)))

    # 3. Stale version rejection.
    stale_dir = out_root / "stale"
    stale_cfg_path = build_config(stale_dir)
    stale_cfg = load_config(stale_cfg_path)
    stale_engine = MixEngine(stale_cfg, stale_dir,
                             stale_dir / "m.wav",
                             state_file=stale_dir / "stale.json")
    try:
        stale_engine.render(interrupt_after=lambda sid: True)
    except RenderInterrupted:
        pass
    changed = json.loads(stale_cfg_path.read_text(encoding="utf-8"))
    changed["crossfade_ms"] = 220.0
    stale_cfg_path.write_text(json.dumps(changed, ensure_ascii=False),
                              encoding="utf-8")
    try:
        MixEngine(load_config(stale_cfg_path), stale_dir,
                  stale_dir / "m.wav",
                  state_file=stale_dir / "stale.json").render()
        stale_rejected = False
        stale_code = ""
    except MixerError as exc:
        stale_rejected = exc.code == "E_STATE_VERSION"
        stale_code = exc.code
    checks.append(check("旧版本断点被拒绝 (E_STATE_VERSION)",
                        stale_rejected, stale_code))

    # 4. Concurrent batch: 4 jobs, one deliberately broken.
    batch_dir = out_root / "batch"
    jobs = []
    for index in range(4):
        workspace = batch_dir / f"job{index}"
        job_cfg = build_config(workspace)
        if index == 3:
            job_cfg.write_text("{broken json", encoding="utf-8")
        jobs.append(BatchJob(
            job_id=f"job{index}",
            config_path=str(job_cfg),
            output_path=str(workspace / "adaptive_mix.wav"),
            state_file=str(workspace / "state.json")))
    summary = process_batch(jobs, max_workers=4)
    (batch_dir / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8")
    failed = summary["jobs"][3]
    checks.append(check("并发 4 任务成功 3 个",
                        summary["succeeded"] == 3
                        and summary["failed"] == 1,
                        f"succeeded={summary['succeeded']}"))
    checks.append(check("失败任务返回 E_ 错误码且不影响其他任务",
                        failed["code"] == "E_CONFIG_PARSE"
                        and all(job["ok"] for job in summary["jobs"][:3]),
                        failed.get("code", "")))

    passed = sum(1 for item in checks if item["status"] == "PASS")
    acceptance = {
        "engine_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "single_render_seconds": round(render_seconds, 3),
        "measured": {
            "integrated_lufs": round(measured_lufs, 4),
            "weighted_target_lufs": round(weighted_target, 4),
            "true_peak_db": round(tp, 4),
            "sample_count": int(len(samples)),
            "duration_seconds": len(samples) / sr,
        },
        "scene_loudness": [
            {"scene_id": s["scene_id"],
             "target_lufs": s["target_lufs"],
             "actual_lufs": round(s["actual_lufs"], 4),
             "error_lu": round(s["actual_lufs"] - s["target_lufs"], 4),
             "bus_trim_db": s["bus_trim_db"]}
            for s in report["scenes"]
        ],
        "crossfades": report["crossfades"],
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "all_passed": passed == len(checks),
    }
    report_file = out_root / "acceptance_report.json"
    report_file.write_text(json.dumps(acceptance, ensure_ascii=False,
                                      indent=2), encoding="utf-8")
    latest_file = latest / "acceptance_report.json"
    latest_file.write_text(json.dumps(acceptance, ensure_ascii=False,
                                      indent=2), encoding="utf-8")
    print(f"\n{passed}/{len(checks)} checks passed; report: {report_file}")
    return 0 if acceptance["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
