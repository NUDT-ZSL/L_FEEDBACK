import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dub_replace.cli import main as cli_main
from dub_replace.runner import run_job
from dub_replace.state import JobState

from tests.conftest import build_stereo_job


def test_run_job_success_and_skip_on_rerun(stereo_job):
    manifest_path, job_dir, _ = stereo_job
    exit_code, summary = run_job(manifest_path)
    assert exit_code == 0
    assert summary["succeeded"] == 1 and summary["failed"] == 0
    report = json.loads((job_dir / "reports" / "anim_stereo.json").read_text(encoding="utf-8"))
    assert report["status"] == "success"

    # Second run: success item is skipped, no reprocessing.
    state_path = job_dir / "job_state.json"
    before = state_path.read_text(encoding="utf-8")
    exit_code, summary = run_job(manifest_path)
    assert exit_code == 0
    assert summary["skipped"] == 1 and summary["succeeded"] == 0
    assert state_path.read_text(encoding="utf-8") == before


def test_batch_isolates_failure_and_exit_code(tmp_path):
    # Two animations: one healthy, one missing its original WAV.
    manifest_path, job_dir, _ = build_stereo_job(tmp_path, animation_id="good")
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad_meta = {
        "animation_id": "bad",
        "audio": {
            "original_track": "missing.wav",
            "sample_rate": 48000,
            "channel_layout": "stereo",
            "channels": [
                {"index": 0, "label": "L", "role": "sfx"},
                {"index": 1, "label": "R", "role": "target_voice"},
            ],
        },
        "stretch_limits": {"min": 0.5, "max": 2.0},
        "dialogue": [{"line_id": "X", "start_ms": 0, "end_ms": 100, "text": "x"}],
        "subtitles": [],
        "lip_events": [],
        "dialect_clips": [],
    }
    (bad_dir / "meta.json").write_text(json.dumps(bad_meta, ensure_ascii=False), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["animations"].append({"animation_id": "bad", "meta": "bad/meta.json"})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    exit_code, summary = run_job(manifest_path)
    assert exit_code == 1
    assert summary["succeeded"] == 1 and summary["failed"] == 1
    statuses = {r["animation_id"]: r for r in summary["results"]}
    assert statuses["bad"]["error"]["code"] == "E_MISSING_AUDIO"
    bad_report = json.loads((job_dir / "reports" / "bad.json").read_text(encoding="utf-8"))
    assert bad_report["status"] == "failed"
    assert bad_report["error"]["hint"]

    # Rerun: good is skipped, bad retried and fails again.
    exit_code, summary = run_job(manifest_path)
    assert exit_code == 1
    assert summary["skipped"] == 1 and summary["failed"] == 1


def test_cli_exit_codes(stereo_job, capsys):
    manifest_path, _, _ = stereo_job
    assert cli_main([str(manifest_path), "-q"]) == 0
    assert cli_main([str(manifest_path)]) == 0  # second run -> skips, still 0


def test_cli_missing_manifest_returns_2(tmp_path):
    assert cli_main([str(tmp_path / "nope.json")]) == 2


def _write_sleeper_manifest(root: Path, count: int = 8) -> Path:
    """Build a job whose processor blocks on a per-attempt env-gated sleep."""
    manifest_path, _, _ = build_stereo_job(root, animation_id="anim_0")
    for i in range(1, count):
        build_stereo_job(root, animation_id=f"anim_{i}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["concurrency"] = 2
    manifest["animations"] = [
        {"animation_id": f"anim_{i}", "meta": f"anim_{i}/meta.json"} for i in range(count)
    ]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def test_hard_kill_recovery_no_duplicate_success(tmp_path, monkeypatch):
    """Kill the process mid-run; restart must not re-output finished items."""
    if sys.platform.startswith("win"):
        pytest.skip("SIGTERM semantics differ on Windows; state logic is platform-independent")
    manifest_path = _write_sleeper_manifest(tmp_path, count=4)
    job_dir = tmp_path / "out"
    gate = job_dir / "gate"
    gate.parent.mkdir(parents=True, exist_ok=True)

    driver = tmp_path / "driver.py"
    driver.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
        "import dub_replace.processor as processor_mod\n"
        "import dub_replace.runner as runner_mod\n"
        f"gate = {str(gate)!r}\n"
        "orig = processor_mod.process_animation\n"
        "def wrapped(*a, **k):\n"
        "    open(gate, 'a').close()\n"
        "    time = __import__('time'); time.sleep(4.0)\n"
        "    return orig(*a, **k)\n"
        "processor_mod.process_animation = wrapped\n"
        "runner_mod.process_animation = wrapped\n"
        "from dub_replace.cli import main\n"
        f"sys.exit(main([{str(manifest_path)!r}, '-q']))\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    proc = subprocess.Popen(
        [sys.executable, str(driver)], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # Wait until at least one worker has entered the processor.
    deadline = time.time() + 15
    while not gate.exists() and time.time() < deadline:
        time.sleep(0.05)
    # Then wait for the first item to fully finish so we can prove a
    # committed success is never re-output after the restart.
    reports_dir = job_dir / "reports"
    deadline = time.time() + 15
    while time.time() < deadline and not (
        reports_dir.exists() and list(reports_dir.glob("*.json"))
    ):
        time.sleep(0.05)
    finished = list(reports_dir.glob("*.json")) if reports_dir.exists() else []
    assert finished, "预期至少一个动画在终止前完成"
    finished_wavs = {}
    for report_path in finished:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        wav = Path(report["output"]["path"])
        assert wav.is_file()
        finished_wavs[wav.name] = wav.read_bytes()
    # SIGKILL simulates a hard kill / power loss: no Python cleanup runs,
    # exactly the scenario the atomic state file must survive. (SIGTERM is
    # not delivered to ThreadPoolExecutor workers blocked in a sleep.)
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=15)
    assert proc.returncode != 0

    exit_code, summary = run_job(manifest_path)
    assert exit_code == 0
    assert summary["failed"] == 0
    assert summary["succeeded"] + summary["skipped"] == 4
    # Committed outputs survive the restart byte-for-byte (no duplicate output).
    for name, payload in finished_wavs.items():
        matches = list(job_dir.rglob(name))
        assert matches and all(wav_path.read_bytes() == payload for wav_path in matches)
    state = JobState.load(job_dir)
    for report_path in finished:
        assert state.is_success(report_path.stem)
