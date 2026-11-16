from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from cutscene_dub.state import render_state_path, sample_state_path

from .helpers import dialogue, timeline_payload, write_json


ROOT = Path(__file__).resolve().parents[1]


def _cli_env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(extra)
    return env


def _run(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cutscene_dub", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _timeline(tmp_path: Path) -> Path:
    path = tmp_path / "timeline.json"
    write_json(
        path,
        timeline_payload(
            [
                dialogue("dlg_001", 0.5, 1.7, source_duration_s=1.5),
                dialogue("dlg_002", 2.0, 3.2, source_duration_s=1.3),
            ],
            duration_s=4.0,
        ),
    )
    return path


def test_sample_generation_keyboard_interrupt_is_recoverable(tmp_path: Path) -> None:
    timeline = _timeline(tmp_path)
    audio_root = tmp_path / "samples"
    args = [
        "generate-samples",
        "-t",
        str(timeline),
        "-o",
        str(audio_root),
        "--seed",
        "189",
    ]

    interrupted = _run(
        args,
        env=_cli_env(CUTSCENE_DUB_TEST_FAILPOINT="sample_partial"),
    )
    assert interrupted.returncode == 130
    error = json.loads(interrupted.stderr)
    assert error["error"]["code"] == "E_INTERRUPTED"
    state = json.loads(sample_state_path(audio_root).read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"
    assert not (audio_root / "bgm.wav").exists()
    assert (audio_root / "bgm.wav.part").exists()

    recovered = _run(args)
    assert recovered.returncode == 0, recovered.stderr
    assert json.loads(recovered.stdout)["ok"] is True
    assert (audio_root / "bgm.wav").exists()
    assert not (audio_root / "bgm.wav.part").exists()
    state = json.loads(sample_state_path(audio_root).read_text(encoding="utf-8"))
    assert state["status"] == "complete"


def test_render_keyboard_interrupt_does_not_leave_final_artifact(tmp_path: Path) -> None:
    timeline = _timeline(tmp_path)
    audio_root = tmp_path / "samples"
    sample_args = [
        "generate-samples",
        "-t",
        str(timeline),
        "-o",
        str(audio_root),
        "--seed",
        "189",
    ]
    assert _run(sample_args).returncode == 0
    output = tmp_path / "build" / "cutscene.wav"
    report = tmp_path / "build" / "alignment_report.json"
    render_args = [
        "render",
        "-t",
        str(timeline),
        "-a",
        str(audio_root),
        "-o",
        str(output),
        "-r",
        str(report),
    ]

    interrupted = _run(
        render_args,
        env=_cli_env(CUTSCENE_DUB_TEST_FAILPOINT="render_audio_partial"),
    )
    assert interrupted.returncode == 130
    assert json.loads(interrupted.stderr)["error"]["code"] == "E_INTERRUPTED"
    assert not output.exists()
    assert not report.exists()
    assert (output.parent / (output.name + ".part")).exists()
    state_path = render_state_path(output)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"

    recovered = _run(render_args)
    assert recovered.returncode == 0, recovered.stderr
    payload = json.loads(recovered.stdout)
    assert payload["ok"] is True and payload["reused"] is False
    assert not (output.parent / (output.name + ".part")).exists()
    report_data = json.loads(report.read_text(encoding="utf-8"))
    assert report_data["status"] == "ok"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "complete"

    reused = _run(render_args)
    assert reused.returncode == 0
    assert json.loads(reused.stdout)["reused"] is True


def test_hard_kill_recovers_from_persisted_writing_state(tmp_path: Path) -> None:
    timeline = _timeline(tmp_path)
    audio_root = tmp_path / "samples"
    sample_args = [
        "generate-samples",
        "-t",
        str(timeline),
        "-o",
        str(audio_root),
        "--seed",
        "189",
    ]
    assert _run(sample_args).returncode == 0
    output = tmp_path / "build" / "cutscene.wav"
    report = tmp_path / "build" / "alignment_report.json"
    render_args = [
        sys.executable,
        "-m",
        "cutscene_dub",
        "render",
        "-t",
        str(timeline),
        "-a",
        str(audio_root),
        "-o",
        str(output),
        "-r",
        str(report),
    ]
    env = _cli_env(
        CUTSCENE_DUB_TEST_FAILPOINT="render_audio_partial",
        CUTSCENE_DUB_TEST_HOLD_MS="10000",
    )
    process = subprocess.Popen(
        render_args,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    partial = output.parent / (output.name + ".part")
    deadline = time.monotonic() + 10.0
    try:
        while time.monotonic() < deadline and not partial.exists():
            if process.poll() is not None:
                break
            time.sleep(0.02)
        assert partial.exists(), "test failpoint did not reach partial output"
        process.kill()
        process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)

    assert not output.exists()
    state = json.loads(render_state_path(output).read_text(encoding="utf-8"))
    assert state["status"] == "writing"
    assert partial.exists()

    recovered = _run(
        [
            "render",
            "-t",
            str(timeline),
            "-a",
            str(audio_root),
            "-o",
            str(output),
            "-r",
            str(report),
        ]
    )
    assert recovered.returncode == 0, recovered.stderr
    assert output.exists() and report.exists() and not partial.exists()
    state = json.loads(render_state_path(output).read_text(encoding="utf-8"))
    assert state["status"] == "complete"

