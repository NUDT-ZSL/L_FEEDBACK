from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .helpers import (
    dialogue,
    silence,
    timeline_payload,
    tone,
    write_json,
    write_mono_wav,
)


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cutscene_dub", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_subtitle_outside_shot_emits_structured_business_error(tmp_path: Path) -> None:
    data = timeline_payload(
        [dialogue("dlg_001", 0.2, 1.0)],
        duration_s=3.0,
    )
    data["shots"][0]["subtitles"][0]["end_s"] = 3.5
    data["shots"][0]["dialogues"][0]["end_s"] = 3.0
    timeline = tmp_path / "timeline.json"
    write_json(timeline, data)

    result = _run_cli("render", "--timeline", str(timeline))

    assert result.returncode != 0
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "E_SUBTITLE_INVALID"
    assert payload["error"]["entry_id"] == "sub_dlg_001"
    assert "path" in payload["error"]


def test_dialogue_outside_subtitle_reports_entry_id(tmp_path: Path) -> None:
    data = timeline_payload(
        [dialogue("dlg_001", 0.2, 2.0)],
        duration_s=3.0,
    )
    data["shots"][0]["subtitles"][0]["end_s"] = 1.5
    timeline = tmp_path / "timeline.json"
    write_json(timeline, data)

    result = _run_cli("render", "--timeline", str(timeline))

    assert result.returncode != 0
    payload = json.loads(result.stderr)
    assert payload["error"]["code"] == "E_DIALOGUE_INVALID"
    assert payload["error"]["entry_id"] == "dlg_001"


def test_overlapping_dialogues_are_rejected(tmp_path: Path) -> None:
    data = timeline_payload(
        [
            dialogue("dlg_001", 0.2, 1.2, subtitle_id="sub_shared"),
            dialogue("dlg_002", 1.0, 2.0, subtitle_id="sub_shared"),
        ],
        duration_s=3.0,
    )
    # Both dialogues share one subtitle so the overlap error is attributable to
    # the dialogue validation rather than overlapping subtitles.
    data["shots"][0]["subtitles"] = [
        {"id": "sub_shared", "start_s": 0.1, "end_s": 2.2, "text": "Shared"}
    ]
    timeline = tmp_path / "timeline.json"
    write_json(timeline, data)

    result = _run_cli("render", "--timeline", str(timeline))

    assert result.returncode != 0
    payload = json.loads(result.stderr)
    assert payload["error"]["code"] == "E_DIALOGUE_INVALID"
    assert payload["error"]["entry_id"] == "dlg_002"
    assert payload["error"]["details"]["previous_id"] == "dlg_001"


def test_invalid_json_has_error_code_and_no_traceback(tmp_path: Path) -> None:
    timeline = tmp_path / "timeline.json"
    timeline.write_text("{not-json", encoding="utf-8")

    result = _run_cli("generate-samples", "-t", str(timeline), "--seed", "1")

    assert result.returncode != 0
    payload = json.loads(result.stderr)
    assert payload["error"]["code"] == "E_TIMELINE_JSON"
    assert "Traceback" not in result.stderr


def test_unadaptable_cli_returns_four_and_writes_failure_report(
    tmp_path: Path,
) -> None:
    timeline = tmp_path / "timeline.json"
    write_json(
        timeline,
        timeline_payload(
            [dialogue("dlg_001", 0.2, 0.8)],
            duration_s=2.0,
        ),
    )
    audio_root = tmp_path / "samples"
    source = np.concatenate(
        [silence(0.01), tone(1.2, amplitude=0.5), silence(0.01)]
    )
    write_mono_wav(audio_root / "en-US" / "dlg_001.wav", source)
    write_mono_wav(
        audio_root / "bgm.wav",
        tone(2.0, frequency=110.0, amplitude=0.15),
    )
    output = tmp_path / "build" / "cutscene.wav"
    report = tmp_path / "build" / "report.json"

    result = _run_cli(
        "render",
        "-t",
        str(timeline),
        "-a",
        str(audio_root),
        "-o",
        str(output),
        "-r",
        str(report),
    )

    assert result.returncode == 4
    payload = json.loads(result.stderr)
    assert payload["error"]["code"] == "E_DIALOGUE_UNFIT"
    assert payload["error"]["entry_id"] == "dlg_001"
    assert payload["error"]["details"]["entry_ids"] == ["dlg_001"]
    assert not output.exists()
    report_data = json.loads(report.read_text(encoding="utf-8"))
    assert report_data["status"] == "failed"
    assert report_data["alignment"]["dialogues"][0]["unadaptable"] is True

