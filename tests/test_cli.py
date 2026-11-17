from __future__ import annotations

import json
from pathlib import Path

from character_preview.config import DEFAULTS
from character_preview.cli import main


def test_cli_validate_reports_error_code(tmp_path: Path, capsys) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"schema_version": 1}', encoding="utf-8")
    code = main(["validate", "--model", str(path)])
    captured = capsys.readouterr()
    assert code == 2
    assert "[E_MODEL_MATERIAL]" in captured.err


def test_cli_validate_warns_and_uses_defaults(tmp_path: Path, capsys) -> None:
    model_path = Path(__file__).resolve().parent.parent / "samples" / "character.json"
    config_path = tmp_path / "partial.json"
    config_path.write_text(json.dumps({"width": 10, "frame_count": 2}), encoding="utf-8")
    code = main(
        ["validate", "--model", str(model_path), "--config", str(config_path)]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "[WARN]" in captured.err
    payload = json.loads(captured.out)
    assert payload["valid"] is True
    assert payload["frame_count"] == 2
    assert 10 != DEFAULTS.width

