from __future__ import annotations

import json
from pathlib import Path

import pytest

from character_preview.config import DEFAULTS, config_from_dict, load_config
from character_preview.errors import ConfigError


def test_missing_values_use_defaults_and_warn() -> None:
    config, warnings = config_from_dict({"width": 800})
    assert config.width == 800
    assert config.height == DEFAULTS.height
    assert config.frame_count == DEFAULTS.frame_count
    assert any("width" not in warning for warning in warnings)
    assert len(warnings) == len(DEFAULTS.__dataclass_fields__) - 1


def test_out_of_range_and_wrong_type_use_defaults() -> None:
    config, warnings = config_from_dict(
        {
            **DEFAULTS.__dict__,
            "width": 10,
            "height": "large",
            "max_point_lights": 999,
            "shadow_bias": -2.0,
        }
    )
    assert config.width == DEFAULTS.width
    assert config.height == DEFAULTS.height
    assert config.max_point_lights == DEFAULTS.max_point_lights
    assert config.shadow_bias == DEFAULTS.shadow_bias
    assert len(warnings) >= 4


def test_unknown_field_warns() -> None:
    config, warnings = config_from_dict({**DEFAULTS.__dict__, "mystery": 1})
    assert config == DEFAULTS
    assert any("unknown field" in warning for warning in warnings)


def test_invalid_config_json_has_stable_code(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert error.value.code == "E_CONFIG_JSON"


def test_complete_config_has_no_warnings(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(DEFAULTS.__dict__), encoding="utf-8")
    config, warnings = load_config(path)
    assert config == DEFAULTS
    assert warnings == []

