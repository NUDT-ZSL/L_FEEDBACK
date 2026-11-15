import json

import pytest

from terrain_gen.config import ConfigError, parse_config


def write_yaml(tmp_path, payload):
    path = tmp_path / "config.yaml"
    path.write_text(payload, encoding="utf-8")
    return path


VALID = """
width: 256
height: 320
iterations: 10
output_prefix: out/terrain
noise:
  seed: 7
erosion:
  hydraulic_weight: 0.4
  thermal_weight: 0.2
"""


def test_valid_config_defaults(tmp_path):
    cfg = parse_config(write_yaml(tmp_path, VALID))
    assert cfg.width == 256
    assert cfg.height == 320
    assert cfg.iterations == 10
    assert cfg.noise.seed == 7
    assert cfg.erosion.hydraulic_weight == 0.4
    assert cfg.erosion.thermal_weight == 0.2
    assert cfg.sweep_iterations == ()


@pytest.mark.parametrize(
    "payload",
    [
        "width: 128\nheight: 256\niterations: 4\noutput_prefix: x\n",
        "width: 256\nheight: 4096\niterations: 4\noutput_prefix: x\n",
        "width: 256\nheight: 256\niterations: 0\noutput_prefix: x\n",
        "width: 256\nheight: 256\niterations: 51\noutput_prefix: x\n",
        "width: true\nheight: 256\niterations: 4\noutput_prefix: x\n",
        "width: 256\nheight: 256\niterations: 4\noutput_prefix: ''\n",
        "width: 256\nheight: 256\niterations: 4\noutput_prefix: x\n"
        "erosion:\n  hydraulic_weight: 1.5\n",
        "width: 256\nheight: 256\niterations: 4\noutput_prefix: x\n"
        "noise:\n  seed: -1\n",
        "width: 256\nheight: 256\niterations: 4\noutput_prefix: x\n"
        "sweep_iterations: [0, 4]\n",
        "width: 256\nheight: 256\niterations: 4\noutput_prefix: x\n"
        "sweep_iterations: []\n",
    ],
)
def test_invalid_configs_raise(tmp_path, payload):
    with pytest.raises(ConfigError):
        parse_config(write_yaml(tmp_path, payload))


def test_missing_key_raises(tmp_path):
    path = write_yaml(tmp_path, "width: 256\nheight: 256\noutput_prefix: x\n")
    with pytest.raises(ConfigError):
        parse_config(path)


def test_broken_yaml_raises(tmp_path):
    path = write_yaml(tmp_path, "width: [256\n")
    with pytest.raises(ConfigError):
        parse_config(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        parse_config(tmp_path / "nope.yaml")


def test_error_is_json_serializable(tmp_path):
    try:
        parse_config(write_yaml(tmp_path, "width: 1\nheight: 256\niterations: 4\noutput_prefix: x\n"))
    except ConfigError as exc:
        rendered = json.dumps({"error_code": "ERR_CFG", "message": exc.message})
        assert json.loads(rendered)["error_code"] == "ERR_CFG"
