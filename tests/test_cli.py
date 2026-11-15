import json
import subprocess
import sys


def run_cli(config_path):
    return subprocess.run(
        [sys.executable, "-m", "terrain_gen", str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_invalid_config_exit_code_2(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("width: 64\nheight: 256\niterations: 4\noutput_prefix: x\n")
    result = run_cli(config)
    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["error_code"] == "ERR_CFG"


def test_cli_missing_file_exit_code_2(tmp_path):
    result = run_cli(tmp_path / "missing.yaml")
    assert result.returncode == 2
    assert json.loads(result.stderr)["error_code"] == "ERR_CFG"


def test_cli_no_args_exit_code_2():
    result = subprocess.run(
        [sys.executable, "-m", "terrain_gen"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["error_code"] == "ERR_CFG"


def test_cli_valid_run_exit_code_0(tmp_path):
    config = tmp_path / "ok.yaml"
    config.write_text(
        "width: 256\nheight: 256\niterations: 3\n"
        f"output_prefix: {tmp_path / 'out' / 'terrain'}\n"
        "noise:\n  seed: 5\n  octaves: 3\n"
    )
    result = run_cli(config)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "terrain_height.png").exists()
    assert (tmp_path / "out" / "terrain_normal.png").exists()
    assert (tmp_path / "out" / "terrain_material.png").exists()
    assert (tmp_path / "out" / "terrain_metadata.json").exists()


def test_cli_sweep_exit_codes(tmp_path):
    config = tmp_path / "sweep.yaml"
    config.write_text(
        "width: 256\nheight: 256\niterations: 2\n"
        f"output_prefix: {tmp_path / 'sw' / 'terrain'}\n"
        "noise:\n  seed: 6\n  octaves: 3\n"
        "sweep_iterations: [4, 2]\nfault_at: 3\n"
    )
    result = run_cli(config)
    assert result.returncode == 1
    summary = json.loads((tmp_path / "sw" / "terrain_sweep_summary.json").read_text())
    statuses = {row["iterations"]: row["status"] for row in summary["results"]}
    assert statuses[2] == "success"
    assert statuses[4] == "failed"
