import json
from pathlib import Path

from terrain_gen.config import ErosionConfig, NoiseConfig, TerrainConfig
from terrain_gen.pipeline import run_sweep


def make_sweep_config(prefix, fault_at=None):
    return TerrainConfig(
        width=256,
        height=256,
        iterations=4,
        output_prefix=str(prefix),
        noise=NoiseConfig(seed=33, octaves=4),
        erosion=ErosionConfig(),
        sweep_iterations=(8, 2, 4),
        fault_at=fault_at,
    )


def test_sweep_runs_ascending_and_all_succeed(tmp_path):
    summary = run_sweep(make_sweep_config(tmp_path / "terrain"))
    assert [row["iterations"] for row in summary["results"]] == [2, 4, 8]
    assert summary["failed_runs"] == 0
    assert summary["exit_code"] == 0
    for count in (2, 4, 8):
        assert (tmp_path / f"terrain_it{count}_height.png").exists()
    assert Path(summary["summary_path"]).exists()


def test_sweep_isolates_failure_and_continues(tmp_path):
    # fault_at=4 only affects runs that actually reach iteration 4; the 2-step
    # run succeeds while both longer runs fail, and the sweep still completes.
    summary = run_sweep(make_sweep_config(tmp_path / "terrain", fault_at=4))
    assert summary["total_runs"] == 3
    assert summary["failed_runs"] == 2
    assert summary["exit_code"] == 1
    by_count = {row["iterations"]: row for row in summary["results"]}
    assert by_count[2]["status"] == "success"
    for row in summary["results"]:
        if row["status"] == "failed":
            assert row["error_type"] == "RuntimeError"
            assert "injected failure" in row["error_message"]
    assert (tmp_path / "terrain_it2_height.png").exists()

    on_disk = json.loads(Path(summary["summary_path"]).read_text())
    assert on_disk["exit_code"] == 1


def test_sweep_partial_failure_mixed(tmp_path):
    # The 2-step run finishes before iteration 3; longer runs hit the fault.
    summary = run_sweep(make_sweep_config(tmp_path / "terrain", fault_at=3))
    by_count = {row["iterations"]: row for row in summary["results"]}
    assert by_count[2]["status"] == "success"
    assert by_count[4]["status"] == "failed"
    assert by_count[8]["status"] == "failed"
    assert summary["failed_runs"] == 2
    assert summary["exit_code"] == 1
    assert (tmp_path / "terrain_it2_height.png").exists()
