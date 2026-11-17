import concurrent.futures
import os
import subprocess
import sys

from conftest import mix_bytes, read_report, run_cli

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_subprocess(manifest_path, out_dir, state_dir):
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [
            sys.executable, "-m", "audiotl",
            str(manifest_path), str(out_dir),
            "--state-dir", str(state_dir),
            "--workers", "4",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def _run_pair(sample_set, out_a, out_b, state_dir):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(
            _run_subprocess, sample_set["manifest_path"], out_a, state_dir
        )
        fut_b = pool.submit(
            _run_subprocess, sample_set["manifest_path"], out_b, state_dir
        )
        return fut_a.result(), fut_b.result()


def test_concurrent_runs_share_state_dir_safely(sample_set, tmp_path):
    state_dir = tmp_path / "shared_state"
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    res_a, res_b = _run_pair(sample_set, out_a, out_b, state_dir)
    assert res_a.returncode == 0, res_a.stderr
    assert res_b.returncode == 0, res_b.stderr

    out_ref = tmp_path / "out_ref"
    assert run_cli(sample_set["manifest_path"], out_ref,
                   state_dir=tmp_path / "ref_state") == 0

    assert mix_bytes(out_a) == mix_bytes(out_ref)
    assert mix_bytes(out_b) == mix_bytes(out_ref)
    assert read_report(out_a)["status"] == "ok"
    assert read_report(out_b)["status"] == "ok"


def test_concurrent_same_output_dir_is_atomic(sample_set, tmp_path):
    state_dir = tmp_path / "shared_state2"
    out = tmp_path / "out_same"
    res_a, res_b = _run_pair(sample_set, out, out, state_dir)
    assert res_a.returncode == 0, res_a.stderr
    assert res_b.returncode == 0, res_b.stderr

    out_ref = tmp_path / "out_ref2"
    assert run_cli(sample_set["manifest_path"], out_ref,
                   state_dir=tmp_path / "ref_state2") == 0
    assert mix_bytes(out) == mix_bytes(out_ref)
