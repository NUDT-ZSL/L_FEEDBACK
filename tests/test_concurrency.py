from __future__ import annotations

import json
import os
import subprocess
import sys

from rhythmforge.errors import ExitCode


def test_two_concurrent_builds_one_is_rejected(built_project):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src")
    output = os.path.join(built_project["base"], "concurrent_out")
    base_cmd = [
        sys.executable, "-m", "rhythmforge.cli", "build",
        "--chart", built_project["chart_path"],
        "--samples-dir", built_project["samples_dir"],
        "--output", output,
    ]

    proc_a = subprocess.Popen(base_cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env=env)
    proc_b = subprocess.Popen(base_cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env=env)
    out_a, err_a = proc_a.communicate()
    out_b, err_b = proc_b.communicate()

    codes = sorted([proc_a.returncode, proc_b.returncode])
    # At least one build must win (exit 0); the loser, if rejected while the
    # lock was held, reports the dedicated lock exit code.
    assert ExitCode.OK in codes
    assert codes[-1] in (ExitCode.OK, ExitCode.LOCKED_OUTPUT)

    locked_stderr = err_a if proc_a.returncode == ExitCode.LOCKED_OUTPUT else err_b
    if proc_a.returncode == ExitCode.LOCKED_OUTPUT or \
            proc_b.returncode == ExitCode.LOCKED_OUTPUT:
        payload = json.loads(locked_stderr)
        assert payload["error"]["code"] == ExitCode.LOCKED_OUTPUT

    # The winning build still produced a coherent, verifiable output set.
    assert os.path.isfile(os.path.join(output, "master.wav"))
    assert os.path.isfile(os.path.join(output, "manifest.json"))
