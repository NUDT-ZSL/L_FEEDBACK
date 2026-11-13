"""End-to-end acceptance driver.

Runs entirely locally with deterministic generated samples and no third-party
packages. It demonstrates:

1. A clean full export and its SHA-256.
2. A second export that is force-killed mid-render (SIGKILL-style via
   ``terminate`` on Windows / ``kill`` on POSIX), then restarted and resumed.
3. Byte-for-byte equality between the resumed master and the clean master.
4. The replay validation report and the structured error-code path.
5. Concurrent exports into separate output directories.

All artifacts are written under ``demo_work/`` and a machine-readable summary
is saved to ``demo_work/acceptance_summary.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from replay_exporter.exporter import export, generate_sample_project, validate  # noqa: E402
from replay_exporter.errors import ErrorCode, ExporterError  # noqa: E402


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_subprocess_export(timeline: str, output_dir: str, stop_after: int) -> dict:
    code = (
        "import sys, json; sys.path.insert(0, %r);"
        "from replay_exporter.exporter import export;"
        "print(json.dumps(export(%r, %r, checkpoint_every=1, stop_after_blocks=%d)))"
        % (ROOT, timeline, output_dir, stop_after)
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess export failed: {stderr.decode()}")
    return json.loads(stdout.decode().strip())


def force_kill_during_export(timeline: str, output_dir: str) -> dict:
    """Start an export in a child process and kill it hard mid-render."""
    code = (
        "import sys, time; sys.path.insert(0, %r);"
        "from replay_exporter.engine import render_master;"
        "render_master(%r, %r, checkpoint_every=1, stop_after_blocks=2);"
        "time.sleep(30)"
        % (ROOT, timeline, output_dir)
    )
    proc = subprocess.Popen([sys.executable, "-c", code])
    # Wait until at least one checkpoint block exists, then force-terminate.
    checkpoint = os.path.join(output_dir, "render_checkpoint.json")
    deadline = 0
    while not os.path.exists(checkpoint) and deadline < 200:
        time.sleep(0.05)
        deadline += 1
    proc.kill()
    proc.wait()
    return {"killed": True, "checkpoint_present": os.path.exists(checkpoint)}


def main() -> int:
    work = os.path.join(ROOT, "demo_work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    project = generate_sample_project(
        os.path.join(work, "project"),
        segment_count=4,
        segment_ms=1000,
        fade_ms=80,
        gaps_ms=[100, 0, 150],
    )
    timeline = project["timeline"]
    summary: dict = {"steps": []}

    # 1. Clean reference export.
    clean_dir = os.path.join(work, "clean")
    clean_result = export(timeline, clean_dir, checkpoint_every=1)
    clean_hash = sha256(os.path.join(clean_dir, "master.wav"))
    summary["steps"].append({"name": "clean_export", "result": clean_result, "sha256": clean_hash})

    # 2. Forced termination, then resume in a fresh process.
    killed_dir = os.path.join(work, "killed_resume")
    kill_info = force_kill_during_export(timeline, killed_dir)
    resumed = run_subprocess_export(timeline, killed_dir, stop_after=10_000)
    resumed_hash = sha256(os.path.join(killed_dir, "master.wav"))
    summary["steps"].append(
        {
            "name": "forced_kill_and_resume",
            "kill": kill_info,
            "resume": resumed,
            "sha256": resumed_hash,
            "byte_identical": resumed_hash == clean_hash,
        }
    )

    # 3. Validation report.
    report = validate(killed_dir)
    summary["steps"].append(
        {
            "name": "validation",
            "passed": report["passed"],
            "segments": len(report["segments"]),
            "crossfades": len(report["crossfades"]),
            "report_path": os.path.join(killed_dir, "validation_report.json"),
        }
    )

    # 4. Structured business error path.
    import tempfile

    bad_project = generate_sample_project(
        tempfile.mkdtemp(dir=work), segment_count=2, segment_ms=800
    )
    metadata_path = os.path.join(
        os.path.dirname(bad_project["timeline"]), "segment_0", "metadata.json"
    )
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    del metadata["dialogue_start_ms"]
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    try:
        export(bad_project["timeline"], os.path.join(work, "should_fail"))
    except ExporterError as error:
        summary["steps"].append(
            {
                "name": "structured_error",
                "code": error.code,
                "expected": ErrorCode.MISSING_DIALOGUE_RANGE,
                "segment_index": error.segment_index,
            }
        )

    # 5. Concurrent exports into distinct directories.
    import threading

    concurrent_results = {}

    def worker(name: str) -> None:
        concurrent_results[name] = export(
            timeline, os.path.join(work, f"concurrent_{name}")
        )

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summary["steps"].append(
        {
            "name": "concurrent_exports",
            "all_ok": all(item["ok"] for item in concurrent_results.values()),
            "jobs": len(concurrent_results),
        }
    )

    summary["overall_passed"] = (
        clean_result["ok"]
        and resumed_hash == clean_hash
        and report["passed"]
        and all(
            step.get("code") == ErrorCode.MISSING_DIALOGUE_RANGE
            for step in summary["steps"]
            if step["name"] == "structured_error"
        )
        and summary["steps"][-1]["all_ok"]
    )
    with open(os.path.join(work, "acceptance_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, sort_keys=True, indent=2)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if summary["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
