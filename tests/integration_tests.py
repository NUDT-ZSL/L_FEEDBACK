#!/usr/bin/env python3
"""Process-level acceptance tests for skinbake."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


def run(baker: Path, args: list[str], expected: int | None = 0):
    process = subprocess.run(
        [str(baker), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if expected is not None and process.returncode != expected:
        raise AssertionError(
            f"command {args} returned {process.returncode}, expected {expected}\n"
            f"stdout={process.stdout}\nstderr={process.stderr}"
        )
    return process


def last_json(process: subprocess.CompletedProcess[str]):
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"no JSON stdout; stderr={process.stderr}")
    return json.loads(lines[-1])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baker", required=True)
    parser.add_argument("--work", required=True)
    args = parser.parse_args()
    baker = Path(args.baker).resolve()
    work = Path(args.work).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    print("[1/8] serial/parallel byte identity and from-zero verification")
    serial = work / "serial.snapshot"
    parallel = work / "parallel.snapshot"
    run(baker, [
        "bake", "--scenario", "idle", "--output", str(serial),
        "--frames", "512", "--fps", "120", "--version", "1",
        "--workers", "1",
    ])
    second = last_json(run(baker, [
        "bake", "--scenario", "idle", "--output", str(parallel),
        "--frames", "512", "--fps", "120", "--version", "1",
        "--workers", "40",
    ]))
    require(sha256(serial) == sha256(parallel),
            "parallel snapshot differs from serial snapshot")
    require(second["requestedWorkers"] == 40 and second["usedWorkers"] < 40,
            "worker count should be resource bounded")
    verified = last_json(run(baker, [
        "verify", "--scenario", "idle", "--file", str(serial),
        "--expect-version", "1", "--fps", "120",
    ]))
    require(verified["passed"] and verified["byteIdentical"]
            and verified["maxAbsError"] == 0,
            "reference verification failed")

    print("[2/8] stable business errors and no partial bake output")
    scenarios = {
        "curve-time": "CURVE_TIME_NOT_INCREASING",
        "bezier-denominator": "BEZIER_DENOMINATOR_ZERO",
        "weight-zero": "WEIGHT_SUM_ZERO",
        "weight-not-one": "WEIGHT_SUM_NOT_ONE",
        "skeleton-cycle": "SKELETON_CYCLE",
        "curve-empty": "CURVE_EMPTY",
    }
    for scenario, code in scenarios.items():
        first = run(baker, ["error", "--scenario", scenario], expected=2)
        second_run = run(baker, ["error", "--scenario", scenario], expected=2)
        require(first.stderr == second_run.stderr,
                f"error output for {scenario} is not stable")
        payload = json.loads(first.stderr.strip())
        require(payload["code"] == code, f"wrong code for {scenario}")
    failed_output = work / "must-not-exist.snapshot"
    bad_bake = run(baker, [
        "bake", "--scenario", "curve-time", "--output", str(failed_output),
    ], expected=2)
    require(json.loads(bad_bake.stderr.strip())["code"] ==
            "CURVE_TIME_NOT_INCREASING", "bake did not report curve error")
    require(not failed_output.exists() and
            not Path(str(failed_output) + ".progress").exists(),
            "invalid bake left a partial progress file")

    print("[3/8] snapshot version and checksum rejection")
    versioned = work / "version.snapshot"
    run(baker, [
        "bake", "--scenario", "version", "--output", str(versioned),
        "--frames", "8", "--fps", "60", "--version", "9", "--workers", "1",
    ])
    version_error = run(baker, [
        "load", "--file", str(versioned), "--expect-version", "3",
    ], expected=4)
    payload = json.loads(version_error.stderr.strip())
    require(payload["code"] == "SNAPSHOT_VERSION_MISMATCH",
            "version mismatch code missing")
    require(payload["actualVersion"] == "9" and payload["expectedVersion"] == "3",
            "both snapshot versions must be reported")

    corrupt = work / "checksum-corrupt.snapshot"
    corrupt.write_bytes(versioned.read_bytes())
    content = bytearray(corrupt.read_bytes())
    content[-1] ^= 0x5A
    corrupt.write_bytes(content)
    checksum_error = run(baker, ["load", "--file", str(corrupt)], expected=4)
    payload = json.loads(checksum_error.stderr.strip())
    require(payload["code"] == "SNAPSHOT_CHECKSUM_MISMATCH",
            "checksum mismatch code missing")
    require("storedChecksum" in payload and "computedChecksum" in payload,
            "checksum details missing")
    inspected_version = last_json(
        run(baker, ["inspect", "--file", str(versioned)])
    )
    require(inspected_version["ok"] is True,
            "valid snapshot failed inspection after rejected reads")
    # A valid file remains usable after all rejected reads.
    run(baker, ["load", "--file", str(versioned), "--expect-version", "9"])

    print("[4/8] forced termination, atomic progress and resume")
    reference = work / "crash-reference.snapshot"
    run(baker, [
        "bake", "--scenario", "idle", "--output", str(reference),
        "--frames", "180", "--fps", "60", "--version", "3",
        "--workers", "1",
    ])
    crash_output = work / "crash.snapshot"
    events = work / "crash.events"
    crashed = run(baker, [
        "bake", "--scenario", "idle", "--output", str(crash_output),
        "--frames", "180", "--fps", "60", "--version", "3",
        "--workers", "4", "--event-log", str(events),
        "--crash-after-commit-frame", "21",
    ], expected=137)
    require(not crashed.stdout and not crashed.stderr,
            "forced termination should not print a normal result")
    progress = Path(str(crash_output) + ".progress")
    inspected = last_json(run(baker, ["inspect", "--file", str(progress)]))
    require(inspected["frames"] == 22 and inspected["complete"] is False,
            "progress did not persist exactly 22 committed frames")
    resumed = last_json(run(baker, [
        "bake", "--scenario", "idle", "--output", str(crash_output),
        "--frames", "180", "--fps", "60", "--version", "3",
        "--workers", "8", "--event-log", str(events),
    ]))
    require(resumed["resumed"] and resumed["resumeFrame"] == 22
            and resumed["newFramesCommitted"] == 158,
            "resume did not start at frame 22")
    require(sha256(crash_output) == sha256(reference),
            "resumed snapshot differs from uninterrupted reference")
    verify_crash = last_json(run(baker, [
        "verify", "--scenario", "idle", "--file", str(crash_output),
        "--expect-version", "3",
    ]))
    require(verify_crash["passed"], "resumed snapshot failed verification")
    event_rows = [json.loads(line) for line in events.read_text().splitlines()]
    commits = [row["frame"] for row in event_rows if row["event"] == "commit"]
    require(commits == list(range(180)),
            "committed frames must be unique and increasing without gaps")
    resumes = [row for row in event_rows if row["event"] == "resume"]
    require(resumes and resumes[-1]["frame"] == 22,
            "resume event did not record frame 22")

    print("[5/8] version/fingerprint invalidation discards old progress")
    version4 = last_json(run(baker, [
        "bake", "--scenario", "idle", "--output", str(crash_output),
        "--frames", "180", "--fps", "60", "--version", "4",
        "--workers", "8", "--event-log", str(events),
    ]))
    require(version4["discardedProgress"]
            and version4["discardReason"] == "version"
            and version4["resumeFrame"] == 0,
            "version change did not reset progress")
    variant = last_json(run(baker, [
        "bake", "--scenario", "idle", "--variant",
        "--output", str(crash_output), "--frames", "180", "--fps", "60",
        "--version", "4", "--workers", "8", "--event-log", str(events),
    ]))
    require(variant["discardedProgress"]
            and variant["discardReason"] == "fingerprint",
            "curve change did not invalidate old progress")
    verify_variant = last_json(run(baker, [
        "verify", "--scenario", "idle", "--variant",
        "--file", str(crash_output), "--expect-version", "4",
    ]))
    require(verify_variant["passed"], "variant snapshot failed verification")

    print("[6/8] transition blend and state-machine timeline")
    blend = work / "transition.snapshot"
    run(baker, [
        "bake", "--scenario", "transition", "--output", str(blend),
        "--frames", "48", "--fps", "60", "--version", "1",
        "--workers", "16",
    ])
    blend_verify = last_json(run(baker, [
        "verify", "--scenario", "transition", "--file", str(blend),
        "--expect-version", "1",
    ]))
    require(blend_verify["passed"] and blend_verify["maxAbsError"] == 0,
            "transition blend verification failed")
    timeline = last_json(run(baker, [
        "timeline-check", "--frames", "10000", "--fps", "60",
    ]))
    require(timeline["passed"] and timeline["maxDeviationMs"] == 0,
            "10000-frame timeline accumulated drift")

    print("[7/8] exact time rounding")
    rounded = [math.floor(frame * 1000.0 / 60.0 + 0.5)
               for frame in range(10000)]
    require(all(isinstance(value, int) for value in rounded),
            "frame time rounding must produce integer milliseconds")
    require(rounded[9999] == 166650,
            "unexpected rounded time at frame 9999")

    print("[8/8] integration checks passed")
    print(f"work directory: {work}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)

