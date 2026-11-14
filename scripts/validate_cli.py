#!/usr/bin/env python3
"""End-to-end assertions for the deterministic light-cluster CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(simulator: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(simulator), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )


def records(result: subprocess.CompletedProcess[str]) -> list[dict]:
    parsed = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parsed.append(json.loads(line))
    return parsed


def frames(parsed: list[dict]) -> list[dict]:
    return [item for item in parsed if item.get("event") == "frame"]


def validate_immediate(simulator: Path) -> None:
    result = run(
        simulator,
        "--frames", "48",
        "--sync", "immediate",
        "--pool-capacity", "4",
        "--seed", "42",
        "--debug",
    )
    expect(result.returncode == 0, f"immediate run failed: {result.stderr}")
    parsed = records(result)
    frame_rows = frames(parsed)
    expect(len(frame_rows) == 48, "immediate run did not emit 48 frame rows")
    expect(frame_rows[0]["uploaded"] is False, "frame 0 should skip upload")
    expect(frame_rows[0]["scene_version"] == 1, "frame 0 version must be 1")
    expect(frame_rows[0]["gpu_version"] == 1, "initial GPU version must be 1")

    previous_version = 0
    for row in frame_rows:
        expect(row["debug_checked"] is True, "debug must run every frame")
        expect(row["debug_ok"] is True, "debug readback must match")
        expect(row["active_lights"] >= 64, "active light floor violated")
        expect(row["point_lights"] >= 64, "point light floor violated")
        expect(
            row["directional_lights"] >= 4,
            "directional light floor violated",
        )
        expect(
            row["active_lights"]
            == row["point_lights"] + row["directional_lights"],
            "light type counts do not sum to active count",
        )
        expect(
            row["gpu_light_count"] == row["active_lights"],
            "immediate GPU count must match frontend",
        )
        expect(
            row["gpu_point_count"] == row["point_lights"],
            "immediate GPU point count must match frontend",
        )
        expect(
            row["gpu_directional_count"] == row["directional_lights"],
            "immediate GPU directional count must match frontend",
        )
        expect(
            row["scene_version"] >= previous_version,
            "scene version must be monotonic",
        )
        previous_version = row["scene_version"]

    expect(
        any(
            row["uploaded"] and not row["buffer_grew"]
            for row in frame_rows[1:]
        ),
        "expected reuse of an existing immediate buffer",
    )
    expect(
        any(
            row["buffer_grew"] and row["buffer_capacity"] > 128
            for row in frame_rows
        ),
        "runtime buffer expansion was not observed",
    )
    expect(
        any(
            item.get("business_code") == "ANIM_POOL_FULL"
            for item in parsed
        ),
        "pool-full rejection was not reported",
    )
    expect(
        any(item.get("event") == "light_added" for item in parsed),
        "deterministic sequence did not add a light",
    )
    expect(
        any(item.get("event") == "light_removed" for item in parsed),
        "deterministic sequence did not remove a light",
    )
    expect(
        any(
            row["animations_applied"] > 0
            for row in frame_rows
        ),
        "animation interpolation never applied",
    )


def validate_deferred(simulator: Path) -> None:
    delay = 4
    result = run(
        simulator,
        "--frames", "48",
        "--sync", "deferred",
        "--delay", str(delay),
        "--pool-capacity", "4",
        "--seed", "42",
        "--debug",
    )
    expect(result.returncode == 0, f"deferred run failed: {result.stderr}")
    frame_rows = frames(records(result))
    expect(len(frame_rows) == 48, "deferred run did not emit 48 frame rows")
    expect(
        any(
            row["uploaded"]
            and row["upload_reason"] == "deferred_due"
            and row["snapshot_age_frames"] == delay
            for row in frame_rows
        ),
        "no upload occurred exactly at the configured delay",
    )
    for row in frame_rows:
        expect(
            row["snapshot_age_frames"] <= delay,
            "deferred lag exceeded N frames",
        )
        expect(
            row["pending_snapshots"] <= delay,
            "pending snapshot queue exceeded N",
        )
        expect(row["debug_ok"] is True, "deferred debug readback failed")

    last = frame_rows[-1]
    expect(
        last["gpu_version"] == last["scene_version"],
        "deferred mode did not converge after mutations stopped",
    )
    expect(
        last["upload_reason"] == "version_unchanged",
        "converged frame should reuse the resident buffer",
    )


def validate_snapshot_restore(simulator: Path) -> None:
    result = run(
        simulator,
        "--frames", "48",
        "--sync", "immediate",
        "--pool-capacity", "4",
        "--seed", "42",
        "--save-snapshot-frame", "8",
        "--restore-snapshot-frame", "20",
        "--debug",
    )
    expect(result.returncode == 0, f"snapshot run failed: {result.stderr}")
    parsed = records(result)
    saved = [item for item in parsed if item.get("event") == "snapshot_saved"]
    restored = [
        item for item in parsed if item.get("event") == "snapshot_restored"
    ]
    expect(len(saved) == 1, "exactly one snapshot should be saved")
    expect(len(restored) == 1, "exactly one snapshot should be restored")
    row = restored[0]
    expect(
        row["version"] > row["version_before"],
        "restore must strictly increase the live version",
    )
    expect(
        row["version"] > row["restored_snapshot_version"],
        "restore version must exceed the saved snapshot version",
    )
    expect(
        row["version_strictly_increased"] is True,
        "restore event must attest strict version increase",
    )
    expect(row["active_lights"] >= 64, "restored scene lost active lights")


def validate_debug_failure(simulator: Path) -> None:
    result = run(
        simulator,
        "--frames", "3",
        "--sync", "immediate",
        "--debug",
        "--inject-debug-fault",
    )
    expect(result.returncode == 75, "debug mismatch must exit with code 75")
    parsed = records(result)
    errors = [item for item in parsed if item.get("event") == "debug_error"]
    expect(len(errors) == 1, "debug mismatch did not emit one error row")
    expect(
        errors[0]["business_code"] == "GPU_COUNT_MISMATCH",
        "wrong debug business code",
    )
    frame_rows = frames(parsed)
    expect(len(frame_rows) == 1, "render loop should stop on debug mismatch")
    expect(frame_rows[0]["debug_ok"] is False, "failed frame debug flag wrong")


def validate_determinism(simulator: Path) -> None:
    arguments = (
        "--frames", "32",
        "--sync", "deferred",
        "--delay", "3",
        "--pool-capacity", "5",
        "--seed", "99173",
    )
    first = run(simulator, *arguments)
    second = run(simulator, *arguments)
    expect(first.returncode == 0, "first deterministic run failed")
    expect(second.returncode == 0, "second deterministic run failed")
    expect(first.stdout == second.stdout, "fixed seed output was not identical")


def validate_cli_error(simulator: Path) -> None:
    result = run(simulator, "--frames", "0")
    expect(result.returncode == 2, "invalid CLI must exit with code 2")
    fatal = [
        item for item in records(result) if item.get("event") == "fatal"
    ]
    expect(len(fatal) == 1, "invalid CLI must emit a fatal JSON row")
    expect(
        fatal[0]["business_code"] == "CLI_INVALID_ARGUMENT",
        "invalid CLI business code is wrong",
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <lightcluster_sim>", file=sys.stderr)
        return 2
    simulator = Path(sys.argv[1])
    if not simulator.is_file():
        print(f"simulator not found: {simulator}", file=sys.stderr)
        return 2

    validate_immediate(simulator)
    validate_deferred(simulator)
    validate_snapshot_restore(simulator)
    validate_debug_failure(simulator)
    validate_determinism(simulator)
    validate_cli_error(simulator)
    print("CLI validation passed: immediate, deferred, snapshot, debug, determinism")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
