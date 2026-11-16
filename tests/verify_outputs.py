#!/usr/bin/env python3
"""Validate deterministic CSM output directories and compare two runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"{path}: invalid PNG signature")
    if data[12:16] != b"IHDR":
        raise AssertionError(f"{path}: first PNG chunk is not IHDR")
    return struct.unpack(">II", data[16:24])


def close(actual: float, expected: float, tolerance: float = 1e-5) -> bool:
    return abs(actual - expected) <= tolerance


def expected_filename(frame: int) -> str:
    return f"frame_{frame:04d}"


def validate_run(directory: Path, frames: int, loss_frame: int | None) -> None:
    expected_names = {"stats.json"}
    for frame in range(frames):
        base = expected_filename(frame)
        expected_names.add(f"{base}.png")
        expected_names.add(f"{base}.json")

    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise AssertionError(
            f"{directory}: file set mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )

    aggregate = json.loads((directory / "stats.json").read_text(encoding="utf-8"))
    assert aggregate["frameCount"] == frames
    assert aggregate["width"] == 1280
    assert aggregate["height"] == 720
    assert len(aggregate["frames"]) == frames
    assert aggregate["vertexCount"] == 70288
    assert aggregate["indexCount"] == 416286
    assert aggregate["drawCallsPerFrame"] == 4
    assert aggregate["shadowMapSize"] == 1024
    assert aggregate["cascadeSplitRatios"] == [0.1, 0.3, 0.6]
    assert aggregate["deviceLossInjectionAfterFrame"] == loss_frame
    assert aggregate["deviceLossRecoveries"] == (1 if loss_frame is not None else 0)
    assert close(aggregate["slopeOffset"], 2.0)
    assert close(aggregate["constOffset"], 0.0015)

    expected_distances = [0.5, 14.45, 42.35, 140.0]
    for actual, expected in zip(
        aggregate["cascadeSplitDistances"], expected_distances, strict=True
    ):
        assert close(actual, expected)

    png_hashes: set[str] = set()
    for frame in range(frames):
        base = expected_filename(frame)
        png_path = directory / f"{base}.png"
        json_path = directory / f"{base}.json"
        width, height = png_dimensions(png_path)
        assert (width, height) == (1280, 720)
        png_hashes.add(sha256(png_path))

        frame_stats = json.loads(json_path.read_text(encoding="utf-8"))
        assert frame_stats["frame"] == frame
        assert close(frame_stats["animationTimeSeconds"], frame / 60.0)
        assert frame_stats["drawCalls"] == 4
        assert frame_stats["shadowMapSizes"] == [1024, 1024, 1024]
        assert frame_stats["cascadeSplitRatios"] == [0.1, 0.3, 0.6]
        assert close(frame_stats["slopeOffset"], 2.0)
        assert close(frame_stats["constOffset"], 0.0015)
        for actual, expected in zip(
            frame_stats["cascadeSplitDistances"], expected_distances, strict=True
        ):
            assert close(actual, expected)
        for cascade in range(3):
            assert frame_stats["cascades"][cascade]["index"] == cascade
            assert frame_stats["cascades"][cascade]["shadowMapSize"] == 1024

        expected_recoveries = (
            1 if loss_frame is not None and frame > loss_frame else 0
        )
        assert frame_stats["deviceLossRecoveries"] == expected_recoveries

        aggregate_frame = aggregate["frames"][frame]
        assert aggregate_frame == frame_stats

    if len(png_hashes) != frames:
        raise AssertionError(
            f"{directory}: expected {frames} unique PNGs, got {len(png_hashes)}"
        )


def compare_runs(run_a: Path, run_b: Path) -> str:
    files_a = sorted(path.name for path in run_a.iterdir() if path.is_file())
    files_b = sorted(path.name for path in run_b.iterdir() if path.is_file())
    assert files_a == files_b
    combined = hashlib.sha256()
    for name in files_a:
        hash_a = sha256(run_a / name)
        hash_b = sha256(run_b / name)
        if hash_a != hash_b:
            raise AssertionError(f"non-deterministic output: {name}")
        combined.update(name.encode("utf-8"))
        combined.update(hash_a.encode("ascii"))
    return combined.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--loss-frame", type=int, default=30)
    args = parser.parse_args()

    validate_run(args.run_a, args.frames, args.loss_frame)
    validate_run(args.run_b, args.frames, args.loss_frame)
    combined = compare_runs(args.run_a, args.run_b)
    print(f"OK: {args.frames} frames per run, byte-identical outputs")
    print(f"OK: {args.frames} unique PNG frames per run")
    print(f"Combined output digest: {combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

