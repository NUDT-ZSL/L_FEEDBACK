#!/usr/bin/env python3
"""Run the delivered demo and write build/validation/report.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skillsfx.dsp import clipping_sample_count, true_peak_stats
from skillsfx.synth import synthesize_sources
from skillsfx.wavio import read_wav


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _cli(*arguments: str) -> dict[str, Any]:
    completed = _run([sys.executable, str(ROOT / "skillsfx.py"), *arguments])
    return json.loads(completed.stdout)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ffprobe(path: Path) -> dict[str, Any] | None:
    executable = shutil.which("ffprobe")
    if executable is None:
        return None
    completed = _run(
        [
            executable,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "a:0",
            "-show_entries",
            (
                "stream=codec_name,sample_rate,channels,bits_per_sample,"
                "bits_per_raw_sample,duration_ts,nb_read_frames"
            ),
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    return streams[0] if streams else None


def _inspect_level(
    output: Path,
    *,
    crosscheck: bool,
) -> dict[str, Any]:
    manifest = _load_json(output / "manifest.json")
    rows: list[dict[str, Any]] = []
    for variant in manifest["variants"]:
        wav_path = output / variant["output"]["path"]
        audio, info = read_wav(wav_path)
        peak = float(np.max(np.abs(audio), initial=0.0))
        true_peak = true_peak_stats(audio, info.sample_rate)
        probe = _ffprobe(wav_path) if crosscheck else None
        if probe is not None:
            expected_bits = (
                32
                if manifest["recipe"]["output"]["bit_depth"] == "float32"
                else manifest["recipe"]["output"]["bit_depth"]
            )
            observed_bits = probe.get("bits_per_sample") or probe.get(
                "bits_per_raw_sample"
            )
            assert int(probe["sample_rate"]) == info.sample_rate
            assert int(probe["channels"]) == info.channels
            assert int(observed_bits) == int(expected_bits)
            probe_frames = int(probe.get("duration_ts") or 0)
            if probe_frames == 0:
                probe_frames = int(
                    round(float(probe["duration"]) * info.sample_rate)
                )
            assert probe_frames == info.frame_count
        rows.append(
            {
                "index": variant["index"],
                "sha256": variant["output"]["sha256"],
                "frames": info.frame_count,
                "duration_seconds": info.frame_count / info.sample_rate,
                "peak_dbfs": 20.0 * np.log10(peak) if peak > 0 else None,
                "true_peak_dbtp": true_peak["true_peak_dbtp"],
                "clipping_sample_count": clipping_sample_count(audio),
                "normalization_gain_db": variant["analysis"][
                    "normalization_gain_db"
                ],
                "limiter_trigger_count": variant["analysis"][
                    "limiter_trigger_count"
                ],
                "limiter_trigger_intervals": variant["analysis"][
                    "limiter_trigger_intervals"
                ],
                "warning_codes": [
                    warning["code"]
                    for warning in variant["analysis"]["warnings"]
                ],
                "ffprobe": probe,
            }
        )
    return {
        "level": manifest["recipe"]["level"],
        "output": str(output.relative_to(ROOT)),
        "run_id": manifest["run_id"],
        "variant_count": len(rows),
        "unique_hash_count": len({row["sha256"] for row in rows}),
        "maximum_peak_dbfs": max(
            row["peak_dbfs"] for row in rows if row["peak_dbfs"] is not None
        ),
        "maximum_true_peak_dbtp": max(
            row["true_peak_dbtp"] for row in rows
        ),
        "clipping_sample_count": sum(
            row["clipping_sample_count"] for row in rows
        ),
        "fade_truncated_variant_count": sum(
            "fade_truncated" in row["warning_codes"] for row in rows
        ),
        "variants": rows,
    }


def main() -> int:
    validation_root = ROOT / "build" / "validation"
    if validation_root.exists():
        shutil.rmtree(validation_root)
    validation_root.mkdir(parents=True)

    sources = ROOT / "examples" / "sources"
    source_result = _cli("synth-sources", "--out", str(sources))

    levels: list[dict[str, Any]] = []
    for level in (1, 2, 3):
        output = validation_root / f"ember_level_{level}"
        _cli(
            "generate",
            "--recipe",
            str(ROOT / "examples" / "ember_cleave.json"),
            "--level",
            str(level),
            "--output",
            str(output),
            "--workers",
            "4",
        )
        _cli("verify", "--manifest", str(output / "manifest.json"))
        levels.append(_inspect_level(output, crosscheck=True))

    frost_output = validation_root / "frost_level_5"
    _cli(
        "generate",
        "--recipe",
        str(ROOT / "examples" / "frost_ring.json"),
        "--level",
        "5",
        "--output",
        str(frost_output),
        "--workers",
        "4",
    )
    _cli("verify", "--manifest", str(frost_output / "manifest.json"))
    frost = _inspect_level(frost_output, crosscheck=True)

    format_matrix: list[dict[str, Any]] = []
    base_recipe = _load_json(ROOT / "examples" / "ember_cleave.json")
    base_recipe["events"] = [
        {
            "role": "slash",
            "source": str(
                (ROOT / "examples" / "sources" / "slash_44k_mono.wav").resolve()
            ),
            "offset_ms": 10.0,
            "duration_ms": 120.0,
            "gain": 0.8,
            "channel_map": "mono_to_stereo",
        }
    ]
    base_recipe["variant_count"] = 1
    base_recipe["randomization"]["gain_jitter_db"] = 0.0
    base_recipe["randomization"]["time_jitter_ms"] = 0.0
    base_recipe["randomization"]["tail_harmonics"]["enabled"] = False
    base_recipe["bus_by_level"] = {"1": base_recipe["bus_by_level"]["1"]}
    for sample_rate, channels, bit_depth in (
        (44100, 1, 16),
        (48000, 2, 24),
        (44100, 2, 32),
        (48000, 1, "float32"),
    ):
        label = f"{sample_rate}_{channels}ch_{bit_depth}"
        recipe = json.loads(json.dumps(base_recipe))
        recipe["skill_name"] = f"Format {label}"
        recipe["output"] = {
            "sample_rate": sample_rate,
            "bit_depth": bit_depth,
            "channels": channels,
            "duration_ms": 300.0,
        }
        recipe["events"][0]["channel_map"] = (
            "mono_to_stereo" if channels == 2 else "identity"
        )
        recipe_path = validation_root / f"format_{label}.json"
        recipe_path.write_text(
            json.dumps(recipe, indent=2),
            encoding="utf-8",
        )
        output = validation_root / f"format_{label}"
        _cli(
            "generate",
            "--recipe",
            str(recipe_path),
            "--level",
            "1",
            "--output",
            str(output),
            "--workers",
            "1",
        )
        _cli("verify", "--manifest", str(output / "manifest.json"))
        inspected = _inspect_level(output, crosscheck=True)
        variant = inspected["variants"][0]
        format_matrix.append(
            {
                "requested_sample_rate": sample_rate,
                "requested_channels": channels,
                "requested_bit_depth": bit_depth,
                "codec_name": (
                    variant["ffprobe"].get("codec_name")
                    if variant["ffprobe"]
                    else None
                ),
                "decoded_frames": variant["frames"],
                "true_peak_dbtp": variant["true_peak_dbtp"],
                "clipping_sample_count": variant["clipping_sample_count"],
                "sha256": variant["sha256"],
            }
        )

    sequential = validation_root / "concurrency_workers_1"
    parallel = validation_root / "concurrency_workers_4"
    _cli(
        "generate",
        "--recipe",
        str(ROOT / "examples" / "ember_cleave.json"),
        "--level",
        "1",
        "--output",
        str(sequential),
        "--workers",
        "1",
    )
    _cli(
        "generate",
        "--recipe",
        str(ROOT / "examples" / "ember_cleave.json"),
        "--level",
        "1",
        "--output",
        str(parallel),
        "--workers",
        "4",
    )
    sequential_manifest = _load_json(sequential / "manifest.json")
    parallel_manifest = _load_json(parallel / "manifest.json")
    concurrency = {
        "same_run_id": (
            sequential_manifest["run_id"] == parallel_manifest["run_id"]
        ),
        "same_variant_hashes": [
            item["output"]["sha256"]
            for item in sequential_manifest["variants"]
        ]
        == [
            item["output"]["sha256"]
            for item in parallel_manifest["variants"]
        ],
        "same_variant_manifest_entries": (
            sequential_manifest["variants"] == parallel_manifest["variants"]
        ),
        "workers": [1, 4],
    }

    bad_recipe = _load_json(ROOT / "examples" / "ember_cleave.json")
    bad_recipe["events"][0]["offset_ms"] = -12.0
    bad_recipe_path = validation_root / "bad_recipe.json"
    bad_recipe_path.write_text(
        json.dumps(bad_recipe, indent=2),
        encoding="utf-8",
    )
    bad_output = validation_root / "bad_output"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "skillsfx.py"),
            "generate",
            "--recipe",
            str(bad_recipe_path),
            "--level",
            "1",
            "--output",
            str(bad_output),
            "--workers",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    failure_error = json.loads(completed.stderr)
    failure_cleanup = {
        "return_code": completed.returncode,
        "stdout_empty": completed.stdout == "",
        "error_code": failure_error["error_code"],
        "event_index": failure_error["context"].get("event_index"),
        "source_path": failure_error["context"].get("source_path"),
        "output_directory_absent": not bad_output.exists(),
        "stage_directories": [
            path.name
            for path in validation_root.glob(".bad_output.stage-*")
        ],
    }

    report = {
        "environment": {
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "ffprobe": shutil.which("ffprobe"),
            "ffmpeg": shutil.which("ffmpeg"),
        },
        "source_synthesis": source_result,
        "ember_levels": levels,
        "frost_level_5": frost,
        "format_matrix": format_matrix,
        "concurrency_determinism": concurrency,
        "failure_cleanup": failure_cleanup,
    }
    report_path = validation_root / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "status": "ok",
        "report": str(report_path.relative_to(ROOT)),
        "levels_validated": [item["level"] for item in levels] + [frost["level"]],
        "clipping_sample_count": sum(
            item["clipping_sample_count"] for item in levels
        )
        + frost["clipping_sample_count"],
        "maximum_true_peak_dbtp": max(
            [item["maximum_true_peak_dbtp"] for item in levels]
            + [frost["maximum_true_peak_dbtp"]]
        ),
        "concurrency_determinism": concurrency,
        "failure_cleanup": failure_cleanup,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
