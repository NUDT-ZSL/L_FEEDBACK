#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from soft_renderer.imageio import linear_to_srgb_preview, read_pfm
from tests.helpers import run_cli, temporary_files


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str, failures: List[str]) -> None:
    if not condition:
        failures.append(message)


def run_acceptance(work: Path) -> Dict[str, Any]:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    failures: List[str] = []

    scene = work / "scene.json"
    completed = run_cli(
        ["generate", str(scene), "--width", "96", "--height", "72", "--quiet"]
    )
    _require(completed.returncode == 0, "sample generation failed: " + completed.stderr, failures)
    scene_hash_1 = _sha256(scene)
    scene_repeat = work / "scene_repeat.json"
    completed = run_cli(
        ["generate", str(scene_repeat), "--width", "96", "--height", "72", "--quiet"]
    )
    _require(completed.returncode == 0, "sample regeneration failed", failures)
    scene_hash_2 = _sha256(scene_repeat)
    _require(scene_hash_1 == scene_hash_2, "generated scene is not deterministic", failures)

    block_records = []
    block_hashes = []
    block_file_hashes = []
    for block_size in (1, 2, 4, 8):
        output = work / ("block_%d.pfm" % block_size)
        report = work / ("block_%d.json" % block_size)
        arguments = [
            "render",
            str(scene),
            str(output),
            "--preset",
            "low",
            "--block-size",
            str(block_size),
            "--report",
            str(report),
            "--quiet",
        ]
        if block_size == 8:
            arguments.extend(["--png", str(work / "block_8.png")])
        completed = run_cli(arguments)
        _require(
            completed.returncode == 0,
            "block size %d failed: %s" % (block_size, completed.stderr),
            failures,
        )
        if completed.returncode == 0:
            report_data = _load_report(report)
            block_hashes.append(report_data["pixel_result_sha256"])
            block_file_hashes.append(_sha256(output))
            block_records.append(
                {
                    "block_size": block_size,
                    "pixel_result_sha256": report_data["pixel_result_sha256"],
                    "pfm_sha256": block_file_hashes[-1],
                    "total_ms": report_data["timings_ms"]["total"],
                }
            )
    _require(len(set(block_hashes)) == 1, "block-size pixel hashes differ", failures)
    _require(len(set(block_file_hashes)) == 1, "block-size PFM bytes differ", failures)

    untiled = work / "untiled.pfm"
    untiled_report = work / "untiled.json"
    completed = run_cli(
        [
            "render",
            str(scene),
            str(untiled),
            "--preset",
            "low",
            "--no-tiling",
            "--report",
            str(untiled_report),
            "--quiet",
        ]
    )
    _require(
        completed.returncode == 0,
        "no-tiling render failed: " + completed.stderr,
        failures,
    )
    untiled_hash = ""
    if completed.returncode == 0:
        untiled_hash = _sha256(untiled)
        _require(
            untiled_hash in block_file_hashes,
            "no-tiling render differs from tiled renders",
            failures,
        )

    png_tree = work / "block_8.png"
    if png_tree.is_file() and (work / "block_8.pfm").is_file():
        hdr = read_pfm(str(work / "block_8.pfm"))
        expected_png = linear_to_srgb_preview(hdr, 1.0)
        actual_png = np.asarray(Image.open(png_tree).convert("RGB"))
        _require(
            np.array_equal(expected_png, actual_png),
            "PNG is not derived from the rendered HDR frame",
            failures,
        )

    whole = work / "layers_whole.pfm"
    layers = work / "layers"
    layer_report = work / "layers_whole.json"
    completed = run_cli(
        [
            "render",
            str(scene),
            str(whole),
            "--preset",
            "medium",
            "--block-size",
            "4",
            "--layers-dir",
            str(layers),
            "--report",
            str(layer_report),
            "--quiet",
        ]
    )
    _require(
        completed.returncode == 0,
        "layered render failed: " + completed.stderr,
        failures,
    )
    composite = work / "layers_composite.pfm"
    composite_report = work / "layers_composite.json"
    completed = run_cli(
        [
            "composite",
            "--layers-dir",
            str(layers),
            "-o",
            str(composite),
            "--report",
            str(composite_report),
            "--quiet",
        ]
    )
    _require(
        completed.returncode == 0,
        "layer composite failed: " + completed.stderr,
        failures,
    )
    if whole.is_file() and composite.is_file():
        _require(
            whole.read_bytes() == composite.read_bytes(),
            "layer composite differs from whole render",
            failures,
        )
    layer_hash = ""
    composite_hash = ""
    if layer_report.is_file():
        layer_hash = _load_report(layer_report)["pixel_result_sha256"]
    if composite_report.is_file():
        composite_hash = _load_report(composite_report)["pixel_result_sha256"]
    _require(layer_hash == composite_hash, "layer report hashes differ", failures)

    preset_scene = work / "preset_scene.json"
    run_cli(
        ["generate", str(preset_scene), "--width", "32", "--height", "24", "--quiet"]
    )
    preset_records = []
    preset_hashes: Dict[str, List[str]] = {"low": [], "medium": [], "high": []}
    for index, preset in enumerate(("low", "low", "medium", "high")):
        output = work / ("preset_%d.pfm" % index)
        report = work / ("preset_%d.json" % index)
        completed = run_cli(
            [
                "render",
                str(preset_scene),
                str(output),
                "--preset",
                preset,
                "--block-size",
                "4",
                "--report",
                str(report),
                "--quiet",
            ]
        )
        _require(
            completed.returncode == 0,
            "preset %s failed: %s" % (preset, completed.stderr),
            failures,
        )
        if completed.returncode == 0:
            data = _load_report(report)
            preset_hashes[preset].append(data["pixel_result_sha256"])
            preset_records.append(
                {
                    "preset": preset,
                    "parameters": data["parameters"],
                    "pixel_result_sha256": data["pixel_result_sha256"],
                    "total_ms": data["timings_ms"]["total"],
                }
            )
    _require(
        len(set(preset_hashes["low"])) == 1,
        "repeated low preset hash is unstable",
        failures,
    )
    if len(preset_records) == 4:
        low = preset_records[0]["parameters"]
        medium = preset_records[2]["parameters"]
        high = preset_records[3]["parameters"]
        _require(low["shadow_resolution"] < medium["shadow_resolution"] < high["shadow_resolution"], "shadow resolutions do not increase by preset", failures)
        _require(low["pcf_samples"] < medium["pcf_samples"] < high["pcf_samples"], "PCF sample counts do not increase by preset", failures)
        _require(low["supersample"] < medium["supersample"] < high["supersample"], "supersample factors do not increase by preset", failures)

    shadow_statistics = {}
    if layer_report.is_file():
        shadow_statistics = _load_report(layer_report)["shadow_statistics"]
    _require(
        any(
            values["fully_lit_fragment_samples"] > 0
            and values["penumbra_fragment_samples"] > 0
            and values["fully_shadowed_fragment_samples"] > 0
            for values in shadow_statistics.values()
        ),
        "shadow statistics do not contain all three visibility classes",
        failures,
    )

    bad_json = work / "bad.json"
    bad_json.write_text('{"materials": [}', encoding="utf-8")
    completed = run_cli(
        ["render", str(bad_json), str(work / "bad.pfm"), "--quiet"]
    )
    _require(
        completed.returncode != 0 and "invalid JSON" in completed.stderr,
        "malformed JSON did not produce a specific nonzero failure",
        failures,
    )
    completed = run_cli(
        [
            "render",
            str(scene),
            str(work / "bad_block.pfm"),
            "--block-size",
            "0",
            "--quiet",
        ]
    )
    _require(
        completed.returncode != 0 and "unsupported block size" in completed.stderr,
        "unsupported block size did not fail specifically",
        failures,
    )

    blocker = work / "not_a_directory"
    blocker.write_text("blocked", encoding="utf-8")
    completed = run_cli(
        [
            "render",
            str(scene),
            str(work / "atomic.pfm"),
            "--preset",
            "low",
            "--png",
            str(blocker / "preview.png"),
            "--quiet",
        ]
    )
    _require(completed.returncode != 0, "unwritable output did not fail", failures)
    _require(not (work / "atomic.pfm").exists(), "failed render left a PFM", failures)
    residue = temporary_files(work)
    _require(not residue, "temporary files remain: %r" % residue, failures)

    return {
        "format": "offline-soft-renderer-acceptance",
        "version": 1,
        "passed": not failures,
        "failures": failures,
        "generated_scene_sha256": scene_hash_1,
        "generated_scene_deterministic": scene_hash_1 == scene_hash_2,
        "block_sizes": block_records,
        "block_bitwise_identical": len(set(block_file_hashes)) == 1,
        "no_tiling_bitwise_identical": untiled_hash in block_file_hashes,
        "no_tiling_pfm_sha256": untiled_hash,
        "png_matches_hdr_transform": not any("PNG" in value for value in failures),
        "layer_whole_sha256": layer_hash,
        "layer_composite_sha256": composite_hash,
        "layer_composite_bitwise_identical": (
            whole.is_file() and composite.is_file() and whole.read_bytes() == composite.read_bytes()
        ),
        "presets": preset_records,
        "preset_hash_stable": len(set(preset_hashes["low"])) == 1,
        "shadow_statistics": shadow_statistics,
        "temporary_files": residue,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic acceptance suite.")
    parser.add_argument("--workdir", help="keep intermediate files in this directory")
    parser.add_argument("--report", help="write the acceptance JSON to this path")
    arguments = parser.parse_args()
    if arguments.workdir:
        work = Path(arguments.workdir).resolve()
        result = run_acceptance(work)
    else:
        with tempfile.TemporaryDirectory() as directory:
            result = run_acceptance(Path(directory))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        report_path = Path(arguments.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
