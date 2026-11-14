"""Built-in end-to-end acceptance self-test."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .errors import RepairFailure
from .jsonutil import fixed_json
from .processor import process_files
from .samples import generate_sample_bundle, write_nonconforming_reference
from .wavio import read_wav


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(fixed_json(value, indent=2) + "\n", encoding="utf-8")


def _error_case(
    manifest: Path,
    output_path: Path,
    report_path: Path,
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    try:
        process_files(manifest.parent / "dialogue.wav", manifest, output_path, report_path)
    except RepairFailure as failure:
        residuals = sorted(
            item.name
            for item in manifest.parent.iterdir()
            if ".stage-" in item.name or ".backup-" in item.name
        )
        passed = (
            not output_path.exists()
            and not report_path.exists()
            and not residuals
        )
        return passed, failure.errors, residuals
    residuals = sorted(
        item.name
        for item in manifest.parent.iterdir()
        if ".stage-" in item.name or ".backup-" in item.name
    )
    return False, [], residuals


def _run_checks(directory: Path) -> dict[str, Any]:
    bundle = generate_sample_bundle(directory)
    output_path = directory / "repaired.wav"
    report_path = directory / "repair-report.json"

    first = process_files(
        bundle.dialogue_wav,
        bundle.manifest_json,
        output_path,
        report_path,
    )
    first_wav = output_path.read_bytes()
    first_report = report_path.read_bytes()

    second = process_files(
        bundle.dialogue_wav,
        bundle.manifest_json,
        output_path,
        report_path,
    )
    second_wav = output_path.read_bytes()
    second_report = report_path.read_bytes()

    source = read_wav(bundle.dialogue_wav)
    repaired = read_wav(output_path)
    length_passed = (
        source.sample_count == repaired.sample_count
        and repaired.sample_count == first.sample_count
        and repaired.sample_count == second.sample_count
    )
    report_contract_passed = (
        first.report.get("output", {}).get("sample_count") == source.sample_count
        and len(first.report.get("operations", [])) == 3
        and first.report["operations"][0]["fade_length_samples"] == 96
        and first.report["operations"][1]["fade_length_samples"] == 144
        and first.report["operations"][1]["splice"]["reference_file"]
        == "reference.wav"
        and first.report["operations"][1]["splice"]["reference_interval"]
        == {"start_sample": 1200, "end_sample": 3600}
    )

    bitwise_passed = (
        first_wav == second_wav
        and first_report == second_report
        and first.wav_sha256 == second.wav_sha256
        and first.report_sha256 == second.report_sha256
    )

    invalid_manifest = {
        "quantization_bits": 16,
        "operations": [
            {
                "index": 0,
                "type": "gain",
                "start_sample": 100,
                "end_sample": 200,
                "gain": -1.0,
            },
            {
                "index": 1,
                "type": "gain",
                "start_sample": 150,
                "end_sample": 250,
                "gain": 1.0,
            },
            {
                "index": 2,
                "type": "mystery",
                "start_sample": 300,
                "end_sample": 400,
            },
            {
                "index": 3,
                "type": "replace",
                "start_sample": 900000,
                "end_sample": 900100,
                "frequency_hz": 440.0,
                "phase_radians": 0.0,
                "fade_ms": -1.0,
            },
        ],
    }
    invalid_manifest_path = directory / "invalid.json"
    _write_json(invalid_manifest_path, invalid_manifest)
    invalid_passed, invalid_errors, invalid_residuals = _error_case(
        invalid_manifest_path,
        directory / "invalid-output.wav",
        directory / "invalid-report.json",
    )
    invalid_categories = {error["category"] for error in invalid_errors}
    invalid_indices = {error["operation_index"] for error in invalid_errors}
    invalid_passed = (
        invalid_passed
        and {
            "negative_gain",
            "overlapping_intervals",
            "unknown_operation_type",
            "interval_out_of_bounds",
            "invalid_fade_length",
        }.issubset(invalid_categories)
        and {0, 1, 2, 3}.issubset(invalid_indices)
    )

    missing_ref_manifest = {
        "quantization_bits": 16,
        "operations": [
            {
                "index": 0,
                "type": "splice",
                "start_sample": 100,
                "end_sample": 200,
                "reference_file": "does-not-exist.wav",
                "reference_start": 0,
                "reference_end": 100,
                "fade_ms": 1.0,
            }
        ],
    }
    missing_ref_path = directory / "missing-reference.json"
    _write_json(missing_ref_path, missing_ref_manifest)
    missing_passed, missing_errors, missing_residuals = _error_case(
        missing_ref_path,
        directory / "missing-output.wav",
        directory / "missing-report.json",
    )
    missing_passed = (
        missing_passed
        and any(
            error["category"] == "missing_reference"
            and error["operation_index"] == 0
            for error in missing_errors
        )
    )

    bad_reference = write_nonconforming_reference(directory / "bad-reference.wav")
    bad_ref_manifest = {
        "quantization_bits": 16,
        "operations": [
            {
                "index": 0,
                "type": "splice",
                "start_sample": 100,
                "end_sample": 200,
                "reference_file": bad_reference.name,
                "reference_start": 0,
                "reference_end": 100,
                "fade_ms": 1.0,
            }
        ],
    }
    bad_ref_path = directory / "bad-reference.json"
    _write_json(bad_ref_path, bad_ref_manifest)
    bad_ref_passed, bad_ref_errors, bad_ref_residuals = _error_case(
        bad_ref_path,
        directory / "bad-output.wav",
        directory / "bad-report.json",
    )
    bad_ref_passed = (
        bad_ref_passed
        and any(
            error["category"] == "reference_format_mismatch"
            and error["operation_index"] == 0
            for error in bad_ref_errors
        )
    )

    sentinel_output = directory / "sentinel-output.wav"
    sentinel_report = directory / "sentinel-report.json"
    sentinel_output.write_bytes(b"ORIGINAL-WAV-MUST-SURVIVE")
    sentinel_report.write_bytes(b"ORIGINAL-REPORT-MUST-SURVIVE")
    try:
        process_files(
            bundle.dialogue_wav,
            invalid_manifest_path,
            sentinel_output,
            sentinel_report,
        )
        sentinel_passed = False
    except RepairFailure:
        sentinel_passed = (
            sentinel_output.read_bytes() == b"ORIGINAL-WAV-MUST-SURVIVE"
            and sentinel_report.read_bytes() == b"ORIGINAL-REPORT-MUST-SURVIVE"
        )

    error_paths_passed = (
        invalid_passed
        and missing_passed
        and bad_ref_passed
        and sentinel_passed
    )
    checks = [
        {
            "name": "track_length_preserved",
            "passed": length_passed and report_contract_passed,
            "details": {
                "source_sample_count": source.sample_count,
                "output_sample_count": repaired.sample_count,
                "report_contract_passed": report_contract_passed,
            },
        },
        {
            "name": "bitwise_reproducible",
            "passed": bitwise_passed,
            "details": {
                "wav_sha256": first.wav_sha256,
                "report_sha256": first.report_sha256,
            },
        },
        {
            "name": "error_paths_produce_no_files",
            "passed": error_paths_passed,
            "details": {
                "invalid_manifest": invalid_passed,
                "invalid_categories": sorted(
                    {error["category"] for error in invalid_errors}
                ),
                "invalid_residuals": invalid_residuals or missing_residuals or bad_ref_residuals,
                "missing_reference": missing_passed,
                "bad_reference_format": bad_ref_passed,
                "existing_targets_preserved": sentinel_passed,
            },
        },
    ]
    return {
        "ok": all(check["passed"] for check in checks),
        "checks": checks,
    }


def run_selftest(workdir: str | Path | None = None) -> dict[str, Any]:
    if workdir is not None:
        directory = Path(workdir)
        directory.mkdir(parents=True, exist_ok=True)
        return _run_checks(directory)
    with tempfile.TemporaryDirectory(prefix="dialogue-repair-selftest-") as temporary:
        return _run_checks(Path(temporary))

