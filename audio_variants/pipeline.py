from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import AppConfig, ConfigError, SoundOverride
from .dsp import create_variant
from .loudness import measure, normalize_loudness
from .wav_io import WavError, WavInfo, read_wav, write_wav

SCHEMA_VERSION = 1


@dataclass
class SourceResult:
    source_path: Path
    status: str
    requested_variants: int
    variants: List[Dict[str, Any]]
    error: Optional[str] = None
    source_info: Optional[WavInfo] = None
    seed: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "source_path": str(self.source_path),
            "source_file": self.source_path.name,
            "status": self.status,
            "requested_variant_count": self.requested_variants,
            "variant_count": self.requested_variants,
            "generated_variant_count": len(self.variants),
            "seed": self.seed,
            "variants": self.variants,
        }
        if self.source_info is not None:
            record["source_format"] = {
                "container": self.source_info.format,
                "subtype": self.source_info.subtype,
                "sample_rate": self.source_info.sample_rate,
                "channels": self.source_info.channels,
                "frames": self.source_info.frames,
            }
        if self.error is not None:
            record["error"] = self.error
        return record


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def variant_seed(base_seed: int, source_name: str, index: int) -> int:
    material = f"game-audio-variants:v1:{base_seed}:{source_name}:{index}".encode(
        "utf-8"
    )
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def random_parameters(
    base_seed: int, source_name: str, index: int, sample_rate: int
) -> Tuple[int, Dict[str, float]]:
    seed = variant_seed(base_seed, source_name, index)
    generator = np.random.Generator(np.random.PCG64(seed))
    pitch = float(generator.uniform(-3.0, 3.0))
    stretch = float(generator.uniform(0.95, 1.05))
    cutoff_ratio = float(generator.uniform(0.70, 1.0))
    nyquist = sample_rate / 2.0
    cutoff_hz = cutoff_ratio * nyquist
    return seed, {
        "pitch_semitones": pitch,
        "time_stretch_ratio": stretch,
        "lowpass_cutoff_ratio": cutoff_ratio,
        "lowpass_cutoff_hz": cutoff_hz,
    }


def _repo_relative_source(source: Path, output_dir: Path) -> str:
    try:
        relative = os.path.relpath(str(source), start=str(output_dir))
    except ValueError:
        return source.name
    return relative.replace(os.sep, "/")


def _relative_output(path: Path, output_dir: Path) -> str:
    return os.path.relpath(str(path), start=str(output_dir)).replace(os.sep, "/")


def _build_output_plan(
    sources: Sequence[Path], config: AppConfig
) -> Tuple[Dict[Path, List[Path]], Dict[Path, str]]:
    source_paths = {str(path.resolve()).casefold() for path in sources}
    generated_owners: Dict[str, List[Path]] = {}
    plan: Dict[Path, List[Path]] = {}
    for source in sources:
        try:
            override = config.override_for(source.name, source.stem)
        except ConfigError:
            plan[source] = []
            continue
        count = override.variants if override else config.default_variants
        outputs = [
            config.output_dir / f"{source.stem}_{index}.wav"
            for index in range(1, count + 1)
        ]
        plan[source] = outputs
        for output in outputs:
            generated_owners.setdefault(output.name.casefold(), []).append(source)

    problems: Dict[Path, str] = {}
    for source, outputs in plan.items():
        for output in outputs:
            key = output.name.casefold()
            owners = generated_owners.get(key, [])
            if len(set(owners)) > 1:
                problems[source] = f"output filename collision for {output.name!r}"
            elif str(output.resolve()).casefold() in source_paths:
                problems.setdefault(
                    source,
                    f"output filename {output.name!r} would overwrite an input WAV",
                )
    return plan, problems


def _json_safe_float(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _process_source(
    source: Path,
    config: AppConfig,
    override: Optional[SoundOverride],
    output_paths: Sequence[Path],
) -> Tuple[SourceResult, List[Dict[str, Any]]]:
    requested = override.variants if override else config.default_variants
    base_seed = (
        override.seed if override and override.seed is not None else config.global_seed
    )
    audio, info = read_wav(source)
    variants: List[Dict[str, Any]] = []
    manifest_entries: List[Dict[str, Any]] = []
    error: Optional[str] = None

    for variant_index, output_path in enumerate(output_paths, start=1):
        try:
            seed, parameters = random_parameters(
                base_seed, source.name, variant_index, info.sample_rate
            )
            processed = create_variant(
                audio,
                info.sample_rate,
                pitch_semitones=parameters["pitch_semitones"],
                stretch_ratio=parameters["time_stretch_ratio"],
                cutoff_hz=parameters["lowpass_cutoff_hz"],
            )
            normalized, normalization = normalize_loudness(
                processed, info.sample_rate, config.target_lufs
            )
            write_wav(
                output_path,
                normalized,
                info.sample_rate,
                info.output_subtype,
            )

            written_audio, written_info = read_wav(output_path)
            actual = measure(written_audio, written_info.sample_rate)
            output_relative = _relative_output(output_path, config.output_dir)
            variant_record = {
                "index": variant_index,
                "variant_seed": seed,
                "parameters": {
                    "pitch_semitones": parameters["pitch_semitones"],
                    "time_stretch_ratio": parameters["time_stretch_ratio"],
                    "lowpass_cutoff_ratio": parameters["lowpass_cutoff_ratio"],
                    "lowpass_cutoff_hz": parameters["lowpass_cutoff_hz"],
                },
                "target_lufs": config.target_lufs,
                "input_lufs": _json_safe_float(normalization["input_lufs"]),
                "input_loudness_method": normalization["input_loudness_method"],
                "loudness_lufs": _json_safe_float(actual["loudness_lufs"]),
                "true_peak_dbtp": _json_safe_float(actual["true_peak_dbtp"]),
                "requested_gain_db": float(normalization["requested_gain_db"]),
                "limiting_gain_db": float(normalization["limiting_gain_db"]),
                "peak_limited": bool(normalization["peak_limited"]),
                "output_file": output_relative,
                "output_path": str(output_path),
                "sample_rate": written_info.sample_rate,
                "channels": written_info.channels,
                "subtype": written_info.subtype,
            }
            variants.append(variant_record)
            manifest_entries.append(
                {
                    "path": output_relative,
                    "file": output_relative,
                    "source": _repo_relative_source(source, config.output_dir),
                    "source_file": source.name,
                    "variant": variant_index,
                    "variant_seed": seed,
                    "parameters": dict(variant_record["parameters"]),
                    "loudness_lufs": variant_record["loudness_lufs"],
                    "true_peak_dbtp": variant_record["true_peak_dbtp"],
                }
            )
        except (WavError, OSError, RuntimeError, ValueError) as exc:
            error = f"variant {variant_index} failed: {exc}"
            break
        except Exception as exc:
            error = f"variant {variant_index} failed with unexpected error: {exc}"
            break

    result = SourceResult(
        source_path=source.resolve(),
        status="success" if error is None else "error",
        requested_variants=requested,
        variants=variants,
        error=error,
        source_info=info,
        seed=base_seed,
    )
    return result, manifest_entries


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        sort_keys=False,
    )
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.write("\n")
        os.replace(str(temp_path), str(path))
    except BaseException:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def run_batch(config: AppConfig, input_dir: Path) -> int:
    if not input_dir.exists():
        raise ConfigError(f"input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ConfigError(f"input path is not a directory: {input_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".wav"
        ),
        key=lambda path: path.name,
    )
    output_plan, plan_problems = _build_output_plan(sources, config)
    timestamp = utc_timestamp()
    results: List[SourceResult] = []
    manifest_entries: List[Dict[str, Any]] = []

    for source in sources:
        try:
            override = config.override_for(source.name, source.stem)
        except ConfigError as exc:
            requested = config.default_variants
            result = SourceResult(
                source_path=source.resolve(),
                status="error",
                requested_variants=requested,
                variants=[],
                error=str(exc),
            )
            results.append(result)
            print(f"error: {source.name}: {exc}", file=sys.stderr)
            continue
        requested = override.variants if override else config.default_variants
        if source in plan_problems:
            result = SourceResult(
                source_path=source.resolve(),
                status="error",
                requested_variants=requested,
                variants=[],
                error=plan_problems[source],
            )
            results.append(result)
            print(
                f"error: {source.name}: {plan_problems[source]}",
                file=sys.stderr,
            )
            continue
        output_paths = output_plan.get(source, [])
        try:
            result, entries = _process_source(source, config, override, output_paths)
            manifest_entries.extend(entries)
            if result.error is not None:
                print(f"error: {source.name}: {result.error}", file=sys.stderr)
        except (WavError, ConfigError, OSError, ValueError) as exc:
            result = SourceResult(
                source_path=source.resolve(),
                status="error",
                requested_variants=requested,
                variants=[],
                error=str(exc),
            )
            print(f"error: {source.name}: {exc}", file=sys.stderr)
        except Exception as exc:  # Keep batch processing resilient.
            result = SourceResult(
                source_path=source.resolve(),
                status="error",
                requested_variants=requested,
                variants=[],
                error=f"unexpected processing error: {exc}",
            )
            print(
                f"error: {source.name}: unexpected processing error: {exc}",
                file=sys.stderr,
            )
        results.append(result)

    successful = sum(result.status == "success" for result in results)
    failed = len(results) - successful
    unmatched = config.unmatched_overrides(
        [source.name for source in sources], [source.stem for source in sources]
    )
    for target in unmatched:
        print(
            f"warning: sound override {target!r} did not match any input WAV",
            file=sys.stderr,
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp,
        "config": config.summary(),
        "input_directory": str(input_dir.resolve()),
        "summary": {
            "sources_total": len(sources),
            "sources_succeeded": successful,
            "sources_failed": failed,
            "variants_generated": len(manifest_entries),
            "unmatched_sound_overrides": unmatched,
        },
        "results": [result.to_json() for result in results],
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp,
        "output_directory": ".",
        "variant_count": len(manifest_entries),
        "variants": manifest_entries,
    }
    try:
        _atomic_write_json(config.output_dir / "report.json", report)
        _atomic_write_json(config.output_dir / "manifest.json", manifest)
    except (OSError, ValueError) as exc:
        print(f"error: cannot write report or manifest: {exc}", file=sys.stderr)
        return 1
    return 0 if failed == 0 else 1
