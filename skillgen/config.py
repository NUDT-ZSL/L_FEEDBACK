from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MANIFEST_VERSION = "1.0"
SUPPORTED_CONFIG_VERSION = 1

@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


class ConfigError(Exception):
    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = list(issues)
        if not self.issues:
            self.issues = [
                ValidationIssue("E_CONFIG", "$", "configuration validation failed")
            ]
        super().__init__("; ".join(issue.message for issue in self.issues))

    @property
    def overall_code(self) -> str:
        codes = {issue.code for issue in self.issues}
        if "E_CONFIG" in codes:
            return "E_CONFIG"
        if "E_MISSING_FILE" in codes:
            return "E_MISSING_FILE"
        return self.issues[0].code


@dataclass(frozen=True)
class SkillConfig:
    skill_id: str
    base_sample: Path
    variant_count: int
    target_duration: float
    transients: tuple[int, ...]
    target_loudness_mlfs: float
    random_seed: int

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "base_sample": str(self.base_sample.resolve()),
            "variant_count": self.variant_count,
            "target_duration": self.target_duration,
            "transients": list(self.transients),
            "target_loudness_mlfs": self.target_loudness_mlfs,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    output_dir: Path
    temp_dir: Path
    ffmpeg: str
    concurrency: int
    skills: tuple[SkillConfig, ...]


def _alias(obj: dict[str, Any], names: tuple[str, ...]) -> tuple[str | None, Any]:
    for name in names:
        if name in obj:
            return name, obj[name]
    return None, None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _add(
    issues: list[ValidationIssue],
    code: str,
    path: str,
    message: str,
) -> None:
    issues.append(ValidationIssue(code, path, message))


def _parse_skill(
    raw: Any,
    skill_index: int,
    base_dir: Path,
    default_seed: Any,
    issues: list[ValidationIssue],
) -> SkillConfig | None:
    prefix = f"skills[{skill_index}]"
    if not isinstance(raw, dict):
        _add(issues, "E_CONFIG", prefix, "skill entry must be an object")
        return None

    _, skill_id_value = _alias(raw, ("skill_id", "skillId", "skillID", "id"))
    skill_id: str | None = None
    if not isinstance(skill_id_value, str) or not skill_id_value.strip():
        _add(issues, "E_CONFIG", f"{prefix}.skill_id", "non-empty string is required")
    else:
        skill_id = skill_id_value.strip()
        if (
            skill_id in {".", ".."}
            or "/" in skill_id
            or "\\" in skill_id
            or "\x00" in skill_id
            or any(ord(character) < 32 for character in skill_id)
        ):
            _add(
                issues,
                "E_CONFIG",
                f"{prefix}.skill_id",
                "must be a safe filename fragment without path separators or controls",
            )

    _, base_value = _alias(
        raw,
        (
            "base_sample",
            "baseSample",
            "base_sample_path",
            "baseSamplePath",
            "sample_path",
            "samplePath",
            "sample",
            "base_audio",
            "baseAudio",
            "source",
        ),
    )
    base_path: Path | None = None
    if not isinstance(base_value, str) or not base_value.strip():
        _add(issues, "E_CONFIG", f"{prefix}.base_sample", "file path is required")
    else:
        base_path = _resolve_path(base_dir, base_value.strip())
        if not base_path.exists():
            _add(
                issues,
                "E_MISSING_FILE",
                f"{prefix}.base_sample",
                f"base sample does not exist: {base_path}",
            )
            base_path = None
        elif not base_path.is_file():
            _add(
                issues,
                "E_MISSING_FILE",
                f"{prefix}.base_sample",
                f"base sample is not a file: {base_path}",
            )
            base_path = None

    _, count_value = _alias(
        raw,
        (
            "variant_count",
            "variantCount",
            "num_variants",
            "numVariants",
            "variants_count",
            "variantsCount",
            "count",
        ),
    )
    variant_count: int | None = None
    if (
        not isinstance(count_value, int)
        or isinstance(count_value, bool)
        or count_value < 1
    ):
        _add(
            issues,
            "E_CONFIG",
            f"{prefix}.variant_count",
            "integer >= 1 is required",
        )
    else:
        variant_count = count_value

    duration_key, duration_value = _alias(
        raw,
        (
            "target_duration",
            "targetDuration",
            "duration",
            "target_duration_sec",
            "targetDurationSec",
            "target_duration_seconds",
            "targetDurationSeconds",
            "target_duration_ms",
            "targetDurationMs",
        ),
    )
    target_duration: float | None = None
    if not _is_number(duration_value) or not math.isfinite(float(duration_value)):
        _add(
            issues,
            "E_CONFIG",
            f"{prefix}.target_duration",
            "finite duration in seconds is required",
        )
    else:
        target_duration = float(duration_value)
        if duration_key and duration_key.lower().endswith("ms"):
            target_duration /= 1000.0
        if target_duration <= 0:
            _add(
                issues,
                "E_CONFIG",
                f"{prefix}.target_duration",
                "duration must be greater than zero",
            )
            target_duration = None

    _, transient_value = _alias(
        raw,
        (
            "transient_markers",
            "transientMarkers",
            "transients",
            "transient_points",
            "transientPoints",
            "transient_markers_samples",
            "transientMarkersSamples",
            "transient_points_samples",
            "transientPointsSamples",
        ),
    )
    transients: tuple[int, ...] | None = None
    if not isinstance(transient_value, list) or not transient_value:
        _add(
            issues,
            "E_CONFIG",
            f"{prefix}.transient_markers",
            "non-empty list of sample positions is required",
        )
    else:
        parsed_transients: list[int] = []
        transient_ok = True
        for marker_index, marker in enumerate(transient_value):
            if isinstance(marker, bool):
                marker_is_valid = False
            elif isinstance(marker, int):
                marker_is_valid = marker >= 0
            elif isinstance(marker, float) and marker.is_integer():
                marker = int(marker)
                marker_is_valid = marker >= 0
            else:
                marker_is_valid = False
            if not marker_is_valid:
                _add(
                    issues,
                    "E_CONFIG",
                    f"{prefix}.transient_markers[{marker_index}]",
                    "marker must be a non-negative integer sample position",
                )
                transient_ok = False
            else:
                parsed_transients.append(marker)
        if transient_ok and parsed_transients:
            transients = tuple(parsed_transients)

    _, loudness_value = _alias(
        raw,
        (
            "target_loudness_mlfs",
            "targetLoudnessMLFS",
            "loudness_target_mlfs",
            "loudnessTargetMLFS",
            "target_loudness",
            "targetLoudness",
            "loudness_target",
            "loudnessTarget",
            "target_lufs",
            "targetLufs",
            "target_mlfs",
            "targetMlfs",
            "targetMLFS",
        ),
    )
    target_loudness: float | None = None
    if not _is_number(loudness_value) or not math.isfinite(float(loudness_value)):
        _add(
            issues,
            "E_CONFIG",
            f"{prefix}.target_loudness_mlfs",
            "finite MLFS/LUFS target is required",
        )
    else:
        target_loudness = float(loudness_value)

    seed_key, seed_value = _alias(
        raw, ("random_seed", "randomSeed", "seed", "rng_seed", "rngSeed")
    )
    if seed_key is None:
        seed_value = default_seed
    seed: int | None = None
    if not isinstance(seed_value, int) or isinstance(seed_value, bool):
        _add(
            issues,
            "E_CONFIG",
            f"{prefix}.random_seed",
            "integer random seed is required",
        )
    else:
        seed = seed_value

    if (
        skill_id is None
        or base_path is None
        or variant_count is None
        or target_duration is None
        or transients is None
        or target_loudness is None
        or seed is None
    ):
        return None

    return SkillConfig(
        skill_id=skill_id,
        base_sample=base_path,
        variant_count=variant_count,
        target_duration=target_duration,
        transients=transients,
        target_loudness_mlfs=target_loudness,
        random_seed=seed,
    )


def _resolve_executable(value: str, base_dir: Path, issues: list[ValidationIssue]) -> str:
    expanded = os.path.expanduser(value)
    if os.sep in expanded or (os.altsep and os.altsep in expanded):
        path = Path(expanded)
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        if not path.is_file():
            _add(
                issues,
                "E_CONFIG",
                "$.ffmpeg",
                f"ffmpeg executable does not exist: {path}",
            )
            return str(path)
        if not os.access(path, os.X_OK):
            _add(
                issues,
                "E_CONFIG",
                "$.ffmpeg",
                f"ffmpeg is not executable: {path}",
            )
        return str(path)

    found = shutil.which(expanded)
    if found is None:
        _add(
            issues,
            "E_CONFIG",
            "$.ffmpeg",
            f"ffmpeg executable not found on PATH: {expanded}",
        )
        return expanded
    return found


def _prepare_directory(
    path: Path, field: str, issues: list[ValidationIssue]
) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _add(issues, "E_CONFIG", field, f"cannot create directory {path}: {exc}")
        return path
    if not path.is_dir():
        _add(issues, "E_CONFIG", field, f"not a directory: {path}")
        return path
    probe = path / f".skillgen_write_probe_{os.getpid()}"
    try:
        probe.write_bytes(b"probe")
        probe.unlink()
    except OSError as exc:
        _add(issues, "E_CONFIG", field, f"directory is not writable {path}: {exc}")
    return path


def load_config(path: str | os.PathLike[str]) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    issues: list[ValidationIssue] = []
    if not config_path.exists():
        raise ConfigError(
            [ValidationIssue("E_MISSING_FILE", "$", f"config does not exist: {config_path}")]
        )
    if not config_path.is_file():
        raise ConfigError(
            [ValidationIssue("E_MISSING_FILE", "$", f"config is not a file: {config_path}")]
        )
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(
            [ValidationIssue("E_CONFIG", "$", f"cannot parse JSON: {exc}")]
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            [ValidationIssue("E_CONFIG", "$", "top-level config must be an object")]
        )

    base_dir = config_path.parent
    config_version = raw.get("version", SUPPORTED_CONFIG_VERSION)
    if (
        not isinstance(config_version, int)
        or isinstance(config_version, bool)
        or config_version != SUPPORTED_CONFIG_VERSION
    ):
        _add(
            issues,
            "E_CONFIG",
            "$.version",
            f"supported config version is {SUPPORTED_CONFIG_VERSION}",
        )

    output_key, output_value = _alias(
        raw,
        (
            "output_dir",
            "outputDir",
            "output_directory",
            "outputDirectory",
            "output",
        ),
    )
    if output_key is None:
        output_value = "output"
    output_dir: Path | None = None
    if not isinstance(output_value, str) or not output_value.strip():
        _add(issues, "E_CONFIG", "$.output_dir", "directory path must be a string")
    else:
        output_dir = _resolve_path(base_dir, output_value.strip())
        _prepare_directory(output_dir, "$.output_dir", issues)

    temp_key, temp_value = _alias(
        raw,
        (
            "temp_dir",
            "tempDir",
            "tmp_dir",
            "tmpDir",
            "temporary_directory",
            "temporaryDirectory",
            "temp",
            "tmp",
        ),
    )
    if temp_key is None:
        temp_value = str(output_dir / ".tmp") if output_dir is not None else ".tmp"
    temp_dir: Path | None = None
    if not isinstance(temp_value, str) or not temp_value.strip():
        _add(issues, "E_CONFIG", "$.temp_dir", "directory path must be a string")
    else:
        temp_dir = _resolve_path(base_dir, temp_value.strip())
        _prepare_directory(temp_dir, "$.temp_dir", issues)

    concurrency_value = raw.get("concurrency", 1)
    concurrency: int | None = None
    if (
        not isinstance(concurrency_value, int)
        or isinstance(concurrency_value, bool)
        or not 1 <= concurrency_value <= 8
    ):
        _add(
            issues,
            "E_CONFIG",
            "$.concurrency",
            "integer between 1 and 8 is required",
        )
    else:
        concurrency = concurrency_value

    ffmpeg_value = raw.get("ffmpeg", raw.get("ffmpeg_path", raw.get("ffmpegPath", "ffmpeg")))
    ffmpeg: str | None = None
    if not isinstance(ffmpeg_value, str) or not ffmpeg_value.strip():
        _add(issues, "E_CONFIG", "$.ffmpeg", "ffmpeg executable path must be a string")
    else:
        ffmpeg = _resolve_executable(ffmpeg_value.strip(), base_dir, issues)

    default_seed = raw.get("random_seed", raw.get("randomSeed", raw.get("seed")))
    skills_value = raw.get("skills", raw.get("skill"))
    if skills_value is None and any(
        name in raw
        for name in (
            "skill_id",
            "skillId",
            "skillID",
            "id",
            "base_sample",
            "baseSample",
            "sample_path",
            "samplePath",
        )
    ):
        skills_value = [raw]
    skills: list[SkillConfig] = []
    if isinstance(skills_value, dict):
        expanded_skills = []
        for key, value in skills_value.items():
            if not isinstance(value, dict):
                expanded_skills.append(value)
                continue
            item = dict(value)
            item.setdefault("skill_id", key)
            expanded_skills.append(item)
        skills_value = expanded_skills
    if not isinstance(skills_value, list) or not skills_value:
        _add(
            issues,
            "E_CONFIG",
            "$.skills",
            "non-empty skills array is required",
        )
        skills_value = []

    seen_ids: set[str] = set()
    for skill_index, raw_skill in enumerate(skills_value):
        parsed = _parse_skill(raw_skill, skill_index, base_dir, default_seed, issues)
        if parsed is None:
            continue
        if parsed.skill_id in seen_ids:
            _add(
                issues,
                "E_CONFIG",
                f"skills[{skill_index}].skill_id",
                f"duplicate skill id: {parsed.skill_id}",
            )
            continue
        seen_ids.add(parsed.skill_id)
        skills.append(parsed)

    if issues:
        raise ConfigError(issues)

    assert output_dir is not None
    assert temp_dir is not None
    assert concurrency is not None
    assert ffmpeg is not None

    config = AppConfig(
        config_path=config_path,
        output_dir=output_dir,
        temp_dir=temp_dir,
        ffmpeg=ffmpeg,
        concurrency=concurrency,
        skills=tuple(skills),
    )
    _validate_audio(config)
    return config


def _validate_audio(config: AppConfig) -> None:
    from .audio_io import load_wav

    issues: list[ValidationIssue] = []
    cache: dict[Path, Any] = {}
    for skill_index, skill in enumerate(config.skills):
        try:
            if skill.base_sample not in cache:
                cache[skill.base_sample] = load_wav(skill.base_sample)
            audio = cache[skill.base_sample]
        except Exception as exc:
            _add(
                issues,
                "E_CONFIG",
                f"skills[{skill_index}].base_sample",
                f"cannot decode base sample: {exc}",
            )
            continue
        if audio.sample_rate < 2000:
            _add(
                issues,
                "E_CONFIG",
                f"skills[{skill_index}].base_sample",
                "sample rate must be at least 2000 Hz",
            )
            continue
        target_samples = int(round(skill.target_duration * audio.sample_rate))
        if target_samples < 1:
            _add(
                issues,
                "E_CONFIG",
                f"skills[{skill_index}].target_duration",
                "target duration is shorter than one sample",
            )
            continue
        for marker_index, marker in enumerate(skill.transients):
            if marker >= len(audio.samples):
                _add(
                    issues,
                    "E_CONFIG",
                    f"skills[{skill_index}].transient_markers[{marker_index}]",
                    f"marker {marker} exceeds base sample frame count {len(audio.samples)}",
                )
            elif marker >= target_samples:
                _add(
                    issues,
                    "E_CONFIG",
                    f"skills[{skill_index}].transient_markers[{marker_index}]",
                    f"marker {marker} exceeds target duration ({target_samples} samples)",
                )
    if issues:
        raise ConfigError(issues)
