from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .audio_io import AudioData, load_wav, sha256_file, write_float_wav
from .config import AppConfig, MANIFEST_VERSION, SkillConfig
from .dsp import (
    align_transient,
    apply_emergency_limit,
    derive_parameters,
    detect_first_transient,
    final_peak as measured_peak,
    synthesize_variant,
)
from .ffmpeg import encode_pcm16_wav
from .loudness import calibrate_loudness, integrated_loudness_mlfs
from .state import StateStore


DSP_VERSION = "1.0.0"
STATE_VERSION = 1


class OutputBusyError(RuntimeError):
    pass


def write_json_atomic(path: Path, value: dict[str, Any], pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if pretty:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class OutputLock:
    def __init__(self, output_dir: Path):
        self.path = output_dir / ".skillgen.lock"
        self._handle: Any | None = None

    def __enter__(self) -> "OutputLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise OutputBusyError(f"output directory is in use: {self.path.parent}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._handle is not None:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None


def _r(value: float | None, digits: int) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _output_token(output_dir: Path) -> str:
    return hashlib.sha256(str(output_dir.resolve()).encode("utf-8")).hexdigest()[:12]


def _skill_token(skill_id: str) -> str:
    return hashlib.sha256(skill_id.encode("utf-8")).hexdigest()[:10]


def _atomic_install(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        temporary = destination.parent / f".skillgen_install_{destination.name}.{os.getpid()}"
        try:
            shutil.copyfile(source, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    try:
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError:
        pass


class Generator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.output_token = _output_token(config.output_dir)
        self.state = StateStore(config.output_dir / "state.jsonl")
        self._cache_lock = threading.Lock()
        self._audio_cache: dict[Path, AudioData] = {}
        self._hash_cache: dict[Path, str] = {}
        self._stderr_lock = threading.Lock()

    def _log(self, message: str) -> None:
        with self._stderr_lock:
            print(message, file=sys.stderr, flush=True)

    def _load_base(self, skill: SkillConfig) -> AudioData:
        with self._cache_lock:
            cached = self._audio_cache.get(skill.base_sample)
        if cached is not None:
            return cached
        loaded = load_wav(skill.base_sample)
        with self._cache_lock:
            self._audio_cache[skill.base_sample] = loaded
        return loaded

    def _base_hash(self, skill: SkillConfig) -> str:
        with self._cache_lock:
            cached = self._hash_cache.get(skill.base_sample)
        if cached is not None:
            return cached
        digest = sha256_file(skill.base_sample)
        with self._cache_lock:
            self._hash_cache[skill.base_sample] = digest
        return digest

    def _fingerprint(self, skill: SkillConfig) -> str:
        payload = {
            "dsp_version": DSP_VERSION,
            "skill": skill.fingerprint_payload(),
            "base_sha256": self._base_hash(skill),
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def cleanup_stale_temporary_files(self) -> None:
        prefix = f"skillgen_tmp_{self.output_token}_"
        if self.config.temp_dir.is_dir():
            for entry in self.config.temp_dir.iterdir():
                if entry.is_dir() and entry.name.startswith(prefix):
                    shutil.rmtree(entry, ignore_errors=True)
        if self.config.output_dir.is_dir():
            for entry in self.config.output_dir.iterdir():
                if entry.name.startswith(".skillgen_install_"):
                    try:
                        entry.unlink()
                    except OSError:
                        pass

    def _variant_temp_dir(self, skill: SkillConfig, variant_index: int) -> Path:
        return self.config.temp_dir / (
            f"skillgen_tmp_{self.output_token}_{_skill_token(skill.skill_id)}_"
            f"{variant_index:03d}"
        )

    def _variant_file_name(self, skill: SkillConfig, variant_index: int) -> str:
        return f"{skill.skill_id}_var{variant_index:03d}.wav"

    def _base_variant_record(
        self,
        skill: SkillConfig,
        variant_index: int,
        sample_rate: int,
        sample_count: int,
        file_name: str,
    ) -> dict[str, Any]:
        return {
            "index": variant_index,
            "file": file_name,
            "status": "error",
            "error_code": None,
            "error_message": None,
            "sample_rate": sample_rate,
            "sample_count": sample_count,
            "parameters": {
                "pitch_semitones": None,
                "time_stretch": None,
                "lowpass_hz": None,
                "gain_db": None,
            },
            "loudness": {
                "target_mlfs": _r(skill.target_loudness_mlfs, 4),
                "before_mlfs": None,
                "after_mlfs": None,
                "gain_db": None,
            },
            "transient": {
                "reference_sample": skill.transients[0],
                "predicted_sample": None,
                "initial_detected_sample": None,
                "initial_deviation_ms": None,
                "detected_sample": None,
                "deviation_ms": None,
                "aligned": False,
                "iterations": 0,
            },
            "peak": None,
            "pre_limit_peak": None,
            "soft_limited": False,
            "sha256": None,
        }

    def _state_record(
        self,
        skill: SkillConfig,
        variant_index: int,
        fingerprint: str,
        variant: dict[str, Any],
        output_file: str | None,
        output_sha256: str | None,
    ) -> dict[str, Any]:
        success = variant.get("status") == "success"
        return {
            "state_version": STATE_VERSION,
            "config_fingerprint": fingerprint,
            "skill_id": skill.skill_id,
            "variant_index": variant_index,
            "status": variant.get("status"),
            "output_file": output_file if success else None,
            "output_sha256": output_sha256 if success else None,
            "processed_by_pid": os.getpid(),
            "variant": variant,
        }

    def _cached_success(
        self,
        skill: SkillConfig,
        variant_index: int,
        fingerprint: str,
        file_name: str,
    ) -> dict[str, Any] | None:
        record = self.state.get(skill.skill_id, variant_index)
        if record is None:
            return None
        if record.get("status") != "success":
            return None
        if record.get("config_fingerprint") != fingerprint:
            return None
        if record.get("output_file") != file_name:
            return None
        output_path = self.config.output_dir / file_name
        if not output_path.is_file():
            return None
        expected_hash = record.get("output_sha256")
        if not isinstance(expected_hash, str):
            return None
        if sha256_file(output_path) != expected_hash:
            return None
        variant = record.get("variant")
        if not isinstance(variant, dict) or variant.get("status") != "success":
            return None
        return record

    def _error_record(
        self,
        skill: SkillConfig,
        variant_index: int,
        fingerprint: str,
        variant: dict[str, Any],
        code: str,
        message: str,
    ) -> dict[str, Any]:
        variant["status"] = "error"
        variant["error_code"] = code
        variant["error_message"] = message
        record = self._state_record(
            skill,
            variant_index,
            fingerprint,
            variant,
            output_file=None,
            output_sha256=None,
        )
        self.state.append(record)
        self._log(
            f"{skill.skill_id} var{variant_index:03d}: {code}: {message}"
        )
        return record

    def _process_variant(
        self,
        skill: SkillConfig,
        variant_index: int,
        fingerprint: str,
        base_audio: AudioData,
    ) -> dict[str, Any]:
        sample_rate = base_audio.sample_rate
        target_count = max(1, int(round(skill.target_duration * sample_rate)))
        file_name = self._variant_file_name(skill, variant_index)
        variant = self._base_variant_record(
            skill, variant_index, sample_rate, target_count, file_name
        )
        parameters = derive_parameters(
            skill.skill_id, skill.random_seed, variant_index, sample_rate
        )
        variant["parameters"]["pitch_semitones"] = _r(
            parameters.pitch_semitones, 6
        )
        variant["parameters"]["time_stretch"] = _r(parameters.time_stretch, 6)
        variant["parameters"]["lowpass_hz"] = _r(parameters.lowpass_hz, 3)

        reference = skill.transients[0]
        temp_dir = self._variant_temp_dir(skill, variant_index)
        try:
            samples, predicted = synthesize_variant(
                base_audio.samples,
                sample_rate,
                parameters,
                skill.target_duration,
                reference,
            )
            alignment = align_transient(
                samples,
                sample_rate,
                reference,
                predicted,
                tolerance_ms=5.0,
            )
            variant["transient"].update(
                {
                    "predicted_sample": predicted,
                    "initial_detected_sample": alignment.initial_detected_sample,
                    "initial_deviation_ms": _r(
                        alignment.initial_deviation_ms, 4
                    ),
                    "detected_sample": alignment.detected_sample,
                    "deviation_ms": _r(alignment.deviation_ms, 4),
                    "aligned": alignment.aligned,
                    "iterations": alignment.iterations,
                }
            )
            if not alignment.aligned:
                return self._error_record(
                    skill,
                    variant_index,
                    fingerprint,
                    variant,
                    "E_TRANSIENT",
                    alignment.error_message or "transient alignment failed",
                )

            calibration = calibrate_loudness(
                alignment.samples,
                sample_rate,
                skill.target_loudness_mlfs,
                tolerance_lu=0.5,
            )
            variant["loudness"].update(
                {
                    "before_mlfs": _r(calibration.before_mlfs, 4),
                    "after_mlfs": _r(calibration.after_mlfs, 4),
                    "gain_db": _r(calibration.gain_db, 4),
                }
            )
            variant["parameters"]["gain_db"] = _r(calibration.gain_db, 4)
            variant["peak"] = _r(calibration.peak, 6)
            variant["pre_limit_peak"] = _r(calibration.pre_limit_peak, 6)
            variant["soft_limited"] = bool(calibration.limited)
            if calibration.error_code:
                return self._error_record(
                    skill,
                    variant_index,
                    fingerprint,
                    variant,
                    calibration.error_code,
                    calibration.error_message or "loudness calibration failed",
                )

            calibrated = apply_emergency_limit(calibration.samples)
            if calibrated.size < target_count:
                calibrated = np.pad(
                    calibrated, (0, target_count - calibrated.size), mode="constant"
                )
            elif calibrated.size > target_count:
                calibrated = calibrated[:target_count]

            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=False)
            input_wav = temp_dir / "input_float.wav"
            encoded_wav = temp_dir / "encoded_pcm16.wav"
            write_float_wav(input_wav, calibrated, sample_rate)
            encode_result = encode_pcm16_wav(
                self.config.ffmpeg,
                input_wav,
                encoded_wav,
                sample_rate,
                temp_dir,
            )
            if not encode_result.ok or not encoded_wav.is_file():
                message = (
                    f"ffmpeg exited with code {encode_result.returncode}"
                    + (f": {encode_result.stderr_summary}" if encode_result.stderr_summary else "")
                )
                return self._error_record(
                    skill,
                    variant_index,
                    fingerprint,
                    variant,
                    "E_ENCODE",
                    message,
                )

            encoded_audio = load_wav(encoded_wav)
            if encoded_audio.sample_rate != sample_rate:
                return self._error_record(
                    skill,
                    variant_index,
                    fingerprint,
                    variant,
                    "E_ENCODE",
                    f"ffmpeg changed sample rate to {encoded_audio.sample_rate}",
                )
            if len(encoded_audio.samples) != target_count:
                return self._error_record(
                    skill,
                    variant_index,
                    fingerprint,
                    variant,
                    "E_ENCODE",
                    f"ffmpeg produced {len(encoded_audio.samples)} frames, "
                    f"expected {target_count}",
                )

            final_detected = detect_first_transient(
                encoded_audio.samples,
                sample_rate,
                expected_sample=reference,
                search_ms=20.0,
            )
            final_deviation_ms = (
                None
                if final_detected is None
                else (final_detected - reference) * 1000.0 / sample_rate
            )
            if final_detected is not None:
                variant["transient"]["detected_sample"] = final_detected
                variant["transient"]["deviation_ms"] = _r(
                    final_deviation_ms, 4
                )
                variant["transient"]["aligned"] = (
                    abs(float(final_deviation_ms)) <= 5.0
                )
            else:
                variant["transient"]["aligned"] = False
            if (
                final_deviation_ms is None
                or abs(float(final_deviation_ms)) > 5.0
            ):
                return self._error_record(
                    skill,
                    variant_index,
                    fingerprint,
                    variant,
                    "E_TRANSIENT",
                    "encoded PCM failed final transient verification",
                )

            final_loudness = integrated_loudness_mlfs(
                encoded_audio.samples, sample_rate
            )
            encoded_peak = measured_peak(encoded_audio.samples)
            variant["loudness"]["after_mlfs"] = _r(final_loudness, 4)
            variant["peak"] = _r(encoded_peak, 6)
            if (
                final_loudness is None
                or abs(final_loudness - skill.target_loudness_mlfs) > 0.5
            ):
                return self._error_record(
                    skill,
                    variant_index,
                    fingerprint,
                    variant,
                    "E_LOUDNESS",
                    "encoded PCM loudness is outside the target tolerance",
                )

            output_path = self.config.output_dir / file_name
            _atomic_install(encoded_wav, output_path)
            output_hash = sha256_file(output_path)
            variant["status"] = "success"
            variant["error_code"] = None
            variant["error_message"] = None
            variant["sha256"] = output_hash
            record = self._state_record(
                skill,
                variant_index,
                fingerprint,
                variant,
                output_file=file_name,
                output_sha256=output_hash,
            )
            self.state.append(record)
            return record
        except Exception as exc:
            return self._error_record(
                skill,
                variant_index,
                fingerprint,
                variant,
                "E_PROCESSING",
                f"variant processing failed: {type(exc).__name__}: {exc}",
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _process_skill(
        self,
        skill: SkillConfig,
        force: bool,
    ) -> list[dict[str, Any]]:
        base_audio = self._load_base(skill)
        fingerprint = self._fingerprint(skill)
        records: list[dict[str, Any]] = []
        for variant_index in range(skill.variant_count):
            file_name = self._variant_file_name(skill, variant_index)
            cached = None
            if not force:
                cached = self._cached_success(
                    skill, variant_index, fingerprint, file_name
                )
            if cached is not None:
                records.append(cached)
                continue
            records.append(
                self._process_variant(
                    skill, variant_index, fingerprint, base_audio
                )
            )
        return records

    def _manifest(
        self, records_by_skill: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        skills: list[dict[str, Any]] = []
        total = 0
        success = 0
        error_codes: dict[str, int] = {}
        for skill in self.config.skills:
            records = sorted(
                records_by_skill[skill.skill_id],
                key=lambda record: int(record["variant_index"]),
            )
            variants = [record["variant"] for record in records]
            skill_success = sum(
                1 for variant in variants if variant.get("status") == "success"
            )
            skill_errors = len(variants) - skill_success
            for variant in variants:
                code = variant.get("error_code")
                if code:
                    error_codes[str(code)] = error_codes.get(str(code), 0) + 1
            total += len(variants)
            success += skill_success
            skills.append(
                {
                    "skill_id": skill.skill_id,
                    "variant_count": skill.variant_count,
                    "success_count": skill_success,
                    "error_count": skill_errors,
                    "variants": variants,
                }
            )
        return {
            "version": MANIFEST_VERSION,
            "skills": skills,
            "summary": {
                "skill_count": len(skills),
                "variant_count": total,
                "success_count": success,
                "error_count": total - success,
                "all_success": total == success,
                "error_codes": dict(sorted(error_codes.items())),
            },
        }

    def run(self, force: bool = False) -> dict[str, Any]:
        self.cleanup_stale_temporary_files()
        if force:
            self.state.clear()
        records_by_skill: dict[str, list[dict[str, Any]]] = {}
        if self.config.concurrency == 1:
            for skill in self.config.skills:
                records_by_skill[skill.skill_id] = self._process_skill(
                    skill, force
                )
        else:
            with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
                futures = {
                    skill.skill_id: executor.submit(
                        self._process_skill, skill, force
                    )
                    for skill in self.config.skills
                }
                for skill_id, future in futures.items():
                    records_by_skill[skill_id] = future.result()

        manifest = self._manifest(records_by_skill)
        state_records = [
            records_by_skill[skill.skill_id][variant_index]
            for skill in self.config.skills
            for variant_index in range(skill.variant_count)
        ]
        expected_count = sum(skill.variant_count for skill in self.config.skills)
        if len(state_records) != expected_count:
            raise RuntimeError(
                f"internal state mismatch: {len(state_records)} records for "
                f"{expected_count} variants"
            )
        self.state.rewrite(state_records)
        manifest_path = self.config.output_dir / "manifest.json"
        write_json_atomic(manifest_path, manifest)
        try:
            (self.config.output_dir / "error_report.json").unlink()
        except FileNotFoundError:
            pass
        return manifest
