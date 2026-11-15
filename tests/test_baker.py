from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dialogue_baker.baker import BakeFailure, bake_project
from dialogue_baker.errors import InputFatal
from dialogue_baker.input import Line, Syllable, load_input
from dialogue_baker.pack import parse_header
from dialogue_baker.synth import bounded_stretch, synthesize_line
from dialogue_baker.util import (
    alignment_error_samples,
    samples_to_seconds,
    seconds_to_samples,
)

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PROJECT = ROOT / "examples" / "clean" / "project.json"
CONFIG = ROOT / "examples" / "config.json"
DEGRADED_PROJECT = ROOT / "examples" / "degraded" / "project.json"


def run_cli(project: Path, output_dir: Path, mode: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "dialogue_baker",
            "--project",
            str(project),
            "--config",
            str(CONFIG),
            "--output-dir",
            str(output_dir),
            "--mode",
            mode,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pack_path_for(output_dir: Path, index: dict) -> Path:
    return output_dir / index["pack"]["file"]


def chunk_samples(index: dict, pack_data: bytes, entry: dict) -> np.ndarray:
    chunk = entry["chunk"]
    payload = pack_data[chunk["offset"] : chunk["offset"] + chunk["length"]]
    return np.frombuffer(payload, dtype="<i2").copy()


def rebuild_locale_mix(index: dict, pack_data: bytes, locale: str) -> np.ndarray:
    timeline_samples = index["timeline"]["timeline_end_sample"]
    mix = np.zeros(timeline_samples, dtype=np.int64)
    for entry in index["entries"]:
        if entry["locale"] != locale:
            continue
        audio = chunk_samples(index, pack_data, entry)
        start = entry["timeline"]["start_sample"]
        mix[start : start + audio.size] += audio.astype(np.int64)
    return mix


@pytest.fixture()
def baked_clean(tmp_path: Path) -> tuple[Path, dict, dict, bytes]:
    output_dir = tmp_path / "clean"
    result = run_cli(CLEAN_PROJECT, output_dir, "full")
    assert result.returncode == 0, result.stderr
    index = read_json(output_dir / "index.json")
    report = read_json(output_dir / "bake_report.json")
    pack_data = pack_path_for(output_dir, index).read_bytes()
    return output_dir, index, report, pack_data


def test_seconds_round_trip_has_no_drift() -> None:
    samples = list(range(0, 100_000, 37)) + [1, 2, 23_999, 47_999, 48_000, 10**9]
    for sample in samples:
        seconds = samples_to_seconds(sample, 48_000)
        assert seconds_to_samples(seconds, 48_000) == sample
        assert alignment_error_samples(sample, seconds, 48_000) == 0


def test_short_syllables_do_not_overflow_crossfade_buffer() -> None:
    loaded = load_input(CLEAN_PROJECT, CONFIG)
    voice = loaded.roles["narrator"]
    assert voice is not None
    line = Line(
        text="tiny",
        syllables=(
            Syllable("a", 1),
            Syllable("b", 2),
            Syllable("c", 1),
        ),
        raw={},
    )
    audio = synthesize_line(line, voice, loaded.config)
    assert audio.size == 4
    assert np.all(np.isfinite(audio))
    stretched, ratio, error = bounded_stretch(
        np.arange(10, dtype=np.float64),
        4,
        loaded.config.stretch_min_ratio,
        loaded.config.stretch_max_ratio,
    )
    assert stretched is None
    assert error == "STRETCH_LIMIT_EXCEEDED"
    assert ratio == 0.4


def test_clean_bake_has_consistent_entries_pack_mix_and_alignment(
    baked_clean: tuple[Path, dict, dict, bytes],
) -> None:
    output_dir, index, report, pack_data = baked_clean
    assert report["final_status"] == "ok"
    assert report["exit_code"] == 0
    assert index["summary"]["entry_count"] == 15
    assert len(index["entries"]) == 15
    assert len(index["chunk_table"]) == 15
    assert index["sample_rate"] == 48_000
    assert index["channels"] == 1
    assert index["sample_format"] == "pcm_s16le"

    header = parse_header(pack_data)
    assert header["sample_rate"] == 48_000
    assert header["channels"] == 1
    assert header["chunk_count"] == 15
    assert header["timeline_samples"] == index["timeline"]["timeline_end_sample"]

    max_error = 0
    for entry in index["entries"]:
        timeline = entry["timeline"]
        chunk = entry["chunk"]
        assert timeline["duration_samples"] == chunk["sample_count"]
        assert (
            timeline["end_sample"] - timeline["start_sample"]
            == chunk["sample_count"]
        )
        assert chunk["length"] == chunk["sample_count"] * 2
        assert chunk["offset"] % 64 == 0
        audio = chunk_samples(index, pack_data, entry)
        assert audio.size == timeline["duration_samples"]
        start_error = alignment_error_samples(
            timeline["start_sample"], timeline["start_seconds"], 48_000
        )
        end_error = alignment_error_samples(
            timeline["end_sample"], timeline["end_seconds"], 48_000
        )
        assert start_error == 0
        assert end_error == 0
        assert entry["subtitle"]["timeline"]["start_sample"] == timeline["start_sample"]
        assert entry["subtitle"]["timeline"]["end_sample"] == timeline["end_sample"]
        assert entry["subtitle"]["syllables"][-1]["end_sample"] == timeline["end_sample"]
        max_error = max(
            max_error,
            entry["alignment"]["max_error_samples"],
            max(
                (
                    syllable["alignment"]["max_error_samples"]
                    for syllable in entry["subtitle"]["syllables"]
                ),
                default=0,
            ),
        )
    assert max_error == 0
    assert report["summary"]["max_alignment_error_samples"] == 0

    by_slot: dict[str, set[tuple[int, int]]] = {}
    for entry in index["entries"]:
        by_slot.setdefault(entry["slot_id"], set()).add(
            (
                entry["timeline"]["start_sample"],
                entry["timeline"]["end_sample"],
            )
        )
    assert all(len(timelines) == 1 for timelines in by_slot.values())

    for locale, locale_data in index["locales"].items():
        mix = rebuild_locale_mix(index, pack_data, locale)
        stats = locale_data["mix"]
        sample_peak = int(np.max(np.abs(mix))) if mix.size else 0
        rms = float(np.sqrt(np.mean(np.square(mix.astype(np.float64)))))
        rms_dbfs = (
            20.0 * math.log10(rms / 32768.0)
            if rms > 0
            else stats["rms_dbfs"]
        )
        assert sample_peak == stats["sample_peak"]
        assert round(rms_dbfs, 9) == stats["rms_dbfs"]
        assert int(mix.size) == stats["timeline_samples"]
        assert stats["clipped_sample_count"] == 0
        if stats["peak_protection_triggered"]:
            assert stats["peak_dbfs"] <= -1.0 + 0.001

    assert report["summary"]["peak_protection_trigger_count"] >= 1
    assert any(
        locale["peak_protection_triggered"]
        for locale in report["locales"].values()
    )
    assert (output_dir / "bake_report.json").is_file()


def test_full_bake_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert run_cli(CLEAN_PROJECT, first, "full").returncode == 0
    assert run_cli(CLEAN_PROJECT, second, "full").returncode == 0
    first_index = read_json(first / "index.json")
    second_index = read_json(second / "index.json")
    assert first_index["resource_version"] == second_index["resource_version"]
    assert first_index == second_index
    assert (first / "index.json").read_bytes() == (second / "index.json").read_bytes()
    assert pack_path_for(first, first_index).read_bytes() == pack_path_for(
        second, second_index
    ).read_bytes()


def test_incremental_skip_and_version_change_rebake(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    input_dir = tmp_path / "input"
    shutil.copytree(ROOT / "examples" / "clean", input_dir)
    project = input_dir / "project.json"

    first = run_cli(project, output_dir, "full")
    assert first.returncode == 0, first.stderr
    old_index = read_json(output_dir / "index.json")
    old_version = old_index["resource_version"]
    old_input_version = old_index["input_version"]

    second = run_cli(project, output_dir, "incremental")
    assert second.returncode == 0, second.stderr
    skipped_report = read_json(output_dir / "bake_report.json")
    skipped_index = read_json(output_dir / "index.json")
    assert skipped_report["summary"]["skipped_entries"] == 15
    assert skipped_report["summary"]["baked_entries"] == 0
    assert skipped_report["summary"]["locale_skipped_counts"] == {
        "zh-CN": 5,
        "en-US": 5,
        "ja-JP": 5,
    }
    assert skipped_index["resource_version"] == old_version

    table_path = input_dir / "tables" / "scene02.json"
    table = read_json(table_path)
    table["slots"][0]["lines"]["zh-CN"]["text"] = "钟声仍在回荡"
    table_path.write_text(
        json.dumps(table, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    third = run_cli(project, output_dir, "incremental")
    assert third.returncode == 0, third.stderr
    changed_report = read_json(output_dir / "bake_report.json")
    changed_index = read_json(output_dir / "index.json")
    assert changed_report["summary"]["baked_entries"] == 5
    assert changed_report["summary"]["skipped_entries"] == 10
    assert changed_report["summary"]["locale_skipped_counts"] == {
        "zh-CN": 0,
        "en-US": 5,
        "ja-JP": 5,
    }
    assert changed_report["previous_resource_version"] == old_version
    assert changed_index["resource_version"] != old_version
    assert changed_index["input_version"] != old_input_version
    assert len(list(output_dir.glob("dialogue-*.pack"))) == 1


def test_config_change_forces_all_entries_to_rebake(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config_path = tmp_path / "config.json"
    config = read_json(CONFIG)
    config["synthesis"]["crossfade_ms"] = 5.0
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    first = subprocess.run(
        [
            sys.executable,
            "-m",
            "dialogue_baker",
            "--project",
            str(CLEAN_PROJECT),
            "--config",
            str(CONFIG),
            "--output-dir",
            str(output_dir),
            "--mode",
            "full",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    old_index = read_json(output_dir / "index.json")
    old_version = old_index["resource_version"]
    old_config_hash = old_index["config_hash"]
    second = subprocess.run(
        [
            sys.executable,
            "-m",
            "dialogue_baker",
            "--project",
            str(CLEAN_PROJECT),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "incremental",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    report = read_json(output_dir / "bake_report.json")
    index = read_json(output_dir / "index.json")
    assert report["summary"]["baked_entries"] == 15
    assert report["summary"]["skipped_entries"] == 0
    assert index["resource_version"] != old_version
    assert index["config_hash"] != old_config_hash


def test_corrupt_old_pack_forces_rebuild(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    assert run_cli(CLEAN_PROJECT, output_dir, "full").returncode == 0
    index = read_json(output_dir / "index.json")
    pack_path = pack_path_for(output_dir, index)
    data = bytearray(pack_path.read_bytes())
    data[-1] ^= 0xFF
    pack_path.write_bytes(data)
    result = run_cli(CLEAN_PROJECT, output_dir, "incremental")
    assert result.returncode == 0, result.stderr
    report = read_json(output_dir / "bake_report.json")
    assert report["summary"]["baked_entries"] == 15
    assert report["summary"]["skipped_entries"] == 0
    assert any(
        warning["code"] == "OLD_PACK_INVALID" for warning in report["warnings"]
    )


def test_changed_layout_recomputes_offsets_instead_of_copying_old_ones(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    shutil.copytree(ROOT / "examples" / "clean", input_dir)
    project = input_dir / "project.json"
    assert run_cli(project, output_dir, "full").returncode == 0
    old_index = read_json(output_dir / "index.json")
    old_entry = next(
        entry
        for entry in old_index["entries"]
        if entry["entry_id"] == "market_heroine/en-US"
    )

    table_path = input_dir / "tables" / "scene01.json"
    table = read_json(table_path)
    table["slots"].append(
        {
            "slot_id": "inserted_before_mix",
            "role": "guard",
            "start_sample": 0,
            "duration_samples": 12000,
            "lines": {
                locale: {
                    "text": "New",
                    "syllables": [
                        {"text": "New", "duration_samples": 12000}
                    ],
                }
                for locale in ("zh-CN", "en-US", "ja-JP")
            },
        }
    )
    table_path.write_text(
        json.dumps(table, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    assert run_cli(project, output_dir, "incremental").returncode == 0
    new_index = read_json(output_dir / "index.json")
    new_entry = next(
        entry
        for entry in new_index["entries"]
        if entry["entry_id"] == "market_heroine/en-US"
    )
    assert new_entry["chunk"]["offset"] != old_entry["chunk"]["offset"]
    for chunk in new_index["chunk_table"]:
        entry = next(
            item for item in new_index["entries"] if item["entry_id"] == chunk["entry_id"]
        )
        assert entry["chunk"]["offset"] == chunk["offset"]


def test_degraded_entries_have_codes_but_valid_entries_are_baked(tmp_path: Path) -> None:
    output_dir = tmp_path / "degraded"
    result = run_cli(DEGRADED_PROJECT, output_dir, "full")
    assert result.returncode == 2
    report = read_json(output_dir / "bake_report.json")
    index = read_json(output_dir / "index.json")
    assert report["final_status"] == "degraded"
    assert report["exit_code"] == 2
    assert report["summary"]["entry_count"] == 6
    assert report["summary"]["error_codes"] == {
        "EMPTY_TEXT": 1,
        "MISSING_LINE": 1,
        "STRETCH_LIMIT_EXCEEDED": 1,
    }
    statuses = {entry["entry_id"]: entry for entry in report["entries"]}
    assert statuses["degraded_middle/zh-CN"]["error_code"] == "EMPTY_TEXT"
    assert statuses["degraded_middle/en-US"]["error_code"] == "MISSING_LINE"
    assert (
        statuses["degraded_middle/ja-JP"]["error_code"]
        == "STRETCH_LIMIT_EXCEEDED"
    )
    assert statuses["valid_first/zh-CN"]["status"] == "baked"
    assert index["summary"]["degraded"] is True
    assert (output_dir / "bake_report.json").read_text(encoding="utf-8")


def test_unparseable_manifest_preserves_previous_pair(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    input_dir = tmp_path / "input"
    shutil.copytree(ROOT / "examples" / "clean", input_dir)
    project = input_dir / "project.json"
    assert run_cli(project, output_dir, "full").returncode == 0
    index_before = (output_dir / "index.json").read_bytes()
    index = read_json(output_dir / "index.json")
    pack_path = pack_path_for(output_dir, index)
    pack_before = pack_path.read_bytes()

    project.write_text("{ not valid json", encoding="utf-8")
    result = run_cli(project, output_dir, "incremental")
    assert result.returncode == 3
    assert (output_dir / "index.json").read_bytes() == index_before
    assert pack_path.read_bytes() == pack_before
    report = read_json(output_dir / "bake_report.json")
    assert report["final_status"] == "failed"
    assert report["exit_code"] == 3
    assert report["errors"][0]["code"] == "INPUT_JSON_INVALID"


def test_corrupt_table_degrades_but_other_tables_continue(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    shutil.copytree(ROOT / "examples" / "clean", input_dir)
    (input_dir / "tables" / "scene02.json").write_text(
        '{"schema_version": 1, "slots": [',
        encoding="utf-8",
    )
    result = run_cli(input_dir / "project.json", output_dir, "full")
    assert result.returncode == 2
    report = read_json(output_dir / "bake_report.json")
    index = read_json(output_dir / "index.json")
    assert report["final_status"] == "degraded"
    assert report["summary"]["error_codes"] == {"INPUT_JSON_INVALID": 1}
    assert report["summary"]["entry_count"] == 9
    assert index["summary"]["entry_count"] == 9


def test_index_commit_failure_keeps_previous_valid_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    input_dir = tmp_path / "input"
    shutil.copytree(ROOT / "examples" / "clean", input_dir)
    project = input_dir / "project.json"
    assert run_cli(project, output_dir, "full").returncode == 0
    index_before = (output_dir / "index.json").read_bytes()
    old_index = read_json(output_dir / "index.json")
    old_pack_path = pack_path_for(output_dir, old_index)
    old_pack_hash = hashlib.sha256(old_pack_path.read_bytes()).hexdigest()

    table_path = input_dir / "tables" / "scene02.json"
    table = read_json(table_path)
    table["slots"][0]["lines"]["zh-CN"]["text"] = "changed for commit failure"
    table_path.write_text(
        json.dumps(table, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    loaded = load_input(project, CONFIG)

    from dialogue_baker import baker as baker_module

    real_write_json_atomic = baker_module.write_json_atomic

    def inject_failure(path: Path, value: object) -> None:
        if path.name == "index.json":
            raise OSError("injected index commit failure")
        real_write_json_atomic(path, value)

    monkeypatch.setattr(baker_module, "write_json_atomic", inject_failure)
    with pytest.raises(BakeFailure):
        bake_project(loaded, output_dir, mode="incremental")
    assert (output_dir / "index.json").read_bytes() == index_before
    assert old_pack_path.is_file()
    assert hashlib.sha256(old_pack_path.read_bytes()).hexdigest() == old_pack_hash
    assert read_json(output_dir / "bake_report.json")["final_status"] == "failed"


def test_non_finite_json_is_a_fatal_input_error(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    shutil.copytree(ROOT / "examples" / "clean", input_dir)
    bad_config = tmp_path / "config.json"
    bad_config.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            '"target_peak_dbfs": -1.0', '"target_peak_dbfs": NaN'
        ),
        encoding="utf-8",
    )
    with pytest.raises(InputFatal) as captured:
        load_input(input_dir / "project.json", bad_config)
    assert captured.value.issue.code == "INPUT_JSON_INVALID"


def test_ffmpeg_decodes_chunk_as_48khz_mono(
    baked_clean: tuple[Path, dict, dict, bytes], tmp_path: Path
) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe not available")
    _, index, _, pack_data = baked_clean
    entry = index["entries"][0]
    raw_path = tmp_path / "chunk.s16le"
    raw_path.write_bytes(
        pack_data[
            entry["chunk"]["offset"] : entry["chunk"]["offset"]
            + entry["chunk"]["length"]
        ]
    )
    wav_path = tmp_path / "chunk.wav"
    decode = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "s16le",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-i",
            str(raw_path),
            str(wav_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert decode.returncode == 0, decode.stderr
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,codec_name,duration",
            "-of",
            "json",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["sample_rate"] == "48000"
    assert stream["channels"] == 1
    assert stream["codec_name"].startswith("pcm_")
