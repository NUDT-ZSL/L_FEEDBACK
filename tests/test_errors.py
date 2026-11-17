import json
import os

from audiotl.errors import ErrorCode, ExitCode

from conftest import read_report, run_cli


def _write_manifest(tmp_path, manifest):
    path = tmp_path / "bad_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_missing_manifest(tmp_path):
    out = tmp_path / "out"
    code = run_cli(tmp_path / "nope.json", out)
    assert code == ExitCode.INPUT_MISSING
    report = read_report(out)
    assert report["errors"][0]["code"] == ErrorCode.E_INPUT_MISSING


def test_missing_segment_input(sample_set, tmp_path):
    manifest = json.loads(json.dumps(sample_set["manifest"]))
    manifest["segments"][0]["path"] = "samples/deleted.wav"
    mpath = _write_manifest(tmp_path, manifest)
    out = tmp_path / "out"
    code = run_cli(mpath, out)
    assert code == ExitCode.INPUT_MISSING
    report = read_report(out)
    assert report["errors"][0]["code"] == ErrorCode.E_INPUT_MISSING
    assert not (out / "mix.wav").exists()


def test_malformed_manifest_json(tmp_path):
    mpath = tmp_path / "broken.json"
    mpath.write_text("{ not json", encoding="utf-8")
    out = tmp_path / "out"
    code = run_cli(mpath, out)
    assert code == ExitCode.MANIFEST_INVALID
    assert read_report(out)["errors"][0]["code"] == ErrorCode.E_MANIFEST_INVALID


def test_manifest_schema_violation(sample_set, tmp_path):
    manifest = json.loads(json.dumps(sample_set["manifest"]))
    manifest["output"]["channels"] = 7
    mpath = _write_manifest(tmp_path, manifest)
    code = run_cli(mpath, tmp_path / "out")
    assert code == ExitCode.MANIFEST_INVALID


def test_negative_fade_length(sample_set, tmp_path):
    manifest = json.loads(json.dumps(sample_set["manifest"]))
    manifest["segments"][1]["fade_in"]["length_ms"] = -5
    mpath = _write_manifest(tmp_path, manifest)
    out = tmp_path / "out"
    code = run_cli(mpath, out)
    assert code == ExitCode.RENDER_ERROR
    report = read_report(out)
    assert report["errors"][0]["code"] == ErrorCode.E_FADE_NEGATIVE
    assert report["errors"][0]["details"]["segment"] == "b_48k_stereo"


def test_fade_longer_than_segment(sample_set, tmp_path):
    manifest = json.loads(json.dumps(sample_set["manifest"]))
    manifest["segments"][2]["fade_in"]["length_ms"] = 5000  # segment is 600ms
    mpath = _write_manifest(tmp_path, manifest)
    out = tmp_path / "out"
    code = run_cli(mpath, out)
    assert code == ExitCode.RENDER_ERROR
    report = read_report(out)
    assert report["errors"][0]["code"] == ErrorCode.E_FADE_TOO_LONG


def test_excessive_overlap(sample_set, tmp_path):
    manifest = json.loads(json.dumps(sample_set["manifest"]))
    # segment a fades out over 700ms but segment b is only 1000ms... make b
    # shorter than the overlap: a fade_out 1200ms > b length 1000ms, while
    # a itself is 800ms+ so fade fits a? a is 800ms, so cap fade to a's own
    # length: use fade_out 800ms (fits a) and shorten b via a new tiny file.
    from audiotl.samplegen import write_tone

    tiny = sample_set["samples"] / "tiny.wav"
    write_tone(str(tiny), 48000, 2, 0.30, 150.0)  # 300ms
    manifest["segments"][1]["path"] = os.path.join("samples", "tiny.wav")
    manifest["segments"][1]["fade_in"]["length_ms"] = 200
    manifest["segments"][0]["fade_out"]["length_ms"] = 700  # > 300ms of b
    mpath = _write_manifest(tmp_path, manifest)
    out = tmp_path / "out"
    code = run_cli(mpath, out)
    assert code == ExitCode.RENDER_ERROR
    report = read_report(out)
    assert report["errors"][0]["code"] == ErrorCode.E_OVERLAP_EXCESSIVE


def test_error_report_leaves_no_valid_outputs(sample_set, tmp_path):
    manifest = json.loads(json.dumps(sample_set["manifest"]))
    manifest["segments"][0]["fade_out"]["length_ms"] = -1
    mpath = _write_manifest(tmp_path, manifest)
    out = tmp_path / "out"
    code = run_cli(mpath, out)
    assert code == ExitCode.RENDER_ERROR
    for name in ("mix.wav", "mapping.json", "summary.json"):
        assert not (out / name).exists()
    report = read_report(out)
    assert report["status"] == "error"
    assert report["exit_code"] == ExitCode.RENDER_ERROR
