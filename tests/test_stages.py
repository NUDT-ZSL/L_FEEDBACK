import json

from conftest import mix_bytes, read_report, run_cli


def test_mix_stage_version_mismatch_forces_recompute(sample_set, tmp_path):
    out = tmp_path / "out"
    assert run_cli(sample_set["manifest_path"], out) == 0
    baseline = mix_bytes(out)
    assert read_report(out)["stages"]["mix"]["reused"] is False

    # second run reuses the mix artifact
    assert run_cli(sample_set["manifest_path"], out) == 0
    assert read_report(out)["stages"]["mix"]["reused"] is True

    # simulate a stage version bump recorded in the artifact
    meta_path = out / "_stages" / "mix.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["version"] = meta["version"] + 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    assert run_cli(sample_set["manifest_path"], out) == 0
    report = read_report(out)
    assert report["stages"]["mix"]["reused"] is False
    assert mix_bytes(out) == baseline  # recompute is deterministic


def test_analysis_artifact_carries_stage_version(sample_set, tmp_path):
    out = tmp_path / "out"
    assert run_cli(sample_set["manifest_path"], out) == 0
    analysis = json.loads((out / "_stages" / "analysis.json").read_text())
    resample = json.loads((out / "_stages" / "resample.json").read_text())
    assert analysis["stage"] == "analyze"
    assert resample["stage"] == "resample"
    assert analysis["version"] >= 1
    assert len(resample["segment_keys"]) == 3
    assert all(len(k) == 64 for k in resample["segment_keys"])  # sha256 hex
