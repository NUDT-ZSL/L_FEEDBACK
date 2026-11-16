import json

from dub_replace.state import (
    JobState,
    STATE_VERSION,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCESS,
)


def test_round_trip_and_success_guard(tmp_path):
    state = JobState.load(tmp_path)
    assert state.is_success("a") is False
    state.mark("a", STATUS_RUNNING)
    state.record_success("a", "r.json", "o.wav")
    reloaded = JobState.load(tmp_path)
    assert reloaded.is_success("a") is True
    assert reloaded.get("a")["status"] == STATUS_SUCCESS
    assert (tmp_path / "job_state.json").is_file()


def test_failure_recorded_and_retried(tmp_path):
    state = JobState.load(tmp_path)
    state.mark("b", STATUS_RUNNING)
    state.record_failure("b", {"code": "E_CORRUPT", "message": "x", "hint": "y"})
    reloaded = JobState.load(tmp_path)
    assert reloaded.is_success("b") is False
    assert reloaded.get("b")["status"] == STATUS_FAILED
    assert reloaded.get("b")["error"]["code"] == "E_CORRUPT"


def test_stale_version_is_discarded(tmp_path):
    stale = {"version": STATE_VERSION - 1, "animations": {"old": {"status": STATUS_SUCCESS}}}
    (tmp_path / "job_state.json").write_text(json.dumps(stale), encoding="utf-8")
    state = JobState.load(tmp_path)
    assert state.animations == {}
    state.record_success("new", "r.json", "o.wav")
    saved = json.loads((tmp_path / "job_state.json").read_text(encoding="utf-8"))
    assert saved["version"] == STATE_VERSION
    assert "old" not in saved["animations"]


def test_corrupt_state_file_starts_clean(tmp_path):
    (tmp_path / "job_state.json").write_text("{partial", encoding="utf-8")
    state = JobState.load(tmp_path)
    assert state.animations == {}
