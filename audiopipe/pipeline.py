"""End-to-end mixing pipeline: per-clip processing, recovery and packaging."""

import csv
import hashlib
import json
import os
import threading
import concurrent.futures
import time

from . import wavio
from .checkpoint import CheckpointStore, parameter_fingerprint
from .errors import (
    WavFormatError,
    AudioPipeError,
    EXIT_FATAL_CLIP,
    EXIT_INTERRUPTED,
)
from .mixer import mix_clip

_SHUTDOWN = threading.Event()


def request_shutdown():
    _SHUTDOWN.set()


def is_shutdown():
    return _SHUTDOWN.is_set()


class SemaphoreLimiter:
    """Bound simultaneously open file descriptors and worker threads."""

    def __init__(self, max_open_files, max_workers):
        self.files = threading.BoundedSemaphore(max_open_files)
        self._max_open = max_open_files
        self._max_workers = max_workers

    @property
    def max_workers(self):
        return self._max_workers


class _LimitedFile:
    def __init__(self, limiter, path, mode="rb"):
        self._limiter = limiter
        self._path = path
        self._mode = mode
        self._f = None

    def __enter__(self):
        self._limiter.files.acquire()
        self._f = open(self._path, self._mode)
        return self._f

    def __exit__(self, *exc):
        try:
            self._f.close()
        finally:
            self._limiter.files.release()


def _read_events(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _hash_inputs(csv_path, rows, wav_dir):
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        h.update(f.read())
    for row in sorted(rows, key=lambda r: r.get("audio_file", "")):
        name = row.get("audio_file", "")
        if not name:
            h.update(b"missing:")
            h.update(str(row.get("missing", "")).encode())
            continue
        path = os.path.join(wav_dir, name)
        if os.path.exists(path):
            h.update(name.encode("utf-8"))
            h.update(wavio.file_sha256(path).encode("ascii"))
        else:
            h.update(b"absent:")
            h.update(name.encode("utf-8"))
    return h.hexdigest()


def _clip_ms(rows):
    end_ms = 0
    for row in rows:
        end_ms = max(end_ms, int(row["start_ms"]) + int(row["duration_ms"]))
    return end_ms


def process_clip(task, limiter):
    """Process one (language, clip). Raises on fatal structural errors."""
    (
        language,
        clip_id,
        csv_path,
        wav_dir,
        out_dir,
        target_rate,
        headroom_db,
        param_fp,
        store,
    ) = task
    token = "%s/%s" % (language, clip_id)
    rows = _read_events(csv_path)
    input_hash = _hash_inputs(csv_path, rows, wav_dir)

    if store.is_current(token, input_hash, param_fp):
        rec = store.get(token)
        return {
            "token": token,
            "reused": True,
            "fatal": rec.get("fatal", []),
            "recoverable": rec.get("recoverable", []),
            "outputs": rec.get("outputs", {}),
        }

    from .mixer import DegradationReport

    report = DegradationReport()
    clip_ms = _clip_ms(rows)
    pcm, event_meta, total_samples, clipped, limit = mix_clip(
        rows, wav_dir, target_rate, clip_ms, headroom_db, report
    )

    lang_dir = os.path.join(out_dir, language)
    os.makedirs(lang_dir, exist_ok=True)
    wav_out = os.path.join(lang_dir, "%s.wav" % clip_id)
    meta_out = os.path.join(lang_dir, "%s.metadata.json" % clip_id)

    with _LimitedFile(limiter, wav_out, "wb") as f:
        # Delegate to wavio through a small bounded writer (reuse its layout).
        _write_bounded(f, target_rate, pcm)

    metadata = {
        "clip_id": clip_id,
        "language": language,
        "target_sample_rate": target_rate,
        "total_samples": total_samples,
        "duration_ms": clip_ms,
        "peak_limit": limit,
        "clipped_samples": clipped,
        "events": event_meta,
        "degradation": report.as_dict(),
    }
    tmp_meta = meta_out + ".tmp"
    with _LimitedFile(limiter, tmp_meta, "w") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
        f.write("\n")
    os.replace(tmp_meta, meta_out)

    outputs = {
        "wav": os.path.relpath(wav_out, out_dir),
        "metadata": os.path.relpath(meta_out, out_dir),
        "wav_sha256": wavio.file_sha256(wav_out),
        "total_samples": total_samples,
    }
    store.record(token, input_hash, param_fp, outputs, report.as_dict())
    return {
        "token": token,
        "reused": False,
        "fatal": report.fatal,
        "recoverable": report.recoverable,
        "outputs": outputs,
    }


def _write_bounded(f, sample_rate, samples):
    import struct

    frame_bytes = len(samples) * 2
    f.write(b"RIFF")
    f.write(struct.pack("<I", 36 + frame_bytes))
    f.write(b"WAVE")
    f.write(b"fmt ")
    f.write(struct.pack("<I", 16))
    f.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
    f.write(b"data")
    f.write(struct.pack("<I", frame_bytes))
    f.write(struct.pack("<%dh" % len(samples), *samples))


def run_pipeline(
    material_dir,
    out_dir,
    target_rate,
    headroom_db=-1.0,
    max_workers=4,
    max_open_files=8,
    languages=None,
):
    events_dir = os.path.join(material_dir, "events")
    wav_dir = os.path.join(material_dir, "wav")
    ckpt_dir = os.path.join(out_dir, "_checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    store = CheckpointStore(ckpt_dir)
    limiter = SemaphoreLimiter(max_open_files, max_workers)
    params = {
        "target_rate": target_rate,
        "headroom_db": headroom_db,
        "mixer": "audiopipe.v1",
        "crossfade_ms": 5,
        "rounding": "round_half_up",
    }
    param_fp = parameter_fingerprint(params)

    csv_files = sorted(
        name for name in os.listdir(events_dir) if name.endswith("_events.csv")
    )
    tasks = []
    for name in csv_files:
        # filename: clipNNN_ll_events.csv
        base = name[: -len("_events.csv")]
        clip_id, language = base.split("_", 1)
        if languages and language not in languages:
            continue
        tasks.append(
            (
                language,
                clip_id,
                os.path.join(events_dir, name),
                wav_dir,
                out_dir,
                target_rate,
                headroom_db,
                param_fp,
                store,
            )
        )

    save_lock = threading.Lock()

    def _worker(task):
        if is_shutdown():
            return {"token": "%s/%s" % (task[0], task[1]), "skipped_shutdown": True}
        _maybe_slow_step()
        try:
            res = process_clip(task, limiter)
        except WavFormatError as exc:
            return {"token": "%s/%s" % (task[0], task[1]), "fatal_error": exc.to_record()}
        with save_lock:
            store.save_atomic()
        return res

    return execute_pipeline(tasks, max_workers, _worker, save_lock, store, out_dir)


def _maybe_slow_step():
    # Test-only deterministic pacing hook (no effect in normal runs).
    if os.environ.get("AUDIOPIPE_SLEEP_STEPS"):
        time.sleep(0.4)


def execute_pipeline(tasks, max_workers, worker, save_lock, store, out_dir):
    """Run workers, handle Ctrl+C, then package; returns (results, exit_code)."""
    results = {"completed": [], "fatal": [], "interrupted": False}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(worker, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res.get("skipped_shutdown"):
                    results["interrupted"] = True
                    continue
                if "fatal_error" in res:
                    results["fatal"].append(res)
                else:
                    results["completed"].append(res)
    except KeyboardInterrupt:
        request_shutdown()
        with save_lock:
            store.save_atomic()
        results["interrupted"] = True
        from .errors import emit_error, AudioPipeError as _APE
        emit_error(_APE("interrupted by user", code="E_INTERRUPTED",
                        exit_code=EXIT_INTERRUPTED))
        return results, EXIT_INTERRUPTED

    _write_language_reports(out_dir, results)
    results["manifest"] = write_manifest(out_dir)
    exit_code = EXIT_FATAL_CLIP if results["fatal"] else 0
    return results, exit_code


def _write_language_reports(out_dir, results):
    per_lang = {}
    for res in results["completed"]:
        language = res["token"].split("/", 1)[0]
        bucket = per_lang.setdefault(language, {"recoverable": [], "fatal": []})
        bucket["recoverable"].extend(res.get("recoverable", []))
        bucket["fatal"].extend(res.get("fatal", []))
    for res in results["fatal"]:
        language = res["token"].split("/", 1)[0]
        bucket = per_lang.setdefault(language, {"recoverable": [], "fatal": []})
        bucket["fatal"].append(res["fatal_error"])

    report_dir = os.path.join(out_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    for language, payload in sorted(per_lang.items()):
        # Order-independent, deterministic content regardless of scheduling.
        payload["recoverable"].sort(key=lambda r: json.dumps(r, sort_keys=True))
        payload["fatal"].sort(key=lambda r: json.dumps(r, sort_keys=True))
        path = os.path.join(report_dir, "%s.degradation.json" % language)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, sort_keys=True, indent=2)
            f.write("\n")
        os.replace(tmp, path)


def write_manifest(out_dir):
    """Hash every artifact (excluding checkpoints) and record duration."""
    entries = []
    for root, _dirs, files in os.walk(out_dir):
        if "_checkpoints" in os.path.relpath(root, out_dir).split(os.sep):
            continue
        for name in sorted(files):
            if name == "manifest.json":
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, out_dir)
            entries.append(
                {
                    "path": rel.replace(os.sep, "/"),
                    "sha256": wavio.file_sha256(path),
                    "bytes": os.path.getsize(path),
                }
            )
    entries.sort(key=lambda e: e["path"])
    manifest = {"artifacts": entries}
    payload = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {"artifacts": len(entries), "manifest_sha256": digest}
