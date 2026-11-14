"""Resumable chunked export, snapshot handling and WAV/manifest packaging."""

import json
import os
import struct
import tempfile
import wave

from . import ENGINE_VERSION
from .errors import (
    EXIT_INTERRUPTED,
    EXIT_OK,
    E_IO,
    E_RESUME,
)
from .renderer import RenderEngine

SNAPSHOT_NAME = "snapshot.json"


def atomic_write_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def chunk_path(work_dir, index):
    return os.path.join(work_dir, f"chunk_{index:08d}.pcm")


def _read_snapshot(work_dir):
    path = os.path.join(work_dir, SNAPSHOT_NAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _verify_chunk(path, expected_size):
    try:
        return os.path.getsize(path) == expected_size
    except OSError:
        return False


def _validate_resume(snapshot, project, frame_size):
    if not isinstance(snapshot, dict):
        return 0, [], None
    if snapshot.get("project_hash") != project.canonical_hash:
        raise ResumeRejected("project hash mismatch; restarting export from zero")
    if snapshot.get("engine_version") != ENGINE_VERSION:
        raise ResumeRejected("engine version mismatch; restarting export from zero")
    if int(snapshot.get("sample_rate", -1)) != project.sample_rate:
        raise ResumeRejected("sample rate mismatch; restarting export from zero")
    if int(snapshot.get("channels", -1)) != (2 if project.raw.get("channel_layout") == "stereo" else 1):
        raise ResumeRejected("channel layout mismatch; restarting export from zero")

    samples_done = int(snapshot.get("samples_done", 0))
    chunk_list = snapshot.get("chunks", [])
    valid_chunks = []
    verified_samples = 0
    for entry in chunk_list:
        index, size = int(entry[0]), int(entry[1])
        expected_samples = size // frame_size
        path = chunk_path(project.export.work_dir, index)
        if not _verify_chunk(path, size):
            break
        valid_chunks.append((index, size, expected_samples))
        verified_samples += expected_samples
    while verified_samples > samples_done and valid_chunks:
        entry = valid_chunks.pop()
        orphan = chunk_path(project.export.work_dir, entry[0])
        try:
            os.unlink(orphan)
        except OSError:
            pass
        verified_samples -= entry["samples"]
    return min(samples_done, verified_samples), valid_chunks, snapshot.get("stats")


class ResumeRejected(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _reset_work_dir(work_dir):
    if os.path.isdir(work_dir):
        for name in os.listdir(work_dir):
            if name.startswith("chunk_") or name == SNAPSHOT_NAME or name.startswith(".tmp-"):
                try:
                    os.unlink(os.path.join(work_dir, name))
                except OSError:
                    pass
    else:
        os.makedirs(work_dir, exist_ok=True)


def export_project(project, timeline, interrupt_flag=None, logger=print):
    options = project.export
    os.makedirs(options.work_dir, exist_ok=True)
    stereo = project.raw.get("channel_layout") == "stereo"
    frame_size = 4 if stereo else 2

    resume_note = None
    snapshot = _read_snapshot(options.work_dir)
    if snapshot is not None:
        try:
            samples_done, valid_chunks, saved_stats = _validate_resume(
                snapshot, project, frame_size
            )
        except ResumeRejected as exc:
            logger(f"[resume] {exc.reason}")
            _reset_work_dir(options.work_dir)
            samples_done, valid_chunks, saved_stats = 0, [], None
            resume_note = exc.reason
        if samples_done > 0:
            logger(f"[resume] continuing from {samples_done} persisted samples")
    else:
        samples_done, valid_chunks, saved_stats = 0, [], None

    engine = RenderEngine(project, timeline, stereo=stereo)
    if samples_done > 0:
        engine.seek_to_sample(samples_done)
    if saved_stats:
        engine.stats.update(
            {
                "peak_premix": float(saved_stats.get("peak_premix", 0.0)),
                "peak_pre_soft": float(saved_stats.get("peak_pre_soft", 0.0)),
                "peak_post_soft": float(saved_stats.get("peak_post_soft", 0.0)),
                "clipped_frames": int(saved_stats.get("clipped_frames", 0)),
                "clipped_samples": int(saved_stats.get("clipped_samples", 0)),
            }
        )

    total_samples = engine.total_samples
    chunk_size = options.chunk_samples
    chunk_entries = list(valid_chunks)
    interrupted = False

    while samples_done < total_samples:
        chunk_start = samples_done
        chunk_end = min(total_samples, chunk_start + chunk_size)
        pcm = engine.render_chunk(chunk_start, chunk_end)
        index = len(chunk_entries)
        path = chunk_path(options.work_dir, index)
        _durable_write(path, pcm)
        chunk_entries.append(
            (index, len(pcm), chunk_end - chunk_start)
        )
        samples_done = chunk_end
        snapshot_payload = {
            "engine_version": ENGINE_VERSION,
            "project_hash": project.canonical_hash,
            "sample_rate": project.sample_rate,
            "channels": 2 if stereo else 1,
            "samples_done": samples_done,
            "total_samples": total_samples,
            "frame_size": frame_size,
            "chunks": [[entry[0], entry[1]] for entry in chunk_entries],
            "stats": engine.stats,
        }
        atomic_write_json(os.path.join(options.work_dir, SNAPSHOT_NAME), snapshot_payload)
        logger(
            f"[render] chunk {index} persisted; {samples_done}/{total_samples} samples"
        )
        if interrupt_flag is not None and interrupt_flag():
            interrupted = True
            break

    if interrupted:
        return EXIT_INTERRUPTED, _incomplete_manifest(project, engine, samples_done, total_samples, interrupted=True)

    _package_wav(project, chunk_entries, stereo)
    manifest = _build_manifest(project, engine, timeline, len(chunk_entries), resume_note)
    atomic_write_json(options.manifest_path, manifest)
    return EXIT_OK, manifest


def _durable_write(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-chunk-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _package_wav(project, chunk_entries, stereo):
    options = project.export
    channels = 2 if stereo else 1
    directory = os.path.dirname(os.path.abspath(options.output_wav))
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-wav-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as raw_handle:
            with wave.open(raw_handle, "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(project.sample_rate)
                for entry in chunk_entries:
                    with open(chunk_path(options.work_dir, entry[0]), "rb") as chunk_handle:
                        while True:
                            block = chunk_handle.read(1 << 20)
                            if not block:
                                break
                            wav_file.writeframesraw(block)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(tmp_path, options.output_wav)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _build_manifest(project, engine, timeline, chunk_count, resume_note):
    options = project.export
    extension_samples = engine.outro_extension()
    return {
        "status": "complete",
        "engine_version": ENGINE_VERSION,
        "project_hash": project.canonical_hash,
        "sample_rate": project.sample_rate,
        "channels": 2 if project.raw.get("channel_layout") == "stereo" else 1,
        "total_samples": engine.total_samples,
        "duration_ms": round(engine.total_samples * 1000.0 / project.sample_rate, 3),
        "chunk_samples": options.chunk_samples,
        "chunk_count": chunk_count,
        "channel_gains_db": {
            cid: channel.gain_db for cid, channel in project.channels.items()
        },
        "muted_channels": [
            cid for cid, channel in project.channels.items() if channel.muted
        ],
        "peaks": {
            "premix": engine.stats["peak_premix"],
            "pre_soft": engine.stats["peak_pre_soft"],
            "post_soft": engine.stats["peak_post_soft"],
        },
        "clipped_samples": engine.stats["clipped_samples"],
        "clipped_frames": engine.stats["clipped_frames"],
        "outro_extension": {
            "samples": extension_samples,
            "ms": round(extension_samples * 1000.0 / project.sample_rate, 3),
        },
        "errors": [],
        "resume": {"rejected": resume_note is not None, "reason": resume_note},
    }


def _incomplete_manifest(project, engine, samples_done, total_samples, interrupted):
    payload = {
        "status": "interrupted" if interrupted else "incomplete",
        "engine_version": ENGINE_VERSION,
        "project_hash": project.canonical_hash,
        "sample_rate": project.sample_rate,
        "samples_done": samples_done,
        "total_samples": total_samples,
        "errors": [
            {
                "code": E_IO if not interrupted else E_RESUME,
                "message": "export interrupted; resume on next run",
            }
        ],
    }
    try:
        atomic_write_json(project.export.manifest_path, payload)
    except OSError:
        pass
    return payload
