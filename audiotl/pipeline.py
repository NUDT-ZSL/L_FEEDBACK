"""Four-stage deterministic render pipeline.

Stages: analyze -> resample -> mix -> writeout.

Each stage emits an intermediate artifact carrying a stage version.
Downstream stages verify the version and input hash of their upstream
artifacts and recompute on any mismatch, so a version bump invalidates
derived data instead of silently reusing stale offsets or statistics.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import cache as cachemod
from .cache import StateStore, atomic_write, atomic_write_json, canonical_json, sha256_bytes
from .dsp import apply_fades, convert_channels, ms_to_frames, resample_length, resample_linear
from .errors import BusinessError, ErrorCode, ExitCode
from .manifest import load_manifest
from .mapping import build_mapping, compute_positions
from .summary import compute_summary
from .wavio import quantize_int16, read_wav, write_wav16_atomic

ENGINE_VERSION = 1
STAGE_VERSIONS = {"analyze": 1, "resample": 1, "mix": 1, "writeout": 1}

FINAL_OUTPUTS = ("mix.wav", "mapping.json", "summary.json")


def _max_workers(requested):
    cores = os.cpu_count() or 1
    if requested is None:
        return cores
    return max(1, min(int(requested), cores))


def _read_json(path):
    try:
        with open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


class Pipeline:
    def __init__(self, manifest_path, out_dir, workers=None, state_dir=None):
        self.manifest_path = manifest_path
        self.out_dir = os.path.abspath(out_dir)
        self.stages_dir = os.path.join(self.out_dir, "_stages")
        self.state_dir = state_dir or os.path.join(self.out_dir, "_state")
        self.workers = _max_workers(workers)
        self.report = {
            "status": "error",
            "exit_code": ExitCode.INTERNAL,
            "errors": [],
            "manifest": os.path.abspath(manifest_path),
            "output_dir": self.out_dir,
            "state_dir": os.path.abspath(self.state_dir),
            "workers": self.workers,
            "engine_version": ENGINE_VERSION,
            "stage_versions": dict(STAGE_VERSIONS),
            "stages": {},
            "cache": {"hits": 0, "misses": 0},
            "durations_ms": {},
        }

    # -- stage 1: analyze -------------------------------------------------

    def stage_analyze(self, manifest):
        analyzed = []
        for seg in manifest.segments:
            if not os.path.isfile(seg.path):
                raise BusinessError(
                    ErrorCode.E_INPUT_MISSING,
                    f"segment '{seg.id}': input file not found: {seg.path}",
                    ExitCode.INPUT_MISSING,
                    {"segment": seg.id, "path": seg.path},
                )
            src_sha = cachemod.hash_file(seg.path)
            data, in_rate = read_wav(seg.path)
            in_frames = int(data.shape[0])
            out_frames = resample_length(in_frames, in_rate, manifest.sample_rate)
            fade_in_frames = ms_to_frames(seg.fade_in.length_ms, manifest.sample_rate)
            fade_out_frames = ms_to_frames(seg.fade_out.length_ms, manifest.sample_rate)
            for which, frames in (("fade_in", fade_in_frames), ("fade_out", fade_out_frames)):
                if frames > out_frames:
                    raise BusinessError(
                        ErrorCode.E_FADE_TOO_LONG,
                        f"segment '{seg.id}': {which} ({frames} frames) exceeds "
                        f"segment length ({out_frames} frames)",
                        ExitCode.RENDER_ERROR,
                        {"segment": seg.id, "fade": which,
                         "fade_frames": frames, "segment_frames": out_frames},
                    )
            analyzed.append(
                {
                    "id": seg.id,
                    "path": seg.path,
                    "src_sha256": src_sha,
                    "in_rate": in_rate,
                    "in_channels": int(data.shape[1]),
                    "in_frames": in_frames,
                    "out_frames": out_frames,
                    "fade_in_frames": fade_in_frames,
                    "fade_out_frames": fade_out_frames,
                    "fade_in_curve": seg.fade_in.curve,
                    "fade_out_curve": seg.fade_out.curve,
                }
            )
        for i in range(len(analyzed) - 1):
            overlap = analyzed[i]["fade_out_frames"]
            neighbor = analyzed[i + 1]["out_frames"]
            if overlap > neighbor:
                raise BusinessError(
                    ErrorCode.E_OVERLAP_EXCESSIVE,
                    f"crossfade overlap after segment '{analyzed[i]['id']}' "
                    f"({overlap} frames) exceeds next segment "
                    f"'{analyzed[i + 1]['id']}' length ({neighbor} frames)",
                    ExitCode.RENDER_ERROR,
                    {"segment": analyzed[i]["id"], "overlap_frames": overlap,
                     "next_segment": analyzed[i + 1]["id"], "next_frames": neighbor},
                )
        artifact = {
            "stage": "analyze",
            "version": STAGE_VERSIONS["analyze"],
            "engine_version": ENGINE_VERSION,
            "manifest_sha256": manifest.sha256,
            "output": {"sample_rate": manifest.sample_rate, "channels": manifest.channels},
            "segments": analyzed,
        }
        atomic_write_json(os.path.join(self.stages_dir, "analysis.json"), artifact)
        return analyzed

    # -- stage 2: resample (cached, parallel) ------------------------------

    def _segment_key(self, manifest, info):
        params = {
            "src_sha256": info["src_sha256"],
            "out_rate": manifest.sample_rate,
            "out_channels": manifest.channels,
            "fade_in": [info["fade_in_frames"], info["fade_in_curve"]],
            "fade_out": [info["fade_out_frames"], info["fade_out_curve"]],
            "versions": {
                "engine": ENGINE_VERSION,
                "analyze": STAGE_VERSIONS["analyze"],
                "resample": STAGE_VERSIONS["resample"],
            },
        }
        return sha256_bytes(canonical_json(params))

    def _render_segment(self, manifest, info):
        data, in_rate = read_wav(info["path"])
        data = resample_linear(data, in_rate, manifest.sample_rate)
        data = convert_channels(data, manifest.channels)
        data = apply_fades(
            data,
            info["fade_in_frames"], info["fade_in_curve"],
            info["fade_out_frames"], info["fade_out_curve"],
        )
        return np.ascontiguousarray(data, dtype=np.float32)

    def _cache_lookup(self, state, key, info, manifest):
        entry = state.get_entry(key)
        if not entry:
            return None
        seg_path = state.segment_path(key)
        meta = _read_json(state.segment_meta_path(key))
        expected_frames = info["out_frames"]
        if not meta or meta.get("key") != key:
            return None
        if meta.get("frames") != expected_frames or meta.get("channels") != manifest.channels:
            return None
        try:
            raw = np.fromfile(seg_path, dtype="<f4")
        except OSError:
            return None
        if raw.size != expected_frames * manifest.channels:
            return None
        return raw.reshape(expected_frames, manifest.channels)

    def stage_resample(self, manifest, analyzed):
        state = StateStore(self.state_dir)
        keys = [self._segment_key(manifest, info) for info in analyzed]
        results = [None] * len(analyzed)
        to_render = []
        for index, (info, key) in enumerate(zip(analyzed, keys)):
            cached = self._cache_lookup(state, key, info, manifest)
            if cached is not None:
                results[index] = cached
                self.report["cache"]["hits"] += 1
            else:
                to_render.append(index)
                self.report["cache"]["misses"] += 1

        if to_render:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {
                    pool.submit(self._render_segment, manifest, analyzed[i]): i
                    for i in to_render
                }
                for future in futures:
                    i = futures[future]
                    results[i] = future.result()

            new_entries = {}
            for i in to_render:
                key = keys[i]
                data = results[i]
                atomic_write(state.segment_path(key), data.tobytes())
                atomic_write_json(
                    state.segment_meta_path(key),
                    {
                        "key": key,
                        "segment": analyzed[i]["id"],
                        "frames": int(data.shape[0]),
                        "channels": int(data.shape[1]),
                        "sample_rate": manifest.sample_rate,
                        "versions": {
                            "engine": ENGINE_VERSION,
                            "analyze": STAGE_VERSIONS["analyze"],
                            "resample": STAGE_VERSIONS["resample"],
                        },
                    },
                )
                new_entries[key] = {
                    "segment": analyzed[i]["id"],
                    "frames": int(data.shape[0]),
                    "channels": int(data.shape[1]),
                }
            state.update_entries(new_entries)

        artifact = {
            "stage": "resample",
            "version": STAGE_VERSIONS["resample"],
            "engine_version": ENGINE_VERSION,
            "manifest_sha256": manifest.sha256,
            "segment_keys": keys,
        }
        atomic_write_json(os.path.join(self.stages_dir, "resample.json"), artifact)
        return keys, results

    # -- stage 3: mix -------------------------------------------------------

    def stage_mix(self, manifest, analyzed, keys, starts, total_frames):
        meta_path = os.path.join(self.stages_dir, "mix.meta.json")
        data_path = os.path.join(self.stages_dir, "mix.f32")
        inputs_hash = sha256_bytes(
            canonical_json(
                {
                    "keys": keys,
                    "starts": starts,
                    "total_frames": total_frames,
                    "sample_rate": manifest.sample_rate,
                    "channels": manifest.channels,
                    "versions": {
                        "engine": ENGINE_VERSION,
                        "resample": STAGE_VERSIONS["resample"],
                        "mix": STAGE_VERSIONS["mix"],
                    },
                }
            )
        )
        meta = _read_json(meta_path)
        expected_size = total_frames * manifest.channels * 4
        if (
            meta
            and meta.get("version") == STAGE_VERSIONS["mix"]
            and meta.get("inputs_sha256") == inputs_hash
            and os.path.isfile(data_path)
            and os.path.getsize(data_path) == expected_size
        ):
            self.report["stages"]["mix"] = {"reused": True}
            return data_path

        state = StateStore(self.state_dir)
        buf = np.zeros((total_frames, manifest.channels), dtype=np.float64)
        for index, info in enumerate(analyzed):
            raw = np.fromfile(state.segment_path(keys[index]), dtype="<f4")
            seg = raw.reshape(info["out_frames"], manifest.channels)
            start = starts[index]
            buf[start:start + info["out_frames"]] += seg
        atomic_write(data_path, buf.astype(np.float32).tobytes())
        atomic_write_json(
            meta_path,
            {
                "stage": "mix",
                "version": STAGE_VERSIONS["mix"],
                "engine_version": ENGINE_VERSION,
                "inputs_sha256": inputs_hash,
                "total_frames": total_frames,
                "channels": manifest.channels,
                "sample_rate": manifest.sample_rate,
            },
        )
        self.report["stages"]["mix"] = {"reused": False}
        return data_path

    # -- stage 4: writeout ---------------------------------------------------

    def stage_writeout(self, manifest, analyzed, starts, total_frames, mix_path):
        raw = np.fromfile(mix_path, dtype="<f4")
        mix = raw.reshape(total_frames, manifest.channels)
        q = quantize_int16(mix)

        mix_wav_path = os.path.join(self.out_dir, "mix.wav")
        write_wav16_atomic(mix_wav_path, mix, manifest.sample_rate)

        mapping = build_mapping(manifest, analyzed, starts, total_frames)
        atomic_write_json(os.path.join(self.out_dir, "mapping.json"), mapping)

        window_frames = max(1, ms_to_frames(manifest.window_ms, manifest.sample_rate))
        summary = compute_summary(
            q, manifest.sample_rate, manifest.channels, window_frames, manifest.window_ms
        )
        atomic_write_json(os.path.join(self.out_dir, "summary.json"), summary)

        return {
            "mix_wav": mix_wav_path,
            "mapping": os.path.join(self.out_dir, "mapping.json"),
            "summary": os.path.join(self.out_dir, "summary.json"),
        }

    # -- orchestration -------------------------------------------------------

    def run(self):
        started = time.perf_counter()
        outputs = None
        exit_code = ExitCode.OK
        try:
            os.makedirs(self.stages_dir, exist_ok=True)
            manifest = load_manifest(self.manifest_path)
            self.report["manifest_sha256"] = manifest.sha256

            t0 = time.perf_counter()
            analyzed = self.stage_analyze(manifest)
            self.report["durations_ms"]["analyze"] = _elapsed_ms(t0)

            t0 = time.perf_counter()
            keys, _segments = self.stage_resample(manifest, analyzed)
            self.report["durations_ms"]["resample"] = _elapsed_ms(t0)

            starts, total_frames = compute_positions(
                [a["out_frames"] for a in analyzed],
                [a["fade_out_frames"] for a in analyzed],
            )

            t0 = time.perf_counter()
            mix_path = self.stage_mix(manifest, analyzed, keys, starts, total_frames)
            self.report["durations_ms"]["mix"] = _elapsed_ms(t0)

            t0 = time.perf_counter()
            outputs = self.stage_writeout(manifest, analyzed, starts, total_frames, mix_path)
            self.report["durations_ms"]["writeout"] = _elapsed_ms(t0)

            self.report["status"] = "ok"
            self.report["exit_code"] = ExitCode.OK
            self.report["outputs"] = outputs
            self.report["total_frames"] = total_frames
        except BusinessError as exc:
            exit_code = exc.exit_code
            self.report["status"] = "error"
            self.report["exit_code"] = exit_code
            self.report["errors"].append(exc.to_dict())
        except Exception as exc:  # unexpected: still emit a JSON report
            exit_code = ExitCode.INTERNAL
            self.report["status"] = "error"
            self.report["exit_code"] = exit_code
            self.report["errors"].append(
                {
                    "code": ErrorCode.E_INTERNAL,
                    "message": f"{type(exc).__name__}: {exc}",
                    "details": {},
                }
            )
        self.report["durations_ms"]["total"] = _elapsed_ms(started)
        # The report is the authoritative run result; it is always written
        # atomically, and final outputs are only published on success.
        atomic_write_json(os.path.join(self.out_dir, "report.json"), self.report)
        return self.report, exit_code


def _elapsed_ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 3)
