"""Block scheduler: full bake, resume, local reconstruct, dry-run planning."""
import hashlib
import json
import os
import signal
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np

from . import heightfield as hfmod
from . import irradiance, state
from .errors import MemoryLimit, NonDeterministic, ReconstructMismatch

COST_PER_PROBE = 1.0e-3          # fixed coefficient; no wall clock is read
CHANNELS = 4
BYTES_PER_PROBE = CHANNELS * 4
THREADS_ENV = "PROBE_BAKE_THREADS"
DELAY_ENV = "PROBE_BAKE_BLOCK_DELAY_MS"   # test hook: slow each block down


def params_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def plan_blocks(probe_nx, probe_ny, block_size):
    blocks = []
    nbx = (probe_nx + block_size - 1) // block_size
    nby = (probe_ny + block_size - 1) // block_size
    for by in range(nby):
        for bx in range(nbx):
            ox = bx * block_size
            oy = by * block_size
            bw = min(block_size, probe_nx - ox)
            bh = min(block_size, probe_ny - oy)
            blocks.append((bx, by, ox, oy, bw, bh))
    return blocks, nbx, nby


def _block_bytes(block_size):
    return block_size * block_size * BYTES_PER_PROBE


def check_memory_budget(threads, total_blocks, block_size, max_cached_blocks):
    cached = min(threads, total_blocks) + 1
    if cached > max_cached_blocks:
        raise MemoryLimit(
            f"requires {cached} cached probe blocks "
            f"({_block_bytes(block_size) * cached} bytes) but budget is "
            f"{max_cached_blocks} blocks",
            required_blocks=cached,
            budget_blocks=max_cached_blocks,
        )
    return cached


def resolve_threads(cli_threads):
    env = os.environ.get(THREADS_ENV)
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, cli_threads)


class _StopFlag:
    def __init__(self):
        self.event = threading.Event()

    def install(self):
        def handler(signum, frame):
            self.event.set()

        signal.signal(signal.SIGINT, handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handler)


def _bake_one(hf, z_scale, spacing, block, light_dir):
    bx, by, ox, oy, bw, bh = block
    delay_ms = os.environ.get(DELAY_ENV)
    if delay_ms:
        threading.Event().wait(int(delay_ms) / 1000.0)
    data = irradiance.bake_block(hf, z_scale, spacing, ox, oy, bw, bh, light_dir)
    return bx, by, data


def _run_blocks(hf, params, blocks_todo, threads, stop, on_block_done):
    """Bake blocks with a worker pool; returns (completed, interrupted)."""
    completed = []
    if not blocks_todo:
        return completed, False
    pending = list(blocks_todo)
    in_flight = {}
    interrupted = False
    with ThreadPoolExecutor(max_workers=threads) as pool:
        while (pending or in_flight) and not interrupted:
            while pending and len(in_flight) < threads and not stop.event.is_set():
                block = pending.pop(0)
                fut = pool.submit(
                    _bake_one, hf, params["z_scale"], params["spacing"],
                    block, params["light_dir"],
                )
                in_flight[fut] = block
            if not in_flight:
                interrupted = True
                break
            done, _ = wait(in_flight, timeout=0.05,
                           return_when=FIRST_COMPLETED)
            for fut in done:
                bx, by, data = fut.result()
                in_flight.pop(fut)
                on_block_done(bx, by, data)
                completed.append((bx, by))
            if stop.event.is_set():
                interrupted = True
        # Drain whatever is still running so their state hits disk.
        for fut in in_flight:
            bx, by, data = fut.result()
            on_block_done(bx, by, data)
            completed.append((bx, by))
    return completed, interrupted


def _assemble(state_dir, version, probe_nx, probe_ny, block_size, blocks):
    out = np.zeros((probe_ny, probe_nx, CHANNELS), dtype=np.float32)
    for bx, by, ox, oy, bw, bh in blocks:
        data = state.read_block(state_dir, bx, by, version)
        if data is None:
            raise ReconstructMismatch(
                f"persisted block ({bx}, {by}) missing or stale during assembly"
            )
        out[oy:oy + bh, ox:ox + bw, :] = data
    return out


def _write_result(out_path, probes, version, params):
    arr = np.ascontiguousarray(probes, dtype="<f4")
    with open(out_path, "wb") as fh:
        fh.write(arr.tobytes())
    sidecar = {
        "format": "rgba-float32le",
        "width": int(probes.shape[1]),
        "height": int(probes.shape[0]),
        "channels": CHANNELS,
        "version_hash": version,
        "params_hash": params_hash(params),
    }
    with open(out_path + ".json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _write_log(log_path, record):
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _load_and_plan(heightfield_path, params):
    hf, version = hfmod.read_heightfield(heightfield_path)
    h, w = hf.shape
    probe_nx, probe_ny = irradiance.probe_grid_dims(w, h, params["spacing"])
    blocks, nbx, nby = plan_blocks(probe_nx, probe_ny, params["block_size"])
    return hf, version, probe_nx, probe_ny, blocks, nbx, nby


def dry_run(heightfield_path, params, threads, max_cached_blocks):
    """Plan only: no integration, no state writes."""
    _, version, probe_nx, probe_ny, blocks, nbx, nby = _load_and_plan(
        heightfield_path, params
    )
    cached = min(threads, len(blocks)) + 1
    report = {
        "mode": "dry-run",
        "version_hash": version,
        "probe_grid": [probe_nx, probe_ny],
        "block_grid": [nbx, nby],
        "block_size": params["block_size"],
        "total_blocks": len(blocks),
        "bytes_per_block": _block_bytes(params["block_size"]),
        "cache_blocks_required": cached,
        "estimated_memory_bytes": cached * _block_bytes(params["block_size"]),
        "within_budget": cached <= max_cached_blocks,
    }
    return report


def bake(heightfield_path, state_dir, out_path, log_path, params, threads,
         max_cached_blocks, verify_determinism=False):
    """Full/incremental bake with resume. Returns (log_record, exit_status)."""
    hf, version, probe_nx, probe_ny, blocks, nbx, nby = _load_and_plan(
        heightfield_path, params
    )
    check_memory_budget(threads, len(blocks), params["block_size"],
                        max_cached_blocks)
    os.makedirs(state_dir, exist_ok=True)
    phash = params_hash(params)

    manifest = state.read_manifest(state_dir)
    if manifest and (
        manifest.get("version_hash") != version
        or manifest.get("params_hash") != phash
    ):
        # Base heightfield (or bake parameters) changed: old block offsets
        # and counts are meaningless; drop everything and rebake fully.
        state.wipe_state(state_dir)
        manifest = None

    done_set = set()
    if manifest:
        for bx, by in manifest.get("blocks_completed", []):
            if state.read_block(state_dir, bx, by, version) is not None:
                done_set.add((bx, by))

    todo = [b for b in blocks if (b[0], b[1]) not in done_set]
    stop = _StopFlag()
    stop.install()
    probes_processed = 0

    def on_block_done(bx, by, data):
        nonlocal probes_processed
        state.write_block(state_dir, bx, by, {"version_hash": version}, data)
        done_set.add((bx, by))
        probes_processed += data.shape[0] * data.shape[1]
        state.write_manifest(state_dir, {
            "version_hash": version,
            "params_hash": phash,
            "probe_grid": [probe_nx, probe_ny],
            "block_size": params["block_size"],
            "blocks_total": len(blocks),
            "blocks_completed": sorted(done_set),
        })

    completed_now, interrupted = _run_blocks(
        hf, params, todo, threads, stop, on_block_done
    )

    record = {
        "mode": "bake",
        "version_hash": version,
        "params_hash": phash,
        "probe_grid": [probe_nx, probe_ny],
        "block_grid": [nbx, nby],
        "block_size": params["block_size"],
        "blocks_total": len(blocks),
        "blocks_completed": sorted(done_set),
        "blocks_baked_this_run": sorted(completed_now),
        "probes_processed": probes_processed,
        "estimated_cost": probes_processed * COST_PER_PROBE,
        "status": "interrupted" if interrupted else "complete",
    }

    if not interrupted:
        probes = _assemble(state_dir, version, probe_nx, probe_ny,
                           params["block_size"], blocks)
        if verify_determinism:
            reference = np.empty_like(probes)
            for bx, by, ox, oy, bw, bh in blocks:
                reference[oy:oy + bh, ox:ox + bw, :] = irradiance.bake_block(
                    hf, params["z_scale"], params["spacing"],
                    ox, oy, bw, bh, params["light_dir"],
                )
            if reference.tobytes() != probes.tobytes():
                raise NonDeterministic(
                    "two bakes of the same input produced different bytes"
                )
        _write_result(out_path, probes, version, params)
    _write_log(log_path, record)
    return record, interrupted


def reconstruct(heightfield_path, state_dir, out_path, log_path, params,
                threads, max_cached_blocks, edit_rect, tolerance):
    """Local rebuild: recompute only blocks intersecting the (dilated) edit
    rect, carry every other block over from persisted state, then verify the
    merged array against a full in-memory bake of the edited heightfield."""
    hf, version, probe_nx, probe_ny, blocks, nbx, nby = _load_and_plan(
        heightfield_path, params
    )
    check_memory_budget(threads, len(blocks), params["block_size"],
                        max_cached_blocks)
    os.makedirs(state_dir, exist_ok=True)
    phash = params_hash(params)

    manifest = state.read_manifest(state_dir)
    reusable = {}
    if manifest and manifest.get("params_hash") == phash:
        old_version = manifest.get("version_hash")
        for bx, by in manifest.get("blocks_completed", []):
            data = state.read_block(state_dir, bx, by, old_version)
            if data is not None:
                reusable[(bx, by)] = data

    affected = set(
        irradiance.affected_blocks(
            edit_rect, params["spacing"], params["block_size"],
            probe_nx, probe_ny,
        )
    )
    todo = [b for b in blocks if (b[0], b[1]) in affected]
    carried = {k: v for k, v in reusable.items() if k not in affected}

    stop = _StopFlag()
    stop.install()
    probes_processed = 0
    done_set = set(carried)

    def on_block_done(bx, by, data):
        nonlocal probes_processed
        state.write_block(state_dir, bx, by, {"version_hash": version}, data)
        done_set.add((bx, by))
        probes_processed += data.shape[0] * data.shape[1]

    # Persist carried blocks under the new version (byte-identical copies).
    for (bx, by), data in sorted(carried.items()):
        state.write_block(state_dir, bx, by, {"version_hash": version}, data)

    completed_now, interrupted = _run_blocks(
        hf, params, todo, threads, stop, on_block_done
    )
    state.write_manifest(state_dir, {
        "version_hash": version,
        "params_hash": phash,
        "probe_grid": [probe_nx, probe_ny],
        "block_size": params["block_size"],
        "blocks_total": len(blocks),
        "blocks_completed": sorted(done_set),
    })

    record = {
        "mode": "reconstruct",
        "version_hash": version,
        "params_hash": phash,
        "probe_grid": [probe_nx, probe_ny],
        "block_grid": [nbx, nby],
        "block_size": params["block_size"],
        "blocks_total": len(blocks),
        "blocks_completed": sorted(done_set),
        "blocks_rebuilt_this_run": sorted(completed_now),
        "blocks_carried_over": sorted(carried),
        "edit_rect": list(edit_rect),
        "probes_processed": probes_processed,
        "estimated_cost": probes_processed * COST_PER_PROBE,
        "status": "interrupted" if interrupted else "complete",
    }

    if not interrupted:
        probes = _assemble(state_dir, version, probe_nx, probe_ny,
                           params["block_size"], blocks)
        # Verify against a full in-memory bake of the edited heightfield.
        bad = []
        for bx, by, ox, oy, bw, bh in blocks:
            ref = irradiance.bake_block(
                hf, params["z_scale"], params["spacing"],
                ox, oy, bw, bh, params["light_dir"],
            )
            got = probes[oy:oy + bh, ox:ox + bw, :]
            if not np.allclose(got, ref, atol=tolerance, rtol=0.0):
                bad.append([bx, by])
        if bad:
            raise ReconstructMismatch(
                "local rebuild diverges from full bake beyond tolerance",
                tolerance=tolerance,
                mismatched_blocks=bad,
            )
        _write_result(out_path, probes, version, params)
    _write_log(log_path, record)
    return record, interrupted


def compare_logs(path_a, path_b):
    with open(path_a, encoding="utf-8") as fh:
        a = json.load(fh)
    with open(path_b, encoding="utf-8") as fh:
        b = json.load(fh)
    if a.get("version_hash") != b.get("version_hash"):
        return {
            "result": "full_rebake_required",
            "reason": "heightfield version hash changed "
                      f"({a.get('version_hash')} -> {b.get('version_hash')}); "
                      "persisted block offsets are invalid",
        }
    set_a = {tuple(b_) for b_ in a.get("blocks_completed", [])}
    set_b = {tuple(b_) for b_ in b.get("blocks_completed", [])}
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    report = {"result": "same_version", "version_hash": a.get("version_hash")}
    if only_a or only_b:
        report["blocks_only_in_first"] = only_a
        report["blocks_only_in_second"] = only_b
    else:
        report["blocks"] = "identical"
    return report
