#!/usr/bin/env python3
"""Deterministic acceptance checks for the offline canyon renderer."""

import copy
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


def fail(message):
    raise AssertionError(message)


def run(renderer, scene, out_dir, mode):
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(renderer), "--scene", str(scene), "--out", str(out_dir), "--mode", mode],
        capture_output=True,
        text=True,
        timeout=180,
    )
    stats_path = out_dir / "stats.json"
    if not stats_path.exists():
        fail(f"missing stats.json after renderer exit {proc.returncode}: {proc.stdout} {proc.stderr}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return proc, stats


def write_variant(source, target, mutate, nonfinite=False):
    data = copy.deepcopy(source)
    mutate(data)
    text = json.dumps(data, indent=2)
    if nonfinite:
        text = text.replace('"__NONFINITE__"', "1e309")
    target.write_text(text + "\n", encoding="utf-8")


def check_ok(proc, stats, label):
    if proc.returncode != 0:
        fail(f"{label}: expected exit 0, got {proc.returncode}: {proc.stdout} {proc.stderr}")
    if stats.get("status") != "ok" or stats.get("error") is not None:
        fail(f"{label}: status is not ok: {stats.get('error')}")


def check_error(proc, stats, code, label):
    if proc.returncode != 2:
        fail(f"{label}: expected exit 2, got {proc.returncode}: {proc.stdout} {proc.stderr}")
    error = stats.get("error") or {}
    if error.get("code") != code:
        fail(f"{label}: expected {code}, got {error}")


def tile_hash_map(stats):
    return {entry["tile"]: entry["hash"] for entry in stats["tileHashes"]}


def main():
    if len(sys.argv) != 4:
        print("usage: acceptance.py <renderer> <scenes-dir> <work-dir>", file=sys.stderr)
        return 2
    renderer = Path(sys.argv[1]).resolve()
    scenes_dir = Path(sys.argv[2]).resolve()
    work_dir = Path(sys.argv[3]).resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    generated = work_dir / "generated_scenes"
    generated.mkdir()

    base_path = scenes_dir / "canyon_demo.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))

    full_proc, full = run(renderer, base_path, work_dir / "full", "full")
    inc_proc, inc = run(renderer, base_path, work_dir / "incremental", "incremental")
    check_ok(full_proc, full, "full render")
    check_ok(inc_proc, inc, "incremental render")

    if full["frameHash"] != inc["frameHash"]:
        fail("full and incremental frame hashes differ")
    if full["depthHash"] != inc["depthHash"]:
        fail("full and incremental depth hashes differ")
    if inc["fullReferenceFrameHash"] != full["frameHash"]:
        fail("incremental reference color hash does not match full render")
    if inc["fullReferenceDepthHash"] != full["depthHash"]:
        fail("incremental reference depth hash does not match full render")
    verification = inc["incrementalVerification"]
    if not (
        verification["divergenceChecked"]
        and verification["framebufferHashesEqual"]
        and verification["tileHashesEqual"]
        and verification["unaffectedTileHashMismatches"] == 0
    ):
        fail(f"incremental verification failed: {verification}")

    if (work_dir / "full" / "color.ppm").read_bytes() != (
        work_dir / "incremental" / "color.ppm"
    ).read_bytes():
        fail("color.ppm differs between full and incremental")
    if (work_dir / "full" / "depth.ppm").read_bytes() != (
        work_dir / "incremental" / "depth.ppm"
    ).read_bytes():
        fail("depth.ppm differs between full and incremental")
    if (work_dir / "full" / "depth_raw.bin").read_bytes() != (
        work_dir / "incremental" / "depth_raw.bin"
    ).read_bytes():
        fail("raw 24-bit depth differs between full and incremental")

    if full["depthBits"] != 24 or full["depthMax"] != 16777215:
        fail("expected a configurable 24-bit fixed-point depth buffer")
    if full["nonBackgroundPixels"] < 10000 or full["distinctDepthValues"] < 1000:
        fail("depth buffer does not contain enough distinguishable geometry samples")
    if full["shadowVolumeTriangles"] <= 0:
        fail("shadow volume triangle count must be positive")
    if full["stencilOverflowCount"] != 0:
        fail("valid scene unexpectedly overflowed the template buffer")
    if full["shadowMaskedBackgroundPixels"] != 0:
        fail("shadow mask leaked outside scene geometry")

    raw_depth = (work_dir / "full" / "depth_raw.bin").read_bytes()
    values = {
        int.from_bytes(raw_depth[i : i + 4], "little")
        for i in range(0, len(raw_depth), 4)
    }
    if len(values) < 1000:
        fail("raw depth data has too few distinct fixed-point values")

    affected = inc["affectedTiles"]
    if affected != sorted(set(affected)):
        fail("affected tile sequence is not unique row-major order")
    if not inc["changeLog"]:
        fail("incremental change log is empty")
    for entry in inc["changeLog"]:
        if entry["affectedTiles"] != sorted(set(entry["affectedTiles"])):
            fail(f"change {entry['index']} affected tiles are not sorted row-major")

    inc_map = tile_hash_map(inc)
    full_map = tile_hash_map(full)
    for tile, digest in full_map.items():
        if tile not in inc_map or inc_map[tile] != digest:
            fail(f"tile {tile} hash differs from the full reference")

    # Repeat the incremental run and compare every deterministic field, excluding timings.
    inc2_proc, inc2 = run(renderer, base_path, work_dir / "incremental_repeat", "incremental")
    check_ok(inc2_proc, inc2, "incremental repeat")
    for field in ("frameHash", "depthHash", "combinedHash", "affectedTiles", "tileHashes", "changeLog"):
        if inc[field] != inc2[field]:
            fail(f"non-deterministic incremental field: {field}")

    # Parent rotation must move descendants in world space.
    no_change_path = generated / "no_change.json"
    parent_change_path = generated / "parent_rotation.json"
    no_change = copy.deepcopy(base)
    no_change["changes"] = []
    no_change_path.write_text(json.dumps(no_change, indent=2) + "\n", encoding="utf-8")
    parent_change = copy.deepcopy(no_change)
    parent_change["changes"] = [
        {
            "target": "canyon_spire",
            "kind": "transform",
            "transform": {"rotationDeg": [8.0, 85.0, 4.0]},
        }
    ]
    parent_change_path.write_text(json.dumps(parent_change, indent=2) + "\n", encoding="utf-8")
    _, baseline = run(renderer, no_change_path, work_dir / "hierarchy_base", "full")
    _, rotated = run(renderer, parent_change_path, work_dir / "hierarchy_rotated", "full")
    base_centers = {entry["id"]: entry["position"] for entry in baseline["rockWorldCenters"]}
    rotated_centers = {entry["id"]: entry["position"] for entry in rotated["rockWorldCenters"]}
    for rock_id in ("spire_shard", "spire_chip"):
        a = base_centers[rock_id]
        b = rotated_centers[rock_id]
        distance = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
        if distance < 0.2:
            fail(f"parent rotation did not move descendant '{rock_id}' (distance {distance})")

    # Invalid scene variants and their mandated business codes.
    cycle_path = generated / "cycle.json"
    write_variant(
        base,
        cycle_path,
        lambda d: (
            d["rocks"][0].__setitem__("parent", d["rocks"][2]["id"]),
            d["rocks"][2].__setitem__("parent", d["rocks"][0]["id"]),
            d.__setitem__("changes", []),
        ),
    )
    light_path = generated / "light.json"
    write_variant(
        base,
        light_path,
        lambda d: (d["lights"][0].__setitem__("intensity", 0.0), d.__setitem__("changes", [])),
    )
    height_path = generated / "height.json"
    write_variant(
        base,
        height_path,
        lambda d: (
            d["heightfield"].__setitem__("heightRange", [-1.0, 14.0]),
            d.__setitem__("changes", []),
        ),
    )
    missing_path = generated / "missing.json"

    def remove_depth_bits(d):
        del d["render"]["depthBits"]
        d["changes"] = []

    write_variant(base, missing_path, remove_depth_bits)
    transform_path = generated / "nonfinite_transform.json"

    def nonfinite_transform(d):
        d["rocks"][0]["transform"]["translation"] = ["__NONFINITE__", 11.0, 3.0]
        d["changes"] = []

    write_variant(base, transform_path, nonfinite_transform, nonfinite=True)

    cases = [
        (cycle_path, "E_SCENE_INVALID"),
        (height_path, "E_SCENE_INVALID"),
        (missing_path, "E_SCENE_INVALID"),
        (light_path, "E_LIGHT_INVALID"),
        (transform_path, "E_TRANSFORM_INVALID"),
    ]
    for index, (scene_path, expected_code) in enumerate(cases):
        proc, stats = run(renderer, scene_path, work_dir / f"error_{index}", "full")
        check_error(proc, stats, expected_code, scene_path.name)

    overflow_path = generated / "stencil_overflow.json"
    overflow = copy.deepcopy(base)
    overflow["render"]["stencilBits"] = 1
    overflow["changes"] = []
    source_rock = copy.deepcopy(overflow["rocks"][0])
    overflow["rocks"] = []
    for i in range(14):
        rock = copy.deepcopy(source_rock)
        rock["id"] = f"overlap_{i}"
        rock["parent"] = None
        rock["transform"]["rotationDeg"] = [8.0 + i * 3.0, 25.0 + i * 5.0, 4.0]
        overflow["rocks"].append(rock)
    overflow_path.write_text(json.dumps(overflow, indent=2) + "\n", encoding="utf-8")
    overflow_proc, overflow_stats = run(
        renderer, overflow_path, work_dir / "overflow", "full"
    )
    check_error(overflow_proc, overflow_stats, "E_STENCIL_OVERFLOW", "stencil overflow")
    if overflow_stats["stencilOverflowCount"] <= 0:
        fail("stencil overflow was reported without a positive overflow count")

    print(
        "PASS: full/incremental byte equality, deterministic tile hashes, "
        "24-bit depth, hierarchy motion, invalid inputs, and stencil overflow"
    )
    print(f"frameHash={inc['frameHash']} depthHash={inc['depthHash']}")
    print(f"affectedTiles={affected}")
    print(f"shadowVolumeTriangles={inc['shadowVolumeTriangles']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
