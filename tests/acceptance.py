#!/usr/bin/env python3
"""Deterministic acceptance checks for the offline deferred renderer."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def fail(message: str) -> None:
    raise AssertionError(message)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(
    renderer: Path,
    work: Path,
    name: str,
    *args: str,
    expected_code: int = 0,
) -> dict:
    output = work / f"{name}.json"
    command = [str(renderer), "--out", str(output), *args]
    completed = subprocess.run(
        command,
        cwd=work,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != expected_code:
        fail(
            f"{name}: expected exit {expected_code}, got {completed.returncode}\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )
    if not output.exists():
        fail(f"{name}: summary file was not written")
    if not output.read_text(encoding="utf-8").lstrip().startswith("{"):
        fail(f"{name}: summary is not complete JSON text")
    return load_json(output)


def caster_report(document: dict) -> dict[int, dict]:
    return {
        light["id"]: light
        for light in document["lights"]
        if light["castsShadow"]
    }


def main() -> int:
    if len(sys.argv) > 1:
        renderer = Path(sys.argv[1]).resolve()
    else:
        renderer = Path("build/dungeon_deferred").resolve()
    if not renderer.exists():
        fail(f"renderer not found: {renderer}")

    temp_root = Path.cwd()
    work = Path(tempfile.mkdtemp(prefix="dungeon_acceptance_", dir=temp_root))
    try:
        base_args = (
            "--width",
            "160",
            "--height",
            "112",
            "--geometry-version",
            "0",
        )

        first = run(renderer, work, "deterministic_a", "--cache", str(work / "a.bin"), *base_args)
        second = run(renderer, work, "deterministic_b", "--cache", str(work / "b.bin"), *base_args)
        first_bytes = (work / "deterministic_a.json").read_bytes()
        second_bytes = (work / "deterministic_b.json").read_bytes()
        if first_bytes != second_bytes:
            fail("two fresh runs with identical inputs produced different summary bytes")

        cache = work / "incremental.bin"
        initial = run(renderer, work, "incremental_initial", "--cache", str(cache), *base_args)
        reused = run(renderer, work, "incremental_reuse", "--cache", str(cache), *base_args)
        if initial["summary"]["tileRebuilds"] != 3 or initial["summary"]["tileHits"] != 0:
            fail("initial shadow cache pass must rebuild all three caster tiles")
        if reused["summary"]["tileRebuilds"] != 0 or reused["summary"]["tileHits"] != 3:
            fail("warm shadow cache pass must reuse all three caster tiles")
        if initial["tiles"] != reused["tiles"]:
            fail("tile light lists/checksums changed across a cache-only reuse")
        if initial["summary"]["pixelLuminanceChecksum"] != reused["summary"]["pixelLuminanceChecksum"]:
            fail("pixel checksum changed across a cache-only reuse")
        sampling_fields = (
            "id",
            "type",
            "tileIndex",
            "atlasX",
            "atlasY",
            "transformHash",
            "shadowTapAttempts",
            "shadowTapSuccesses",
            "shadowTapShadowed",
            "shadowSampleSuccessRate",
            "shadowedRate",
            "litPixels",
        )
        initial_sampling = [
            {field: light[field] for field in sampling_fields}
            for light in initial["lights"]
        ]
        reused_sampling = [
            {field: light[field] for field in sampling_fields}
            for light in reused["lights"]
        ]
        if initial_sampling != reused_sampling:
            fail("light sampling statistics changed across a cache-only reuse")

        shifted = run(
            renderer,
            work,
            "light_shift",
            "--cache",
            str(cache),
            "--light-shift",
            "0.25",
            *base_args,
        )
        shifted_casters = caster_report(shifted)
        if shifted["summary"]["tileHits"] != 2 or shifted["summary"]["tileRebuilds"] != 1:
            fail("moving only light id 2 must reuse two tiles and rebuild one")
        if shifted_casters[2]["action"] != "rebuild":
            fail("moved light id 2 did not rebuild")
        if shifted_casters[1]["action"] != "hit" or shifted_casters[3]["action"] != "hit":
            fail("unmodified caster tiles were not reused after light id 2 moved")

        changed = run(
            renderer,
            work,
            "geometry_changed",
            "--cache",
            str(cache),
            "--geometry-version",
            "1",
            "--width",
            "160",
            "--height",
            "112",
        )
        if changed["summary"]["tileHits"] != 0 or changed["summary"]["tileRebuilds"] != 3:
            fail("geometry version change must invalidate every caster tile")
        if changed["config"]["geometry"]["hash"] == reused["config"]["geometry"]["hash"]:
            fail("geometry-version injection did not alter the scene hash")
        if changed["summary"]["pixelLuminanceChecksum"] == reused["summary"]["pixelLuminanceChecksum"]:
            fail("geometry-version injection did not alter the luminance checksum")

        zero_light_tiles = [tile for tile in initial["tiles"] if tile["lightCount"] == 0]
        if not zero_light_tiles:
            fail("deterministic sample does not exercise ambient-only tiles")
        if any(tile["averageLuminance"] <= 0.0 for tile in zero_light_tiles):
            fail("ambient-only tile has zero brightness")
        if any(tile["luminanceChecksum"] == "0x0000000000000000" for tile in zero_light_tiles):
            fail("ambient-only tile checksum is zero")
        if any(not tile["lightIds"] == sorted(tile["lightIds"]) for tile in initial["tiles"]):
            fail("tile light IDs are not ascending")

        light_types = {light["type"] for light in initial["lights"]}
        if light_types != {"point", "spot"}:
            fail("sample does not contain both point and spot lights")
        for light in caster_report(initial).values():
            if light["tileIndex"] < 0:
                fail(f"caster {light['id']} has no atlas tile")
            if light["litPixels"] == 0:
                fail(f"caster {light['id']} did not illuminate any sampled pixel")
            if light["shadowSampleSuccessRate"] < 0.9:
                fail(f"caster {light['id']} shadow sample success rate is too low")

        atlas_full = run(
            renderer,
            work,
            "atlas_full",
            "--atlas-cols",
            "1",
            "--atlas-rows",
            "1",
            *base_args,
        )
        if "E_ATLAS_FULL" not in atlas_full["errorCodes"]:
            fail("1x1 atlas did not report E_ATLAS_FULL")
        if atlas_full["summary"]["degradedLights"] != 2:
            fail("1x1 atlas must degrade exactly the two overflow casters")
        degraded = [light for light in atlas_full["lights"] if light["action"] == "degraded"]
        if [light["id"] for light in degraded] != [2, 3]:
            fail("deterministic atlas-full degradation order changed")
        if [light["tileIndex"] for light in degraded] != [-1, -1]:
            fail("degraded lights must not receive atlas tiles")

        bias_path = work / "bias_range.json"
        bias_path.write_text('{"sentinel": true}\n', encoding="utf-8")
        bias_error = run(
            renderer,
            work,
            "bias_range",
            "--bias",
            "0.5",
            expected_code=3,
        )
        if bias_error["errorCodes"] != ["E_BIAS_RANGE"]:
            fail("out-of-range bias did not emit E_BIAS_RANGE")
        if not bias_path.read_text(encoding="utf-8").rstrip().endswith("}"):
            fail("bias error summary was partially written")
        if (work / "bias_range.json.tmp").exists():
            fail("atomic summary temp file was left behind")

        fov_error = run(renderer, work, "zero_fov", "--fov", "0", expected_code=2)
        if fov_error["status"] != "error" or "E_CAMERA_INVALID" not in fov_error["errorCodes"]:
            fail("zero frustum did not produce E_CAMERA_INVALID")
        resolution_error = run(
            renderer,
            work,
            "negative_resolution",
            "--width",
            "-1",
            expected_code=2,
        )
        if (
            resolution_error["status"] != "error"
            or "E_RESOLUTION_INVALID" not in resolution_error["errorCodes"]
        ):
            fail("negative resolution did not produce E_RESOLUTION_INVALID")

        edge = run(
            renderer,
            work,
            "edge_tiles",
            "--width",
            "161",
            "--height",
            "113",
            "--geometry-version",
            "0",
        )
        if edge["config"]["tileColumns"] != 11 or edge["config"]["tileRows"] != 8:
            fail("non-multiple-of-16 resolution produced the wrong tile grid")
        if len(edge["tiles"]) != 88:
            fail("edge resolution did not produce 11*8 clipped tiles")
        if edge["tiles"][-1]["width"] != 1 or edge["tiles"][-1]["height"] != 1:
            fail("last edge tile was not clipped to 1x1")
        if sorted({(tile["width"], tile["height"]) for tile in edge["tiles"]}) != [
            (1, 1),
            (1, 16),
            (16, 1),
            (16, 16),
        ]:
            fail("edge tile dimensions are inconsistent")

        print("PASS deterministic repeated run: byte-identical summaries")
        print("PASS cache reuse: initial 3 rebuild / warm 3 hit, identical shading")
        print("PASS selective invalidation: light shift 2 hit / 1 rebuild")
        print("PASS geometry injection: scene hash changed, 3 rebuild")
        print("PASS ambient-only tiles: non-zero luma and checksum")
        print("PASS point + spot lights and shadow comparison sampling")
        print("PASS E_ATLAS_FULL: deterministic 1x1 degradation for lights 2 and 3")
        print("PASS E_BIAS_RANGE: exit 3 and complete atomic error summary")
        print("PASS invalid frustum/resolution: exit 2 and error JSON")
        print("PASS 161x113 clipped tiles: 11x8 grid with 1x1 corner")
        print(
            "RESULT "
            + json.dumps(
                {
                    "checksum": initial["summary"]["pixelLuminanceChecksum"],
                    "lightCountHistogram": {
                        str(count): sum(
                            tile["lightCount"] == count for tile in initial["tiles"]
                        )
                        for count in sorted({tile["lightCount"] for tile in initial["tiles"]})
                    },
                    "shadowComparisons": initial["summary"]["shadowComparisons"],
                    "geometryHashes": {
                        "v0": initial["config"]["geometry"]["hash"],
                        "v1": changed["config"]["geometry"]["hash"],
                    },
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
