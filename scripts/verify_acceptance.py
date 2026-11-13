"""Run the local acceptance checks and write artifacts/verification.json."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_scene.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_cli(
    scene_path: Path,
    output_path: Path,
    snapshot_dir: Path | None,
    report_path: Path,
    quality: str | None = None,
) -> dict:
    command = [
        sys.executable,
        "-m",
        "render_pipeline",
        "--scene",
        str(scene_path),
        "--output",
        str(output_path),
        "--report",
        str(report_path),
    ]
    if snapshot_dir is None:
        command.append("--no-snapshot")
    else:
        command.extend(["--snapshot-dir", str(snapshot_dir)])
    if quality is not None:
        command.extend(["--quality", quality])
    process = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(process.stdout)


def assemble_reference_match(scene_data: dict) -> dict:
    from render_pipeline import PipelineOptions, load_scene, render_reference, render_scene

    scene = load_scene(scene_data)
    reference = render_reference(scene)
    tiled = render_scene(scene, PipelineOptions(workers=4, snapshot_dir=None))
    by_coordinate = {(tile.x, tile.y): tile for tile in tiled.gbuffer_tiles}
    channel_mismatches = 0
    for pixel_y in range(scene.height):
        for pixel_x in range(scene.width):
            tile = by_coordinate[
                (
                    (pixel_x // scene.tile_width) * scene.tile_width,
                    (pixel_y // scene.tile_height) * scene.tile_height,
                )
            ]
            local_index = (pixel_y % scene.tile_height) * tile.width + pixel_x % scene.tile_width
            whole_index = pixel_y * scene.width + pixel_x
            if reference.gbuffer.depth[whole_index] != tile.depth[local_index]:
                channel_mismatches += 1
            if reference.gbuffer.material_ids[whole_index] != tile.material_ids[local_index]:
                channel_mismatches += 1
            for component in range(3):
                if (
                    reference.gbuffer.normal[whole_index * 3 + component]
                    != tile.normal[local_index * 3 + component]
                ):
                    channel_mismatches += 1
                if (
                    reference.gbuffer.albedo[whole_index * 3 + component]
                    != tile.albedo[local_index * 3 + component]
                ):
                    channel_mismatches += 1
    return {
        "frame_equal": reference.frame == tiled.frame,
        "channel_mismatches": channel_mismatches,
        "tile_count": len(tiled.gbuffer_tiles),
    }


def main() -> int:
    base_scene = json.loads(SAMPLE.read_text(encoding="utf-8"))
    test_process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-v", "tests"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    reference = assemble_reference_match(base_scene)
    record: dict = {
        "recorded_at": "2026-11-14",
        "python": sys.version.split()[0],
        "command": "python -m render_pipeline --scene sample_scene.json --output out.ppm",
        "tests": {
            "returncode": test_process.returncode,
            "count": next(
                (
                    int(line.split()[1])
                    for line in test_process.stderr.splitlines()
                    if line.startswith("Ran ") and len(line.split()) >= 2
                ),
                None,
            ),
            "summary": test_process.stderr.strip().splitlines()[-1] if test_process.stderr.strip() else "",
            "passed": test_process.returncode == 0,
        },
        "reference_match": reference,
    }
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        snapshot_dir = directory / "snapshots"
        first_output = directory / "first.ppm"
        second_output = directory / "second.ppm"
        first = run_cli(SAMPLE, first_output, snapshot_dir, directory / "first.report.json")
        second = run_cli(SAMPLE, second_output, snapshot_dir, directory / "second.report.json")
        first_report = json.loads((directory / "first.report.json").read_text(encoding="utf-8"))
        second_report = json.loads((directory / "second.report.json").read_text(encoding="utf-8"))
        record["exact_cli"] = {
            "first_hash": first["output_sha256"],
            "second_hash": second["output_sha256"],
            "incremental_hash_equal": first["output_sha256"] == second["output_sha256"],
            "first_rendered_tiles": first_report["summary"]["rendered_tiles"],
            "second_reused_tiles": second_report["summary"]["reused_tiles"],
            "snapshot_count": len(list(snapshot_dir.glob("*.snapshot.json"))),
        }

        changed_scene = copy.deepcopy(base_scene)
        changed_scene["materials"][2]["albedo"] = [0.24, 0.17, 0.61]
        changed_scene["lights"][1]["color"] = [1.0, 0.33, 0.14]
        changed_path = directory / "changed.json"
        changed_path.write_text(json.dumps(changed_scene, sort_keys=True), encoding="utf-8")
        changed_output = directory / "changed.ppm"
        changed_fresh_output = directory / "changed_fresh.ppm"
        incremental = run_cli(
            changed_path,
            changed_output,
            snapshot_dir,
            directory / "changed.report.json",
        )
        fresh = run_cli(
            changed_path,
            changed_fresh_output,
            None,
            directory / "changed_fresh.report.json",
        )
        changed_report = json.loads((directory / "changed.report.json").read_text(encoding="utf-8"))
        record["incremental"] = {
            "incremental_hash": incremental["output_sha256"],
            "fresh_hash": fresh["output_sha256"],
            "hash_equal_to_fresh": incremental["output_sha256"] == fresh["output_sha256"],
            "reused_tiles": changed_report["summary"]["reused_tiles"],
            "rendered_tiles": changed_report["summary"]["rendered_tiles"],
            "changed_materials": changed_report["incremental"]["changed_materials"],
            "changed_light_colors": changed_report["incremental"]["changed_light_colors"],
        }

        fast_output = directory / "fast.ppm"
        fast = run_cli(
            SAMPLE,
            fast_output,
            directory / "fast_snapshots",
            directory / "fast.report.json",
            quality="fast",
        )
        fast_report = json.loads((directory / "fast.report.json").read_text(encoding="utf-8"))
        record["quality"] = {
            "high_hash": first["output_sha256"],
            "fast_hash": fast["output_sha256"],
            "outputs_differ": first["output_sha256"] != fast["output_sha256"],
            "fast_effective": fast_report["shadow"]["effective"],
        }

        invalid_scene = copy.deepcopy(base_scene)
        invalid_scene["materials"][2]["albedo"] = [-0.4, 0.1, 0.1]
        invalid_path = directory / "invalid.json"
        invalid_path.write_text(json.dumps(invalid_scene, sort_keys=True), encoding="utf-8")
        invalid_snapshots = directory / "invalid_snapshots"
        invalid = run_cli(
            invalid_path,
            directory / "invalid.ppm",
            invalid_snapshots,
            directory / "invalid.report.json",
        )
        invalid_report = json.loads((directory / "invalid.report.json").read_text(encoding="utf-8"))
        failed_ids = [tile["id"] for tile in invalid_report["failed_tiles"]]
        record["failure_isolation"] = {
            "status": invalid_report["status"],
            "succeeded_tiles": invalid_report["summary"]["succeeded_tiles"],
            "failed_tiles": len(failed_ids),
            "failed_codes": sorted({tile["error"]["code"] for tile in invalid_report["failed_tiles"]}),
            "failed_snapshots_absent": all(
                not (invalid_snapshots / f"{tile_id}.snapshot.json").exists() for tile_id in failed_ids
            ),
            "snapshot_count": len(list(invalid_snapshots.glob("*.snapshot.json"))),
        }

        from render_pipeline import SnapshotCorruptError, load_snapshot

        first_snapshot = next(snapshot_dir.glob("*.snapshot.json"))
        raw_snapshot = json.loads(first_snapshot.read_text(encoding="utf-8"))
        raw_snapshot["scene"]["summary"] = "tampered"
        first_snapshot.write_text(json.dumps(raw_snapshot, sort_keys=True), encoding="utf-8")
        try:
            load_snapshot(first_snapshot)
            corruption_code = None
        except SnapshotCorruptError as error:
            corruption_code = error.code
        record["snapshot_corruption"] = {
            "rejected": corruption_code == "E_SNAPSHOT_CORRUPT",
            "code": corruption_code,
        }

    destination = ROOT / "artifacts" / "verification.json"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    checks_ok = all(
        (
            record["tests"]["passed"],
            record["reference_match"]["frame_equal"],
            record["reference_match"]["channel_mismatches"] == 0,
            record["exact_cli"]["incremental_hash_equal"],
            record["exact_cli"]["snapshot_count"] == record["reference_match"]["tile_count"],
            record["incremental"]["hash_equal_to_fresh"],
            record["quality"]["outputs_differ"],
            record["failure_isolation"]["failed_snapshots_absent"],
            record["failure_isolation"]["status"] == "partial",
            record["snapshot_corruption"]["rejected"],
        )
    )
    return 0 if checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
