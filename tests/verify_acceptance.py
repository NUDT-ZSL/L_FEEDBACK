#!/usr/bin/env python3
"""End-to-end acceptance checks for the offline terrain renderer.

The script intentionally uses only the Python standard library.  It decodes the
generated RGB PNG files with zlib so validation also works on machines without
Pillow or NumPy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path


class CheckFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def decode_png(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    require(data[:8] == b"\x89PNG\r\n\x1a\n", f"{path}: invalid PNG signature")
    offset = 8
    width = height = 0
    color_type = 0
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            require(depth == 8, f"{path}: expected 8-bit PNG")
            require(color_type in (2, 6), f"{path}: expected RGB/RGBA PNG")
            require(interlace == 0, f"{path}: interlaced PNG is not supported")
        elif chunk_type == b"IDAT":
            compressed.extend(chunk)
        elif chunk_type == b"IEND":
            break
    require(width > 0 and height > 0, f"{path}: missing IHDR")
    raw = zlib.decompress(bytes(compressed))
    channels = 3 if color_type == 2 else 4
    stride = width * channels
    previous = bytearray(stride)
    rgb = bytearray()
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        row = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        for index in range(stride):
            left = row[index - channels] if index >= channels else 0
            up = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                row[index] = (row[index] + left) & 0xFF
            elif filter_type == 2:
                row[index] = (row[index] + up) & 0xFF
            elif filter_type == 3:
                row[index] = (row[index] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                estimate = left + up - upper_left
                pa = abs(estimate - left)
                pb = abs(estimate - up)
                pc = abs(estimate - upper_left)
                predictor = left if pa <= pb and pa <= pc else (up if pb <= pc else upper_left)
                row[index] = (row[index] + predictor) & 0xFF
            elif filter_type != 0:
                raise CheckFailure(f"{path}: unsupported PNG filter {filter_type}")
        if channels == 3:
            rgb.extend(row)
        else:
            for index in range(0, len(row), 4):
                rgb.extend(row[index : index + 3])
        previous = row
    return width, height, bytes(rgb)


def image_difference(
    first: tuple[int, int, bytes], second: tuple[int, int, bytes]
) -> tuple[int, int, int]:
    require(first[:2] == second[:2], "image dimensions differ")
    maximum = 0
    different_pixels = 0
    over_tolerance = 0
    width, height = first[:2]
    a = first[2]
    b = second[2]
    for pixel in range(width * height):
        pixel_max = 0
        for channel in range(3):
            difference = abs(a[pixel * 3 + channel] - b[pixel * 3 + channel])
            maximum = max(maximum, difference)
            pixel_max = max(pixel_max, difference)
        if pixel_max:
            different_pixels += 1
        if pixel_max > 1:
            over_tolerance += 1
    return maximum, different_pixels, over_tolerance


def hard_edge_count(image: tuple[int, int, bytes], threshold: int = 25) -> int:
    width, height = image[:2]
    data = image[2]
    count = 0
    for y in range(height):
        for x in range(width):
            index = (y * width + x) * 3
            luminance = (data[index] * 3 + data[index + 1] * 6 + data[index + 2]) // 10
            if x + 1 < width:
                offset = index + 3
                other = (data[offset] * 3 + data[offset + 1] * 6 + data[offset + 2]) // 10
                if abs(luminance - other) > threshold:
                    count += 1
            if y + 1 < height:
                offset = index + width * 3
                other = (data[offset] * 3 + data[offset + 1] * 6 + data[offset + 2]) // 10
                if abs(luminance - other) > threshold:
                    count += 1
    return count


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Harness:
    def __init__(self, executable: Path, work: Path) -> None:
        self.executable = executable.resolve()
        self.work = work.resolve()
        self.data = self.work / "data"
        self.checks = 0

    def run(
        self,
        arguments: list[str],
        expected_exit: int,
        expect_json: bool = True,
    ) -> dict:
        result = subprocess.run(
            [str(self.executable), *arguments],
            cwd=self.work.parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )
        require(
            result.returncode == expected_exit,
            f"command exited {result.returncode}, expected {expected_exit}: "
            f"{' '.join(arguments)}\nstdout={result.stdout}\nstderr={result.stderr}",
        )
        payload = None
        if expect_json:
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError as error:
                raise CheckFailure(
                    f"stdout is not JSON for {' '.join(arguments)}: {error}\n{result.stdout}"
                ) from error
        self.checks += 1
        return payload

    def check(self, condition: bool, message: str) -> None:
        require(condition, message)
        self.checks += 1

    def execute(self) -> None:
        if self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir(parents=True)

        generated = self.run(
            ["generate-data", "--data-dir", str(self.data), "--seed", "1337"],
            0,
        )
        self.check(generated["width"] == 256 and generated["height"] == 256, "data dimensions")
        self.check(generated["zero_weight_texels"] > 0, "zero-weight fault region missing")
        self.check(
            any(w["code"] == "MATW_ZERO" for w in generated["material_warnings"]),
            "MATW_ZERO warning missing",
        )

        angles_dir = self.work / "angles"
        angles = self.run(
            [
                "angles",
                "--data-dir",
                str(self.data),
                "--out-dir",
                str(angles_dir),
                "--no-restore",
                "--snapshot",
                str(angles_dir / "snapshot.json"),
            ],
            0,
        )
        self.check(angles["triangle_count"] == 130050, "unexpected terrain triangle count")
        self.check(len(angles["outputs"]) == 8, "angles output count")
        self.check(
            any(w["code"] == "MATW_ZERO" for w in angles["material_warnings"]),
            "angles MATW_ZERO warning missing",
        )
        for yaw in (0, 45, 90):
            hard_path = angles_dir / f"yaw{yaw}_hard.png"
            soft_path = angles_dir / f"yaw{yaw}_soft.png"
            hard = decode_png(hard_path)
            soft = decode_png(soft_path)
            self.check(hard[:2] == (1024, 768), f"yaw {yaw} hard dimensions")
            self.check(soft[:2] == (1024, 768), f"yaw {yaw} soft dimensions")
            self.check(hard[2] != soft[2], f"yaw {yaw} hard/soft images are identical")
            self.check(
                hard_edge_count(soft) <= hard_edge_count(hard),
                f"yaw {yaw} soft shadow has more hard edges than hard shadow",
            )
        bias2 = decode_png(angles_dir / "bias2_hard.png")
        bias4 = decode_png(angles_dir / "bias4_hard.png")
        self.check(bias2[2] != bias4[2], "2-texel and 4-texel hard shadows are identical")
        for report_path in (
            angles_dir / "yaw0_hard.json",
            angles_dir / "bias2_hard.json",
            angles_dir / "bias4_hard.json",
        ):
            report = json.loads(report_path.read_text())
            for field in (
                "render_time_ms",
                "triangle_count",
                "redrawn_blocks",
                "shadow_samples",
                "material_warnings",
            ):
                self.check(field in report, f"report field missing: {field}")

        deterministic_dir = self.work / "determinism"
        deterministic_dir.mkdir()
        deterministic_args = [
            "render",
            "--data-dir",
            str(self.data),
            "--out",
            str(deterministic_dir / "image.png"),
            "--report",
            str(deterministic_dir / "report.json"),
            "--yaw",
            "45",
            "--pan-x",
            "0.5",
            "--shadow",
            "soft",
            "--bias-texels",
            "2",
            "--no-restore",
            "--no-save-snapshot",
        ]
        self.run(deterministic_args, 0)
        first_png = (deterministic_dir / "image.png").read_bytes()
        first_report = (deterministic_dir / "report.json").read_bytes()
        self.run(deterministic_args, 0)
        self.check(first_png == (deterministic_dir / "image.png").read_bytes(), "PNG not deterministic")
        self.check(
            first_report == (deterministic_dir / "report.json").read_bytes(),
            "JSON report not deterministic",
        )

        incremental_dir = self.work / "incremental"
        incremental = self.run(
            [
                "incremental",
                "--data-dir",
                str(self.data),
                "--out-dir",
                str(incremental_dir),
                "--yaw",
                "0",
                "--step",
                "0.65",
                "--shadow",
                "soft",
                "--bias-texels",
                "2",
                "--no-restore",
                "--snapshot",
                str(incremental_dir / "snapshot.json"),
            ],
            0,
        )
        self.check(incremental["frame_count"] == 7, "incremental frame count")
        pans = [frame["pan_x"] for frame in incremental["frames"]]
        expected_pans = [0.0, 0.65, 1.30, 1.95, 2.60, 3.25, 0.0]
        self.check(
            all(abs(a - b) < 1e-6 for a, b in zip(pans, expected_pans)),
            f"unexpected pan sequence: {pans}",
        )
        self.check(
            incremental["total_redrawn_blocks"]
            < int(incremental["full_equivalent_blocks"] * 0.8),
            "incremental redraw savings below 20%",
        )
        self.check(incremental["final_within_tolerance"] is True, "incremental tolerance flag")
        self.check(incremental["final_max_channel_diff"] <= 1, "incremental max channel diff")
        final_image = decode_png(incremental_dir / "frame_06.png")
        reference_image = decode_png(incremental_dir / "reference_full.png")
        max_diff, _, over_tolerance = image_difference(final_image, reference_image)
        self.check(max_diff <= 1 and over_tolerance == 0, "incremental/full image comparison")

        invalid = self.run(["render", "--bogus"], 2)
        self.check(invalid["error"]["code"] == "INVALID_ARGUMENT", "invalid-argument code")
        missing_value = self.run(["render", "--out"], 2)
        self.check(missing_value["error"]["code"] == "INVALID_ARGUMENT", "missing-value code")
        degenerate = self.run(
            [
                "render",
                "--data-dir",
                str(self.data),
                "--fov-y",
                "0",
                "--no-save-snapshot",
            ],
            1,
        )
        self.check(degenerate["error"]["code"] == "DEGENERATE_PROJ", "degenerate projection code")
        for fault, code in (
            ("dim_mismatch", "DIM_MISMATCH"),
            ("degenerate_proj", "DEGENERATE_PROJ"),
            ("shadow_alloc", "SHADOW_ALLOC"),
        ):
            result = self.run(["render", "--test-fault", fault, "--no-save-snapshot"], 1)
            self.check(result["error"]["code"] == code, f"fault code {fault}")

        small_data = self.work / "small_data"
        self.run(
            [
                "generate-data",
                "--data-dir",
                str(small_data),
                "--width",
                "128",
                "--height",
                "128",
            ],
            0,
        )
        dimension_failure = self.run(
            [
                "render",
                "--data-dir",
                str(small_data),
                "--out",
                str(self.work / "dimension_failure.png"),
                "--no-save-snapshot",
            ],
            1,
        )
        self.check(
            dimension_failure["error"]["code"] == "DIM_MISMATCH",
            "actual dimension mismatch code",
        )

        mismatched_data = self.work / "mismatched_data"
        shutil.copytree(self.data, mismatched_data)
        weight_path = mismatched_data / "material_weights.wgt8"
        weight_bytes = bytearray(weight_path.read_bytes())
        weight_bytes[8:12] = struct.pack("<I", 255)
        weight_path.write_bytes(weight_bytes)
        header_failure = self.run(
            [
                "render",
                "--data-dir",
                str(mismatched_data),
                "--out",
                str(self.work / "header_failure.png"),
                "--no-save-snapshot",
            ],
            1,
        )
        self.check(
            header_failure["error"]["code"] == "DIM_MISMATCH",
            "height/weight header mismatch code",
        )

        cancel_dir = self.work / "cancel"
        cancel_dir.mkdir()
        sentinel = cancel_dir / "preserved.png"
        sentinel.write_bytes(b"existing-output-sentinel")
        before = sha256(sentinel)
        canceled = self.run(
            [
                "render",
                "--data-dir",
                str(self.data),
                "--out",
                str(sentinel),
                "--report",
                str(cancel_dir / "canceled.json"),
                "--cancel-after-triangles",
                "5000",
                "--no-save-snapshot",
            ],
            1,
        )
        self.check(canceled["error"]["code"] == "CANCELED", "cancellation code")
        self.check(sha256(sentinel) == before, "cancellation overwrote existing PNG")
        self.check(
            not (cancel_dir / "canceled.json").exists(),
            "cancellation produced a partial JSON report",
        )

        snapshot_dir = self.work / "snapshot"
        snapshot_dir.mkdir()
        snapshot_path = snapshot_dir / "state.json"
        self.run(
            [
                "render",
                "--data-dir",
                str(self.data),
                "--out",
                str(snapshot_dir / "first.png"),
                "--report",
                str(snapshot_dir / "first.json"),
                "--yaw",
                "45",
                "--pan-x",
                "1.25",
                "--shadow",
                "soft",
                "--snapshot",
                str(snapshot_path),
                "--no-restore",
            ],
            0,
        )
        restored = self.run(
            [
                "render",
                "--data-dir",
                str(self.data),
                "--out",
                str(snapshot_dir / "restored.png"),
                "--report",
                str(snapshot_dir / "restored.json"),
                "--snapshot",
                str(snapshot_path),
            ],
            0,
        )
        self.check(abs(restored["camera"]["yaw_deg"] - 45.0) < 1e-5, "snapshot camera restore")
        self.check(abs(restored["camera"]["pan_x"] - 1.25) < 1e-5, "snapshot pan restore")
        self.check(
            (snapshot_dir / "first.png").read_bytes()
            == (snapshot_dir / "restored.png").read_bytes(),
            "snapshot restore changes rendered image",
        )

        snapshot_path.write_text(
            snapshot_path.read_text().replace('"version": 3', '"version": 99', 1)
        )
        version_mismatch = self.run(
            [
                "render",
                "--data-dir",
                str(self.data),
                "--out",
                str(snapshot_dir / "version.png"),
                "--report",
                str(snapshot_dir / "version.json"),
                "--snapshot",
                str(snapshot_path),
            ],
            0,
        )
        self.check(
            abs(version_mismatch["camera"]["yaw_deg"]) < 1e-6,
            "version-mismatched snapshot was not ignored",
        )
        snapshot_path.write_text("{ malformed snapshot")
        malformed = self.run(
            [
                "render",
                "--data-dir",
                str(self.data),
                "--out",
                str(snapshot_dir / "malformed.png"),
                "--report",
                str(snapshot_dir / "malformed.json"),
                "--snapshot",
                str(snapshot_path),
            ],
            0,
        )
        self.check(
            abs(malformed["camera"]["yaw_deg"]) < 1e-6,
            "malformed snapshot was not ignored safely",
        )

        print(
            json.dumps(
                {
                    "ok": True,
                    "checks": self.checks,
                    "work_dir": str(self.work),
                    "incremental_total_redrawn_blocks": incremental[
                        "total_redrawn_blocks"
                    ],
                    "incremental_full_equivalent_blocks": incremental[
                        "full_equivalent_blocks"
                    ],
                    "incremental_final_max_channel_diff": incremental[
                        "final_max_channel_diff"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--exe",
        type=Path,
        default=repository / "build" / "bin" / "terrain_renderer",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=repository / "outputs" / "acceptance_test",
    )
    args = parser.parse_args()
    try:
        require(args.exe.exists(), f"executable not found: {args.exe}")
        Harness(args.exe, args.work_dir).execute()
        return 0
    except CheckFailure as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

