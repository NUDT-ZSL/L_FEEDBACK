#!/usr/bin/env python3

import argparse
import copy
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path


EXIT_CODES = {
    "E_CONFIG": 10,
    "E_OBJ_MISSING": 11,
    "E_INDEX_INVALID": 12,
    "E_MAT_MISSING": 13,
    "E_IMAGE_DECODE": 14,
    "E_OUTPUT_WRITE": 15,
    "E_INTERNAL": 16,
}


def base_config():
    return {
        "width": 320,
        "height": 320,
        "obj_path": "../assets/tetra.obj",
        "output_path": "../out/render.bmp",
        "double_sided": False,
        "ambient": [0.06, 0.07, 0.09],
        "background": [0.015, 0.02, 0.03],
        "camera": {
            "position": [0.0, 1.7, 4.8],
            "target": [0.0, -0.1, 0.0],
            "up": [0.0, 1.0, 0.0],
            "vertical_fov_degrees": 42.0,
            "near": 0.1,
            "far": 100.0,
        },
        "lights": [
            {
                "type": "directional",
                "direction": [-0.45, -1.0, -0.55],
                "color": [1.0, 0.96, 0.88],
                "intensity": 0.9,
            },
            {
                "type": "point",
                "position": [-1.8, 1.5, 2.4],
                "color": [1.0, 0.72, 0.45],
                "intensity": 9.0,
                "attenuation": {
                    "constant": 1.0,
                    "linear": 0.18,
                    "quadratic": 0.08,
                },
            },
        ],
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_renderer(executable, config_path):
    return subprocess.run(
        [str(executable), str(config_path)],
        cwd=str(config_path.parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def assert_success(result, description):
    if result.returncode != 0:
        raise AssertionError(
            f"{description}: expected success, got {result.returncode}; "
            f"stderr={result.stderr!r}"
        )
    if result.stderr:
        raise AssertionError(f"{description}: unexpected stderr {result.stderr!r}")


def read_bmp(path):
    data = path.read_bytes()
    if len(data) < 54:
        raise AssertionError(f"{path}: BMP is shorter than 54 bytes")
    if data[0:2] != b"BM":
        raise AssertionError(f"{path}: missing BM signature")

    (
        file_size,
        reserved1,
        reserved2,
        pixel_offset,
    ) = struct.unpack_from("<IHHI", data, 2)
    (
        dib_size,
        width,
        height,
        planes,
        bits_per_pixel,
        compression,
        image_size,
        x_ppm,
        y_ppm,
        colors_used,
        colors_important,
    ) = struct.unpack_from("<IiiHHIIiiII", data, 14)

    if file_size != len(data):
        raise AssertionError(f"{path}: BF_SIZE={file_size}, actual={len(data)}")
    if reserved1 != 0 or reserved2 != 0:
        raise AssertionError(f"{path}: reserved header fields are nonzero")
    if pixel_offset != 54:
        raise AssertionError(f"{path}: expected pixel offset 54, got {pixel_offset}")
    if dib_size != 40:
        raise AssertionError(f"{path}: expected BITMAPINFOHEADER size 40, got {dib_size}")
    if width <= 0 or height <= 0:
        raise AssertionError(f"{path}: invalid dimensions {width}x{height}")
    if planes != 1 or bits_per_pixel != 24 or compression != 0:
        raise AssertionError(
            f"{path}: expected 1 plane, 24bpp, BI_RGB; got "
            f"{planes}, {bits_per_pixel}, {compression}"
        )
    if x_ppm <= 0 or y_ppm <= 0 or colors_used != 0 or colors_important != 0:
        raise AssertionError(f"{path}: invalid DIB metadata")

    row_stride = (width * 3 + 3) & ~3
    if image_size != row_stride * height:
        raise AssertionError(f"{path}: BI_SIZE mismatch")
    if len(data) != pixel_offset + image_size:
        raise AssertionError(f"{path}: file length does not match headers")

    rows = []
    for y in range(height):
        start = pixel_offset + y * row_stride
        row = data[start : start + width * 3]
        padding = data[start + width * 3 : start + row_stride]
        if any(padding):
            raise AssertionError(f"{path}: nonzero BMP row padding at row {y}")
        rows.append(row)

    return {
        "width": width,
        "height": height,
        "row_stride": row_stride,
        "rows_bottom_up": rows,
        "data": data,
    }


def bgr_at_top_down(bmp, x, y):
    row = bmp["rows_bottom_up"][bmp["height"] - 1 - y]
    return tuple(row[x * 3 : x * 3 + 3])


def flat_pixels_top_down(bmp):
    return b"".join(reversed(bmp["rows_bottom_up"]))


def color_bytes(rgb):
    red, green, blue = rgb
    return (
        int(min(max(blue, 0.0), 1.0) * 255.0 + 0.5),
        int(min(max(green, 0.0), 1.0) * 255.0 + 0.5),
        int(min(max(red, 0.0), 1.0) * 255.0 + 0.5),
    )


def validate_rendered_pixels(bmp, background_rgb):
    background = color_bytes(background_rgb)
    if bgr_at_top_down(bmp, 0, 0) != background:
        raise AssertionError("top-left pixel is not the configured background")
    if bgr_at_top_down(bmp, bmp["width"] - 1, bmp["height"] - 1) != background:
        raise AssertionError("bottom-right pixel is not the configured background")

    pixels = flat_pixels_top_down(bmp)
    background_count = 0
    lit_count = 0
    color_count = set()
    maximum = 0
    for offset in range(0, len(pixels), 3):
        pixel = pixels[offset : offset + 3]
        color_count.add(pixel)
        maximum = max(maximum, max(pixel))
        if tuple(pixel) == background:
            background_count += 1
        else:
            lit_count += 1

    if lit_count < 10_000:
        raise AssertionError(f"too few foreground pixels: {lit_count}")
    if background_count < 10_000:
        raise AssertionError(f"too few background pixels: {background_count}")
    if len(color_count) < 500:
        raise AssertionError(f"lighting/texture produced too few colors: {len(color_count)}")
    if maximum < 96:
        raise AssertionError("rendered image has no clearly lit pixels")


def expect_error(executable, config_path, expected_code, description):
    result = run_renderer(executable, config_path)
    expected_exit = EXIT_CODES[expected_code]
    if result.returncode != expected_exit:
        raise AssertionError(
            f"{description}: expected exit {expected_exit}, got {result.returncode}; "
            f"stderr={result.stderr!r}"
        )
    first_line = result.stderr.splitlines()[0] if result.stderr else ""
    if not first_line.startswith(expected_code + ":"):
        raise AssertionError(
            f"{description}: expected stderr starting with {expected_code}, "
            f"got {result.stderr!r}"
        )


def prepare_error_assets(assets):
    original_obj = (assets / "tetra.obj").read_text(encoding="utf-8")
    tangent_obj = (assets / "tetra_with_tangents.obj").read_text(encoding="utf-8")
    source_lines = original_obj.splitlines()
    quad_records = []
    for prefix in ("v ", "vt ", "vn "):
        quad_records.extend(
            [line for line in source_lines if line.startswith(prefix)][:4]
        )
    (assets / "quad.obj").write_text(
        "mtllib tetra.mtl\n"
        + "\n".join(quad_records)
        + "\nusemtl tetra\nf 1/1/1 2/2/2 3/3/3 4/4/4\n",
        encoding="utf-8",
    )

    (assets / "bad_index.obj").write_text(
        original_obj.replace("f 1/1/1 2/2/2 3/3/3", "f 999/1/1 2/2/2 3/3/3"),
        encoding="utf-8",
    )
    (assets / "missing_mtl.obj").write_text(
        original_obj.replace("mtllib tetra.mtl", "mtllib absent.mtl"),
        encoding="utf-8",
    )
    (assets / "missing_attribute.obj").write_text(
        original_obj.replace("f 1/1/1 2/2/2 3/3/3", "f 1 2 3"),
        encoding="utf-8",
    )
    (assets / "missing_diffuse.obj").write_text(
        original_obj.replace("mtllib tetra.mtl", "mtllib missing_diffuse.mtl"),
        encoding="utf-8",
    )
    (assets / "missing_diffuse.mtl").write_text(
        "newmtl tetra\nKd 1 1 1\nmap_Kd absent_diffuse.png\n",
        encoding="utf-8",
    )
    (assets / "missing_normal_path.obj").write_text(
        original_obj.replace("mtllib tetra.mtl", "mtllib missing_normal_path.mtl"),
        encoding="utf-8",
    )
    (assets / "missing_normal_path.mtl").write_text(
        "newmtl tetra\nKd 1 1 1\nmap_Kd tetra_diffuse.png\n"
        "norm absent_normal.png\n",
        encoding="utf-8",
    )
    (assets / "corrupt_diffuse.obj").write_text(
        original_obj.replace("mtllib tetra.mtl", "mtllib corrupt_diffuse.mtl"),
        encoding="utf-8",
    )
    (assets / "corrupt_diffuse.mtl").write_text(
        "newmtl tetra\nKd 1 1 1\nmap_Kd corrupt_diffuse.png\n",
        encoding="utf-8",
    )
    (assets / "corrupt_diffuse.png").write_bytes(b"not a png")
    (assets / "zero_tangent.obj").write_text(
        re.sub(r"(?m)^vtangent .*$", "vtangent 0 0 0", tangent_obj),
        encoding="utf-8",
    )
    (assets / "degenerate_uv.obj").write_text(
        re.sub(r"(?m)^vt .*$", "vt 0 0", original_obj),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    executable = Path(args.exe).resolve()
    generator = Path(args.generator).resolve()
    source_dir = Path(args.source_dir).resolve()
    work_dir = Path(args.work_dir).resolve()

    if work_dir.exists():
        shutil.rmtree(work_dir)
    assets = work_dir / "assets"
    configs = work_dir / "configs"
    output = work_dir / "out"
    configs.mkdir(parents=True)

    generated = subprocess.run(
        [str(generator), str(assets)],
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if generated.returncode != 0:
        raise AssertionError(f"sample generator failed: {generated.stderr!r}")

    repeat_assets = work_dir / "assets_repeat"
    repeated = subprocess.run(
        [str(generator), str(repeat_assets)],
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if repeated.returncode != 0:
        raise AssertionError(f"sample generator repeat failed: {repeated.stderr!r}")
    for first in sorted(assets.iterdir()):
        second = repeat_assets / first.name
        if first.read_bytes() != second.read_bytes():
            raise AssertionError(f"sample generator is not deterministic: {first.name}")

    prepare_error_assets(assets)

    normal_config = base_config()
    normal_config_path = configs / "normal.json"
    write_json(normal_config_path, normal_config)
    result = run_renderer(executable, normal_config_path)
    assert_success(result, "normal-map render")
    normal_bmp = read_bmp(output / "render.bmp")
    if (normal_bmp["width"], normal_bmp["height"]) != (320, 320):
        raise AssertionError("rendered BMP dimensions do not match config")
    validate_rendered_pixels(normal_bmp, normal_config["background"])

    padded_config = copy.deepcopy(base_config())
    padded_config["width"] = 323
    padded_config["height"] = 257
    padded_config["output_path"] = "../out/padded.bmp"
    padded_path = configs / "padded.json"
    write_json(padded_path, padded_config)
    assert_success(run_renderer(executable, padded_path), "BMP row-padding render")
    padded_bmp = read_bmp(output / "padded.bmp")
    if padded_bmp["row_stride"] != 972:
        raise AssertionError(
            f"expected 972-byte padded BMP rows, got {padded_bmp['row_stride']}"
        )
    validate_rendered_pixels(padded_bmp, padded_config["background"])

    fallback_config = copy.deepcopy(base_config())
    fallback_config["obj_path"] = "../assets/tetra_no_normal.obj"
    fallback_config["output_path"] = "../out/fallback.bmp"
    fallback_path = configs / "fallback.json"
    write_json(fallback_path, fallback_config)
    assert_success(run_renderer(executable, fallback_path), "no-normal-map fallback")
    fallback_bmp = read_bmp(output / "fallback.bmp")

    tangent_config = copy.deepcopy(base_config())
    tangent_config["obj_path"] = "../assets/tetra_with_tangents.obj"
    tangent_config["output_path"] = "../out/tangents.bmp"
    tangent_path = configs / "tangents.json"
    write_json(tangent_path, tangent_config)
    assert_success(run_renderer(executable, tangent_path), "supplied tangent render")
    tangent_bmp = read_bmp(output / "tangents.bmp")

    normal_pixels = flat_pixels_top_down(normal_bmp)
    fallback_pixels = flat_pixels_top_down(fallback_bmp)
    tangent_pixels = flat_pixels_top_down(tangent_bmp)
    normal_difference = sum(
        abs(a - b) for a, b in zip(normal_pixels, fallback_pixels)
    )
    tangent_difference = sum(
        abs(a - b) for a, b in zip(normal_pixels, tangent_pixels)
    )
    if normal_difference < 100_000:
        raise AssertionError(
            "normal map had no meaningful pixel effect compared with geometry normals"
        )
    if tangent_difference > 100:
        raise AssertionError(
            f"generated and supplied tangents diverge too much: {tangent_difference}"
        )

    missing_normal_config = copy.deepcopy(base_config())
    missing_normal_config["obj_path"] = "../assets/missing_normal_path.obj"
    missing_normal_config["output_path"] = "../out/missing_normal_path.bmp"
    missing_normal_path = configs / "missing_normal_path.json"
    write_json(missing_normal_path, missing_normal_config)
    assert_success(
        run_renderer(executable, missing_normal_path),
        "missing normal-map file fallback",
    )
    read_bmp(output / "missing_normal_path.bmp")

    zero_tangent_config = copy.deepcopy(base_config())
    zero_tangent_config["obj_path"] = "../assets/zero_tangent.obj"
    zero_tangent_config["output_path"] = "../out/zero_tangent.bmp"
    zero_tangent_path = configs / "zero_tangent.json"
    write_json(zero_tangent_path, zero_tangent_config)
    assert_success(
        run_renderer(executable, zero_tangent_path),
        "zero-length tangent fallback",
    )
    read_bmp(output / "zero_tangent.bmp")

    degenerate_uv_config = copy.deepcopy(base_config())
    degenerate_uv_config["obj_path"] = "../assets/degenerate_uv.obj"
    degenerate_uv_config["output_path"] = "../out/degenerate_uv.bmp"
    degenerate_uv_path = configs / "degenerate_uv.json"
    write_json(degenerate_uv_path, degenerate_uv_config)
    assert_success(
        run_renderer(executable, degenerate_uv_path),
        "degenerate UV tangent fallback",
    )
    read_bmp(output / "degenerate_uv.bmp")

    quad_config = copy.deepcopy(base_config())
    quad_config["obj_path"] = "../assets/quad.obj"
    quad_config["output_path"] = "../out/quad.bmp"
    quad_path = configs / "quad.json"
    write_json(quad_path, quad_config)
    assert_success(run_renderer(executable, quad_path), "quad triangulation")
    read_bmp(output / "quad.bmp")

    clipped_config = copy.deepcopy(base_config())
    clipped_config["output_path"] = "../out/clipped.bmp"
    clipped_config["camera"]["position"] = [0.0, 0.9, 1.0]
    clipped_config["camera"]["near"] = 0.5
    clipped_config["camera"]["far"] = 8.0
    clipped_path = configs / "clipped.json"
    write_json(clipped_path, clipped_config)
    assert_success(run_renderer(executable, clipped_path), "near/frustum clipping")
    read_bmp(output / "clipped.bmp")

    bad_json_path = configs / "bad_json.json"
    bad_json_path.write_text("{", encoding="utf-8")
    expect_error(executable, bad_json_path, "E_CONFIG", "malformed JSON")

    for name, mutate in [
        ("zero_fov", lambda c: c["camera"].__setitem__("vertical_fov_degrees", 0.0)),
        ("zero_width", lambda c: c.__setitem__("width", 0)),
        ("bad_near", lambda c: c["camera"].__setitem__("near", -1.0)),
        (
            "near_after_far",
            lambda c: c["camera"].__setitem__("near", c["camera"]["far"]),
        ),
        ("missing_lights", lambda c: c.pop("lights")),
        (
            "zero_direction",
            lambda c: c["lights"][0].__setitem__("direction", [0.0, 0.0, 0.0]),
        ),
    ]:
        config = copy.deepcopy(base_config())
        mutate(config)
        path = configs / f"{name}.json"
        write_json(path, config)
        expect_error(executable, path, "E_CONFIG", name)

    missing_obj = copy.deepcopy(base_config())
    missing_obj["obj_path"] = "../assets/does_not_exist.obj"
    missing_obj_path = configs / "missing_obj.json"
    write_json(missing_obj_path, missing_obj)
    expect_error(executable, missing_obj_path, "E_OBJ_MISSING", "missing OBJ")

    for name, obj_name, expected_code in [
        ("bad_index", "bad_index.obj", "E_INDEX_INVALID"),
        ("missing_attribute", "missing_attribute.obj", "E_OBJ_MISSING"),
        ("missing_mtl", "missing_mtl.obj", "E_MAT_MISSING"),
        ("missing_diffuse", "missing_diffuse.obj", "E_MAT_MISSING"),
        ("corrupt_diffuse", "corrupt_diffuse.obj", "E_IMAGE_DECODE"),
    ]:
        config = copy.deepcopy(base_config())
        config["obj_path"] = f"../assets/{obj_name}"
        config["output_path"] = f"../out/{name}.bmp"
        path = configs / f"{name}.json"
        write_json(path, config)
        expect_error(executable, path, expected_code, name)

    unwritable_dir = work_dir / "output_is_a_directory"
    unwritable_dir.mkdir()
    unwritable = copy.deepcopy(base_config())
    unwritable["output_path"] = "../output_is_a_directory"
    unwritable_path = configs / "unwritable.json"
    write_json(unwritable_path, unwritable)
    expect_error(executable, unwritable_path, "E_OUTPUT_WRITE", "unwritable output")

    usage = subprocess.run(
        [str(executable)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if usage.returncode != 2 or not usage.stderr.startswith("E_USAGE:"):
        raise AssertionError("missing argument did not produce E_USAGE")

    summary = {
        "rendered_pixels": len(normal_pixels) // 3,
        "padded_bmp_row_stride": padded_bmp["row_stride"],
        "normal_vs_fallback_abs_diff": normal_difference,
        "generated_vs_supplied_tangent_abs_diff": tangent_difference,
        "validated_error_codes": [
            "E_CONFIG",
            "E_OBJ_MISSING",
            "E_INDEX_INVALID",
            "E_MAT_MISSING",
            "E_IMAGE_DECODE",
            "E_OUTPUT_WRITE",
            "E_USAGE",
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"VALIDATION FAILED: {error}", file=sys.stderr)
        sys.exit(1)
