#!/usr/bin/env python3

import argparse
import json
import math
import pathlib
import struct
import subprocess
import sys
import tempfile


def run(command, expected_code, warning_contains=None):
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != expected_code:
        raise AssertionError(
            "command returned {} (expected {}): {}\nstdout:\n{}\nstderr:\n{}".format(
                result.returncode, expected_code, " ".join(map(str, command)),
                result.stdout, result.stderr
            )
        )
    if warning_contains is not None and warning_contains not in result.stderr:
        raise AssertionError(
            "expected stderr to contain {!r}, got {!r}".format(
                warning_contains, result.stderr
            )
        )
    return result


def load_bmp(path):
    data = pathlib.Path(path).read_bytes()
    if len(data) < 54 or data[:2] != b"BM":
        raise AssertionError("not a BMP file: {}".format(path))
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    dib_size = struct.unpack_from("<I", data, 14)[0]
    width, signed_height = struct.unpack_from("<ii", data, 18)
    planes, bits = struct.unpack_from("<HH", data, 26)
    compression = struct.unpack_from("<I", data, 30)[0]
    if dib_size < 40 or planes != 1 or bits != 24 or compression != 0:
        raise AssertionError("unexpected BMP format")
    height = abs(signed_height)
    stride = ((width * 3 + 3) // 4) * 4

    def pixel(x, y):
        source_y = height - 1 - y if signed_height > 0 else y
        offset = pixel_offset + source_y * stride + x * 3
        return tuple(data[offset:offset + 3])

    return width, height, pixel


def sample_texture(path, u, v):
    width, height, pixel = load_bmp(path)
    wrapped_u = u - math.floor(u)
    wrapped_v = v - math.floor(v)
    x = min(width - 1, int(math.floor(wrapped_u * width)))
    y_from_bottom = min(height - 1, int(math.floor(wrapped_v * height)))
    return pixel(x, height - 1 - y_from_bottom)


def expected_center_bgr(scene, base_path, detail_path):
    material = scene["materials"][0]
    base_bgr = sample_texture(base_path, 1.5, 1.5)
    detail_bgr = sample_texture(detail_path, 1.5, 1.5)
    base_rgb = [base_bgr[2] / 255.0, base_bgr[1] / 255.0, base_bgr[0] / 255.0]
    detail_rgb = [
        detail_bgr[2] / 255.0, detail_bgr[1] / 255.0, detail_bgr[0] / 255.0
    ]
    alpha = 0.5
    layered = [
        base_rgb[i] * (1.0 - alpha) + detail_rgb[i] * alpha for i in range(3)
    ]
    albedo = [material["diffuse"][i] * layered[i] for i in range(3)]

    light_sum = [0.0, 0.0, 0.0]
    for light in scene["lights"]:
        for i in range(3):
            light_sum[i] += light["color"][i] * light["intensity"]
    light_sum = [min(1.0, max(0.0, value)) for value in light_sum]

    output = []
    for i in range(3):
        hdr = albedo[i] * (light_sum[i] + 0.1)
        mapped = hdr / (1.0 + hdr)
        gamma = math.pow(mapped, 1.0 / 2.2)
        output.append(int(math.floor(gamma * 255.0 + 0.5)))
    return (output[2], output[1], output[0])


def write_scene(path, scene):
    path.write_text(json.dumps(scene), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--texture-gen", required=True)
    parser.add_argument("--sample", required=True)
    args = parser.parse_args()

    renderer = pathlib.Path(args.renderer).resolve()
    texture_gen = pathlib.Path(args.texture_gen).resolve()
    sample_path = pathlib.Path(args.sample).resolve()
    scene = json.loads(sample_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="soft-raster-test-") as directory:
        root = pathlib.Path(directory)
        texture_dir = root / "textures"
        run([texture_gen, texture_dir], 0)
        scene_path = root / "scene.json"
        write_scene(scene_path, scene)

        output1 = root / "base.bmp"
        result = run(
            [renderer, "--input", scene_path, "--output", output1], 0
        )
        stats = json.loads(result.stdout)
        expected_stats = {
            "totalTriangles": 6,
            "culledMeshes": 1,
            "rasterizedTriangles": 4,
        }
        for key, expected in expected_stats.items():
            if stats[key] != expected:
                raise AssertionError(
                    "{} was {}, expected {}".format(key, stats[key], expected)
                )
        if stats["totalFragments"] <= 0:
            raise AssertionError("totalFragments must be positive")
        if stats["depthTestFailures"] <= 0:
            raise AssertionError("depthTestFailures must be positive")
        if stats["depthTestFailures"] > stats["totalFragments"]:
            raise AssertionError("depth failures exceed generated fragments")

        width, height, pixel = load_bmp(output1)
        if (width, height) != (320, 240):
            raise AssertionError("unexpected BMP dimensions")
        center = pixel(160, 120)
        expected_center = expected_center_bgr(
            scene, texture_dir / "base.bmp", texture_dir / "detail.bmp"
        )
        if center != expected_center:
            raise AssertionError(
                "center pixel BGR was {}, expected {}".format(
                    center, expected_center
                )
            )
        if pixel(5, 5) != (0, 0, 0):
            raise AssertionError("background pixel must be black")

        output_override = root / "override.bmp"
        run(
            [
                renderer, "--input", scene_path, "--output", output_override,
                "--width", "96", "--height", "64"
            ],
            0,
        )
        if load_bmp(output_override)[:2] != (96, 64):
            raise AssertionError("resolution override was not applied")

        output2 = root / "repeat.bmp"
        run([renderer, "--input", scene_path, "--output", output2], 0)
        if output1.read_bytes() != output2.read_bytes():
            raise AssertionError("renderer output is not byte-deterministic")

        no_cull_scene = json.loads(json.dumps(scene))
        no_cull_scene["meshes"] = no_cull_scene["meshes"][:2]
        no_cull_path = root / "no_cull.json"
        write_scene(no_cull_path, no_cull_scene)
        no_cull_output = root / "no_cull.bmp"
        no_cull_result = run(
            [renderer, "--input", no_cull_path, "--output", no_cull_output], 0
        )
        no_cull_stats = json.loads(no_cull_result.stdout)
        if no_cull_stats["culledMeshes"] != 0:
            raise AssertionError("scene without off-screen mesh unexpectedly culled")
        if output1.read_bytes() != no_cull_output.read_bytes():
            raise AssertionError("frustum culling changed visible pixels")

        for name, translation in [
            ("right", (100.0, 0.0, 0.0)),
            ("left", (-100.0, 0.0, 0.0)),
            ("above", (0.0, 100.0, 0.0)),
            ("below", (0.0, -100.0, 0.0)),
            ("behind", (0.0, 0.0, -100.0)),
            ("beyond_far", (0.0, 0.0, 100.0)),
        ]:
            plane_scene = json.loads(json.dumps(scene))
            plane_scene["meshes"][2]["model"][3] = translation[0]
            plane_scene["meshes"][2]["model"][7] = translation[1]
            plane_scene["meshes"][2]["model"][11] = translation[2]
            plane_scene_path = root / "cull_{}.json".format(name)
            write_scene(plane_scene_path, plane_scene)
            plane_result = run(
                [
                    renderer, "--input", plane_scene_path,
                    "--output", root / "cull_{}.bmp".format(name)
                ],
                0,
            )
            if json.loads(plane_result.stdout)["culledMeshes"] != 1:
                raise AssertionError(
                    "{} frustum plane did not cull the off-screen mesh".format(
                        name
                    )
                )

        missing_scene = json.loads(json.dumps(scene))
        missing_scene["materials"][0]["baseTexture"] = "textures/missing.bmp"
        missing_path = root / "missing_texture.json"
        write_scene(missing_path, missing_scene)
        missing_output = root / "missing.bmp"
        run(
            [renderer, "--input", missing_path, "--output", missing_output],
            0,
            warning_contains="missing.bmp",
        )

        clip_scene = {
            "resolution": [160, 120],
            "camera": scene["camera"],
            "lights": scene["lights"],
            "materials": scene["materials"],
            "meshes": [
                {
                    "name": "near_plane_crossing",
                    "positions": [
                        0.0, 0.0, -3.9,
                        -1.0, -1.0, 0.0,
                        1.0, -1.0, 0.0
                    ],
                    "normals": [
                        0.0, 0.0, -1.0,
                        0.0, 0.0, -1.0,
                        0.0, 0.0, -1.0
                    ],
                    "uvs": [0.5, 0.5, 0.0, 0.0, 1.0, 0.0],
                    "colors": [
                        1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0
                    ],
                    "indices": [0, 1, 2],
                    "material": 1
                }
            ]
        }
        clip_path = root / "near_clip.json"
        write_scene(clip_path, clip_scene)
        clip_output = root / "near_clip.bmp"
        clip_result = run(
            [renderer, "--input", clip_path, "--output", clip_output], 0
        )
        clip_stats = json.loads(clip_result.stdout)
        if clip_stats["totalTriangles"] != 1:
            raise AssertionError("near-clip test triangle count is wrong")
        if clip_stats["rasterizedTriangles"] < 1:
            raise AssertionError("near-clipped triangle was not rasterized")
        if clip_stats["totalFragments"] <= 0:
            raise AssertionError("near-clipped triangle generated no fragments")
        clip_width, clip_height, clip_pixel = load_bmp(clip_output)
        has_visible_pixel = any(
            clip_pixel(x, y) != (0, 0, 0)
            for y in range(clip_height)
            for x in range(clip_width)
        )
        if not has_visible_pixel:
            raise AssertionError("near-clipped triangle produced a blank image")

        missing_field_scene = json.loads(json.dumps(scene))
        del missing_field_scene["camera"]["position"]
        missing_field_path = root / "missing_field.json"
        write_scene(missing_field_path, missing_field_scene)
        run(
            [
                renderer, "--input", missing_field_path,
                "--output", root / "invalid.bmp"
            ],
            1,
        )

        bad_material_scene = json.loads(json.dumps(scene))
        bad_material_scene["meshes"][0]["material"] = 99
        bad_material_path = root / "bad_material.json"
        write_scene(bad_material_path, bad_material_scene)
        run(
            [
                renderer, "--input", bad_material_path,
                "--output", root / "invalid-material.bmp"
            ],
            1,
        )

        bad_vertices_scene = json.loads(json.dumps(scene))
        bad_vertices_scene["meshes"][0]["normals"] = (
            bad_vertices_scene["meshes"][0]["normals"][:-3]
        )
        bad_vertices_path = root / "bad_vertices.json"
        write_scene(bad_vertices_path, bad_vertices_scene)
        run(
            [
                renderer, "--input", bad_vertices_path,
                "--output", root / "invalid-vertices.bmp"
            ],
            1,
        )

        nan_scene = json.loads(json.dumps(scene))
        nan_scene["camera"]["position"][0] = float("nan")
        nan_path = root / "nan.json"
        nan_path.write_text(json.dumps(nan_scene), encoding="utf-8")
        run(
            [renderer, "--input", nan_path, "--output", root / "nan.bmp"],
            1,
        )

        infinity_scene = json.loads(json.dumps(scene))
        infinity_scene["camera"]["position"][0] = float("inf")
        infinity_path = root / "infinity.json"
        infinity_path.write_text(json.dumps(infinity_scene), encoding="utf-8")
        run(
            [
                renderer, "--input", infinity_path,
                "--output", root / "infinity.bmp"
            ],
            1,
        )

        unwritable = root / "missing-directory" / "result.bmp"
        run(
            [renderer, "--input", scene_path, "--output", unwritable],
            2,
        )

        print(
            "PASS: stats={} center_bgr={} background_bgr={}".format(
                json.dumps(stats, sort_keys=True), center, pixel(5, 5)
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exception:
        print("FAIL: {}".format(exception), file=sys.stderr)
        raise

