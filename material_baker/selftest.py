"""Deterministic end-to-end and numerical self-test."""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable
import zlib

import numpy as np

from .bake import run_bake
from .config import load_config
from .errors import ToolError
from .lighting import evaluate_material
from .pngio import read_png_rgb, write_rgb_png
from .resample import decode_normal_map, normalize_vectors, resize_image


SEED = 0x5EED2026
SAMPLE_SIZE = 256
THEORY_TOLERANCE = 3.0e-4
WORK_DIR_MARKER = ".material_baker_selftest"


def _encode_linear(value: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(np.clip(value, 0.0, 1.0), 1.0 / gamma)


def _normalize_scalar(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-2:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / length


def _schlick_scalar(f0: np.ndarray, cosine: float) -> np.ndarray:
    return f0 + (1.0 - f0) * ((1.0 - max(0.0, min(1.0, cosine))) ** 5)


def _reference_pixel(
    albedo: np.ndarray,
    roughness_value: float,
    metallic_value: float,
    encoded_normal: np.ndarray,
    *,
    flip_green: bool,
    view: np.ndarray,
    light_direction: np.ndarray,
    light_color: np.ndarray,
    light_intensity: float,
    sky: np.ndarray,
    ground: np.ndarray,
    ambient_intensity: float,
    alpha_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scalar, independent implementation of the documented lighting model."""

    normal = _normalize_scalar(encoded_normal * 2.0 - 1.0)
    if flip_green:
        normal = _normalize_scalar(normal * np.asarray([1.0, -1.0, 1.0]))
    light = _normalize_scalar(light_direction)
    view = _normalize_scalar(view)
    half = _normalize_scalar(light + view)

    ndot_l = max(0.0, min(1.0, float(np.dot(normal, light))))
    ndot_v = max(0.0, min(1.0, float(np.dot(normal, view))))
    ndot_h = max(0.0, min(1.0, float(np.dot(normal, half))))
    vdot_h = max(0.0, min(1.0, float(np.dot(view, half))))

    roughness = max(0.0, min(1.0, float(roughness_value)))
    metallic = max(0.0, min(1.0, float(metallic_value)))
    alpha = max(roughness * roughness, alpha_epsilon)
    alpha_sq = alpha * alpha
    ggx_denominator = math.pi * (
        (ndot_h * ndot_h * (alpha_sq - 1.0) + 1.0) ** 2 + 1.0e-12
    )
    distribution = alpha_sq / ggx_denominator

    def g1(cosine: float) -> float:
        root = math.sqrt(alpha_sq + (1.0 - alpha_sq) * cosine * cosine)
        return 2.0 * cosine / max(cosine + root, 1.0e-12)

    geometry = g1(ndot_v) * g1(ndot_l)
    f0 = 0.04 * (1.0 - metallic) + albedo * metallic
    fresnel = _schlick_scalar(f0, vdot_h)
    specular_brdf = (
        distribution
        * geometry
        * fresnel
        / max(4.0 * ndot_v * ndot_l, 1.0e-12)
    )
    diffuse_brdf = (1.0 - fresnel) * (1.0 - metallic) * albedo / math.pi
    incident = light_color * light_intensity * ndot_l
    direct_diffuse = diffuse_brdf * incident
    direct_specular = specular_brdf * incident
    combined = direct_diffuse + direct_specular
    safety = np.minimum(1.0, incident / np.maximum(combined, 1.0e-12))
    direct_diffuse = direct_diffuse * safety
    direct_specular = direct_specular * safety

    up_weight = max(0.0, min(1.0, 0.5 * normal[2] + 0.5))
    ambient_irradiance = (
        ground + (sky - ground) * up_weight
    ) * ambient_intensity
    fresnel_ambient = _schlick_scalar(f0, ndot_v)
    environment_diffuse = (
        (1.0 - fresnel_ambient)
        * (1.0 - metallic)
        * albedo
        * ambient_irradiance
    )

    reflection = _normalize_scalar(2.0 * ndot_v * normal - view)
    reflection_weight = max(0.0, min(1.0, 0.5 * reflection[2] + 0.5))
    sharp = ground + (sky - ground) * reflection_weight
    broad = 0.5 * (sky + ground)
    rough_blend = roughness * roughness
    reflected_radiance = (
        sharp * (1.0 - rough_blend) + broad * rough_blend
    ) * ambient_intensity
    environment_specular = fresnel_ambient * reflected_radiance

    irradiance = environment_diffuse
    specular = direct_specular + environment_specular
    final = (
        direct_diffuse
        + direct_specular
        + environment_diffuse
        + environment_specular
    )
    return irradiance, specular, final


def _write_sample_inputs(input_dir: Path) -> dict[str, np.ndarray]:
    input_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    y, x = np.mgrid[0:SAMPLE_SIZE, 0:SAMPLE_SIZE].astype(np.float64)
    xn = x / (SAMPLE_SIZE - 1)
    yn = y / (SAMPLE_SIZE - 1)
    noise = rng.random((SAMPLE_SIZE, SAMPLE_SIZE))

    albedo = np.stack(
        [
            np.clip(0.12 + 0.72 * xn + 0.08 * noise, 0.0, 1.0),
            np.clip(0.18 + 0.65 * yn + 0.08 * noise, 0.0, 1.0),
            np.clip(0.25 + 0.45 * (1.0 - xn) + 0.12 * noise, 0.0, 1.0),
        ],
        axis=2,
    )
    roughness = np.clip(0.08 + 0.78 * xn + 0.06 * noise, 0.0, 1.0)
    roughness[:, :16] = 0.0
    roughness[:, -16:] = 1.0
    metallic = np.clip(0.15 + 0.7 * yn + 0.08 * noise, 0.0, 1.0)
    metallic[:16, :] = 0.0
    metallic[-16:, :] = 1.0

    normal = np.stack(
        [
            0.55 * np.sin(5.0 * xn * math.pi) * np.cos(3.0 * yn * math.pi),
            0.55 * np.cos(3.5 * yn * math.pi) * np.sin(2.0 * xn * math.pi),
            np.ones_like(xn),
        ],
        axis=2,
    )
    normal, _ = normalize_vectors(normal)
    encoded_normal = np.clip(normal * 0.5 + 0.5, 0.0, 1.0)
    encoded_normal[23, 41] = (0.5, 0.5, 0.5)

    write_rgb_png(input_dir / "albedo.png", albedo, 8, 1.0)
    write_rgb_png(
        input_dir / "roughness.png", np.repeat(roughness[:, :, None], 3, axis=2), 8, 1.0
    )
    write_rgb_png(
        input_dir / "metallic.png", np.repeat(metallic[:, :, None], 3, axis=2), 16, 1.0
    )
    write_rgb_png(input_dir / "normal.png", encoded_normal, 16, 1.0)
    return {
        "albedo": albedo,
        "roughness": roughness,
        "metallic": metallic,
        "normal": encoded_normal,
    }


def _make_config(
    path: Path, *, resolution: int, bits: int, gamma: float = 2.2
) -> dict:
    config = {
        "resolution": resolution,
        "output_bits": bits,
        "output_gamma": gamma,
        "flip_green": True,
        "alpha_epsilon": 1.0e-3,
        "view_direction": [0.0, 0.0, 1.0],
        "directional_light": {
            "direction": [0.4, 0.5, 0.8],
            "color": [1.0, 0.9, 0.8],
            "intensity": 2.5,
        },
        "hemisphere_light": {
            "sky_color": [0.3, 0.55, 0.9],
            "ground_color": [0.12, 0.08, 0.06],
            "intensity": 0.8,
        },
    }
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def _make_invalid_depth_png(source: Path, destination: Path) -> None:
    data = bytearray(source.read_bytes())
    data[24] = 4
    payload = data[16:29]
    crc = zlib.crc32(b"IHDR")
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    data[29:33] = crc.to_bytes(4, "big")
    destination.write_bytes(bytes(data))


def _run_cli(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "material_baker", *arguments],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_close(
    actual: np.ndarray, expected: np.ndarray, message: str
) -> float:
    difference = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    if difference > THEORY_TOLERANCE:
        raise AssertionError(f"{message}; max difference {difference:.8g}")
    return difference


def _assert_code(operation: Callable[[], object], expected_code: str) -> None:
    try:
        operation()
    except ToolError as exc:
        if exc.code != expected_code:
            raise AssertionError(
                f"expected {expected_code}, got {exc.code}: {exc.message}"
            ) from exc
    else:
        raise AssertionError(f"expected {expected_code}, but no error was raised")


def run_selftest(work_dir: str | Path, *, keep: bool = False) -> int:
    root = Path(work_dir)
    if root.exists():
        if not root.is_dir():
            raise ToolError("E_IO", f"Self-test work path is not a directory: {root}")
        try:
            contents = list(root.iterdir())
        except OSError as exc:
            raise ToolError("E_IO", f"Cannot inspect self-test work dir {root}: {exc}") from exc
        if contents and not (root / WORK_DIR_MARKER).exists():
            raise ToolError(
                "E_IO",
                f"Refusing to overwrite non-empty non-selftest directory: {root}",
            )
        shutil.rmtree(root)
    input_dir = root / "inputs"
    output_256 = root / "out_256"
    output_128 = root / "out_128"
    output_512 = root / "out_512"
    root.mkdir(parents=True, exist_ok=True)
    (root / WORK_DIR_MARKER).touch()

    failures: list[tuple[str, str]] = []
    passes = 0
    theory_max_difference = 0.0

    def check(name: str, operation: Callable[[], None]) -> None:
        nonlocal passes
        try:
            operation()
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"[FAIL] {name}: {exc}")
        else:
            passes += 1
            print(f"[PASS] {name}")

    print("material-baker deterministic self-test")
    print(f"seed={SEED} sample_size={SAMPLE_SIZE}")

    try:
        _write_sample_inputs(input_dir)
    except Exception as exc:
        print(f"[FAIL] sample generation: {exc}")
        print("SELFTEST FAILED: 1 check(s) failed")
        return 1

    def normal_decode_check() -> None:
        encoded = np.asarray(
            [
                [[0.5, 0.5, 1.0], [0.5, 0.0, 0.5]],
                [[0.5, 0.5, 0.5], [0.75, 0.25, 0.5]],
            ],
            dtype=np.float64,
        )
        decoded, zero_count = decode_normal_map(encoded, flip_green=False)
        if zero_count != 1:
            raise AssertionError(f"zero-length normal count is {zero_count}, expected 1")
        expected_up = np.asarray([0.0, 0.0, 1.0])
        if np.max(np.abs(decoded[0, 0] - expected_up)) > 1.0e-12:
            raise AssertionError("neutral normal did not decode to (0,0,1)")
        if decoded[0, 1, 1] > -0.99:
            raise AssertionError("Y-up normal did not decode green=0 to y=-1")
        flipped, _ = decode_normal_map(encoded, flip_green=True)
        if flipped[0, 1, 1] < 0.99:
            raise AssertionError("flip_green did not invert the green/Y channel")
        if np.max(np.abs(decoded[1, 0] - expected_up)) > 1.0e-12:
            raise AssertionError("zero normal did not fall back to (0,0,1)")

    def extremes_check() -> None:
        albedo = np.asarray([[[0.9, 0.5, 0.2]]], dtype=np.float64)
        normal = np.asarray([[[0.0, 0.0, 1.0]]], dtype=np.float64)
        rough = np.asarray([[0.0]], dtype=np.float64)
        metal = np.asarray([[1.0]], dtype=np.float64)
        config = load_config(root / "config.json")
        metal_result = evaluate_material(
            albedo,
            rough,
            metal,
            normal,
            config.view_direction,
            config.directional,
            config.hemisphere,
            alpha_epsilon=config.alpha_epsilon,
        )
        if np.max(np.abs(metal_result.direct_diffuse)) > 1.0e-12:
            raise AssertionError("pure metal has non-zero diffuse response")
        if np.max(np.abs(metal_result.environment_diffuse)) > 1.0e-12:
            raise AssertionError("pure metal has non-zero environment diffuse")
        if not np.all(np.isfinite(metal_result.final_color)):
            raise AssertionError("zero-roughness metal produced non-finite values")

        dielectric = evaluate_material(
            albedo,
            rough,
            np.asarray([[0.0]], dtype=np.float64),
            normal,
            config.view_direction,
            config.directional,
            config.hemisphere,
            alpha_epsilon=config.alpha_epsilon,
        )
        if float(dielectric.direct_diffuse.max()) <= 0.0:
            raise AssertionError("dielectric should have a diffuse response")
        if float(metal_result.environment_specular.max()) <= float(
            dielectric.environment_specular.max()
        ):
            raise AssertionError(
                "bright metal should have stronger environment specular than dielectric"
            )

    def resample_check() -> None:
        constant = np.full((16, 16, 3), 0.25, dtype=np.float64)
        reduced = resize_image(constant, 8, 8)
        if np.max(np.abs(reduced - 0.25)) > 1.0e-12:
            raise AssertionError("constant roughness/metallic area average changed value")
        rng = np.random.default_rng(SEED + 1)
        vectors = rng.normal(size=(16, 16, 3))
        vectors, _ = normalize_vectors(vectors)
        resized = resize_image(vectors, 8, 8)
        resized, zero_count = normalize_vectors(resized)
        lengths = np.linalg.norm(resized, axis=2)
        if zero_count != 0 or np.max(np.abs(lengths - 1.0)) > 1.0e-10:
            raise AssertionError("resampled normals were not re-normalized")

    def grayscale_16bit_check() -> None:
        try:
            from PIL import Image
        except ImportError as exc:
            raise AssertionError("Pillow is required to generate this test sample") from exc
        path = root / "gray16.png"
        source = np.arange(256 * 32, dtype=np.uint32).reshape(32, 256) % 65536
        source = source.astype(np.uint16)
        Image.fromarray(source).save(path)
        image = read_png_rgb(path)
        if image.bit_depth != 16 or image.width != 256 or image.height != 32:
            raise AssertionError("16-bit grayscale PNG dimensions/depth were misread")
        expected = source.astype(np.float64) / 65535.0
        if np.max(np.abs(image.rgb[:, :, 0] - expected)) > 1.0e-12:
            raise AssertionError("16-bit grayscale sample values were decoded incorrectly")
        if not (
            np.array_equal(image.rgb[:, :, 0], image.rgb[:, :, 1])
            and np.array_equal(image.rgb[:, :, 1], image.rgb[:, :, 2])
        ):
            raise AssertionError("grayscale PNG was not replicated to RGB")

    _make_config(root / "config.json", resolution=256, bits=16)
    _make_config(root / "config_128.json", resolution=128, bits=8)
    _make_config(root / "config_512.json", resolution=512, bits=8, gamma=1.0)

    check("normal decoding, flip_green, zero fallback", normal_decode_check)
    check("zero roughness and metal/dielectric extremes", extremes_check)
    check("area-average and normal re-normalization", resample_check)
    check("16-bit grayscale PNG input support", grayscale_16bit_check)

    warnings = io.StringIO()
    bake_ok = True
    try:
        run_bake(
            root / "config.json",
            input_dir=input_dir,
            output_dir=output_256,
            stderr=warnings,
        )
    except Exception as exc:
        bake_ok = False
        failures.append(("256x256 16-bit bake", str(exc)))
        print(f"[FAIL] 256x256 16-bit bake: {exc}")
    else:
        passes += 1
        print("[PASS] 256x256 16-bit bake with zero-normal warning")

    if not bake_ok:
        _finish_failure(root, keep, failures)
        return 1

    try:
        config = load_config(root / "config.json")
        metadata = json.loads((output_256 / "metadata.json").read_text(encoding="utf-8"))
        loaded = {
            name: read_png_rgb(input_dir / f"{name}.png")
            for name in ("albedo", "roughness", "metallic", "normal")
        }
        outputs = {
            name: read_png_rgb(output_256 / metadata["output"]["images"][name])
            for name in ("irradiance", "specular", "final_color")
        }
    except Exception as exc:
        failures.append(("output loading", str(exc)))
        print(f"[FAIL] output loading: {exc}")
        _finish_failure(root, keep, failures)
        return 1

    def output_format_check() -> None:
        if "WARNING:" not in warnings.getvalue():
            raise AssertionError("zero-length normal warning was not written to stderr")
        if metadata["output"]["width"] != SAMPLE_SIZE or metadata["output"]["height"] != SAMPLE_SIZE:
            raise AssertionError("metadata dimensions do not match requested output")
        if metadata["output_gamma"] != 2.2 or metadata["output"]["gamma"] != 2.2:
            raise AssertionError("metadata gamma is inconsistent")
        if metadata["output_bits"] != 16:
            raise AssertionError("metadata bit depth is inconsistent")
        if metadata["warnings"]["zero_length_normals"] != 1:
            raise AssertionError("metadata zero-normal warning count is inconsistent")
        for name in ("irradiance", "specular", "final_color"):
            image = outputs[name]
            if image.width != SAMPLE_SIZE or image.height != SAMPLE_SIZE:
                raise AssertionError(f"{name} has incorrect dimensions")
            if image.bit_depth != 16:
                raise AssertionError(f"{name} is not 16-bit")
            if np.min(image.rgb) < 0.0 or np.max(image.rgb) > 1.0:
                raise AssertionError(f"{name} values are outside [0,1]")
        for filename in metadata["output"]["images"].values():
            path = output_256 / filename
            try:
                from PIL import Image
            except ImportError:
                continue
            with Image.open(path) as image:
                image.verify()
                if image.format != "PNG":
                    raise AssertionError(f"Pillow did not identify {filename} as PNG")
        if not all(loaded[name].bit_depth in (8, 16) for name in loaded):
            raise AssertionError("input PNG bit-depth metadata is invalid")

    def metadata_lighting_check() -> None:
        directional = metadata["lighting"]["directional_light"]
        hemisphere = metadata["lighting"]["hemisphere_light"]
        if directional["intensity"] != 2.5:
            raise AssertionError("directional intensity is inconsistent")
        if directional["color"] != [1.0, 0.9, 0.8]:
            raise AssertionError("directional color is inconsistent")
        if not math.isclose(
            float(np.linalg.norm(directional["direction"])), 1.0, abs_tol=1e-12
        ):
            raise AssertionError("stored light direction is not normalized")
        if hemisphere["sky_color"] != [0.3, 0.55, 0.9]:
            raise AssertionError("hemisphere sky color is inconsistent")
        if hemisphere["ground_color"] != [0.12, 0.08, 0.06]:
            raise AssertionError("hemisphere ground color is inconsistent")
        if metadata["flip_green"] is not True:
            raise AssertionError("flip_green metadata is inconsistent")

    def theory_check() -> None:
        nonlocal theory_max_difference
        rng = np.random.default_rng(SEED + 2)
        coordinates = [
            (0, 0),
            (0, SAMPLE_SIZE - 1),
            (SAMPLE_SIZE - 1, 0),
            (SAMPLE_SIZE - 1, SAMPLE_SIZE - 1),
            (23, 41),
            (64, 64),
            (127, 191),
            (255, 1),
        ]
        random_coordinates = rng.integers(0, SAMPLE_SIZE, size=(8, 2))
        coordinates.extend(tuple(int(v) for v in row) for row in random_coordinates)
        albedo_source = loaded["albedo"].rgb
        roughness_source = loaded["roughness"].rgb[:, :, 0]
        metallic_source = loaded["metallic"].rgb[:, :, 0]
        normal_source = loaded["normal"].rgb
        light = np.asarray(config.directional.direction)
        color = np.asarray(config.directional.color)
        sky = np.asarray(config.hemisphere.sky_color)
        ground = np.asarray(config.hemisphere.ground_color)
        view = np.asarray(config.view_direction)

        for y, x in coordinates:
            irradiance, specular, final = _reference_pixel(
                albedo_source[y, x],
                roughness_source[y, x],
                metallic_source[y, x],
                normal_source[y, x],
                flip_green=config.flip_green,
                view=view,
                light_direction=light,
                light_color=color,
                light_intensity=config.directional.intensity,
                sky=sky,
                ground=ground,
                ambient_intensity=config.hemisphere.intensity,
                alpha_epsilon=config.alpha_epsilon,
            )
            expected = {
                "irradiance": _encode_linear(irradiance, config.output_gamma),
                "specular": _encode_linear(specular, config.output_gamma),
                "final_color": _encode_linear(final, config.output_gamma),
            }
            for name in expected:
                difference = _assert_close(
                    outputs[name].rgb[y, x],
                    expected[name],
                    f"{name} theory mismatch at ({x},{y})",
                )
                theory_max_difference = max(theory_max_difference, difference)

    def energy_check() -> None:
        result = evaluate_material(
            loaded["albedo"].rgb,
            loaded["roughness"].rgb[:, :, 0],
            loaded["metallic"].rgb[:, :, 0],
            decode_normal_map(
                loaded["normal"].rgb, flip_green=config.flip_green
            )[0],
            config.view_direction,
            config.directional,
            config.hemisphere,
            alpha_epsilon=config.alpha_epsilon,
        )
        light = np.asarray(config.directional.direction)
        normal = decode_normal_map(
            loaded["normal"].rgb, flip_green=config.flip_green
        )[0]
        ndot_l = np.clip(np.sum(normal * light, axis=-1), 0.0, 1.0)[..., None]
        incident = (
            np.asarray(config.directional.color)
            * config.directional.intensity
            * ndot_l
        )
        total = result.direct_diffuse + result.direct_specular
        excess = float(np.max(total - incident))
        if excess > 1.0e-10:
            raise AssertionError(f"direct energy exceeds incident energy by {excess}")
        if not np.all(np.isfinite(result.final_color)):
            raise AssertionError("bake result contains non-finite values")

    check("PNG format, dimensions, gamma metadata, Pillow validation", output_format_check)
    check("metadata lighting parameters", metadata_lighting_check)
    check("independent sampled-pixel theory comparison", theory_check)
    check("directional diffuse+specular energy bound", energy_check)

    def downsample_check() -> None:
        warnings_128 = io.StringIO()
        summary = run_bake(
            root / "config_128.json",
            input_dir=input_dir,
            output_dir=output_128,
            stderr=warnings_128,
        )
        meta = json.loads((output_128 / "metadata.json").read_text(encoding="utf-8"))
        if summary["resolution"] != 128 or summary["bit_depth"] != 8:
            raise AssertionError("128 output summary is inconsistent")
        if meta["output_gamma"] != 2.2 or meta["output"]["gamma"] != 2.2:
            raise AssertionError("128 output metadata gamma is inconsistent")
        for name in ("irradiance", "specular", "final_color"):
            image = read_png_rgb(output_128 / meta["output"]["images"][name])
            if image.width != 128 or image.height != 128:
                raise AssertionError(f"downsampled {name} has incorrect dimensions")
            if image.bit_depth != 8:
                raise AssertionError(f"downsampled {name} is not 8-bit")
            if not np.all(np.isfinite(image.rgb)):
                raise AssertionError(f"downsampled {name} contains non-finite values")
            if np.min(image.rgb) < 0.0 or np.max(image.rgb) > 1.0:
                raise AssertionError(f"downsampled {name} is outside [0,1]")

    check("Gaussian downsample to 128 and 8-bit output", downsample_check)

    def upscale_gamma_check() -> None:
        summary = run_bake(
            root / "config_512.json",
            input_dir=input_dir,
            output_dir=output_512,
            stderr=io.StringIO(),
        )
        meta = json.loads((output_512 / "metadata.json").read_text(encoding="utf-8"))
        if summary["resolution"] != 512 or summary["gamma"] != 1.0:
            raise AssertionError("512 gamma=1 output summary is inconsistent")
        if meta["output_gamma"] != 1.0 or meta["output"]["gamma"] != 1.0:
            raise AssertionError("512 output metadata gamma is inconsistent")
        for name in ("irradiance", "specular", "final_color"):
            image = read_png_rgb(output_512 / meta["output"]["images"][name])
            if image.width != 512 or image.height != 512 or image.bit_depth != 8:
                raise AssertionError(
                    f"upscaled {name} does not match 512x512/8-bit metadata"
                )
            if np.min(image.rgb) < 0.0 or np.max(image.rgb) > 1.0:
                raise AssertionError(f"upscaled {name} is outside [0,1]")

    check("512 upscale and gamma=1.0 output", upscale_gamma_check)

    def config_error_check() -> None:
        bad = json.loads((root / "config.json").read_text(encoding="utf-8"))
        bad["output_gamma"] = 1.8
        bad_path = root / "bad_gamma.json"
        bad_path.write_text(json.dumps(bad), encoding="utf-8")
        _assert_code(lambda: load_config(bad_path), "E_INVALID_CONFIG")
        bad["output_gamma"] = 2.2
        bad["resolution"] = 100
        bad_path.write_text(json.dumps(bad), encoding="utf-8")
        _assert_code(lambda: load_config(bad_path), "E_INVALID_CONFIG")

    def missing_file_cli_check() -> None:
        completed = _run_cli(
            [
                "bake",
                str(root / "config.json"),
                "--input",
                str(root / "does_not_exist"),
                "--output",
                str(root / "missing_output"),
            ]
        )
        if completed.returncode == 0:
            raise AssertionError("missing input did not return a non-zero exit code")
        if "E_MISSING_FILE" not in completed.stderr:
            raise AssertionError("missing input did not report E_MISSING_FILE on stderr")

    def invalid_format_cli_check() -> None:
        bad_input = root / "bad_inputs"
        shutil.copytree(input_dir, bad_input)
        _make_invalid_depth_png(
            bad_input / "normal.png", bad_input / "normal.png"
        )
        completed = _run_cli(
            [
                "bake",
                str(root / "config.json"),
                "--input",
                str(bad_input),
                "--output",
                str(root / "bad_output"),
            ]
        )
        if completed.returncode == 0:
            raise AssertionError("4-bit PNG did not return a non-zero exit code")
        if "E_INVALID_FORMAT" not in completed.stderr:
            raise AssertionError("4-bit PNG did not report E_INVALID_FORMAT on stderr")

    def mismatch_format_check() -> None:
        bad_input = root / "mismatch_inputs"
        shutil.copytree(input_dir, bad_input)
        wrong = np.zeros((64, 64, 3), dtype=np.float64)
        write_rgb_png(bad_input / "normal.png", wrong, 8, 1.0)
        _assert_code(
            lambda: run_bake(
                root / "config.json",
                input_dir=bad_input,
                output_dir=root / "mismatch_output",
                stderr=io.StringIO(),
            ),
            "E_INVALID_FORMAT",
        )

    check("invalid configuration rejection", config_error_check)
    check("CLI missing file exit code and stderr code", missing_file_cli_check)
    check("CLI invalid PNG depth exit code and stderr code", invalid_format_cli_check)
    check("mismatched input dimensions rejection", mismatch_format_check)

    if failures:
        _finish_failure(root, keep, failures)
        return 1

    print(
        f"SELFTEST PASSED: {passes} checks; seed={SEED}; "
        f"theory_tolerance={THEORY_TOLERANCE:g}; "
        f"max_theory_delta={theory_max_difference:.8g}"
    )
    if not keep:
        shutil.rmtree(root, ignore_errors=True)
    else:
        print(f"Artifacts kept in {root.resolve()}")
    return 0


def _finish_failure(
    root: Path, keep: bool, failures: list[tuple[str, str]]
) -> None:
    print("SELFTEST FAILED", file=sys.stdout)
    for name, message in failures:
        print(f"FAILED ITEM: {name}: {message}", file=sys.stderr)
    print(f"SELFTEST FAILED: {len(failures)} check(s) failed")
    if keep:
        print(f"Artifacts kept in {root.resolve()}")
    else:
        shutil.rmtree(root, ignore_errors=True)

