"""Render orchestration, exact checkpoint resume, statistics, and reporting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .checkpoint import Checkpoint, RenderState
from .errors import business_error
from .integrator import Renderer
from .output import (
    atomic_write_bytes,
    auxiliary_ppm,
    sha256_bytes,
    stable_json_bytes,
    tonemap_ppm,
)
from .scene import Scene
from .spectral import RGB_RESPONSE_LOBES


@dataclass(frozen=True)
class RenderConfig:
    width: int
    height: int
    spp: int
    seed: int
    tile: int
    max_depth: int
    use_indirect: bool
    caustic_photons: int
    caustic_map_resolution: int = 192
    exposure: float = 1.35


def automatic_caustic_photons(width: int, height: int) -> int:
    return max(150_000, min(1_200_000, width * height * 4))


def _checkpoint_fingerprint(scene: Scene, config: RenderConfig) -> str:
    payload = {
        "scene_sha256": scene.fingerprint(),
        "width": config.width,
        "height": config.height,
        "spp": config.spp,
        "seed": config.seed,
        "tile": config.tile,
        "max_depth": config.max_depth,
        "use_indirect": config.use_indirect,
        "caustic_photons": config.caustic_photons,
        "caustic_map_resolution": config.caustic_map_resolution,
        "exposure": config.exposure,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _output_stem(config: RenderConfig) -> str:
    indirect = "" if config.use_indirect else "_noindirect"
    return (
        f"prism_{config.width}x{config.height}_spp{config.spp}"
        f"_seed{config.seed}_d{config.max_depth}{indirect}"
    )


def _channel_statistics(
    sums: np.ndarray,
    sums_squared: np.ndarray,
    direct_sums: np.ndarray,
    indirect_sums: np.ndarray,
    spp: int,
) -> dict[str, dict[str, float]]:
    means = sums / spp
    sample_variance = np.maximum(sums_squared / spp - means * means, 0.0)
    direct_means = direct_sums / spp
    indirect_means = indirect_sums / spp
    labels = ("R", "G", "B")
    result: dict[str, dict[str, float]] = {}
    for channel, label in enumerate(labels):
        pixel_means = means[..., channel]
        pixel_variance = sample_variance[..., channel]
        mean_value = float(np.mean(pixel_means))
        variance_value = float(np.mean(pixel_variance))
        direct_mean = float(np.mean(direct_means[..., channel]))
        indirect_mean = float(np.mean(indirect_means[..., channel]))
        estimate_stddev = math.sqrt(max(variance_value / spp, 0.0))
        result[label] = {
            "mean": mean_value,
            "variance": variance_value,
            "stddev": math.sqrt(max(variance_value, 0.0)),
            "estimate_stddev": estimate_stddev,
            "relative_noise": estimate_stddev / max(abs(mean_value), 1.0e-12),
            "pixel_mean_variance": float(np.var(pixel_means)),
            "minimum_pixel_mean": float(np.min(pixel_means)),
            "maximum_pixel_mean": float(np.max(pixel_means)),
            "direct_mean": direct_mean,
            "indirect_mean": indirect_mean,
            "energy_closure_error": abs(
                mean_value - (direct_mean + indirect_mean)
            ),
        }
    return result


def _build_report(
    scene: Scene,
    config: RenderConfig,
    state: RenderState,
    renderer: Renderer | None,
    checkpoint_fingerprint: str,
    stem: str,
    image_bytes: bytes,
    auxiliary_bytes: bytes,
) -> dict:
    stats = _channel_statistics(
        state.sum_rgb,
        state.sum_sq_rgb,
        state.sum_direct_rgb,
        state.sum_indirect_rgb,
        config.spp,
    )
    aux_energy = np.maximum(np.sum(state.aux_contribution, axis=(0, 1)), 0.0)
    aux_hits = np.maximum(np.sum(state.aux_hits, axis=(0, 1)), 0.0)
    aux_total = float(np.sum(aux_energy))
    if aux_total > 0.0:
        fractions = (aux_energy / aux_total).tolist()
    else:
        fractions = [0.0, 0.0, 0.0]

    if renderer is not None and renderer.caustic_map is not None:
        cmap = renderer.caustic_map
        caustic = {
            "photons_emitted": int(cmap.photons_emitted),
            "photons_deposited_refractive": int(cmap.photons_deposited_refract),
            "photons_deposited_tir": int(cmap.photons_deposited_tir),
            "glass_transmission_events": int(cmap.photons_with_glass_transmission),
            "map_resolution": int(cmap.resolution),
            "smoothing_passes": int(cmap.smoothing_passes),
            "ground_extent_x": [float(cmap.x0), float(cmap.x1)],
            "ground_extent_z": [float(cmap.z0), float(cmap.z1)],
            "normalization": "photon power per map cell divided by cell area",
            "double_counting_control": (
                "NEE treats glass as an occluder; the photon map contains only "
                "paths that transmitted through glass before the first diffuse hit"
            ),
        }
    else:
        caustic = {
            "photons_emitted": 0,
            "photons_deposited_refractive": 0,
            "photons_deposited_tir": 0,
            "glass_transmission_events": 0,
            "map_resolution": config.caustic_map_resolution,
            "smoothing_passes": 0,
            "normalization": "photon power per map cell divided by cell area",
            "double_counting_control": "no photon map contribution",
        }

    assert scene.spectral is not None
    white_error = scene.spectral.white_error().tolist()
    report = {
        "schema_version": 1,
        "scene": "prism",
        "resolution": {"width": config.width, "height": config.height},
        "samples_per_pixel": config.spp,
        "seed": config.seed,
        "tile": config.tile,
        "max_path_depth": config.max_depth,
        "indirect_enabled": config.use_indirect,
        "spectral_sampling": {
            "range_nm": [scene.min_nm, scene.max_nm],
            "method": "uniform hero-wavelength Monte Carlo sampling",
            "rgb_conversion": (
                "three fixed smooth positive RGB response lobes, each normalized "
                "so equal-energy white integrates to (1, 1, 1)"
            ),
            "rgb_response_lobes_nm": [
                {"center": center, "width": width}
                for center, width in RGB_RESPONSE_LOBES
            ],
            "fixed_weight_white_error_rgb": white_error,
        },
        "materials": {
            "glass": {
                "ior_table": scene.data["prism"]["ior"],
                "absorption_per_m_table": scene.data["prism"]["absorption"],
            },
            "water": {
                "ior_table": scene.data["water"]["ior"],
                "absorption_per_m_table": scene.data["water"]["absorption"],
            },
            "air_ior": 1.0,
        },
        "light": {
            "shape": "rectangle",
            "center": scene.data["light"]["center"],
            "u": scene.data["light"]["u"],
            "v": scene.data["light"]["v"],
            "area": float(scene.light_area),
            "spectral_emission": float(scene.light_emission),
            "sampling": "uniform area samples, independent RNG stream from BSDF sampling",
        },
        "integrator": {
            "camera": "backward path tracing, cosine-weighted diffuse bounces",
            "direct_lighting": (
                "one stratified area-light sample per diffuse hit (NEE); after a "
                "diffuse vertex, direct BSDF-emitter hits already covered by NEE "
                "or the caustic map are suppressed"
            ),
            "specular": "smooth unpolarized Fresnel reflection/refraction with TIR",
            "fresnel_branch": (
                "reflection sampled with probability F, transmission with 1-F; "
                "weights are not multiplied by the chosen Fresnel probability"
            ),
            "radiance_transmission": "eta^2 factor at refractive radiance transport",
            "caustics": "deterministic forward glass-path photon map",
            "indirect_policy": (
                "indirect radiance continues after cosine-weighted diffuse bounces; "
                "glass-path direct caustics are queried from the photon map at every "
                "diffuse hit"
            ),
            "max_photon_depth": 16,
        },
        "caustic_map": caustic,
        "channel_stats": stats,
        "noise": {
            label: {
                "mean": values["mean"],
                "variance": values["variance"],
                "estimate_stddev": values["estimate_stddev"],
                "relative_noise": values["relative_noise"],
            }
            for label, values in stats.items()
        },
        "path_type_aux": {
            "channel_legend": {
                "R": "direct diffuse transport",
                "G": "refractive transmission through glass",
                "B": "total-internal-reflection transmission chain",
            },
            "energy_share": dict(zip(("direct_diffuse", "refractive", "tir"), fractions)),
            "sample_class_hits": {
                "direct_diffuse": int(round(aux_hits[0])),
                "refractive": int(round(aux_hits[1])),
                "tir": int(round(aux_hits[2])),
            },
            "channel_normalization": "each AOV channel independently scaled to its own maximum",
        },
        "tone_mapping": {
            "exposure": config.exposure,
            "operator": "x/(1+x)",
            "encoding": "sRGB transfer",
            "output_bit_depth": 8,
        },
        "determinism": {
            "random_generator": (
                "counter-based SplitMix64, keyed per pixel/sample/event; independent "
                "per-pixel strata permutations decorrelate wavelength/light/material axes"
            ),
            "tile_independent_samples": True,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "scene_sha256": scene.fingerprint(),
            "report_contains_no_timestamps": True,
        },
        "outputs": {
            "image": f"{stem}.ppm",
            "auxiliary": f"{stem}_aux.ppm",
            "report": f"{stem}_report.json",
        },
        "hashes": {
            "algorithm": "sha256",
            "image_sha256": sha256_bytes(image_bytes),
            "auxiliary_sha256": sha256_bytes(auxiliary_bytes),
        },
    }
    return report


def run_pipeline(
    scene: Scene,
    config: RenderConfig,
    out_dir: Path,
    *,
    stop_after_tiles: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Run or resume a render.  This function intentionally has no timestamps."""

    assert scene.spectral is not None
    stem = _output_stem(config)
    fingerprint = _checkpoint_fingerprint(scene, config)
    checkpoint = Checkpoint(
        out_dir / f"{stem}.partial-{fingerprint[:12]}.npz", fingerprint
    )
    renderer = Renderer(
        scene,
        config.width,
        config.height,
        config.spp,
        config.seed,
        config.tile,
        config.max_depth,
        config.use_indirect,
        config.caustic_photons,
        config.caustic_map_resolution,
    )

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        state = checkpoint.load_or_create(config.width, config.height, config.tile)
        already_complete = bool(np.all(state.completed))
        if not already_complete:
            if progress:
                progress("building caustic map")
            renderer.build_caustic_map()
            if progress and renderer.caustic_map is not None:
                progress(
                    "caustic map: "
                    f"{renderer.caustic_map.photons_deposited_refract} refractive + "
                    f"{renderer.caustic_map.photons_deposited_tir} TIR deposits"
                )

            processed_now = 0
            try:
                for tile_y in range(state.tiles_y):
                    for tile_x in range(state.tiles_x):
                        if state.completed[tile_y, tile_x]:
                            continue
                        tile_data = renderer.render_tile(tile_x, tile_y)
                        state.add_tile(tile_x, tile_y, tile_data)
                        checkpoint.save(state)
                        processed_now += 1
                        if progress:
                            completed = int(np.count_nonzero(state.completed))
                            progress(
                                f"tile {completed}/{state.tiles_x * state.tiles_y} checkpointed"
                            )
                        if (
                            stop_after_tiles is not None
                            and processed_now >= stop_after_tiles
                            and not np.all(state.completed)
                        ):
                            raise KeyboardInterrupt("intentional test interruption")
            except KeyboardInterrupt:
                checkpoint.save(state)
                raise

        if not np.all(state.completed):
            raise business_error("E_CHECKPOINT", "render stopped before all tiles completed")

        # A checkpoint can be complete if the process was interrupted between
        # final atomic file replacements. Rebuild the map so the report is
        # byte-identical to a non-resumed run.
        if renderer.caustic_map is None:
            renderer.build_caustic_map()

        means = state.sum_rgb / config.spp
        image_bytes = tonemap_ppm(means, config.exposure)
        auxiliary_bytes = auxiliary_ppm(state.aux_contribution)
        report = _build_report(
            scene,
            config,
            state,
            renderer,
            fingerprint,
            stem,
            image_bytes,
            auxiliary_bytes,
        )
        report_bytes = stable_json_bytes(report)

        image_path = out_dir / f"{stem}.ppm"
        auxiliary_path = out_dir / f"{stem}_aux.ppm"
        report_path = out_dir / f"{stem}_report.json"
        atomic_write_bytes(image_path, image_bytes)
        atomic_write_bytes(auxiliary_path, auxiliary_bytes)
        atomic_write_bytes(report_path, report_bytes)
        checkpoint.remove()
        return report
    except OSError as exc:
        raise business_error("E_IO", f"output I/O failed: {exc}") from exc

