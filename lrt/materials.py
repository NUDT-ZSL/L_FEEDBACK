"""Layered dielectric coating and diffuse/metal base optics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .geometry import Vec3, dot


@dataclass(frozen=True)
class BaseMaterial:
    kind: str
    color: Vec3
    roughness: float = 0.0


@dataclass(frozen=True)
class Coating:
    ior: float
    thickness: float
    tint: Vec3 = (1.0, 1.0, 1.0)
    absorption: Vec3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Material:
    name: str
    kind: str
    base: BaseMaterial
    coating: Coating


def fresnel_dielectric(cos_incident: float, eta_incident: float, eta_transmitted: float) -> float:
    """Unpolarized Fresnel reflectance for a dielectric interface."""

    cos_i = max(0.0, min(1.0, float(cos_incident)))
    if eta_incident <= 0.0 or eta_transmitted <= 0.0:
        return 1.0
    eta = eta_incident / eta_transmitted
    sin_t2 = eta * eta * max(0.0, 1.0 - cos_i * cos_i)
    if sin_t2 >= 1.0:
        return 1.0
    cos_t = math.sqrt(max(0.0, 1.0 - sin_t2))
    rs_num = eta_incident * cos_i - eta_transmitted * cos_t
    rs_den = eta_incident * cos_i + eta_transmitted * cos_t
    rp_num = eta_incident * cos_t - eta_transmitted * cos_i
    rp_den = eta_incident * cos_t + eta_transmitted * cos_i
    if abs(rs_den) <= 1.0e-15 or abs(rp_den) <= 1.0e-15:
        return 1.0
    rs = rs_num / rs_den
    rp = rp_num / rp_den
    return max(0.0, min(1.0, 0.5 * (rs * rs + rp * rp)))


def refract(
    incident: Vec3, normal: Vec3, eta_incident: float, eta_transmitted: float
) -> Tuple[bool, Vec3]:
    """Refract a unit direction. ``normal`` must face the incident ray."""

    cos_i = max(0.0, min(1.0, -dot(incident, normal)))
    eta = eta_incident / eta_transmitted
    sin_t2 = eta * eta * max(0.0, 1.0 - cos_i * cos_i)
    if sin_t2 > 1.0:
        return False, (
            incident[0] + 2.0 * cos_i * normal[0],
            incident[1] + 2.0 * cos_i * normal[1],
            incident[2] + 2.0 * cos_i * normal[2],
        )
    cos_t = math.sqrt(max(0.0, 1.0 - sin_t2))
    scale = eta * cos_i - cos_t
    return True, (
        eta * incident[0] + scale * normal[0],
        eta * incident[1] + scale * normal[1],
        eta * incident[2] + scale * normal[2],
    )


def coating_fresnel_and_transmission(
    material: Material, cos_incident: float
) -> Tuple[float, Vec3]:
    """Return top-interface Fresnel reflectance and base-branch transmission.

    The branch transmission is the probability-conditional factor used after a
    base branch has been selected. It already includes the exit-interface
    Fresnel factor, Beer-Lambert attenuation through the stated layer
    thickness, and the coating tint:

        base_gain = (1 - F) * exp(-absorption * thickness / cos_i) * tint

    When multiplied by the base reflectance ``Rb``, the directional effective
    reflectance is ``F + (1 - F)^2 * exp(...) * tint * Rb``, which cannot exceed
    1 for validated inputs.
    """

    cos_i = max(1.0e-5, min(1.0, float(cos_incident)))
    fresnel = fresnel_dielectric(cos_i, 1.0, material.coating.ior)
    path_length = material.coating.thickness / cos_i
    attenuation = (
        math.exp(-material.coating.absorption[0] * path_length),
        math.exp(-material.coating.absorption[1] * path_length),
        math.exp(-material.coating.absorption[2] * path_length),
    )
    base_gain = (
        (1.0 - fresnel) * attenuation[0] * material.coating.tint[0],
        (1.0 - fresnel) * attenuation[1] * material.coating.tint[1],
        (1.0 - fresnel) * attenuation[2] * material.coating.tint[2],
    )
    return fresnel, base_gain


def effective_layer_reflectance(
    material: Material, cos_incident: float
) -> Vec3:
    fresnel, base_gain = coating_fresnel_and_transmission(material, cos_incident)
    return (
        fresnel + base_gain[0] * material.base.color[0],
        fresnel + base_gain[1] * material.base.color[1],
        fresnel + base_gain[2] * material.base.color[2],
    )


def calibration_report(
    materials: Iterable[Material],
    camera_position: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Check the analytical layer composition over fixed angles and materials."""

    angles = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0)
    material_reports: List[Dict[str, object]] = []
    all_in_range = True
    maximum = 0.0
    minimum = 1.0
    warnings: List[str] = []
    for material in materials:
        rows: List[Dict[str, object]] = []
        for angle in angles:
            cos_i = max(1.0e-5, math.cos(math.radians(angle)))
            value = effective_layer_reflectance(material, cos_i)
            local_max = max(value)
            local_min = min(value)
            maximum = max(maximum, local_max)
            minimum = min(minimum, local_min)
            in_range = all(0.0 <= channel <= 1.0 for channel in value)
            all_in_range = all_in_range and in_range
            if not in_range:
                warnings.append(
                    "material %s at %.1f degrees has reflectance %r outside [0,1]"
                    % (material.name, angle, value)
                )
            rows.append(
                {
                    "angle_degrees": angle,
                    "reflectance_rgb": [round(channel, 12) for channel in value],
                    "in_range": in_range,
                }
            )
        material_reports.append({"name": material.name, "angles": rows})
    report: Dict[str, object] = {
        "mode": "fixed_camera_fixed_materials_analytical",
        "angles_degrees": list(angles),
        "all_in_range": all_in_range,
        "minimum_reflectance": round(minimum, 12),
        "maximum_reflectance": round(maximum, 12),
        "warnings": warnings,
        "materials": material_reports,
    }
    if camera_position is not None:
        report["fixed_camera_position"] = [float(value) for value in camera_position]
    return report


def energy_bounds(materials: Iterable[Material]) -> Tuple[float, float, Sequence[str]]:
    """Return the min/max effective reflectance without building a full report."""

    minimum = 1.0
    maximum = 0.0
    warnings: List[str] = []
    for material in materials:
        for angle in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0):
            cos_i = max(1.0e-5, math.cos(math.radians(angle)))
            value = effective_layer_reflectance(material, cos_i)
            minimum = min(minimum, min(value))
            maximum = max(maximum, max(value))
            if any(channel < 0.0 or channel > 1.0 for channel in value):
                warnings.append(
                    "material %s has out-of-range reflectance %r" % (material.name, value)
                )
    return minimum, maximum, warnings

