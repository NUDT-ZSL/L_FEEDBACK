"""Build the deterministic built-in character preview scene."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


class MeshBuilder:
    def __init__(self) -> None:
        self.vertices: list[list[float]] = []
        self.normals: list[list[float]] = []
        self.uvs: list[list[float]] = []
        self.groups: dict[str, list[int]] = {}

    def add_vertex(self, position: np.ndarray, normal: np.ndarray, uv: tuple[float, float]) -> int:
        index = len(self.vertices)
        self.vertices.append([float(value) for value in position])
        self.normals.append([float(value) for value in normal])
        self.uvs.append([float(uv[0]), float(uv[1])])
        return index

    def add_triangle(self, material: str, a: int, b: int, c: int) -> None:
        self.groups.setdefault(material, []).extend((a, b, c))

    def add_ellipsoid(
        self,
        material: str,
        center: tuple[float, float, float],
        radii: tuple[float, float, float],
        segments: int,
        rings: int,
    ) -> None:
        center_array = np.asarray(center, dtype=np.float64)
        radius_array = np.asarray(radii, dtype=np.float64)
        grid: list[list[int]] = []
        for ring in range(rings + 1):
            theta = math.pi * ring / rings
            row: list[int] = []
            for segment in range(segments + 1):
                phi = 2.0 * math.pi * segment / segments
                unit = np.array(
                    [
                        math.sin(theta) * math.cos(phi),
                        math.cos(theta),
                        math.sin(theta) * math.sin(phi),
                    ]
                )
                position = center_array + radius_array * unit
                normal = np.array(
                    [
                        unit[0] / radius_array[0],
                        unit[1] / radius_array[1],
                        unit[2] / radius_array[2],
                    ]
                )
                normal /= np.linalg.norm(normal)
                row.append(
                    self.add_vertex(
                        position,
                        normal,
                        (segment / segments, ring / rings),
                    )
                )
            grid.append(row)
        for ring in range(rings):
            for segment in range(segments):
                a = grid[ring][segment]
                b = grid[ring][segment + 1]
                c = grid[ring + 1][segment]
                d = grid[ring + 1][segment + 1]
                if ring != 0:
                    self.add_triangle(material, a, b, c)
                if ring != rings - 1:
                    self.add_triangle(material, b, d, c)

    def add_tube(
        self,
        material: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        start_radius: float,
        end_radius: float,
        segments: int,
        length_segments: int,
        caps: bool = True,
    ) -> None:
        start_array = np.asarray(start, dtype=np.float64)
        end_array = np.asarray(end, dtype=np.float64)
        axis_vector = end_array - start_array
        length = float(np.linalg.norm(axis_vector))
        if length <= 1e-9:
            raise ValueError("tube endpoints must differ")
        axis = axis_vector / length
        helper = np.array([0.0, 1.0, 0.0]) if abs(float(axis[1])) < 0.9 else np.array([1.0, 0.0, 0.0])
        tangent_u = np.cross(helper, axis)
        tangent_u /= np.linalg.norm(tangent_u)
        tangent_v = np.cross(axis, tangent_u)
        radius_slope = (end_radius - start_radius) / length

        rings: list[list[int]] = []
        for length_index in range(length_segments + 1):
            fraction = length_index / length_segments
            center = start_array + axis_vector * fraction
            radius = start_radius + (end_radius - start_radius) * fraction
            row: list[int] = []
            for segment in range(segments + 1):
                phi = 2.0 * math.pi * segment / segments
                radial = tangent_u * math.cos(phi) + tangent_v * math.sin(phi)
                normal = radial - axis * radius_slope
                normal /= np.linalg.norm(normal)
                row.append(
                    self.add_vertex(
                        center + radial * radius,
                        normal,
                        (segment / segments, fraction),
                    )
                )
            rings.append(row)
        for length_index in range(length_segments):
            for segment in range(segments):
                a = rings[length_index][segment]
                b = rings[length_index][segment + 1]
                c = rings[length_index + 1][segment]
                d = rings[length_index + 1][segment + 1]
                self.add_triangle(material, a, b, c)
                self.add_triangle(material, b, d, c)

        if caps:
            start_center = self.add_vertex(
                start_array, -axis, (0.5, 0.5)
            )
            end_center = self.add_vertex(
                end_array, axis, (0.5, 0.5)
            )
            for segment in range(segments):
                phi0 = 2.0 * math.pi * segment / segments
                phi1 = 2.0 * math.pi * (segment + 1) / segments
                radial0 = tangent_u * math.cos(phi0) + tangent_v * math.sin(phi0)
                radial1 = tangent_u * math.cos(phi1) + tangent_v * math.sin(phi1)
                start0 = self.add_vertex(
                    start_array + radial0 * start_radius,
                    -axis,
                    (0.5 + 0.5 * math.cos(phi0), 0.5 + 0.5 * math.sin(phi0)),
                )
                start1 = self.add_vertex(
                    start_array + radial1 * start_radius,
                    -axis,
                    (0.5 + 0.5 * math.cos(phi1), 0.5 + 0.5 * math.sin(phi1)),
                )
                end0 = self.add_vertex(
                    end_array + radial0 * end_radius,
                    axis,
                    (0.5 + 0.5 * math.cos(phi0), 0.5 + 0.5 * math.sin(phi0)),
                )
                end1 = self.add_vertex(
                    end_array + radial1 * end_radius,
                    axis,
                    (0.5 + 0.5 * math.cos(phi1), 0.5 + 0.5 * math.sin(phi1)),
                )
                self.add_triangle(material, start_center, start1, start0)
                self.add_triangle(material, end_center, end0, end1)

    def add_sphere_patch(
        self,
        material: str,
        center: tuple[float, float, float],
        radius: float,
        theta_start: float,
        theta_end: float,
        phi_start: float,
        phi_end: float,
        segments: int,
        rings: int,
    ) -> None:
        center_array = np.asarray(center, dtype=np.float64)
        grid: list[list[int]] = []
        for ring in range(rings + 1):
            theta = theta_start + (theta_end - theta_start) * ring / rings
            row: list[int] = []
            for segment in range(segments + 1):
                phi = phi_start + (phi_end - phi_start) * segment / segments
                unit = np.array(
                    [
                        math.sin(theta) * math.cos(phi),
                        math.cos(theta),
                        math.sin(theta) * math.sin(phi),
                    ]
                )
                row.append(
                    self.add_vertex(
                        center_array + unit * radius,
                        unit,
                        (segment / segments, ring / rings),
                    )
                )
            grid.append(row)
        for ring in range(rings):
            for segment in range(segments):
                a = grid[ring][segment]
                b = grid[ring][segment + 1]
                c = grid[ring + 1][segment]
                d = grid[ring + 1][segment + 1]
                self.add_triangle(material, a, b, c)
                self.add_triangle(material, b, d, c)

    def to_json(self) -> dict[str, object]:
        return {
            "vertices": self.vertices,
            "normals": self.normals,
            "uvs": self.uvs,
            "primitives": [
                {"material": material, "indices": indices}
                for material, indices in self.groups.items()
            ],
        }


def _skin_texture_base64() -> str:
    image = Image.new("RGB", (16, 16), (168, 103, 77))
    pixels = image.load()
    assert pixels is not None
    for y in range(16):
        for x in range(16):
            variation = 8 * math.sin((x + y) * 0.7)
            pixels[x, y] = (
                int(max(0, min(255, 168 + variation))),
                int(max(0, min(255, 103 + variation * 0.55))),
                int(max(0, min(255, 77 + variation * 0.35))),
            )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _material(
    name: str,
    base_color: list[float],
    diffuse_id: str,
    normal_id: str,
    spec_id: str,
    specular: float,
    shininess: float,
    *,
    opacity: float = 1.0,
    blend: str = "opaque",
    double_sided: bool = False,
    emission: list[float] | None = None,
    diffuse_spec: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "baseColor": base_color,
        "diffuseMap": diffuse_spec
        or {"kind": "generated", "id": diffuse_id, "width": 96, "height": 96},
        "normalMap": {"kind": "generated", "id": normal_id, "width": 96, "height": 96},
        "specularMap": {"kind": "generated", "id": spec_id, "width": 96, "height": 96},
        "specular": specular,
        "shininess": shininess,
        "opacity": opacity,
        "blend": blend,
        "doubleSided": double_sided,
        "normalStrength": 1.0,
        "emission": emission or [0.0, 0.0, 0.0],
    }


def build_mesh() -> dict[str, object]:
    mesh = MeshBuilder()
    mesh.add_tube("platform", (0.0, -0.02, 0.0), (0.0, 0.16, 0.0), 1.34, 1.34, 48, 2)
    mesh.add_tube("platform", (0.0, 0.16, 0.0), (0.0, 0.24, 0.0), 1.16, 1.16, 48, 1)
    mesh.add_ellipsoid("cloth", (0.0, 1.07, 0.0), (0.38, 0.29, 0.28), 24, 14)
    mesh.add_ellipsoid("armor", (0.0, 1.55, 0.0), (0.53, 0.56, 0.31), 32, 20)
    # Inner frame intentionally enclosed by the chest shell. Its front-facing
    # triangles are valid Hi-Z candidates: depth is retained by the prepass but
    # per-pixel shading is skipped.
    mesh.add_ellipsoid("core", (0.0, 1.56, 0.0), (0.29, 0.32, 0.17), 20, 12)
    mesh.add_ellipsoid("armor", (-0.48, 1.88, 0.0), (0.24, 0.20, 0.25), 20, 12)
    mesh.add_ellipsoid("armor", (0.48, 1.88, 0.0), (0.24, 0.20, 0.25), 20, 12)
    mesh.add_tube("cloth", (0.0, 1.92, 0.0), (0.0, 2.10, 0.01), 0.13, 0.12, 16, 2)
    mesh.add_ellipsoid("skin", (0.0, 2.35, 0.03), (0.31, 0.36, 0.30), 24, 16)
    mesh.add_sphere_patch(
        "visor",
        (0.0, 2.35, 0.03),
        0.325,
        math.pi * 0.39,
        math.pi * 0.63,
        math.pi * 0.5 - 0.82,
        math.pi * 0.5 + 0.82,
        24,
        8,
    )

    for side in (-1.0, 1.0):
        shoulder = (0.49 * side, 1.85, 0.0)
        elbow = (0.68 * side, 1.46, 0.035)
        wrist = (0.78 * side, 1.10, 0.10)
        mesh.add_tube("cloth", shoulder, elbow, 0.145, 0.115, 16, 4)
        mesh.add_tube("armor", elbow, wrist, 0.125, 0.095, 16, 4)
        mesh.add_ellipsoid("skin", (0.80 * side, 1.03, 0.13), (0.105, 0.15, 0.09), 16, 10)
        hip = (0.20 * side, 1.03, 0.0)
        knee = (0.23 * side, 0.63, 0.015)
        ankle = (0.23 * side, 0.27, 0.02)
        mesh.add_tube("cloth", hip, knee, 0.185, 0.145, 18, 5)
        mesh.add_tube("armor", knee, ankle, 0.155, 0.12, 18, 4)
        mesh.add_tube(
            "armor",
            (0.23 * side, 0.24, 0.02),
            (0.23 * side, 0.17, 0.15),
            0.15,
            0.13,
            18,
            2,
        )

    mesh.add_ellipsoid("core", (0.0, 1.62, 0.285), (0.15, 0.15, 0.055), 20, 12)
    return mesh.to_json()


def build_scene() -> dict[str, object]:
    skin_diffuse = {
        "kind": "base64",
        "format": "png",
        "srgb": True,
        "data": _skin_texture_base64(),
    }
    materials = [
        _material("armor", [1.0, 1.0, 1.0], "armor_diffuse", "armor_normal", "armor_spec", 0.95, 72.0),
        _material("cloth", [1.0, 1.0, 1.0], "cloth_diffuse", "cloth_normal", "cloth_spec", 0.22, 18.0),
        _material(
            "skin",
            [1.0, 1.0, 1.0],
            "skin_diffuse",
            "skin_normal",
            "skin_spec",
            0.30,
            28.0,
            diffuse_spec=skin_diffuse,
        ),
        _material(
            "visor",
            [0.65, 0.92, 1.0],
            "visor_diffuse",
            "visor_normal",
            "visor_spec",
            1.25,
            96.0,
            opacity=0.34,
            blend="alpha",
            double_sided=True,
            emission=[0.04, 0.20, 0.26],
        ),
        _material(
            "platform",
            [1.0, 1.0, 1.0],
            "platform_diffuse",
            "platform_normal",
            "platform_spec",
            0.55,
            42.0,
        ),
        _material(
            "core",
            [0.28, 0.95, 1.0],
            "armor_diffuse",
            "armor_normal",
            "visor_spec",
            1.5,
            110.0,
            emission=[0.08, 0.55, 0.70],
        ),
    ]
    return {
        "schemaVersion": 1,
        "camera": {
            "position": [3.05, 2.55, 5.35],
            "target": [0.0, 1.35, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fovY": 36.0,
            "near": 0.05,
            "far": 40.0,
            "width": 512,
            "height": 512,
            "background": [0.018, 0.023, 0.034],
            "ambient": [0.11, 0.13, 0.17],
            "exposure": 2.25,
            "cullMode": "back",
        },
        "mesh": {
            "transform": {
                "translation": [0.0, 0.0, 0.0],
                "rotationDegrees": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
            **build_mesh(),
        },
        "materials": materials,
        "lights": [
            {
                "name": "warm-key",
                "type": "directional",
                "direction": [0.48, 0.80, 0.55],
                "color": [1.0, 0.83, 0.64],
                "intensity": 3.4,
                "shadow": {
                    "enabled": True,
                    "resolution": 512,
                    "bias": 0.0012,
                    "normalBias": 0.012,
                    "slopeBias": 1.4,
                    "pcfRadius": 1,
                },
            },
            {
                "name": "cool-fill",
                "type": "point",
                "position": [-2.25, 2.85, 1.85],
                "range": 8.0,
                "color": [0.33, 0.57, 1.0],
                "intensity": 3.2,
                "attenuation": {"constant": 1.0, "linear": 0.05, "quadratic": 0.32},
                "shadow": {
                    "enabled": True,
                    "resolution": 256,
                    "bias": 0.0018,
                    "normalBias": 0.020,
                    "slopeBias": 1.8,
                    "pcfRadius": 1,
                },
            },
        ],
        "debug": {
            "wireframe": False,
            "wireframeColor": [0.15, 1.0, 0.35],
            "normals": False,
            "normalColor": [1.0, 0.18, 0.72],
            "normalLength": 0.11,
        },
    }


def main() -> None:
    output = Path(__file__).with_name("character_scene.json")
    output.write_text(
        json.dumps(build_scene(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
