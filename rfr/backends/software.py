"""Pure NumPy software rasterizer.

The backend keeps a logically equivalent state machine to the OpenGL backend
and implements the same pipeline order per pixel:

1. viewport transform and back/front face culling (CCW front faces);
2. perspective-correct interpolation of lit RGBA vertex colors;
3. depth test against a float32 depth buffer;
4. blending using equations equivalent to ``glBlendFunc``;
5. quantization to RGBA8 with round-to-nearest (matching GPU output).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .. import mathutils
from ..errors import RenderError
from ..pipeline import DrawCall
from ..scene import Scene
from .base import RenderBackend


def _quantize_unorm8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(np.clip(values, 0.0, 1.0) * 255.0 + 0.5), 0, 255).astype(np.uint8)


class SoftwareBackend(RenderBackend):
    name = "software"

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise RenderError("E_SIZE_MISMATCH", "Invalid software framebuffer size.",
                              {"width": width, "height": height})
        self.width = int(width)
        self.height = int(height)
        self.color_f = np.zeros((self.height, self.width, 4), dtype=np.float32)
        self.depth = np.ones((self.height, self.width), dtype=np.float32)
        self.framebuffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self._blend = "opaque"
        self._depth_test = True
        self._cull = "back"
        self.rasterized_triangles = 0
        self.shaded_pixels = 0
        self._view = np.eye(4, dtype=np.float32)
        self._proj = np.eye(4, dtype=np.float32)

    # ------------------------------------------------------------ frame flow
    def begin_frame(self, scene: Scene) -> np.ndarray:
        if scene.width != self.width or scene.height != self.height:
            raise RenderError("E_SIZE_MISMATCH", "Scene/backend framebuffer size mismatch.",
                              {"scene": [scene.width, scene.height],
                               "backend": [self.width, self.height]})
        bg = np.asarray(scene.background, dtype=np.float32)
        self.color_f[:] = bg[None, None, :]
        self.depth.fill(1.0)
        aspect = self.width / self.height
        view = mathutils.look_at(scene.camera.eye, scene.camera.target, scene.camera.up)
        proj = mathutils.perspective(scene.camera.fov_y, aspect,
                                     scene.camera.near, scene.camera.far)
        self._view = view.astype(np.float32)
        self._proj = proj.astype(np.float32)
        return self.framebuffer

    def apply_state(self, draw: DrawCall, changes: Dict[str, bool]) -> None:
        if changes["blend"]:
            self._blend = draw.blend
        if changes["depth_test"]:
            self._depth_test = draw.depth_test
        if changes["cull"]:
            self._cull = draw.cull

    def end_frame(self) -> None:
        self.framebuffer = _quantize_unorm8(self.color_f)

    def read_pixels(self) -> np.ndarray:
        return self.framebuffer.copy()

    def backend_statistics(self) -> Dict[str, int]:
        return {
            "rasterized_triangles": self.rasterized_triangles,
            "shaded_pixels": self.shaded_pixels,
        }

    # ------------------------------------------------------------------ draw
    def draw(self, draw: DrawCall) -> None:
        if draw.positions.shape[0] != draw.colors.shape[0]:
            raise RenderError("E_SIZE_MISMATCH", "Position/color vertex count mismatch.",
                              {"node": draw.node_name})
        mvp = (self._proj @ self._view).astype(np.float32)
        vertex_count = draw.positions.shape[0]
        homog = np.concatenate(
            [draw.positions, np.ones((vertex_count, 1), dtype=np.float32)], axis=1)
        clip = (mvp @ homog.T).T.astype(np.float32)
        w_clip = clip[:, 3]
        if np.any(np.abs(w_clip) < 1e-6):
            return
        ndc = (clip[:, :3] / w_clip[:, None]).astype(np.float32)

        # OpenGL window coordinates: x to the right, y UPWARD, depth in [0,1].
        # Keeping y-up makes winding, edge functions and barycentric weights
        # identical to the GPU; conversion to top-down image rows happens only
        # when indexing the framebuffer.
        screen_x = (ndc[:, 0] * 0.5 + 0.5) * self.width
        screen_y = (ndc[:, 1] * 0.5 + 0.5) * self.height
        screen_z = ndc[:, 2] * 0.5 + 0.5
        inv_w = (1.0 / w_clip).astype(np.float32)

        idx = draw.indices.reshape(-1, 3)
        for tri in idx:
            self._rasterize_triangle(draw, tri, screen_x, screen_y, screen_z, inv_w)

    def _rasterize_triangle(self, draw, tri, sx, sy, sz, inv_w) -> None:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        x0, y0 = float(sx[i0]), float(sy[i0])
        x1, y1 = float(sx[i1]), float(sy[i1])
        x2, y2 = float(sx[i2]), float(sy[i2])

        # Twice the signed window area (B-A) x (C-A), in the same convention
        # as the edge functions; positive for a CCW front face (GL_CCW).
        screen_area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if abs(screen_area) < 1e-6:
            return
        ccw = screen_area > 0.0
        if self._cull == "back" and not ccw:
            return
        if self._cull == "front" and ccw:
            return

        # Bounding box over pixel centers.  Column c samples x = c + 0.5 and
        # top-down row r samples window y = (H - 1 - r) + 0.5.
        min_c = max(int(np.floor(min(x0, x1, x2) - 0.5)), 0)
        max_c = min(int(np.ceil(max(x0, x1, x2) - 0.5)), self.width - 1)
        min_wy = max(int(np.floor(min(y0, y1, y2) - 0.5)), 0)
        max_wy = min(int(np.ceil(max(y0, y1, y2) - 0.5)), self.height - 1)
        if min_c > max_c or min_wy > max_wy:
            return

        col_grid, wy_grid = np.meshgrid(
            np.arange(min_c, max_c + 1, dtype=np.float32),
            np.arange(min_wy, max_wy + 1, dtype=np.float32),
        )
        px = col_grid + np.float32(0.5)
        py = wy_grid + np.float32(0.5)
        # Positive-inside edge functions E(P) = (B - A) x (P - A) in y-up
        # window space.  For a CCW front-facing triangle every edge is >= 0
        # over the interior.  OpenGL's top-left rule makes a horizontal
        # left-to-right edge ("top") or an upward edge, dy > 0, ("left")
        # inclusive and the other edges strict.
        e0p = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
        e1p = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        e2p = (x0 - x2) * (py - y2) - (y0 - y2) * (px - x2)
        edges = [(x0, y0, x1, y1), (x1, y1, x2, y2), (x2, y2, x0, y0)]
        inside = np.ones(px.shape, dtype=bool)
        for (ax, ay, bx, by), ev in zip(edges, (e0p, e1p, e2p)):
            is_top_left = (by == ay and bx > ax) or (by > ay)
            if is_top_left:
                inside &= ev >= 0.0
            else:
                # A length-scaled epsilon (about a tenth of a pixel at the
                # typical edge length) absorbs sub-pixel float32 boundary noise
                # versus the GPU's fixed-function rasterizer while leaving
                # clearly exterior samples excluded.
                tolerance = np.float32(1e-3) * np.float32(np.hypot(bx - ax, by - ay))
                inside &= ev > -tolerance
        # Barycentric weights: the edge function on edge (v_a -> v_b) is
        # proportional to the weight of the opposite vertex.  e0 spans v0->v1
        # so it weights v2, e1 weights v0 and e2 weights v1.
        weight0 = e1p / screen_area   # weight of vertex i0
        weight1 = e2p / screen_area   # weight of vertex i1
        weight2 = e0p / screen_area   # weight of vertex i2
        if not np.any(inside):
            return
        self.rasterized_triangles += 1

        persp0 = weight0 * inv_w[i0]
        persp1 = weight1 * inv_w[i1]
        persp2 = weight2 * inv_w[i2]
        persp_sum = persp0 + persp1 + persp2
        persp_sum = np.where(persp_sum == 0.0, np.float32(1.0), persp_sum)
        b0 = persp0 / persp_sum
        b1 = persp1 / persp_sum
        b2 = persp2 / persp_sum

        # Window depth is affine in screen space (OpenGL derives gl_FragCoord.z
        # by perspective-correct interpolation of clip z/w, which reduces to a
        # screen-linear combination of window depths), so use the raw edge
        # barycentrics rather than the perspective-correct weights reserved
        # for generic varyings such as color.
        raw_w0 = e1p / screen_area
        raw_w1 = e2p / screen_area
        raw_w2 = e0p / screen_area
        interp_z = raw_w0 * sz[i0] + raw_w1 * sz[i1] + raw_w2 * sz[i2]
        colors = draw.colors
        interp_color = (
            b0[..., None] * colors[i0]
            + b1[..., None] * colors[i1]
            + b2[..., None] * colors[i2]
        ).astype(np.float32)

        rows = (self.height - 1 - wy_grid).astype(np.int64)
        cols = col_grid.astype(np.int64)
        valid = inside
        if self._depth_test:
            valid = inside & (interp_z < self.depth[rows, cols])
        if not np.any(valid):
            return

        sel_rows = rows[valid]
        sel_cols = cols[valid]
        sel_z = interp_z[valid].astype(np.float32)
        src = np.clip(interp_color[valid], 0.0, 1.0).astype(np.float32)
        dst = self.color_f[sel_rows, sel_cols]

        if self._blend == "opaque":
            out = src
        elif self._blend == "alpha":
            src_a = src[:, 3:4]
            out_rgb = src[:, :3] * src_a + dst[:, :3] * (1.0 - src_a)
            out_a = src[:, 3:4] * src_a + dst[:, 3:4] * (1.0 - src_a)
            out = np.concatenate([out_rgb, out_a], axis=1).astype(np.float32)
        else:  # additive: (SRC_ALPHA, ONE)
            out_rgb = src[:, :3] * src[:, 3:4] + dst[:, :3]
            out_a = src[:, 3:4] * src[:, 3:4] + dst[:, 3:4]
            out = np.concatenate([out_rgb, out_a], axis=1).astype(np.float32)

        self.color_f[sel_rows, sel_cols] = np.clip(out, 0.0, 1.0)
        if self._depth_test and self._blend == "opaque":
            self.depth[sel_rows, sel_cols] = sel_z
        self.shaded_pixels += int(valid.sum())
