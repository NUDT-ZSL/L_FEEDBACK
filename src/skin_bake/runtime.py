"""Runtime asset loading and CPU-side curve sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .basis import BindBasis, build_bind_basis
from .errors import output_io, schema, texture_corrupt
from .scene import Scene, load_scene
from .skinning import AnimationSampler, FrameSkin
from .texture import TextureData, read_texture


@dataclass(frozen=True)
class VertexBlock:
    vertex: int
    pixel_offset: tuple[int, int]
    size: tuple[int, int]
    sample_count: int
    sample_rate: float

    @property
    def capacity(self) -> int:
        return self.size[0] * self.size[1]

    def texel(self, sample_index: int) -> tuple[int, int]:
        if sample_index < 0 or sample_index >= self.sample_count:
            raise IndexError("curve sample index is out of range")
        if sample_index >= self.capacity:
            raise texture_corrupt(
                f"sample {sample_index} exceeds block capacity for vertex {self.vertex}"
            )
        row, column = divmod(sample_index, self.size[0])
        return (
            self.pixel_offset[0] + column,
            self.pixel_offset[1] + row,
        )


@dataclass
class BakedAsset:
    scene: Scene
    texture: TextureData
    atlas_map: dict[str, Any]
    blocks: dict[int, VertexBlock]
    sampler: AnimationSampler
    bind_basis: BindBasis

    @property
    def frame_count(self) -> int:
        frames = int(round(self.scene.duration * self.scene.fps))
        return frames if self.scene.loop_mode == "loop" else frames + 1

    @property
    def sample_count(self) -> int:
        return int(self.atlas_map["sampleCount"])

    def clamped_time(self, time: float) -> float:
        if self.scene.loop_mode == "loop":
            return float(time % self.scene.duration)
        return float(np.clip(time, 0.0, self.scene.duration))

    def frame_time(self, frame: int) -> float:
        if frame < 0 or frame >= self.frame_count:
            raise schema(f"frame {frame} is outside [0, {self.frame_count - 1}]")
        return frame / self.scene.fps

    def curve_sample_index(self, time: float) -> int:
        if self.scene.loop_mode == "loop":
            normalized = (float(time) % self.scene.duration) / self.scene.duration
            return int(round(normalized * self.sample_count)) % self.sample_count
        normalized = float(np.clip(time, 0.0, self.scene.duration)) / self.scene.duration
        return int(round(normalized * (self.sample_count - 1)))

    def sample_vertex_displacement(self, vertex: int, time: float) -> np.ndarray:
        block = self.blocks[vertex]
        sample_index = self.curve_sample_index(time)
        x, y = block.texel(sample_index)
        return self.texture.pixels[y, x].copy()

    def frame_displacements(self, time: float) -> np.ndarray:
        return np.stack(
            [self.sample_vertex_displacement(vertex, time) for vertex in range(self.scene.vertex_count)],
            axis=0,
        )

    def frame_skin(self, time: float) -> FrameSkin:
        return self.sampler.skin(self.clamped_time(time))

    def displaced_frame(
        self, time: float
    ) -> tuple[FrameSkin, np.ndarray, np.ndarray]:
        skin = self.frame_skin(time)
        displacement = self.frame_displacements(time)
        positions = (
            skin.positions
            + skin.tangents * displacement[:, 0:1]
            + skin.bitangents * displacement[:, 1:2]
        )
        return skin, displacement, positions


def _mapping_block(raw: dict[str, Any], index: int) -> VertexBlock:
    try:
        vertex = int(raw["vertex"])
        pixel_offset = tuple(int(v) for v in raw["pixelOffset"])
        size = tuple(int(v) for v in raw["blockSize"])
        sample_count = int(raw["sampleCount"])
        sample_rate = float(raw["sampleRate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise schema(f"atlas_map.blocks[{index}] is malformed") from exc
    if len(pixel_offset) != 2 or len(size) != 2 or min(size) <= 0 or sample_count <= 0:
        raise schema(f"atlas_map.blocks[{index}] has invalid dimensions")
    return VertexBlock(vertex, pixel_offset, size, sample_count, sample_rate)


def load_baked_asset(scene_path: str | Path, bake_dir: str | Path) -> BakedAsset:
    scene = load_scene(scene_path)
    directory = Path(bake_dir)
    map_path = directory / "atlas_map.json"
    try:
        atlas_map = json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise output_io(f"cannot load atlas map '{map_path}': {exc}") from exc
    if atlas_map.get("format") != "skin-bake-atlas-map":
        raise schema("atlas_map.json format is invalid")
    if atlas_map.get("sceneContentHash") != scene.content_hash:
        raise schema("atlas map was baked from a different scene")
    texture_name = atlas_map.get("texture")
    if not isinstance(texture_name, str):
        raise schema("atlas map does not name its texture file")
    texture = read_texture(directory / texture_name)
    try:
        atlas_size = tuple(int(v) for v in atlas_map.get("atlasSize", []))
    except (TypeError, ValueError) as exc:
        raise schema("atlas map atlasSize is malformed") from exc
    if len(atlas_size) != 2:
        raise schema("atlas map atlasSize must have two components")
    if atlas_size != (texture.width, texture.height):
        raise schema("atlas map and texture dimensions differ")
    blocks: dict[int, VertexBlock] = {}
    raw_blocks = atlas_map.get("blocks")
    if not isinstance(raw_blocks, list):
        raise schema("atlas map blocks must be an array")
    for index, raw in enumerate(raw_blocks):
        block = _mapping_block(raw, index)
        if block.vertex < 0 or block.vertex >= scene.vertex_count:
            raise schema(f"atlas map vertex {block.vertex} is out of range")
        if block.vertex in blocks:
            raise schema(f"atlas map contains duplicate vertex {block.vertex}")
        if (
            block.pixel_offset[0] < 0
            or block.pixel_offset[1] < 0
            or block.pixel_offset[0] + block.size[0] > texture.width
            or block.pixel_offset[1] + block.size[1] > texture.height
        ):
            raise schema(f"atlas block for vertex {block.vertex} is outside the texture")
        blocks[block.vertex] = block
    if len(blocks) != scene.vertex_count:
        raise schema("atlas map does not contain exactly one block per mesh vertex")
    try:
        global_sample_count = int(atlas_map.get("sampleCount", 0))
    except (TypeError, ValueError) as exc:
        raise schema("atlas map sampleCount is malformed") from exc
    if global_sample_count <= 0:
        raise schema("atlas map sampleCount must be positive")
    for block in blocks.values():
        if block.sample_count != global_sample_count:
            raise schema("all atlas blocks must use the same global sampleCount")
        if block.sample_rate <= 0.0:
            raise schema("atlas block sampleRate must be positive")

    return BakedAsset(
        scene=scene,
        texture=texture,
        atlas_map=atlas_map,
        blocks=blocks,
        sampler=AnimationSampler(scene),
        bind_basis=build_bind_basis(scene.normals, scene.tangents),
    )

