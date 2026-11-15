"""Immutable render-state model.

All numeric animation and material values use integer fixed point. This avoids
platform-dependent floating-point accumulation and makes repeated runs
bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
TICKS_PER_SECOND = 600_000
LOGIC_STEP_TICKS = 10_000  # Exactly 60 Hz on a 600 kHz time base.
FP_SCALE = 1_000_000

ANIMATION_ORDER: Tuple[str, ...] = ("locomotion", "upper_body", "breath")
ANIMATION_BLEND_MODES: Mapping[str, str] = {
    "locomotion": "override",
    "upper_body": "override",
    "breath": "additive",
}

MATERIAL_ORDER: Tuple[str, ...] = (
    "base_color",
    "normal",
    "emissive",
    "transparent",
)
MATERIAL_BLEND_MODES: Mapping[str, str] = {
    "base_color": "opaque",
    "normal": "normal_map",
    "emissive": "additive",
    "transparent": "alpha",
}
MATERIAL_PARAM_COUNTS: Mapping[str, int] = {
    "base_color": 4,
    "normal": 3,
    "emissive": 4,
    "transparent": 4,
}

ZERO_EMISSIVE_PARAMS: Tuple[int, int, int, int] = (0, 0, 0, 0)
ZERO_TRANSPARENT_PARAMS: Tuple[int, int, int, int] = (0, 0, 0, 0)


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _as_int_tuple(value: Any, length: int, field_name: str) -> Tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{field_name} must contain exactly {length} integers")
    return tuple(_require_int(item, field_name) for item in value)


def _assert_exact_keys(
    data: Mapping[str, Any], expected: Sequence[str], context: str
) -> None:
    if not isinstance(data, Mapping):
        raise ValueError(f"{context} must be a mapping")
    expected_set = set(expected)
    actual_set = set(data)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(
            f"{context} fields mismatch; missing={missing}, extra={extra}"
        )


@dataclass(frozen=True)
class Keyframe:
    time_ticks: int
    position: Tuple[int, int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_ticks": self.time_ticks,
            "position": list(self.position),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Keyframe":
        _assert_exact_keys(
            data, ("time_ticks", "position"), "keyframe"
        )
        return cls(
            time_ticks=_require_int(data["time_ticks"], "keyframe.time_ticks"),
            position=_as_int_tuple(
                data["position"], 3, "keyframe.position"
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class AnimationLayer:
    layer_id: str
    bone: str
    current_time_ticks: int
    weight_fp: int
    blend_mode: str
    duration_ticks: int
    keyframes: Tuple[Keyframe, ...]
    version: int

    def validate(self) -> None:
        if not self.layer_id:
            raise ValueError("animation layer_id must not be empty")
        if not self.bone:
            raise ValueError(f"animation layer {self.layer_id} has empty bone")
        _require_int(self.current_time_ticks, "animation.current_time_ticks")
        if self.current_time_ticks < 0:
            raise ValueError("animation current_time_ticks must be non-negative")
        _require_int(self.weight_fp, "animation.weight_fp")
        if not 0 <= self.weight_fp <= FP_SCALE:
            raise ValueError("animation weight_fp must be in [0, FP_SCALE]")
        if self.duration_ticks <= 0:
            raise ValueError("animation duration_ticks must be positive")
        if len(self.keyframes) < 2:
            raise ValueError("animation requires at least two keyframes")
        if self.keyframes[0].time_ticks != 0:
            raise ValueError("first animation keyframe must start at time 0")
        if self.keyframes[-1].time_ticks != self.duration_ticks:
            raise ValueError("last animation keyframe must equal duration_ticks")
        previous = -1
        for keyframe in self.keyframes:
            if keyframe.time_ticks <= previous:
                raise ValueError("animation keyframe times must be strictly increasing")
            previous = keyframe.time_ticks
        if self.version < 1:
            raise ValueError("animation layer version must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bone": self.bone,
            "current_time_ticks": self.current_time_ticks,
            "weight_fp": self.weight_fp,
            "blend_mode": self.blend_mode,
            "duration_ticks": self.duration_ticks,
            "keyframes": [item.to_dict() for item in self.keyframes],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, layer_id: str, data: Mapping[str, Any]) -> "AnimationLayer":
        _assert_exact_keys(
            data,
            (
                "bone",
                "current_time_ticks",
                "weight_fp",
                "blend_mode",
                "duration_ticks",
                "keyframes",
                "version",
            ),
            f"animation layer {layer_id}",
        )
        return cls(
            layer_id=layer_id,
            bone=str(data["bone"]),
            current_time_ticks=_require_int(
                data["current_time_ticks"], "animation.current_time_ticks"
            ),
            weight_fp=_require_int(data["weight_fp"], "animation.weight_fp"),
            blend_mode=str(data["blend_mode"]),
            duration_ticks=_require_int(
                data["duration_ticks"], "animation.duration_ticks"
            ),
            keyframes=tuple(
                Keyframe.from_dict(item) for item in data["keyframes"]
            ),
            version=_require_int(data["version"], "animation.version"),
        )


@dataclass(frozen=True)
class MaterialLayer:
    layer_type: str
    params: Tuple[int, ...]
    enabled: bool
    version: int
    blend_mode: str

    def validate(self) -> None:
        if self.layer_type not in MATERIAL_ORDER:
            raise ValueError(f"unknown material layer {self.layer_type}")
        expected = MATERIAL_PARAM_COUNTS[self.layer_type]
        if len(self.params) != expected:
            raise ValueError(
                f"{self.layer_type} requires {expected} parameters, got {len(self.params)}"
            )
        for value in self.params:
            _require_int(value, f"{self.layer_type}.params")
        if not isinstance(self.enabled, bool):
            raise ValueError("material enabled must be boolean")
        if self.version < 1:
            raise ValueError("material layer version must be >= 1")
        if self.blend_mode != MATERIAL_BLEND_MODES[self.layer_type]:
            raise ValueError(
                f"invalid blend mode for {self.layer_type}: {self.blend_mode}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_type": self.layer_type,
            "params": list(self.params),
            "enabled": self.enabled,
            "version": self.version,
            "blend_mode": self.blend_mode,
        }

    @classmethod
    def from_dict(
        cls, layer_type: str, data: Mapping[str, Any]
    ) -> "MaterialLayer":
        _assert_exact_keys(
            data,
            ("layer_type", "params", "enabled", "version", "blend_mode"),
            f"material layer {layer_type}",
        )
        params = data["params"]
        if not isinstance(params, (list, tuple)):
            raise ValueError("material params must be a sequence")
        if not isinstance(data["enabled"], bool):
            raise ValueError("material enabled must be boolean")
        return cls(
            layer_type=layer_type,
            params=tuple(_require_int(v, "material.params") for v in params),
            enabled=data["enabled"],
            version=_require_int(data["version"], "material.version"),
            blend_mode=str(data["blend_mode"]),
        )


@dataclass(frozen=True)
class RenderCacheChannel:
    params: Tuple[int, ...]
    value_fp: int
    generation: int

    def validate(self) -> None:
        if len(self.params) != 4:
            raise ValueError("render cache params must contain four integers")
        for value in self.params:
            _require_int(value, "render_cache.params")
        _require_int(self.value_fp, "render_cache.value_fp")
        _require_int(self.generation, "render_cache.generation")
        if self.value_fp < 0 or self.generation < 0:
            raise ValueError("render cache values must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": list(self.params),
            "value_fp": self.value_fp,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RenderCacheChannel":
        _assert_exact_keys(
            data,
            ("params", "value_fp", "generation"),
            "render cache channel",
        )
        params = data["params"]
        if not isinstance(params, (list, tuple)):
            raise ValueError("render cache params must be a sequence")
        return cls(
            params=tuple(_require_int(v, "render_cache.params") for v in params),
            value_fp=_require_int(data["value_fp"], "render_cache.value_fp"),
            generation=_require_int(data["generation"], "render_cache.generation"),
        )


@dataclass(frozen=True)
class RenderCache:
    emissive: RenderCacheChannel
    transparent: RenderCacheChannel

    def validate(self) -> None:
        self.emissive.validate()
        self.transparent.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emissive": self.emissive.to_dict(),
            "transparent": self.transparent.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RenderCache":
        _assert_exact_keys(
            data, ("emissive", "transparent"), "render cache"
        )
        return cls(
            emissive=RenderCacheChannel.from_dict(data["emissive"]),
            transparent=RenderCacheChannel.from_dict(data["transparent"]),
        )


def emissive_value_fp(params: Sequence[int]) -> int:
    r, g, b, intensity = params
    return ((r + g + b) * intensity) // (3 * FP_SCALE)


def transparent_value_fp(params: Sequence[int]) -> int:
    return params[3]


def material_cache_channel(
    layer: MaterialLayer,
    *,
    generation: Optional[int] = None,
    force_clear: bool = False,
) -> RenderCacheChannel:
    current_generation = 0 if generation is None else generation
    if force_clear or not layer.enabled:
        return RenderCacheChannel(
            params=(0, 0, 0, 0),
            value_fp=0,
            generation=current_generation,
        )
    if layer.layer_type == "emissive":
        value = emissive_value_fp(layer.params)
    else:
        value = transparent_value_fp(layer.params)
    return RenderCacheChannel(
        params=tuple(layer.params),
        value_fp=value,
        generation=current_generation,
    )


@dataclass(frozen=True)
class Light:
    light_id: str
    color: Tuple[int, int, int]
    intensity_fp: int
    position: Tuple[int, int, int]

    def validate(self) -> None:
        if not self.light_id:
            raise ValueError("light_id must not be empty")
        if len(self.color) != 3:
            raise ValueError("light color must contain three integers")
        if len(self.position) != 3:
            raise ValueError("light position must contain three integers")
        for channel in self.color:
            _require_int(channel, "light.color")
            if not 0 <= channel <= FP_SCALE:
                raise ValueError("light color channels must be in [0, FP_SCALE]")
        _require_int(self.intensity_fp, "light.intensity_fp")
        if self.intensity_fp < 0:
            raise ValueError("light intensity_fp must be non-negative")
        for value in self.position:
            _require_int(value, "light.position")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "color": list(self.color),
            "intensity_fp": self.intensity_fp,
            "position": list(self.position),
        }

    @classmethod
    def from_dict(cls, light_id: str, data: Mapping[str, Any]) -> "Light":
        _assert_exact_keys(
            data,
            ("color", "intensity_fp", "position"),
            f"light {light_id}",
        )
        return cls(
            light_id=light_id,
            color=_as_int_tuple(data["color"], 3, "light.color"),  # type: ignore[arg-type]
            intensity_fp=_require_int(data["intensity_fp"], "light.intensity_fp"),
            position=_as_int_tuple(
                data["position"], 3, "light.position"
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class LightSnapshot:
    state_version: int
    light_revision: int
    lights: Tuple[Light, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_version": self.state_version,
            "light_revision": self.light_revision,
            "lights": {
                light.light_id: light.to_dict()
                for light in sorted(self.lights, key=lambda item: item.light_id)
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LightSnapshot":
        _assert_exact_keys(
            data,
            ("state_version", "light_revision", "lights"),
            "light snapshot",
        )
        lights_data = data["lights"]
        if not isinstance(lights_data, Mapping):
            raise ValueError("light snapshot lights must be a mapping")
        return cls(
            state_version=_require_int(
                data["state_version"], "light_snapshot.state_version"
            ),
            light_revision=_require_int(
                data["light_revision"], "light_snapshot.light_revision"
            ),
            lights=tuple(
                Light.from_dict(light_id, lights_data[light_id])
                for light_id in sorted(lights_data)
            ),
        )


@dataclass(frozen=True)
class StateSnapshot:
    state_version: int
    logic_frame: int
    logic_time_ticks: int
    animation_version: int
    material_version: int
    light_revision: int
    animation_layers: Tuple[AnimationLayer, ...]
    material_layers: Tuple[MaterialLayer, ...]
    render_cache: RenderCache
    lights: Tuple[Light, ...]
    light_snapshot: LightSnapshot
    emissive_transparent_rule: str = "exclusive"

    def animation_map(self) -> Dict[str, AnimationLayer]:
        return {layer.layer_id: layer for layer in self.animation_layers}

    def material_map(self) -> Dict[str, MaterialLayer]:
        return {layer.layer_type: layer for layer in self.material_layers}

    def light_map(self) -> Dict[str, Light]:
        return {light.light_id: light for light in self.lights}

    def validate(self) -> None:
        if self.state_version < 1:
            raise ValueError("state_version must be >= 1")
        if self.logic_frame < 0 or self.logic_time_ticks < 0:
            raise ValueError("logic clock must be non-negative")
        if self.logic_time_ticks != self.logic_frame * LOGIC_STEP_TICKS:
            raise ValueError(
                "logic_time_ticks must equal logic_frame * LOGIC_STEP_TICKS"
            )
        for value, name in (
            (self.animation_version, "animation_version"),
            (self.material_version, "material_version"),
            (self.light_revision, "light_revision"),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.emissive_transparent_rule not in ("exclusive", "stacked"):
            raise ValueError("unknown emissive/transparent rule")

        animation_ids = tuple(layer.layer_id for layer in self.animation_layers)
        if animation_ids != ANIMATION_ORDER:
            raise ValueError(
                f"animation layer order must be {ANIMATION_ORDER}, got {animation_ids}"
            )
        for layer in self.animation_layers:
            layer.validate()
            if layer.blend_mode != ANIMATION_BLEND_MODES[layer.layer_id]:
                raise ValueError(
                    f"invalid animation blend mode for {layer.layer_id}: {layer.blend_mode}"
                )

        material_ids = tuple(layer.layer_type for layer in self.material_layers)
        if material_ids != MATERIAL_ORDER:
            raise ValueError(
                f"material layer order must be {MATERIAL_ORDER}, got {material_ids}"
            )
        materials = self.material_map()
        for layer in self.material_layers:
            layer.validate()
        if (
            self.emissive_transparent_rule == "exclusive"
            and materials["emissive"].enabled
            and materials["transparent"].enabled
        ):
            raise ValueError("emissive and transparent layers are mutually exclusive")

        self.render_cache.validate()
        for layer_type, cache in (
            ("emissive", self.render_cache.emissive),
            ("transparent", self.render_cache.transparent),
        ):
            material = materials[layer_type]
            if material.enabled:
                expected = material_cache_channel(
                    material, generation=cache.generation
                )
                if cache != expected:
                    raise ValueError(
                        f"{layer_type} render cache does not match enabled material"
                    )
            elif cache.params != (0, 0, 0, 0) or cache.value_fp != 0:
                raise ValueError(
                    f"{layer_type} render cache must be zero when disabled"
                )

        light_ids = tuple(light.light_id for light in self.lights)
        if light_ids != tuple(sorted(light_ids)):
            raise ValueError("lights must be stored in stable light_id order")
        if len(set(light_ids)) != len(light_ids):
            raise ValueError("duplicate light_id in state")
        for light in self.lights:
            light.validate()
        if self.light_snapshot.state_version != self.state_version:
            raise ValueError("light snapshot version does not match state version")
        if self.light_snapshot.light_revision != self.light_revision:
            raise ValueError("light snapshot revision does not match light_revision")
        if self.light_snapshot.lights != self.lights:
            raise ValueError("light snapshot content does not match current lights")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "state_version": self.state_version,
            "logic_frame": self.logic_frame,
            "logic_time_ticks": self.logic_time_ticks,
            "animation_version": self.animation_version,
            "material_version": self.material_version,
            "light_revision": self.light_revision,
            "emissive_transparent_rule": self.emissive_transparent_rule,
            "animation_order": list(ANIMATION_ORDER),
            "animation_blend_modes": dict(ANIMATION_BLEND_MODES),
            "animation_layers": {
                layer.layer_id: layer.to_dict() for layer in self.animation_layers
            },
            "material_order": list(MATERIAL_ORDER),
            "material_blend_modes": dict(MATERIAL_BLEND_MODES),
            "material_param_counts": dict(MATERIAL_PARAM_COUNTS),
            "material_layers": {
                layer.layer_type: layer.to_dict() for layer in self.material_layers
            },
            "render_cache": self.render_cache.to_dict(),
            "lights": {
                light.light_id: light.to_dict()
                for light in sorted(self.lights, key=lambda item: item.light_id)
            },
            "light_snapshot": self.light_snapshot.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateSnapshot":
        _assert_exact_keys(
            data,
            (
                "schema_version",
                "state_version",
                "logic_frame",
                "logic_time_ticks",
                "animation_version",
                "material_version",
                "light_revision",
                "emissive_transparent_rule",
                "animation_order",
                "animation_blend_modes",
                "animation_layers",
                "material_order",
                "material_blend_modes",
                "material_param_counts",
                "material_layers",
                "render_cache",
                "lights",
                "light_snapshot",
            ),
            "state snapshot",
        )
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        if list(data.get("animation_order", [])) != list(ANIMATION_ORDER):
            raise ValueError("animation_order is not the fixed supported order")
        if list(data.get("material_order", [])) != list(MATERIAL_ORDER):
            raise ValueError("material_order is not the fixed supported order")

        animation_data = data["animation_layers"]
        material_data = data["material_layers"]
        lights_data = data["lights"]
        if not isinstance(animation_data, Mapping) or not isinstance(
            material_data, Mapping
        ):
            raise ValueError("layer collections must be mappings")
        if not isinstance(lights_data, Mapping):
            raise ValueError("lights must be a mapping")

        snapshot = cls(
            state_version=_require_int(data["state_version"], "state_version"),
            logic_frame=_require_int(data["logic_frame"], "logic_frame"),
            logic_time_ticks=_require_int(
                data["logic_time_ticks"], "logic_time_ticks"
            ),
            animation_version=_require_int(
                data["animation_version"], "animation_version"
            ),
            material_version=_require_int(
                data["material_version"], "material_version"
            ),
            light_revision=_require_int(data["light_revision"], "light_revision"),
            animation_layers=tuple(
                AnimationLayer.from_dict(layer_id, animation_data[layer_id])
                for layer_id in ANIMATION_ORDER
            ),
            material_layers=tuple(
                MaterialLayer.from_dict(layer_type, material_data[layer_type])
                for layer_type in MATERIAL_ORDER
            ),
            render_cache=RenderCache.from_dict(data["render_cache"]),
            lights=tuple(
                Light.from_dict(light_id, lights_data[light_id])
                for light_id in sorted(lights_data)
            ),
            light_snapshot=LightSnapshot.from_dict(data["light_snapshot"]),
            emissive_transparent_rule=str(
                data.get("emissive_transparent_rule", "exclusive")
            ),
        )
        snapshot.validate()
        return snapshot


def _default_animation_layers() -> Tuple[AnimationLayer, ...]:
    cycle = 600_000
    return (
        AnimationLayer(
            layer_id="locomotion",
            bone="root",
            current_time_ticks=0,
            weight_fp=FP_SCALE,
            blend_mode=ANIMATION_BLEND_MODES["locomotion"],
            duration_ticks=cycle,
            keyframes=(
                Keyframe(0, (0, 0, 0)),
                Keyframe(150_000, (100_000, 20_000, 0)),
                Keyframe(300_000, (-50_000, 100_000, 30_000)),
                Keyframe(450_000, (0, -20_000, -10_000)),
                Keyframe(cycle, (0, 0, 0)),
            ),
            version=1,
        ),
        AnimationLayer(
            layer_id="upper_body",
            bone="spine",
            current_time_ticks=0,
            weight_fp=800_000,
            blend_mode=ANIMATION_BLEND_MODES["upper_body"],
            duration_ticks=cycle,
            keyframes=(
                Keyframe(0, (0, 0, 0)),
                Keyframe(200_000, (30_000, 50_000, 0)),
                Keyframe(400_000, (-30_000, 50_000, 0)),
                Keyframe(cycle, (0, 0, 0)),
            ),
            version=1,
        ),
        AnimationLayer(
            layer_id="breath",
            bone="spine",
            current_time_ticks=0,
            weight_fp=500_000,
            blend_mode=ANIMATION_BLEND_MODES["breath"],
            duration_ticks=cycle,
            keyframes=(
                Keyframe(0, (0, 0, 0)),
                Keyframe(300_000, (0, 20_000, 0)),
                Keyframe(cycle, (0, 0, 0)),
            ),
            version=1,
        ),
    )


def make_default_state(
    emissive_transparent_rule: str = "exclusive",
) -> StateSnapshot:
    """Create the deterministic demo/sample state."""

    material_layers = (
        MaterialLayer(
            layer_type="base_color",
            params=(800_000, 600_000, 500_000, FP_SCALE),
            enabled=True,
            version=1,
            blend_mode=MATERIAL_BLEND_MODES["base_color"],
        ),
        MaterialLayer(
            layer_type="normal",
            params=(0, 0, FP_SCALE),
            enabled=True,
            version=1,
            blend_mode=MATERIAL_BLEND_MODES["normal"],
        ),
        MaterialLayer(
            layer_type="emissive",
            params=ZERO_EMISSIVE_PARAMS,
            enabled=False,
            version=1,
            blend_mode=MATERIAL_BLEND_MODES["emissive"],
        ),
        MaterialLayer(
            layer_type="transparent",
            params=ZERO_TRANSPARENT_PARAMS,
            enabled=False,
            version=1,
            blend_mode=MATERIAL_BLEND_MODES["transparent"],
        ),
    )
    materials = {layer.layer_type: layer for layer in material_layers}
    lights = tuple(
        sorted(
            (
                Light(
                    light_id="key",
                    color=(FP_SCALE, 900_000, 800_000),
                    intensity_fp=FP_SCALE,
                    position=(0, 300_000, 200_000),
                ),
                Light(
                    light_id="rim",
                    color=(200_000, 400_000, FP_SCALE),
                    intensity_fp=500_000,
                    position=(0, -200_000, -300_000),
                ),
            ),
            key=lambda item: item.light_id,
        )
    )
    state = StateSnapshot(
        state_version=1,
        logic_frame=0,
        logic_time_ticks=0,
        animation_version=1,
        material_version=1,
        light_revision=1,
        animation_layers=_default_animation_layers(),
        material_layers=material_layers,
        render_cache=RenderCache(
            emissive=material_cache_channel(materials["emissive"], generation=0),
            transparent=material_cache_channel(
                materials["transparent"], generation=0
            ),
        ),
        lights=lights,
        light_snapshot=LightSnapshot(
            state_version=1,
            light_revision=1,
            lights=lights,
        ),
        emissive_transparent_rule=emissive_transparent_rule,
    )
    state.validate()
    return state
