"""Render-state synchronization controller.

The controller is the single writer for the authoritative state. Rendering
consumes immutable snapshots and field-level deltas; no wall-clock reads or
floating-point animation values are used.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .errors import ErrorCode, SyncError, make_error
from .model import (
    ANIMATION_BLEND_MODES,
    ANIMATION_ORDER,
    FP_SCALE,
    LOGIC_STEP_TICKS,
    MATERIAL_BLEND_MODES,
    MATERIAL_ORDER,
    MATERIAL_PARAM_COUNTS,
    AnimationLayer,
    Light,
    LightSnapshot,
    MaterialLayer,
    RenderCache,
    RenderCacheChannel,
    StateSnapshot,
    make_default_state,
    material_cache_channel,
)
from .patch import (
    DeltaPatch,
    build_delta_patch,
    digest_to_dict,
    replay_delta_patch,
    state_digest,
)


RECOVERY_READY = "ready"
RECOVERY_ROLLBACK = "recoverable_error"
MAX_LOGIC_STEPS_PER_ADVANCE = 600


@dataclass(frozen=True)
class MutationResult:
    operation: str
    previous_version: int
    state_version: int
    changed: bool
    dirty_fields: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "previous_version": self.previous_version,
            "state_version": self.state_version,
            "changed": self.changed,
            "dirty_fields": list(self.dirty_fields),
        }


@dataclass(frozen=True)
class RenderTicket:
    state_version: int
    animation_version: int
    material_version: int
    light_revision: int
    logic_frame: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "state_version": self.state_version,
            "animation_version": self.animation_version,
            "material_version": self.material_version,
            "light_revision": self.light_revision,
            "logic_frame": self.logic_frame,
        }


@dataclass(frozen=True)
class RenderFrame:
    frame_index: int
    render_time_ticks: int
    state_version: int
    logic_frame: int
    logic_time_ticks: int
    animation_version: int
    material_version: int
    light_revision: int
    light_snapshot_version: int
    pose: Mapping[str, Tuple[int, int, int]]
    materials: Tuple[Mapping[str, Any], ...]
    render_cache: Mapping[str, Any]
    lights: Tuple[Mapping[str, Any], ...]
    patch_summary: Mapping[str, Any]
    checksum: str

    def payload_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "render_time_ticks": self.render_time_ticks,
            "state_version": self.state_version,
            "logic_frame": self.logic_frame,
            "logic_time_ticks": self.logic_time_ticks,
            "animation_version": self.animation_version,
            "material_version": self.material_version,
            "light_revision": self.light_revision,
            "light_snapshot_version": self.light_snapshot_version,
            "pose": {
                bone: list(position)
                for bone, position in sorted(self.pose.items())
            },
            "materials": [dict(item) for item in self.materials],
            "render_cache": copy.deepcopy(self.render_cache),
            "lights": [dict(item) for item in self.lights],
            "patch": dict(self.patch_summary),
        }

    def to_dict(self) -> Dict[str, Any]:
        data = self.payload_dict()
        data["checksum"] = self.checksum
        return data


def _signed_scale(value: int, weight_fp: int) -> int:
    numerator = value * weight_fp
    if numerator >= 0:
        return numerator // FP_SCALE
    return -((-numerator) // FP_SCALE)


def _signed_blend(first: int, second: int, weight_fp: int) -> int:
    return first + _signed_scale(second - first, weight_fp)


def _interpolate_fp(first: int, second: int, fraction_fp: int) -> int:
    return first + _signed_scale(second - first, fraction_fp)


def _dirty_names(paths: Iterable[Tuple[Any, ...]]) -> Tuple[str, ...]:
    return tuple("/".join(str(part) for part in path) for path in sorted(paths))


class RenderSyncCore:
    """Thread-safe deterministic render-state coordinator."""

    def __init__(self, initial_state: Optional[StateSnapshot] = None):
        state = make_default_state() if initial_state is None else initial_state
        try:
            state.validate()
        except ValueError as exc:
            raise make_error(
                ErrorCode.STATE_INVALID,
                "initial state validation failed",
                details={"cause": str(exc)},
                state_version=getattr(state, "state_version", None),
            ) from exc

        self._lock = threading.RLock()
        self._state = state
        self._base_snapshot = state
        self._last_valid = state
        self._dirty: set = set()
        self._status = RECOVERY_READY
        self._last_error: Optional[SyncError] = None
        self._diagnostics: List[Dict[str, Any]] = []
        self._diagnostic_sequence = 0
        self._render_frame_index = 0
        self._recovery_epoch = 0

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def state_version(self) -> int:
        with self._lock:
            return self._state.state_version

    @property
    def last_valid_version(self) -> int:
        with self._lock:
            return self._last_valid.state_version

    @property
    def recovery_epoch(self) -> int:
        with self._lock:
            return self._recovery_epoch

    @property
    def snapshot(self) -> StateSnapshot:
        with self._lock:
            return self._state

    @property
    def last_valid_snapshot(self) -> StateSnapshot:
        with self._lock:
            return self._last_valid

    def full_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state.to_dict())

    def query_layer_order(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "animation": {
                    "order": list(ANIMATION_ORDER),
                    "blend_modes": dict(ANIMATION_BLEND_MODES),
                },
                "material": {
                    "order": list(MATERIAL_ORDER),
                    "blend_modes": dict(MATERIAL_BLEND_MODES),
                    "param_counts": dict(MATERIAL_PARAM_COUNTS),
                },
                "emissive_transparent_rule": self._state.emissive_transparent_rule,
            }

    def current_ticket(self) -> RenderTicket:
        with self._lock:
            state = self._state
            return RenderTicket(
                state_version=state.state_version,
                animation_version=state.animation_version,
                material_version=state.material_version,
                light_revision=state.light_revision,
                logic_frame=state.logic_frame,
            )

    def diagnostics(self) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._diagnostics)

    def last_error(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return None if self._last_error is None else self._last_error.to_dict()

    def _record(
        self,
        event: str,
        code: str,
        message: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._diagnostic_sequence += 1
        self._diagnostics.append(
            {
                "sequence": self._diagnostic_sequence,
                "event": event,
                "code": code,
                "message": message,
                "details": copy.deepcopy(dict(details or {})),
                "status": self._status,
                "state_version": self._state.state_version,
                "last_valid_version": self._last_valid.state_version,
            }
        )

    def _ensure_ready(self) -> None:
        if self._status != RECOVERY_READY:
            if self._last_error is not None:
                raise self._last_error
            raise self._raise_runtime(
                ErrorCode.RECOVERABLE_ERROR_ACTIVE,
                "controller is in a recoverable error state",
            )

    def _raise_validation(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        error = make_error(
            code,
            message,
            details=details,
            recoverable=False,
            state_version=self._state.state_version,
            last_valid_version=self._last_valid.state_version,
        )
        self._record("business_error", code.value, message, details)
        raise error

    def _raise_runtime(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> SyncError:
        error = make_error(
            code,
            message,
            details=details,
            recoverable=True,
            state_version=self._state.state_version,
            last_valid_version=self._last_valid.state_version,
        )
        self._status = RECOVERY_ROLLBACK
        self._last_error = error
        self._record("recoverable_error", code.value, message, details)
        return error

    def _commit(
        self,
        new_state: StateSnapshot,
        dirty: Iterable[Tuple[Any, ...]],
    ) -> MutationResult:
        try:
            new_state.validate()
        except ValueError as exc:
            raise self._raise_runtime(
                ErrorCode.STATE_INVALID,
                "transaction produced an invalid state",
                {"cause": str(exc)},
            ) from exc

        previous_version = self._state.state_version
        dirty_tuple = tuple(tuple(path) for path in dirty)
        self._state = new_state
        self._last_valid = new_state
        self._dirty.update(dirty_tuple)
        return MutationResult(
            operation="unknown",
            previous_version=previous_version,
            state_version=new_state.state_version,
            changed=True,
            dirty_fields=_dirty_names(dirty_tuple),
        )

    def _bump_state(
        self,
        state: StateSnapshot,
        *,
        animation_layers: Optional[Tuple[AnimationLayer, ...]] = None,
        material_layers: Optional[Tuple[MaterialLayer, ...]] = None,
        render_cache: Optional[RenderCache] = None,
        lights: Optional[Tuple[Light, ...]] = None,
        logic_frame: Optional[int] = None,
        logic_time_ticks: Optional[int] = None,
        animation_bump: bool = False,
        material_bump: bool = False,
        light_bump: bool = False,
    ) -> StateSnapshot:
        new_state_version = state.state_version + 1
        new_lights = state.lights if lights is None else lights
        new_light_revision = state.light_revision + (1 if light_bump else 0)
        return replace(
            state,
            state_version=new_state_version,
            logic_frame=state.logic_frame if logic_frame is None else logic_frame,
            logic_time_ticks=(
                state.logic_time_ticks
                if logic_time_ticks is None
                else logic_time_ticks
            ),
            animation_version=state.animation_version
            + (1 if animation_bump else 0),
            material_version=state.material_version + (1 if material_bump else 0),
            light_revision=new_light_revision,
            animation_layers=(
                state.animation_layers
                if animation_layers is None
                else animation_layers
            ),
            material_layers=(
                state.material_layers
                if material_layers is None
                else material_layers
            ),
            render_cache=state.render_cache if render_cache is None else render_cache,
            lights=new_lights,
            light_snapshot=LightSnapshot(
                state_version=new_state_version,
                light_revision=new_light_revision,
                lights=new_lights,
            ),
        )

    def advance_animation(self, steps: int) -> MutationResult:
        """Advance the 60 Hz animation clock by an integer logical step count."""

        with self._lock:
            self._ensure_ready()
            if (
                isinstance(steps, bool)
                or not isinstance(steps, int)
                or steps < 1
                or steps > MAX_LOGIC_STEPS_PER_ADVANCE
            ):
                error = self._raise_runtime(
                    ErrorCode.LOGIC_STEP_RANGE,
                    "logic step count is outside the accepted range",
                    {
                        "steps": steps,
                        "minimum": 1,
                        "maximum": MAX_LOGIC_STEPS_PER_ADVANCE,
                    },
                )
                raise error

            old_state = self._state
            delta_ticks = steps * LOGIC_STEP_TICKS
            layers = tuple(
                replace(
                    layer,
                    current_time_ticks=layer.current_time_ticks + delta_ticks,
                    version=layer.version + 1,
                )
                for layer in old_state.animation_layers
            )
            new_state = self._bump_state(
                old_state,
                animation_layers=layers,
                logic_frame=old_state.logic_frame + steps,
                logic_time_ticks=old_state.logic_time_ticks + delta_ticks,
                animation_bump=True,
            )
            dirty = (
                ("state_version",),
                ("logic_frame",),
                ("logic_time_ticks",),
                ("animation_version",),
                ("animation_layers",),
                ("light_snapshot",),
            )
            result = self._commit(new_state, dirty)
            return replace(result, operation="advance_animation")

    def set_animation_weight(
        self, layer_id: str, weight_fp: int
    ) -> MutationResult:
        with self._lock:
            self._ensure_ready()
            layer = self._state.animation_map().get(layer_id)
            if layer is None:
                self._raise_validation(
                    ErrorCode.UNKNOWN_LAYER,
                    "unknown animation layer",
                    {"layer_id": layer_id},
                )
            if (
                isinstance(weight_fp, bool)
                or not isinstance(weight_fp, int)
                or not 0 <= weight_fp <= FP_SCALE
            ):
                self._raise_validation(
                    ErrorCode.ANIMATION_WEIGHT_RANGE,
                    "animation weight must be an integer fixed-point value",
                    {"weight_fp": weight_fp, "minimum": 0, "maximum": FP_SCALE},
                )
            assert layer is not None
            if layer.weight_fp == weight_fp:
                return MutationResult(
                    "set_animation_weight",
                    self._state.state_version,
                    self._state.state_version,
                    False,
                    (),
                )

            layers = tuple(
                replace(
                    item,
                    weight_fp=weight_fp,
                    version=item.version + 1,
                )
                if item.layer_id == layer_id
                else item
                for item in self._state.animation_layers
            )
            new_state = self._bump_state(
                self._state,
                animation_layers=layers,
                animation_bump=True,
            )
            result = self._commit(
                new_state,
                (
                    ("state_version",),
                    ("animation_version",),
                    ("animation_layers",),
                    ("light_snapshot",),
                ),
            )
            return replace(result, operation="set_animation_weight")

    def _validated_material_params(
        self, layer_type: str, params: Sequence[int]
    ) -> Tuple[int, ...]:
        if layer_type not in MATERIAL_PARAM_COUNTS:
            self._raise_validation(
                ErrorCode.UNKNOWN_LAYER,
                "unknown material layer",
                {"layer_type": layer_type},
            )
        if not isinstance(params, (tuple, list)):
            self._raise_validation(
                ErrorCode.MATERIAL_PARAM_TYPE,
                "material parameters must be a sequence of integers",
                {"layer_type": layer_type, "received_type": type(params).__name__},
            )
        expected = MATERIAL_PARAM_COUNTS[layer_type]
        if len(params) != expected:
            self._raise_validation(
                ErrorCode.MATERIAL_PARAM_LENGTH,
                "material parameter vector length is invalid",
                {
                    "layer_type": layer_type,
                    "expected": expected,
                    "actual": len(params),
                },
            )
        normalized = tuple(params)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in normalized):
            self._raise_validation(
                ErrorCode.MATERIAL_PARAM_TYPE,
                "material parameter vector must contain only integers",
                {"layer_type": layer_type, "params": list(normalized)},
            )
        return normalized

    def _validate_emissive_transparent_rule(
        self, materials: Tuple[MaterialLayer, ...]
    ) -> None:
        by_type = {layer.layer_type: layer for layer in materials}
        if (
            self._state.emissive_transparent_rule == "exclusive"
            and by_type["emissive"].enabled
            and by_type["transparent"].enabled
        ):
            self._raise_validation(
                ErrorCode.EMISSIVE_TRANSPARENT_CONFLICT,
                "emissive and transparent layers are mutually exclusive",
                {
                    "rule": "exclusive",
                    "emissive_version": by_type["emissive"].version,
                    "transparent_version": by_type["transparent"].version,
                },
            )

    def _cache_after_toggle(
        self,
        layer_type: str,
        updated_layer: MaterialLayer,
        old_cache: RenderCacheChannel,
        *,
        transition: bool,
    ) -> RenderCacheChannel:
        if layer_type not in ("emissive", "transparent"):
            return old_cache
        if transition:
            return material_cache_channel(
                updated_layer,
                generation=old_cache.generation + 1,
                force_clear=not updated_layer.enabled,
            )
        return material_cache_channel(
            updated_layer,
            generation=old_cache.generation,
        )

    def set_material_enabled(
        self, layer_type: str, enabled: bool
    ) -> MutationResult:
        """Enable/disable a material slot atomically.

        Re-enabling emissive/transparent always starts from zero parameters and
        a cleared render cache.
        """

        with self._lock:
            self._ensure_ready()
            if layer_type not in MATERIAL_ORDER:
                self._raise_validation(
                    ErrorCode.UNKNOWN_LAYER,
                    "unknown material layer",
                    {"layer_type": layer_type},
                )
            if not isinstance(enabled, bool):
                self._raise_validation(
                    ErrorCode.MATERIAL_PARAM_TYPE,
                    "material enabled flag must be boolean",
                    {"layer_type": layer_type, "enabled": enabled},
                )
            old_state = self._state
            old_layer = old_state.material_map()[layer_type]
            if old_layer.enabled == enabled:
                return MutationResult(
                    "set_material_enabled",
                    old_state.state_version,
                    old_state.state_version,
                    False,
                    (),
                )

            new_layer = replace(
                old_layer,
                enabled=enabled,
                version=old_layer.version + 1,
            )
            if layer_type in ("emissive", "transparent"):
                zero_length = MATERIAL_PARAM_COUNTS[layer_type]
                new_layer = replace(new_layer, params=(0,) * zero_length)

            materials = tuple(
                new_layer if layer.layer_type == layer_type else layer
                for layer in old_state.material_layers
            )
            self._validate_emissive_transparent_rule(materials)

            cache = old_state.render_cache
            if layer_type == "emissive":
                cache = replace(
                    cache,
                    emissive=self._cache_after_toggle(
                        layer_type,
                        new_layer,
                        cache.emissive,
                        transition=True,
                    ),
                )
            elif layer_type == "transparent":
                cache = replace(
                    cache,
                    transparent=self._cache_after_toggle(
                        layer_type,
                        new_layer,
                        cache.transparent,
                        transition=True,
                    ),
                )

            new_state = self._bump_state(
                old_state,
                material_layers=materials,
                render_cache=cache,
                material_bump=True,
            )
            result = self._commit(
                new_state,
                (
                    ("state_version",),
                    ("material_version",),
                    ("material_layers",),
                    ("render_cache",),
                    ("light_snapshot",),
                ),
            )
            return replace(result, operation="set_material_enabled")

    def enable_material(self, layer_type: str) -> MutationResult:
        return self.set_material_enabled(layer_type, True)

    def disable_material(self, layer_type: str) -> MutationResult:
        return self.set_material_enabled(layer_type, False)

    def update_material_params(
        self, layer_type: str, params: Sequence[int]
    ) -> MutationResult:
        with self._lock:
            self._ensure_ready()
            normalized = self._validated_material_params(layer_type, params)
            old_state = self._state
            old_layer = old_state.material_map()[layer_type]
            if old_layer.params == normalized:
                return MutationResult(
                    "update_material_params",
                    old_state.state_version,
                    old_state.state_version,
                    False,
                    (),
                )

            new_layer = replace(
                old_layer,
                params=normalized,
                version=old_layer.version + 1,
            )
            materials = tuple(
                new_layer if layer.layer_type == layer_type else layer
                for layer in old_state.material_layers
            )
            self._validate_emissive_transparent_rule(materials)

            cache = old_state.render_cache
            if layer_type == "emissive":
                cache = replace(
                    cache,
                    emissive=material_cache_channel(
                        new_layer, generation=cache.emissive.generation
                    ),
                )
            elif layer_type == "transparent":
                cache = replace(
                    cache,
                    transparent=material_cache_channel(
                        new_layer, generation=cache.transparent.generation
                    ),
                )

            new_state = self._bump_state(
                old_state,
                material_layers=materials,
                render_cache=cache,
                material_bump=True,
            )
            result = self._commit(
                new_state,
                (
                    ("state_version",),
                    ("material_version",),
                    ("material_layers",),
                    ("render_cache",),
                    ("light_snapshot",),
                ),
            )
            return replace(result, operation="update_material_params")

    def remove_material(self, layer_type: str) -> MutationResult:
        """Remove a fixed slot by disabling and clearing all historical data."""

        with self._lock:
            self._ensure_ready()
            if layer_type not in MATERIAL_ORDER:
                self._raise_validation(
                    ErrorCode.UNKNOWN_LAYER,
                    "unknown material layer",
                    {"layer_type": layer_type},
                )
            old_state = self._state
            old_layer = old_state.material_map()[layer_type]
            new_layer = replace(
                old_layer,
                enabled=False,
                params=(0,) * MATERIAL_PARAM_COUNTS[layer_type],
                version=old_layer.version + 1,
            )
            materials = tuple(
                new_layer if layer.layer_type == layer_type else layer
                for layer in old_state.material_layers
            )
            self._validate_emissive_transparent_rule(materials)

            cache = old_state.render_cache
            if layer_type == "emissive":
                cache = replace(
                    cache,
                    emissive=material_cache_channel(
                        new_layer,
                        generation=cache.emissive.generation + 1,
                        force_clear=True,
                    ),
                )
            elif layer_type == "transparent":
                cache = replace(
                    cache,
                    transparent=material_cache_channel(
                        new_layer,
                        generation=cache.transparent.generation + 1,
                        force_clear=True,
                    ),
                )

            new_state = self._bump_state(
                old_state,
                material_layers=materials,
                render_cache=cache,
                material_bump=True,
            )
            result = self._commit(
                new_state,
                (
                    ("state_version",),
                    ("material_version",),
                    ("material_layers",),
                    ("render_cache",),
                    ("light_snapshot",),
                ),
            )
            return replace(result, operation="remove_material")

    def add_light(self, light: Light) -> MutationResult:
        with self._lock:
            self._ensure_ready()
            try:
                light.validate()
            except (ValueError, TypeError) as exc:
                self._raise_validation(
                    ErrorCode.INVALID_LIGHT,
                    "light description is invalid",
                    {"cause": str(exc)},
                )
            if light.light_id in self._state.light_map():
                self._raise_validation(
                    ErrorCode.DUPLICATE_LIGHT,
                    "light id already exists",
                    {"light_id": light.light_id},
                )

            lights = tuple(
                sorted(
                    self._state.lights + (light,),
                    key=lambda item: item.light_id,
                )
            )
            new_state = self._bump_state(
                self._state,
                lights=lights,
                light_bump=True,
            )
            result = self._commit(
                new_state,
                (
                    ("state_version",),
                    ("light_revision",),
                    ("lights",),
                    ("light_snapshot",),
                ),
            )
            return replace(result, operation="add_light")

    def remove_light(self, light_id: str) -> MutationResult:
        with self._lock:
            self._ensure_ready()
            lights_by_id = self._state.light_map()
            if light_id not in lights_by_id:
                self._raise_validation(
                    ErrorCode.LIGHT_NOT_FOUND,
                    "light id does not exist",
                    {"light_id": light_id},
                )
            lights = tuple(
                light
                for light in self._state.lights
                if light.light_id != light_id
            )
            new_state = self._bump_state(
                self._state,
                lights=lights,
                light_bump=True,
            )
            result = self._commit(
                new_state,
                (
                    ("state_version",),
                    ("light_revision",),
                    ("lights",),
                    ("light_snapshot",),
                ),
            )
            return replace(result, operation="remove_light")

    def inject_light_snapshot_version_mismatch(self) -> None:
        """Inject a version mismatch without changing the last valid snapshot."""

        with self._lock:
            self._ensure_ready()
            snapshot = self._state.light_snapshot
            broken_version = max(0, snapshot.state_version - 1)
            self._state = replace(
                self._state,
                light_snapshot=replace(
                    snapshot, state_version=broken_version
                ),
            )
            self._dirty.add(("light_snapshot",))
            self._record(
                "fault_injected",
                "I_LIGHT_SNAPSHOT_VERSION",
                "injected light snapshot version mismatch",
                {
                    "expected": self._state.state_version,
                    "actual": broken_version,
                },
            )

    def pending_patch(self) -> DeltaPatch:
        with self._lock:
            self._ensure_ready()
            return build_delta_patch(
                self._base_snapshot,
                self._state,
                self._dirty,
            )

    def replay_patch(
        self,
        patch: DeltaPatch,
        *,
        base_snapshot: Optional[StateSnapshot] = None,
        verify_current: bool = True,
    ) -> StateSnapshot:
        """Replay a patch; any failure enters the recoverable error state."""

        with self._lock:
            self._ensure_ready()
            base = self._base_snapshot if base_snapshot is None else base_snapshot
            try:
                replayed = replay_delta_patch(base, patch)
                if verify_current and replayed != self._state:
                    raise make_error(
                        ErrorCode.PATCH_REPLAY_FAILED,
                        "replayed snapshot does not equal the current state",
                        details={
                            "replayed_digest": state_digest(replayed),
                            "current_digest": state_digest(self._state),
                        },
                        recoverable=True,
                        state_version=self._state.state_version,
                        last_valid_version=self._last_valid.state_version,
                    )
            except SyncError as exc:
                raise self._raise_runtime(
                    exc.code,
                    exc.error.message,
                    dict(exc.error.details),
                ) from exc
            return replayed

    def _validate_light_snapshot_for_render(self) -> None:
        state = self._state
        snapshot = state.light_snapshot
        if snapshot.state_version != state.state_version:
            raise self._raise_runtime(
                ErrorCode.LIGHT_SNAPSHOT_VERSION_MISMATCH,
                "light snapshot version does not match current state version",
                {
                    "snapshot_version": snapshot.state_version,
                    "state_version": state.state_version,
                },
            )
        if snapshot.light_revision != state.light_revision:
            raise self._raise_runtime(
                ErrorCode.LIGHT_SNAPSHOT_REVISION_MISMATCH,
                "light snapshot revision does not match current light revision",
                {
                    "snapshot_revision": snapshot.light_revision,
                    "light_revision": state.light_revision,
                },
            )
        if snapshot.lights != state.lights:
            raise self._raise_runtime(
                ErrorCode.LIGHT_SNAPSHOT_CONTENT_MISMATCH,
                "light snapshot content does not match current lights",
                {
                    "snapshot_light_ids": [
                        light.light_id for light in snapshot.lights
                    ],
                    "state_light_ids": [
                        light.light_id for light in state.lights
                    ],
                },
            )

    def _validate_ticket(self, ticket: RenderTicket) -> None:
        state = self._state
        expected = {
            "state_version": state.state_version,
            "animation_version": state.animation_version,
            "material_version": state.material_version,
            "light_revision": state.light_revision,
            "logic_frame": state.logic_frame,
        }
        actual = ticket.to_dict()
        if actual != expected:
            raise self._raise_runtime(
                ErrorCode.COUPLING_VERSION_MISMATCH,
                "render ticket is coupled to a different state revision",
                {"expected": expected, "actual": actual},
            )

    def _sample_layer(
        self, layer: AnimationLayer, local_time_ticks: int
    ) -> Tuple[int, int, int]:
        time_ticks = local_time_ticks % layer.duration_ticks
        keyframes = layer.keyframes
        for index in range(1, len(keyframes)):
            right = keyframes[index]
            if time_ticks <= right.time_ticks:
                left = keyframes[index - 1]
                span = right.time_ticks - left.time_ticks
                fraction = (
                    0
                    if span == 0
                    else ((time_ticks - left.time_ticks) * FP_SCALE) // span
                )
                return tuple(
                    _interpolate_fp(left.position[axis], right.position[axis], fraction)
                    for axis in range(3)
                )  # type: ignore[return-value]
        return keyframes[-1].position

    def _sample_pose(
        self, state: StateSnapshot, render_time_ticks: int
    ) -> Dict[str, Tuple[int, int, int]]:
        render_delta = render_time_ticks - state.logic_time_ticks
        pose: Dict[str, Tuple[int, int, int]] = {}
        for layer in state.animation_layers:
            sample_clock = max(0, layer.current_time_ticks + render_delta)
            sampled = self._sample_layer(layer, sample_clock)
            existing = pose.get(layer.bone, (0, 0, 0))
            if layer.blend_mode == "override":
                pose[layer.bone] = tuple(
                    _signed_blend(existing[axis], sampled[axis], layer.weight_fp)
                    for axis in range(3)
                )  # type: ignore[assignment]
            elif layer.blend_mode == "additive":
                pose[layer.bone] = tuple(
                    existing[axis] + _signed_scale(sampled[axis], layer.weight_fp)
                    for axis in range(3)
                )  # type: ignore[assignment]
            else:
                raise self._raise_runtime(
                    ErrorCode.INVALID_ANIMATION_LAYER,
                    "unsupported animation blend mode",
                    {"layer_id": layer.layer_id, "blend_mode": layer.blend_mode},
                )
        return pose

    def submit_render(
        self,
        render_time_ticks: int,
        *,
        ticket: Optional[RenderTicket] = None,
    ) -> RenderFrame:
        """Validate and commit one variable-rate render submission."""

        with self._lock:
            self._ensure_ready()
            if (
                isinstance(render_time_ticks, bool)
                or not isinstance(render_time_ticks, int)
                or render_time_ticks < 0
            ):
                self._raise_validation(
                    ErrorCode.RENDER_TIME_INVALID,
                    "render time must be a non-negative integer tick value",
                    {"render_time_ticks": render_time_ticks},
                )

            if ticket is not None:
                self._validate_ticket(ticket)

            self._validate_light_snapshot_for_render()
            try:
                self._state.validate()
            except ValueError as exc:
                raise self._raise_runtime(
                    ErrorCode.STATE_INVALID,
                    "current state is invalid",
                    {"cause": str(exc)},
                ) from exc

            try:
                patch = build_delta_patch(
                    self._base_snapshot,
                    self._state,
                    self._dirty,
                )
                replayed = replay_delta_patch(self._base_snapshot, patch)
            except SyncError as exc:
                raise self._raise_runtime(
                    ErrorCode.PATCH_REPLAY_FAILED,
                    "pre-submit patch replay failed",
                    {"cause": exc.to_dict()},
                ) from exc
            if replayed != self._state:
                raise self._raise_runtime(
                    ErrorCode.PATCH_REPLAY_FAILED,
                    "replayed snapshot does not equal current state",
                    {
                        "replayed_digest": state_digest(replayed),
                        "current_digest": state_digest(self._state),
                    },
                )

            state = self._state
            pose = self._sample_pose(state, render_time_ticks)
            materials = tuple(
                {
                    "layer_type": layer.layer_type,
                    "enabled": layer.enabled,
                    "params": list(layer.params),
                    "version": layer.version,
                    "blend_mode": layer.blend_mode,
                }
                for layer in state.material_layers
            )
            lights = tuple(
                {
                    "id": light.light_id,
                    **light.to_dict(),
                }
                for light in state.light_snapshot.lights
            )
            patch_summary = {
                "base_version": patch.base_version,
                "target_version": patch.target_version,
                "base_digest": patch.base_digest,
                "target_digest": patch.target_digest,
                "operation_count": len(patch.operations),
            }
            payload = {
                "frame_index": self._render_frame_index,
                "render_time_ticks": render_time_ticks,
                "state_version": state.state_version,
                "logic_frame": state.logic_frame,
                "logic_time_ticks": state.logic_time_ticks,
                "animation_version": state.animation_version,
                "material_version": state.material_version,
                "light_revision": state.light_revision,
                "light_snapshot_version": state.light_snapshot.state_version,
                "pose": {
                    bone: list(position)
                    for bone, position in sorted(pose.items())
                },
                "materials": [dict(item) for item in materials],
                "render_cache": state.render_cache.to_dict(),
                "lights": [dict(item) for item in lights],
                "patch": patch_summary,
            }
            checksum = digest_to_dict(payload)
            frame = RenderFrame(
                frame_index=self._render_frame_index,
                render_time_ticks=render_time_ticks,
                state_version=state.state_version,
                logic_frame=state.logic_frame,
                logic_time_ticks=state.logic_time_ticks,
                animation_version=state.animation_version,
                material_version=state.material_version,
                light_revision=state.light_revision,
                light_snapshot_version=state.light_snapshot.state_version,
                pose=pose,
                materials=materials,
                render_cache=state.render_cache.to_dict(),
                lights=lights,
                patch_summary=patch_summary,
                checksum=checksum,
            )
            self._base_snapshot = state
            self._last_valid = state
            self._dirty.clear()
            self._render_frame_index += 1
            return frame

    def rollback(self) -> Dict[str, Any]:
        """Restore the most recent valid snapshot and clear recoverable error."""

        with self._lock:
            previous_status = self._status
            target = self._last_valid
            try:
                target.validate()
            except ValueError as exc:
                error = self._raise_runtime(
                    ErrorCode.ROLLBACK_UNAVAILABLE,
                    "last valid snapshot cannot be restored",
                    {"cause": str(exc)},
                )
                raise error from exc
            self._state = target
            self._base_snapshot = target
            self._dirty.clear()
            self._status = RECOVERY_READY
            self._last_error = None
            self._recovery_epoch += 1
            self._record(
                "rollback",
                "I_ROLLBACK_COMPLETED",
                "restored the most recent valid snapshot",
                {
                    "previous_status": previous_status,
                    "restored_version": target.state_version,
                    "recovery_epoch": self._recovery_epoch,
                    "snapshot_digest": state_digest(target),
                },
            )
            return {
                "status": self._status,
                "restored_version": target.state_version,
                "restored_logic_frame": target.logic_frame,
                "recovery_epoch": self._recovery_epoch,
                "snapshot_digest": state_digest(target),
            }
