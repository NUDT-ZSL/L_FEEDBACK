"""Deterministic offline demo for the render synchronization core."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from typing import Any, Dict, List, Optional

from .core import MAX_LOGIC_STEPS_PER_ADVANCE, RenderSyncCore
from .errors import ErrorCode, SyncError
from .model import Light
from .patch import PatchOperation


class DemoValidationError(RuntimeError):
    pass


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise DemoValidationError(message)


def run_demo(include_faults: bool = True) -> Dict[str, Any]:
    core = RenderSyncCore()
    frames: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    faults: List[Dict[str, Any]] = []
    checksum_chain = hashlib.sha256()
    replay_checks = 0

    def event(operation: str, result: Dict[str, Any]) -> None:
        events.append({"kind": operation, **result})

    def render(label: str, offset_ticks: int) -> Dict[str, Any]:
        nonlocal replay_checks
        state = core.snapshot
        _assert(0 <= offset_ticks < 10_000, "render offset outside logic interval")
        patch = core.pending_patch()
        replayed = core.replay_patch(patch)
        _assert(
            replayed.to_dict() == core.full_snapshot(),
            "patch replay did not match full snapshot",
        )
        replay_checks += 1
        frame = core.submit_render(
            state.logic_time_ticks + offset_ticks,
            ticket=core.current_ticket(),
        )
        _assert(
            frame.light_snapshot_version == frame.state_version,
            "render used a light snapshot from another state version",
        )
        record = frame.to_dict()
        record["label"] = label
        frames.append(record)
        checksum_chain.update(frame.checksum.encode("ascii"))
        checksum_chain.update(b"\n")
        return record

    event("initial_state", {"state_version": core.state_version})
    first = render("boot", 0)
    _assert(first["patch"]["operation_count"] == 0, "initial patch should be empty")

    result = core.advance_animation(6)
    event(result.operation, result.to_dict())
    result = core.update_material_params(
        "base_color", (700_000, 600_000, 500_000, 1_000_000)
    )
    event(result.operation, result.to_dict())
    render("base_color_hot_update", 1_000)

    result = core.set_material_enabled("emissive", True)
    event(result.operation, result.to_dict())
    result = core.update_material_params(
        "emissive", (300_000, 120_000, 60_000, 2_000_000)
    )
    event(result.operation, result.to_dict())
    emissive_on = render("emissive_on", 5_000)
    _assert(
        emissive_on["render_cache"]["emissive"]["value_fp"] > 0,
        "enabled emissive cache was not populated",
    )

    result = core.advance_animation(4)
    event(result.operation, result.to_dict())
    result = core.set_animation_weight("breath", 700_000)
    event(result.operation, result.to_dict())
    render("animation_material_coupled", 9_000)

    result = core.disable_material("emissive")
    event(result.operation, result.to_dict())
    emissive_off = render("emissive_disabled", 2_000)
    _assert(
        emissive_off["render_cache"]["emissive"]["params"] == [0, 0, 0, 0]
        and emissive_off["render_cache"]["emissive"]["value_fp"] == 0,
        "emissive cache leaked after disable",
    )

    result = core.set_material_enabled("emissive", True)
    event(result.operation, result.to_dict())
    emissive_reenabled = render("emissive_reenabled_zero", 3_000)
    emissive_material = next(
        item
        for item in emissive_reenabled["materials"]
        if item["layer_type"] == "emissive"
    )
    _assert(
        emissive_material["params"] == [0, 0, 0, 0]
        and emissive_reenabled["render_cache"]["emissive"]["value_fp"] == 0,
        "re-enabled emissive reused historical parameters",
    )

    core.disable_material("emissive")
    result = core.set_material_enabled("transparent", True)
    event(result.operation, result.to_dict())
    result = core.update_material_params(
        "transparent", (200_000, 400_000, 600_000, 350_000)
    )
    event(result.operation, result.to_dict())
    transparent_on = render("transparent_on", 4_000)
    _assert(
        transparent_on["render_cache"]["transparent"]["value_fp"] == 350_000,
        "transparent cache was not populated",
    )
    result = core.remove_material("transparent")
    event(result.operation, result.to_dict())
    transparent_removed = render("transparent_removed_zero", 6_000)
    _assert(
        transparent_removed["render_cache"]["transparent"]["params"] == [0, 0, 0, 0]
        and transparent_removed["render_cache"]["transparent"]["value_fp"] == 0,
        "transparent cache leaked after removal",
    )

    result = core.add_light(
        Light(
            light_id="fill",
            color=(300_000, 700_000, 900_000),
            intensity_fp=750_000,
            position=(400_000, 100_000, 250_000),
        )
    )
    event(result.operation, result.to_dict())
    result = core.advance_animation(2)
    event(result.operation, result.to_dict())
    light_added = render("light_added", 500)
    _assert(
        {item["id"] for item in light_added["lights"]}
        == {"key", "rim", "fill"},
        "added light missing from render",
    )
    result = core.remove_light("rim")
    event(result.operation, result.to_dict())
    light_removed = render("light_removed", 1_500)
    _assert(
        {item["id"] for item in light_removed["lights"]} == {"key", "fill"},
        "removed light still active in render",
    )

    if include_faults:
        before_fault = core.state_version
        try:
            core.advance_animation(MAX_LOGIC_STEPS_PER_ADVANCE + 1)
            raise DemoValidationError("logic step fault was not rejected")
        except SyncError as exc:
            _assert(
                exc.code == ErrorCode.LOGIC_STEP_RANGE and exc.error.recoverable,
                "logic step fault had the wrong error contract",
            )
            faults.append({"kind": "logic_step_range", "error": exc.to_dict()})
        _assert(
            core.status == "recoverable_error"
            and core.state_version == before_fault,
            "logic step fault changed state or did not enter recovery",
        )
        recovery = core.rollback()
        _assert(
            recovery["status"] == "ready"
            and recovery["restored_version"] == before_fault,
            "logic step rollback did not restore version",
        )
        faults[-1]["recovery"] = recovery
        render("after_logic_fault_recovery", 7_000)

        result = core.advance_animation(1)
        event(result.operation, result.to_dict())
        good_patch = core.pending_patch()
        corrupt_patch = replace(
            good_patch,
            operations=good_patch.operations
            + (PatchOperation("set", ("not_a_state_field",), 1),),
        )
        try:
            core.replay_patch(corrupt_patch)
            raise DemoValidationError("corrupt patch replay was not rejected")
        except SyncError as exc:
            _assert(
                exc.code == ErrorCode.PATCH_REPLAY_FAILED
                and exc.error.recoverable,
                "corrupt patch had the wrong error contract",
            )
            faults.append(
                {"kind": "patch_replay_failure", "error": exc.to_dict()}
            )
        _assert(
            core.status == "recoverable_error",
            "patch replay failure did not enter recovery",
        )
        recovery = core.rollback()
        restored = core.full_snapshot()
        _assert(
            recovery["restored_version"] == core.last_valid_version
            and restored["state_version"] == recovery["restored_version"],
            "patch replay rollback version mismatch",
        )
        faults[-1]["recovery"] = recovery
        render("after_patch_fault_recovery", 8_000)

        result = core.advance_animation(1)
        event(result.operation, result.to_dict())
        expected_version = core.state_version
        core.inject_light_snapshot_version_mismatch()
        try:
            core.submit_render(
                core.snapshot.logic_time_ticks + 1_000,
                ticket=core.current_ticket(),
            )
            raise DemoValidationError("light snapshot mismatch was not rejected")
        except SyncError as exc:
            _assert(
                exc.code == ErrorCode.LIGHT_SNAPSHOT_VERSION_MISMATCH
                and exc.error.recoverable,
                "light snapshot fault had the wrong error contract",
            )
            faults.append(
                {"kind": "light_snapshot_version_mismatch", "error": exc.to_dict()}
            )
        _assert(
            core.status == "recoverable_error"
            and core.state_version == expected_version,
            "light snapshot mismatch did not enter recovery",
        )
        _assert(
            core.diagnostics()[-1]["event"] == "recoverable_error",
            "recoverable failure was not recorded in diagnostics",
        )
        recovery = core.rollback()
        _assert(
            recovery["restored_version"] == expected_version
            and core.snapshot.light_snapshot.state_version == expected_version,
            "light snapshot rollback did not restore a coherent version",
        )
        faults[-1]["recovery"] = recovery
        render("after_light_fault_recovery", 2_500)

        result = core.advance_animation(2)
        event(result.operation, result.to_dict())
        render("post_recovery_continue", 5_500)

    final_frame = frames[-1]
    final_state = core.snapshot
    _assert(core.status == "ready", "controller did not finish in ready state")
    _assert(
        final_state.light_snapshot.state_version == final_state.state_version,
        "final light snapshot version mismatch",
    )
    _assert(
        final_frame["state_version"] == final_state.state_version,
        "final frame/state version mismatch",
    )

    final_checksum = checksum_chain.hexdigest()
    return {
        "schema_version": 1,
        "demo": "layered_render_sync",
        "deterministic": True,
        "include_faults": include_faults,
        "frames": frames,
        "events": events,
        "faults": faults,
        "checks": {
            "patch_replays_equal_full_snapshot": True,
            "patch_replay_count": replay_checks,
            "emissive_cache_zero_after_disable": True,
            "emissive_zero_after_reenable": True,
            "transparent_cache_zero_after_remove": True,
            "light_snapshot_versions_coherent": True,
            "recoverable_faults_tested": 3 if include_faults else 0,
        },
        "final": {
            "exit_code": 0,
            "status": core.status,
            "state_version": final_state.state_version,
            "logic_frame": final_state.logic_frame,
            "animation_version": final_state.animation_version,
            "material_version": final_state.material_version,
            "light_revision": final_state.light_revision,
            "render_frame_count": len(frames),
            "last_frame_checksum": final_frame["checksum"],
            "final_render_checksum": final_checksum,
            "snapshot_digest": hashlib.sha256(
                json.dumps(
                    final_state.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
            "recovery_epoch": core.recovery_epoch,
            "diagnostic_count": len(core.diagnostics()),
        },
    }


def _render_text(report: Dict[str, Any]) -> str:
    lines = []
    for frame in report["frames"]:
        enabled = ",".join(
            item["layer_type"]
            for item in frame["materials"]
            if item["enabled"]
        )
        lines.append(
            "frame={frame:02d} label={label} state_version={state_version} "
            "logic_frame={logic_frame} render_time={render_time_ticks} "
            "anim_v={animation_version} mat_v={material_version} "
            "light_v={light_revision} enabled=[{enabled}] "
            "patch_ops={ops} checksum={checksum}".format(
                frame=frame["frame_index"],
                label=frame["label"],
                state_version=frame["state_version"],
                logic_frame=frame["logic_frame"],
                render_time_ticks=frame["render_time_ticks"],
                animation_version=frame["animation_version"],
                material_version=frame["material_version"],
                light_revision=frame["light_revision"],
                enabled=enabled,
                ops=frame["patch"]["operation_count"],
                checksum=frame["checksum"],
            )
        )
    for fault in report["faults"]:
        lines.append(
            "fault={kind} code={code} recoverable={recoverable} "
            "rolled_back_to={restored}".format(
                kind=fault["kind"],
                code=fault["error"]["code"],
                recoverable=str(fault["error"]["recoverable"]).lower(),
                restored=fault["recovery"]["restored_version"],
            )
        )
    final = report["final"]
    lines.append(
        "final status={status} state_version={state_version} "
        "logic_frame={logic_frame} frames={render_frame_count} "
        "final_render_checksum={checksum} exit_code={exit_code}".format(
            status=final["status"],
            state_version=final["state_version"],
            logic_frame=final["logic_frame"],
            render_frame_count=final["render_frame_count"],
            checksum=final["final_render_checksum"],
            exit_code=final["exit_code"],
        )
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic layered render synchronization demo"
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format",
    )
    parser.add_argument(
        "--no-faults",
        action="store_true",
        help="skip intentional fault injection and recovery",
    )
    args = parser.parse_args(argv)

    try:
        report = run_demo(include_faults=not args.no_faults)
    except SyncError as exc:
        print(
            json.dumps(
                {"type": "error", "error": exc.to_dict()},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        error = {
            "code": ErrorCode.INTERNAL.value,
            "message": "unexpected demo failure",
            "details": {"cause": f"{type(exc).__name__}: {exc}"},
            "recoverable": False,
            "state_version": None,
            "last_valid_version": None,
        }
        print(
            json.dumps(
                {"type": "error", "error": error},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2

    if args.format == "json":
        print(
            json.dumps(
                report,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
    else:
        print(_render_text(report))
    return int(report["final"]["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
