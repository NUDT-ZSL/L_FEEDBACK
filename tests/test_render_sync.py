from __future__ import annotations

import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from render_sync import (
    ANIMATION_BLEND_MODES,
    ANIMATION_ORDER,
    LOGIC_STEP_TICKS,
    MATERIAL_BLEND_MODES,
    MATERIAL_ORDER,
    Light,
    PatchOperation,
    RenderSyncCore,
    StateSnapshot,
    SyncError,
    make_default_state,
    replay_delta_patch,
    state_digest,
)
from render_sync.core import MAX_LOGIC_STEPS_PER_ADVANCE
from render_sync.errors import ErrorCode


ROOT = Path(__file__).resolve().parents[1]


def submit_at_logic_offset(core: RenderSyncCore, offset: int = 0):
    return core.submit_render(
        core.snapshot.logic_time_ticks + offset,
        ticket=core.current_ticket(),
    )


class StateModelTests(unittest.TestCase):
    def test_layer_order_and_blend_modes_are_queryable_and_fixed(self) -> None:
        core = RenderSyncCore()
        order = core.query_layer_order()
        self.assertEqual(tuple(order["animation"]["order"]), ANIMATION_ORDER)
        self.assertEqual(
            tuple(order["material"]["order"]), MATERIAL_ORDER
        )
        self.assertEqual(order["animation"]["blend_modes"], ANIMATION_BLEND_MODES)
        self.assertEqual(order["material"]["blend_modes"], MATERIAL_BLEND_MODES)
        self.assertEqual(
            tuple(layer.layer_id for layer in core.snapshot.animation_layers),
            ANIMATION_ORDER,
        )
        self.assertEqual(
            tuple(layer.layer_type for layer in core.snapshot.material_layers),
            MATERIAL_ORDER,
        )

    def test_full_snapshot_round_trip_is_field_equal(self) -> None:
        core = RenderSyncCore()
        core.advance_animation(3)
        core.enable_material("emissive")
        core.update_material_params(
            "emissive", (100_000, 200_000, 300_000, 2_000_000)
        )
        data = core.full_snapshot()
        restored = StateSnapshot.from_dict(data)
        self.assertEqual(restored, core.snapshot)
        self.assertEqual(restored.to_dict(), data)
        self.assertEqual(state_digest(restored), state_digest(core.snapshot))

    def test_snapshot_rejects_unknown_fields(self) -> None:
        data = RenderSyncCore().full_snapshot()
        data["unexpected"] = 1
        with self.assertRaises(ValueError):
            StateSnapshot.from_dict(data)


class MaterialValidationTests(unittest.TestCase):
    def test_invalid_param_length_is_structured_and_atomic(self) -> None:
        core = RenderSyncCore()
        before = core.snapshot
        with self.assertRaises(SyncError) as caught:
            core.update_material_params("emissive", (1, 2, 3))
        self.assertEqual(caught.exception.code, ErrorCode.MATERIAL_PARAM_LENGTH)
        self.assertFalse(caught.exception.error.recoverable)
        self.assertEqual(core.snapshot, before)
        self.assertEqual(core.status, "ready")

    def test_unknown_material_is_structured_and_atomic(self) -> None:
        core = RenderSyncCore()
        before = core.snapshot
        with self.assertRaises(SyncError) as caught:
            core.enable_material("not_a_layer")
        self.assertEqual(caught.exception.code, ErrorCode.UNKNOWN_LAYER)
        self.assertEqual(core.snapshot, before)

    def test_exclusive_emissive_transparent_rejects_without_mutation(self) -> None:
        core = RenderSyncCore()
        core.enable_material("emissive")
        before = core.snapshot
        version = core.state_version
        with self.assertRaises(SyncError) as caught:
            core.enable_material("transparent")
        self.assertEqual(
            caught.exception.code,
            ErrorCode.EMISSIVE_TRANSPARENT_CONFLICT,
        )
        self.assertEqual(core.snapshot, before)
        self.assertEqual(core.state_version, version)
        self.assertEqual(core.status, "ready")

    def test_stacked_rule_allows_both_layers_in_fixed_order(self) -> None:
        core = RenderSyncCore(make_default_state("stacked"))
        core.enable_material("emissive")
        core.update_material_params(
            "emissive", (100_000, 200_000, 300_000, 1_000_000)
        )
        core.enable_material("transparent")
        core.update_material_params(
            "transparent", (10_000, 20_000, 30_000, 400_000)
        )
        materials = core.snapshot.material_map()
        self.assertTrue(materials["emissive"].enabled)
        self.assertTrue(materials["transparent"].enabled)
        self.assertLess(
            MATERIAL_ORDER.index("emissive"),
            MATERIAL_ORDER.index("transparent"),
        )
        frame = submit_at_logic_offset(core)
        self.assertEqual(
            [item["layer_type"] for item in frame.materials],
            list(MATERIAL_ORDER),
        )


class PatchTests(unittest.TestCase):
    def test_patch_replay_reconstructs_exact_full_snapshot(self) -> None:
        core = RenderSyncCore()
        base = core.snapshot
        core.advance_animation(7)
        core.update_material_params(
            "base_color", (710_000, 610_000, 510_000, 1_000_000)
        )
        core.enable_material("emissive")
        core.update_material_params(
            "emissive", (100_000, 150_000, 200_000, 2_500_000)
        )
        core.add_light(
            Light(
                light_id="fill",
                color=(300_000, 700_000, 900_000),
                intensity_fp=750_000,
                position=(400_000, 100_000, 250_000),
            )
        )
        core.remove_light("rim")

        patch = core.pending_patch()
        replayed = replay_delta_patch(base, patch)
        self.assertEqual(replayed.to_dict(), core.full_snapshot())
        self.assertEqual(state_digest(replayed), state_digest(core.snapshot))
        self.assertEqual(patch.base_version, base.state_version)
        self.assertEqual(patch.target_version, core.state_version)

        paths = {operation.path for operation in patch.operations}
        self.assertNotIn(("schema_version",), paths)
        self.assertNotIn(("animation_order",), paths)
        self.assertFalse(any("keyframes" in path for path in paths))
        self.assertFalse(
            any(
                path[:3] == ("material_layers", "normal", "params")
                for path in paths
            )
        )
        self.assertTrue(
            any(
                path[:3] == ("material_layers", "emissive", "params")
                for path in paths
            )
        )
        self.assertIn(
            ("animation_layers", "locomotion", "current_time_ticks"),
            paths,
        )

    def test_empty_patch_after_commit(self) -> None:
        core = RenderSyncCore()
        submit_at_logic_offset(core)
        patch = core.pending_patch()
        self.assertEqual(patch.operations, ())
        self.assertEqual(patch.base_version, patch.target_version)
        self.assertEqual(patch.base_digest, patch.target_digest)

    def test_patch_replay_failure_enters_recovery_then_rolls_back(self) -> None:
        core = RenderSyncCore()
        core.advance_animation(2)
        good = core.pending_patch()
        corrupt = replace(
            good,
            operations=good.operations
            + (PatchOperation("set", ("unknown_field",), 1),),
        )
        version = core.state_version
        before_state = core.snapshot
        with self.assertRaises(SyncError) as caught:
            core.replay_patch(corrupt)
        self.assertEqual(caught.exception.code, ErrorCode.PATCH_REPLAY_FAILED)
        self.assertTrue(caught.exception.error.recoverable)
        self.assertEqual(core.status, "recoverable_error")
        self.assertEqual(core.state_version, version)
        self.assertEqual(core.snapshot, before_state)
        self.assertEqual(core.diagnostics()[-1]["event"], "recoverable_error")

        recovery = core.rollback()
        self.assertEqual(recovery["restored_version"], version)
        self.assertEqual(core.status, "ready")
        core.advance_animation(1)
        self.assertEqual(core.state_version, version + 1)

    def test_wrong_patch_base_is_rejected(self) -> None:
        core = RenderSyncCore()
        core.advance_animation(1)
        patch = core.pending_patch()
        wrong_base = core.snapshot
        with self.assertRaises(SyncError) as caught:
            replay_delta_patch(wrong_base, patch)
        self.assertEqual(caught.exception.code, ErrorCode.PATCH_BASE_MISMATCH)


class ClockAndRenderTests(unittest.TestCase):
    def test_logic_step_bounds_enter_recoverable_error(self) -> None:
        core = RenderSyncCore()
        core.advance_animation(2)
        version = core.state_version
        before = core.snapshot
        with self.assertRaises(SyncError) as caught:
            core.advance_animation(MAX_LOGIC_STEPS_PER_ADVANCE + 1)
        self.assertEqual(caught.exception.code, ErrorCode.LOGIC_STEP_RANGE)
        self.assertTrue(caught.exception.error.recoverable)
        self.assertEqual(core.status, "recoverable_error")
        self.assertEqual(core.snapshot, before)
        self.assertEqual(core.state_version, version)
        recovery = core.rollback()
        self.assertEqual(recovery["restored_version"], version)
        self.assertEqual(recovery["restored_logic_frame"], 2)
        core.advance_animation(1)
        self.assertEqual(core.snapshot.logic_frame, 3)
        self.assertEqual(
            core.snapshot.logic_time_ticks,
            3 * LOGIC_STEP_TICKS,
        )

    def test_variable_render_timestamps_sample_without_clock_drift(self) -> None:
        core = RenderSyncCore()
        first = submit_at_logic_offset(core, 0)
        second = submit_at_logic_offset(core, 5_000)
        third = submit_at_logic_offset(core, 9_999)
        self.assertNotEqual(first.pose, second.pose)
        self.assertNotEqual(second.pose, third.pose)
        self.assertEqual(
            [first.frame_index, second.frame_index, third.frame_index],
            [0, 1, 2],
        )
        self.assertEqual(core.snapshot.logic_frame, 0)

    def test_same_input_sequence_is_bit_identical(self) -> None:
        def run_sequence():
            core = RenderSyncCore()
            records = []
            for steps, timestamp in ((1, 0), (4, 3_000), (7, 8_000), (2, 1_500)):
                core.advance_animation(steps)
                if steps % 2:
                    core.update_material_params(
                        "base_color",
                        (800_000 - steps, 600_000, 500_000, 1_000_000),
                    )
                patch = core.pending_patch()
                records.append(patch.to_dict())
                frame = core.submit_render(
                    core.snapshot.logic_time_ticks + timestamp,
                    ticket=core.current_ticket(),
                )
                records.append(frame.to_dict())
            return records

        first = run_sequence()
        second = run_sequence()
        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first, sort_keys=True, separators=(",", ":")),
            json.dumps(second, sort_keys=True, separators=(",", ":")),
        )

    def test_stale_animation_material_ticket_is_rejected(self) -> None:
        core = RenderSyncCore()
        stale = core.current_ticket()
        core.advance_animation(1)
        core.update_material_params(
            "base_color", (600_000, 500_000, 400_000, 1_000_000)
        )
        with self.assertRaises(SyncError) as caught:
            core.submit_render(core.snapshot.logic_time_ticks + 1, ticket=stale)
        self.assertEqual(caught.exception.code, ErrorCode.COUPLING_VERSION_MISMATCH)
        self.assertTrue(caught.exception.error.recoverable)
        self.assertEqual(core.status, "recoverable_error")
        core.rollback()
        submit_at_logic_offset(core)


class CacheAndLightTests(unittest.TestCase):
    def test_emissive_cache_is_cleared_and_reenable_starts_at_zero(self) -> None:
        core = RenderSyncCore()
        core.enable_material("emissive")
        core.update_material_params(
            "emissive", (300_000, 200_000, 100_000, 2_000_000)
        )
        active = submit_at_logic_offset(core)
        self.assertGreater(active.render_cache["emissive"]["value_fp"], 0)
        generation = active.render_cache["emissive"]["generation"]

        core.disable_material("emissive")
        disabled = submit_at_logic_offset(core, 100)
        self.assertEqual(
            disabled.render_cache["emissive"]["params"], [0, 0, 0, 0]
        )
        self.assertEqual(disabled.render_cache["emissive"]["value_fp"], 0)
        self.assertGreater(
            disabled.render_cache["emissive"]["generation"], generation
        )
        self.assertEqual(
            core.snapshot.material_map()["emissive"].params, (0, 0, 0, 0)
        )

        core.enable_material("emissive")
        reenabled = submit_at_logic_offset(core, 200)
        material = next(
            item
            for item in reenabled.materials
            if item["layer_type"] == "emissive"
        )
        self.assertEqual(material["params"], [0, 0, 0, 0])
        self.assertEqual(reenabled.render_cache["emissive"]["value_fp"], 0)
        self.assertEqual(reenabled.render_cache["emissive"]["params"], [0, 0, 0, 0])

    def test_transparent_cache_is_cleared_after_remove(self) -> None:
        core = RenderSyncCore()
        core.enable_material("transparent")
        core.update_material_params(
            "transparent", (100_000, 200_000, 300_000, 450_000)
        )
        active = submit_at_logic_offset(core)
        self.assertEqual(
            active.render_cache["transparent"]["value_fp"], 450_000
        )
        core.remove_material("transparent")
        removed = submit_at_logic_offset(core, 100)
        self.assertEqual(
            removed.render_cache["transparent"]["params"], [0, 0, 0, 0]
        )
        self.assertEqual(removed.render_cache["transparent"]["value_fp"], 0)
        core.enable_material("transparent")
        reenabled = submit_at_logic_offset(core, 200)
        self.assertEqual(
            reenabled.render_cache["transparent"]["params"], [0, 0, 0, 0]
        )

    def test_light_add_remove_updates_versioned_snapshot(self) -> None:
        core = RenderSyncCore()
        initial_revision = core.snapshot.light_revision
        core.add_light(
            Light(
                light_id="fill",
                color=(300_000, 700_000, 900_000),
                intensity_fp=750_000,
                position=(400_000, 100_000, 250_000),
            )
        )
        added = core.snapshot
        self.assertEqual(added.light_revision, initial_revision + 1)
        self.assertEqual(
            added.light_snapshot.state_version, added.state_version
        )
        self.assertEqual(added.light_snapshot.lights, added.lights)
        frame = submit_at_logic_offset(core)
        self.assertIn("fill", {item["id"] for item in frame.lights})

        core.remove_light("rim")
        removed = core.snapshot
        self.assertEqual(removed.light_revision, initial_revision + 2)
        self.assertEqual(
            removed.light_snapshot.state_version, removed.state_version
        )
        self.assertNotIn("rim", {item["id"] for item in submit_at_logic_offset(core, 100).lights})

    def test_light_snapshot_mismatch_rejects_and_recovers(self) -> None:
        core = RenderSyncCore()
        core.advance_animation(1)
        version = core.state_version
        expected_light_version = core.snapshot.light_snapshot.state_version
        core.inject_light_snapshot_version_mismatch()
        with self.assertRaises(SyncError) as caught:
            submit_at_logic_offset(core, 100)
        self.assertEqual(
            caught.exception.code,
            ErrorCode.LIGHT_SNAPSHOT_VERSION_MISMATCH,
        )
        self.assertTrue(caught.exception.error.recoverable)
        self.assertEqual(core.status, "recoverable_error")
        self.assertEqual(core.state_version, version)
        diagnostics = core.diagnostics()
        self.assertEqual(
            diagnostics[-1]["code"],
            ErrorCode.LIGHT_SNAPSHOT_VERSION_MISMATCH.value,
        )
        recovery = core.rollback()
        self.assertEqual(recovery["restored_version"], version)
        self.assertEqual(
            core.snapshot.light_snapshot.state_version,
            expected_light_version,
        )
        core.advance_animation(1)
        frame = submit_at_logic_offset(core, 1_000)
        self.assertEqual(frame.light_snapshot_version, frame.state_version)


class DemoEntryTests(unittest.TestCase):
    def test_demo_cli_is_deterministic_and_exits_zero(self) -> None:
        command = [sys.executable, str(ROOT / "demo.py"), "--format", "json"]
        first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout, second.stdout)
        report = json.loads(first.stdout)
        self.assertTrue(report["deterministic"])
        self.assertEqual(report["final"]["exit_code"], 0)
        self.assertEqual(report["checks"]["patch_replay_count"], len(report["frames"]))
        self.assertEqual(len(report["faults"]), 3)
        self.assertEqual(report["final"]["status"], "ready")
        self.assertEqual(len(report["final"]["final_render_checksum"]), 64)

    def test_demo_without_faults_still_exits_zero(self) -> None:
        command = [
            sys.executable,
            str(ROOT / "demo.py"),
            "--format",
            "json",
            "--no-faults",
        ]
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["faults"], [])
        self.assertEqual(report["checks"]["recoverable_faults_tested"], 0)


if __name__ == "__main__":
    unittest.main()
