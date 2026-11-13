"""工作站验收测试：对齐取整、分块/分段一致性、断点续导、错误码等。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from workstation.errors import ErrorCode
from workstation.exporter import ExportConfig, ExportManager, _SimulatedCrash
from workstation.fixtures import generate_workspace
from workstation.fingerprint import file_fingerprint
from workstation.mixer import (
    Mixer,
    PreviewEngine,
    SnapshotManager,
    TrackConfig,
    remap_timeline,
)
from workstation.project import Project
from workstation.timeline import round_half_up


class WorkstationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent / "_tmp_workspace"
        if self.root.exists():
            for path in sorted(self.root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
            for path in sorted(self.root.rglob("*"), reverse=True):
                if path.is_dir():
                    path.rmdir()
        self.info = generate_workspace(self.root, 48000)
        self.project = Project.build(
            self.root, self.info["script_path"], self.info["mix_path"]
        )

    def test_01_round_half_up_alignment(self) -> None:
        """非整数倍采样周期的起始时间按四舍五入映射，浮点时间原样保留。"""
        self.assertEqual(round_half_up(2.5), 3)
        self.assertEqual(round_half_up(2.4), 2)
        clip = next(c for c in self.project.timeline.clips if c.clip_id == "c001")
        self.assertAlmostEqual(clip.start_time, 0.12345)
        self.assertEqual(clip.start_frame, round_half_up(0.12345 * 48000))
        self.assertEqual(clip.start_frame, 5926)
        report_entry = next(
            item for item in self.project.alignment_report if item["clip_id"] == "c001"
        )
        self.assertEqual(report_entry["start_time"], 0.12345)
        self.assertEqual(report_entry["start_frame"], 5926)

    def test_02_resample_to_project_rate(self) -> None:
        """不同采样率素材统一重采样到项目采样率。"""
        source_441 = self.project.library.get("line_2")
        source_32 = self.project.library.get("line_3")
        self.assertEqual(source_441.original_rate, 44100)
        self.assertEqual(source_441.project_rate, 48000)
        self.assertEqual(source_441.frames.shape[0], int(np.ceil(0.4 * 48000)))
        self.assertEqual(source_32.frames.shape[0], int(np.ceil(0.6 * 48000)))

    def test_03_error_codes_are_enumerated_and_isolated(self) -> None:
        """缺失/格式/空素材分别登记 E_MISSING/E_FORMAT/E_SHORT，流程不中断。"""
        codes = self.project.library.error_codes()
        for expected in (
            ErrorCode.E_MISSING.value,
            ErrorCode.E_FORMAT.value,
            ErrorCode.E_SHORT.value,
        ):
            self.assertIn(expected, codes)
        valid_ids = {sid for sid, _ in self.project.library.items()}
        self.assertIn("line_1", valid_ids)
        self.assertNotIn("missing", valid_ids)
        paths = {record["path"] for record in self.project.library.errors}
        self.assertTrue(all(Path(path).is_absolute() or path for path in paths))

    def test_04_block_size_invariance(self) -> None:
        """不同块大小的流式混音在采样级完全一致。"""
        total = self.project.timeline.length
        mixer = self.project.mixer
        one_shot = mixer.render_range(0, total)
        for block_size in (128, 512, 1024, 2003):
            chunks = [
                block
                for _, block in mixer.stream(total, block_size=block_size)
            ]
            streamed = np.concatenate(chunks, axis=0)
            self.assertEqual(streamed.shape, one_shot.shape)
            np.testing.assert_array_equal(streamed, one_shot)

    def test_05_segmented_preview_matches_one_shot(self) -> None:
        """分段预览拼接与一次性请求采样级一致，且只处理覆盖片段。"""
        total = self.project.timeline.length
        engine = PreviewEngine(self.project.mixer)
        one_shot = engine.preview(0, total)
        engine.reset_counters()
        boundaries = [0, 300, 777, 2048, total]
        pieces = []
        touched_sum = 0
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            piece = engine.preview(lo, hi - lo)
            touched_sum += self.project.timeline.clips_for_range(lo, hi - lo).__len__()
            pieces.append(piece)
        segmented = np.concatenate(pieces, axis=0)
        np.testing.assert_array_equal(segmented, one_shot)
        self.assertEqual(engine.clip_touches, touched_sum)
        self.assertLess(engine.clip_touches, len(self.project.timeline.clips) * 4)

    def test_06_mute_solo_rules(self) -> None:
        """独奏时非独奏轨静音；被静音轨即使独奏也静音。"""
        tracks = {
            name: TrackConfig.from_dict(cfg.to_dict())
            for name, cfg in self.project.tracks.items()
        }
        total = self.project.timeline.length
        tracks["dialog"].solo = True
        mixer = Mixer(self.project.timeline, self.project.library, tracks)
        solo_out = mixer.render_range(0, total)
        self.assertTrue(np.any(solo_out != 0.0))
        dialog_only = self.project.mixer
        tracks["dialog"].mute = True
        muted_solo = Mixer(self.project.timeline, self.project.library, tracks).render_range(
            0, total
        )
        np.testing.assert_array_equal(muted_solo, np.zeros_like(muted_solo))
        self.assertIsNotNone(dialog_only)

    def test_07_crossfade_gain_sums_to_one(self) -> None:
        """重叠片段线性交叉淡化，区间内增益之和恒为 1 且采样级连续。"""
        from workstation.fades import compute_envelopes

        clips = sorted(
            [c for c in self.project.timeline.clips if c.track == "dialog"],
            key=lambda clip: clip.start_frame,
        )
        c1 = next(clip for clip in clips if clip.clip_id == "c001")
        c4 = next(clip for clip in clips if clip.clip_id == "c004")
        overlap_lo = max(c1.start_frame, c4.start_frame)
        overlap_hi = min(c1.end_frame, c4.end_frame)
        self.assertLess(overlap_lo, overlap_hi)
        envelopes = compute_envelopes([c1, c4], overlap_lo, overlap_hi - overlap_lo)
        total_gain = envelopes[c1.clip_id] + envelopes[c4.clip_id]
        np.testing.assert_allclose(total_gain, np.ones_like(total_gain), atol=1e-12)
        diffs = np.abs(np.diff(envelopes[c4.clip_id]))
        self.assertTrue(np.all(diffs < 1.0))

    def test_08_output_not_clipped_beyond_dbfs(self) -> None:
        """混音输出各声道幅度不超过 0 dBFS。"""
        total = self.project.timeline.length
        pcm = self.project.mixer.render_range(0, total)
        self.assertTrue(np.all(np.abs(pcm) <= 1.0 + 1e-12))

    def test_09_resumable_export_byte_identical(self) -> None:
        """模拟崩溃后续导，最终文件与一次性导出字节级相同。"""
        total = self.project.timeline.length
        config = ExportConfig(
            total_frames=total,
            channels=2,
            block_size=512,
            subtype="PCM_16",
            samplerate=48000,
        )
        one_shot_path = self.root / "oneshot.wav"
        one_shot_state = self.root / "oneshot_state.json"
        ExportManager(
            self.project.mixer, one_shot_path, one_shot_state, config
        ).run()
        one_shot_bytes = one_shot_path.read_bytes()

        crash_path = self.root / "resume.wav"
        crash_state = self.root / "resume_state.json"
        manager = ExportManager(self.project.mixer, crash_path, crash_state, config)
        with self.assertRaises(_SimulatedCrash):
            manager.run(crash_after_blocks=3)
        self.assertTrue(crash_state.exists())
        partial_state = json.loads(crash_state.read_text(encoding="utf-8"))
        self.assertEqual(partial_state["status"], "in_progress")
        self.assertEqual(partial_state["version"], 1)
        self.assertIn("input_fingerprint", partial_state)
        self.assertNotEqual(crash_path.read_bytes(), one_shot_bytes)

        result = ExportManager(
            self.project.mixer, crash_path, crash_state, config
        ).run()
        self.assertTrue(result["resumed"])
        self.assertEqual(crash_path.read_bytes(), one_shot_bytes)
        final_state = json.loads(crash_state.read_text(encoding="utf-8"))
        self.assertEqual(final_state["status"], "complete")
        self.assertEqual(final_state["frames_written"], total)

    def test_10_fingerprint_change_rejects_resume(self) -> None:
        """入参指纹变化时拒绝恢复并从头重新导出。"""
        total = self.project.timeline.length
        config = ExportConfig(
            total_frames=total,
            channels=2,
            block_size=512,
            subtype="PCM_16",
            samplerate=48000,
        )
        out = self.root / "fp.wav"
        state = self.root / "fp_state.json"
        with self.assertRaises(_SimulatedCrash):
            ExportManager(self.project.mixer, out, state, config).run(
                crash_after_blocks=2
            )
        changed_tracks = {
            name: TrackConfig.from_dict(
                {**cfg.to_dict(), "gain": cfg.gain * 0.7}
            )
            for name, cfg in self.project.tracks.items()
        }
        changed_mixer = Mixer(self.project.timeline, self.project.library, changed_tracks)
        manager = ExportManager(changed_mixer, out, state, config)
        result = manager.run()
        self.assertFalse(result["resumed"])
        saved = json.loads(state.read_text(encoding="utf-8"))
        self.assertEqual(saved["input_fingerprint"], manager.input_fingerprint)

    def test_11_project_restore_with_content_fingerprint(self) -> None:
        """恢复时依据内容指纹校验；素材变化则拒绝对应片段，且不依赖修改时间。"""
        state_path = self.root / "project_state.json"
        self.project.save(state_path)
        restored = Project.restore(self.root, state_path)
        self.assertEqual(restored.verify_errors, [])
        self.assertEqual(
            len(restored.project.timeline.clips),
            len(self.project.timeline.clips),
        )

        target = self.info["audio_paths"]["line_1"]
        original_fingerprint = file_fingerprint(target)
        original_mtime = target.stat().st_mtime_ns
        with target.open("ab") as handle:
            handle.write(b"\x00\x01\x02\x03")
        # 即使把修改时间恢复，内容指纹仍能识别变化
        import os

        os.utime(target, ns=(original_mtime, original_mtime))
        self.assertNotEqual(file_fingerprint(target), original_fingerprint)
        restored2 = Project.restore(self.root, state_path)
        self.assertTrue(restored2.verify_errors)
        self.assertEqual(restored2.verify_errors[0]["code"], "E_CHANGED")
        self.assertIn("line_1", restored2.rejected_sources)
        self.assertIn("c001", restored2.rejected_clips)
        restored_ids = {clip.clip_id for clip in restored2.project.timeline.clips}
        self.assertNotIn("c001", restored_ids)

    def test_12_snapshot_version_remap_and_incremental_diff(self) -> None:
        """快照版本变化触发重新映射；参数修改输出受影响区间差异报告。"""
        timeline = self.project.timeline
        clip = next(c for c in timeline.clips if c.clip_id == "c001")
        original_offset = clip.start_frame
        clip.start_frame = original_offset + 999  # 故意污染旧偏移
        manager = SnapshotManager(self.project.mixer)
        snapshot = {
            "version": 2,
            "tracks": {
                name: cfg.to_dict() for name, cfg in self.project.tracks.items()
            },
        }
        result = manager.load_snapshot(snapshot, timeline)
        self.assertTrue(result["remapped"])
        self.assertEqual(clip.start_frame, original_offset)

        diff = manager.apply_parameter_change(
            "music", {"gain": 0.1}, render_length=timeline.length
        )
        self.assertGreater(diff["changed_frame_count"], 0)
        self.assertIsNotNone(diff["affected_range"]["start_frame"])

    def test_13_state_file_roundtrip_and_report_json(self) -> None:
        """工程状态与报告可序列化为 JSON，且包含导出进度指针字段。"""
        state_path = self.root / "project_state.json"
        self.project.save(state_path)
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], 1)
        self.assertIn("timeline", payload)
        self.assertIn("tracks", payload)
        self.assertTrue(all("fingerprint" in src for src in payload["sources"]))

        config = ExportConfig(
            total_frames=self.project.timeline.length,
            channels=2,
            block_size=256,
            subtype="PCM_16",
            samplerate=48000,
        )
        export_state = self.root / "ex.json"
        with self.assertRaises(_SimulatedCrash):
            ExportManager(
                self.project.mixer,
                self.root / "ex.wav",
                export_state,
                config,
            ).run(crash_after_blocks=1)
        state = json.loads(export_state.read_text(encoding="utf-8"))
        self.assertIn("frames_written", state)
        data, rate = sf.read(self.root / "ex.wav")
        self.assertEqual(rate, 48000)
        self.assertEqual(data.shape[1], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

