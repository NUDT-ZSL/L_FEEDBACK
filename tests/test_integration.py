from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from bcsubts.core import RunConfig, run_pipeline
from bcsubts.mpegts import (
    StreamInfo,
    build_pmt_section,
    extract_private_sections,
    find_pmt,
)
from bcsubts.mux import AuxAction, inject_actions
from bcsubts.samplegen import SampleSpec, generate_sample
from bcsubts.scte35 import ScteEvent, build_splice_info, parse_splice_info


def _find_tool(name: str) -> str | None:
    return os.environ.get("FFMPEG_BIN" if name == "ffmpeg" else "FFPROBE_BIN") or shutil.which(name)


@unittest.skipUnless(_find_tool("ffmpeg") and _find_tool("ffprobe"), "FFmpeg tools are not on PATH")
class PipelineIntegrationTests(unittest.TestCase):
    def test_full_pipeline_and_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            sample = generate_sample(
                root / "sample",
                SampleSpec("integration", 44100, 2, 950, "csv", 777),
            )
            config = RunConfig(
                audio=Path(sample["audio"]),
                subtitles=Path(sample["subtitles"]),
                scte35=Path(sample["scte35"]),
                output_dir=root / "out",
            )
            first = run_pipeline(config)
            self.assertEqual(first["status"], "ok")
            self.assertEqual(first["verification"]["scte35_event_count"], 3)
            self.assertEqual(first["verification"]["ffprobe_subtitle_codec"], "dvb_subtitle")
            self.assertLess(abs(first["report"]["alignment"]["estimated_offset_ms"] - 950), 20)
            ts_path = Path(first["outputs"]["mpeg_ts"])
            first_size = ts_path.stat().st_size

            first_start = first["report"]["subtitles"]["corrected_starts_ms"][0]
            decoded = subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{(first_start + 100) / 1000:.3f}",
                    "-i",
                    str(ts_path),
                    "-filter_complex",
                    "[0:v][0:s]overlay",
                    "-frames:v",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(decoded.returncode, 0, decoded.stderr.decode(errors="replace"))
            self.assertEqual(len(decoded.stdout), 720 * 576 * 3)
            self.assertGreater(max(decoded.stdout), 40)

            second = run_pipeline(config)
            self.assertEqual(second["verification"]["scte35_event_ids"], first["verification"]["scte35_event_ids"])
            self.assertEqual(ts_path.stat().st_size, first_size)

            # Exercise an interrupted action checkpoint and prove that the
            # resumed run does not insert the already-completed event twice.
            base_ts = root / "out" / ".work" / "base_audio_video.ts"
            pmt = find_pmt(base_ts.read_bytes())
            scte_pid = 0x1F0
            pmt_section = build_pmt_section(pmt, [StreamInfo(0x86, scte_pid)])
            event = ScteEvent(
                event_id=501,
                event_type="splice_insert",
                offset_ms=1000,
                duration_ms=1000,
                target_pts=90000,
            )
            action = AuxAction(
                action_id="scte:501",
                base_packet_index=100,
                target_ms=1000,
                kind="section",
                pid=scte_pid,
                payload=build_splice_info(event),
                event_id=501,
            )
            recovery_state: dict = {}
            interrupted = False

            def save_then_interrupt(value: dict) -> None:
                nonlocal interrupted
                snapshot = dict(value)
                recovery_state.clear()
                recovery_state.update(snapshot)
                if snapshot.get("completed_event_ids") and not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt

            partial = root / "recovery" / "partial.ts"
            final = root / "recovery" / "final.ts"
            with self.assertRaises(KeyboardInterrupt):
                inject_actions(
                    base_ts,
                    partial,
                    final,
                    [action],
                    pmt.pmt_pid,
                    pmt_section,
                    recovery_state,
                    save_then_interrupt,
                )
            self.assertEqual(recovery_state["completed_event_ids"], [501])
            self.assertGreater(recovery_state["partial_size"], 0)
            inject_actions(
                base_ts,
                partial,
                final,
                [action],
                pmt.pmt_pid,
                pmt_section,
                recovery_state,
                lambda value: recovery_state.update(value),
            )
            sections = extract_private_sections(final.read_bytes(), scte_pid)
            self.assertEqual(len(sections), 1)
            self.assertEqual(parse_splice_info(sections[0])["event_id"], 501)


if __name__ == "__main__":
    unittest.main()
