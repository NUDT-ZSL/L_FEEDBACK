from __future__ import annotations

import json
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from radio_pipeline.audio import generate_source_pcm, pcm_bytes
from radio_pipeline.config import ConfigError, load_config
from radio_pipeline.jitter import FrameDecision
from radio_pipeline.packets import (
    INVALID_BASE64_TOKEN,
    PACKET_SCHEMA,
    ParsedArrival,
    WirePacket,
    generate_network_events,
    parse_packet_line,
)
from radio_pipeline.plc import render_plc
from radio_pipeline.receiver import receive_packets


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_default_config(self) -> None:
        config = load_config(ROOT / "config" / "default.json")
        self.assertEqual(config.audio.total_packets, 400)
        self.assertEqual(config.audio.total_samples, 384000)
        self.assertEqual(config.jitter_buffer.max_delay_ms, 160)

    def test_invalid_schema_values_are_rejected(self) -> None:
        config = json.loads((ROOT / "config" / "default.json").read_text("utf-8"))
        config["audio"]["duration_ms"] = 8010
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid_config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_duration_tolerance_is_at_most_one_sample(self) -> None:
        config = json.loads((ROOT / "config" / "default.json").read_text("utf-8"))
        config["output"]["duration_tolerance_samples"] = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid_tolerance.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)


class PacketTests(unittest.TestCase):
    def test_generator_is_deterministic(self) -> None:
        config = load_config(ROOT / "config" / "default.json")
        frames = generate_source_pcm(config.audio, config.seed)
        first = generate_network_events(frames, config.network, config.seed)
        second = generate_network_events(frames, config.network, config.seed)
        first_json = json.dumps(first, sort_keys=True, separators=(",", ":"))
        second_json = json.dumps(second, sort_keys=True, separators=(",", ":"))
        self.assertEqual(first_json, second_json)

    def test_invalid_payload_is_diagnosed(self) -> None:
        raw = json.dumps(
            {
                "schema": PACKET_SCHEMA,
                "sequence": 3,
                "timestamp_ms": 60,
                "arrival_ms": 70.0,
                "payload_b64": INVALID_BASE64_TOKEN,
                "payload_bytes": 1920,
                "checksum": "00000000",
            }
        )
        parsed = parse_packet_line(raw, 1, 1920)
        self.assertIsNone(parsed.packet)
        self.assertIsNotNone(parsed.issue)
        self.assertIn("base64", parsed.issue.diagnostic)


def _packet(sequence: int, arrival_ms: float, seed: int = 0) -> ParsedArrival:
    samples = np.arange(960, dtype=np.int32) + seed
    payload = ((samples * 13) % 1000 - 500).astype("<i2").tobytes()
    checksum = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
    return ParsedArrival(
        packet=WirePacket(
            sequence=sequence,
            timestamp_ms=sequence * 20,
            arrival_ms=arrival_ms,
            payload=payload,
            checksum=checksum,
            line_number=sequence + 1,
        ),
        issue=None,
    )


def _invalid(sequence: int, arrival_ms: float) -> ParsedArrival:
    raw = json.dumps(
        {
            "schema": PACKET_SCHEMA,
            "sequence": sequence,
            "timestamp_ms": sequence * 20,
            "arrival_ms": arrival_ms,
            "payload_b64": INVALID_BASE64_TOKEN,
        }
    )
    return parse_packet_line(raw, sequence + 1, 1920)


class ReceiverTests(unittest.TestCase):
    def test_duplicate_timeout_loss_and_invalid_handling(self) -> None:
        config = load_config(ROOT / "config" / "default.json")
        arrivals = [
            _packet(0, 8.0, 0),
            _invalid(1, 9.0),
            _packet(1, 10.0, 1),
            _packet(1, 12.0, 1),
            _packet(4, 20.0, 4),
            _packet(4, 30.0, 4),
            _packet(5, 28.0, 5),
            _packet(3, 500.0, 3),
        ]
        result = receive_packets(
            arrivals,
            total_packets=6,
            jitter_config=config.jitter_buffer,
            plc_config=config.plc,
        )
        statuses = [event["status"] for event in result.event_logs]
        self.assertEqual(statuses.count("invalid"), 1)
        self.assertEqual(statuses.count("duplicate"), 2)
        self.assertEqual(statuses.count("timeout"), 1)
        self.assertEqual(statuses.count("lost"), 1)
        self.assertEqual([packet.sequence for packet in result.accepted_packets], [0, 1, 4, 5])
        sources = [frame.outcome for frame in result.frames]
        self.assertEqual(sources, ["received", "received", "lost", "timeout", "received", "received"])
        timestamps = [packet.timestamp_ms for packet in result.accepted_packets]
        self.assertEqual(timestamps, sorted(timestamps))


class PlcTests(unittest.TestCase):
    def test_plc_preserves_timeline_and_fades_edges(self) -> None:
        config = load_config(ROOT / "config" / "default.json")
        frames: list[FrameDecision] = []
        for sequence in range(5):
            packet = None
            if sequence in {0, 4}:
                payload = pcm_bytes(
                    (
                        np.sin(np.arange(960) * 2 * np.pi * 220 / 48000) * 7000
                    ).astype("<i2")
                )
                packet = WirePacket(
                    sequence=sequence,
                    timestamp_ms=sequence * 20,
                    arrival_ms=sequence * 20 + 8,
                    payload=payload,
                    checksum="",
                    line_number=sequence + 1,
                )
            frames.append(
                FrameDecision(
                    sequence=sequence,
                    timestamp_ms=sequence * 20,
                    outcome="received" if packet else "lost",
                    packet=packet,
                    buffer_water_ms=60,
                    missing=packet is None,
                    low_water_plc=packet is None,
                    fast_consume=False,
                    conservative=False,
                    plc_aggression=1.0,
                )
            )
        result = render_plc(frames, config.audio, config.plc, config.seed)
        self.assertEqual(result.pcm.size, 5 * 960)
        self.assertEqual(result.pitch_frames, 3)
        self.assertEqual(result.comfort_frames, 0)
        self.assertTrue(all(frame.plc_triggered for frame in frames[1:4]))
        self.assertTrue(all(frame.plc_strategy == "pitch" for frame in frames[1:4]))
        self.assertEqual(result.pcm[959], 0)
        self.assertEqual(result.pcm[960], 0)
        self.assertEqual(result.pcm[3839], 0)
        self.assertEqual(result.pcm[3840], 0)
        self.assertTrue(frames[1].fade_regions)
        self.assertTrue(frames[3].fade_regions)


if __name__ == "__main__":
    unittest.main()
