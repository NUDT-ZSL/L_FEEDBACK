#!/usr/bin/env python3
"""Generate deterministic automotive audio test material.

Only Python's standard library is used. PCM WAV files are produced with the
wave/struct modules; the IEEE-float WAV header is written with struct so no
third-party audio package is required.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import wave
from pathlib import Path
from typing import Iterable, Sequence

PI = math.pi


def _pcm_bytes(samples: Sequence[float], bits: int) -> bytes:
    if bits == 8:
        return bytes(
            max(0, min(255, round(value * 127.0 + 128.0)))
            for value in samples
        )
    if bits == 16:
        return b"".join(
            struct.pack("<h", max(-32768, min(32767, round(value * 32767.0))))
            for value in samples
        )
    if bits == 24:
        output = bytearray()
        for value in samples:
            integer = max(-8388608, min(8388607, round(value * 8388607.0)))
            if integer < 0:
                integer += 1 << 24
            output.extend(
                bytes((integer & 0xFF, (integer >> 8) & 0xFF, (integer >> 16) & 0xFF))
            )
        return bytes(output)
    raise ValueError(f"unsupported PCM width: {bits}")


def write_pcm_wav(path: Path, sample_rate: int, channels: int, bits: int,
                  samples: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(bits // 8)
        output.setframerate(sample_rate)
        output.writeframes(_pcm_bytes(samples, bits))


def write_float_wav(path: Path, sample_rate: int, channels: int,
                    samples: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = struct.pack("<" + "f" * len(samples), *samples)
    byte_rate = sample_rate * channels * 4
    block_align = channels * 4
    header = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 3, channels, sample_rate,
                                     byte_rate, block_align, 32)
    header += b"data" + struct.pack("<I", len(payload))
    path.write_bytes(header + payload)


def envelope(index: int, count: int, fade_in: float, fade_out: float) -> float:
    t = index / count
    fade_in = min(max(fade_in, 1e-9), 0.49)
    fade_out = min(max(fade_out, 1e-9), 0.49)
    if t < fade_in:
        return 0.5 - 0.5 * math.cos(PI * t / fade_in)
    if t > 1.0 - fade_out:
        return 0.5 - 0.5 * math.cos(PI * (1.0 - t) / fade_out)
    return 1.0


def sine(rate: int, count: int, frequency: float, amplitude: float,
         phase: float = 0.0) -> list[float]:
    angular = 2.0 * PI * frequency / rate
    return [amplitude * math.sin(angular * index + phase) for index in range(count)]


def mix_tones(rate: int, count: int,
              tones: Iterable[tuple[float, float, float]], count_envelope: bool,
              fade_in: float = 0.03, fade_out: float = 0.05) -> list[float]:
    output = [0.0] * count
    for frequency, amplitude, phase in tones:
        angular = 2.0 * PI * frequency / rate
        for index in range(count):
            output[index] += amplitude * math.sin(angular * index + phase)
    if count_envelope:
        for index in range(count):
            output[index] *= envelope(index, count, fade_in, fade_out)
    return output


def _read_u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _read_u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def read_wav(path: Path) -> tuple[int, int, list[float]]:
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {path}")

    offset = 12
    fmt: tuple[int, int, int, int] | None = None
    payload: bytes | None = None
    while offset + 8 <= len(data):
        chunk_id = data[offset:offset + 4]
        chunk_size = _read_u32(data, offset + 4)
        body = offset + 8
        end = body + chunk_size
        if end > len(data):
            raise ValueError(f"truncated {chunk_id!r} chunk in {path}")
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise ValueError("short fmt chunk")
            format_tag, channels, sample_rate, _, _, bits = struct.unpack_from(
                "<HHIIHH", data, body
            )
            fmt = (format_tag, channels, sample_rate, bits)
        elif chunk_id == b"data":
            payload = data[body:end]
        offset = end + (chunk_size & 1)

    if fmt is None or payload is None:
        raise ValueError(f"missing fmt or data chunk: {path}")
    format_tag, channels, sample_rate, bits = fmt
    bytes_per_sample = (bits + 7) // 8
    frame_bytes = bytes_per_sample * channels
    if frame_bytes <= 0:
        raise ValueError("invalid frame size")
    frame_count = len(payload) // frame_bytes
    samples: list[float] = []

    if format_tag == 1 and bits == 16:
        samples = [value / 32768.0 for value in struct.unpack(
            "<" + "h" * (frame_count * channels), payload[:frame_count * frame_bytes]
        )]
    elif format_tag == 1 and bits == 24:
        for offset in range(0, frame_count * frame_bytes, 3):
            integer = payload[offset] | (payload[offset + 1] << 8) | (
                payload[offset + 2] << 16
            )
            if integer & 0x800000:
                integer -= 1 << 24
            samples.append(integer / 8388608.0)
    elif format_tag == 3 and bits == 32:
        samples = list(struct.unpack(
            "<" + "f" * (frame_count * channels), payload[:frame_count * frame_bytes]
        ))
    else:
        raise ValueError(f"unsupported WAV format tag={format_tag} bits={bits}")
    return sample_rate, channels, samples


_TRUE_PEAK_COEFFICIENTS = (
    (0.0017089843750, -0.0291748046875, -0.0189208984375, -0.0083007812500),
    (0.0109863281250, 0.0292968750000, 0.0330810546875, 0.0148925781250),
    (-0.0196533203125, -0.0517578125000, -0.0582275390625, -0.0266113281250),
    (0.0332031250000, 0.0891113281250, 0.1015625000000, 0.0476074218750),
    (-0.0594482421875, -0.1665039062500, -0.2003173828125, -0.1022949218750),
    (0.1373291015625, 0.4650878906250, 0.7797851562500, 0.9721679687500),
    (0.9721679687500, 0.7797851562500, 0.4650878906250, 0.1373291015625),
    (-0.1022949218750, -0.2003173828125, -0.1665039062500, -0.0594482421875),
    (0.0476074218750, 0.1015625000000, 0.0891113281250, 0.0332031250000),
    (-0.0266113281250, -0.0582275390625, -0.0517578125000, -0.0196533203125),
    (0.0148925781250, 0.0330810546875, 0.0292968750000, 0.0109863281250),
    (-0.0083007812500, -0.0189208984375, -0.0291748046875, 0.0017089843750),
)


class Biquad:
    def __init__(self, b0: float, b1: float, b2: float,
                 a1: float, a2: float) -> None:
        self.b0, self.b1, self.b2 = b0, b1, b2
        self.a1, self.a2 = a1, a2
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process(self, value: float) -> float:
        output = (
            self.b0 * value + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1 - self.a2 * self.y2
        )
        self.x2, self.x1 = self.x1, value
        self.y2, self.y1 = self.y1, output
        return output


def true_peak_channel(channel: Sequence[float]) -> float:
    peak = max((abs(value) for value in channel), default=0.0)
    count = len(channel)
    for index in range(count):
        for phase in range(4):
            total = 0.0
            for tap, row in enumerate(_TRUE_PEAK_COEFFICIENTS):
                source = index + tap - 5
                if 0 <= source < count:
                    total += row[phase] * channel[source]
            peak = max(peak, abs(total))
    return peak


def measure_bs1770(sample_rate: int, channels: int,
                   samples: Sequence[float]) -> dict[str, float | int | None]:
    if sample_rate != 48000:
        raise ValueError("reference loudness implementation expects 48 kHz")
    if channels < 1:
        raise ValueError("at least one channel is required")
    left = [samples[index] for index in range(0, len(samples), channels)]
    right = (
        [samples[index + 1] for index in range(0, len(samples), channels)]
        if channels > 1 else left
    )
    if len(left) != len(right):
        right = right[:len(left)]

    shelf_l = Biquad(1.53512485958697, -2.69169618940638, 1.19839281085285,
                     -1.69065929318241, 0.73248077421585)
    shelf_r = Biquad(1.53512485958697, -2.69169618940638, 1.19839281085285,
                     -1.69065929318241, 0.73248077421585)
    high_l = Biquad(1.0, -2.0, 1.0, -1.99004745483398, 0.99007225036621)
    high_r = Biquad(1.0, -2.0, 1.0, -1.99004745483398, 0.99007225036621)
    weighted_l = [high_l.process(shelf_l.process(value)) for value in left]
    weighted_r = [high_r.process(shelf_r.process(value)) for value in right]

    tp_linear = max(true_peak_channel(left), true_peak_channel(right))
    true_peak = 20.0 * math.log10(tp_linear) if tp_linear > 0.0 else None

    block_frames = round(0.4 * sample_rate)
    step_frames = round(0.1 * sample_rate)
    blocks: list[tuple[float, float]] = []
    start = 0
    while start + block_frames <= len(left):
        energy = 0.0
        for index in range(start, start + block_frames):
            energy += weighted_l[index] ** 2 + weighted_r[index] ** 2
        energy /= block_frames
        loudness = -0.691 + 10.0 * math.log10(energy) if energy > 0 else -math.inf
        blocks.append((energy, loudness))
        start += step_frames

    above_absolute = [(energy, loudness) for energy, loudness in blocks
                      if loudness > -70.0]
    relative_gate = None
    integrated = None
    if above_absolute:
        mean_energy = sum(item[0] for item in above_absolute) / len(above_absolute)
        relative_gate = -0.691 + 10.0 * math.log10(mean_energy) - 10.0
        gated = [energy for energy, loudness in blocks
                 if loudness > -70.0 and loudness > relative_gate]
        if gated:
            integrated = -0.691 + 10.0 * math.log10(
                sum(gated) / len(gated)
            )
    return {
        "integrated_lufs": integrated,
        "true_peak_dbtp": true_peak,
        "true_peak_linear": tp_linear,
        "relative_gate_lufs": relative_gate,
        "total_blocks": len(blocks),
        "gated_blocks": len(gated) if above_absolute else 0,
    }


def generate(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    nav_rate = 44100
    nav_count = int(nav_rate * 4.0)
    nav = mix_tones(
        nav_rate, nav_count,
        [(880.0, 10 ** (-20 / 20), 0.2), (1760.0, 10 ** (-31 / 20), 1.1)],
        True, fade_in=0.08, fade_out=0.12,
    )
    write_pcm_wav(out_dir / "nav_44k1_16bit_mono.wav", nav_rate, 1, 16, nav)

    media_rate = 48000
    media_count = int(media_rate * 5.0)
    media_left = mix_tones(
        media_rate, media_count,
        [(440.0, 10 ** (-24 / 20), 0.4), (1320.0, 10 ** (-39 / 20), 0.0)],
        True, fade_in=0.1, fade_out=0.1,
    )
    media_right = mix_tones(
        media_rate, media_count,
        [(660.0, 10 ** (-26 / 20), 0.9), (1980.0, 10 ** (-41 / 20), 0.0)],
        True, fade_in=0.1, fade_out=0.1,
    )
    media = [value for pair in zip(media_left, media_right) for value in pair]
    write_pcm_wav(out_dir / "media_48k_24bit_stereo.wav", media_rate, 2, 24, media)

    phone_rate = 48000
    phone_count = int(phone_rate * 3.5)
    phone = mix_tones(
        phone_rate, phone_count,
        [(300.0, 10 ** (-21 / 20), 0.0), (1200.0, 10 ** (-31 / 20), 0.7)],
        True, fade_in=0.03, fade_out=0.05,
    )
    write_float_wav(out_dir / "phone_48k_float32_mono.wav", phone_rate, 1,
                    phone)

    system_rate = 44100
    system_count = int(system_rate * 1.2)
    beep = mix_tones(
        system_rate, system_count,
        [(1500.0, 10 ** (-18 / 20), 0.0), (3000.0, 10 ** (-33 / 20), 0.4)],
        True, fade_in=0.01, fade_out=0.2,
    )
    system = [value for sample in beep for value in (sample, sample)]
    write_pcm_wav(out_dir / "system_44k1_16bit_stereo.wav", system_rate, 2, 16,
                  system)

    reference_rate = 48000
    reference_count = int(reference_rate * 4.0)
    reference = sine(reference_rate, reference_count, 997.0,
                     10 ** (-20 / 20), 0.3)
    reference_stereo = [value for sample in reference for value in (sample, sample)]
    reference_path = out_dir / "reference_48k_24bit_stereo.wav"
    write_pcm_wav(reference_path, reference_rate, 2, 24, reference_stereo)

    reference_rate_read, reference_channels, reference_samples = read_wav(reference_path)
    reference_measurement = measure_bs1770(
        reference_rate_read, reference_channels, reference_samples
    )

    invalid_rate = 44100
    invalid_count = int(invalid_rate * 0.2)
    invalid_samples = mix_tones(
        invalid_rate, invalid_count, [(1000.0, 0.1, 0.0)], False
    )
    write_pcm_wav(out_dir / "unsupported_8bit.wav", invalid_rate, 1, 8,
                  invalid_samples)

    nav_bytes = (out_dir / "nav_44k1_16bit_mono.wav").read_bytes()
    (out_dir / "corrupt_truncated.wav").write_bytes(nav_bytes[:-4096])

    reference_json = {
        "reference_file": reference_path.name,
        "sample_rate": reference_rate_read,
        "channels": reference_channels,
        "integrated_lufs": reference_measurement["integrated_lufs"],
        "true_peak_dbtp": reference_measurement["true_peak_dbtp"],
        "true_peak_linear": reference_measurement["true_peak_linear"],
        "gated_blocks": reference_measurement["gated_blocks"],
        "total_blocks": reference_measurement["total_blocks"],
    }
    (out_dir / "reference.json").write_text(
        json.dumps(reference_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return reference_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    reference = generate(args.out_dir)
    print(json.dumps(reference, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
