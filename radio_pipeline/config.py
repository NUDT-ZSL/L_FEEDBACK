from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError


DEFAULT_CONFIG: dict[str, Any] = {
    "schema": "radio-pipeline-config-v1",
    "seed": 20261115,
    "audio": {
        "sample_rate": 48000,
        "channels": 1,
        "frame_ms": 20,
        "duration_ms": 8000,
        "amplitude": 0.24,
        "base_frequency_hz": 220.0,
    },
    "network": {
        "ordering": "reordered",
        "reorder_probability": 0.22,
        "reorder_min_extra_ms": 35.0,
        "reorder_max_extra_ms": 130.0,
        "loss_rate": 0.045,
        "duplicate_rate": 0.045,
        "burst_length": 16,
        "burst_probability": 0.012,
        "arrival_jitter": {
            "distribution": "normal",
            "scale_ms": 18.0,
            "max_abs_ms": 110.0,
        },
        "base_latency_ms": 8.0,
        "corrupt_rate": 0.0,
        "forced_bursts": [{"start_sequence": 230, "length": 18}],
        "forced_late_sequences": [61],
        "forced_corrupt_sequences": [150],
    },
    "jitter_buffer": {
        "min_delay_ms": 60,
        "max_delay_ms": 160,
        "timeout_ms": 240,
        "fast_consume_step_ms": 20,
        "conservative_max_delay_ms": 320,
    },
    "plc": {
        "strategy": "adaptive",
        "fade_ms": 6.0,
        "pitch_min_hz": 80.0,
        "pitch_max_hz": 500.0,
        "pitch_decay": 0.94,
        "isolated_decay": 0.70,
        "comfort_noise_gain": 0.22,
        "conservative_aggression": 0.45,
    },
    "output": {
        "wav_name": "radio.wav",
        "ogg_name": "radio.ogg",
        "opus_bitrate_bps": 64000,
        "loudness_tolerance_lufs": 1.0,
        "duration_tolerance_samples": 1,
    },
}


@dataclass(frozen=True)
class ForcedBurst:
    start_sequence: int
    length: int


@dataclass(frozen=True)
class AudioSettings:
    sample_rate: int
    channels: int
    frame_ms: int
    duration_ms: int
    amplitude: float
    base_frequency_hz: float

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000

    @property
    def total_packets(self) -> int:
        return self.duration_ms // self.frame_ms

    @property
    def total_samples(self) -> int:
        return self.total_packets * self.frame_samples


@dataclass(frozen=True)
class ArrivalJitterSettings:
    distribution: str
    scale_ms: float
    max_abs_ms: float


@dataclass(frozen=True)
class NetworkSettings:
    ordering: str
    reorder_probability: float
    reorder_min_extra_ms: float
    reorder_max_extra_ms: float
    loss_rate: float
    duplicate_rate: float
    burst_length: int
    burst_probability: float
    arrival_jitter: ArrivalJitterSettings
    base_latency_ms: float
    corrupt_rate: float
    forced_bursts: tuple[ForcedBurst, ...]
    forced_late_sequences: tuple[int, ...]
    forced_corrupt_sequences: tuple[int, ...]


@dataclass(frozen=True)
class JitterSettings:
    min_delay_ms: int
    max_delay_ms: int
    timeout_ms: int
    fast_consume_step_ms: int
    conservative_max_delay_ms: int


@dataclass(frozen=True)
class PlcSettings:
    strategy: str
    fade_ms: float
    pitch_min_hz: float
    pitch_max_hz: float
    pitch_decay: float
    isolated_decay: float
    comfort_noise_gain: float
    conservative_aggression: float


@dataclass(frozen=True)
class OutputSettings:
    wav_name: str
    ogg_name: str
    opus_bitrate_bps: int
    loudness_tolerance_lufs: float
    duration_tolerance_samples: int


@dataclass(frozen=True)
class PipelineConfig:
    seed: int
    audio: AudioSettings
    network: NetworkSettings
    jitter_buffer: JitterSettings
    plc: PlcSettings
    output: OutputSettings
    normalized: dict[str, Any]
    source_path: str | None = None

    def canonical_json(self) -> str:
        return json.dumps(
            self.normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {key: _copy_value(value) for key, value in base.items()}
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = _copy_value(value)
    return result


def _copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a JSON object")
    return value


def _as_int(value: Any, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{label} must be >= {minimum}")
    return value


def _as_number(
    value: Any,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ConfigError(f"{label} must be <= {maximum}")
    return result


def _as_probability(value: Any, label: str) -> float:
    return _as_number(value, label, 0.0, 1.0)


def _as_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty string")
    return value


def _parse_forced_bursts(value: Any) -> tuple[ForcedBurst, ...]:
    if not isinstance(value, list):
        raise ConfigError("network.forced_bursts must be a list")
    bursts: list[ForcedBurst] = []
    for index, item in enumerate(value):
        item_map = _require_mapping(item, f"network.forced_bursts[{index}]")
        bursts.append(
            ForcedBurst(
                start_sequence=_as_int(
                    item_map.get("start_sequence"),
                    f"network.forced_bursts[{index}].start_sequence",
                    0,
                ),
                length=_as_int(
                    item_map.get("length"),
                    f"network.forced_bursts[{index}].length",
                    1,
                ),
            )
        )
    return tuple(bursts)


def _parse_int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be a list")
    return tuple(_as_int(item, f"{label}[]", 0) for item in value)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            supplied = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"invalid JSON in {config_path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {config_path}: {exc}") from exc

    if not isinstance(supplied, Mapping):
        raise ConfigError("configuration root must be a JSON object")
    merged = _deep_merge(DEFAULT_CONFIG, supplied)
    audio_map = _require_mapping(merged.get("audio"), "audio")
    network_map = _require_mapping(merged.get("network"), "network")
    jitter_map = _require_mapping(merged.get("jitter_buffer"), "jitter_buffer")
    plc_map = _require_mapping(merged.get("plc"), "plc")
    output_map = _require_mapping(merged.get("output"), "output")
    jitter_map_dist = _require_mapping(
        network_map.get("arrival_jitter"), "network.arrival_jitter"
    )

    sample_rate = _as_int(audio_map.get("sample_rate"), "audio.sample_rate", 8000)
    channels = _as_int(audio_map.get("channels"), "audio.channels", 1)
    frame_ms = _as_int(audio_map.get("frame_ms"), "audio.frame_ms", 1)
    duration_ms = _as_int(audio_map.get("duration_ms"), "audio.duration_ms", 1)
    if channels != 1:
        raise ConfigError("audio.channels must be 1 for the mandated mono pipeline")
    if sample_rate != 48000:
        raise ConfigError("audio.sample_rate must be 48000")
    if frame_ms != 20:
        raise ConfigError("audio.frame_ms must be 20")
    if duration_ms % frame_ms != 0:
        raise ConfigError("audio.duration_ms must be an exact multiple of audio.frame_ms")

    audio = AudioSettings(
        sample_rate=sample_rate,
        channels=channels,
        frame_ms=frame_ms,
        duration_ms=duration_ms,
        amplitude=_as_number(audio_map.get("amplitude"), "audio.amplitude", 0.001, 0.95),
        base_frequency_hz=_as_number(
            audio_map.get("base_frequency_hz"), "audio.base_frequency_hz", 20.0, 2000.0
        ),
    )

    ordering = _as_string(network_map.get("ordering"), "network.ordering")
    if ordering not in {"reordered", "sequential"}:
        raise ConfigError("network.ordering must be 'reordered' or 'sequential'")
    distribution = _as_string(
        jitter_map_dist.get("distribution"), "network.arrival_jitter.distribution"
    )
    if distribution not in {"normal", "uniform", "exponential", "none"}:
        raise ConfigError(
            "network.arrival_jitter.distribution must be normal, uniform, exponential, or none"
        )
    reorder_min = _as_number(
        network_map.get("reorder_min_extra_ms"),
        "network.reorder_min_extra_ms",
        0.0,
    )
    reorder_max = _as_number(
        network_map.get("reorder_max_extra_ms"),
        "network.reorder_max_extra_ms",
        reorder_min,
    )
    if reorder_max < reorder_min:
        raise ConfigError("network.reorder_max_extra_ms must be >= reorder_min_extra_ms")

    network = NetworkSettings(
        ordering=ordering,
        reorder_probability=_as_probability(
            network_map.get("reorder_probability"), "network.reorder_probability"
        ),
        reorder_min_extra_ms=reorder_min,
        reorder_max_extra_ms=reorder_max,
        loss_rate=_as_probability(network_map.get("loss_rate"), "network.loss_rate"),
        duplicate_rate=_as_probability(
            network_map.get("duplicate_rate"), "network.duplicate_rate"
        ),
        burst_length=_as_int(network_map.get("burst_length"), "network.burst_length", 1),
        burst_probability=_as_probability(
            network_map.get("burst_probability"), "network.burst_probability"
        ),
        arrival_jitter=ArrivalJitterSettings(
            distribution=distribution,
            scale_ms=_as_number(
                jitter_map_dist.get("scale_ms"),
                "network.arrival_jitter.scale_ms",
                0.0,
            ),
            max_abs_ms=_as_number(
                jitter_map_dist.get("max_abs_ms"),
                "network.arrival_jitter.max_abs_ms",
                0.0,
            ),
        ),
        base_latency_ms=_as_number(
            network_map.get("base_latency_ms"), "network.base_latency_ms", 0.0
        ),
        corrupt_rate=_as_probability(
            network_map.get("corrupt_rate"), "network.corrupt_rate"
        ),
        forced_bursts=_parse_forced_bursts(network_map.get("forced_bursts")),
        forced_late_sequences=_parse_int_tuple(
            network_map.get("forced_late_sequences"),
            "network.forced_late_sequences",
        ),
        forced_corrupt_sequences=_parse_int_tuple(
            network_map.get("forced_corrupt_sequences"),
            "network.forced_corrupt_sequences",
        ),
    )

    jitter_buffer = JitterSettings(
        min_delay_ms=_as_int(
            jitter_map.get("min_delay_ms"), "jitter_buffer.min_delay_ms", 0
        ),
        max_delay_ms=_as_int(
            jitter_map.get("max_delay_ms"), "jitter_buffer.max_delay_ms", 1
        ),
        timeout_ms=_as_int(jitter_map.get("timeout_ms"), "jitter_buffer.timeout_ms", 1),
        fast_consume_step_ms=_as_int(
            jitter_map.get("fast_consume_step_ms"),
            "jitter_buffer.fast_consume_step_ms",
            1,
        ),
        conservative_max_delay_ms=_as_int(
            jitter_map.get("conservative_max_delay_ms"),
            "jitter_buffer.conservative_max_delay_ms",
            1,
        ),
    )
    if jitter_buffer.max_delay_ms < jitter_buffer.min_delay_ms:
        raise ConfigError("jitter_buffer.max_delay_ms must be >= min_delay_ms")
    if jitter_buffer.conservative_max_delay_ms < jitter_buffer.max_delay_ms:
        raise ConfigError(
            "jitter_buffer.conservative_max_delay_ms must be >= max_delay_ms"
        )
    if jitter_buffer.timeout_ms <= jitter_buffer.min_delay_ms:
        raise ConfigError("jitter_buffer.timeout_ms must exceed min_delay_ms")

    plc_strategy = _as_string(plc_map.get("strategy"), "plc.strategy")
    if plc_strategy not in {"adaptive", "pitch", "comfort", "mute"}:
        raise ConfigError("plc.strategy must be adaptive, pitch, comfort, or mute")
    fade_ms = _as_number(plc_map.get("fade_ms"), "plc.fade_ms", 0.0, float(frame_ms))
    pitch_min = _as_number(plc_map.get("pitch_min_hz"), "plc.pitch_min_hz", 20.0)
    pitch_max = _as_number(plc_map.get("pitch_max_hz"), "plc.pitch_max_hz", pitch_min)
    if pitch_max <= pitch_min:
        raise ConfigError("plc.pitch_max_hz must exceed pitch_min_hz")
    pitch_max = min(pitch_max, sample_rate / 2.0)
    plc = PlcSettings(
        strategy=plc_strategy,
        fade_ms=fade_ms,
        pitch_min_hz=pitch_min,
        pitch_max_hz=pitch_max,
        pitch_decay=_as_number(
            plc_map.get("pitch_decay"), "plc.pitch_decay", 0.01, 1.0
        ),
        isolated_decay=_as_number(
            plc_map.get("isolated_decay"), "plc.isolated_decay", 0.01, 1.0
        ),
        comfort_noise_gain=_as_number(
            plc_map.get("comfort_noise_gain"), "plc.comfort_noise_gain", 0.0, 2.0
        ),
        conservative_aggression=_as_number(
            plc_map.get("conservative_aggression"),
            "plc.conservative_aggression",
            0.0,
            1.0,
        ),
    )

    wav_name = _as_string(output_map.get("wav_name"), "output.wav_name")
    ogg_name = _as_string(output_map.get("ogg_name"), "output.ogg_name")
    if Path(wav_name).name != wav_name or Path(ogg_name).name != ogg_name:
        raise ConfigError("output file names must not contain directory components")
    if not wav_name.lower().endswith(".wav"):
        raise ConfigError("output.wav_name must end with .wav")
    if not ogg_name.lower().endswith(".ogg"):
        raise ConfigError("output.ogg_name must end with .ogg")
    duration_tolerance_samples = _as_int(
        output_map.get("duration_tolerance_samples"),
        "output.duration_tolerance_samples",
        0,
    )
    if duration_tolerance_samples > 1:
        raise ConfigError("output.duration_tolerance_samples must be 0 or 1")
    output = OutputSettings(
        wav_name=wav_name,
        ogg_name=ogg_name,
        opus_bitrate_bps=_as_int(
            output_map.get("opus_bitrate_bps"), "output.opus_bitrate_bps", 6000
        ),
        loudness_tolerance_lufs=_as_number(
            output_map.get("loudness_tolerance_lufs"),
            "output.loudness_tolerance_lufs",
            0.0,
        ),
        duration_tolerance_samples=duration_tolerance_samples,
    )

    return PipelineConfig(
        seed=_as_int(merged.get("seed"), "seed", 0),
        audio=audio,
        network=network,
        jitter_buffer=jitter_buffer,
        plc=plc,
        output=output,
        normalized=merged,
        source_path=str(config_path.resolve()),
    )


def config_as_dict(config: PipelineConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "audio": asdict(config.audio),
        "network": asdict(config.network),
        "jitter_buffer": asdict(config.jitter_buffer),
        "plc": asdict(config.plc),
        "output": asdict(config.output),
    }
