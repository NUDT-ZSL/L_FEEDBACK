"""端到端烘焙管线：参数校验、切分、音高偏移、循环、归一化与原子导出。"""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from .errors import BakeError, ErrorCode
from .loops import apply_equal_power_crossfade, detect_loop
from .normalize import normalize_segments
from .pitch import estimate_f0, pitch_shift_preserve_duration
from .synth import SynthConfig, synthesize


@dataclass
class RpmTable:
    start_rpm: float
    end_rpm: float
    step_rpm: float

    def values(self) -> list:
        count = int(round((self.end_rpm - self.start_rpm) / self.step_rpm))
        return [self.start_rpm + i * self.step_rpm for i in range(count + 1)]


@dataclass
class BakeConfig:
    output_dir: str = "out"
    sample_rate: int = 44100
    duration_s: float = 22.0
    seed: int = 20260115
    source_rpm_start: float = 800.0
    source_rpm_end: float = 2400.0
    source_rpm_to_hz: float = 0.2
    rpm_table: RpmTable = field(default_factory=lambda: RpmTable(1000.0, 2200.0, 400.0))
    target_rpm_to_hz: float = 0.18
    segment_seconds: float = 0.5
    crossfade_ratio: float = 0.125
    peak_ceiling: float = 0.89
    loudness_tol_db: float = 2.0
    target_loop_seconds: float = 0.08
    phase_error_tol_rad: float = 0.5


def validate_config(cfg: BakeConfig) -> None:
    """集中参数校验，任何非法值都以业务错误码终止（此时尚未写任何文件）。"""
    table = cfg.rpm_table
    if table.step_rpm <= 0:
        raise BakeError(ErrorCode.ERR_RPM_TABLE, "转速步进必须为正数")
    if table.end_rpm < table.start_rpm:
        raise BakeError(ErrorCode.ERR_RPM_TABLE, "终止转速不得小于起始转速")
    span = table.end_rpm - table.start_rpm
    quotient = span / table.step_rpm
    if abs(quotient - round(quotient)) > 1e-9:
        raise BakeError(ErrorCode.ERR_RPM_TABLE, "转速范围无法被步进整除")
    if not (0.0 < cfg.peak_ceiling < 1.0):
        raise BakeError(ErrorCode.ERR_PEAK_CEILING,
                        f"峰值上限 {cfg.peak_ceiling} 必须位于 (0, 1)")
    if not (0.0 < cfg.crossfade_ratio <= 0.25):
        raise BakeError(ErrorCode.ERR_CROSSFADE,
                        "交叉淡化比例必须位于 (0, 0.25]")
    if cfg.segment_seconds <= 0 or cfg.target_loop_seconds <= 0:
        raise BakeError(ErrorCode.ERR_CONFIG, "时长参数必须为正数")
    if cfg.target_loop_seconds >= cfg.segment_seconds:
        raise BakeError(ErrorCode.ERR_CONFIG, "循环时长必须小于片段时长")
    if cfg.target_rpm_to_hz <= 0 or cfg.source_rpm_to_hz <= 0:
        raise BakeError(ErrorCode.ERR_CONFIG, "转速-频率比例必须为正数")


def build_rpm_segments(cfg: BakeConfig, synth_cfg: SynthConfig) -> list:
    """把转速表映射到与基频轨迹对齐的切分区间（等长、连续、居中）。"""
    values = cfg.rpm_table.values()
    seg_len = int(round(cfg.segment_seconds * synth_cfg.sample_rate))
    half = seg_len / 2.0
    total = synth_cfg.sample_rate * synth_cfg.duration_s
    segments = []

    for rpm_value in values:
        span = synth_cfg.rpm_end - synth_cfg.rpm_start
        center_time = synth_cfg.duration_s * (rpm_value - synth_cfg.rpm_start) / span
        center_sample = center_time * synth_cfg.sample_rate
        start = int(round(center_sample - half))
        end = start + seg_len
        if start < 0 or end > total:
            raise BakeError(
                ErrorCode.ERR_SEGMENT_RANGE,
                f"转速 {rpm_value:.0f} 的切分区间 [{start}, {end}) 超出录音范围",
            )
        segments.append({
            "rpm": float(rpm_value),
            "start_sample": int(start),
            "end_sample": int(end),
            "start_time_s": start / synth_cfg.sample_rate,
            "end_time_s": end / synth_cfg.sample_rate,
            "center_time_s": center_time,
        })
    return segments


def run_bake(config: BakeConfig | None = None) -> dict:
    """执行完整烘焙，返回索引清单字典；同时把 WAV 与 JSON 原子写入磁盘。"""
    cfg = config or BakeConfig()
    validate_config(cfg)

    synth_cfg = SynthConfig(
        sample_rate=cfg.sample_rate,
        duration_s=cfg.duration_s,
        seed=cfg.seed,
        rpm_start=cfg.source_rpm_start,
        rpm_end=cfg.source_rpm_end,
        rpm_to_hz=cfg.source_rpm_to_hz,
    )
    sample_rate = synth_cfg.sample_rate
    synth = synthesize(synth_cfg)
    segment_specs = build_rpm_segments(cfg, synth_cfg)

    processed = []
    for spec in segment_specs:
        source = synth.audio[spec["start_sample"]: spec["end_sample"]].copy()
        measured_f0 = estimate_f0(source, sample_rate)
        target_f0 = spec["rpm"] * cfg.target_rpm_to_hz
        pitch_factor = target_f0 / measured_f0

        shifted = pitch_shift_preserve_duration(source, sample_rate, pitch_factor)
        # 等长保证：禁止只变速而不做时长补偿。
        if shifted.size != source.size:
            raise BakeError(ErrorCode.ERR_CONFIG, "音高偏移后片段时长发生变化")

        loop = detect_loop(
            shifted, sample_rate, target_f0,
            target_loop_s=cfg.target_loop_seconds,
            crossfade_ratio=cfg.crossfade_ratio,
        )
        if loop.phase_error_rad > cfg.phase_error_tol_rad:
            raise BakeError(
                ErrorCode.ERR_LOOP_DETECT,
                f"相位误差 {loop.phase_error_rad:.3f}rad 超过容差 "
                f"{cfg.phase_error_tol_rad:.3f}rad",
            )
        baked = apply_equal_power_crossfade(shifted, loop)

        processed.append({
            "spec": spec,
            "audio": baked,
            "measured_source_f0": measured_f0,
            "target_f0": target_f0,
            "pitch_factor": pitch_factor,
            "loop": loop,
        })

    norm = normalize_segments(
        [item["audio"] for item in processed],
        cfg.peak_ceiling,
        cfg.loudness_tol_db,
    )

    entries = []
    for idx, item in enumerate(processed):
        loop = item["loop"]
        entries.append({
            "index": idx,
            "file": f"segment_{idx:03d}.wav",
            "rpm": item["spec"]["rpm"],
            "source_interval": {
                "start_sample": item["spec"]["start_sample"],
                "end_sample": item["spec"]["end_sample"],
                "start_time_s": round(item["spec"]["start_time_s"], 6),
                "end_time_s": round(item["spec"]["end_time_s"], 6),
                "center_time_s": round(item["spec"]["center_time_s"], 6),
            },
            "measured_source_f0_hz": round(item["measured_source_f0"], 4),
            "target_f0_hz": round(item["target_f0"], 4),
            "pitch_factor": round(item["pitch_factor"], 6),
            "loop": {
                "start_sample": loop.loop_start,
                "end_sample": loop.loop_end,
                "length_samples": loop.loop_length,
                "cycle_count": loop.cycle_count,
                "period_samples": round(loop.period_samples, 4),
                "fade_samples": loop.fade_length,
                "phase_error_rad": round(loop.phase_error_rad, 6),
                "seam_mismatch": round(loop.seam_mismatch, 8),
            },
            "gain": round(norm.gain, 8),
            "peak": round(norm.peaks[idx], 8),
            "rms_dbfs": round(norm.rms_db[idx], 4),
        })

    manifest = {
        "format_version": 1,
        "sample_rate": sample_rate,
        "channels": 1,
        "sample_format": "PCM_16",
        "seed": cfg.seed,
        "segment_count": len(entries),
        "peak_ceiling": cfg.peak_ceiling,
        "uniform_gain": round(norm.gain, 8),
        "crossfade_ratio": cfg.crossfade_ratio,
        "target_rpm_to_hz": cfg.target_rpm_to_hz,
        "loudness_tol_db": cfg.loudness_tol_db,
        "segments": entries,
    }

    _atomic_export(cfg, manifest, norm.segments, sample_rate)
    return manifest


def _atomic_export(cfg: BakeConfig, manifest: dict, signals: list,
                   sample_rate: int) -> None:
    """在临时目录写全 WAV 与 JSON，成功后整体替换输出目录。

    任何中途异常都会清理临时目录，磁盘上不会出现半成品索引文件。
    """
    out_dir = os.path.abspath(cfg.output_dir)
    parent = os.path.dirname(out_dir) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=".bake_", dir=parent)
    try:
        for idx, signal in enumerate(signals):
            sf.write(
                os.path.join(tmp_dir, f"segment_{idx:03d}.wav"),
                signal.astype(np.float64),
                sample_rate,
                subtype="PCM_16",
            )
        with open(os.path.join(tmp_dir, "manifest.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2,
                      sort_keys=True)

        backup = out_dir + ".bak"
        if os.path.isdir(out_dir):
            if os.path.isdir(backup):
                shutil.rmtree(backup)
            os.replace(out_dir, backup)
        os.replace(tmp_dir, out_dir)
        if os.path.isdir(backup):
            shutil.rmtree(backup)
    except BaseException:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        raise


def build_pipeline(config: BakeConfig | None = None) -> dict:
    """``run_bake`` 的语义化别名。"""
    return run_bake(config)
