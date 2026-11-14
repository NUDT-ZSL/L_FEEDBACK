"""确定性引擎原始录音合成器。

不依赖任何外部录音。给定固定随机种子后，两次运行逐样本一致：
基频通过对瞬时频率做积分得到连续相位，谐波结构固定，宽带噪声仅来自
种子化的 ``numpy.random.Generator``。
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SynthConfig:
    sample_rate: int = 44100
    duration_s: float = 22.0
    seed: int = 20260115
    rpm_start: float = 800.0
    rpm_end: float = 2400.0
    rpm_to_hz: float = 0.5          # 发动机阶次比例：f0 = rpm * rpm_to_hz
    # 固定谐波结构（相对基频的整数倍幅度），不含基频键 0
    harmonics: tuple = ((1, 1.0), (2, 0.55), (3, 0.33),
                        (4, 0.2), (5, 0.12), (6, 0.07))
    noise_gain: float = 0.018       # 宽带机械/燃烧噪声
    vibrato_depth: float = 0.004    # 轻微周期抖动（比例）
    vibrato_rate_hz: float = 7.5


@dataclass
class SynthResult:
    audio: np.ndarray               # float64, mono, 幅度约 +/-1
    time_s: np.ndarray
    f0_hz: np.ndarray               # 每个样本的瞬时基频轨迹
    rpm: np.ndarray                 # 每个样本的瞬时转速
    config: SynthConfig


def synthesize(config: SynthConfig | None = None) -> SynthResult:
    """生成一段确定性的引擎原始录音及其基频/转速轨迹。"""
    cfg = config or SynthConfig()
    if cfg.sample_rate <= 0:
        raise ValueError("sample_rate 必须为正整数")
    if cfg.duration_s <= 0:
        raise ValueError("duration_s 必须为正数")

    n = int(round(cfg.sample_rate * cfg.duration_s))
    t = np.arange(n, dtype=np.float64) / cfg.sample_rate

    # 线性上升的转速轨迹与基频轨迹。
    rpm = cfg.rpm_start + (cfg.rpm_end - cfg.rpm_start) * (t / cfg.duration_s)
    f0_base = rpm * cfg.rpm_to_hz

    # 确定性的轻微抖动（固定正弦，不引入随机源）。
    vibrato = 1.0 + cfg.vibrato_depth * np.sin(
        2.0 * np.pi * cfg.vibrato_rate_hz * t
    )
    inst_f = f0_base * vibrato

    # 对瞬时频率积分得到连续相位，保证逐样本可复现且无相位跳变。
    phase = 2.0 * np.pi * np.cumsum(inst_f) / cfg.sample_rate

    audio = np.zeros(n, dtype=np.float64)
    for order, amplitude in cfg.harmonics:
        audio += amplitude * np.sin(order * phase)

    # 固定种子的宽带噪声，两次运行逐样本一致。
    rng = np.random.default_rng(cfg.seed)
    audio += cfg.noise_gain * rng.standard_normal(n)

    # 整体缩放到与后续归一化安全配合的范围。
    audio = audio / 3.0
    return SynthResult(audio=audio, time_s=t, f0_hz=inst_f, rpm=rpm, config=cfg)


def f0_at_time(cfg: SynthConfig, time_s: float) -> float:
    """按合成器轨迹返回某时刻（秒）的中心基频（忽略微幅抖动）。"""
    ratio = time_s / cfg.duration_s
    rpm = cfg.rpm_start + (cfg.rpm_end - cfg.rpm_start) * ratio
    return rpm * cfg.rpm_to_hz


def time_of_rpm(cfg: SynthConfig, rpm_value: float) -> float:
    """按线性轨迹返回某转速对应的时间（秒）。"""
    span = cfg.rpm_end - cfg.rpm_start
    return cfg.duration_s * (rpm_value - cfg.rpm_start) / span
