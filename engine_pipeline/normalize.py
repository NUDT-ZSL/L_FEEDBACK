"""统一峰值余量归一化与跨片段响度一致性校验。"""

from dataclasses import dataclass

import numpy as np

from .errors import BakeError, ErrorCode


@dataclass
class NormalizationResult:
    segments: list           # 归一化后的 float64 数组列表
    gain: float              # 全片段统一线性增益
    peaks: list              # 各片段归一化后峰值
    rms_db: list             # 各片段 RMS 响度（dBFS）


def rms_db(signal: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(signal, dtype=np.float64) ** 2)))
    if rms <= 1e-12:
        return -np.inf
    return 20.0 * np.log10(rms)


def normalize_segments(segments: list, peak_ceiling: float,
                       loudness_tol_db: float) -> NormalizationResult:
    """对全部片段施加同一增益，满足统一峰值上限并校验响度差异。

    采用单一全局增益（而非各片段独立增益），从根本上保证相对响度不被
    改变；增益取“使最大峰值恰好不超过上限”的值。
    """
    if not (0.0 < peak_ceiling < 1.0):
        raise BakeError(
            ErrorCode.ERR_PEAK_CEILING,
            f"峰值上限 {peak_ceiling} 越界，必须位于开区间 (0, 1)",
        )

    pre_peaks = [float(np.max(np.abs(seg))) for seg in segments]
    max_peak = max(pre_peaks)
    if max_peak <= 0.0:
        raise BakeError(ErrorCode.ERR_LOOP_DETECT, "存在全零片段，无法归一化")

    gain = peak_ceiling / max_peak
    scaled = [seg * gain for seg in segments]

    peaks = [float(np.max(np.abs(seg))) for seg in scaled]
    if any(p > peak_ceiling + 1e-9 for p in peaks):
        raise BakeError(ErrorCode.ERR_PEAK_CEILING, "归一化后峰值超出上限")

    loudness = [rms_db(seg) for seg in scaled]
    spread = max(loudness) - min(loudness)
    if spread > loudness_tol_db + 1e-9:
        raise BakeError(
            ErrorCode.ERR_LOUDNESS_SPREAD,
            f"片段响度差异 {spread:.2f}dB 超过容差 {loudness_tol_db:.2f}dB",
        )

    return NormalizationResult(
        segments=scaled, gain=float(gain), peaks=peaks, rms_db=loudness
    )
