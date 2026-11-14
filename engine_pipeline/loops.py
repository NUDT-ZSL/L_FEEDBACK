"""循环点检测、整数周期约束与等功率交叉淡化。

检测策略：先估计片段中央基频得到周期 T0，按目标循环时长取整得到整数
周期数；再在首尾一个周期的网格上搜索使接缝失配最小的起点，并分别估计
首尾局部周期，量化 chirp 漂移导致的累计相位误差。交叉淡化采用
等功率（equal-power）曲线，淡化区不超过循环长度的四分之一。
"""

from dataclasses import dataclass

import numpy as np

from .errors import BakeError, ErrorCode
from .pitch import estimate_f0


@dataclass
class LoopInfo:
    loop_start: int
    loop_end: int
    loop_length: int          # 样本数
    cycle_count: int          # 整数基频周期数
    period_samples: float     # 平均周期（样本，子样本精度）
    phase_error_rad: float    # 首尾接缝累计相位误差估计（弧度，已折回 [-pi, pi]）
    seam_mismatch: float      # 归一化接缝失配能量（0~1，越小越好）
    fade_length: int          # 交叉淡化区样本数


def _wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _instant_phase(signal: np.ndarray, f0_hint: float, sample_rate: int,
                   index: int) -> float:
    """用带通滤波后的解析信号估计基频在某样本处的瞬时相位。"""
    period = max(4.0, sample_rate / f0_hint)
    half = max(8, int(round(3.0 * period)))
    lo = max(0, index - half)
    hi = min(signal.size, index + half + 1)
    chunk = signal[lo:hi].astype(np.float64)
    if chunk.size < 8:
        return 0.0
    # 在基频附近做窄带复调制 + 低通，等效带通后取解析信号。
    tt = np.arange(lo, hi) / sample_rate
    mixed = chunk * np.exp(-1j * 2.0 * np.pi * f0_hint * tt)
    kernel = np.hanning(max(8, int(round(2.0 * period))))
    kernel = kernel / kernel.sum()
    analytic = np.convolve(mixed, kernel, mode="same")
    local = index - lo
    base = 2.0 * np.pi * f0_hint * (index / sample_rate)
    return float(_wrap_pi(base + np.angle(analytic[local])))


def detect_loop(signal: np.ndarray, sample_rate: int, target_f0: float,
                target_loop_s: float = 0.25,
                crossfade_ratio: float = 0.125) -> LoopInfo:
    """检测满足整数周期约束的循环区间并量化接缝相位误差。

    循环长度被严格约束为整数个局部基频周期（``loop_length == ncyc * T``），
    再在一个周期的相位网格上搜索接缝失配最小的起点，保证首尾同相。
    """
    x = np.asarray(signal, dtype=np.float64)
    n = x.size

    f0_center = estimate_f0(x, sample_rate, expected_f0=target_f0)
    if not (0.5 * target_f0 <= f0_center <= 1.5 * target_f0):
        raise BakeError(
            ErrorCode.ERR_LOOP_DETECT,
            f"检测基频 {f0_center:.1f}Hz 与目标 {target_f0:.1f}Hz 偏离过大",
        )

    period = sample_rate / f0_center
    target_len = target_loop_s * sample_rate
    ncyc = max(1, int(round(target_len / period)))
    loop_length = int(round(ncyc * period))
    win = max(4, int(round(period)))

    # 首尾各预留 >=3 个周期的引导/收尾，并保证淡化区不越界。
    margin = int(np.ceil(3.0 * period))
    min_start = margin
    max_start = n - margin - loop_length
    if max_start <= min_start:
        raise BakeError(ErrorCode.ERR_LOOP_DETECT, "片段过短，无法容纳有效循环")

    best = None
    # 在整个可行区间按一个周期的步长采样候选（候选间同相），再在最优点
    # 附近做逐样本精修；整数周期由固定 loop_length 严格保证。
    coarse_step = max(1, int(round(period)))
    candidates = list(range(min_start, max_start + 1, coarse_step))
    for start in candidates:
        end = start + loop_length
        head = x[start: start + win]
        tail = x[end - win: end]
        denom = np.dot(head, head) + np.dot(tail, tail) + 1e-12
        mismatch = float(np.sum((head - tail) ** 2) / denom)
        if best is None or mismatch < best[0]:
            best = (mismatch, start, end)

    _, coarse_start, _ = best
    refine_lo = max(min_start, coarse_start - coarse_step)
    refine_hi = min(max_start, coarse_start + coarse_step)
    for start in range(refine_lo, refine_hi + 1):
        end = start + loop_length
        head = x[start: start + win]
        tail = x[end - win: end]
        denom = np.dot(head, head) + np.dot(tail, tail) + 1e-12
        mismatch = float(np.sum((head - tail) ** 2) / denom)
        if mismatch < best[0]:
            best = (mismatch, start, end)

    if best is None:
        raise BakeError(ErrorCode.ERR_LOOP_DETECT, "循环点搜索未得到有效候选")

    mismatch, loop_start, loop_end = best

    # 首尾局部周期，量化线性 chirp 在循环内累计的相位漂移。
    region = max(8 * win, 2048)
    head_region = x[loop_start: min(n, loop_start + region)]
    tail_region = x[max(0, loop_end - region): loop_end]
    f0_head = estimate_f0(head_region, sample_rate, expected_f0=f0_center)
    f0_tail = estimate_f0(tail_region, sample_rate, expected_f0=f0_center)
    f0_mean = 0.5 * (f0_head + f0_tail)
    period_mean = sample_rate / f0_mean

    # 用解析信号瞬时相位直接测量接缝处首尾基频的真实相位差。
    phase_at_start = _instant_phase(x, f0_head, sample_rate, loop_start)
    phase_at_end = _instant_phase(x, f0_tail, sample_rate, loop_end)
    phase_error = abs(_wrap_pi(phase_at_end - phase_at_start))

    fade_length = max(1, int(round(crossfade_ratio * loop_length)))
    return LoopInfo(
        loop_start=int(loop_start),
        loop_end=int(loop_end),
        loop_length=int(loop_length),
        cycle_count=int(ncyc),
        period_samples=float(period_mean),
        phase_error_rad=float(phase_error),
        seam_mismatch=float(mismatch),
        fade_length=int(fade_length),
    )


def apply_equal_power_crossfade(signal: np.ndarray, loop: LoopInfo) -> np.ndarray:
    """在循环边界烘焙等功率交叉淡化，消除接缝爆音。

    淡化曲线满足 g_in^2 + g_out^2 = 1（整段能量恒定）。把淡出的尾部与
    淡入的头部对等混合后，循环回绕处的样本严格一致，实现无缝循环。
    """
    x = np.array(signal, dtype=np.float64, copy=True)
    f = loop.fade_length
    if f <= 0:
        raise BakeError(ErrorCode.ERR_CROSSFADE, "交叉淡化区长度必须为正")
    if f > loop.loop_length // 4:
        raise BakeError(
            ErrorCode.ERR_CROSSFADE,
            f"淡化区 {f} 超过循环长度 {loop.loop_length} 的四分之一",
        )

    s = loop.loop_start
    e = loop.loop_end
    if e + f > x.size or s < 0:
        raise BakeError(ErrorCode.ERR_CROSSFADE, "淡化区越界，无法施加交叉淡化")

    k = np.arange(f, dtype=np.float64) / max(1, f - 1)
    g_in = np.sqrt(k)
    g_out = np.sqrt(1.0 - k)

    # 仅改写循环起点处的头部淡化区，把它与循环结束后的尾部淡化区做等功率
    # 混合；尾部保持原样。于是回绕点两侧样本严格相等：
    # head'[i] = g_in*head[i] + g_out*tail[i]，而播放到 tail 时即同值淡出，
    # 满足 g_in^2 + g_out^2 = 1 的恒定能量且无接缝爆音。
    head = x[s: s + f].copy()
    tail = x[e: e + f].copy()
    x[s: s + f] = g_in * head + g_out * tail
    return x
