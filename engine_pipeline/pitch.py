"""时长保持的音高偏移。

直接时域重采样（变速播放）会同时改变时长并使接缝相位漂移。这里采用
经典两步法：

1. 相位声码器先做时间拉伸（time stretch），改变信号时长但保持音高；
2. 窗函数 sinc 重采样把拉伸后的信号缩回原长度，改变音高但恢复时长。

二者组合后，输出与输入等长（时长不变），而基频按 ``pitch_factor`` 缩放，
满足“禁止只改变播放速率而不做时长补偿”的要求。
"""

import numpy as np

from .errors import BakeError, ErrorCode


def estimate_f0(signal: np.ndarray, sample_rate: int,
                min_f0: float = 25.0, max_f0: float = 4000.0,
                expected_f0: float | None = None,
                tolerance_ratio: float = 0.35) -> float:
    """用归一化自相关 + 抛物线插值估计基频，失败时抛出业务错误。"""
    x = np.asarray(signal, dtype=np.float64)
    if x.size < 4:
        raise BakeError(ErrorCode.ERR_LOOP_DETECT, "信号过短，无法估计基频")

    x = x - np.mean(x)
    energy = np.dot(x, x)
    if energy <= 1e-10 * x.size:
        raise BakeError(ErrorCode.ERR_LOOP_DETECT, "信号近似静音，无法检测循环")

    if expected_f0 is not None:
        lo = max(min_f0, expected_f0 * (1.0 - tolerance_ratio))
        hi = min(max_f0, expected_f0 * (1.0 + tolerance_ratio))
        min_lag = max(1, int(sample_rate / hi))
        max_lag = min(x.size - 2, int(sample_rate / lo))
    else:
        min_lag = max(1, int(sample_rate / max_f0))
        max_lag = min(x.size - 2, int(sample_rate / min_f0))
    if max_lag <= min_lag:
        raise BakeError(ErrorCode.ERR_LOOP_DETECT, "基频搜索范围无效")

    # 全长度自相关（FFT 实现），取滞后区间。
    n_fft = 1 << (2 * x.size - 1).bit_length()
    spec = np.fft.rfft(x, n_fft)
    corr = np.fft.irfft(spec * np.conj(spec), n_fft)[: x.size]
    corr = corr / corr[0]

    search = corr[min_lag: max_lag + 1]
    peak_offset = int(np.argmax(search))
    lag = min_lag + peak_offset
    if corr[lag] <= 0.0:
        raise BakeError(ErrorCode.ERR_LOOP_DETECT, "自相关未发现周期性峰值")

    # 抛物线子样本插值，提升周期估计精度。
    if 1 <= peak_offset < search.size - 1:
        a = search[peak_offset - 1]
        b = search[peak_offset]
        c = search[peak_offset + 1]
        denom = (a - 2.0 * b + c)
        shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        lag = lag + float(np.clip(shift, -1.0, 1.0))

    return sample_rate / lag


def _hann_window(n: int) -> np.ndarray:
    if n == 1:
        return np.ones(1, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def phase_vocoder_stretch(signal: np.ndarray, factor: float,
                          n_fft: int = 4096) -> np.ndarray:
    """相位声码器时间拉伸。factor>1 变长、<1 变短，音高保持不变。"""
    if factor <= 0:
        raise ValueError("时间拉伸因子必须为正数")
    x = np.asarray(signal, dtype=np.float64)
    hop_a = n_fft // 4
    hop_s = int(round(hop_a * factor))
    if hop_s < 1:
        hop_s = 1

    window = _hann_window(n_fft)
    # 仅在头部补一个窗长，保证首个分析帧完整；帧覆盖直到信号末尾。
    pad = n_fft
    padded = np.pad(x, (pad, pad), mode="reflect" if x.size > n_fft
                    else "constant")
    n_frames = 1 + (x.size - 1) // hop_a
    # 输出长度按拉伸因子精确确定（仅基于信号本身，不含反射补零）。
    target_out_len = int(round(x.size * factor))
    expected_len = target_out_len + n_fft
    output = np.zeros(expected_len, dtype=np.float64)
    norm = np.zeros(expected_len, dtype=np.float64)
    w2 = window * window

    prev_phase = None
    phase_acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    omega = 2.0 * np.pi * np.arange(n_fft // 2 + 1) / n_fft

    for frame in range(n_frames):
        start = frame * hop_a
        chunk = padded[start: start + n_fft]
        if chunk.size < n_fft:
            chunk = np.pad(chunk, (0, n_fft - chunk.size))
        spectrum = np.fft.rfft(chunk * window)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        if prev_phase is None:
            phase_acc = phase
        else:
            delta = phase - prev_phase - omega * hop_a
            delta = delta - 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
            freq = (omega * hop_a + delta) / hop_a
            phase_acc = phase_acc + freq * hop_s
        prev_phase = phase

        synthesized = np.fft.irfft(magnitude * np.exp(1j * phase_acc), n_fft)
        out_start = frame * hop_s
        output[out_start: out_start + n_fft] += synthesized * window * window
        norm[out_start: out_start + n_fft] += w2

    safe = norm > 1e-6
    output[safe] /= norm[safe]
    # 去掉头部补零对应区域，再裁剪/补齐到精确的拉伸目标长度。
    head_cut = pad
    out = output[head_cut: head_cut + target_out_len]
    if out.size < target_out_len:
        out = np.pad(out, (0, target_out_len - out.size))
    return out[:target_out_len]


def resample_to_length(signal: np.ndarray, target_len: int,
                       half_taps: int = 8) -> np.ndarray:
    """窗函数 sinc 插值重采样到指定长度（改变采样率/音高，不改样本时长）。"""
    x = np.asarray(signal, dtype=np.float64)
    if target_len <= 0:
        raise ValueError("target_len 必须为正数")
    if x.size == target_len:
        return x.copy()

    ratio = (x.size - 1) / (target_len - 1)      # 源位置相对输出位置的比例
    src_pos = np.arange(target_len, dtype=np.float64) * ratio
    src_idx = np.floor(src_pos).astype(np.int64)
    frac = src_pos - src_idx

    padded = np.pad(x, (half_taps, half_taps), mode="reflect"
                    if x.size > half_taps else "constant")
    out = np.zeros(target_len, dtype=np.float64)
    taps = np.arange(-half_taps + 1, half_taps + 1)
    for i in range(target_len):
        center = src_idx[i] + half_taps
        d = taps - frac[i]
        kernel = np.sinc(d)
        window = 0.5 + 0.5 * np.cos(np.pi * d / half_taps)  # Hann 窗
        out[i] = np.sum(padded[center - half_taps + 1:
                                center + half_taps + 1] * kernel * window)
    return out


def pitch_shift_preserve_duration(signal: np.ndarray, sample_rate: int,
                                  pitch_factor: float) -> np.ndarray:
    """等长音高偏移：输出长度与输入完全一致，基频乘 pitch_factor。"""
    if pitch_factor <= 0:
        raise BakeError(ErrorCode.ERR_CONFIG, "音高比例必须为正数")
    x = np.asarray(signal, dtype=np.float64)
    original_len = x.size

    # 第一步：时间拉伸 pitch_factor 倍长度（相位声码器保持音高）；
    # 第二步：重采样压缩回原长度，音高乘 pitch_factor，且时长完全恢复。
    # 直接时域变速（仅第二步）会改变时长并使接缝相位漂移，故必须有第一步补偿。
    stretched = phase_vocoder_stretch(x, pitch_factor)
    shifted = resample_to_length(stretched, original_len)
    return shifted
