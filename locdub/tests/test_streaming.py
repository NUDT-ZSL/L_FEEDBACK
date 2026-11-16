import numpy as np

from locdub.wsola import StreamingWSOLA

SR = 44100


def test_streaming_produces_finite_output():
    rng = np.random.default_rng(7)
    signal = (
        0.4 * np.sin(2 * np.pi * 220 * np.arange(SR) / SR)
        + 0.02 * rng.standard_normal(SR)
    )
    streamer = StreamingWSOLA(ratio=1.25, sample_rate=SR,
                              block_samples=SR // 2)
    chunks = []
    for start in range(0, len(signal), SR // 2):
        for frame in streamer.push(signal[start:start + SR // 2, None]):
            chunks.append(frame)
    for frame in streamer.flush():
        chunks.append(frame)
    assert chunks
    combined = np.concatenate(chunks)
    assert np.isfinite(combined).all()
    # Streaming path emits windowed frames advancing by the synthesis hop.
    assert len(combined) > 0
