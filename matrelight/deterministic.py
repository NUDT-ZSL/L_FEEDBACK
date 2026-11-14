"""确定性运行上下文。

所有随机/置换行为以及关键浮点归约（法线重归一化、GGX 半角向量归一化、
颜色/法线加权求和）都经过本模块；种子固定时中间运算顺序逐字节可复现。
"""

from __future__ import annotations

import numpy as np


class DeterministicContext:
    """以种子初始化的确定性上下文，按阶段派生独立 RNG。"""

    def __init__(self, seed: int, tag: str = "root"):
        self.seed = int(seed)
        self.tag = tag
        self._sequence = np.random.SeedSequence(self.seed)
        self._stage_index = 0

    def stage_rng(self, label: str) -> np.random.Generator:
        """按固定派生顺序返回某阶段的 RNG。"""
        child = self._sequence.spawn(1)[0]
        self._stage_index += 1
        return np.random.default_rng(child)

    @staticmethod
    def permutation(rng: np.random.Generator, n: int) -> np.ndarray:
        if n <= 1:
            return np.arange(int(n))
        return rng.permutation(int(n))

    @staticmethod
    def ordered_sum(rng: np.random.Generator, arrays: list[np.ndarray]) -> np.ndarray:
        """按种子决定的置换顺序逐项累加，替代不确定顺序的归约。"""
        if not arrays:
            raise ValueError("ordered_sum 至少需要一个数组")
        order = DeterministicContext.permutation(rng, len(arrays))
        total = np.zeros_like(arrays[0], dtype=np.float64)
        for idx in order:
            total = total + arrays[idx]
        return total

    @staticmethod
    def seeded_normalize(
        rng: np.random.Generator,
        vectors: np.ndarray,
        axis: int = -1,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """重归一化：平方项按种子置换顺序累加后再开方。"""
        moved = np.moveaxis(vectors.astype(np.float64, copy=False), axis, 0)
        components = [moved[i] for i in range(moved.shape[0])]
        square_sum = DeterministicContext.ordered_sum(rng, [c * c for c in components])
        norm = np.sqrt(square_sum)
        norm = np.where(norm < eps, eps, norm)
        normalized = moved / np.expand_dims(norm, 0)
        return np.moveaxis(normalized, 0, axis)
