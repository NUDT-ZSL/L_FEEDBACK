"""离线引擎循环片段烘焙管线。

模块划分：
errors    业务错误码与异常
synth     确定性引擎原始录音合成器
pitch     时长保持的音高偏移（相位声码器 + 窗函数 sinc 重采样）
loops     循环点检测、整数周期约束与等功率交叉淡化
normalize 统一峰值余量归一化与响度校验
pipeline  端到端编排、参数校验与原子导出
"""

from .errors import BakeError, ErrorCode
from .pipeline import BakeConfig, build_pipeline, run_bake

__all__ = [
    "BakeConfig",
    "BakeError",
    "ErrorCode",
    "build_pipeline",
    "run_bake",
]
