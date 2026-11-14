"""业务错误码定义。

所有可预期的业务失败都抛出 :class:`BakeError`，CLI 据此返回非零退出码，
且管线保证在抛出错误前不会写出索引清单。
"""

from enum import IntEnum


class ErrorCode(IntEnum):
    ERR_RPM_TABLE = 1001        # 转速表非法（步进 <= 0、范围不可整除等）
    ERR_PEAK_CEILING = 1002     # 峰值上限越界
    ERR_CROSSFADE = 1003        # 交叉淡化比例越界或淡化区过长
    ERR_LOOP_DETECT = 1004      # 无法检测到有效循环点
    ERR_SEGMENT_RANGE = 1005    # 转速片段落在原始录音时间范围之外
    ERR_LOUDNESS_SPREAD = 1006  # 片段间响度差异超出容差
    ERR_CONFIG = 1007           # 其它配置非法


class BakeError(Exception):
    """携带业务错误码的异常。"""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = ErrorCode(code)

    @property
    def exit_code(self) -> int:
        # shell 退出码只保留低 8 位，映射为稳定的小整数（1~7），
        # 完整业务码 1001~1007 仍保留在 self.code 与日志/清单语义中。
        return int(self.code) - 1000
