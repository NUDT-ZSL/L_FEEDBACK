# subalign

面向影视本地化与流媒体分发的字幕时间轴校正命令行工具：以一条参考字幕
（或视频关键时间点）为基准，自动检测并校正多条目标语言字幕的线性/分段
漂移，输出对齐后的 SRT/ASS 文件与可复现的 JSON 校正报告。

## 环境要求

- Python 3.10+（本地验证环境为 Python 3.14）
- 依赖：`numpy`、`pysubs2`
- 可选：FFmpeg/FFprobe（仅视频参考时间点提取需要，缺失时自动回退并告警）
- 纯本地运行，不访问网络或外部服务

```bash
python -m pip install numpy pysubs2
```

## 安装

```bash
python -m pip install -e .
```

之后可直接使用 `subalign` 命令，或通过 `python -m subalign` 调用。

## 命令行用法

```bash
python -m subalign \
  --reference reference.srt \
  --targets en_us.srt zh_cn.ass ja_jp.srt \
  --output-dir ./aligned \
  --report ./aligned/report.json
```

| 参数 | 说明 |
| --- | --- |
| `-r, --reference` | 参考字幕（.srt/.ass/.ssa）或视频文件（.mp4/.mkv 等） |
| `-t, --targets` | 一个或多个待校正字幕路径 |
| `-o, --output-dir` | 输出目录；缺省时输出到各输入文件同目录 |
| `--report` | JSON 报告路径，默认 `subalign_report.json` |

输出文件与输入同格式，文件名追加 `_aligned` 后缀（如
`zh_cn.srt -> zh_cn_aligned.srt`），输出统一使用 UTF-8 编码。

## 功能要点

- 自动识别 UTF-8（含 BOM）、UTF-16 LE/BE；无法解析时输出含文件路径与
  错误类型的结构化 JSON 错误并以非零退出码结束。
- 单调动态规划融合文本相似度与时间邻近性，逐匹配给出置信度；翻译语言
  自动切换纯时间对齐。
- 线性漂移估计全局缩放与偏移（确定性样本误差 0–10ms 内）；分段漂移通过
  残差突变自顶向下检测边界并分段拟合。
- 校正保持每条事件持续时间不变；重叠、零时长、负起始时间按确定性策略
  处理并在报告 `warnings` 中体现。
- 保留 ASS Dialogue 样式字段与 SRT 基本格式；NFC 规范化仅用于相似度
  计算，原始多语言文本（中/日/阿/俄等）原样保留，RTL 顺序不变。
- 报告包含漂移参数、匹配统计、警告、耗时；毫秒精度，ISO 8601 时间字段，
  可被再次解析校验。

## 验证

所有样本由测试代码确定性构造，无需外部下载：

```bash
python -m unittest discover -s tests -v
python acceptance_demo.py
```

退出码约定：`0` 成功；`1` 存在文件级处理错误（其他文件与报告仍产出）；
`2` 参数非法或路径不存在。

## 文档

- 算法与模块设计：`docs/DESIGN.md`
- 实测结果：`docs/VERIFICATION.md`
