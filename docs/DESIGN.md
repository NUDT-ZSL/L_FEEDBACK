# subalign 技术设计说明

## 1. 模块划分

| 模块 | 职责 |
| --- | --- |
| `subalign/encoding.py` | BOM 驱动的编码识别（UTF-8/UTF-8 BOM/UTF-16 LE/BE）与解码 |
| `subalign/subtitles.py` | SRT/ASS/SSA 读写、事件模型、样式保留、结构化错误 |
| `subalign/textutil.py` | NFC 规范化、ASS 覆写标签剥离、`SequenceMatcher` 文本相似度 |
| `subalign/matching.py` | 单调动态规划事件匹配、文本锚点、翻译目标时间回退 |
| `subalign/drift.py` | 加权最小二乘、稳健重拟合、自顶向下分段、确定性校正与消毒 |
| `subalign/video.py` | FFprobe/FFmpeg 音频包时间戳提取与聚类 |
| `subalign/report.py` | 毫秒精度 JSON 报告、ISO 8601 时间戳与持续时间 |
| `subalign/pipeline.py` / `cli.py` | 编排与命令行入口 |

## 2. 事件匹配算法

1. 计算参考事件与目标事件的文本相似度矩阵（NFC 后比较，原文不改）。
2. 第一遍单调 DP：配对成本
   `0.6 * (1 - text_similarity) + 0.4 * time_cost`，其中
   `time_cost = d / (d + tau)`，`tau` 取两条时间轴中位事件间隔；跳过一行/列
   的固定代价为 `1.0`，因此缺失或多余事件可被跳过。
3. 由首遍匹配的中位时间差估计全局位移，再跑第二遍 DP，消除大漂移下的局部
   成本陷阱。
4. 若首遍平均文本相似度低于 0.35，判定为翻译目标，改用纯时间单调 DP，
   不依赖跨语言文本。
5. 每条匹配输出置信度
   `0.5 * exp(-d / tau) + 0.5 * text_similarity`（翻译目标时间权重 0.85）。

## 3. 漂移模型

- 线性模型 `reference = scale * target + offset`，以匹配置信度为权重做
  加权最小二乘（numpy 闭式解），随后用 `2 * RMS` 残差门限做一次稳健重拟合。
- 分段模型采用确定性自顶向下二分：仅当全局 RMS ≥ 200ms 时触发；对每个区间
  枚举满足最小 4 个匹配的切分点，若切分后 SSE 下降超过 50% 且最大残差
  ≥ 200ms 则分裂并递归。区间按目标起始时间排序，分段边界即相似度/残差
  突变位置。若分段 SSE 未显著优于全局模型则回退为线性模型。

## 4. 校正与确定性消毒

- 每条事件持续时间保持不变，仅按所属分段模型重算起始时间。
- 未匹配事件保留原时间并记录 `UNMATCHED_EVENT` 警告。
- 零/负时长记录 `ZERO_DURATION`；负起始时间先警告再钳制到 0ms
  （`NEGATIVE_START`）。
- 重叠事件保留首个事件，后续事件起始时间推后到前一事件结束
  （`OVERLAP`）；乱序事件不重排，仅记录 `OUT_OF_ORDER`。

## 5. 格式与多语言

- 按扩展名选择解析器；写出时复用输入 `SSAFile` 的样式/脚本信息，仅替换
  时间字段，因此 ASS 的 Style、Name、Layer、边距等完整保留，输出统一 UTF-8。
- 文本在比较前做 NFC，输出始终写回原始字符串；阿拉伯文等 RTL 文本不改变
  事件顺序与时间关系。

## 6. 视频参考路径

优先调用 `ffprobe -show_packets` 读取音频包 `pts_time`，ffprobe 缺失时
回退 `ffmpeg -f null` 解析 `pts_time:`；包时间戳按 500ms 间隔聚类成参考
事件。两者均不可用时抛出 `FFmpegUnavailable`，管道记录 `FFMPEG_UNAVAILABLE`
警告并尝试将路径作为字幕回退读取。

## 7. 报告与退出码

- 报告为确定性键排序的 JSON，数值以毫秒取整，生成时间为 UTC ISO 8601，
  耗时同时给出 ISO 8601 duration 与毫秒整数，可直接 `json.load` 复验。
- 退出码：`0` 全部成功；`1` 运行期存在文件级错误（仍产出有效文件与报告）；
  `2` 参数/路径校验失败。
