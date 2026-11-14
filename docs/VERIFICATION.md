# 实测验证结果

验证环境：Linux 容器，Python 3.14.4，numpy 2.5.4，pysubs2 1.9.1；本机未
安装 FFmpeg/FFprobe（视频路径通过 mock 验证，缺失场景实测回退告警）。
所有样本均由 `tests/samples.py` 确定性构造，无外部下载、无网络访问。

## 单元与集成测试

命令：`python -m unittest discover -s tests`

结果：`Ran 32 tests ... OK`（32 个测试全部通过）。覆盖范围：

- 编码：UTF-8/UTF-8 BOM/UTF-16 LE/BE 识别与解码、BOM 剥离。
- 格式：SRT 与 ASS 解析、ASS `Style` 字段保留、缺失/不支持扩展名/
  非法字节的结构化错误（`FileNotFound`、`UnsupportedFormat`、`EncodingError`）。
- 线性漂移：纯偏移、缩放+偏移，校正后起始时间误差 0ms（容差 ±10ms），
  持续时间保持不变。
- 分段漂移：两段不同 scale/offset 自动识别为 `segmented`，分段边界落在
  注入边界 5s 邻域内，全部事件误差 0ms；恒定漂移保持 `linear`。
- 缺失/多余/乱序事件：缺失参考事件正确跳过（23/24 匹配），多余目标事件
  标记 `UNMATCHED_EVENT`，乱序事件保留文件顺序并记录 `OUT_OF_ORDER`。
- 多语言：中文翻译目标走纯时间对齐；阿拉伯文 RTL 文本与事件顺序保持；
  NFC/NFD 等价比较且原始文本不变。
- 消毒规则：负起始时间钳制并告警、重叠事件保留首个、零时长事件告警。
- CLI/报告：批处理 4 个不同编码/格式目标、报告可再次 `json.load`、
  ISO 8601 生成时间与 `PT` 持续时间、非法参数退出码 2、部分坏文件
  退出码 1 且其余文件正常产出。
- 视频路径：FFmpeg 缺失抛 `FFmpegUnavailable`、包时间戳聚类、回退路径
  携带 `FFMPEG_UNAVAILABLE` 警告并写结构化错误报告。

## 端到端验收脚本

命令：`python acceptance_demo.py`（退出码 1，因其中故意包含一个坏文件）

确定性输出（9 个有效目标 + 1 个坏文件）：

| 目标文件 | 模型 | 最大校正误差 | 匹配数 | 警告 |
| --- | --- | --- | --- | --- |
| linear_en.srt（偏移 +1250ms） | linear | 0.0ms | 24 | 0 |
| scaled_en.srt（scale 1.04 + 偏移） | linear | 0.0ms | 24 | 0 |
| segmented_en.srt（两段漂移） | segmented | 0.0ms | 24 | 0 |
| missing_event.srt（缺 1 事件） | linear | 0.0ms | 23 | 0 |
| reordered.srt（交换 2 事件） | linear | 0.0ms（非交换事件） | 24 | 2 |
| chinese.srt（中文翻译） | linear | 0.0ms | 24 | 0 |
| arabic.srt（阿拉伯文 RTL） | linear | 0.0ms | 24 | 0 |
| russian_utf16.srt（UTF-16 LE） | linear | 0.0ms | 24 | 0 |
| japanese.ass（ASS 样式保留） | linear | 0.0ms | 24 | 0 |
| broken.srt（非法字节） | — | 未处理 | — | 结构化错误 `EncodingError` |

说明：确定性样本以整数毫秒构造，因此误差为 0ms，优于 ±10ms 验收门限。
乱序样本中被人工交换的两条事件按需求保留原始文件顺序（不重排），仅产生
`OUT_OF_ORDER` 警告，其余事件误差为 0。

## 退出码核对

| 场景 | 退出码 |
| --- | --- |
| 全部文件成功 | 0 |
| 目标中含无法解析文件（其余正常） | 1 |
| 参考/目标路径不存在、扩展名非法、缺少必填参数 | 2 |

## 已知边界

- SRT 与 ASS 语法本身不支持负时间戳，pysubs2 在文件解析阶段会将其钳制
    到 0；内存级负起始时间的检测与钳制由 `drift.preflight_warnings` /
    `sanitize` 单元测试覆盖。
- 视频参考依赖本地 FFprobe/FFmpeg；当前环境无该二进制，相关行为通过
    `unittest.mock` 确定性验证。
