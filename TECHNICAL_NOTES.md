# 技术说明

## 模块结构

| 模块 | 职责 |
| --- | --- |
| `dub_replace/timestretch.py` | 纯 numpy 的 WSOLA 不变音高时间伸缩与比例钳制 |
| `dub_replace/audioio.py` | WAV 读写、峰值 dBFS、FFmpeg 校验/重采样（锁 + 独立临时文件） |
| `dub_replace/metadata.py` | 元数据 dataclass、布局/重叠校验、方言片段匹配 |
| `dub_replace/processor.py` | 单动画流水线：校验 → 匹配 → 伸缩 → 混音 → 写 WAV → 报告 |
| `dub_replace/state.py` | 版本化、原子写入的作业状态存储 |
| `dub_replace/runner.py` | 清单解析、线程池并发、失败隔离、作业汇总与退出码 |
| `dub_replace/cli.py` | 命令行入口 |
| `dub_replace/synth.py` / `scripts/generate_sample_job.py` | 确定性样本合成与样本作业生成 |

## WSOLA 为什么能保持音高

实现遵循 Verhelst & Roelands (1993)：合成端以固定 50% 重叠（`half = win/2`，
20 ms Hanning 窗）推进；输入侧维护一个连续浮点锚点，每帧前进
`analysis_hop = synthesis_hop / alpha`（`alpha = 目标长度/原长度`）。每帧只在
锚点 ±5 ms 容差内，用归一化互相关选与参考段最相似的具体片段。锚点轨迹与局部相似度
偏移解耦，因此周期信号的载波频率（音高）不变，只改变时长。最后做亚样本线性修正以
精确命中目标帧数（修正量远小于一个 WSOLA 帧）。极短素材回退到线性插值。

保音高验证：对 440 Hz、2 s 纯音做 0.6×/1.5× 伸缩，FFT 抛物线插值测频，频偏 < 6 Hz
（见 `tests/test_timestretch.py`）。

## 比例钳制语义

设槽位长度 `T`、方言长度 `N`、比例 `r = T/N`：

- `stretch_min <= r <= stretch_max`：直接 WSOLA 到 `T`，无事件。
- `r > stretch_max`（方言太短）：WSOLA 拉长到 `N*stretch_max`，其后在槽位内补静音，
  事件 `pad`，报告比例记为 `stretch_max`。
- `r < stretch_min`（方言太长）：取前缀长度 `round(T/stretch_min)` 做 WSOLA 到 `T`，
  等价于在最低允许速度下裁掉尾部，事件 `trim`，报告比例记为 `stretch_min`。

## 混音与对齐

- 仅 `role` 为 `voice`/`target_voice` 的声道在台词槽位 `[start_ms, end_ms]` 内被改写，
  槽位两端 5 ms 线性交叉淡化避免爆音；其余声道与槽位之外的内容原样保留。
- 输出帧数与原始音轨一致，字幕/口型事件时间轴无需平移即可继续对齐。
- 多目标声道（如声明了多个 target_voice）会写入相同的单声道方言混音。

## 持久化与并发

- 所有 JSON/WAV 落盘均为“同目录临时文件 + `os.replace`”原子替换；硬终止不会留下
  半文件。状态文件带 `version`，版本变化或 JSON 损坏时整体丢弃而非沿用。
- 线程池并发处理多个动画；FFmpeg 子进程由模块级锁串行化，且每次调用使用
  `tempfile.mkstemp` 唯一名，避免共享临时资源冲突。
- 失败动画在状态中记 `failed` 并写失败报告（含错误码与提示），重跑自动重试；成功项
  记 `success`，重跑直接跳过，不会重复输出。

## 已验证结果

- `pytest -q`：29 项全部通过。
- 样本作业（stereo / 5.1 / 损坏）：2 成功 1 隔离失败，进程退出码 1；重跑成功项跳过、
  损坏项重试仍失败。
- `ffprobe` 确认输出为 `pcm_s16le`、48000 Hz、声道数与布局一致（2 / 6）。
- SIGKILL 强杀后重启：已提交 WAV 字节级不变、未完成项补齐、无重复输出。
- 非目标声道与原始音轨最大绝对差为 0（PCM16 量化范围内）。
