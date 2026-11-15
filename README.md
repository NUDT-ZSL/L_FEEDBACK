# Cutscene Pipeline — 离线过场动画音频处理工具

本地离线命令行工具：读取 JSON 工程描述，确定性合成音频素材，完成时间轴对齐、
多轨混音、EBU R128 响度归一化与真峰值限制，并用 FFmpeg 封装为带章节标记的 MP4，
产出统一处理报告。全流程不访问网络，同一输入产出确定性结果。

## 环境要求

- Python 3.9+，NumPy（`pip install numpy`）
- FFmpeg 4.4+（含 `libx264` 与 `aac` 编码器），`ffmpeg`/`ffprobe` 加入 `PATH`
- 仅标准库 + NumPy，无其他 Python 依赖；不使用 Docker

## 使用方法

```bash
python3 -m cutscene_pipeline examples/project.json -o out/demo
python3 -m cutscene_pipeline examples/project.json -o out/dry --dry-run
```

- 正常模式：在输出目录写入分轨对齐 WAV（`aligned/track_*.wav`，48kHz/24bit 单声道）、
  混音（`mix_48k_24bit_stereo.wav`）、归一化音频（`normalized_48k_24bit_stereo.wav`）、
  成品 `cutscene.mp4` 与统一报告 `report.json`（UTF-8）。
- `--dry-run`：复用同一套校验与响度估算逻辑，不写出任何音视频文件，
  报告输出到 stdout，字段结构与正常模式完全一致。
- 退出码：`0` 成功；`1` 部分跳过（存在警告/跳过事件）；`2` 整体失败。

## 工程 JSON 格式

见 `examples/project.json`。要点：

- `sources`：源素材声明，必须为 44.1kHz/16bit/单声道，否则对应事件按
  `format_mismatch` 跳过；事件引用未声明的 source 按 `missing_source` 跳过。
- `tracks`：轨道名到 `gain_db` 的映射；事件引用未知轨道按 `unknown_track` 跳过。
- `timeline.events`：`event_id` / `track` / `start` / `duration` / `source`；
  负时长、负起点跳过；同轨时间重叠记入 `validation` 与 `warnings`。
- `chapters`：`title` + `start`（秒）；起点超出总时长的章节丢弃并告警。
- `video`：`width` / `height` / `fps` / `color`；`loudness`：`target_lufs`（默认 -16）
  与 `true_peak_dbtp`（默认 -1）。

## 处理流水线

1. **解析校验**（`project.py`）：schema 校验 + 事件级隔离（无效事件记警告并跳过，
   不影响其余事件的相对对齐）。
2. **确定性合成**（`synth.py`）：由 `sha256(event_id|source)` 推导频率/谐波/包络，
   生成 44.1kHz/16bit 单声道信号，同一输入比特级一致。
3. **时间轴对齐**（`pipeline.align_events`）：160/147 有理比窗函数 sinc 重采样到
   48kHz（零群延迟），按 `round(start*48000)` 放置到统一时间轴，偏移误差 ≤ 0.5 样本
   （≈0.01ms），分轨 WAV 落盘。
4. **多轨混音**（`pipeline.mix_tracks`）：float64 域按轨道增益线性叠加（无整数削波），
   输出 48kHz/24bit 立体声 PCM，记录各轨与混音峰值。
5. **响度归一化**（`pipeline.normalize_loudness` + `dsp.py`）：
   BS.1770-4 K 加权（48kHz 标准系数，FFT 卷积实现，与直接递归误差 ~1e-15）+
   400ms/75% 重叠门控积分响度；增益至目标 LUFS 后，用反射填充 FFT 频域插值做
   4 倍过采样真峰值测量，超限时由确定性前瞻峰值限制器压至 -1 dBTP 以内，
   迭代收敛后复测并记录前后值。
6. **封装**（`mux.py`）：FFmpeg `color` 源生成纯色视频，libx264 + AAC 封装 MP4，
   经 ffmetadata 写入章节标记。

## 统一报告

`report.json` 合并处理报告与响度报告，包含：输入/处理/跳过事件数及原因、
校验发现（重叠等）、逐事件样本偏移与误差、各轨与混音峰值、归一化前后
响度/真峰值、章节列表、各阶段耗时、警告列表、输出文件清单。

## 验证

```bash
python3 -m unittest discover -s tests -v
```

16 项测试全部通过（本机实测约 13s），覆盖验收标准：

- 对齐误差：报告内逐事件偏移误差 ≤ 1ms（实测 0.0ms），并从落盘分轨 WAV
  独立检测事件起点（阈值法）误差 ≤ 48 样本（1ms）。
- 响度：归一化后 -16.0 LUFS（目标 -16±0.5），真峰值 -5.3 dBTP（≤ -1）；
  并与 `ffmpeg -af ebur128=peak=true` 独立交叉校验一致（±0.5 LU 内）。
- 跳过逻辑：示例工程 11 个事件中 4 个（负时长/未知轨道/源缺失/格式不一致）
  被跳过并记录，退出码为 1；其余事件相对对齐不受影响。
- 章节：超出总时长的章节被丢弃，ffprobe 确认 MP4 内含 3 个章节标记。
- dry-run：不产出任何音视频文件，报告字段集合与正常模式完全一致。
- 确定性：同一 event_id/source 两次合成结果比特级一致。

示例工程（12s 时间轴、7 个有效事件）本机实测各阶段耗时：
解析 0.0002s / 对齐 1.35s / 混音 0.01s / 响度 3.6s / 封装 0.7s，总计约 5.7s。
