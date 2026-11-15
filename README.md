# dialogue-sync

离线对白套片与广播交付工具。它读取多条对白 WAV、SRT 和时间轴 JSON，完成时间轴拟合、48 kHz/24-bit 统一、加权混合、EBU R128 响度归一化、SRT 同步、章节生成和 MKV 封装。整个流程不依赖 Docker 或网络服务，逐条处理结果可检查点恢复。

## 环境

- Python 3.9+
- NumPy
- FFmpeg 5.0+（需要 `ffmpeg`、`ffprobe` 在 PATH 中）

```bash
python3 -m pip install numpy
ffmpeg -version
python3 -m unittest discover -v
python3 scripts/acceptance.py
```

工具直接以仓库根目录运行，无需安装：

```bash
python3 -m dialogue_sync build \
  --timeline project/timeline.json \
  --srt project/dialogue.srt \
  --output-dir project/output
```

也可以安装为命令：

```bash
python3 -m pip install -e .
dialogue-sync build ...
```

## 时间轴 JSON

时间码统一使用严格、与帧率无关的 `HH:MM:SS.mmm`。`source_start` 是源时间轴锚点，不是 WAV 文件内的 seek 位置；WAV 默认从文件开头读取。若确实需要裁掉 WAV 文件头，使用可选的 `source_offset`。

```json
{
  "schema_version": 1,
  "srt": "dialogue.srt",
  "total_duration": "00:00:12.000",
  "audio": {
    "sample_rate": 48000,
    "bit_depth": 24,
    "channels": 2,
    "target_lufs": -23.0,
    "max_true_peak_dbtp": -1.0,
    "stretch_min": 0.8,
    "stretch_max": 1.25,
    "min_stretch_delta_ms": 25
  },
  "output": {
    "language": "eng",
    "subtitle_language": "eng",
    "title": "Episode 1",
    "audio_filename": "dialogue_mix.wav",
    "subtitle_filename": "dialogue_mix.srt",
    "container_filename": "dialogue_mix.mkv",
    "report_filename": "report.json"
  },
  "dialogue_entries": [
    {
      "id": "line_001",
      "source": "media/line_001.wav",
      "source_start": "00:01:12.500",
      "target_start": "00:00:04.000",
      "subtitle_index": 1,
      "weight": 1.0
    }
  ]
}
```

每个条目的必需字段是 `source`、`source_start`、`target_start` 和 `subtitle_index`。支持的可选字段：

- `source_offset`：WAV 文件内读取起点，默认 `00:00:00.000`。
- `source_end`：源时间轴上的窗口终点；窗口时长为 `source_end - source_start`，WAV 内从 `source_offset` 开始读取。
- `target_end` 或 `target_duration`：显式目标时长；未提供时使用对应 SRT 条目的原时长。
- `weight`：混合权重，范围 0 到 4，默认 1。
- `id`：报告和章节中使用的稳定标识。

`source` 相对路径以时间轴 JSON 所在目录为基准。`--srt`、`--channels`、`--language`、`--subtitle-language`、`--total-duration` 可覆盖 JSON 设置。输出固定为 48 kHz、24-bit PCM，配置为其他采样率或位深会被拒绝。

## 处理语义

1. 通过 `ffprobe` 验证源文件是 16/24-bit PCM WAV，采样率为 44.1 或 48 kHz，声道数为 1 或 2。
2. 使用 FFmpeg SoXR 重采样到 48 kHz。单声道转立体声时每声道乘 `-3.01 dB` 以保持总功率；立体声转单声道时使用 L/R 平均。
3. 目标时长优先采用 `target_end`/`target_duration`，否则采用 SRT 条目原时长。源窗口时长与目标时长之比为报告的 `stretch_rate = target / source`。
4. 比例必须位于 `[0.8, 1.25]`，否则该条目标记 `E_RANGE` 并跳过音频处理。长度差不超过 `min_stretch_delta_ms` 时优先裁剪或补静音；更大的允许差异使用 FFmpeg `atempo` 变速不变调，再做采样级裁剪/补静音，保证片段样本数精确。
5. 所有成功片段先写成 48 kHz/24-bit WAV 中间文件，再按目标样本位置和权重叠加。叠加峰值超过满刻度时对整条混音施加实际衰减，并在报告中记录 `mix_gain_db` 与 `E_CLIP`。
6. FFmpeg `loudnorm` 测量后执行两遍 EBU R128 归一化，目标为 `-23 LUFS`。真峰值目标预留 `0.2 dB` 安全余量（默认内部目标 `-1.2 dBTP`），并以 `ebur128=peak=true` 和 `loudnorm` 双重验证最终 `I <= -22.5 LUFS`、`I >= -23.5 LUFS`、`TP <= -1 dBTP`。归一化前后值、迭代阶段和额外峰值修正增益均写入报告。
7. 最终样本数由 `total_duration` 精确决定；封装后 MKV 时长与目标时长误差必须不超过 50 ms。

## SRT 同步

- SRT 输出条目顺序和序号保持不变。
- 映射偏移为 `target_start - source_start`。没有显式目标时长时，字幕起止时间同时平移；有显式目标时长时，字幕结束时间调整为目标起点加目标时长。
- 映射结果若为负值或零长度，会夹紧到非负且至少 1 ms，并在条目中写入 `E_TIMECODE` 和警告。不会产生非法 SRT 时间戳。
- 输出 SRT 使用标准逗号时间戳 `HH:MM:SS,mmm`；JSON 报告中的时间码统一为点号 `HH:MM:SS.mmm`。
- 任意两个映射后字幕区间重叠时，报告写入 `W_SUBTITLE_OVERLAP`，音频仍按重叠样本相加。

## 报告与错误码

`report.json` 包含每个条目的源路径、源/目标时间码、时间轴偏移、实际操作、伸缩率、atempo 比率、权重、增益分解、片段校验信息、字幕映射和错误码。

条目错误不会作为进程退出码返回。只要至少一条音频可生成，命令行仍退出 0，总体状态为 `partial`。主要错误码：

| 错误码 | 含义 |
| --- | --- |
| `E_TIMECODE` | 字幕映射需要夹紧，或其他条目级时间码问题 |
| `E_RANGE` | 伸缩比例超出限制、源窗口越界等，条目被跳过 |
| `E_CLIP` | 条目或加权混音超过满刻度，已施加衰减；条目通常仍可交付 |
| `E_IO` | 源 WAV 不存在或不可读取 |
| `E_FORMAT` | PCM 格式、采样率、位深或声道不符合输入约束 |
| `E_DECODE` | FFmpeg 解码或中间音频处理失败 |
| `E_INTERNAL` | 未预期的条目级异常 |
| `W_SUBTITLE_OVERLAP` | 映射后字幕重叠 |
| `W_SUBTITLE_CLAMPED` | 字幕映射被夹紧 |

配置、SRT 语法、FFmpeg 不可用或整体响度无法达标等全局错误使用非零进程退出码，并输出 `E_CONFIG`、`E_SRT`、`E_EXTERNAL`、`E_LOUDNESS` 等错误。

## 恢复与原子性

默认工作目录是 `<output-dir>/.dialogue_sync_work`，其中保存：

- `state.json`：带 SHA-256 封装的检查点，原子替换写入。
- `report_snapshot.json`：每完成一个条目后更新的报告快照。
- `fragments/entry_*.wav`：已成功条目的 48 kHz/24-bit 中间片段。
- `mix.wav`、`normalized.wav`、`package.mkv`：阶段缓存。

每个片段的 SHA-256 记录在状态中。续跑时若片段缺失或校验失败，只重做该条目；状态 JSON 截断、校验和不匹配或输入 manifest 变化时拒绝续跑，需使用 `--force` 重建。最终 WAV/SRT/MKV/报告都先写同目录临时文件，经 `fsync` 后 `os.replace`；MKV 先在私有工作目录生成并验证，再原子复制到输出位置。

`--simulate-abort-after N` 是验收用的强杀钩子：在第 N 个新检查点保存后执行 `os._exit(75)`，不会运行清理逻辑，用于验证真实中断恢复。

诊断状态：

```bash
python3 -m dialogue_sync state --work-dir output/.dialogue_sync_work
```

## 测试与实测结果

确定性样本覆盖 44.1 kHz/单声道/16-bit、48 kHz/立体声/24-bit、48 kHz/单声道/24-bit，以及伸缩、裁剪、补静音、范围错误跳过、重叠混音削波、字幕重叠和日语标签。完整自动化测试：

```text
Ran 16 tests in 12.167s
OK
```

`python3 scripts/acceptance.py` 在 2026-11-16 的实际结果（详见 `artifacts/verification.json`）：

- 环境：Python 3.14.4，FFmpeg 8.0.1。
- 17 项验收检查全部通过。
- 输出 WAV：48000 Hz、双声道、24-bit、576000 帧、12.000000 秒。
- 独立 `ebur128` 实测：`-23.2 LUFS`，真峰值 `-15.2 dBTP`。
- 加权混音峰值 `+2.383 dBFS`，自动衰减 `-2.392 dB`，条目 2/5 记录 `E_CLIP`。
- 伸缩、裁剪、补静音、跳过四种主操作均被触发并验证。
- MKV 音频与字幕语言均为 `eng`，5 个章节，音轨时长误差 `0.0 ms`。
- 强制中断退出码 75，检查点保留 2 个条目和 2 个校验通过的片段；篡改状态文件被 `E_STATE_INTEGRITY` 拒绝；恢复后从第 3 条继续，最终状态 `complete`。

可查看的验收产物：

- `artifacts/verification.json`：机器可读实测结果。
- `artifacts/acceptance/output/`：最终 WAV、SRT、MKV、报告。
- `artifacts/acceptance/recovery_output/`：中断恢复后的交付物。

## 项目结构

- `dialogue_sync/model.py`：JSON 模式、配置与时间轴校验。
- `dialogue_sync/audio.py`：重采样、声道映射、atempo、混音与 R128。
- `dialogue_sync/srt.py`：SRT 解析、映射后的重叠检测与输出。
- `dialogue_sync/state.py`：检查点、SHA-256 完整性验证与恢复。
- `dialogue_sync/pipeline.py`：端到端流程、报告、章节与 MKV 验证。
- `dialogue_sync/fftools.py`：FFmpeg/ffprobe 包装和原子写入。
- `tests/`、`scripts/acceptance.py`：确定性自动化测试与验收。

## 技术说明

- 时间轴内部使用整数毫秒，最终音频位置和长度按 48 kHz 样本四舍五入，因而与视频帧率无关。
- `atempo` 是 FFmpeg 内置的保持音高变速滤镜；输出仍可能包含算法瞬态，最终真峰值以 EBU R128 测量为准。
- 工具面向完整本地文件处理，单条对白片段会完整载入内存；超长整集单条素材应按对白切段输入。
- 输入 WAV 的 `source_start` 只作为原时间轴锚点；若源文件带前导素材，用 `source_offset` 显式指定文件内起点。
