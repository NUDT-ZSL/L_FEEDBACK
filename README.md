# Cutscene Dub Conformer

`cutscene-dub` 是一个面向 Windows 10+ / Python 3.11+ 的本地命令行工具。它读取过场动画时间轴 JSON、某一目标语种的对白 WAV 和一条背景音乐 WAV，完成：

- 对白严格时间轴校验；
- 首尾静默裁剪与受限语速伸缩；
- 背景音乐自动闪避；
- 48 kHz / 16-bit / 交错立体声混流；
- 峰值限制与可复核 JSON 对齐报告；
- `.part` 临时文件 + 持久化状态的中断恢复和幂等重跑。

样本波形由固定种子确定性生成，不访问网络、账号或外部数据。

## 安装

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
ffmpeg -version
ffprobe -version
```

核心混流不需要 FFmpeg；只有实际发生变速时才调用 `ffmpeg -af atempo`。请确保 `ffmpeg` 在 `PATH` 中，或通过 `--ffmpeg-bin` 指定绝对路径。

不安装项目也可以直接使用模块入口：`python -m cutscene_dub ...`。

## 快速开始

```powershell
python -m cutscene_dub generate-samples `
  --timeline examples\timeline.json `
  --output-dir examples\samples `
  --seed 189

python -m cutscene_dub render `
  --timeline examples\timeline.json `
  --audio-root examples\samples `
  --output build\cutscene.wav `
  --report build\alignment_report.json
```

也可使用安装后的 `cutscene-dub` 命令替代 `python -m cutscene_dub`。

输出：

- `build\cutscene.wav`：48 kHz、16-bit PCM、2 声道交错 WAV；
- `build\alignment_report.json`：每个对白和闪避区间的复核数据；
- `build\cutscene.wav.state.json`：可选恢复状态，不是业务产物。

同一输入、配置和种子重跑时，工具会校验产物 SHA-256，完整则直接复用；缺文件、文件被改动或上次未完成时会整体重建，不会拼接半成品。`--force` 可强制重建。

## 时间轴 Schema

每份时间轴对应一个目标语种。多语种批处理时，为每种语言准备一份时间轴与对应 `audio-root`。

```json
{
  "schema_version": "1.0",
  "project_id": "harbor_light",
  "target_language": "en-US",
  "duration_s": 10.0,
  "shots": [
    {
      "id": "shot_001",
      "start_s": 0.0,
      "end_s": 5.0,
      "events": [
        {"id": "evt_001", "type": "camera_cut", "at_s": 0.0},
        {"id": "evt_002", "type": "camera_pan", "at_s": 2.4, "end_s": 4.2}
      ],
      "subtitles": [
        {
          "id": "sub_001",
          "start_s": 0.5,
          "end_s": 3.0,
          "text": "The lighthouse has been dark for three nights."
        }
      ],
      "dialogues": [
        {
          "id": "dlg_001",
          "start_s": 0.5,
          "end_s": 3.0,
          "subtitle_id": "sub_001",
          "text": "The lighthouse has been dark for three nights.",
          "target_language": "en-US",
          "source_duration_s": 3.25
        }
      ]
    }
  ]
}
```

字段规则：

- `schema_version` 必须为 `"1.0"`；顶层、镜头、事件、字幕、对白均拒绝未知字段。
- `project_id`、对象 `id` 仅允许 `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`，且各类 ID 全局唯一。
- `target_language` 允许 BCP-47 风格字符 `[A-Za-z0-9][A-Za-z0-9_-]{0,31}`。对白可省略该字段；若提供则必须等于顶层值。
- 所有时间单位为秒，必须为有限数。镜头、字幕、对白均要求 `end_s > start_s`。
- 镜头按开始时间递增且不重叠，全部位于 `[0, duration_s]`。
- 字幕按开始时间递增、不重叠，且完整落在所属镜头内。
- 对白按开始时间递增、不重叠；`subtitle_id` 必须引用同一镜头中的字幕，且对白区间必须完整落在该字幕区间内。
- 事件 `at_s` 必须落在所属镜头内；可选 `end_s` 不早于 `at_s`。
- `source_duration_s` 仅作为 `generate-samples` 的目标素材总时长（包含静默），不会替代实际 WAV 时长判断。

输入 WAV 必须是 48 kHz、16-bit PCM、单声道、无压缩。BGM 默认路径为 `<audio-root>/bgm.wav`，对白默认路径为 `<audio-root>/<target_language>/<dialogue-id>.wav`。

## 适配、闪避与响度

1. 读取实际 WAV 时长。只有素材超出对白分配窗口时才尝试裁剪，避免无必要修改。
2. 以 `--silence-threshold-dbfs` 为峰值阈值，裁剪首尾静默，并用 `--silence-margin-ms` 保留检测到的语音边缘余量。
3. 裁剪后仍超出时，所需比例 `original_frames / window_frames` 若不超过 `--max-speedup`，调用 `ffmpeg atempo` 保音高变速，并记录实际比例。
4. 所需比例超限时不截断语音，条目标记为 `unadaptable`。整次渲染以 `E_DIALOGUE_UNFIT` 失败，不发布最终 WAV；失败报告仍会写出，包含全部不可适配条目。
5. 闪避区间按对白实际落点计算。若相邻对白间隔小于 `duck_fade_out + duck_fade_in`，两个区间合并为一个持续压低区间，间隔内不回弹。
6. 闪避曲线使用升余弦渐变，并逐采样检查增益变化；超过 `--max-gain-jump-db` 时以 `E_DUCK_RAMP_UNSAFE` 明确失败。
7. 单声道对白与 BGM 均按恒定功率居中规则映射：左右声道各乘 `1/sqrt(2)`。输出为真正交错的双声道 PCM。
8. 混音后若峰值高于 `--peak-ceiling-dbfs`（最高只能配置为 `-1.0 dBFS`），对整体施加衰减并记录 `global_attenuation_db`。闪避深度与整体衰减分别记录，前者不是后者替代品。

## 对齐报告

成功报告的顶层字段固定为：

```text
schema_version, tool_version, status, source, timeline, config, alignment, music, ducking, mix, output, errors
```

主要复核字段：

- `alignment.dialogues[].subtitle`：字幕起止；
- `alignment.dialogues[].window`：对白分配窗口；
- `alignment.dialogues[].original`：素材实际时长；
- `alignment.dialogues[].silence_trim`：首尾裁剪帧数、秒数、阈值；
- `alignment.dialogues[].stretch.requested_ratio / applied_ratio`：请求与实际伸缩比例；
- `alignment.dialogues[].final`：最终起止、帧数、是否在窗口内；
- `alignment.dialogues[].unadaptable / status / error`：不可适配标记及同一业务错误对象；
- `ducking.intervals[]`：闪避开始/结束、对白活跃区间、实际淡入淡出、关联对白；
- `ducking.measured_dialogue_attenuation_db`：对白活跃区的实测音乐衰减；
- `mix.global_attenuation_db`：仅因峰值限制产生的整体衰减；
- `mix.dialogue_to_music_ratio_db`：对白与闪避后 BGM 的实测 RMS 差；
- `output.timeline_duration_error_frames`：输出与时间轴帧数误差，成功时恒为 `0`。

所有浮点时间同时记录全局采样帧，便于逐帧复核。时间轴总时长按 `round(duration_s * 48000)` 落帧，因此与秒值的差异小于一个采样帧。

## 错误与退出码

所有业务错误以单个 JSON 对象写入 `stderr`，成功摘要或样本生成摘要以 JSON 写入 `stdout`。错误对象始终包含 `code` 与 `entry_id`（无条目时为 `null`）：

```json
{
  "ok": false,
  "command": "render",
  "error": {
    "code": "E_DIALOGUE_UNFIT",
    "message": "1 dialogue(s) cannot fit within their subtitle windows under the configured speed limit.",
    "entry_id": "dlg_001",
    "details": {
      "entry_ids": ["dlg_001"],
      "report_path": "build/alignment_report.json"
    }
  }
}
```

退出码：

- `0`：成功或幂等复用；
- `2`：CLI 参数、配置或时间轴校验错误；
- `3`：输入音频格式、读取、FFmpeg、伸缩或闪避曲线安全错误；
- `4`：存在不可适配对白；
- `5`：样本、输出或状态写入错误；
- `70`：未预期的内部错误；
- `130`：Ctrl+C 中断。

稳定业务码包括：

`E_CLI_USAGE`, `E_CONFIG`, `E_TIMELINE_READ`, `E_TIMELINE_JSON`,
`E_TIMELINE_SCHEMA`, `E_TIMELINE_RANGE`, `E_SHOT_INVALID`, `E_EVENT_INVALID`,
`E_SUBTITLE_INVALID`, `E_DIALOGUE_INVALID`, `E_DIALOGUE_UNFIT`, `E_AUDIO_READ`,
`E_AUDIO_FORMAT`, `E_FFMPEG_NOT_FOUND`, `E_STRETCH_FAILED`,
`E_DUCK_RAMP_UNSAFE`, `E_SAMPLE_GENERATION`, `E_OUTPUT_WRITE`, `E_STATE_WRITE`,
`E_INTERRUPTED`, `E_INTERNAL`。

## 中断恢复

- 所有 WAV/JSON 先写同目录 `<name>.part`，关闭、`fsync` 后通过 `os.replace` 原子发布。
- 状态文件在写产物前先标记 `writing`，完成后标记 `complete`。
- Ctrl+C 会关闭文件并保留可识别的 `.part`，状态标记为 `interrupted`，退出码 130。
- 即使进程被强制终止，持久化状态仍为 `writing`；下次运行不会信任现有 `.part` 或同名最终文件，会清理后整体重建。
- 完整状态记录输入与配置指纹以及产物 SHA-256；任一输入、配置或产物变化都会使复用失效。

`CUTSCENE_DUB_TEST_FAILPOINT` / `CUTSCENE_DUB_TEST_HOLD_MS` 是仅用于自动化测试的确定性中断注入点，正常使用无需设置。

## 验证

```powershell
python -m pytest -q
```

当前测试覆盖时间轴错误、结构化错误对象、样本字节确定性、静默裁剪、atempo 伸缩、不可适配汇总、闪避区间合并、增益突变上限、峰值限制、报告字段、幂等复用、Ctrl+C 恢复和强制终止恢复。实际验证记录见 [VALIDATION.md](VALIDATION.md)。

