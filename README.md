# bcsubts

Windows 本地命令行工具：将人工字幕时间轴对齐到播出主音频，并把 SCTE-35
插入指令和 DVB 字幕封装进单个 MPEG-TS。项目不依赖 Docker，不访问网络，
样本、报告和回归验证均可由固定随机种子重建。

## 功能

- 16-bit PCM WAV 输入，支持 44.1/48/96 kHz、单声道/双声道。
- 统一重采样到 48 kHz，采用带抗混叠低通的窗函数 sinc 插值。
- SRT/CSV 字幕解析，毫秒时间码、重叠、负时长、超过 5 秒空洞统计。
- 10 ms 粗搜（±2000 ms）加 48 kHz 采样点级细化的全局偏移估计。
- JSON SCTE-35 指令表校验、按采样点调度、splice_insert/splice_null 封装。
- AAC-LC + 黑色 MPEG-2 视频底板 + DVB 字幕 + SCTE-35 专用 PID。
- FFprobe/FFmpeg 自动校验、JSON 日志、断点恢复和结构化错误码。

## 环境

- Windows 10/11 x64
- Python 3.10+
- NumPy
- FFmpeg 6.0+（需要 `ffmpeg`、`ffprobe` 在 `PATH`；DVB 字幕渲染建议
  FFmpeg 构建包含 `libass`，没有时会退化为内置 ASCII 点阵字体）

包含 `libass` 的 FFmpeg 可直接把 UTF-8/中文文本渲染为 DVB 位图；实际验证
使用 `中文字幕测试 Unicode` 成功生成 234x40 的子标题区域。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ffmpeg -version
ffprobe -version
python -m bcsubts --version
```

也可以执行 `pip install -e .` 后直接使用 `bcsubts` 命令，或使用仓库中的
`bcsubts.cmd`。如果 FFmpeg 不在 `PATH`：

```powershell
$env:FFMPEG_BIN = "C:\ffmpeg\bin\ffmpeg.exe"
$env:FFPROBE_BIN = "C:\ffmpeg\bin\ffprobe.exe"
```

## 运行

```powershell
python -m bcsubts run `
  --audio samples\rate48_mono_pos\audio.wav `
  --subtitles samples\rate48_mono_pos\subtitles.csv `
  --scte35 samples\rate48_mono_pos\scte35.json `
  --output-dir out `
  --overlap-policy truncate
```

中断后原命令重跑即可从恢复点继续；要丢弃恢复状态并重新开始，增加
`--force`。`--ffmpeg`、`--ffprobe` 可显式指定可执行文件。

确定性样本和完整回归：

```powershell
python -m bcsubts generate-samples --out samples
python -m bcsubts regress --out regression
```

`regress` 覆盖 44.1/48/96 kHz、单/双声道、正/负延迟，生成
`regression\regression_summary.json` 并以非零状态返回任何失败。

## 输入格式

### 字幕

SRT 使用标准 `HH:MM:SS,mmm --> HH:MM:SS,mmm`。CSV 必须有表头：

```csv
start,end,text
"00:00:02,000","00:00:03,600","The evening bulletin begins now."
```

支持 UTF-8（含 BOM）、CRLF。不符合上述时间码的行返回 `E_TIMECODE`。

### SCTE-35 JSON

```json
{
  "events": [
    {
      "event_id": 49001,
      "type": "splice_insert",
      "offset_ms": 6000,
      "duration_ms": 4500,
      "segmentation_descriptor": {
        "segmentation_type_id": 52,
        "segment_num": 1,
        "segments_expected": 1,
        "upid_type": 8,
        "upid": "program-1"
      }
    },
    {
      "event_id": 50001,
      "type": "splice_null",
      "offset_ms": 17250,
      "duration_ms": 0
    }
  ]
}
```

`event_id` 必须是唯一 32-bit 无符号整数。`offset_ms` 必须位于
`[0, audio_duration]`；`duration_ms` 不得为负。`segmentation_descriptor`
可选。`splice_null` 本身在线路上没有 event_id，输入 event_id 用于指令表
唯一性和日志关联，`scte35_log.json` 中 `wire_event_id` 为 `null`。

## 输出

在 `--output-dir` 下：

- `<audio_stem>.corrected.srt`：校正后的独立 SRT 旁路。
- `alignment_report.json`：采样率/声道、偏移、置信度、残差方差、
  重叠/空洞前后统计、输出路径。
- `subtitle_corrections.json`：截断/跳过/越界夹取修正日志。
- `scte35_log.json`：事件 ID、采样点、目标 PTS、pts_adjustment、
  section CRC、PES 前插入位置和传输 PTS。
- `<audio_stem>.with_scte35.ts`：最终单文件 MPEG-TS。
- `verification_summary.json`：FFprobe/FFmpeg 和包级校验结果。
- `.work/recovery.json`：阶段、已完成转码段、已封装事件、动作游标、
  部分 TS 大小和失败阶段。

最终 TS 包含：

- 视频 PID：720x576/25 fps 黑色 MPEG-2 底板，用于标准播放器显示字幕。
- 音频 PID：48 kHz AAC-LC，PTS 连续。
- 字幕 PID：DVB subtitle，PMT 中 `stream_type=0x06` +
  subtitling descriptor，页面 ID 1。
- SCTE-35 PID：建议值 `0x0100`；冲突时自动选择空闲 PID。

## 时间同步口径

- 字幕时间以音频内容起点为零点。
- SCTE-35 的 `pts_adjustment` 设为目标音频 PTS（33-bit 取模），
  `splice_time=0`，因此有效 splice PTS 等于目标音频 PTS。
- 48 kHz 采样点映射到 90 kHz 时取最近 tick，量化误差小于 1 tick
  （约 11.1 µs）。
- 辅助 PES/section 在目标 PTS 之前插入到传输流中；不修改任何音频
  TS 包、音频 PTS 或连续性计数器。

### SCTE-35 传输封装说明

SCTE-35 over MPEG-TS 的标准线路封装是：

- PMT 中 `stream_type=0x86`；
- 节目级 `registration_descriptor` 包含 `CUEI`；
- 在专用 PID 上直接承载 `table_id=0xFC` 的 MPEG-2 private section。

本工具按该标准封装。FFmpeg 6+ 的 MPEG-TS demuxer 只按 section 语法
识别 `scte_35`；把 `splice_info_section` 再包进 PES 会出现一个额外
PES 头，标准解析器无法定位 `0xFC` table_id。因此这里没有输出
“PES 中的 SCTE-35”，而是使用播出链中可实际解析的 section 承载。
每条指令仍按音频 PTS 精确调度，日志中记录了对应 PES 插入边界。

## 断点恢复

- FFmpeg 转码成功后记录 `base_audio_video_mpegts` 完成段。
- 注入阶段在每个辅助动作后原子写恢复点；记录已完成的 SCTE event ID
  和字幕动作 ID、音频包索引、动作游标、TS 连续计数器和部分文件长度。
- Ctrl+C 时保留 `.work/*.partial` 和恢复 JSON。
- 重跑会截断到最后的已提交长度，跳过已完成动作，再继续注入。
- 最终文件先写临时名再原子替换；即使中断在替换边界，重跑也不会重复
  输出同一 event ID。

## 错误码

错误以单行 JSON 写到 stderr，例如：

```json
{"error":{"code":"E_TIMECODE","stage":"subtitle_parse","message":"Malformed SRT timecode","details":{}}}
```

| 退出码 | 业务码 | 触发条件 |
|---:|---|---|
| 2 | `E_FORMAT` | 非 16-bit PCM、浮点/24-bit WAV、非法 CSV/JSON/SRT 结构 |
| 3 | `E_TIMECODE` | 非法 SRT/CSV 时间码 |
| 4 | `E_OVERFLOW` | SCTE-35 偏移越界或持续时长为负 |
| 5 | `E_TRANSCODE` | FFmpeg 缺失/版本过低、渲染或 TS 转码失败 |
| 6 | `E_INPUT` | 输入不存在或恢复状态与输入指纹不一致 |
| 7 | `E_STATE` | 恢复状态/中间产物不一致 |
| 8 | `E_VERIFY` | TS 包级、FFprobe 或解码校验失败 |
| 130 | `E_INTERRUPTED` | Ctrl+C；恢复点和部分输出已保留 |

失败时已生成的 SRT、WAV、报告或部分 TS 不会删除，`recovery.json`
会写入 `stage=failed`、`failed_stage` 和结构化错误。

## 验证

单元和集成测试：

```powershell
python -m unittest discover -s tests -v
```

固定种子回归的实际结果见 `validation/validation_results.md`。核心验收点：

- 6 个采样率/声道/延迟组合全部通过，偏移误差 0.000–1.313 ms。
- 每条 TS 均由 FFprobe 识别为 AAC 48 kHz、`dvb_subtitle`、`scte_35`。
- 每条 TS 的 3 个 SCTE-35 section CRC/命令序列/`pts_adjustment` 校验通过。
- 每条 TS 的 5 个字幕 display set 可被 FFmpeg DVB 解码器解码。
- `validation/subtitle_overlay.png` 是从最终 TS 解码并 overlay 的字幕
  画面实拍，字幕位于 720x576 画面底部。
- 自动校验还会抽帧 overlay，并要求字幕区域最大 RGB 通道大于 40；
  六个回归样本实测均为 255。
- 音频 PTS 步长恒为 1920 ticks（1024 samples @ 48 kHz），连续无断裂。
- `E_FORMAT/E_TIMECODE/E_OVERFLOW/E_TRANSCODE` 分别实测返回 2/3/4/5。
- 注入中断恢复测试确认同一 event ID 只出现一次。

## 代码结构

- `bcsubts/wavio.py`、`resample.py`：WAV 校验和 sinc 重采样。
- `subtitles.py`：SRT/CSV、时间轴修正、偏移夹取。
- `alignment.py`：能量包络、伪语音活动、粗搜/细搜。
- `scte35.py`：指令表、section 编解码、CRC。
- `dvb.py`：libass 位图渲染、4-bit DVB 调色板/RLE、显示集/PES。
- `mpegts.py`、`mux.py`：PAT/PMT、PES/section 打包、调度注入。
- `verify.py`：包级和 FFprobe/FFmpeg 验收。
- `samplegen.py`：固定种子样本矩阵。
- `core.py`、`cli.py`：全流程、报告、恢复、命令入口。
