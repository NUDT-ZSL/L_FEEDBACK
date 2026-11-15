# 实际验证记录

验证日期：2026-11-16（Asia/Shanghai）

## 环境

- 运行环境：WSL2 Linux x86_64，用于本地自动验证
- Python：3.14.4
- NumPy：2.5.4
- FFmpeg/FFprobe：8.0.1（工具要求 FFmpeg 6.0+）
- 未使用 Docker、容器或网络媒体源

本环境 https://johnvansickle.com 的静态 FFmpeg 7.0.2 在 WSL 内核上
`ffprobe` 崩溃，因此实际验收使用 Ubuntu 软件包解出的 FFmpeg 8.0.1。
工具自身的版本检查允许 6.0+，命令和 API 未使用 6.0 之后新增的 CLI
选项。

## 自动测试

```text
python -m unittest tests.test_unit -v
Ran 9 tests ... OK

PATH=<ffmpeg-bin>:$PATH python -m unittest tests.test_integration -v
Ran 1 test ... OK

PATH=<ffmpeg-bin>:$PATH python -m unittest tests.test_cli_errors -v
Ran 1 test ... OK

PATH=<ffmpeg-bin>:$PATH python -m unittest discover -s tests -v
Ran 12 tests ... OK
```

集成测试实际覆盖：

- 44.1 kHz 双声道、+950 ms 延迟、CSV 字幕。
- 全流程生成最终 TS。
- 重复执行不产生重复 SCTE-35 event ID，最终文件大小不变。
- 模拟在第一个 SCTE event 封装完成后中断；检查恢复 JSON 已记录
  `completed_event_ids=[501]` 和部分文件长度。
- 从部分 TS 恢复后，最终流中 event 501 只出现一次。

## 固定种子回归

命令：

```bash
python -m bcsubts regress --out validation/regression
```

结果文件：`validation/regression/regression_summary.json`

| 样本 | 采样率 | 声道 | 已知延迟 | 估计延迟 | 误差 | TS 校验 |
|---|---:|---:|---:|---:|---:|---|
| rate44_mono_neg | 44.1k | 1 | -850 ms | -848.6875 ms | 1.313 ms | OK |
| rate44_stereo_pos | 44.1k | 2 | +950 ms | +950.0000 ms | 0.000 ms | OK |
| rate48_mono_pos | 48k | 1 | +250 ms | +250.0000 ms | 0.000 ms | OK |
| rate48_stereo_neg | 48k | 2 | -400 ms | -398.9792 ms | 1.021 ms | OK |
| rate96_mono_pos | 96k | 1 | +1500 ms | +1500.0000 ms | 0.000 ms | OK |
| rate96_stereo_neg | 96k | 2 | -1500 ms | -1499.1667 ms | 0.833 ms | OK |

每个样本的 `verification_summary.json` 均为：

```text
status=ok
ffprobe_audio_codec=aac
ffprobe_audio_sample_rate=48000
ffprobe_subtitle_codec=dvb_subtitle
ffprobe_data_codec=scte_35
scte35_event_count=3
dvb_display_sets=5
audio_pts_packet_count=939
audio_pts_step_90k_values=[1920]
ffmpeg_decode_check=ok
subtitle_visual_check=ok
subtitle_overlay_max_channel=255
```

`dvb_clear_sets=4`：第 1、2 条字幕首尾相接，清除集合被抑制，
由同一 PTS 的下一条显示集合直接替换，避免 FFmpeg DVB parser 在
同一 PTS 上连续收到两个 PES 时误判包头。

## 包级检查

- PMT 同时存在 AAC-LC、DVB subtitle（`stream_type=0x06` +
  subtitling descriptor）和 SCTE-35（`stream_type=0x86`）。
- PMT 节目级 registration descriptor 为 `CUEI`。
- SCTE-35 PID 为 `0x0100`，字幕 PID 为 `0x0120`。
- 3 条 SCTE section 的 CRC32/MPEG、命令序列、event_id、
  `pts_adjustment` 和 90 kHz 目标 PTS 与日志一致。
- 5 条 DVB show display set 被 FFmpeg `dvbsub` 解码器实际解码并
  重新编码通过；FFmpeg 退出码为 0。
- UTF-8 中文文本经 libass 渲染为 234x40 的 DVB 位图区域，成功生成
  显示集。
- 从最终 TS 在 2.4 s 处解码视频和 DVB 字幕并执行 overlay，输出
  `validation/subtitle_overlay.png`；画面底部实际显示
  “The evening bulletin begins now.”，证明内嵌字幕可渲染。
- 输出 TS 的音频 PTS 连续，步长仅包含 1920 ticks，未因辅助包插入
  产生断裂。

## 错误码

实测命令及 stderr 结构化结果：

| 输入 | stderr 业务码 | 进程退出码 |
|---|---|---:|
| 24-bit WAV | `E_FORMAT` | 2 |
| SRT `00:00:75,000` | `E_TIMECODE` | 3 |
| SCTE offset 99999 ms / 音频 20000 ms | `E_OVERFLOW` | 4 |
| 替身 FFmpeg 在转码阶段返回 1 | `E_TRANSCODE` | 5 |

`E_TRANSCODE` 失败后 `audio.corrected.srt` 和
`.work/resampled_48k.wav` 保留，`recovery.json` 记录：

```json
{
  "stage": "failed",
  "failed_stage": "ffmpeg_transcode",
  "error": {
    "code": "E_TRANSCODE",
    "stage": "ffmpeg_transcode"
  }
}
```
