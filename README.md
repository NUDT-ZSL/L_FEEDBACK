# 本地剧情演出音频混音管线

这是一个不依赖第三方 Python 包的本地命令行管线。它扫描演出根目录中的
`manifest.json`，调用本地 `ffmpeg`/`ffprobe`，把对白、音乐和环境音统一为
立体声 48 kHz，按对白积分响度计算整体增益，输出严格合法的 16-bit PCM WAV，
并维护可断点续跑的原子作业清单。

实现不访问网络、不使用容器、不需要系统 PATH 中的 FFmpeg，也不会下载测试素材。

## 环境

- Python 3.11 或更新版本
- 本地 FFmpeg 6.x 的 `ffmpeg` 与 `ffprobe`
- Windows 10+ 或 Linux/macOS

本次实际验证环境是 Ubuntu 26.04、Python 3.14.4、FFmpeg 6.0 static。代码使用
跨平台 `pathlib`、`subprocess`、`os.replace` 和 Windows 专用进程组标志，但没有
在真实 Windows 主机上执行本轮验收。

## 快速开始

```bash
python audio_pipeline.py run /path/to/shows \
  --ffmpeg /path/to/ffmpeg \
  --ffprobe /path/to/ffprobe \
  --jobs 4
```

根目录也可用 `--root /path/to/shows` 传入，`--jobs` 也可写作 `--concurrency`。

Windows 示例：

```powershell
py audio_pipeline.py run D:\shows `
  --ffmpeg C:\tools\ffmpeg.exe `
  --ffprobe C:\tools\ffprobe.exe `
  --jobs 4
```

也可以先放好 FFmpeg 目录：

```bash
python audio_pipeline.py run /path/to/shows --tools-dir /path/to/ffmpeg/bin
```

运行内置确定性自检：

```bash
python audio_pipeline.py self-test \
  --dir .audio-pipeline-selftest \
  --ffmpeg /path/to/ffmpeg \
  --ffprobe /path/to/ffprobe \
  --jobs 4
```

自检退出码为 `0` 表示所有断言通过。自检内部包含预期失败作业，因此它报告的
“首次管线退出码”会是 `1`。

## Manifest

每个演出目录必须直接包含 `manifest.json`。示例包含全部字段；前 8 个字段必需，
两个钳制上限可省略：

```json
{
  "dialogue_path": "audio/dialogue.wav",
  "music_path": "audio/music.wav",
  "ambience_path": "audio/ambience.wav",
  "target_lufs": -16.0,
  "music_fade_start_sec": 1.0,
  "music_fade_duration_sec": 1.5,
  "music_gain_db": -6.0,
  "ambience_gain_db": -12.0,
  "max_attenuation_db": 12.0,
  "max_boost_db": 6.0
}
```

字段约束：

- 三条音轨路径必须是相对路径，不能逃出演出目录。
- 输入音频采样率只接受 `44100` 或 `48000`，声道数只接受 `1` 或 `2`。
- `target_lufs` 范围为 `[-70, 0]`。
- `music_fade_start_sec >= 0`，`music_fade_duration_sec > 0`。
- `music_gain_db` 和 `ambience_gain_db` 范围为 `[-120, 24]`。
- `max_attenuation_db >= 0`，`max_boost_db >= 0`。两者可省略，默认分别为
  `24.0` 和 `12.0`。

兼容的等价写法：

- `tracks.dialogue`、`tracks.music`、`tracks.ambience`
- `dialogue`、`music`、`ambience`
- `environment_gain_db` 代替 `ambience_gain_db`
- `dialogue_target_lufs` 代替 `target_lufs`
- `music_fade.start_sec` / `music_fade.duration_sec` 代替两个淡出字段
- `music_fade_start` / `music_fade_duration` 等不带 `_sec` 的等价字段
- `max_attenuation`、`max_boost` 以及不带 `_db` 的增益字段

缺失必需字段、类型错误、路径逃逸、冲突字段和非法 JSON 均产生
`MANIFEST_INVALID`，不会中断其他作业。

## 处理算法

每个有效作业严格按以下顺序执行：

1. `ffprobe` 检查三条输入轨的采样率、声道数和持续时间。
2. 三条轨分别解码并重采样为 `48000 Hz`、双声道、`s16le`。
3. 对白轨保持 `0 dB`，只测量其未加增益的积分响度 `L_dialogue`。
4. 音乐轨应用 manifest 指定的线性淡出，再应用 `music_gain_db`。
5. 环境音应用 `ambience_gain_db`。
6. 三轨以 `normalize=0` 直接求和，得到浮点预增益混音。
7. 计算 `requested_gain = target_lufs - L_dialogue`。
8. 将实际增益钳制到 `[-max_attenuation_db, +max_boost_db]`。
9. 对整个预增益混音应用实际增益，输出 16-bit PCM。

因此音乐淡出与对白重叠不会改变响度测量基准。`clamped`、`clamp`、
`requested_gain_db` 和 `applied_gain_db` 都会写入结果。

音乐淡出由 FFmpeg `afade` 的 `tri` 曲线实现；该曲线在幅度域线性下降。自检会
从最终 WAV 中按 660 Hz 测试音测量淡出前、淡出中、淡出后的幅度，验证中间点
接近 50% 且结束后接近 0。

## 输出与清单

默认文件布局：

```text
SHOW_ROOT/
  audio_pipeline_jobs.json     # 持久作业清单
  audio_pipeline_result.json   # 本次运行的完整结构化结果
  .audio-pipeline-tmp/         # 默认临时工作区
  renders/
    <job-id>/mix.wav           # 最终产物，相对 SHOW_ROOT 记录
```

作业标识是 manifest 所在目录相对 `SHOW_ROOT` 的 POSIX 路径。根目录自身若含
manifest，其标识为 `.`。

清单以 `jobs` 数组保存，每个作业最多一条，按作业标识排序：

```json
{
  "schema_version": 1,
  "updated_at": "2026-11-14T00:00:00.000Z",
  "jobs": [
    {
      "id": "scene-001",
      "manifest": "scene-001/manifest.json",
      "fingerprint": "sha256...",
      "status": "completed",
      "output": "renders/scene-001/mix.wav",
      "output_size": 768044,
      "measured_dialogue_lufs": -17.15,
      "requested_gain_db": 1.15,
      "applied_gain_db": 1.15,
      "clamped": false,
      "clamp": "none",
      "error": null,
      "updated_at": "2026-11-14T00:00:00.000Z"
    }
  ]
}
```

指纹包含管线版本、作业标识、manifest 原始内容，以及三条源音频的实际内容
SHA-256。只改文件时间而内容不变不会重做；manifest 或任一源音频内容变化都会
重做。重做通过作业标识 upsert，不会追加重复条目。

只有以下条件全部满足才跳过：

- 清单中的指纹与当前指纹一致；
- 清单记录了产物相对路径；
- 该路径存在、是普通文件；
- 文件大小与清单记录一致。

跳过作业在本次结果中状态为 `skipped`，并保留测量值、增益和钳制信息。

## 并发、临时空间与原子性

- `--jobs N` 限制同时处理的作业数。
- 每个作业使用一个独立目录 `<work-dir>/jobs/job-<hash>`，三条解码轨、
  预混音和最终 raw PCM 都放在这个目录中。
- `--job-temp-limit-mb` 是每作业硬性上限。启动解码前会根据时长估算所需空间；
  每个临时产物生成后再次检查实际目录大小。超过上限或磁盘保留空间不足时返回
  `TEMP_SPACE_INSUFFICIENT`。
- `--min-free-mb` 是临时文件系统保留空间，默认 32 MiB。
- 最终 WAV 先写入产物同目录的唯一 `.part` 临时文件，完成 `fsync`、严格 RIFF
  校验后再用 `os.replace` 原子替换。
- WAV 的 RIFF 大小、`fmt ` 和 `data` 大小由 Python 按实际 PCM 字节数写入，因此
  `data_size == file_size - 44`，不存在并发更新头部导致的长度不一致。

建议不要在多个进程中同时写同一个清单文件。管线内部并发是安全的，但没有实现
跨进程清单锁。

## 中断与断电恢复

- 每个成功、跳过或失败的作业完成后，清单立即通过“同目录临时文件 + fsync +
  原子替换”刷新。
- 第一次 Ctrl+C 只停止提交新作业；已启动作业继续执行到原子落盘和清单刷新。
  未启动作业在本次结果中标记为 `interrupted`，退出码为 `130`。
- FFmpeg 子进程运行在独立进程组/会话中，不会因终端 Ctrl+C 被提前杀死。
- 强制终止或断电后，下次运行只依据持久清单和产物大小判断是否复用。中途留下
  的临时目录和不被清单引用的 `.part` 文件不会被视为完成产物。
- 最终 WAV 替换成功但清单尚未刷新时发生断电，只是安全地多处理一次；不会产生
  重复清单项，也不会引用半成品。
- 清单损坏返回致命错误 `JOURNAL_CORRUPT`，本轮不会静默覆盖损坏文件。

## 退出码

| 退出码 | 含义 |
|---:|---|
| `0` | 全部作业完成或跳过，无失败 |
| `1` | 至少一个作业失败，其他作业已继续处理 |
| `2` | 致命配置/工具/清单/工作区错误 |
| `130` | 收到 Ctrl+C 并完成可控收敛 |

结果文件中的计数满足：

```text
total = completed + skipped + failed + interrupted
```

stderr 始终只输出一行摘要，例如：

```text
audio-pipeline exit=1 total=8 completed=4 skipped=0 failed=4 interrupted=0 result=...
```

该行中的计数直接来自结果 JSON 的 `counts`。

## 稳定业务码

| 业务码 | 含义 |
|---|---|
| `MANIFEST_INVALID` | manifest 缺失字段、类型错误、冲突、路径逃逸或 JSON 非法 |
| `SOURCE_MISSING` | 源音频路径不存在 |
| `SOURCE_UNREADABLE` | 源音频存在但不可读 |
| `SOURCE_HAS_NO_AUDIO` | 文件没有音频流 |
| `UNSUPPORTED_AUDIO_FORMAT` | 采样率不是 44.1/48 kHz，或声道不是 1/2 |
| `INPUT_PROBE_FAILED` | ffprobe 执行或输出解析失败 |
| `FFMPEG_UNAVAILABLE` | ffmpeg 不存在、不可执行或运行中消失 |
| `FFPROBE_UNAVAILABLE` | ffprobe 不存在或不可执行 |
| `TEMP_SPACE_INSUFFICIENT` | 每作业上限或临时磁盘空间不足 |
| `TEMP_IO_ERROR` | 临时目录创建、清理或访问错误 |
| `AUDIO_DECODE_FAILED` | 解码/重采样失败 |
| `LOUDNESS_MEASUREMENT_FAILED` | 对白积分响度无法测量 |
| `MIX_FAILED` | 三轨混音失败 |
| `FINALIZE_FAILED` | 最终增益或 16-bit PCM 渲染失败 |
| `OUTPUT_INVALID` | 写出的 WAV 结构校验失败 |
| `OUTPUT_IO_ERROR` | 最终目录或文件写入失败 |
| `OUTPUT_TOO_LARGE` | PCM 数据超过 RIFF 32 位长度限制 |
| `JOURNAL_CORRUPT` | 清单不是合法结构 |
| `JOURNAL_WRITE_FAILED` | 清单无法原子持久化 |
| `RESULT_WRITE_FAILED` | 结果文件无法写入（只能通过退出码和 stderr 报告） |
| `ROOT_INVALID` | 演出根目录不存在或不可访问 |
| `INTERRUPTED` | 作业因 Ctrl+C 未被启动 |
| `INTERNAL_ERROR` | 未分类内部错误，消息保留可读异常摘要 |

## 自检覆盖

自检使用固定随机种子 `0xA0D10F1C` 在本地生成 8 个作业：

- 4 个有效作业，覆盖 44.1/48 kHz、单声道/立体声的所有组合；
- 对白目标响度的正向和负向钳制；
- 音乐淡出与对白重叠；
- 1 个缺失源文件；
- 1 个缺字段 manifest；
- 1 个 32 kHz 源文件；
- 1 个 3 声道源文件。

自检不只检查退出码，还会检查：

- 每个最终 WAV 的 RIFF、`data` 长度、采样率、声道和位深；
- 对白测量值直接等于源对白轨测量值，而不是带音乐的最终混音值；
- 淡出中段和结束段的幅度比例；
- 钳制方向及实际增益；
- 第二次运行是否全部复用；
- manifest 内容变化是否只重做对应作业；
- 源音频内容变化是否只重做对应作业；
- 删除产物后是否重做且清单不重复；
- 遗留 `.part` 和临时文件是否被安全忽略；
- 临时空间不足和 ffmpeg 不可用的业务码与计数。

详细实测记录见 [docs/VERIFICATION.md](docs/VERIFICATION.md)。

## 自动化测试

```bash
FFMPEG=/path/to/ffmpeg \
FFPROBE=/path/to/ffprobe \
python3 -m unittest -v tests.test_pipeline
```

没有设置 `FFMPEG`/`FFPROBE` 时，单元测试仍可运行，端到端测试会跳过。
