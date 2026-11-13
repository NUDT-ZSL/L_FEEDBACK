# 实际验证记录

验证日期：2026-11-14

## 环境

| 项目 | 版本 |
|---|---|
| Linux 测试 | Python 3.11.18，FFmpeg 6.0.1 static |
| Windows 测试 | Python 3.11.9，Windows FFmpeg 6.0 full_build（GyanD） |
| Python 依赖 | numpy 1.26.4，soundfile 0.12.1 |
| Windows 补充 | 同一脚本也在本机 Windows FFmpeg 8.0 上通过 |

所有测试使用本地确定性生成文件，不使用 Docker、网络素材或远程媒体。

## 执行命令

Linux：

```bash
.venv/bin/python tests/acceptance.py \
  --ffmpeg /tmp/ffmpeg6/ffmpeg \
  --ffprobe /tmp/ffmpeg6/ffprobe --keep
```

Windows：

```powershell
C:\Users\LIANG\AppData\Local\Programs\Python\Python311\python.exe `
  tests\acceptance.py `
  --ffmpeg D:\tmp_ffmpeg_test\ffmpeg.exe `
  --ffprobe D:\tmp_ffmpeg_test\ffprobe.exe --keep
```

Windows 使用官方 GyanD `ffmpeg-6.0-full_build.zip` 中的静态 `ffmpeg.exe`、
`ffprobe.exe`。两个平台最终均输出：

```text
ALL ACCEPTANCE CHECKS PASSED
```

## 样本与格式

- 3 种语言：`eng`、`fre`、`jpn`。
- 每种语言 3 个片段，共 9 个片段。
- 时间轴总长 2600ms，目标 48kHz、双声道、16-bit PCM。
- 源片段覆盖 16kHz/22.05kHz/44.1kHz/48kHz、单/双声道、
  PCM-16/PCM-24/float32。
- 包含 5ms 相邻交叉淡化重叠、5ms 以下连续间隔、片段间静音与尾部静音。
- 字幕包含换行、空白行、`&`、`<`、`>`、`-->`、中文和 Unicode 行分隔符。

## 核验结果

- 归一化后 mono/stereo WAV 均为 48000Hz、PCM-16、124800 帧；两种版本峰值实测
  均为 `-0.9999 dBFS`。
- 三个语言 VTT 的每条 cue 起止时间与清单整数毫秒完全一致，偏差 0ms；
  文件为 UTF-8 无 BOM、LF 换行。
- MP4 包含 1 条视频和 3 条 AAC 音频；轨道标签依次为 `eng`、`fre`、`jpn`。
- 视频和各音轨时长均为 2.600000s，音轨/视频差 0ms；所有音轨 48kHz 双声道。
- `moov` 位于 `mdat` 之前；首帧像素为黑。
- 同环境两次运行（不同 `--jobs`，第二次达到 CPU 上限）的全部 WAV、VTT、MP4
  SHA-256 完全一致。
- 跨 Linux/Windows FFmpeg 6 构建比较：全部 WAV 与 VTT 字节一致；MP4 因不同
  FFmpeg 构建的编码/复用差异而不同。同构建内重复运行仍字节一致。

## 错误与中断

| 场景 | 实测结果 |
|---|---|
| 同语言片段重叠 11ms | 退出码 2，stderr 为 `E_TIMELINE` |
| 引用缺失 WAV | stderr 为 `E_MEDIA`，其他语言继续并生成 MP4，退出码 4 |
| `--no-overwrite` 且输出存在 | 退出码 3，stderr 为 `E_EXISTS` |
| MP4 时长校验注入错误 | 抛出 `E_MUX` |
| 处理中收到中断 | 退出码 130，当前语言 3 个输出原子提交，无 `.tmp` |

中断实测：Linux SIGINT 在 `wav_stereo` 阶段发送，进程约 21ms 后退出；Windows
自动化测试发送 `CTRL_BREAK_EVENT`（同一处理器同时覆盖 `SIGINT`/`SIGBREAK`），
进程约 63ms 后退出。两种情况下均只保留当前语言的 stereo WAV、mono WAV 和 VTT，
输出目录中 `.tmp` 数量为 0。

## 复现要点

- 验收脚本会生成 `build/acceptance`；`--keep` 可保留产物供 FFprobe 或播放器检查。
- 中断测试使用 8 种语言、每种 16 个片段，确保能在语言处理阶段稳定注入信号。
- `E_MUX` 保留目录使用唯一名称 `.intermediate.<pid>.<id>`，避免并发运行互相覆盖。
