# dubmux — 本地化过场动画配音对齐 / 字幕重排 / 封装工具

本地命令行程序：将多语言配音 WAV、字幕时间轴 JSON 与既有视频画面组合，
确定性地完成静音检测、片段切分、局部时间伸缩、字幕重排与封装，输出
按语言分别封装的 MP4 及一份对齐报告 JSON。全程离线，不访问网络。

## 环境

- Windows 10/11（本仓库代码为纯 Python + FFmpeg 子进程，亦可在 Linux/macOS 运行）
- Python 3.11 或 3.12，NumPy
- FFmpeg 6.x（`ffmpeg` / `ffprobe` 在 PATH 中）

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install numpy pytest
```

## 快速开始

```bat
REM 1. 确定性生成测试样本（视频 + 三语配音 WAV + 字幕 JSON）
python tools\make_samples.py --out samples

REM 2. 运行对齐 / 重排 / 封装
python -m dubmux ^
  --video samples\video.mp4 ^
  --lang en=samples\dub_en.wav --subs en=samples\subs_en.json ^
  --lang ja=samples\dub_ja.wav --subs ja=samples\subs_ja.json ^
  --lang de=samples\dub_de.wav --subs de=samples\subs_de.json ^
  --out out --report out\report.json --jobs 4
```

输出：`out\<lang>.mp4`（视频流 copy 不重编码，配音 AAC，字幕 mov_text）、
`out\<lang>.subtitles.json`（重排后字幕）、`out\report.json`（对齐报告）。

## 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--jobs` | 4 | 按语言并行处理的并发上限 |
| `--sample-rate` | 48000 | 配音 WAV 必须满足的采样率 |
| `--tolerance` | 0.05 | 片段边界相对目标区间的容差（秒） |
| `--max-sub-duration` / `--min-sub-duration` | 7.0 / 1.0 | 字幕最大显示 / 最少停留（秒） |
| `--silence-threshold` | 0.01 | 固定 RMS 静音阈值（帧 25ms / 步长 10ms） |

## 确定性设计

- 静音检测：固定阈值 + 帧级 RMS（25ms 帧 / 10ms 步长），全部参数写入报告。
- 片段切分：语音段合并间隔 300ms、最短段 150ms，纯 NumPy 实现，同输入同结果。
- 时间伸缩：按参考视频对白段落逐段线性重采样到目标区间精确长度，
  报告记录每段原始时长、伸缩比例、最终时长与边界误差（≤ 容差）。
- 字幕重排：条目按 `(start, id)` 排序；超长按字符数确定性拆分，
  过短先延长、与后条冲突则合并，调整前后时间全部入报告。
- 报告：含 `format_version`，每次运行整体快照写入（临时文件 + `os.replace`
  原子替换），识别旧版本号（`previous_format_version`），不增量追加；
  报告含全部输入文件 SHA-256 摘要，支持复现核验。
- 并行：`ThreadPoolExecutor` 按语言隔离临时目录（`out/.tmp/<lang>/`），
  运行结束或中止后清理；报告不含并发数与时间戳，不同 `--jobs` 下输出与
  报告逐字节一致（测试覆盖）。

## 业务错误码与退出码

| 代码 | 含义 |
| --- | --- |
| E1001 | 输入文件不可读 / 无法解码 |
| E1002 | 配音片段缺失（少于参考段落数） |
| E1003 | 配音片段多于参考段落数 |
| E1004 | 字幕条目超出视频总时长 |
| E1005 | 配音采样率与要求不一致 |
| E1006 | 对齐边界超出容差 |
| E1007 | ffmpeg/ffprobe 调用失败 |

退出码：`0` 全部成功；`1` 部分语言失败（其余照常交付）；`2` 全部失败 /
输入或配置错误 / 运行中止。单语言失败不阻断其他语言。

## 测试与验证

```bat
python -m pytest tests\ -q
```

覆盖：静音检测确定性、伸缩对齐（比例与边界容差）、字幕约束（拆分/合并/
延长）、并发一致性（jobs=1 与 jobs=4 报告与 MP4 逐字节相同）、三类异常
错误码与退出码、报告版本替换与输入摘要。

## 实际验证记录（本仓库环境：Python 3.12 + NumPy 2.5 + FFmpeg 8）

- `pytest tests/ -q` → **7 passed**（约 5s）。
- 样本运行：en/ja/de 三语全部 `[ok]`，退出码 0。
- 报告摘录：ja 起始静音检测 0.89s（构造值 0.90s）；片段 0 原始 1.665s →
  目标 2.015s，比例 1.21021（≈ 1/0.82），边界误差 0.0；
  各语言 `duration_diff`（音频-视频时长差）为 0.0。
- 异常运行：构造缺段 / 44100Hz / 字幕越界三语言 + 正常 en，退出码 1，
  报告分别记录 E1002 / E1005 / E1004，en 正常交付，失败语言无残留输出。
