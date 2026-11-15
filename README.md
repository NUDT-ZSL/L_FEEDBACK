# 互动剧情音频离线对齐与响度处理器

读取 JSON 事件清单与 WAV/FLAC 素材，生成时间轴精确对齐、响度统一到
**-16 LUFS**、真峰值不超过 **-1 dBTP** 的 48 kHz / 24-bit 立体声 WAV，
并输出逐事件 JSON 报告。纯本地、确定性，不依赖任何外部账号或数据集。

## 环境要求

- Python ≥ 3.10，依赖 `numpy>=1.24`（见 `requirements.txt`）
- FFmpeg ≥ 6.0，`ffmpeg` 与 `ffprobe` 可在命令行调用（需含 libsoxr）

本仓库实测环境：Python 3.14 + numpy 2.5 + FFmpeg 8.0.1（含 libsoxr）。

## 安装

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 命令行用法

```bash
python -m audiopipeline.cli <事件清单.json> <输出.wav> \
    [-c 配置.json] [-r 报告.json]
```

Windows 下也可直接使用批处理入口：

```bat
run_pipeline.bat manifest.json output.wav -c config.json
```

事件级错误（解码失败、文件缺失等）只跳过该事件，进程退出码仍为 0；
清单或配置本身非法时输出致命错误码并以退出码 2 结束。

## 事件清单字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `total_duration_ms` | 是（顶层） | 全局时间轴总时长，毫秒，正数 |
| `events` | 是（顶层） | 事件数组 |
| `id` | 是 | 事件唯一标识 |
| `path` | 是 | 素材路径（相对清单目录或绝对路径），仅 .wav/.flac |
| `start_ms` | 是 | 事件在全局时间轴的起点，毫秒，非负 |
| `offset_ms` | 否 | 素材内偏移，默认 0 |
| `duration_ms` | 否 | 事件持续时长；循环事件缺省时填充到时间轴末尾 |
| `loop` | 否 | 是否循环，默认 false |
| `priority` | 否 | 重叠优先级整数，越大越优先，默认 0 |
| `category` | 否 | `dialog` / `sfx` / `music`，默认 sfx |
| `overlap` | 否 | `replace`（默认，按优先级替换）或 `mix`（加法混音） |

## 一键演示与验收

```bash
python -m demo.generate_samples     # 生成确定性素材与两份清单
python demo/verify.py               # 响度/真峰值/对齐/错误码/确定性验收
python -m unittest discover -s tests
```

10 分钟性能样本：

```bash
python -m audiopipeline.cli demo/workspace/manifest_long.json demo/workspace/long.wav
```

## 目录结构

```
audiopipeline/   核心包（清单/解码/响度/淡化/混音/母带/写盘/CLI）
demo/            确定性样本生成器与验收脚本
tests/           36 个单元与集成测试
docs/DESIGN.md   算法、错误码与确定性设计说明
run_pipeline.bat Windows 命令行入口
```

## 输出报告

报告与输出 WAV 同目录、同名加 `.report.json`（或用 `-r` 指定），包含：
逐事件原始响度、调整增益、实际起止时间、是否循环/裁剪/静音填充、
错误码与警告，以及汇总统计（总数/成功/跳过、整体 LUFS、真峰值）和完整错误码目录。
