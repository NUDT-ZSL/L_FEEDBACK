# 实际验证记录

验证日期：2026-11-16  
工作目录：`/home/liang/feedback/uid100`

## 环境

| 项目 | 版本 |
|---|---|
| Python | 3.14.4 |
| NumPy | 2.5.4 |
| FFmpeg | 8.0.1-3ubuntu2 |
| ffprobe | 8.0.1-3ubuntu2 |

实现声明的最低环境为 Python 3.9 / NumPy 1.21。FFmpeg 仅用于独立响度对照，
不参与修复处理。

## 自动测试

```bash
python3 -m unittest discover -s tests -v
python3 tools/acceptance.py
```

结果：

- 单元/集成测试：`7/7` 通过。
- 端到端验收：`18/18` 通过。

验收覆盖输出格式保持、响度、峰值、直流、边界误差、报告回读、FFmpeg 响度对照、
单/多线程逐字节一致性、线程回收，以及缺失文件、无效 JSON、时间轴越界、伸缩
不可达和失败无残留输出。

## 固定种子样本实测

样本生成器使用固定种子 `20261116`。输出时长由两段各 2.8 秒素材经伸缩和 20 ms
交叉淡化得到。

| 指标 | 48 kHz 单声道 | 44.1 kHz 立体声 | 约束 |
|---|---:|---:|---:|
| 内置 BS.1770 积分响度 | -22.999999 LUFS | -23.000002 LUFS | -23 ±0.5 |
| FFmpeg `ebur128` | -23.0 LUFS | -23.0 LUFS | -23 ±0.5 |
| 样本峰值 | -15.390336 dBFS | -18.984964 dBFS | ≤ -1.0 |
| 整体直流 | -1.0e-9 | -7.0e-9 | abs < 0.001 |
| 最大边界误差 | 4.000 ms | 5.805 ms | ≤ 20 ms |
| 输出采样数 | 268512 | 246695 | — |

输出格式检查：

- 48 kHz 单声道输出为 `pcm_s16le / 48000 Hz / 1 ch / 16 bit`；
- 44.1 kHz 立体声输出为 `pcm_s16le / 44100 Hz / 2 ch / 16 bit`。

## 确定性

同一批处理清单分别以 `--jobs 1` 和 `--jobs 4` 执行后，两份 WAV 和两份逐文件
报告逐字节一致：

| 文件 | SHA-256 |
|---|---|
| 48 kHz 单声道 WAV | `d96b7ad0896cf6cac62714f68e21ef86a443ef4e034470db73ddaa6ab697a4af` |
| 44.1 kHz 立体声 WAV | `4647561694f7fdef32d401fd83d2cabf5e0465e2bd3a33d9b09aa6c2c156a50b` |

批处理结束后未发现残留的 `audiorepair` 工作线程。

## 错误码实测

| 场景 | 业务错误码 | CLI 退出码 |
|---|---|---:|
| WAV 缺失 | `INPUT_MISSING` | 10 |
| JSON 损坏 | `TIMELINE_INVALID` | 11 |
| 时间轴越界 | `TIMELINE_OUT_OF_RANGE` | 12 |
| 伸缩率超出声明范围 | `STRETCH_UNREACHABLE` | 13 |
| 24-bit WAV | `AUDIO_UNSUPPORTED_FORMAT` | 14 |

所有失败场景下目标 WAV、报告和 `.incomplete` 标记均不存在；若目标文件在处理前
已存在，输入/时间轴校验失败不会修改它。

