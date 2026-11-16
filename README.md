# 离线多语言对白混音器（offline-mixer）

纯本地、确定性的离线混音流水线。输入结构化时间轴 JSON、多语言多采样率 WAV
对白、背景音乐与音效轨，输出对齐、响度均衡、带侧链闪避与峰值保护的
48 kHz / 16-bit / 立体声 WAV，并生成结构化 JSON 报告。

## 环境准备

- Python 3.10+（实测 Python 3.14 亦可；3.13+ 需额外安装 `audioop-lts` 供 pydub 使用）
- FFmpeg 已安装并在 `PATH` 中（`ffmpeg -version` 可见）
- 依赖安装：

```bash
python -m pip install -r requirements.txt
# Python 3.13/3.14 若 pydub 报 audioop 缺失：
python -m pip install audioop-lts
```

## 命令行用法

所有命令通过 `python -m offline_mixer ...` 调用（在仓库根目录执行）。

```bash
# 1) 生成确定性测试工程（3 语言、10 段对白、音乐/音效轨）
python -m offline_mixer generate demo_project --seed 20261117

# 2) 执行完整混音流水线
python -m offline_mixer mix demo_project/timeline.json \
    -o demo_project/mix_output.wav \
    -r demo_project/mix_report.json

# 3) 按验收规则校验报告（退出码 0 = 全部通过）
python -m offline_mixer verify demo_project/mix_report.json

# 单元 + 端到端测试（21 个用例，仅依赖标准库 unittest）
python -m unittest tests.test_pipeline -v
```

时间轴校验失败时进程以非零退出码终止，并向 stderr 输出含错误码与
`timeline_index` 的 JSON 诊断，例如：

```json
{"status": "error", "error": {"code": "E2002",
 "details": {"issues": [{"code": "E2002", "timeline_index": 0,
 "message": "source_range end exceeds source audio duration"}]}}}
```

## 时间轴 JSON 结构

```json
{
  "project": {"name": "demo", "sample_rate": 48000, "duration_seconds": 30.0},
  "reference_tracks": {"music": "audio/music.wav", "sfx": ["audio/sfx.wav"]},
  "segments": [
    {
      "index": 0,
      "start": 1.0, "end": 2.8,
      "source_range": [0.0, 2.0],
      "file": "audio/voice_en.wav",
      "language": "en",
      "target_lufs": -23.0
    }
  ]
}
```

- `start/end`：片段在最终时间轴上的位置（秒）。
- `source_range`：在源 WAV 中截取的 `[起, 止]`（秒），校验其不超出文件总长。
- 允许 `start == end` 的零长度片段（通过校验，混音阶段标记为无法对齐但不阻塞其他片段）。

### 错误码

| 代码 | 含义 |
| --- | --- |
| E1001 | 音频/时间轴文件不存在 |
| E1002 | 音频文件存在但无法解码 |
| E2001 | 时间数值非法（负值或 end < start） |
| E2002 | 源范围超出音频文件总长 |
| E2003 | JSON/字段结构错误 |
| E3001 | 变速比超出 [0.8, 1.25] 或零目标时长（片段级异常，不中断混音） |
| E9000 | 未预期内部错误 |

## 算法与模块

| 模块 | 职责 |
| --- | --- |
| `timeline.py` | JSON 解析、全量校验、错误码 + 片段索引诊断 |
| `audioio.py` | libsndfile 读写、多声道下混、16-bit 立体声 WAV 输出、FFmpeg 探测 |
| `resample.py` | `scipy.resample_poly` 带限多相插值，统一到 48 kHz 单声道，抗混叠 |
| `timescale.py` | 40 ms Hann 颗粒、10 ms 分析跳距的 OLA 颗粒式时间缩放（保音高） |
| `loudness.py` | ITU-R BS.1770-4 K 加权、400 ms/75% 重叠块、-70 LKFS 绝对门限与 -10 LU 相对门限 |
| `ducking.py` | 对白幅度包络侧链，3 dB/倍频程、上限 -12 dB、10 ms 启动、200 ms 内恢复 |
| `limiter.py` | -1 dBFS 二次软拐点限幅（导数连续、硬顶保证不越界），统计触发样本/事件数 |
| `mixer.py` | 逐样本叠加、报告组装、耗时统计 |
| `samplegen.py` | 固定种子确定性样本生成 |

### 关键设计说明

- **变速对齐**：`ratio = 源时长 / 目标时长`，限定 `0.8 <= ratio <= 1.25`；
  越界片段写入 `alignment_error` 并跳过，其余片段继续混音。
- **响度归一化**：按门限积分 LUFS 测量值与目标值之差施加线性增益，并二次
  测量确认实现值与目标偏差在 ±0.5 LU 内；静音片段增益固定 0 dB，不做放大。
- **侧链闪避**：仅作用于音乐轨，音效轨保持原样；恢复时间按
  `tau = 0.2 / -ln(0.01)` 设计，保证对白结束后 200 ms 内增益回到 99%。
- **混音与限幅**：对白 + 闪避后音乐 + 音效逐样本求和，超过 -1 dBFS 的样本
  进入软限幅，报告记录触发样本数与连续事件数。

## 确定性样本覆盖

固定种子 `20261117` 生成的工程覆盖：

- 3 种语言（en 48 kHz / ja 32 kHz / fr 16 kHz），重采样状态写入报告；
- 10 个对白片段，含时间轴重叠（片段 1/2）、零长度片段（4）、静音片段（5）、
  超变速比片段（7，ratio=0.2）、边界变速比（8，ratio=1.25）、
  削波/高响度片段（9，触发软限幅）；
- 同种子重复生成的 WAV 与 JSON 字节级一致（测试 `test_determinism_across_runs`）。

## 实测验收结果

在本机（Python 3.14.4、FFmpeg 8.0.1、numpy 2.5 / scipy 1.18 / soundfile 0.14）
运行 `mix` + `verify`：

- 对齐时长与目标时长误差 <= 1 个采样点（1/48000 s）；
- 全部非静音片段响度偏差 = 0.0 LU（优于 ±0.5 LU 要求）；
- 闪避最大衰减约 11.9 dB，恢复时间约 0.187 s（<= 0.2 s）；
- 限幅器在削波段触发，限幅后峰值 <= -1 dBFS；
- 异常时间轴用例（E1001/E2001/E2002/E2003）均以退出码 2 返回 JSON 诊断；
- `python -m unittest tests.test_pipeline`：21/21 通过。
