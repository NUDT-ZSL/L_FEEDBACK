# Rhythm Mixer — 节拍对齐的动态分层离线混音工具

读取 44100 Hz / 16-bit PCM 单声道 WAV 分层音轨、BPM 谱面 JSON 与分层控制
事件 JSON，将事件吸附到节拍网格，按时间轴动态启用/停用鼓、贝斯、旋律等
分层，循环点使用 10 ms 交叉淡化保证无缝，最终输出单声道 16-bit PCM WAV 与
节拍对齐报告。所有结果在固定输入下字节一致。

## 环境准备

- Python 3.10+（开发验证使用 Python 3.14）
- 依赖：`pip install numpy soundfile`
- 无 Docker、无外部音频素材，样本由固定种子 `20261117` 确定性合成

## 目录结构

```
rhythm_mixer/
  cli.py            命令行入口，导入 NumPy 前强制单线程环境变量
  config.py         JSON/WAV 解析与业务校验
  beatgrid.py       节拍网格、就近吸附、半拍上限、毫秒精度校验
  looping.py        周期化循环源 + overlap-add 交叉淡化
  mixer.py          事件状态机、分块混音、-1 dBFS 归一化、PCM16 写出
  engine.py         流水线编排、部分输出清理、报告
  make_samples.py   固定种子合成测试素材与谱面
tests/              39 个 unittest 用例
```

## 输入文件

输入目录包含 `chart.json`、`events.json` 与各分层 WAV。

`chart.json`：

```json
{
  "bpm": 120.0,
  "total_duration_ms": 8000.0,
  "crossfade_ms": 10.0,
  "loop_bars": 2,
  "beats_per_bar": 4,
  "layers": [
    {"name": "drums", "file": "drums.wav"},
    {"name": "bass", "file": "bass.wav"},
    {"name": "melody", "file": "melody.wav"}
  ]
}
```

`events.json`（时间支持 `time_ms` 或 `time_s`，毫秒精度，非负）：

```json
{"events": [
  {"time_ms": 0, "layer": "drums", "action": "enable", "gain": 0.9},
  {"time_ms": 4000, "layer": "drums", "action": "disable"}
]}
```

## 使用方式

```bash
# 1. 生成确定性素材
python -m rhythm_mixer.make_samples --output-dir ./demo_input
# 2. 混音
python -m rhythm_mixer --input-dir ./demo_input --output-dir ./demo_output
```

输出固定为 `mix_output.wav` 与 `beat_alignment_report.json`。

## 关键设计

- **节拍吸附**：`beat k` 起始于 `k * 60/BPM` 秒；事件就近吸附，半拍为硬性
  上限，半拍平局向后续拍吸附。控制事件还必须落在节拍边界的毫秒精度内
  （容差 0.5 ms），否则报 `E_NOT_ON_BEAT`。偏差定义为“对齐时间减原时间”。
- **动态控制状态机**：重复 enable/disable 幂等；启用中改增益会关闭旧片段
  并开启新片段，两段增益线性交叉淡化。每个片段两端默认 10 ms 线性淡化，
  消除启停咔嗒声且不改变节拍时刻。
- **无缝循环**：声明 `loop_bars`/`beats_per_bar` 时，循环周期强制为整数小节
  的精确采样数（节拍对齐不变）；素材先平铺到该周期，再以周期 `L-X` 重复，
  重叠区 `X`（默认 10 ms = 441 采样）做权重和恒为 1 的 overlap-add。
- **分块两遍混音**：第一遍分块测量峰值并统计每分层非零采样数，第二遍分块
  应用峰值归一化（目标 -1 dBFS，线性系数 `10**(-1/20)/peak`）直接写 WAV，
  内存驻留与输出时长无关。
- **确定性**：float64 累加、固定分块顺序、`np.rint` 后转 int16；报告不含
  绝对路径，JSON 使用固定缩进与 `\n` 行尾，同输入两次运行字节一致。
- **单线程**：`cli.py` 在导入 NumPy 前设置 `OMP_NUM_THREADS` 等变量为 1。

## 错误码

stderr 输出单行 JSON，例如 `{"error": "E_UNKNOWN_LAYER", "message": "..."}`。

| 错误码 | 场景 | 退出码 |
| --- | --- | --- |
| `E_BAD_WAV` | WAV 头非法、非单声道、非 PCM_16、文件缺失 | 2 |
| `E_SAMPLE_RATE` | 采样率不是 44100 Hz | 2 |
| `E_OUTPUT` | 输出目录不可写 / 报告写失败 | 2 |
| `E_UNKNOWN_LAYER` | 事件引用未定义分层 | 1 |
| `E_GAIN` | 增益为 NaN/Inf 等非有限数 | 1 |
| `E_NOT_ON_BEAT` | 事件超出半拍吸附上限或不在节拍边界 | 1 |
| `E_CHART` / `E_EVENT` | 谱面/事件结构与取值非法 | 1 |

任何失败都会清理已产生的部分输出文件。

## 测试与验证

```bash
python -m unittest discover -s tests -v
```

覆盖：输出格式与 1 采样内时长精度、-1 dBFS 峰值、动态启停听感区间、
非整数 BPM 吸附与拒绝、循环点连续性与周期性、全部错误码与退出码、
不可写目录清理、字节级确定性、120 秒长输出内存（实测混音增长约为输入
音频的 1%，远低于 5 倍上限）、子进程单线程环境变量。
