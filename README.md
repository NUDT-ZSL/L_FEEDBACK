# Telemetry-driven Racing Audio Synth

一个完全离线、可复现的赛车音频合成管线：用 5 ms 一帧的遥测驱动引擎阶次泛音、胎噪滤波与离散事件，流式输出 48 kHz / 16-bit / 双声道 PCM WAV。合成过程不读取任何外部音频素材，噪声和瞬态都由参数模型与确定性种子生成。

## 快速开始

要求 Python 3.10 或 3.11，以及 NumPy 1.24 以上版本。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

生成本仓库自带的确定性遥测和参数样本：

```bash
python tools/generate_samples.py --out-dir samples
```

运行合成：

```bash
python -m telemetry_synth.cli \
  --telemetry samples/telemetry.json \
  --params samples/params.json \
  --output artifacts/demo/race.wav \
  --seed 20261115 \
  --block-size 4096
```

未指定 `--report` 时，报告写入 `race.report.json`（与 WAV 同目录）。成功时 stderr 会输出一行实际峰值内存信息：

```text
peak_memory_estimate_bytes=8224640 tracemalloc_peak_bytes=... process_peak_rss_bytes=... method=modeled_streaming_working_set
```

## 数据约定

### 遥测 JSON

时间轴固定 5 ms 一帧，帧数组必须从 `frame_index=0` 连续递增，`time_ms` 必须严格递增并等于 `frame_index * 5.0`。缺失帧、重复帧、时间跳变、字段类型错误、越界值和未知路面/事件类型都会被拒绝。

```json
{
  "schema_version": 1,
  "format": {
    "sample_rate": 48000,
    "channels": 2,
    "bit_depth": 16
  },
  "frame_interval_ms": 5.0,
  "frame_count": 2400,
  "frames": [
    {
      "frame_index": 0,
      "time_ms": 0.0,
      "rpm": 900.0,
      "throttle": 0.147176,
      "slip": 0.038,
      "surface": "asphalt"
    }
  ],
  "events": [
    {
      "frame_index": 165,
      "type": "suspension_impact",
      "intensity": 0.88,
      "pan": -0.35
    }
  ]
}
```

事件的时间坐标取所在帧的中心，即：

```text
event_sample = round_half_up((frame_index + 0.5) * 5ms * 48000)
             = round_half_up((frame_index + 0.5) * 240)
```

总输出采样数按遥测总时长计算：

```text
total_samples = round_half_up(frame_count * 5ms * 48000)
```

在固定 5 ms / 48 kHz 下正好等于 `frame_count * 240`。实现会校验实际写入数和 WAV `data` 块大小，任何不一致都以 `E_ALIGNMENT` 或 `E_WAV_HEADER` 失败。

### 参数模型 JSON

参数包含四组：

- `engine`：转速范围、点火阶次、泛音阶次/增益/相位、油门幅度调制、换挡 RPM 阈值、交叉淡化时长、立体声相位差。
- `tires`：滑移阈值/释放比例、平滑时间、各路面材质的 `grip` 与 `slip` 两套截止频率和增益。
- `events`：事件类型、波形、持续时间、攻击/衰减包络、频率、噪声混合比例、重叠时间窗和衰减系数。
- `mix`：三条独立总线增益、主增益和软限幅阈值。

完整可运行实例见 `samples/params.json`。

## 合成设计

### 帧到音频采样

音频块按全局采样索引推进。RPM、油门、滑移、路面滤波参数都用局部帧索引做线性重采样，不构造全时长音频数组。输入遥测帧数组只保留每帧的标量数组。

### 引擎声

基频为 `rpm / 60 * firing_order`，再叠加参数表中各阶次的泛音。每个泛音使用相位积分：

```text
phase[n] = phase[n-1] + 2*pi*frequency[n]/sample_rate
```

因此 RPM 变化不会直接重置相位。检测到相邻帧 RPM 阶跃超过阈值时，旧声音保持原相位并冻结在换挡前 RPM，新声音从当前相位开始以配置时长淡入；旧声音淡出，避免阶跃爆音。多个换挡交叉淡化交叠时，每个声部在切换点冻结当前增益并继续衰减。

### 胎噪

滑移率使用带释放阈值的迟滞状态机，越过 `slip_threshold` 进入滑移态，低于 `slip_threshold * release_ratio` 回到抓地态。状态包络按 `attack_ms` / `release_ms` 平滑；每个路面材质按 `grip`/`slip` 两套查表值混合，得到逐帧截止频率与增益，再线性重采样到音频采样。

白噪声不用顺序 RNG 流，而是对 `seed + 流标识 + 全局采样索引 + 声道` 做 SplitMix64 哈希。这样块大小或调用顺序不会改变任一采样的噪声值。单极点低通滤波器的状态跨块保存。

### 离散事件

事件按半帧中心对齐到音频采样。事件包络包含攻击、指数衰减和尾部淡出；悬挂撞击和换挡提示音使用不同参数波形。事件起点落在同一 `overlap_window_ms` 时间窗时，后触发事件按 `overlap_attenuation^n` 衰减。事件包络直接按全局采样坐标计算，因此可以跨块无缝延续。

### 混音与输出

引擎、胎噪、事件分别乘独立总线增益并累加，再应用主增益。软限幅在阈值以下保持线性，以上使用渐近到 ±1 的 `tanh` 过渡，最后量化到 little-endian PCM16。WAV 和报告先写入同目录临时文件，全部完成并通过 RIFF 头校验后才原子替换目标文件；失败时删除临时文件，不发布半成品。

内存占用按块内数组规模、遥测帧数和事件数建立保守估算，同时记录 `tracemalloc` 峰值和进程 RSS 峰值。

## 错误码

所有预期业务错误写入 stderr，格式为：

```text
error_code=E_XXX exit_code=NN message=...
```

| 错误码 | 退出码 | 含义 |
|---|---:|---|
| `E_USAGE` | 2 | CLI 参数或路径冲突 |
| `E_TELEMETRY_IO` | 10 | 遥测文件不可读 |
| `E_TELEMETRY_JSON` | 11 | 遥测不是合法 JSON |
| `E_TELEMETRY_SCHEMA` | 12 | 顶层结构或固定格式错误 |
| `E_TELEMETRY_MISSING_FRAME` | 13 | 帧数不符、帧索引或时间轴缺帧 |
| `E_TELEMETRY_TIMESTAMP` | 14 | 时间戳不单调或重复 |
| `E_TELEMETRY_FIELD` | 15 | 帧/事件字段缺失或类型错误 |
| `E_TELEMETRY_RANGE` | 16 | RPM/油门/滑移/路面/事件越界 |
| `E_PARAMS_IO` | 20 | 参数文件不可读 |
| `E_PARAMS_JSON` | 21 | 参数不是合法 JSON |
| `E_PARAMS_SCHEMA` | 22 | 参数结构错误 |
| `E_PARAMS_RANGE` | 23 | 参数数值越界 |
| `E_ALIGNMENT` | 30 | 帧时长与采样数无法严格对齐 |
| `E_OUTPUT_IO` | 31 | 最终文件发布失败 |
| `E_WAV_IO` | 40 | WAV 写入或读取失败 |
| `E_WAV_HEADER` | 41 | RIFF/WAVE 头字段与预期不符 |
| `E_REPORT_IO` | 42 | 报告写入失败 |
| `E_SYNTH` | 50 | 流式状态机内部一致性错误 |
| `E_INTERNAL` | 99 | 未预期异常 |

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖：

- WAV RIFF 头、PCM16 双声道、48 kHz、精确采样数。
- 报告中的总线峰值、软限幅峰值、事件触发和重叠衰减次数。
- 同参数复跑逐字节一致；不同块大小输出仍一致。
- 缺帧、帧索引跳变、非单调时间戳、越界 RPM、未知路面、坏 JSON、坏参数、非法块大小。
- 失败时不生成 WAV/报告，且不覆盖已有目标文件；篡改或截断的 WAV 头被独立解析器拒绝。
- 全局采样索引噪声分块一致性和软限幅边界。

本次实测环境、哈希和结果记录见 `VALIDATION.md`。

