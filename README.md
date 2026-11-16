# L_FEEDBACK 离线混音引擎

一个仅依赖 Python、NumPy 与 SoundFile 的确定性离线多轨混音器。它读取 JSON
配置，把不同采样率、不同长度、不同起始偏移的 WAV 音轨对齐到统一时间轴，
执行“静态增益 → 压缩 → 限幅 → 混音”，分块写出 WAV，并生成包含
ITU-R BS.1770-4 响度、真峰值、逐轨峰值/RMS/增益缩减及内存峰值的 JSON 报告。

## 快速开始

```bat
py -3.9 -m pip install -r requirements.txt
python examples\generate_demo.py
python mixer.py examples\demo.json
```

等价的模块入口：

```bat
python -m offline_mixer examples\demo.json
```

示例会确定性生成 44.1 kHz、48 kHz、96 kHz 三个 WAV，并输出：

- `examples/demo_mix.wav`
- `examples/demo_mix.report.json`

常用参数：

```text
python mixer.py CONFIG.json
  --output PATH         覆盖输出 WAV 路径
  --report PATH         覆盖报告 JSON 路径
  --workers N           音轨并行线程数
  --serial              强制串行
  --chunk-frames N      分块大小，默认 32768 帧
  --quiet               成功时不输出摘要
  --debug               失败时附带 traceback
```

配置内相对路径相对配置文件目录解析；命令行显式给出的相对
`--output`/`--report` 相对当前工作目录解析。

## 配置格式

```json
{
  "output": {
    "path": "mix.wav",
    "sample_rate": 48000,
    "channels": 2,
    "duration_seconds": 10.0,
    "subtype": "PCM_32",
    "limiter_placement": "bus"
  },
  "bus_limiter": {
    "ceiling_dbfs": -1.0,
    "lookahead_ms": 5.0,
    "release_ms": 80.0
  },
  "processing": {
    "workers": 8,
    "chunk_frames": 32768
  },
  "tracks": [
    {
      "id": "dialogue",
      "input": "dialogue.wav",
      "offset_ms": 0.0,
      "gain_db": -3.0,
      "channel_map": [0, 0],
      "compressor": {
        "threshold_db": -20.0,
        "ratio": 3.0,
        "knee_db": 6.0,
        "attack_ms": 8.0,
        "release_ms": 120.0
      }
    },
    {
      "id": "music",
      "input": "music.wav",
      "offset_ms": 123.4,
      "gain_db": -6.0,
      "limiter": {
        "ceiling_dbfs": -1.0,
        "lookahead_ms": 5.0,
        "release_ms": 80.0
      }
    }
  ]
}
```

关键字段：

- `output.sample_rate`：输出采样率，允许 8000–192000 Hz。
- `output.channels`：1–16 声道。
- `output.duration_seconds`：可选；省略时按所有轨道裁剪后的最晚结束点推断。
- `output.limiter_placement`：`bus`（默认）、`per_track` 或 `both`。
- `tracks[].offset_ms`：浮点毫秒，最终按输出采样率量化，48 kHz 下量化步长约
  0.0208 ms，满足 0.1 ms 精度。负偏移裁掉轨道开头；超出输出末尾的轨道被裁掉。
- `tracks[].gain_db`：范围 -96–24 dB。
- `compressor.ratio`：必须不小于 1。`knee_db=0` 或
  `knee_type="hard"` 为硬拐点。
- `tracks[].limiter`：逐轨限幅参数；`per_track`/`both` 模式下未给出时继承
  总线限幅参数。
- `channel_map`：可选，长度必须等于输出声道数；元素是输入声道下标或 `null`
  （静音）。未给出时，单声道自动复制到所有输出声道，多声道转单声道取平均，
  相同声道数直接对应。

也支持若干简写：`start_ms`、`gain`、`threshold`、`attack`、`release`、
`ceiling`、`lookahead`，以及顶层 `sample_rate`、`output_path`、
`report_path`。推荐使用完整字段名。

## DSP 实现

### 带限 sinc 重采样

- 每个输出点在其输入时间坐标上用 64 抽头 Kaiser 窗（`beta=12`）带限 sinc
  求值；降采样截止频率取 `0.97 * min(1, Fs_out/Fs_in)`。
- 支持任意 8–192 kHz 输入/输出组合；相同采样率直接旁路。
- 重采样器保留固定长度的历史/前瞻样本，跨块连续计算绝对输出相位，末尾零填充，
  输出帧数按 `round(input_frames * Fs_out/Fs_in)` 固定。
- 本地 1 kHz 满幅正弦实测：44.1→48 kHz 为 129.04 dB，96→48 kHz 为
  113.21 dB，均高于 90 dB 要求。

### 压缩器

采用声道联动的前馈峰值检测和 dB 域平滑增益：

```text
over = level_db - threshold_db
硬拐点:   reduction = (1 - 1/ratio) * over
软拐点:   拐点区内按二次曲线过渡
```

起振/释放分别用采样率相关的单极点系数平滑增益缩减量，避免逐样本增益跳变。

### 前瞻真峰值限幅器

- 对基带每帧用 4 个相位（0、1/4、1/2、3/4）做 sinc 插值，形成真峰值包络。
- 将超过天花板的衰减需求向前后各保持 `2 * lookahead`，再做平滑的起振/释放。
- 音频按序列延迟后输出，但对外返回的样本数与输入完全一致，不改变时间轴。
- 内部保留 0.15 dB 安全余量。示例配置 `-1 dBFS` 天花板时，最终实测
  `-1.14999 dBTP`，没有超过天花板。
- `per_track` 只保证单轨限幅；多轨叠加仍可能超过总线天花板，此时报告发出
  `W_TRUE_PEAK_ABOVE_CEILING`。需要节目整体峰值合规时使用 `bus` 或 `both`。

### 响度与真峰值

- K 加权使用 BS.1770-4 的高架滤波与 38 Hz 高通，系数按实际采样率生成；
  48 kHz 时复现标准附录系数。
- 积分响度使用 400 ms 块、100 ms 步进、-70 LUFS 绝对门限和 -10 LU 相对门限。
- 短期响度最大值为 3 s 滑动窗、100 ms 步进；不足 3 s 时报告实际窗口长度。
- 真峰值使用与限幅器一致的 4 倍过采样包络。

## 并行、分块与内存

- 每条轨道由独立线程处理，默认线程数为 `os.cpu_count()`；线程池数量由
  `--workers` 控制。
- 每条轨道先写入独立的 32-bit float 临时 WAV，总线阶段始终按配置顺序单线程
  叠加并限幅。因此并行与串行使用相同的浮点累加顺序，输出逐字节一致。
- 输入、重采样、压缩、限幅、总线叠加和最终写盘均为分块处理；内存状态只与
  声道数、块大小、前瞻/释放窗有关，不随时长增长。
- 最终文件先写到同一目录下的临时文件，成功后用原子替换发布。任何失败都会由
  `TemporaryDirectory` 清理未完成结果。
- 超过约 3.9 GB 的输出自动使用 RF64 容器，扩展名仍为 `.wav`。
- 内存报告同时给出 OS 峰值工作集、输入文件字节数、解码 float32 字节数和比值。
  小样本会被 Python/NumPy 运行时基线主导；流式架构的渐近目标针对长节目。

## 报告

报告包含：

- 输出路径、采样率、声道数、格式、时长、帧数、SHA-256；
- 每轨原始/目标采样率、输入声道、重采样帧数、偏移、有效起点、裁剪样本数；
- 每轨处理前/后的峰值与 RMS（线性值和 dBFS）；
- 压缩器和限幅器的 `max_db`、`mean_db` 增益缩减量；
- 混音总线限幅器的参数与应用状态；
- 混音前（总线限幅之前）和最终输出的积分响度、短期最大值、真峰值；
- 偏移/尾端裁剪警告；
- 并行线程数、块大小、耗时及内存峰值。

主要响度字段既位于 `loudness.after_mix`，也提供
`loudness.integrated_lufs`、`integrated_lufs` 等顶层便捷字段。

## 错误处理

失败时以非零退出码结束，并向 stderr 输出单行 JSON，例如：

```json
{"error":{"code":"E_INPUT_MISSING","message":"input audio file does not exist","details":{"track":"a","path":"C:\\media\\missing.wav"}}}
```

| 错误码 | 含义 | 退出码 |
|---|---|---:|
| `E_CONFIG_MISSING` | 配置文件不存在 | 2 |
| `E_CONFIG_INVALID_JSON` | JSON 语法错误 | 2 |
| `E_CONFIG_MISSING_FIELD` | 必填字段缺失 | 2 |
| `E_INVALID_COMPRESSOR` | 压缩参数非法，如 ratio < 1 | 2 |
| `E_INVALID_LIMITER` | 限幅参数非法 | 2 |
| `E_INVALID_TRACK` | 轨道字段、ID 或声道映射非法 | 2/3 |
| `E_INPUT_MISSING` | 输入音频不存在 | 3 |
| `E_UNSUPPORTED_FORMAT` | 非 WAV 或不可解码 | 3 |
| `E_UNSUPPORTED_RATE` | 采样率超出 8–192 kHz | 2/3 |
| `E_PROCESSING` / `E_WORKER` | DSP 或工作线程失败 | 4 |
| `E_OUTPUT_WRITE` | 输出或报告写盘失败 | 5 |

配置和输入解析发生在创建临时工作区之前；处理中失败不会发布半成品输出。

## 验证结果

本容器实际环境：Ubuntu 26.04、Python 3.14.4、NumPy 2.5.4、SoundFile 0.14.0。
代码避免 POSIX 专属依赖，Windows 10/11 使用 `ctypes`/PSAPI 采集工作集，
文件操作使用 `pathlib` 和 `os.replace`，并行使用线程池以规避 Windows
`spawn` 的进程引导差异。当前环境没有 Windows 主机，因此未声称已完成
Windows 实机运行；Windows 实机验证命令与本文相同。

执行：

```bash
python -m unittest -v tests.test_mixer
```

实际结果（2026-11-16）：

- 11 个测试全部通过，耗时 11.136 s。
- 44.1→48 kHz、1 kHz 满幅正弦重采样 SNR：**129.0369 dB**。
- 96→48 kHz 同类测试 SNR：**113.2064 dB**。
- 7 s 确定性双声道测试信号积分响度：
  `pyloudnorm` 参考 **-12.289619 LUFS**，引擎 **-12.291179 LUFS**，
  绝对误差 **0.001559 LU**（要求 ≤ 0.5 LU）。
- 演示工程总线限幅天花板 -1 dBFS：最终真峰值 **-1.149993 dBTP**；
  限幅前真峰值为 **+1.400829 dBTP**，说明叠加峰值确被处理。
- 演示工程串行（`--workers 1`）与并行（`--workers 4`）输出 SHA-256
  均为 `1dd433529682af2f9ee44a42a121f25b4728f9f85fda58c35cfb7ec525fb1112`。
  演示积分响度 **-4.541653 LUFS**。
- 演示工程（4 线程）内存峰值 **166,285,312 bytes**。输入仅约 6 MB，因此比值由约
  39 MB 的 Python/NumPy 运行时基线主导；报告如实保留 OS 峰值和比值。
- 非法配置实测：
  `E_INPUT_MISSING` 退出码 3，`E_INVALID_COMPRESSOR` 退出码 2，
  `E_UNSUPPORTED_RATE` 退出码 3，`E_UNSUPPORTED_FORMAT` 退出码 3，
  `E_CONFIG_MISSING_FIELD` 退出码 2；失败后无残留 `.offline_mixer_*`
  临时目录或半成品 WAV。

测试覆盖：多采样率重采样、BS.1770 参考响度、多轨偏移/负偏移裁剪、自动时长
推断、压缩器、总线与逐轨真峰值限幅、逐轨统计、报告字段、串行/并行字节一致性、
六类非法输入错误码及失败清理。
