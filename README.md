# 赛车回放离线混音器

一个面向 Windows 10/11 PowerShell 的确定性离线混音 CLI。它读取一局比赛回放的 JSON 时间轴，把多路解说按锚点分段变速到回放时间轴，裁剪呼吸停顿，与引擎声/环境声混音，执行两遍式 RMS 归一化和峰值保护，最后输出 16-bit 交错 PCM WAV。导出按固定块写检查点，支持中断后续导。

运行时不访问网络、GPU、外部二进制或真实账号。WAV 读写仅用 Python 标准库 `wave`/`struct`，交织样本使用标准库 `array`；不依赖 `soundfile`、FFmpeg、SoX 等工具。环境按任务要求锁定 Python 3.11 与 NumPy 1.26.4，但 CLI 本身不导入 NumPy，因此没有 NumPy 也能运行。

## 快速开始

在项目根目录打开 PowerShell：

```powershell
py -3.11 --version
py -3.11 -m pip install numpy==1.26.4

py -3.11 tools\generate_sample.py --output demo --force
py -3.11 -m replay_mixer --config demo\replay.json
```

或直接执行脚本：

```powershell
.\scripts\run_demo.ps1 -Force
.\scripts\run_tests.ps1
```

成功时 stdout 只输出一行机器可读 JSON；失败时 stderr 只输出一行 JSON，不以异常堆栈或符号型错误码作为主要输出。

## 文件结构

- `replay_mixer/schema.py`：严格 JSON schema、业务码、路径与语义校验。
- `replay_mixer/audio.py`：WAV 输入读取、声道/采样率/位深交叉校验。
- `replay_mixer/dsp.py`：锚点时间映射、线性重采样、滑窗 RMS、声像、软限幅、量化。
- `replay_mixer/checkpoint.py`：块布局、检查点完整性、配置摘要、原始 PCM 摘要。
- `replay_mixer/engine.py`：两遍测量/导出、续导、统计与 WAV 封装。
- `replay_mixer/cli.py`：命令行参数、退出码和 stdout/stderr 契约。
- `tools/generate_sample.py`：确定性 WAV/JSON 样本生成器。
- `tests/`：21 项自动化验收测试。
- `scripts/run_demo.ps1`、`scripts/run_tests.ps1`：PowerShell 入口。

## JSON 契约

JSON 中所有内部路径都必须是相对路径，以 JSON 文件所在目录为基准；Windows 反斜杠会被转换为当前平台的路径分隔符。绝对路径、盘符路径、`..` 越界路径会被拒绝。

```json
{
  "version": 1,
  "sample_rate": 48000,
  "bit_depth": 16,
  "channels": 2,
  "output_path": "out\\replay_mix.wav",
  "replay": {
    "duration_ms": 9000.0,
    "frame_timestamps_ms": [0.0, 16.7, 33.4, 51.2, 68.3]
  },
  "mix": {
    "target_rms_dbfs": -20.0,
    "max_gain_db": 18.0,
    "peak_limit_dbfs": -1.0,
    "soft_knee_ratio": 0.9,
    "block_frames": 2048
  },
  "tracks": {
    "commentary": [
      {
        "name": "lead",
        "path": "inputs\\commentary_lead.wav",
        "channels": 2,
        "start_ms": 0.0,
        "end_ms": 8460.0,
        "gain_db": -3.0,
        "pan": -0.45,
        "anchors": [
          {"replay_ms": 0.0, "source_ms": 0.0},
          {"replay_ms": 1980.0, "source_ms": 1800.0},
          {"replay_ms": 4320.0, "source_ms": 3870.0},
          {"replay_ms": 6030.0, "source_ms": 5670.0},
          {"replay_ms": 8460.0, "source_ms": 7470.0}
        ],
        "silence": {
          "window_ms": 20.0,
          "threshold_dbfs": -45.0,
          "min_duration_ms": 135.0
        }
      }
    ],
    "engine": {
      "name": "engine",
      "path": "inputs\\engine.wav",
      "channels": 2,
      "start_ms": 0.0,
      "end_ms": 8910.0,
      "gain_db": -23.0,
      "pan": 0.0
    },
    "ambience": {
      "name": "ambience",
      "path": "inputs\\ambience.wav",
      "channels": 2,
      "start_ms": 0.0,
      "end_ms": 9000.0,
      "gain_db": -31.0,
      "pan": 0.0
    }
  }
}
```

约束要点：

- 根级 `channels` 是输出声道数，只支持 1 或 2；每轨 `channels` 是声明输入声道数，必须与 WAV 头一致。
- 支持采样率：8000、11025、12000、16000、22050、24000、32000、44100、48000、88200、96000 Hz。
- 仅支持 16-bit PCM WAV 输入/输出。
- `replay.duration_ms` 决定总时长；`frame_timestamps_ms` 是 VFR 回放帧序列，必须严格递增且不超过总时长。
- 所有轨道 `start_ms < end_ms`，且不超出回放总时长。
- 解说 `anchors` 至少两个；`replay_ms` 与 `source_ms` 都必须严格单调递增，`replay_ms` 必须位于该轨 `start_ms/end_ms` 之间。
- 缺失字段、类型错误、重复 JSON key、未知字段、NaN/Infinity 都在解析阶段以 `INVALID_CONFIG` 拒绝。

## 对齐与裁剪算法

### 分段线性变速

锚点定义源 WAV 时间到回放时间的映射。对回放时刻 `t`，位于锚点 `(r_i, s_i)` 与 `(r_{i+1}, s_{i+1})` 之间时：

```text
source_ms(t) = s_i + (t - r_i) * (s_{i+1} - s_i) / (r_{i+1} - r_i)
```

第一锚点之前沿用第一段斜率，最后锚点之后沿用最后一段斜率，因此锚点外是恒定速率。最后将 `source_ms` 转为源样本位置，在相邻源样本之间做线性插值。

- 每路解说在 `start_ms/end_ms` 之外补零，源位置超出 WAV 范围时也补零。
- 所有解说最终都严格产生 `round(duration_ms * sample_rate / 1000)` 帧，误差不超过 1 个采样点。
- 锚点 `replay_ms` 或 `source_ms` 非单调会在解析阶段报错。

### 呼吸停顿裁剪

- 每帧能量为各输入声道平方的均值；以 `window_ms` 对应的样本窗口计算滑窗 RMS。
- RMS 低于 `threshold_dbfs` 的连续区间，只有持续时间严格大于 `min_duration_ms` 才进入裁剪列表。
- 裁剪采用“源时间区间屏蔽”：当某个回放帧映射到的源样本落在裁剪区间时，该帧不再贡献解说声；不会把后面的非静音内容整体前移。
- 锚点始终按原始源时间映射，因此裁剪路径与不裁剪路径在所有未被屏蔽的回放帧上逐样本一致。缺失的静音位置由回放总时长内的零样本填充。

### 声像

- 单声道转立体声使用等功率声像：`left=cos((pan+1)*pi/4)`，`right=sin((pan+1)*pi/4)`。
- 立体声转立体声使用左右平衡；`pan=-1` 只保留左声道，`pan=1` 只保留右声道。
- 立体声转单声道时按声像权重下混，并保持增益可控。

## 混音、限幅与归一化

每轨先应用独立线性增益（dB），再按声像展开，所有轨道按固定顺序相加。混音总线使用软限幅，不用硬截断：

```text
ceiling = 10^(peak_limit_dbfs / 20)
knee    = ceiling * soft_knee_ratio

|x| <= knee: y = x
|x| >  knee: y = sign(x) * (knee + (ceiling-knee) * tanh((|x|-knee)/(ceiling-knee)))
```

软限幅输出渐近但不超过 `ceiling`。混音块先量化为 float32，再写测量流，确保首次导出的测量值与续导时的重算路径完全一致。

两遍式响度归一化：

1. 第一遍把软限幅后的混音块写到测量流，扫描全部样本计算整体 RMS 和峰值。
2. 计算目标增益：

```text
desired_gain_db = target_rms_dbfs - measured_rms_dbfs
gain_db = min(desired_gain_db, max_gain_db)
gain_db = min(gain_db, peak_limit_dbfs - measured_peak_dbfs)
```

第二步若被峰值上限截断，`peak_limit_triggered=true`。最终按 `gain_db` 进行第二次量化到 int16，并限制在 `[-32768, 32767]`。正常情况下峰值上限已使 int16 不会溢出，硬边界仅作为防御。

## 检查点与断点续混

输出旁会生成：

- `<output>.ckpt.json`：检查点，包含阶段、完成块数、已写数据字节数、累计 SHA-256、配置摘要、块布局、归一化结果。
- `<output>.measure.raw`：第一遍 float32 混音测量流。
- `<output>.final.raw`：第二遍 int16 交错 PCM 流。
- `<output>.tmp.wav`：最终 WAV 的临时组装文件。

行为：

- 每写一个固定帧块，就 flush/fsync 数据并原子更新检查点。
- 配置摘要覆盖规范化后的 JSON 和所有输入 WAV 的 SHA-256，因此配置或输入变化都会导致摘要变化。
- 数据流记录累计 SHA-256；检查点自身另有元数据完整性摘要。
- 默认模式下，摘要不一致、检查点损坏或数据摘要不符会清理不可信中间文件并重新开始。
- 加 `--strict-checkpoint` 时，同类问题返回 `CHECKPOINT_INVALID`（退出码 6），满足独立错误码验收场景。
- 测量阶段中断可从已完成测量块继续；最终阶段中断可从已完成导出块继续。最终 WAV 与一次性完整导出逐字节一致。
- `--stop-after-blocks N` 与 `--stop-after-phase measure|final` 是确定性的中断测试钩子，正常导出不需要使用。

## 退出码与错误 JSON

| 退出码 | 业务码 | 含义 |
|---:|---|---|
| 0 | - | 成功，stdout 输出统计 JSON |
| 1 | `INTERNAL_ERROR` | 未预期内部错误 |
| 2 | `MISSING_INPUT` | 配置或输入 WAV 缺失 |
| 3 | `INVALID_CONFIG` | JSON 缺失字段、类型/范围/顺序非法、路径非法 |
| 4 | `UNSUPPORTED_SAMPLE_RATE` | 配置或输入 WAV 采样率不支持/不一致 |
| 5 | `CHANNEL_COUNT_OUT_OF_RANGE` | 输出或轨道声明声道数越界 |
| 6 | `CHECKPOINT_INVALID` | 严格检查点模式下摘要/损坏校验失败 |
| 7 | `AUDIO_IO_ERROR` | WAV 头/数据损坏或声明声道不匹配 |
| 8 | `UNSUPPORTED_BIT_DEPTH` | 非 16-bit PCM |
| 9 | `OUTPUT_ERROR` | 输出或中间文件 I/O 失败 |
| 75 | `EXPORT_INTERRUPTED` | 测试钩子或用户中断 |

错误示例（单行 stderr）：

```json
{"ok":false,"code":"MISSING_INPUT","exit_code":2,"message":"required input file does not exist","context":{"path":"inputs\\commentary_lead.wav","role":"commentary"}}
```

## 统计字段

成功时 stdout 的 JSON 包含：

- `output_frames`、`output_samples`、`output_duration_ms`、`data_bytes`：与实际 WAV 数据一致。
- `commentary[].effective_frames`：该解说在回放时间轴内、映射有效且未被裁剪屏蔽的帧数。
- `commentary[].effective_samples`：上述有效帧乘以输出声道数。
- `commentary[].trimmed_intervals`、`trimmed_source_frames`、`trimmed_total_ms`：裁掉的源区间数和时长。
- `normalization.measured_rms_dbfs`：第一遍软限幅后混音的整体 RMS。
- `normalization.applied_gain_db` / `applied_normalization_gain_db`：实际应用的第二遍增益。
- `normalization.peak_limit_triggered`：归一化增益是否因峰值上限被截断。
- `normalization.soft_limit_triggered`：总线软限幅是否工作。
- `normalization.final_rms_dbfs`、`final_peak_dbfs`：从最终 int16 流复算的响度与峰值。
- `resumed_blocks`、`resumed_phase`、`measure_resumed_blocks`、`final_resumed_blocks`：本次复用的检查点块数。
- `total_trimmed_ms`、`elapsed_seconds`、`total_elapsed_ms`、`output_sha256`。

`peak_limit_triggered` 专指第二遍归一化增益被峰值上限截断；总线软限幅的触发另见 `soft_limit_triggered`，任一发生可由 `any_peak_limiting_triggered` 汇总。

## 实际验证

验证环境：

- Windows PowerShell 5.1，Windows Python 3.11.9，NumPy 1.26.4。
- 测试目录通过 WSL UNC 路径挂载到 Windows，额外覆盖了 Windows 网络共享上的文件替换语义。
- 同一份代码也在一台 Linux/Python 3.14 无 NumPy 环境中通过测试，说明运行时不依赖 NumPy。

自动化测试结果：

```text
Ran 21 tests in 17.123s
OK
```

覆盖项包括：锚点内外映射、线性重采样、滑窗 RMS、长短静音区间、裁剪前后非静音逐样本一致、单/立体声声像、软限幅、两遍归一化、峰值截断、WAV 头/数据长度、五类错误业务码、测量阶段/最终阶段续导、检查点损坏与配置摘要不一致。

9 秒、48 kHz 确定性演示样本的实测结果：

- 输出：432000 帧，2 声道，16-bit，数据长度 1,728,000 字节。
- 第一遍整体 RMS：`-23.945800 dBFS`。
- 应用归一化增益：`+3.945800 dB`。
- 最终 RMS：`-20.000000 dBFS`。
- 最终峰值：`-8.405347 dBFS`，未触发峰值截断或软限幅。
- 两路解说各裁掉 2 段：`723.125 ms` 与 `763.021 ms`，总计 `1486.146 ms`。
- 一次性导出、测量阶段中断后续导、最终阶段中断后续导三种路径的 SHA-256 均为 `32b2ee3ec9aec0eb4744b28fe327ecced66638fc12eca60cdba6bc69c7489c3e`。
- WAV 头核对：`RIFF size = file_size - 8`，`data size = 1,728,000`，`nframes * channels * sampwidth = 1,728,000`。

## 已知边界

- 输入轨整段读入内存；这是离线批处理工具，面向单场比赛回放。
- 仅支持 16-bit PCM、单/双声道和文中列出的采样率。
- 输出只在完整导出后以原子替换形式出现；中断时旧输出可能仍保留，中间状态以检查点为准。
