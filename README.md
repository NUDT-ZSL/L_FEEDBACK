# Offline Dialogue Mixer

面向游戏剧情音频的 Windows 离线混音工具。输入本地 WAV（对白、音乐、音效）和 JSON 清单，输出 48 kHz / 24-bit PCM 多声道 WAV 以及 JSON 处理元数据。

核心能力：

- 对白滑动窗口噪声门限、噪声底噪估计和 SNR 余量上限。
- ITU-R BS.1770 K 加权、400 ms 短时响度、绝对/相对门限积分响度。
- 对白动态响度均衡，目标 ±1 LU，峰值不超过 0 dBFS。
- 以对白活跃区间为唯一时间基准的样本级侧链闪避。
- 立体声 / 5.1 输出，显式上混或下混矩阵。
- 分段输出、跨段共享处理状态和边界样本交接。
- 缺轨、坏格式和单段写出失败隔离；完整参数启动校验。

## 环境

- Python 3.11+
- NumPy 1.26+
- FFmpeg 6.0+，`ffmpeg` 必须在 `PATH` 中
- 推荐同时提供 `ffprobe`，用于更严格的格式校验；缺失时工具会回退到 `ffmpeg` 和 Python WAV 探测

Windows PowerShell 示例：

```powershell
py -3.11 -m pip install -r requirements.txt
ffmpeg -version
python -m offline_mixer --version
```

无需网络、数据集、Docker 或其他容器。

## 快速开始

生成确定性的对白、音乐、音效和示例清单：

```powershell
python tools/synthesize_demo.py artifacts/demo/input
```

运行混音：

```powershell
python -m offline_mixer artifacts/demo/input/manifest.json artifacts/demo/output
```

也支持：

```powershell
python mixer.py MANIFEST.json OUTPUT_DIR
python -m offline_mixer -m MANIFEST.json -o OUTPUT_DIR
offline-mixer MANIFEST.json OUTPUT_DIR   # pip install -e . 后
```

成功时会输出：

```text
OK: wrote 3 WAV segment(s); metadata=...\mix_metadata.json
```

## 清单格式

路径相对 JSON 清单所在目录解析。示例：

```json
{
  "schema_version": "1.0",
  "sample_rate": 48000,
  "output": {
    "channels": 2,
    "channel_layout": "stereo"
  },
  "target_lufs": -18.0,
  "noise_gate": {
    "window_ms": 20.0,
    "hop_ms": 10.0,
    "threshold_db": -45.0,
    "attenuation_db": 18.0,
    "snr_margin_db": 6.0,
    "noise_estimate_window_ms": 500.0,
    "noise_percentile": 20.0,
    "attack_ms": 4.0,
    "release_ms": 35.0
  },
  "loudness": {
    "window_ms": 400.0,
    "tolerance_lu": 1.0,
    "attack_ms": 35.0,
    "release_ms": 220.0,
    "max_gain_db": 30.0
  },
  "ducking": {
    "depth_db": 9.0,
    "attack_ms": 5.0,
    "release_ms": 300.0,
    "vad_threshold_db": -40.0,
    "vad_snr_db": 6.0,
    "vad_window_ms": 10.0,
    "min_gap_ms": 70.0
  },
  "tracks": [
    {
      "id": "dialog",
      "path": "dialogue_stereo.wav",
      "role": "dialogue",
      "input_channels": 2,
      "target_lufs": -18.0
    },
    {
      "id": "music",
      "path": "music_stereo.wav",
      "role": "music",
      "input_channels": 2,
      "gain_db": -1.0,
      "ducking": {
        "depth_db": 9.0,
        "release_ms": 300.0
      }
    },
    {
      "id": "effects",
      "path": "sfx_stereo.wav",
      "role": "sfx",
      "input_channels": 2,
      "gain_db": -2.0
    }
  ],
  "segments": [
    {"id": "part_01", "start_s": 0.0, "end_s": 2.0},
    {"id": "part_02", "start_s": 2.0, "end_s": 4.0},
    {"id": "part_03", "start_s": 4.0, "end_s": 6.4}
  ]
}
```

兼容字段别名包括 `dialogue/dialog/对白`、`music/音乐`、`sfx/effect/音效`，以及 `start/start_s`、`end/end_s`、`audio_tracks/tracks` 等。建议使用上面的规范字段。

### 音轨角色与参数覆盖

- `dialogue`：执行门限降噪、活跃检测、响度均衡。
- `music`、`sfx`：执行侧链闪避和静态增益。
- 每条音轨可覆盖 `noise_gate`、`loudness`、`ducking`、`target_lufs`、`gain_db`。
- `input_channels` 建议声明。它与文件实际声道数不一致时，该轨跳过并记录原因。

### 声道映射矩阵

输出声道数与输入不同时，必须提供显式矩阵：

```json
{
  "id": "dialog",
  "path": "dialogue_mono.wav",
  "role": "dialogue",
  "input_channels": 1,
  "channel_map": {
    "method": "upmix",
    "description": "mono to stereo",
    "matrix": [
      [0.70710678],
      [0.70710678]
    ]
  }
}
```

矩阵形状为 `输出声道数 x 输入声道数`，按行声明输出声道。输出 5.1 顺序为：

```text
FL, FR, FC, LFE, BL, BR
```

`method` 可取 `upmix`、`downmix`、`identity`、`explicit`。缺少匹配矩阵或矩阵方向错误时，该轨跳过而非静默处理。

## 参数校验区间

参数在启动阶段、任何输出目录创建或音频写入之前校验。

| 参数 | 合法区间 / 约束 |
| --- | --- |
| `sample_rate` | 仅 48000 |
| `output.channels` | 2 或 6 |
| `target_lufs` | -60 至 -5 |
| `duration_s` | 0.01 至 21600 |
| `noise_gate.window_ms` | 5 至 1000 |
| `noise_gate.hop_ms` | 1 至 window_ms |
| `noise_gate.threshold_db` | -100 至 -5 |
| `noise_gate.attenuation_db` | 0 至 60 |
| `noise_gate.snr_margin_db` | 0 至 24 |
| `noise_gate.noise_estimate_window_ms` | window_ms 至 10000 |
| `noise_gate.noise_percentile` | 0 至 50 |
| `noise_gate.attack_ms` | 0 至 1000 |
| `noise_gate.release_ms` | 0 至 5000 |
| `loudness.window_ms` | 必须恰为 400 |
| `loudness.tolerance_lu` | 0 至 1 |
| `loudness.attack_ms` | 0 至 500 |
| `loudness.release_ms` | 0 至 5000 |
| `loudness.max_gain_db` | 0 至 40 |
| `ducking.depth_db` | 0 至 48 |
| `ducking.attack_ms` | 0 至 500 |
| `ducking.release_ms` | 0 至 10000 |
| `ducking.vad_threshold_db` | -100 至 -5 |
| `ducking.vad_snr_db` | 0 至 40 |
| `ducking.vad_window_ms` | 1 至 100 |
| `ducking.min_gap_ms` | 0 至 1000 |
| `tracks[].gain_db` | -60 至 24 |
| `tracks[].input_channels` | 1 至 16 |
| `segments[].start_s/end_s` | 非负且 start < end |

## 处理链

1. 解析清单并完成全部参数、分段、矩阵形状和范围校验。
2. 用 FFmpeg 解码并统一到 48 kHz，检查声道和有限样本。
3. 对每条可读音轨扣除直流偏移，按时间线补零或截断，再执行显式声道映射。
4. 对白：
   - 以配置窗口计算帧能量；
   - 以滑动窗口分位数估计局部噪声底噪；
   - 低于门限的帧按底噪差值和 `attenuation_db` 衰减；
   - 平滑门限包络，并用输入/输出 SNR 约束限制最大改善量；
   - 用 K 加权 400 ms 窗口测量短时响度；
   - 以测量窗口对齐的平滑增益包络迭代均衡到目标响度；
   - 最后执行全采样峰值保护。
5. 对白活跃检测：
   - 使用居中滑动 RMS；
   - 自适应阈值取配置绝对阈值与局部底噪加 `vad_snr_db` 的较大者；
   - 连续无静音素材回退到绝对阈值；
   - 合并短间隙，输出样本索引区间。
6. 音乐/音效：
   - 对活跃区间并集生成样本级闪避包络；
   - 活跃起点开始 attack，活跃结束后在 release 时间内恢复到 0 dB；
   - 再应用音轨静态增益。
7. 汇总多声道混音，执行最终峰值保护。
8. 按分段范围切片并分别写 24-bit PCM WAV。
9. 写 JSON 元数据。

### BS.1770 口径

- 使用标准 48 kHz K 加权两级预滤波。
- 积分响度使用 400 ms 块、门限积分：
  - 绝对门限 `-70 LUFS`；
  - 相对门限为绝对门限通过块的平均响度 `-10 LU`；
  - 常量偏移 `-0.691`。
- 5.1 声道权重为 `1, 1, 1, 0, 1.41, 1.41`，LFE 不参与响度积分。
- 短时响度采用 400 ms 窗口，元数据中的偏差按活跃帧统计。

## 分段与边界

工具先按完整时间线完成一次有状态处理，再切出各段独立 WAV。这等价于按顺序流式处理分段，并天然共享：

- 降噪底噪估计；
- 响度增益包络；
- 对白活跃区间；
- 闪避包络历史。

相邻段严格保持标称采样数。边界处使用一个平均样本交接：前段最后一个样本和后段第一个样本都设为边界两侧原样本的平均值。因此：

- 每个输出 WAV 长度为 `round((end_s - start_s) * 48000)`；
- 相邻段边界对应样本差为 0；
- 无 overlap、无增益重置、无相位跳变。

元数据中的 `boundary_method` 和 `boundary_handoff_samples` 记录该处理。

## 输出与元数据

- 单个全时段输出：`full.wav`
- 分段输出：`<segment id>.wav`
- 元数据：`mix_metadata.json`

元数据包含：

- 输出格式、声道顺序、总样本数；
- 每轨状态、跳过原因、解码信息；
- 上混/下混方式、矩阵和应用状态；
- 每条对白的输入/输出 SNR、底噪估计、SNR 余量约束结果；
- 每条对白的积分响度、400 ms 最大偏差和是否在 ±1 LU 内；
- 样本级对白活跃区间和对白段积分响度；
- 每条音乐/音效的闪避深度、attack/release、最小增益；
- 每个输出段的文件、样本范围、积分响度和峰值；
- 汇总状态和出口码。

`None` 表示该测量不适用或没有有限值，例如全静音素材的积分响度。

## 错误码与退出码

| 退出码 | 场景 | stderr 业务码 |
| --- | --- | --- |
| 0 | 至少一条音轨成功处理并写出 | — |
| 0 | 部分轨缺失/格式不支持/无匹配矩阵，其余继续 | 成功摘要 + 警告 |
| 1 | 全部音轨不可读或全部处理失败，或全部段写出失败 | `MIX2001` / `MIX3001` |
| 2 | 清单、参数、矩阵、分段或命令行非法 | `MIX1000` / `MIX1001` / `MIX1002` / `MIX1003` |
| 2 | 缺少 FFmpeg | `MIX1005` |

非法参数不会创建输出目录，不会留下 WAV 或 JSON。全部音轨不可读时会写诊断元数据，便于定位，但不会写 WAV。

## 测试与实际验证

运行：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

本次执行结果：

```text
........                                                                 [100%]
9 passed in 11.42s
```

测试覆盖：

- 三段时间线的确定性对白、音乐、音效混音；
- 每段 400 ms 短时响度偏差 <= 1 LU；
- 闪避 attack/release 的样本级起点和释放完成点；
- 相邻段边界对应样本差 <= 1e-3；
- 缺轨和坏格式跳过且退出码 0；
- 全部不可读退出码 1；
- 非法参数退出码 2 且输出目录不存在；
- 5.1 显式上混、6 -> 2 显式下混和缺矩阵跳过；
- BS.1770 K 加权积分响度与参考实现的一致性。

实际合成样本验证值见 `artifacts/validation-report.json`。这些数值由仓库内确定性合成器生成，不依赖外部数据。

## 设计说明

- 24-bit 量化前会执行峰值保护，保证样本不超过 0 dBFS。
- 输入支持 FFmpeg 可解码的音频容器；任务验收素材为 WAV。
- 如果目标响度要求超过素材可提供的余量，元数据会保留实际偏差和峰值保护状态，不会伪造达标。
- 目前没有实时音频接口，也不依赖音频设备；全部处理为离线文件流程。
