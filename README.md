# 对白驱动混音器（Dialogue Mixer）

本地、确定性的 Windows 音频工具：读取项目 JSON，按对白区间对音乐/音效轨执行采样级自动闪避（ducking），对多语言配音执行保持音高的 WSOLA 时间伸缩，完成多轨混合、等功率交叉淡化、峰值归一化，并输出轨道索引与结构化事件日志。

所有测试样本由仓库内脚本生成，不访问网络，不依赖外部素材。

## 环境

- Windows 10/11 64 位
- Python 3.10+
- NumPy 1.24+
- FFmpeg 6.0+，且 `ffmpeg` 在 `PATH` 中

当前代码只使用 Python 标准库、NumPy 和 FFmpeg 命令行，不依赖 Docker 或其他容器。

## 快速开始

Windows `cmd.exe`：

```bat
py -3 -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
ffmpeg -version

py scripts\generate_samples.py --out samples
py -m dialogue_mixer mix samples\valid\project.json
echo %ERRORLEVEL%

py -m unittest discover -s tests -v
py scripts\verify_acceptance.py --samples samples --report reports\acceptance_report.json
echo %ERRORLEVEL%
```

PowerShell 中查看退出码使用 `$LASTEXITCODE`。也可以直接运行 `py -m dialogue_mixer samples\valid\project.json`，CLI 会自动补上 `mix` 子命令。

可用 `--audio`、`--index`、`--log` 覆盖项目中的输出路径。

生成文件：

- `samples/valid/output/demo_mix.wav`：48 kHz、单声道、32-bit float WAV
- `samples/valid/output/track_index.json`：轨道索引
- `samples/valid/output/run_log.json`：处理日志
- `samples/invalid/`：缺失文件、缺失闪避目标、不可合并重叠的部分失败样本
- `samples/fatal/`：非法采样率样本
- `reports/acceptance_report.json`：自动验收报告

## 项目 JSON

相对路径均相对于项目 JSON 所在目录解析。`gain` 为线性增益；`depth_db` 为负 dB 闪避深度。

```json
{
  "schema_version": 1,
  "name": "deterministic_dialogue_demo",
  "sample_rate": 48000,
  "duration_seconds": 9.0,
  "tracks": [
    {"id": "dialogue", "kind": "dialogue", "gain": 1.0},
    {"id": "music", "kind": "music", "gain": 0.8},
    {"id": "sfx", "kind": "sfx", "gain": 0.7}
  ],
  "clips": [
    {
      "id": "voice_zh",
      "track": "dialogue",
      "file": "audio/voice_zh.wav",
      "start": 1.0,
      "gain": 1.0
    },
    {
      "id": "music_a",
      "track": "music",
      "file": "audio/music_a.wav",
      "start": 0.0,
      "gain": 0.95
    }
  ],
  "dialogues": [
    {
      "id": "line_zh_01",
      "start": 1.0,
      "end": 2.2,
      "stretch_ratio": 1.2,
      "voice_clip_id": "voice_zh",
      "duck_tracks": ["music", "sfx"]
    }
  ],
  "ducking": {
    "enabled": true,
    "target_tracks": ["music", "sfx"],
    "depth_db": -12.0,
    "attack_seconds": 0.06,
    "release_seconds": 0.35,
    "attack_curve": "cosine",
    "release_curve": "cosine"
  },
  "crossfades": [
    {
      "from_clip_id": "music_a",
      "to_clip_id": "music_b",
      "duration_seconds": 0.5
    }
  ],
  "normalization": {"peak_ceiling_dbfs": -0.2},
  "output": {
    "audio": "output/demo_mix.wav",
    "index": "output/track_index.json",
    "log": "output/run_log.json"
  }
}
```

字段说明：

- `tracks[].kind`：`dialogue`、`music`、`sfx` 或 `other`。只有 `music` 轨允许声明交叉淡化。
- `clips[]`：每项包含轨道、文件、起始秒和线性增益。
- `dialogues[]`：对白区间为半开区间 `[start, end)`，时间戳最终转换为整数采样边界。
- `dialogues[].stretch_ratio`：请求的“目标时长/源时长”比例；实际输出以 `start`/`end` 区间为权威目标，偏差超过 1 个采样点时记录 `STRETCH_RATIO_ADJUSTED` 警告。
- `dialogues[].voice_clip_id`：带 `stretch_ratio` 时必填。该配音原片段不再按普通片段重复混入，伸缩结果放置在 `dialogue.start`。
- `dialogues[].duck_tracks`：省略时使用 `ducking.target_tracks`；空数组表示该对白不闪避任何轨道。
- 对白可覆盖 `depth_db`、`attack_seconds`、`release_seconds`、`attack_curve`、`release_curve`。
- 曲线支持 `linear`、`sine`/`equal_power`、`cosine`、`smoothstep`、`exponential`。
- `crossfades[]`：区域从后一个片段起始采样开始，长度由 `duration_seconds` 决定，必须完整落在前段尾部与后段头部内。
- `normalization.peak_ceiling_dbfs`：默认 `0.0`，只允许 `<= 0`；归一化只降低峰值，不放大低于上限的混音。

## 处理语义

### 时间轴

所有边界使用 `round(seconds * sample_rate)` 转为整数采样。对白区间、WSOLA 输出、索引和混音都共享同一整数采样时间轴。索引中的 `actual_start_seconds` / `actual_duration_seconds` 由实际采样数反算，验收要求最大误差 `<= 1` 采样。

### WSOLA 时间伸缩

- 默认 48 kHz 参数：帧长 2048、合成 hop 512、搜索半径 480，Hann 窗重叠相加。
- 每个分析帧在名义位置附近搜索与前一帧自然延续最相似的波形，从而改变时长但保持局部周期和音高。
- 输出长度被精确裁剪/补齐到对白区间的采样数，因此片段起止边界为 0 采样误差。
- 结果索引标记 `source_kind: "stretched"`；若请求比例对应长度与源完全一致，则标记为 `"original"`，并记录 `transform_applied: false`。

### 闪避包络

深度 `d = 10^(depth_db/20)`，曲线函数为 `C(p)`，`p ∈ [0,1]`：

- 攻击阶段：`g = 1 + (d - 1) * C(p_attack)`
- 对白活动段：`g = d`
- 释放阶段：`g = d + (1 - d) * C(p_release)`
- 释放阶段从对白结束采样开始，持续 `round(release_seconds * sample_rate)` 个采样。
- 多个重叠对白对同一轨道的包络逐采样取最小值。

闪避先独立计算轨道包络，再与轨道增益和逐样本音频相乘；交叉淡化只作用于片段的增益包络，不参与闪避包络计算。

### 重叠策略

- 不同配音片段的对白区间可以重叠，包络按逐采样最小值合并，并记录 `mergeable: true` 的重叠事件。
- 同一 `voice_clip_id` 的多个伸缩实例若目标区间重叠，无法无歧义地重放同一源时间轴；后一个对白整体跳过并记录 `UNMERGEABLE_OVERLAP`，其余片段继续处理。
- 同轨交叉淡化区域相互重叠时，后一个交叉淡化跳过并记录 `UNMERGEABLE_OVERLAP`。

### 交叉淡化

交叉淡化采用等功率曲线，`p` 从 0 到 1：

- 前段淡出：`cos(pi/2 * p)`
- 后段淡入：`sin(pi/2 * p)`
- 两条曲线平方和为 1。

### 峰值归一化

所有轨道按 `clip_gain * track_gain * duck_envelope * crossfade_envelope` 逐样本混合后测量峰值。

- 若峰值高于项目上限，`normalization_factor = ceiling / peak`；否则因子为 `1.0`，不放大素材。
- 最终样本限制在 `[-1, 1]`，输出峰值保证 `<= 0 dBFS`。
- 日志同时记录归一化前/后峰值、因子、因子 dB、是否应用及缩放后削波采样数。
- 输出 WAV 写完后会重新读取并验证采样率、采样数和峰值。

## 输出

### 轨道索引

每个输出片段包含：

- `clip_id`、`track_id`、可选 `dialogue_id`
- `source_kind`：`original` 或 `stretched`
- `source_file`、`source_duration_samples`
- `start_sample`、`end_sample`、`duration_samples`
- `actual_start_seconds`、`actual_duration_seconds`、`actual_end_seconds`
- `stretch` 伸缩参数、`crossfade_in`/`crossfade_out`
- `timeline_alignment_error_samples`

顶层 `alignment.max_alignment_error_samples` 应 `<= 1`。

### 结构化日志

日志包含：

- 各阶段状态与摘要
- 片段、对白、重叠、伸缩、闪避事件和交叉淡化计数
- 每个闪避事件的采样边界、攻击/释放参数与曲线
- 每条目标轨道的包络最小/最大值和受闪避采样数
- WSOLA 参数及请求/实际比例
- 归一化因子与实际输出峰值
- `errors`、`warnings`、`business_error_codes`

日志不写墙钟时间，重复运行同一项目时音频和索引字节一致。

## 退出码与错误处理

| 退出码 | 状态 | 含义 |
|---|---|---|
| `0` | `success` | 全部处理成功 |
| `3` | `partial` | 有业务错误，已跳过受影响片段，并保留其余音频/索引/日志 |
| `2` | `failed` | 项目无法建立有效时间轴，或输出写入/验证失败 |

常见业务码：

- `INVALID_SAMPLE_RATE`、`INVALID_DURATION`、`INVALID_PROJECT_JSON`、`MISSING_TRACKS`：致命输入错误
- `MISSING_AUDIO_FILE`、`AUDIO_DECODE_ERROR`：音频源不可用，片段跳过
- `MISSING_TRACK`、`MISSING_DUCK_TARGET`、`MISSING_VOICE_CLIP`、`INVALID_VOICE_TRACK`：轨道/引用冲突
- `UNMERGEABLE_OVERLAP`：不可无歧义合并的重叠
- `CROSSFADE_CLIP_MISSING`、`CROSSFADE_REGION_INVALID`、`CROSSFADE_TRACK_MISMATCH`、`CROSSFADE_NOT_MUSIC`：交叉淡化错误
- `TIME_STRETCH_FAILED`、`TIME_STRETCH_LENGTH_MISMATCH`：伸缩失败
- `INDEX_WRITE_ERROR`、`AUDIO_WRITE_ERROR`、`OUTPUT_VERIFICATION_FAILED`：输出错误

即使发生业务错误，只要采样率、总时长和输出路径有效，工具仍会写出未受影响部分的混音、索引与日志。

## 验证

单元与集成测试：

```bat
py -m unittest discover -s tests -v
```

覆盖内容：WSOLA 时长与音高、闪避攻击/保持/释放采样位置、重叠包络最小值、正常项目、部分失败、非法采样率、索引对齐、等功率交叉淡化、退出码和字节级确定性。

自动验收：

```bat
py scripts\verify_acceptance.py --samples samples --report reports\acceptance_report.json
```

本次实际验证结果（当前环境 Python 3.14.4、NumPy 2.5.5、FFmpeg 8.0.1；目标 Windows 运行方式使用同一跨平台代码路径）：

| 项目 | 结果 |
|---|---|
| 测试 | 13/13 通过 |
| 验收检查 | 20/20 通过 |
| 正常项目退出码 | `0` |
| 输出峰值 | `-0.200000 dBFS`（32-bit float 实测） |
| 峰值上限 | `-0.2 dBFS`，满足 `<= 0 dBFS` |
| 归一化因子 | `0.8793871210292404` |
| 索引最大对齐误差 | `1.455e-11` 采样 |
| 闪避事件 / 重叠事件 | `8` / `1` |
| 时间伸缩片段 | 4 个（3 个实际 WSOLA，1 个比例 1.0） |
| 等功率交叉淡化 | 1 个，独立于闪避 |
| 部分失败项目 | 退出码 `3`，保留 WAV/索引/日志 |
| 部分失败业务码 | `MISSING_AUDIO_FILE`, `MISSING_DUCK_TARGET`, `UNMERGEABLE_OVERLAP` |
| 致命项目 | 退出码 `2`，日志记录 `INVALID_SAMPLE_RATE`，不生成混音 |
| 重复运行音频 SHA-256 | `36d2652802e8567c8d52de38044e8148d4aea8d68ce792ed19d2484735152d26` |
| 样本重新生成 | 12 个输入文件 SHA-256 全部不变 |

详细逐项结果见 `reports/acceptance_report.json`。当前验收在 WSL Linux 上执行；代码未使用 POSIX 专用命令，Windows 分支仅用于隐藏 FFmpeg 控制台窗口。实际部署到 Windows 时按上方命令运行即可。

## 代码结构

- `dialogue_mixer/config.py`：项目 JSON 解析与业务校验
- `dialogue_mixer/wsola.py`：保持音高的 WSOLA
- `dialogue_mixer/envelope.py`：采样级闪避包络与重叠检测
- `dialogue_mixer/engine.py`：伸缩、交叉淡化、混音、归一化、索引和日志
- `dialogue_mixer/audio_io.py`：FFmpeg 解码与 32-bit float WAV 输出/验证
- `dialogue_mixer/samplegen.py`：确定性语音、音乐、音效样本生成
- `scripts/generate_samples.py`：样本生成入口
- `scripts/verify_acceptance.py`：端到端验收入口

