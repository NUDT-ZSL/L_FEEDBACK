# SkillSFX

`SkillSFX` 是一个从零实现的可复现游戏技能音效批量生成工具。它把起手挥砍、命中冲击、金属尾音等 WAV 事件段叠加到一条目标音轨上，按技能等级执行总线处理，并按配方种子生成可逐样本复现、互不干扰的随机变体。

项目只依赖 Python 3.11+ 与 NumPy。FFmpeg/ffprobe 仅用于可选的交叉核对，不参与编码或生成。

## 已实现能力

- JSON 配方严格校验，缺失字段、未知字段、非法数值和非法 JSON 均返回稳定业务错误码。
- 内置确定性合成器生成 44.1/48 kHz、单/双声道、16/24/32-bit/float32 测试源素材。
- 源重采样、任意“源声道 → 目标声道”矩阵映射、短源补零、长源裁剪。
- 每个变体按 `SeedSequence([base_seed, variant_index, event_index, stream_id])` 派生独立随机流，事件增益抖动、时移抖动和尾音泛音互不影响。
- 总线链：直流去除 → 参数化软压缩 → 尾音淡出 → 响度归一化 → 真峰值限制器。
- 4 倍过采样真峰值估计，限制器记录触发次数和触发时间区间；输出编码后再次检查削波。
- 支持 PCM 16/24/32-bit 和 IEEE float32 WAV，采样率 44.1/48 kHz，单/双声道。
- 写入后回读验证 RIFF/WAVE 头、数据块长度、帧数、采样率、声道、位深和格式。
- 生成包含源哈希、时间线、随机实际值、总线参数、输出哈希和时长的 `manifest.json`。
- `verify` 只读取清单与清单引用的源文件，重建每个变体并逐字节比较输出 WAV。
- 并发批量生成使用原子目录发布；任一工作线程失败时删除整个暂存批次，不发布半成品。

## 快速开始

```bash
python -m pip install "numpy>=1.26"

# 1. 生成内置测试素材
python skillsfx.py synth-sources --out examples/sources

# 2. 选择一个技能等级并批量生成
python skillsfx.py generate \
  --recipe examples/ember_cleave.json \
  --level 3 \
  --output build/ember_level_3 \
  --workers 4

# 3. 只依据清单重放并校验
python skillsfx.py verify \
  --manifest build/ember_level_3/manifest.json
```

Windows PowerShell 使用相同命令，将续行符 `\` 换成反引号或写成单行即可。

第二个示例使用 44.1 kHz、单声道、float32：

```bash
python skillsfx.py generate \
  --recipe examples/frost_ring.json \
  --level 5 \
  --output build/frost_level_5 \
  --workers 4
```

也可通过模块或安装后的控制台命令运行：

```bash
python -m skillsfx --version
python -m pip install -e .
skillsfx --version
```

## 配方结构

完整示例见 `examples/ember_cleave.json`。

```json
{
  "schema_version": 1,
  "skill_name": "Ember Cleave",
  "variant_count": 6,
  "seed": 20261117,
  "output": {
    "sample_rate": 48000,
    "bit_depth": 24,
    "channels": 2,
    "duration_ms": 3200
  },
  "events": [
    {
      "role": "slash",
      "source": "sources/slash_44k_mono.wav",
      "offset_ms": 20,
      "duration_ms": 240,
      "gain": 0.82,
      "channel_map": "mono_to_stereo"
    }
  ],
  "randomization": {
    "gain_jitter_db": 0.65,
    "time_jitter_ms": 0.45,
    "tail_harmonics": {
      "enabled": true,
      "fundamental_hz": 1750,
      "ratios": [1.0, 1.47, 2.09, 2.71],
      "gain_db_min": -30,
      "gain_db_max": -18,
      "decay_ms": 920,
      "attack_ms": 2.5
    }
  },
  "bus_by_level": {
    "3": {
      "compressor": {
        "threshold_db": -30,
        "ratio": 4.2,
        "knee_db": 4,
        "attack_ms": 3,
        "release_ms": 70
      },
      "tail_fade_ms": 600,
      "limiter": {
        "ceiling_dbtp": -1.0,
        "lookahead_ms": 6,
        "attack_ms": 0.6,
        "release_ms": 75
      }
    }
  },
  "normalization": {
    "target_lufs": -12,
    "min_gain_db": -18,
    "max_gain_db": 18
  }
}
```

关键规则：

- `sample_rate` 仅允许 `44100`、`48000`；`channels` 仅允许 `1`、`2`。
- `bit_depth` 允许 `16`、`24`、`32`、`"float32"`；也可用 `32` 加 `"sample_format": "float"`。
- `variant_count` 为 `1..1000`，`seed` 为 `0..2^63-1`。
- `duration_ms` 是整条输出音轨长度。事件必须从输出时长内开始，尾部超出部分会被截断。
- `duration_ms`（事件级）可省略；省略时使用重采样后的完整源长度。指定时，短源补零、长源裁剪。
- 所有随机时移的最坏情况也必须落在输出时间线内，因此靠近 0 的事件应给 `offset_ms` 留出大于 `time_jitter_ms` 的余量。
- `role: "tail"` 的事件会叠加按种子随机化的泛音层。
- `bus_by_level` 可包含任意非负整数等级键，`generate --level` 必须命中其中一个。
- `normalization` 必须且只能指定 `target_lufs` 或 `target_rms_dbfs` 之一。

### 声道映射

支持字符串模式：

- `identity` / `stereo` / `passthrough`
- `mono_to_stereo` / `duplicate`
- `stereo_to_mono` / `mono` / `downmix`
- `sum_to_mono`
- `swap`
- `left` / `right`

支持显式规则数组，索引从 0 开始，可对同一目标声道叠加多个源：

```json
"channel_map": [
  {"source": 0, "target": 0, "gain": 0.92},
  {"source": 0, "target": 1, "gain": 0.76}
]
```

支持目标行 × 源列矩阵：

```json
"channel_map": {
  "mode": "matrix",
  "matrix": [[1.0, 0.0], [0.0, 1.0]]
}
```

## 输出与清单

每次 `generate` 产生一个新目录：

```text
build/ember_level_3/
  Ember_Cleave_l3_v0000.wav
  ...
  manifest.json
```

`manifest.json` 记录：

- 配方路径与 SHA-256、等级、基础种子和随机化配置；
- 每个变体派生的种子、源文件哈希、实际帧偏移、毫秒时移、增益抖动实际值；
- 泛音频率、增益、相位和声道权重等实际抽样值；
- 总线/压缩/淡出/归一化/限制器参数；
- 归一化增益、限制器触发次数与区间、最终峰值和真峰值；
- 输出 WAV 的完整 SHA-256、帧数和时长；
- 淡出截断、归一化钳制、限制器安全修正等警告。

输出目录必须不存在或为空，批次在隐藏暂存目录完成全部编码、回读和清单写入后才整体改名发布。

## 业务错误码

错误以单行 JSON 写入 stderr，退出码为 `2`：

```json
{"error_code":"SFX-E2003","message":"event offset is outside the output timeline","context":{"event_index":2,"source_path":"sources/tail.wav"}}
```

| 错误码 | 含义 |
| --- | --- |
| `SFX-E1001` | 配方不可读、不是严格 JSON |
| `SFX-E1002` | 字段缺失、类型错误、未知字段 |
| `SFX-E1003` | 数值或配置组合非法 |
| `SFX-E2001` | 源 WAV 不可读 |
| `SFX-E2002` | 源 WAV 结构或编码不受支持 |
| `SFX-E2003` | 事件偏移为负或不在输出时长内 |
| `SFX-E2004` | 声道映射无法应用 |
| `SFX-E3001` | 总线/DSP 参数或泛音频率非法 |
| `SFX-E4001` | WAV 编码或回读头校验失败 |
| `SFX-E4002` | 发现削波或超过真峰值上限 |
| `SFX-E5001` | 清单重放不一致 |
| `SFX-E6001` | 批次执行失败并清理暂存产物 |
| `SFX-E9001` | 未预期内部错误 |

## 验证

```bash
# 11 项单元/集成测试
python -m unittest discover -s tests -v

# 完整演示验证：多等级、四种输出格式、并发确定性、失败清理、ffprobe 交叉检查
python tools/validate_demo.py
```

验证报告写入 `build/validation/report.json`。实际执行结果见 `VALIDATION.md`。

## 项目结构

```text
skillsfx/
  cli.py         CLI、stdout/stderr 协议
  recipe.py      严格配方解析与数据模型
  synth.py       确定性测试素材合成器
  dsp.py         重采样、映射、压缩、淡出、响度、限制器
  wavio.py       WAV 读写、编码、回读校验
  engine.py      时间线、批量并发、原子发布、清单重放
  errors.py      稳定业务错误码
examples/        两个可直接运行的配方
tests/           unittest 测试套件
tools/           完整演示验证脚本
docs/            技术设计说明
```

## 明确边界

- 源 WAV 仅支持单/双声道、16/24/32-bit PCM 或 float32；不支持压缩 WAV、RF64。
- 输出仅支持 44.1/48 kHz、单/双声道、16/24/32-bit PCM 或 float32。
- 清单重放不读取配方，但会读取清单引用并校验哈希的源 WAV；移动输出目录时建议保留原始源路径或同步移动源素材。
- 同一运行时、同一配方和种子可逐字节复现；跨 Python/NumPy/数学库版本不承诺最后一位浮点完全一致。
- 源素材路径相对于配方文件解析；清单中的源路径通常相对于清单目录记录（Windows 跨盘符时为绝对路径），并同时保留绝对诊断路径。

更详细的算法、随机流、封装与并发设计见 `docs/TECHNICAL.md`。
