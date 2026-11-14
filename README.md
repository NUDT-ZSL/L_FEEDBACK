# Cinemix 离线多轨音频处理器

Cinemix 面向过场动画的离线混音交付。它读取 16/24-bit PCM WAV 多轨音频与 JSON 工程，完成独立重采样、增益/声像、能量恒定交叉淡化、自适应公共增益修正、BS.1770 响度对齐、前馈压缩限幅，最后输出 24-bit PCM WAV。

核心处理只依赖 Python 与 NumPy；FFmpeg 用于验收阶段的独立响度/真峰值交叉测量。

## 环境

- Python 3.10+，64 位
- NumPy 1.24–2.x
- FFmpeg 6.0+，`ffmpeg`/`ffprobe` 位于 `PATH`（验收脚本需要）
- 建议至少 500 MB 可用磁盘空间

Windows PowerShell：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Linux/macOS：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

FFmpeg 需单独安装并把 `ffmpeg.exe` 目录加入 `PATH`。也可在验收时用 `--ffmpeg` 指定绝对路径。

## 快速开始

```bash
# 确定性生成 44.1/48/96 kHz、16/24-bit 多轨样本、三组响度目标及错误工程
python scripts/generate_samples.py --out generated/sample_project

# 广播目标：-23 LUFS / -1 dBTP
python -m cinemix process generated/sample_project/project.json \
  --report generated/sample_project/broadcast_report.json

# 热混音：-14 LUFS / -1 dBTP
python -m cinemix process generated/sample_project/project_hot.json

# 限幅压力目标：-10.2 LUFS / -3 dBTP
python -m cinemix process generated/sample_project/project_stress.json

# 单文件响度测量
python -m cinemix measure generated/sample_project/out/broadcast_mix.wav

# 单元 + 集成测试
python -m unittest discover -s tests -v

# 完整验收：三组目标、FFmpeg 交叉测量、确定性、错误注入
python scripts/verify_acceptance.py \
  --ffmpeg /path/to/ffmpeg \
  --project-dir generated/acceptance \
  --artifact artifacts/acceptance_report.json
```

`process` 默认在 stdout 输出一行成功摘要；所有 info/warning/error 均为 stderr 上的单行 JSON。`--print-report` 会把完整报告写到 stdout，`--report` 会以稳定键序写入 JSON 文件。

## 工程配置

完整示例见 `examples/project.example.json`。路径相对配置文件所在目录解析。

```json
{
  "project": {
    "sample_rate": 48000,
    "output": "out/mix.wav",
    "output_channels": 2,
    "target_lufs": -23.0,
    "true_peak_dbtp": -1.0,
    "duration_seconds": 10.0
  },
  "tracks": [
    {
      "id": "dialogue",
      "path": "dialogue.wav",
      "start": 0.25,
      "gain_db": -2.0,
      "pan": 0.0,
      "fade_in": {"start": 0.25, "end": 0.55, "curve": "s_curve"},
      "fade_out": {"start": 9.3, "end": 9.8, "curve": "s_curve"},
      "resample_to": 48000
    }
  ],
  "crossfades": [
    {
      "from": "dialogue",
      "to": "music",
      "start": 5.0,
      "end": 7.0,
      "curve": "equal_power"
    }
  ],
  "limiter": {
    "attack_ms": 5.0,
    "release_ms": 120.0,
    "ratio": 20.0,
    "lookahead_ms": 7.0,
    "safety_margin_db": 0.25
  },
  "adaptive_gain": {
    "enabled": true,
    "threshold_lufs": -23.0,
    "max_cut_db": 8.0
  }
}
```

字段说明：

- `sample_rate` 是主时间轴/输出采样率；省略时使用第一条轨的采样率。输入采样率不一致时，每轨必须设置 `resample_to` 为主采样率，或设置 `project.resample=true`。
- `start`、所有淡化时间均为秒，可带小数；内部映射到最近的主采样点。
- `gain_db` 范围 `[-96, 24]`；越界会被钳制并产生 `W_GAIN_CLAMPED`，继续处理。`pan` 范围 `[-1, 1]`。
- `fade_in`/`fade_out` 可为秒数（表示时长），也可为 `start/end/duration` 对象。轨道淡出省略时间时对齐轨道尾部。
- `crossfades` 的 `from` 轨在区间内淡出，`to` 轨淡入；同一轨参与的两个区间不得重叠。
- `curve` 可取 `linear`、`equal_power`、`s_curve`。三者均满足重叠区 `g_out² + g_in² = 1`。线性曲线是线性功率斜坡，等功率曲线为正弦/余弦，S 曲线使用 smoothstep 后的 S 形角度映射。
- `adaptive_gain.threshold_lufs` 是触发阈值。只要有轨超过阈值，就对**所有轨施加同一个公共修剪量**，因此配置的相对增益关系保持不变，淡化时间点也不变。
- `limiter.compression_threshold_dbtp` 可显式指定压缩起点；省略时为真峰值上限下方 0.5 dB。

常用别名也受支持，例如 `file`/`path`、`volume_db`/`gain_db`、`target_loudness_lufs`/`target_lufs`、`max_true_peak_dbtp`/`true_peak_dbtp`。

## 处理链

1. 校验 JSON、字段、文件、位深、通道数、采样率与交叉淡化冲突。
2. 各轨独立按有理比例重采样。输出时刻按 `n * in_rate / out_rate` 精确计算，没有累计漂移；插值核为 Kaiser 窗 sinc，降采样时自动缩放截止频率与抽头数。
3. 应用声像、配置增益、轨道淡入淡出、交叉淡化与自适应公共修剪。
4. 在未限幅的混音总线上反复测量整体响度；以目标误差驱动节目增益，并对每个候选执行限幅和真峰值复测。
5. 限幅器使用未限幅混音作为旁链，包含前瞻、attack/release、可配置 ratio、硬采样上限和采样间峰值复测迭代。
6. 量化到 24-bit 后再次测量。若量化后真峰值接近/超过上限，则安全微调并重新量化。
7. 原子写入 WAV 与确定性 JSON 报告。

### ITU-R BS.1770 响度

- 按实际采样率由模拟原型双线性变换生成两级 K 加权 biquad：高架滤波与 38 Hz 高通。
- 使用 400 ms 块、100 ms 步进，声道权重 1.0。
- 先做绝对门限 `-70 LUFS`，再做相对门限 `-10 LU`，返回积分响度。
- IIR 通过 16384 点冲激响应的 FFT 卷积实现，经测试与 FFmpeg `ebur128` 在 44.1/48/96 kHz 上一致。

### 4 倍真峰值

- 每个基带样点生成 4 个相位，使用 65 抽头 Kaiser 窗 sinc 重建。
- 忽略滤波器启动/结束边缘，避免把窗口瞬态误计为节目峰值。
- `fs/4`、相位 `π/4` 的测试信号样品峰值为 `-3.01 dBFS`，4 倍真峰值测得 `-0.00006 dBTP`。

### 限幅器

- 旁链检测器取未限幅立体声总线的逐点最大绝对值。
- 压缩增益为 `(T_comp / peak)^(1 - 1/ratio)`，再由真峰值上限对应的硬增益封顶。
- 前瞻窗口取未来最小目标增益，attack/release 使用一阶平滑；硬封顶保证当前样点不越过内部 ceiling。
- 对限幅结果做 4 倍真峰值测量；若仍过冲，降低内部 ceiling 后重跑，最多 5 次。
- 24-bit 量化后再次验证，保证交付文件而非仅浮点中间结果满足上限。

### 确定性

样本生成器使用固定 PCG64 种子，不依赖外部音频；处理器无时间戳、无随机数、JSON 键排序稳定、WAV 无元数据时间字段。相同输入与相同代码在同一环境中重复运行得到相同 SHA-256。验收中对广播输出重复处理，哈希完全一致。

## 错误码与诊断

所有诊断写 stderr，单行 JSON，含 `level`、`code`、`message`，必要时含 `context`。

| 代码 | 退出码 | 含义 |
|---|---:|---|
| `E_INPUT_MISSING` | 20 | 配置文件或 WAV 不存在/不可读 |
| `E_INPUT_FORMAT` | 21 | 非 PCM、位深/通道数不支持、WAV 损坏 |
| `E_INPUT_SAMPLE_RATE` | 22 | 采样率不一致且未配置重采样 |
| `E_CONFIG_JSON` | 10 | JSON 语法错误 |
| `E_CONFIG_SCHEMA` | 11 | 顶层/节点类型错误 |
| `E_CONFIG_FIELD` | 12 | 字段值非法或时间区间非法 |
| `E_CROSSFADE_CONFLICT` | 30 | 交叉淡化区间重叠或边界量化异常 |
| `E_INPUT_SILENT` | 23 | 混音为静音，无法对齐响度 |
| `E_LIMIT_UNREACHABLE` | 40 | 在给定峰值上限下无法达到目标响度 ±0.5 LU |
| `E_OUTPUT_WRITE` | 50 | 输出 WAV/报告写入失败 |

可恢复警告包括 `W_GAIN_CLAMPED`、`W_PAN_CLAMPED`、`W_LOUDNESS_TOLERANCE_MARGIN`；信息码包括 `I_TRACK_RESAMPLED`、`I_ADAPTIVE_GAIN_APPLIED`。

## 实际验收结果

最终结果同时记录在 `artifacts/acceptance_report.json`。环境：Python 3.12.16、NumPy 2.5.4、FFmpeg 7.0.2-static。

| 工程 | 目标 LUFS | 实测 LUFS | 误差 LU | 真峰值 dBTP | 上限 dBTP | 限幅最大衰减 | 淡化边界误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 广播 | -23.000 | -23.000 | 0.000 | -13.268 | -1.000 | 0.000 dB | 0.000 ms |
| 热混音 | -14.000 | -14.010 | 0.010 | -4.278 | -1.000 | 0.000 dB | 0.000 ms |
| 压力 | -10.200 | -10.245 | 0.045 | -3.249 | -3.000 | 5.934 dB | 0.000 ms |

FFmpeg `ebur128=peak=true` 独立读数分别为：

- 广播：`-23.0 LUFS / -13.3 dBTP`
- 热混音：`-14.0 LUFS / -4.3 dBTP`
- 压力：`-10.2 LUFS / -3.2 dBTP`

广播输出重复运行哈希一致：

```text
ef90409ad6c115055edce707f1b3534559a78a65ec95086cbccab80b2dfa5f73
```

错误注入结果：

- 缺 WAV：`E_INPUT_MISSING` / exit 20
- 非法字段：`E_CONFIG_FIELD` / exit 12
- 采样率不一致：`E_INPUT_SAMPLE_RATE` / exit 22
- 交叉淡化重叠：`E_CROSSFADE_CONFLICT` / exit 30
- 增益 `+99 dB`：钳制到 `+24 dB`，输出成功并产生 `W_GAIN_CLAMPED`

自适应公共修剪在实际样本中为 `-1.268298 dB`，同时作用于对话、音乐和音效轨，配置的相对增益差保持不变。压力工程中限幅器确实介入，最大衰减 `5.934 dB`，量化后最终真峰值仍低于 `-3 dBTP`。

16 项自动化测试全部通过：

```text
Ran 16 tests in 1.652s
OK
```

## 已知边界

- 输入轨支持 mono/stereo、16/24-bit PCM；输出固定 24-bit PCM，通道数 1 或 2。
- 目标响度与峰值上限若在物理上不可同时满足，则返回 `E_LIMIT_UNREACHABLE`，不会静默交付不合格文件。
- 小于 400 ms 的极短信号会以整段作为单个测量块；正式节目应使用完整长度素材。
- 核心处理无 FFmpeg 依赖；FFmpeg 只用于验收交叉测量。
