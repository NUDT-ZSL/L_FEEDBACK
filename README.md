# 离线多声道响度归一化管线

这是一个不依赖容器、可直接在制作流水线中调用的离线音频处理工具。它读取
PCM/浮点 WAV，按 ITU-R BS.1770-4 进行 K 加权和门限积分，按 EBU R128
目标响度计算增益；需要时使用确定性前视峰值限制器，并以原子写入方式输出
新 WAV 与 JSON 报告。

## 功能

- 读取 16/24/32 位整型 PCM 和 32/64 位浮点 WAV/WAVEX。
- 支持 32、44.1、48、88.2、96、176.4、192 kHz。
- 按标准系数和模拟原型重新推导 K 加权滤波器，支持任意受支持采样率。
- 使用 400 ms 积分块、100 ms 步进（75% 重叠）计算响度序列。
- 实现 -70 LUFS 绝对门限和“绝对门限结果 -10 LU”相对门限。
- 5.1 声道顺序为 `L, R, C, LFE, Ls, Rs`，LFE 权重为 0。
- 全局增益后若峰值超过 0 dBFS，可启用前视限峰并记录最大限制量。
- 输出旁路默认生成 `<output>.json`，包含前后整体响度、分段响度和积分块。
- 相同输入和参数下，WAV PCM 与 JSON 报告字节级可复现。
- 输入文件绝不写入；输出先写同目录临时文件，成功后再原子替换。

## 安装

Python 3.9 或更高版本，且需要系统安装 `libsndfile`：

```bash
sudo apt-get install libsndfile1 python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

也可以使用 `uv`：

```bash
uv venv .venv --python python3
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

## 生成测试样本

```bash
python scripts/generate_test_samples.py --output-dir testdata --duration 5.2
```

脚本生成以下确定性素材：

- 立体声：32 kHz PCM_32、44.1 kHz PCM_16、48 kHz PCM_24、96 kHz FLOAT。
- 5.1：44.1 kHz DOUBLE、48 kHz PCM_24。
- 限峰测试：48 kHz 立体声 FLOAT，包含短暂高电平瞬态。
- 异常测试：22050 Hz 不支持采样率、零帧 WAV、损坏 WAV。

## 命令行

```bash
python -m loudness_pipeline INPUT.wav OUTPUT.wav \
  --target -23 \
  --report OUTPUT.report.json \
  --segment-seconds 3 \
  --limiter
```

参数：

| 参数 | 说明 |
| --- | --- |
| `INPUT.wav` | 输入 WAV |
| `OUTPUT.wav` | 输出 WAV，必须与输入路径不同 |
| `-t/--target/--target-lufs` | 目标整体响度，范围 `-35` 到 `-5`，默认 `-23` |
| `--limiter` / `--no-limiter` | 是否启用前视峰值限制，默认启用 |
| `--report PATH` | JSON 报告路径，默认 `<OUTPUT.wav>.json` |
| `--segment-seconds SECONDS` | 报告分段时长，默认 `3` |

成功时 stdout 输出：

```text
OK
input_lufs: -19.82 LUFS
output_lufs: -23.00 LUFS
gain_db: -3.18 dB
applied_base_gain_db: -3.18 dB
limiter_applied: false
max_limiting_db: 0.00 dB
elapsed_seconds: 0.386
output: /path/out.wav
report: /path/out.wav.json
```

安装包后也可使用 `loudness-normalize` 命令。`elapsed_seconds` 只出现在
终端摘要中，不写入 JSON，因此不影响报告复现性。

## 测量与处理实现

### K 加权

48 kHz 系数与 BS.1770-4 表 1 一致。其他采样率使用标准给出的 48 kHz 模拟
原型中心频率、增益和 Q 值，通过双线性变换重新推导：

- 高架滤波器：`1681.974450955533 Hz`、`3.999843853973347 dB`、
  `Q=0.7071752369554196`
- RLB 高通：`38.13547087602444 Hz`、`Q=0.5003270373238773`

两级双二阶滤波器级联后，在频域执行零填充 FFT。预留至少一秒尾零，避免
IIR 尾部的循环卷绕。测试中用逐采样直接型实现进行独立对照，最大误差小于
`1e-9`。

### 积分与门限

每个积分块的加权能量为：

```text
E = sum_channels(weight[c] * mean(K_weighted[c]^2))
L = -0.691 + 10 * log10(E)
```

先筛选 `L >= -70 LUFS` 的块，计算绝对门限积分值；再保留
`L >= 绝对门限积分值 - 10 LU` 的块，得到最终整体响度。全部积分块的
响度及门限状态都会写入报告的 `integration_blocks`。短于 400 ms 的输入
按实际长度形成一个标注为 `is_partial` 的积分块。

### 声道权重

| 布局 | 权重 |
| --- | --- |
| 单声道 | `[1]` |
| 立体声 | `[1, 1]` |
| 5.1 | `[1, 1, 1, 0, 1, 1]` |
| 7.1 | `[1, 1, 1, 0, 1, 1, 1, 1]` |

输出数组按 `(frames, channels)` 写入，保持输入声道顺序和交错方式。

### 增益与限峰

初始增益为 `target_lufs - input_lufs`。若增益后样本峰值不超过 `1.0`，
直接施加全局增益。若会超过 0 dBFS：

- 启用限制器：使用 5 ms 前视、50 ms 释放和多声道联动包络。下降增益提前
  覆盖未来峰值，因此每个输出样本都不超过上限；迭代增加限制前的基础增益，
  使限峰后的整体响度回到目标附近。
- 禁用限制器：使用峰值安全的静态增益，牺牲目标响度而不产生削波。

限峰结果记录 `max_limiting_db`、受限样本数和 `maximum_overshoot`。实现在
最终输出前还会做数值边界保护，但正常路径不会触发额外硬削波。

## JSON 报告

所有浮点数在文本中固定保留两位小数。核心结构如下：

```json
{
  "standard": "ITU-R BS.1770-4 / EBU R128",
  "audio": {
    "sample_rate": 48000,
    "channels": 6,
    "format": "WAV",
    "subtype": "PCM_24",
    "frames": 249600,
    "duration_seconds": 5.20,
    "channel_weights": [1.00, 1.00, 1.00, 0.00, 1.00, 1.00]
  },
  "parameters": {
    "target_lufs": -23.00,
    "segment_seconds": 3.00,
    "limiter_enabled": true
  },
  "processing": {
    "requested_gain_db": -3.18,
    "applied_base_gain_db": -3.18,
    "limiter_applied": false,
    "max_limiting_db": 0.00,
    "limited_sample_count": 0,
    "maximum_overshoot": 0.00,
    "limiter_iterations": 0
  },
  "overall": {
    "before": {"integrated_lufs": -19.82, "sample_peak_dbfs": -10.53},
    "after": {"integrated_lufs": -23.00, "sample_peak_dbfs": -13.71},
    "delta_lufs": -3.18
  },
  "segments": [
    {
      "index": 0,
      "start_seconds": 0.00,
      "end_seconds": 3.00,
      "duration_seconds": 3.00,
      "before_lufs": -19.30,
      "after_lufs": -22.48,
      "delta_lufs": -3.18,
      "is_partial_segment": false,
      "window_duration_seconds": 0.40,
      "is_partial_window": false
    }
  ],
  "integration_blocks": {"before": [], "after": []}
}
```

`segments` 按固定时长切分。“末段不足 400 ms”时，
`is_partial_segment`、`is_partial_window` 为 `true`，
`duration_seconds` 与 `window_duration_seconds` 记录实际长度。

## 错误码

| 错误码 | 退出码 | 场景 |
| --- | --- | --- |
| `E_USAGE` | 2 | 参数或路径关系非法；argparse 错误也以 2 退出 |
| `E_FORMAT` | 3 | 损坏文件、非 WAV、或非支持子类型 |
| `E_EMPTY` | 4 | 零帧音频 |
| `E_SAMPLERATE` | 5 | 采样率不在支持集合 |
| `E_WRITE` | 6 | 临时文件创建、WAV 写入或报告写入失败 |
| `E_SILENT` | 7 | 全部积分块低于 -70 LUFS 绝对门限 |
| `E_INTERNAL` | 8 | 未预期内部异常 |

业务错误信息只写 stderr，例如：

```text
E_FORMAT: cannot inspect WAV file 'testdata/corrupt.wav': ...
```

## 验证

```bash
python -m pytest -q
```

当前测试覆盖 23 个用例，包括：5 种输入子类型、损坏/空/采样率错误、BS.1770
系数、FFT 与直接型滤波一致性、1 kHz 响度基准、绝对/相对门限、5.1 LFE
零权重、短音频部分窗口、PCM 字节级复现、JSON 两位小数、限峰无过冲、
CLI 成功与错误退出码、禁限峰静态安全增益。

完整实测数据见 `VERIFICATION.md`。

## 说明与边界

- 峰值限制使用样本峰值，不计算 BS.1770 Annex 2 的过采样真峰值。
- 支持列表外的采样率会主动拒绝；这是业务约束，不是滤波器能力限制。
- WAV/WAVEX 的声道掩码不属于 `soundfile` 公共读写元数据；本管线保证相同
  声道数、相同通道顺序和相同交错存储方式。
- 禁用限峰时，如果素材波峰因数过高，输出会优先保证 0 dBFS 以下，实际响度
  可能低于目标。
- 报告含输入/输出路径文本，因此“完全一致”指相同命令行与相同路径下重复
  运行；处理耗时不会进入报告。

