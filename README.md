# 确定性对白定点修复器

离线命令行工具，用采样点级清单修复 48 kHz 单声道对白 WAV。支持合成片段替换、参考片段补入、区间增益三类操作；不改变整轨采样数，不修改清单区间之外的内容。

实现只依赖 Python 3.11 标准库和 NumPy，不需要 ffmpeg、网络或外部素材。输入和输出均为 PCM WAV。

## 快速开始

```powershell
python repair.py selftest
python repair.py generate-sample --output-dir .\sample
python repair.py process `
  --input .\sample\dialogue.wav `
  --manifest .\sample\repair.json `
  --output .\sample\repaired.wav `
  --report .\sample\repair-report.json
```

也可使用 `python -m dialogue_repair ...`。成功时退出码为 `0`，处理摘要写到 stdout。

## 清单格式

规范字段如下。所有采样位置必须是非负整数，区间均为左闭右开 `[start_sample, end_sample)`。

```json
{
  "version": 1,
  "sample_rate": 48000,
  "quantization_bits": 24,
  "operations": [
    {
      "index": 0,
      "type": "replace",
      "start_sample": 11600,
      "end_sample": 12800,
      "frequency_hz": 523.25,
      "phase_radians": 0.125,
      "amplitude": 0.22,
      "fade_ms": 2.0
    },
    {
      "index": 1,
      "type": "splice",
      "start_sample": 18000,
      "end_sample": 20400,
      "reference_file": "reference.wav",
      "reference_start": 1200,
      "reference_end": 3600,
      "fade_ms": 3.0
    },
    {
      "index": 2,
      "type": "gain",
      "start_sample": 34000,
      "end_sample": 36000,
      "gain": 0.35
    }
  ]
}
```

约束：

- `quantization_bits` 必须是 `16` 或 `24`。
- `operations[*].index` 必须存在，且等于其在数组中的 0 基位置。
- 所有目标区间必须满足 `0 <= start_sample < end_sample <= 输入采样数`，且两两不重叠；首尾相接合法。
- `replace`：`frequency_hz`、`phase_radians` 必填，`amplitude` 可省略，默认 `1.0`，且不得为负。第 `n` 个输出采样点的合成值为 `amplitude * sin(2*pi*frequency_hz*n/48000 + phase_radians)`，`n` 使用整轨绝对采样坐标。
- `splice`：`reference_file` 相对清单所在目录解析；参考 WAV 必须是 48 kHz 单声道 PCM；参考区间与目标区间样本数必须相等。
- `gain`：目标区间每个样本乘以常数 `gain`，`gain >= 0`。gain 不执行淡化，也不触碰区间外样本。
- `replace` 与 `splice` 必须提供 `fade_ms`；`gain` 可省略或给 `0`，给正数会报错。

为便于调用，还接受常见别名：`operation_index`、`operation`、`start`/`end`、`target_start`/`target_end`、`crossfade_ms`、`frequency`、`phase`、`reference`/`reference_path`、`reference_start_sample`/`reference_end_sample`、`source_start`/`source_end`、`linear_gain`、`quantization`、`bit_depth`。冲突别名会产生错误。

## 淡化与拼接语义

`replace` 和 `splice` 的交叉淡化完全位于目标区间内部。淡化样本数按

```text
fade_samples = floor(fade_ms * 48000 / 1000 + 0.5)
```

计算。正的 `fade_ms` 至少需要 2 个样本，且 `2 * fade_samples <= 目标区间长度`。

淡化使用升余弦窗 `0.5 - 0.5*cos(pi*t)`：

- 左淡化从 `[start_sample, start_sample+fade_samples)` 内的原信号过渡到新信号。
- 右淡化从 `[end_sample-fade_samples, end_sample)` 内的新信号过渡回原信号。
- 目标区间首样本保留原值 `original[start_sample]`，末样本保留原值 `original[end_sample-1]`，因而删除/替换边界与未修改区间逐样本连续。
- 曲线两端权重精确为 0/1，满足零阶连续。`fade_ms` 为 0 时不淡化，但仍只修改目标区间。

## 确定性

- 全部内部运算使用 `float64`，没有时钟、随机数、随机抖动或并行浮点归约。
- 合成正弦以整轨绝对采样坐标计算，不依赖操作遍历顺序。
- 写盘采用最近舍入（恰好半样本时远离 0）和饱和截断，不回绕。
- 16 位映射到 `[-32768, 32767]`，24 位映射到 `[-8388608, 8388607]`。
- WAV 使用规范的 44 字节 PCM 头；24 位数据块按 RIFF 规则补偶数字节。
- 报告浮点数固定输出 12 位小数；区间端点和采样数始终为 JSON 整数。
- 报告不写入时间戳、临时路径或运行环境信息。相同输入文件、相同清单和相同程序版本产生逐字节相同的 WAV 与报告。

## 处理报告

报告主要字段：

```json
{
  "schema_version": 1,
  "input": {
    "sample_rate": 48000,
    "channels": 1,
    "bit_depth": 24,
    "sample_count": 72000,
    "file_sha256": "..."
  },
  "output": {
    "format": "WAV_PCM",
    "sample_rate": 48000,
    "channels": 1,
    "sample_count": 72000,
    "quantization_bits": 24,
    "file_sha256": "...",
    "pcm_sha256": "..."
  },
  "processing": {
    "internal_precision": "float64",
    "fade_curve": "raised_cosine",
    "quantization_rounding": "nearest_half_away_from_zero",
    "quantization_saturation": true,
    "deterministic": true
  },
  "operations": [
    {
      "operation_index": 1,
      "operation_type": "splice",
      "target_interval": {"start_sample": 18000, "end_sample": 20400},
      "actual_interval": {"start_sample": 18000, "end_sample": 20400},
      "fade_length_samples": 144,
      "fade": {
        "requested_ms": 3.000000000000,
        "actual_ms": 3.000000000000,
        "length_samples": 144,
        "left_interval": {"start_sample": 18000, "end_sample": 18144},
        "right_interval": {"start_sample": 20256, "end_sample": 20400}
      },
      "levels": {
        "before": {
          "peak_linear": 0.000000000000,
          "rms_linear": 0.000000000000,
          "peak_dbfs": null,
          "rms_dbfs": null
        },
        "after": {
          "peak_linear": 0.000000000000,
          "rms_linear": 0.000000000000,
          "peak_dbfs": null,
          "rms_dbfs": null
        }
      },
      "splice": {
        "reference_file": "reference.wav",
        "reference_interval": {"start_sample": 1200, "end_sample": 3600},
        "reference_sample_count": 2400,
        "reference_sample_rate": 48000,
        "reference_channels": 1,
        "reference_bit_depth": 24,
        "reference_file_sha256": "..."
      }
    }
  ]
}
```

电平在量化前按目标区间计算；静音的 dBFS 写作 JSON `null`。`replace` 报告频率、相位、幅度，`gain` 报告线性增益。

## 错误与原子写盘

业务错误退出码为 `2`，写盘等执行错误为 `3`。错误只写 stderr，格式为单行 JSON：

```json
{"ok":false,"exit_code":2,"error_count":2,"errors":[{"category":"negative_gain","operation_index":0,"message":"gain must be non-negative","field":"gain","details":{"gain":-1.000000000000}},{"category":"overlapping_intervals","operation_index":1,"message":"target interval overlaps another operation","related_operation_indices":[0]}]}
```

程序会先完成整份清单的解析和校验，再一次性报告所有能确定的问题。稳定错误类别包括：

- 结构：`manifest_read_error`、`invalid_json`、`invalid_manifest`、`invalid_operation`、`missing_field`、`invalid_field`、`operation_index_mismatch`
- 语义：`unknown_operation_type`、`invalid_interval`、`interval_out_of_bounds`、`overlapping_intervals`、`negative_gain`、`invalid_fade_length`
- 素材：`missing_reference`、`reference_format_mismatch`、`reference_out_of_bounds`、`reference_length_mismatch`
- 量化与输入：`invalid_quantization`、`manifest_sample_rate_mismatch`、`input_missing`、`input_format_mismatch`
- 输出：`output_path_conflict`、`output_write_failed`、`processing_failed`

校验通过后才开始写盘。WAV 与报告先分别完整写入目标同目录的暂存文件并 `fsync`；两个文件都成功后，才用 `os.replace` 提交。提交过程中任一替换失败，会删除新目标、恢复已存在的旧目标并清理暂存/备份文件。业务错误不会创建、截断或替换任何目标文件。

## 自检与测试

```powershell
python repair.py selftest
python -m unittest discover -s tests -v
```

`selftest` 使用程序内置生成的 48 kHz 对白、爆音、静音缝隙、参考片段和示例清单，检查：

1. `track_length_preserved`：输入/输出整轨采样数一致，报告契约正确。
2. `bitwise_reproducible`：连续两次运行得到完全相同且 SHA-256 相同的 WAV 与报告。
3. `error_paths_produce_no_files`：非法清单、缺失参考、格式错误参考均不产出目标文件；已有目标在失败时保持不变。

## 实际验证结果

本仓库交付时的本地验证环境为 Linux、Python 3.14.4、NumPy 2.5.4；代码仅使用 Python 3.11 可用的语言/标准库特性。验证结果：

```text
python -m unittest discover -s tests -v
Ran 12 tests ... OK

python repair.py selftest
track_length_preserved: passed
bitwise_reproducible: passed
error_paths_produce_no_files: passed

内置样本整轨: 72000 -> 72000 样本
输出 WAV SHA-256: 0c405818bbc8f78fe4d536d15e0557868404dc8a0d1e9b332d89327890a0efaf
处理报告 SHA-256: 725ee324d6488a2e624519459acebe30c0e4154306163341c635aba411753e93
```

另用两个不同输出文件名重复处理同一输入，WAV 与报告均逐字节相同。组合非法清单返回 6 条错误，覆盖 `negative_gain`、`overlapping_intervals`、`unknown_operation_type`、`interval_out_of_bounds`、`invalid_fade_length`，目标 WAV/报告均不存在，也没有 `.stage-`/`.backup-` 残留文件。

## 文件

- `repair.py`：命令行入口。
- `dialogue_repair/manifest.py`：清单解析、别名与组合错误收集。
- `dialogue_repair/processor.py`：修复、淡化、电平统计和报告。
- `dialogue_repair/wavio.py`：PCM WAV 读取、量化与确定性写出。
- `dialogue_repair/atomic.py`：同目录暂存、提交和回滚。
- `dialogue_repair/samples.py`：内置确定性素材。
- `dialogue_repair/selftest.py`：端到端自检。
- `tests/test_dialogue_repair.py`：回归测试。
- `requirements.txt`：唯一运行时第三方依赖 NumPy。

