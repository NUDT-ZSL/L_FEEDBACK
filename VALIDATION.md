# 实际验证记录

验证时间：2026-11-17（Asia/Shanghai）

## 环境

| 项目 | 实际值 |
| --- | --- |
| 平台 | Linux / WSL Ubuntu 26.04 |
| Python | 3.14.4 |
| NumPy | 2.5.4 |
| FFmpeg | 8.0.1-3ubuntu2 |
| ffprobe | `/home/liang/.local/bin/ffprobe` |

任务目标环境为 Windows 10/11 + Python 3.11。当前执行容器没有 Windows 或 Python 3.11 运行时，因此本次实跑在 Linux + Python 3.14 上完成。实现仅使用 Python 3.11 可用语法和标准库，路径处理基于 `pathlib`/`os.path`，未使用 Unix 专用外部命令；但不能把本次结果表述为 Windows 实机验证。

## 自动化测试

执行：

```bash
python -m unittest discover -s tests -v
```

结果：

```text
Ran 11 tests in 3.366s
OK
```

覆盖项：

- WAV 16/24/32-bit PCM 与 float32 编解码、header/data/frame 一致性；
- 内置源合成器逐字节确定性，源格式覆盖 44.1/48 kHz、单/双声道；
- 配方缺失字段、未知字段、负偏移、越界偏移及事件/源路径上下文；
- 短源补零、长源裁剪；
- 增益/时移随机实际值记录和变体区分；
- 多等级总线选择与淡出余量截断警告；
- 响度归一化、真峰值限制、零削波；
- 只依赖 manifest 的重放校验；
- `manifest` 被修改后失败；
- 工作线程失败后最终目录不存在、暂存目录清空；
- CLI 失败时 stdout 为空、stderr 为统一业务错误 JSON、退出码为 2。

## 完整演示验证

执行：

```bash
python tools/validate_demo.py
```

最终 stdout：

```json
{
  "clipping_sample_count": 0,
  "concurrency_determinism": {
    "same_run_id": true,
    "same_variant_hashes": true,
    "same_variant_manifest_entries": true,
    "workers": [1, 4]
  },
  "failure_cleanup": {
    "error_code": "SFX-E2003",
    "event_index": 0,
    "output_directory_absent": true,
    "return_code": 2,
    "source_path": "sources/slash_44k_mono.wav",
    "stage_directories": [],
    "stdout_empty": true
  },
  "levels_validated": [1, 2, 3, 5],
  "maximum_true_peak_dbtp": -1.019999247088091,
  "report": "build/validation/report.json",
  "status": "ok"
}
```

详细机器报告位于 `build/validation/report.json`（构建产物，不提交）。

### 多等级批量结果

`examples/ember_cleave.json` 每级生成 6 个变体，`examples/frost_ring.json` 生成 4 个变体，共 22 个变体：

| 配方/等级 | 变体数 | 唯一输出哈希 | 最大样本峰值 dBFS | 最大真峰值 dBTP | 削波样本 | 淡出截断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ember level 1 | 6 | 6 | -1.025 | -1.020 | 0 | 0 |
| Ember level 2 | 6 | 6 | -1.036 | -1.020 | 0 | 0 |
| Ember level 3 | 6 | 6 | -1.062 | -1.020 | 0 | 6 |
| Frost level 5 | 4 | 4 | -1.222 | -1.220 | 0 | 0 |

所有批次均执行了 `skillsfx verify --manifest ...`，结果为 `verified_variants` 等于对应变体数。

Ember level 3 请求淡出 `600 ms`，最后事件结束后的实际余量约 `399.94 ms`。6 个变体均按余量截断，并在清单 `analysis.warnings` 中记录：

```json
{
  "code": "fade_truncated",
  "requested_fade_ms": 600.0,
  "applied_fade_ms": 399.9375,
  "tail_margin_ms": 399.9375
}
```

每个变体均记录了非零归一化增益和限制器触发区间。例如 Ember level 1 的变体 0：

```text
normalization_gain_db = 5.565
limiter_trigger_count = 1
limiter_trigger_intervals[0] = 0.017 s .. 2.852 s
final_true_peak_dbtp = -1.020
```

### 输出格式矩阵

`ffprobe` 逐文件确认 codec、采样率、声道和帧数：

| 请求格式 | ffprobe codec | 帧数 | 真峰值 dBTP | 削波样本 |
| --- | --- | ---: | ---: | ---: |
| 44.1 kHz / mono / 16-bit | `pcm_s16le` | 13230 | -1.0201 | 0 |
| 48 kHz / stereo / 24-bit | `pcm_s24le` | 14400 | -1.0200 | 0 |
| 44.1 kHz / stereo / 32-bit | `pcm_s32le` | 13230 | -1.0200 | 0 |
| 48 kHz / mono / float32 | `pcm_f32le` | 14400 | -1.0200 | 0 |

限制器内部预留 `0.02 dB` 量化余量，因此配置 ceiling 为 `-1.0 dBTP` 时，最终测得约 `-1.02 dBTP`，16-bit 量化后也没有越过配置上限。

### 并发确定性

同一配方、等级和种子分别用 1 与 4 个工作线程生成：

```text
same_run_id = true
same_variant_hashes = true
same_variant_manifest_entries = true
```

测试数据为 6 个变体，输出 WAV 全文件 SHA-256 全部一致，清单中的时间线、随机抽样实际值和报告也一致。

### 失败与清理

CLI 使用负事件偏移的临时配方：

```text
return_code = 2
stdout = <empty>
error_code = SFX-E2003
event_index = 0
source_path = sources/slash_44k_mono.wav
output_directory_absent = true
stage_directories = []
```

另有单元测试直接让第 2 个变体的工作线程抛错，确认整批失败、最终目录未发布且 `.stage-*` 暂存目录全部删除。

### 重放一致性

- 生成后删除原配方文件，再执行 `verify_manifest`，仍可按清单重放并逐字节验证全部变体。
- 单元测试翻转输出 WAV 中一个字节，`verify_manifest` 返回 `SFX-E5001`，上下文包含失败变体与哈希不匹配信息。
- 清单记录源 WAV SHA-256；源素材在重放前变化也会失败。

## 交叉核对说明

FFmpeg/ffprobe 只用于读取生成的 WAV 并核对 codec、采样率、声道、位深和帧数，未用于生成、编码或修复文件。即使移除 FFmpeg，`generate`、`verify` 和测试套件仍可用。
