# loudnorm-batch

离线批量音频响度归一化工具：对清单中的每个音频文件执行 ITU-R
BS.1770-4 综合响度测量、EBU R128 风格归一化、真峰值限制、时间轴对齐，
并输出统一规格（48 kHz / 24-bit / 立体声 WAV）的音频与可复现 JSON 报告。

## 环境要求

- Python 3.10+，`pip install -r requirements.txt`（NumPy + soundfile）
- FFmpeg 6.x+（含 ffprobe），`bin` 目录加入 `PATH`
- 不使用 Docker；样本由本地确定性脚本生成，无需网络与真实录音

## 快速开始

```bash
# 1. 生成确定性多格式样本与 manifest.json
python3 tools/generate_samples.py

# 2. 批量处理
python3 -m loudnorm_batch --manifest manifest.json --out-dir out --report report.json

# 3. 运行验收测试
python3 tests/test_acceptance.py
```

## 清单格式

```json
{
  "files": [
    {"path": "samples/tone_48k_16bit_stereo.wav", "offset_ms": 0},
    {"path": "samples/voice_44k1_24bit_mono.wav", "offset_ms": 250}
  ]
}
```

`offset_ms` 为该文件在统一时间轴上的参考起始偏移（毫秒）。

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--target-lufs` | -23.0 | 目标综合响度（EBU R128 风格） |
| `--true-peak-ceiling` | -1.0 | 真峰值上限（dBTP） |
| `--gain-min` / `--gain-max` | -24 / +24 | 允许施加的增益范围（dB） |
| `--total-duration-ms` | 自动 | 统一时间轴总长；默认为 max(offset+时长) |
| `--jobs` | 4 | 并发处理数 |
| `--timeout` | 30 | 单文件处理超时（秒） |

## 处理流程

1. **解析与元信息**：ffprobe 提取采样率、位深、声道数/布局、时长、起始
   时间戳；ffprobe 失败 → 状态 `unreadable`，记录 stderr 摘录，批次继续。
2. **响度测量**（BS.1770-4）：原生采样率下解码为浮点 PCM，K 计权
   （高 shelf + RLB 高通，系数按采样率重算，RLB 分子与 ffmpeg ebur128
   实现保持一致以控制偏差），400 ms 窗口 / 75% 重叠，绝对门限 -70 LUFS、
   相对门限 -10 LU 两级门控；结果保留两位小数。
3. **真峰值**：FFT 频域零填充过采样至 192 kHz 域测量 dBTP。
4. **归一化**：增益 = 目标 - 测量响度，限制在 [-24, +24] dB；越界 →
   状态 `loudness_out_of_range`，跳过归一化但保留测量值与拒绝原因。
5. **限幅**：1 ms 控制率、5 ms 前瞻、100 ms 释放的联动立体声峰值限制器；
   限幅后重新测量真峰值，若仍超限则施加安全微调并计入限幅量，保证输出
   真峰值 ≤ 上限。
6. **声道映射**：单声道复制为双声道；多声道按 ITU-R BS.775 下混
   （C/环绕以 0.7071 混入，LFE 丢弃）。
7. **时间轴对齐**：按清单偏移补静音/截断至统一总时长；文件自身起始
   时间戳与清单偏移偏差 > 5 ms 时标记 `timestamp_mismatch` 并以清单为准。
8. **输出校验**：输出样本数必须等于 `round(total_ms * 48)`，否则
   `length_mismatch` 并拒绝写入；封装为 48 kHz / 24-bit / 立体声 WAV。
9. **调度**：线程池最多 4 并发，按清单顺序提交；单文件超时标记
   `timeout` 并终止其子进程，不影响其他文件；报告按清单原始顺序排列。

## 报告

`report.json` 包含：工具版本、参数快照、每个文件的 SHA-256 前 16 位、
原始元信息、测量响度/真峰值、施加增益、限幅量、对齐偏移、状态标记与
输出路径。所有浮点测量值定点舍入、stderr 摘录中的内存地址归一化，
同一输入与参数重复运行产生逐字节相同的报告（`generated_at` 字段除外）。

状态取值：`ok`、`unreadable`、`decode_error`、`loudness_out_of_range`、
`length_mismatch`、`timeout`；附加标记：`timestamp_mismatch`、
`partial_decode`（损坏帧但可部分解码）。

## 确定性样本集

`tools/generate_samples.py` 以固定种子生成：48k/16bit/立体声、
44.1k/24bit/单声道、96k/float32/5.1、32k/16bit 极静音（触发
`loudness_out_of_range`）、静音底+满幅脉冲（触发限幅器）、高响度削波
素材、带 2.5 s 容器偏移的 OGG（与清单一致）、带 1.2 s 偏移但清单为 0
的 OGG（触发 `timestamp_mismatch`）、随机垃圾文件（`unreadable`）、
中间帧损坏的 MP3（`partial_decode`）。

## 验证结果（本机实测）

- 综合响度与参考工具（ffmpeg `ebur128`）偏差：全部样本 ≤ 0.04 LU
  （要求 ≤ 0.1 LU）
- 真峰值与参考偏差 ≤ 0.05 dB；所有输出真峰值 ≤ -1.0 dBTP
- 输出规格 48 kHz / 24-bit / 立体声，样本数 = 统一时长 × 48000
- 异常标记（unreadable / partial_decode / timestamp_mismatch /
  loudness_out_of_range / timeout）均按预期触发
- 连续两次运行报告逐字节一致（除 `generated_at`）
- `tests/test_acceptance.py`：15/15 通过
