# broadcast-audiorepair

确定性离线音频修复流水线：读取 16-bit PCM WAV 和 JSON 时间轴，按拼接点分段做
ITU-R BS.1770 响度归一化、线性交叉淡化、锚点保持的 WSOLA 时间伸缩，输出修复后
WAV、JSON 映射报告和批处理汇总。处理不依赖云端、账号或随机系统状态。

## 核心能力

- 支持 16-bit PCM、单声道/立体声、44.1 kHz/48 kHz；输出格式与输入一致。
- 每段测量 BS.1770 积分响度并归一化，整轨闭环校正到 `-23 ±0.5 LUFS`。
- 时间轴锚点定义源音节边界与目标边界；段内按锚点分区间执行 WSOLA。
- 拼接点使用时间轴指定时长的线性交叉淡化。
- 输出量化后重新解析、测量并把实测值写回报告。
- 峰值不超过 `-1 dBFS`，整体直流绝对值小于 `0.001`。
- 批处理 `--jobs N`；`jobs=1` 与多线程的每个 WAV/报告逐字节一致。
- 临时文件 + 完整性标记 + 原子替换，失败时不提交半成品。

## 环境

- Python 3.9+
- NumPy 1.21+
- FFmpeg 5.0+（仅用于独立验收和 `ebur128` 对照，不是运行时依赖）

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
ffmpeg -version
ffprobe -version
```

本机已安装并验证 `ffmpeg 8.0.1` / `ffprobe 8.0.1`。流水线本身只依赖 Python 和
NumPy。若暂时不安装包，也可在仓库根目录使用
`PYTHONPATH=. python3 -m audiorepair ...`。

## 快速开始

生成固定随机种子的本地样本：

```bash
python3 tools/generate_samples.py /tmp/audiorepair-samples
```

处理单个文件：

```bash
python3 -m audiorepair \
  /tmp/audiorepair-samples/dialogue_48k_mono.wav \
  /tmp/audiorepair-samples/dialogue_48k_mono.timeline.json \
  --output /tmp/out/mono.wav \
  --report /tmp/out/mono.report.json
```

并行处理多个独立文件：

```bash
python3 -m audiorepair \
  --batch /tmp/audiorepair-samples/batch.json \
  --jobs 4 \
  --output-dir /tmp/out-batch \
  --batch-report /tmp/out-batch/batch_report.json
```

## 时间轴 JSON

推荐格式如下。`source_boundaries_sec` 与 `expected_boundaries_sec` 都是
**片段局部时间**：前者相对原始片段起点，后者相对修复后片段起点。

```json
{
  "version": 1,
  "target_lufs": -23.0,
  "tolerance_lu": 0.5,
  "crossfade_ms": 20.0,
  "splice_points_sec": [2.8],
  "segments": [
    {
      "id": "seg-0",
      "stretch_ratio": 1.02,
      "stretch_range": [0.97, 1.05],
      "source_boundaries_sec": [0.55, 1.15, 1.85],
      "expected_boundaries_sec": [0.558, 1.176, 1.887]
    },
    {
      "id": "seg-1",
      "stretch_ratio": 0.985,
      "stretch_range": [0.96, 1.02],
      "source_boundaries_sec": [0.65, 1.35, 2.00],
      "expected_boundaries_sec": [0.641, 1.330, 1.970]
    }
  ]
}
```

规则：

- 有 `splice_points_sec` 且片段未写 `start_sec/end_sec` 时，片段范围由拼接点自动
  推导；片段数必须等于拼接点数加一。
- 也可为每段显式写 `start_sec/end_sec`，或使用 `start_sample/end_sample`。所有
  片段必须首尾相接并覆盖整条输入，否则返回 `TIMELINE_INVALID` 或
  `TIMELINE_OUT_OF_RANGE`。
- `stretch_ratio` 决定整体目标长度：
  `round((end_sample - start_sample) * stretch_ratio)`。
- 每个锚点区间的实际伸缩比、以及整体伸缩比，都必须落在 `stretch_range` 内；
  否则返回 `STRETCH_UNREACHABLE`。
- 也支持 `boundaries` 对象数组，每项包含
  `{"source_sec": ..., "expected_sec": ...}`。
- `crossfade_ms` 可统一指定；也可用 `crossfades_ms` 数组，或在拼接点对象中写
  `crossfade_ms`，或给片段写 `crossfade_out_ms`。
- 所有数值必须有限；片段不能重叠、留隙或越界。

## 批处理清单

```json
{
  "jobs": [
    {
      "id": "mono48",
      "input": "dialogue_48k_mono.wav",
      "timeline": "dialogue_48k_mono.timeline.json",
      "output": "batch_out/mono48.wav",
      "report": "batch_out/mono48.report.json"
    }
  ]
}
```

相对路径按批处理清单所在目录解析。使用 `--output-dir` 时，输出统一命名为
`<id>.wav` 和 `<id>.report.json`。汇总文件在任务结束后原子写入。

## 处理与报告

每个片段依次执行：

1. 按采样点截取并测量原始积分响度。
2. 按锚点区间执行 WSOLA，得到精确目标长度。
3. 根据伸缩后片段响度计算目标增益。
4. 线性交叉淡化并按顺序拼接。
5. 对整轨做响度闭环校正、平滑峰值保护、逐声道直流消除。
6. 量化到 16-bit PCM，重新解析输出并测量最终指标。

报告包含：

- 每段原始采样起止点、输出起止点、目标长度；
- 原始/伸缩后/最终输出区间响度、目标增益和总应用增益；
- 总体与各锚点区间伸缩系数、可达区间；
- 每个拼接点的交叉淡化采样区间和毫秒长度；
- 每个预期边界的目标/实际采样点和误差毫秒数；
- 输出 SHA-256、最终响度、峰值、直流和最大边界误差。

最终报告中的验证数值来自量化后的输出 WAV 回读，而不是处理中间浮点结果。

## 业务错误码

单文件 CLI 失败时向 stderr 输出 JSON，并使用以下退出码：

| 错误码 | 退出码 | 含义 |
|---|---:|---|
| `INPUT_MISSING` | 10 | WAV、时间轴或批处理清单不存在/不可读 |
| `TIMELINE_INVALID` | 11 | JSON 损坏、字段类型错误、片段留隙/重叠 |
| `TIMELINE_OUT_OF_RANGE` | 12 | 采样点越界、交叉淡化超过相邻片段 |
| `STRETCH_UNREACHABLE` | 13 | 所需伸缩比超出声明范围 |
| `AUDIO_UNSUPPORTED_FORMAT` | 14 | 非 16-bit PCM、非 44.1/48 kHz、非单/双声道 |
| `BOUNDARY_MISALIGNED` | 16 | 输出实测边界误差超过 20 ms |
| `LOUDNESS_UNREACHABLE` | 17 | 静音或峰值约束下无法达到响度目标 |
| `PROCESSING_FAILED` | 20 | 其他处理/提交失败 |

输入校验在写盘前完成。提交阶段先创建 `<output>.incomplete` 标记，再原子替换 WAV
和报告；只有两者都提交后才删除标记。异常时会清理已提交文件和临时文件，或保留
标记说明输出不完整。

## 测试与验收

```bash
python3 -m unittest discover -s tests -v
python3 tools/acceptance.py
```

验收脚本自动生成固定种子样本并检查：

- 输出格式保持；
- 整体响度、峰值、直流；
- 边界误差和报告回读一致性；
- 内置 BS.1770 与 FFmpeg `ebur128` 对照；
- `jobs=1/4` 的 WAV 与报告逐字节一致；
- 线程池回收；
- 四类主要业务错误码和失败无残留文件。

本次实测：单元测试 `7/7`，验收 `18/18`。详见
[VERIFICATION.md](VERIFICATION.md) 和
[docs/TECHNICAL.md](docs/TECHNICAL.md)。

## 已知边界

- 输入必须是普通 PCM-16 WAV；24-bit、float、WAVE_FORMAT_EXTENSIBLE 等会明确
  拒绝。
- 静音片段无法归一化到目标响度，会返回 `LOUDNESS_UNREACHABLE`。
- 边界实测采用局部能量谷/谱通量检测。固定样本包含清晰音节间隙；对真实素材，
  时间轴锚点仍会作为 WSOLA 的硬合成边界。
- 极高峰均比素材可能触发平滑峰值保护；若响度与峰值约束不可同时满足则拒绝提交。
