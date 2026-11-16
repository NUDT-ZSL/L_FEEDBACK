# 技术说明与验证记录

## 1. 模块职责

| 模块 | 职责 |
| --- | --- |
| `project.py` | JSON 工程解析、必填字段/范围/ID 唯一性校验、时间轴采样长度 |
| `ffmpeg_io.py` | ffprobe 探测、ffmpeg 解码为 float32、WAV RIFF/data 完整性校验 |
| `resample.py` | soxr HQ 重采样 + NumPy 加窗 sinc 后备；单/双声道互转 |
| `timeline.py` | 毫秒↔采样换算、事件放置、负时间/越界截断、源不足补静音、索引项 |
| `compressor.py` | 峰值检测软拐点压缩器、独立起音/释放、可选前瞻（≤5ms） |
| `loudness.py` | BS.1770-4 K 加权、400ms 块/100ms 步进、双门限积分响度 |
| `wav_io.py` | 16/24/32-bit PCM 输出，固定种子 TPDF 抖动保证字节确定性 |
| `state.py` | 工程+源文件 SHA-256 校验检查点、缓存、损坏回退 |
| `renderer.py` | 端到端编排：加载→压缩→增益→对齐→混音→归一化→响度→索引 |
| `pipeline.py` / `cli.py` | 顶层入口与错误码→退出码映射、stderr 结构化 JSON 日志 |

## 2. 关键设计决策

### 2.1 处理顺序

每轨对每个事件先取源片段送入**有状态压缩器**（包络在事件间自然延续），
再施加 `事件增益 × 轨道增益`，最后放置到时间轴缓冲。重叠事件按采样点
累加，因此压缩器看到的是连续素材而非分段跳变。

### 2.2 延迟约束

压缩器默认零前瞻（无延迟）。前瞻在工程 schema 中被限制在 `[0, 5] ms`，
实测 `compressor_latency_ms` 等于前瞻采样数换算值，最大不超过 5ms。

### 2.3 损坏识别

FFmpeg 对截断的 PCM WAV 会容错解码，且截断同时改写了 data 块大小，
ffprobe 报告的是截断后的“合法”时长。因此在探测前先解析 RIFF/data 块，
比对声明字节数与实际文件大小，差异即判定 `E_TRACK_LOAD_FAIL`。
解码后还会校验非有限值、零样本与帧长缺口。

### 2.4 响度合规

先做 -1 dBFS 峰值归一化防削波，再测 BS.1770-4 积分响度；与目标偏差
超过 ±0.5 LU 时施加**且仅施加一次**线性标量增益。标量缩放不改变任何
采样偏移，索引保持不变；若修正后逼近满幅，再按比例折回到安全天花板，
仍然是纯标量，不引入时间偏移。

### 2.5 中断恢复

检查点保存工程描述的规范化 SHA-256 与每个源文件 SHA-256，逐轨完成后
原子写入状态文件并缓存该轨渲染缓冲（`.npy`）。恢复时仅复用
`status=done` 且校验和匹配的轨道；失败轨会被重新尝试。状态 JSON 解析
失败、版本不符或缓存缺失均发出 `W_STATE_CORRUPT` 并全量重处理。

### 2.6 确定性

样本由固定种子 `SEED=20261116` 合成；16/24-bit 抖动使用每文件固定种子的
TPDF，32-bit 不抖动。测试验证同一工程两次渲染的 WAV 与索引逐字节一致，
中断恢复后的产出与干净运行也逐字节一致。

## 3. 错误码与退出码

| 代码 | 退出码 | 致命性 |
| --- | --- | --- |
| `E_PROJECT_SCHEMA` | 2 | 致命 |
| `E_PROJECT_NOT_FOUND` | 3 | 致命 |
| `E_TRACK_LOAD_FAIL` | 4 | 非致命，跳过该轨并静音 |
| `E_ALL_TRACKS_FAILED` | 5 | 致命，不产出文件 |
| `E_FFMPEG_MISSING` | 7 | 致命 |
| `E_OUTPUT_FAIL` | 8 | 致命 |

警告：`W_EVENT_NEGATIVE_START`、`W_EVENT_PAST_TIMELINE`、
`W_SOURCE_PAD_SILENCE`、`W_STATE_CORRUPT`。

## 4. 实际验证结果

验证环境：Python 3.14.4、NumPy 2.5.5、soxr 1.1.0、pytest 9.2.0、
FFmpeg/ffprobe 8.0.1（PATH 可用）。

- 单元/集成测试：`pytest tests/` → **54 passed**。
- 外部核验：`ffprobe` 确认输出 48kHz/立体声/16bit/2.000s，帧数 96000，
  与工程描述及索引 `length_samples` 严格一致。
- 响度：演示工程初值 -15.19 LUFS，自动一次线性修正后稳定在 -23.0 LUFS。
- 损坏/缺失：均报 `E_TRACK_LOAD_FAIL` 并仅静音该轨，其余轨正常；
  全损时报 `E_ALL_TRACKS_FAILED`（退出码 5）且无输出文件。
- 中断恢复：`--fault-after 2` 子进程退出码 130，状态文件记录 2 轨完成；
  二次运行命中缓存续跑，产出与干净运行**逐字节相同**，成功后清理检查点。
- 状态损坏：写入非法 JSON 后运行，发 `W_STATE_CORRUPT` 并全量重处理成功。
- CLI：schema 错误退出码 2、工程缺失退出码 3、成功 0，stderr 均为结构化
  JSON 行。
- K 加权频响实测与 BS.1770-4 参考表吻合（100Hz≈-1.2dB、2kHz≈+2.7dB、
  高频渐近 +4dB）。

## 5. 复现实验

```bash
uv venv .venv
uv pip install --python .venv/bin/python numpy pytest soxr
.venv/bin/python -m pytest tests/ -v
PYTHONPATH=src .venv/bin/python -m cinemixer <project.json>
```

固化演示位于 `artifacts/final_demo/`（含合成源、`out/mix.wav` 与
`mix.index.json`）。
