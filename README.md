# cinemixer

确定性的游戏过场动画离线多轨混音工具。它读取 JSON 工程描述，使用 FFmpeg
解码各轨源文件，重采样并统一声道布局，按事件时间码对齐、逐轨压缩与增益，
混音为立体声 WAV，并生成采样级精确的事件索引文件，同时保证 BS.1770-4
响度合规。所有测试样本由固定随机种子合成，无需任何外部素材。

## 环境要求

- Python ≥ 3.10（验证环境为 3.14）
- FFmpeg ≥ 6.0，且 `ffmpeg`、`ffprobe` 在 `PATH` 中（验证环境为 8.0）
- Python 依赖：`numpy`、`soxr`、`pytest`

## 安装

```bash
uv venv .venv
uv pip install --python .venv/bin/python numpy pytest soxr
# 或激活后：source .venv/bin/activate && pip install -e .
```

## 使用

```bash
python -m cinemixer path/to/project.json
```

成功退出码为 `0`；业务错误以非零整数退出码结束，并在 stderr 输出一行
结构化 JSON（含 `level`/`code`/`message` 及轨道、事件上下文字段）。

## 工程描述

```json
{
  "project_id": "demo",
  "sample_rate": 48000,
  "channels": 2,
  "duration_ms": 2000,
  "target_lufs": -23.0,
  "output": { "path": "out/mix.wav", "format": "wav", "bit_depth": 16 },
  "compressor": {
    "threshold_db": -18.0, "ratio": 4.0, "knee_db": 6.0,
    "attack_ms": 5.0, "release_ms": 50.0, "lookahead_ms": 0.0
  },
  "tracks": [
    {
      "track_id": "voice",
      "source": "samples/voice.wav",
      "gain_db": -1.0,
      "muted": false,
      "events": [
        { "event_id": "v1", "start_ms": 0, "duration_ms": 800,
          "source_offset_ms": 0, "gain_db": 0.0 }
      ]
    }
  ]
}
```

## 业务错误码

| 错误码 | 退出码 | 含义 |
| --- | --- | --- |
| `E_PROJECT_SCHEMA` | 2 | 工程字段缺失、类型或取值范围非法、JSON 损坏 |
| `E_PROJECT_NOT_FOUND` | 3 | 工程文件不存在 |
| `E_TRACK_LOAD_FAIL` | 4（非致命） | 源文件缺失/损坏/解码失败，跳过该轨并静音 |
| `E_ALL_TRACKS_FAILED` | 5 | 所有轨道均加载失败，终止且不产出文件 |
| `E_STATE_CORRUPT` | 警告 | 状态文件损坏，回退全量重处理 |
| `E_FFMPEG_MISSING` | 7 | PATH 中找不到 ffmpeg/ffprobe |
| `E_OUTPUT_FAIL` | 8 | 输出 WAV 写入/读取失败 |

警告码：`W_EVENT_NEGATIVE_START`、`W_EVENT_PAST_TIMELINE`、
`W_SOURCE_PAD_SILENCE`、`W_STATE_CORRUPT`。

## 处理管线

1. **解析与校验**：`cinemixer.project` 严格校验必填字段、数值范围、ID 唯一性。
2. **解码**：`cinemixer.ffmpeg_io` 用 ffprobe 探测、ffmpeg 解码为 float32，
   并对 WAV 做 RIFF/data 块完整性校验以识别截断损坏。
3. **重采样/声道**：`cinemixer.resample` 优先 soxr HQ，内置 NumPy 加窗 sinc
   后备；单声道转立体声复制，立体声转单声道按 `1/sqrt(2)` 能量平均。
4. **时间轴**：`cinemixer.timeline` 用四舍五入做毫秒↔采样换算，负时间截断、
   越界截断、源不足补静音，允许毫秒级重叠并按采样累加。
5. **压缩与增益**：`cinemixer.compressor` 峰值检测软拐点压缩器，独立
   起音/释放系数，前瞻延迟上限 5 ms；再施加事件级与轨道级 dB 增益。
6. **混音**：逐轨对齐累加，峰值归一化到 -1 dBFS 防削波。
7. **响度**：`cinemixer.loudness` 实现 BS.1770-4 K 加权与双门限；偏差超过
   ±0.5 LU 时施加一次线性增益修正，纯标量缩放不改变索引采样偏移。
8. **索引**：输出与 WAV 同目录的 `*.index.json`，逐事件给出
   `track_id`/`event_id`/`start_sample`/`end_sample`/`length_samples`。
9. **状态恢复**：`cinemixer.state` 以工程校验和 + 源文件 SHA-256 校验缓存；
   中断后从最近成功轨继续，状态损坏则全量重处理。

## 测试

```bash
.venv/bin/python -m pytest tests/ -v
```

测试覆盖：schema 校验与错误码、重采样与声道转换、时间轴边界与重叠、
压缩器起音/释放/延迟、BS.1770 响度、端到端输出与索引同步、字节级确定性、
损坏/缺失/全失败、子进程中断恢复与状态损坏回退、CLI 退出码。
