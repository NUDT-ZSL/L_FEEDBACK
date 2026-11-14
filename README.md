# 小队语音通讯复盘混音器

面向战术射击游戏回放的多路语音时间线混音工具。仅依赖 Python 3.11+ 标准库，
所有语音、按键杂音与提示音均由内置确定性合成器生成，不读取任何外部音频文件、
不访问网络。

## 运行

```bash
# 生成确定性样本工程并导出单声道/立体声 WAV 与清单
python main.py --demo out

# 导出指定 JSON 工程
python main.py path/to/project.json

# 模拟在完成 2 个分片后收到可处理中断，再续渲
python main.py path/to/project.json --simulate-interrupt-after 2
python main.py path/to/project.json

# 运行全部标准库单元测试
python main.py --selftest
```

Windows 上使用官方 Python 3.11 的 `python` 命令即可，无需安装第三方包。

## 工程文件结构

- `sample_rate`：整数采样率，范围 8000–192000。
- `channel_layout`：`mono` 或 `stereo`。
- `tick_rate`：每秒 tick 数，默认 1000，即 1 tick = 1 毫秒。
- `channels[]`：频道 `id`、`gain_db`、`muted`。
- `speakers[]`：说话人 `id`、`seed`、`base_frequency`、`harmonic_ratio`、
  `pitch_semitones`、`source_duration_ms`、`bound_channel`、`pan`。
- `events[]`：按 `tick` 升序，类型为 `start`、`end`、`ptt`、`switch`、
  `block`、`unblock`，可携带 `channel` 与 `speaker` 引用。
- `export`：输出 WAV、清单、工作目录、`chunk_samples`、提示音时长等。

## 退出码

- `0`：导出成功（含因快照哈希不一致而拒绝续渲、从零重导成功）。
- `2`：工程或时间轴校验失败，不产生输出文件或临时残留。
- `3`：IO 错误。
- `10`：收到 Ctrl+C / SIGTERM，已完成当前分片并落盘快照，重跑即可续渲。
- `1`：其他运行时错误。

## 错误对象

校验失败输出结构化错误数组，字段含 `code`、`message`、`path` 与可选
`event_index`。错误码包括 `E_SCHEMA`、`E_TICK_ORDER`、`E_UNKNOWN_REF`、
`E_SAMPLE_RATE`、`E_EVENT`、`E_IO`、`E_RESUME`。

## 模块

- `voice_mixer/parser.py`：JSON 解析、结构校验、规范化 SHA-256。
- `voice_mixer/timeline.py`：事件状态机与采样级片段切取。
- `voice_mixer/synthesis.py`：确定性 PCM、抗混叠变调、PTT 与提示音。
- `voice_mixer/renderer.py`：分块混音、dB 增益、软限幅与统计。
- `voice_mixer/exporter.py`：分片、原子快照、续渲、WAV 封装与清单。
- `voice_mixer/test/`：标准库 `unittest` 测试套件。
