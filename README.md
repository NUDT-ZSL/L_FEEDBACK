# audio_sync — 本地确定性音频帧同步与混流服务

在不可靠传输之上，把多路 48 kHz / 单声道 / 16-bit PCM 的 RTP 风格音频包
重组为连续 20 ms 帧流，完成抖动缓冲、丢包补偿（PLC）、多路混音，并输出
WAV 与逐帧 JSON Lines 诊断日志。全部数据由本地代码用固定随机种子确定性
生成，不访问网络、不依赖真实音频设备，可在 Windows / Linux / macOS 重复运行。

## 环境要求

- 64 位 Python 3.9+（开发与验证使用 Python 3.14）。
- numpy >= 1.21（不需要 PyAudio / 声卡 / Docker）。
- 可写工作目录，用于生成包、WAV 与日志。

### 安装

Windows（官方 Python 启动器）：

```bat
py -3 -m pip install -r requirements.txt
```

若使用本仓库已建好的虚拟环境（本仓库用 `uv` 生成在 `.venv`），直接用
`.venv\Scripts\python.exe` 即可。Linux/macOS：

```bash
python3 -m pip install -r requirements.txt
# 或使用已存在的虚拟环境
.venv/bin/python -m pip install -r requirements.txt
```

## 包格式

每个自定义二进制包由 8 字节小端头部 + 负载组成：

| 偏移 | 长度 | 字段 | 说明 |
| --- | --- | --- | --- |
| 0 | 2 | sequence number | uint16，65536 回绕 |
| 2 | 4 | timestamp | uint32，48 kHz 采样时钟 |
| 6 | 2 | payload length | uint16，负载字节数 |
| 8 | N | payload | 48 kHz 单声道 16-bit 小端 PCM |

一个标准 20 ms 帧负载固定为 `960 样本 × 2 字节 = 1920 字节`。
全局帧索引由时间戳推导：`frame_index = timestamp // 960`。

## 命令行用法

提供三个子命令，均通过 `python -m audio_sync ...` 调用。

### 1. 生成确定性受损包集

```bash
python -m audio_sync simulate --output-dir artifacts/demo \
  --streams 3 --frames 250 --seed 20261114 \
  --loss-rate 0.12 --reorder-rate 0.06 \
  --jitter-distribution normal --burst-loss --inject-faults
```

产物：`stream_<id>.pkt`、`ground_truth.jsonl`（逐帧逐路真值来源）、
`simulation_meta.json`（到达时刻表与损伤参数）。

### 2. 运行帧同步 / PLC / 混流

```bash
python -m audio_sync run --input-dir artifacts/demo \
  --wav artifacts/demo/mixed.wav --log artifacts/demo/frames.jsonl \
  --buffer-depth 4 --mixer-mode equal_power
```

`--buffer-depth` 取值 2–10；`--mixer-mode` 为 `equal_power` 或 `linear`。
结束打印摘要（总帧数、PLC 帧数、平均缓冲水位、最大延迟、背压事件数）。

### 3. 真值核验

```bash
python -m audio_sync verify --input-dir artifacts/demo \
  --wav artifacts/demo/mixed.wav --log artifacts/demo/frames.jsonl
```

Windows 一键演示脚本：`scripts\run_demo.bat`。

## 模块结构

- `audio_sync/constants.py` — 采样率、帧长、协议头、事件与错误码常量。
- `audio_sync/protocol.py` — 二进制包解析、长度/连续性校验、业务错误码。
- `audio_sync/signals.py` — 确定性信号生成与 int16/float PCM 转换。
- `audio_sync/jitter_buffer.py` — 每路自适应抖动缓冲、溢出/背压、256KB 内存预算。
- `audio_sync/plc.py` — 基音周期重复 + 二阶线性预测，>3 帧静音与恢复点。
- `audio_sync/mixer.py` — 2–4 路等功率/线性混音、活动检测、防削波。
- `audio_sync/simulator.py` — 固定种子的丢包/乱序/抖动/突发/异常包损伤模拟。
- `audio_sync/engine.py` — 单线程虚拟时钟事件循环，驱动到达与 20ms 播放。
- `audio_sync/wavio.py` — 标准库 WAV 读写。
- `audio_sync/verifier.py` — WAV 时长、JSONL 帧序、PLC 与真值对齐核验。
- `audio_sync/cli.py` — 命令行入口与退出码。
- `tests/` — 29 个 unittest 用例，覆盖全部模块与端到端流程。

## 错误码与事件

stderr 业务错误码（不中断主流程）：

- `E_TRUNCATED` 头部或负载被截断；`E_BAD_LENGTH` 负载长度非法（跳过该记录）。
- `W_SEQ_GAP` 序列号前向缺口；`I_SEQ_REORDER` 乱序回跳（信息提示）。

JSONL 帧事件：`buffer_underflow`、`buffer_overflow`、`backpressure`、
`late_frame_dropped`、`plc_recovery`、`silence_run`。

退出码：成功 `0`；参数错误 `2`；配置/IO 错误 `3`（信息写 stderr）。

## 验证结果

固定种子端到端运行（3 路 × 250 帧，丢包 12%、乱序 6%、突发丢包、注入异常包）：

- 单元测试：`python -m unittest discover -s tests` → 全部通过。
- WAV：48kHz/16bit/单声道，时长 5.000s（250 × 20ms），与真值帧数一致。
- JSONL：250 行，`frame_index` 严格 `0..249` 单调，时间戳对齐。
- 核验：`verify` 返回 `ok=true`，PLC 帧、静音段、恢复点、背压均可追踪。
- 确定性：同种子两次生成的 `.pkt` / 真值 / meta 字节级一致。
- 资源：每路缓冲内存峰值远低于 256KB；背压时丢弃最旧非关键帧且不阻塞接收。

更多设计细节见 `docs/technical_design.md`。
