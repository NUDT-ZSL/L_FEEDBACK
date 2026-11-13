# EventMix

EventMix 是一个面向电竞赛事直播后期制作的命令行工具。它读取多路不同采样率、位深和录制起点的解说/环境声 WAV，根据赛事时间轴锚点完成时钟漂移校正和分数延迟对齐，再把各轨按声道布局封装为一条 `WAVE_FORMAT_EXTENSIBLE` 多声道 WAV，同时输出机器可读的播放时间轴 JSON、检查点和批处理报告。

实现只依赖 Python 3.11+ 和 NumPy，不依赖 FFmpeg、soundfile、容器或外部音频服务。

## 已实现能力

- 任意合法输入/目标采样率组合（如 44100→48000、48000→32000），使用 Kaiser 窗化 sinc 分数延迟插值；降采样时自动降低截止频率以抑制混叠。
- 16/24/32 位整数 PCM 输入，额外支持 32 位 IEEE float 输入；输出支持 16/24/32 位整数 PCM。
- 单声道扩展、显式多源混排和立体声→单声道加权缩混。
- 按锚点最小二乘拟合 `program = offset + drift × source`，输出 offset、drift factor、ppm 和每个锚点的毫秒/帧残差。
- WAV 中写入标准 `fmt `、`cue `、`LIST/adtl/labl`、`LIST/INFO`，并额外写入完整播放时间轴 `JSON` chunk。
- sidecar 播放时间轴与 WAV 内嵌时间轴使用同一份 canonical UTF-8 字节；cue、offset、drift 和告警不存在两套独立计算。
- 静音轨、内部静音段、样本覆盖不足、坏 WAV 头/读取失败均产生带轨道 ID 和节目时间范围的告警；`silence` 策略下局部零填充继续封装，其他轨道不受影响。
- 每任务完成后原子落盘检查点，保存时间轴及全部输入指纹、输出文件大小/SHA-256、工具与格式版本；恢复时重新计算指纹并校验现有产物。
- 批处理支持线程并发、全局 WAV 句柄上限和内存预算；输出与并发度无关，逐字节确定。
- 输出目录内重名任务在启动前直接判为致命冲突；跨进程同名写入使用文件锁和任务归属检查。
- CLI stdout 始终是可解析 JSON，退出码稳定：成功 `0`、可恢复降级 `1`、致命错误 `2`。

## 安装

Windows 10/11：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\eventmix.exe --version
```

Linux/macOS：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/eventmix --version
```

项目依赖声明为 `numpy>=1.26,<3`。本次实际验证环境是 Python 3.11.18 + NumPy 1.26.4。

## 快速开始

生成确定性的三轨样本、两份时间轴和一份两任务批处理清单：

```bash
python3 -m eventmix demo --output-dir .work/demo
# 或
python3 scripts/generate_fixtures.py --output-dir .work/demo
```

构建：

```bash
eventmix build .work/demo/batch.json \
  --output-dir .work/output \
  --jobs 2 \
  --max-open-wav 1 \
  --max-memory-mb 16 \
  --report .work/output/report.json
```

恢复同一批处理：

```bash
eventmix build .work/demo/batch.json --output-dir .work/output --jobs 2
```

若输入时间轴、WAV 内容或影响输出的参数发生变化，对应任务会重新构建；未变化且产物哈希仍匹配的任务直接跳过。`--fresh` 会忽略并重建检查点。

单时间轴也可直接构建：

```bash
eventmix build timeline.json --output-dir out --wav-name program.wav \
  --timeline-output-name program.timeline.json
```

## 时间轴格式

```json
{
  "schema_version": "eventmix-timeline/v1",
  "event": {
    "id": "final-2026",
    "title": "Grand Final",
    "venue": "Shanghai Arena",
    "date": "2026-11-14"
  },
  "program": {
    "sample_rate": 48000,
    "duration_seconds": 7.0,
    "bit_depth": 24
  },
  "policy": {
    "on_track_error": "silence",
    "silence_threshold_dbfs": -75.0,
    "min_silence_seconds": 0.25,
    "max_anchor_error_ms": 0.25
  },
  "channel_layout": [
    {"name": "main.L", "track": "main", "speaker": "FL", "sources": [0]},
    {"name": "main.R", "track": "main", "speaker": "FR", "sources": [1]},
    {"name": "sub.L", "track": "sub", "speaker": "BL", "sources": [0]},
    {"name": "sub.R", "track": "sub", "speaker": "BR", "sources": [0]},
    {"name": "ambient.L", "track": "ambient", "speaker": "SL", "sources": [0]},
    {"name": "ambient.R", "track": "ambient", "speaker": "SR", "sources": [1]}
  ],
  "tracks": [
    {
      "id": "main",
      "name": "Main commentary",
      "role": "main_commentary",
      "path": "audio/main.wav",
      "anchors": [
        {"source_seconds": 0.88, "program_seconds": 0.80},
        {"source_seconds": 3.48, "program_seconds": 3.40},
        {"source_seconds": 6.18, "program_seconds": 6.10}
      ]
    },
    {
      "id": "sub",
      "name": "Co-commentary",
      "role": "co_commentary",
      "path": "audio/sub.wav",
      "anchors": [
        {"source_seconds": 0.84, "program_seconds": 0.80},
        {"source_seconds": 3.44, "program_seconds": 3.40},
        {"source_seconds": 6.14, "program_seconds": 6.10}
      ]
    }
  ],
  "cues": [
    {"id": "draft", "name": "Draft phase", "program_seconds": 0.45},
    {"id": "fight", "name": "Objective fight", "program_seconds": 3.40}
  ]
}
```

关键规则：

- `path` 相对时间轴文件所在目录解析。
- `source_seconds` 是输入 WAV 自身时间轴上的时刻；兼容别名 `track_seconds`。
- 每轨至少一个锚点。一个锚点采用 `drift=1`；两个及以上锚点做最小二乘拟合。
- `channel_layout` 按最终 WAV 声道顺序排列。`speaker` 使用 WAVE speaker id（`FL/FR/FC/LFE/BL/BR/SL/SR` 等）。
- `sources` 可写源声道整数，或 `{"channel": 0, "gain": 0.5}`。省略时：单声道自动扩展；立体声按轨道目的声道序号映射；一个目的声道对应多源声道时自动等权缩混。
- `program.channel_mask` 可选；若显式提供，必须与 `channel_layout.speaker` 推导出的掩码一致。
- cue 按帧时间排序，`wav_cue_id` 从 1 开始；`program_seconds` 最终以 `frame / sample_rate` 为唯一时间语义。

## 批处理清单

```json
{
  "schema_version": "eventmix-batch/v1",
  "defaults": {
    "on_track_error": "silence",
    "bit_depth": 24
  },
  "jobs": [
    {
      "id": "event-a",
      "timeline": "timeline-a.json",
      "wav": "event-a.wav",
      "timeline_output": "event-a.timeline.json"
    },
    {
      "id": "event-b",
      "timeline": "timeline-b.json",
      "wav": "event-b.wav",
      "timeline_output": "event-b.timeline.json"
    }
  ]
}
```

`timeline` 相对清单目录解析；`wav` 和 `timeline_output` 相对 `--output-dir` 解析。任务也可用 `policy` 或 `engine` 对象覆盖 `on_track_error`、`bit_depth`、`target_sample_rate`、静音阈值等。

## 输出与时间语义

每个任务输出一条多声道整数 PCM WAV、一个 sidecar 播放时间轴 JSON，以及汇报中的任务摘要、告警、锚点参数和产物 SHA-256。

WAV 结构：

| chunk | 内容 |
| --- | --- |
| `fmt ` | 40 字节 `WAVE_FORMAT_EXTENSIBLE`，PCM sub-format GUID、valid bits、channel mask |
| `LIST` `INFO` | `INAM/IART/ICRD/ISFT/ICMT`，`ICMT` 含 timeline SHA-256 和 cue 数 |
| `cue ` | 每条 cue 的 WAV 数字 ID 与 data chunk 绝对样本帧位置 |
| `LIST` `adtl` | `labl` cue 名称 |
| `JSON` | 完整 canonical 播放时间轴 UTF-8 字节，与 sidecar 逐字节相同 |
| `data` | 按配置顺序交织的多声道 PCM |

播放时间轴核心字段：

- `program.total_frames`、`program.duration_seconds = total_frames / sample_rate`
- `tracks[].alignment.offset_seconds`
- `tracks[].alignment.drift_factor`、`drift_ppm`
- `tracks[].alignment.anchor_errors[]` 中每个锚点的预测值、毫秒误差和帧误差
- `cues[].frame`、`cues[].program_seconds`、`cues[].wav_cue_position`
- `integrity.digest`：对移除 `integrity` 后的 canonical JSON 计算 SHA-256

`cue.frame` 是 data chunk 内的绝对帧偏移，`program_seconds` 只由该帧号和目标采样率计算；WAV 的 `cue ` 与 `JSON` chunk 都使用同一个 cue 对象列表。

## 对齐与输出安全

输入/节目时间映射：

```text
program_seconds = offset_seconds + drift_factor × source_seconds
source_seconds  = (program_seconds - offset_seconds) / drift_factor
```

每个输出帧先映射到输入 WAV 的分数样本位置，再用窗化 sinc 重建。核函数按可用样本范围自动提高抽头数，并对每行核系数归一化。锚点集合上的拟合残差直接写入播放时间轴和最终报告，可通过 `max_abs_anchor_error_ms`、`rms_anchor_error_ms` 和逐锚点 `error_frames` 量化验证。

为避免插值过冲造成整数越界，每轨先写临时 float32 轨并统计峰值，再按 `1 - 0.5/2^(bits-1)` 的安全上限应用必要时增益。最终量化器仍保留整数钳位计数；若发生任何钳位会作为实现错误终止，而不是静默输出削波。

## 告警与退出码

| 退出码 | 状态 | 含义 |
| --- | --- | --- |
| `0` | `ok` | 全部任务成功且无可恢复告警 |
| `1` | `partial` | 至少一路静音/短轨/坏头被报告或局部降级，其余轨道继续产出 |
| `2` | `fatal` | 配置错误、输出冲突、`on_track_error=fail` 的轨道错误或不可继续的内部错误 |

主要告警：

- `track_silent`：整轨低于阈值，范围为 `[0, duration]`
- `track_silent_segment`：内部静音段，带开始/结束帧和秒
- `track_short_start` / `track_short_end`：输入未覆盖节目起止范围
- `wav_header_corrupt`：RIFF/fmt/data 等头信息损坏，策略允许时整轨零占位
- `wav_read_error`：通过头校验后发生读取失败，失败点到节目末尾零占位
- `anchor_fit_residual_high`：锚点线性拟合残差超过策略阈值

静音轨、短轨和坏 WAV 头的降级均不会改动其他轨的 offset、drift 或音频样本。

## 检查点与并发

默认检查点为 `<output-dir>/.eventmix-checkpoint.json`。每个任务完成后，在文件锁内重读、合并并原子替换检查点。记录包含：

- 任务状态与工具/播放格式/WAV 格式版本
- 时间轴文件、每条输入轨的路径、大小和 SHA-256
- 输出 WAV 与 sidecar 的路径、大小和 SHA-256
- 与产物同源的轨道对齐、cue 和告警摘要

恢复时不会直接信任旧参数：时间轴会重新解析和拟合，输入指纹会重算；只有指纹、版本、产物大小与哈希全部匹配才跳过。

并发控制由单个进程内共享的 `ResourceGovernor` 执行：

- `--max-open-wav` 限制全局同时打开的 WAV 句柄。
- `--max-memory-mb` 限制同时持有的源窗口、sinc 核和混合缓冲。
- 每个任务串行渲染自己的轨道，线程并发只影响任务调度，不改变浮点计算顺序。
- 同一输出目录的跨进程写入通过 `.eventmix.lock` 文件锁串行化；若现有 WAV/JSON 属于另一任务则拒绝覆盖。

## CLI 摘要

```text
eventmix build INPUT --output-dir DIR
  [--jobs N] [--resume|--fresh|--no-resume] [--checkpoint PATH] [--report PATH]
  [--max-open-wav N] [--max-memory-mb MiB]
  [--chunk-frames N] [--sinc-half-width N] [--sinc-max-half-width N]
  [--on-track-error silence|fail]
  [--target-rate HZ] [--bit-depth 16|24|32]
  [--silence-threshold-dbfs DB] [--min-silence-seconds S]
  [--wav-name NAME] [--timeline-output-name NAME] [--no-progress]

eventmix demo --output-dir DIR
```

进度写 stderr，stdout 只写 JSON 报告；`--report` 另写一份格式化 JSON。

## 实际验证

核心测试：

```text
$ PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_eventmix
...
Ran 9 tests in 27.425s
OK
```

覆盖内容：

1. 44100/16/2ch、48000/24/1ch、48000/32/2ch 三路输入，目标 48000/24/6ch。
2. 每路不同起点与 ppm 漂移；锚点残差和输出 click onset 均验证。
3. 任意重采样到 32000/16bit、cue 帧重算、显式立体声→单声道加权缩混。
4. extensible 头、channel mask、cue/labl、INFO、内嵌 JSON 与 sidecar 逐字节一致。
5. 未完成任务恢复、输入时间轴变更重跑、陈旧 sidecar 哈希不匹配重跑。
6. 静音、短轨、坏 WAV 头局部降级；健康轨道样本逐位不变。
7. `jobs=1/4`、全局单 WAV 句柄、16 MiB 全局内存预算下输出逐字节一致；双进程共享输出目录成功。
8. CLI 的 `0/1/2` 退出码和 stdout JSON。
9. 16/24/32 位 extensible 写入以及 24 位奇数 data chunk 的 RIFF 长度/填充边界。

独立验收脚本：

```bash
PYTHONPATH=src .venv/bin/python scripts/run_acceptance.py
```

本次运行结果（同时保存在 `artifacts/acceptance-report.json`）：

```json
{
  "status": "ok",
  "serial_exit_code": 0,
  "parallel_exit_code": 0,
  "byte_identical_jobs_1_vs_2": true,
  "resume_skipped": 2,
  "wav": {
    "format_tag": 65534,
    "channels": 6,
    "sample_rate": 48000,
    "bit_depth": 24,
    "channel_mask": 1587,
    "frames": 336000,
    "peak": 0.5025148391723633
  },
  "max_anchor_fit_error_ms": 8.881784197001252e-13,
  "click_onset_delta_frames": {
    "main": [1, 2, 2],
    "sub": [1, 2, 1],
    "ambient": [2, 2, 2]
  },
  "embedded_timeline_equals_sidecar": true,
  "cue_count": 3
}
```

click 检测的是输入合成 burst 的起始阈值点；1–2 帧偏差包含整数帧取整和 burst 起始相位，对应的锚点模型残差低于 `1e-9 ms`。

## 已知边界

- 标准 RIFF 的 4 GiB 限制；超长节目需拆分为多个 program segment，当前不生成 RF64。
- 输入支持 16/24/32 位整数 PCM 和 32 位 float；不支持压缩 WAV、ADPCM、RF64、BWF `bext` 自动解析。
- 对齐假设每路在锚点范围内为线性时钟漂移；锚点本身需要由赛事时间轴提供，不做内容识别或自动找锚点。
- Windows 路径、原子替换和 `msvcrt` 文件锁已按跨平台实现；本次自动化运行在 Linux 开发环境完成，目标 Windows 环境使用相同代码路径。
