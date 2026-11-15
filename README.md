# cutmaster

`cutmaster` 是一个本地、离线、确定性的剧情音频母版封装器。它读取 JSON
绝对时间轴，把不同采样率、声道数、起始时间、增益和角色类别的音轨统一到工程
采样率，按帧对齐混音，对重叠对白做优先级避让，测量并校准 EBU R128 响度与
真峰值，原子发布立体声 WAV，并生成可复核的对时 JSON 报告。

## 环境

- Python 3.11+
- NumPy、soundfile
- FFmpeg 6.x+，`ffmpeg` 必须在 `PATH` 中；也可用 `FFMPEG_BINARY` 指定完整路径

Windows PowerShell 示例：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
ffmpeg -version
```

## 快速运行

生成完全由本地确定性算法驱动的 4 轨演示样本：

```powershell
python scripts\generate_samples.py --output .\demo
python -m cutmaster --manifest .\demo\manifest.json
```

也可以使用安装后的入口 `cutmaster --manifest ...`。输出默认取清单中的
`output.master` 与 `output.report`，可用 `--output`、`--report` 覆盖。
状态目录默认是清单同级的 `.cutmaster-state`，可用 `--state-dir` 覆盖。

成功时 stdout 输出一行 JSON 摘要；失败时 stderr 输出一行结构化 JSON，
最终母版与报告不会被替换。例如：

```json
{"ok":false,"error":{"code":"E_MISSING_TRACK","message":"audio track does not exist","details":{"path":"..."}}}
```

## 时间轴清单

```json
{
  "schema_version": 1,
  "project": {
    "name": "episode-03",
    "version": "v12",
    "segment_id": "scene-04",
    "sample_rate": 48000,
    "duration_ms": 8000,
    "channels": 2
  },
  "loudness": {
    "target_lufs": -16.0,
    "true_peak_db": -1.0
  },
  "dialogue_ducking": {
    "duck_db": -12.0,
    "safe_headroom_db": -0.5
  },
  "tracks": [
    {
      "id": "dialogue_a",
      "path": "assets/dialogue_a.wav",
      "sample_rate": 44100,
      "channels": 1,
      "start_ms": 300.0,
      "gain_db": -2.0,
      "role": "dialogue",
      "priority": 20
    },
    {
      "id": "ambience",
      "path": "assets/ambience.wav",
      "sample_rate": 22050,
      "channels": 2,
      "start_ms": 0.0,
      "gain_db": -16.0,
      "role": "ambience",
      "priority": 0
    }
  ],
  "output": {
    "master": "out/master.wav",
    "report": "out/timing_report.json",
    "sample_format": "PCM_24"
  }
}
```

规则：

- 轨道 `path` 相对清单目录；命令行输出路径相对当前目录。
- `project.channels` 必须为 2。单声道轨复制到左右；多声道按下混规则变为立体声。
- `role` 为 `dialogue`、`dialogue_*` 或 `voice` 的轨道参与对白避让，其他角色不参与。
- 所有采样率必须是 8000–384000 的整数；文件实际采样率/声道数必须与声明一致。
- `start_ms` 使用十进制定点语义四舍五入到最近采样点（0.5 采样向上），误差不超过 1 个采样。
- 起始帧必须小于工程时长；轨道越过工程时长时报 `E_TRACK_OVERRUN`。
- 对白重叠区中，最高 `priority` 的轨道保持原增益，其余对白增加 `duck_db`；
  多个轨道并列最高优先级时无法确定语义胜者，报 `E_DUCK_UNSATISFIABLE`。
- `sample_format` 支持 `PCM_16`、`PCM_24`、`FLOAT`，默认 `PCM_24`。

## 处理与审计

- 重采样使用 64-tap Kaiser 窗 sinc、零相位有理比例映射；轨道内不插入时间漂移。
- 每轨先按“源文件内容 SHA-256 + 源元数据 + 工程采样率 + 下混版本”缓存为
  立体声 float32 中间态。缓存键不包含版本号或起始偏移，因此版本变化时只复用
  音频样本，所有帧偏移均从新清单重新计算。
- 混音按当前清单顺序流式累加，内存中只保留一条 float64 混音总线。
- 响度和真峰值由 FFmpeg `ebur128=peak=true` 测量，保留 4× true-peak 测量精度。
- 先按目标响度施加整体增益；若真峰值越限，使用 FFmpeg `alimiter` 前瞻限幅，
  再迭代校准响度，并以编码后 WAV 的最终测量作为验收值。
- 母版先写入输出目录内隐藏临时文件，完整关闭后再 `os.replace`；报告同样原子发布。
  失败、中断或强杀不会把临时文件暴露为最终母版。

检查点位于 `<state-dir>/projects/<project>-<segment>/checkpoint.json`。阶段包括
`validating`、`preparing_tracks`、`mixing`、`normalizing`、`publishing`、
`committed`。每完成一条轨道缓存都会更新检查点；进程被强杀后，下一次运行会识别
`in_progress` 状态、校验已提交缓存并继续，未完成混音则从有效轨道缓存全量重混。

对时报告包含：

- 每轨 `timeline_start_frame`、`first_valid_sample`、`first_nonzero_sample`、
  `last_valid_sample_exclusive`、重采样帧数及起始毫秒取整误差。
- 对白避让的绝对帧区间、对手轨道、避让增益，以及覆盖整条轨道的
  `gain_segments`（首尾与母版时间轴严格对应）。
- 每轨峰值、RMS、post-gain 平方和、`loudness_contribution` 能量占比及
  float32 贡献 SHA-256。
- 混音前后响度/真峰值测量、整体增益、限幅阈值、最终编码母版测量值和输出 SHA-256。

## 错误协议

所有业务错误写入 stderr 的单个 JSON 对象：`error.code`、`error.message`、
`error.details`。常见错误码：

| 错误码 | 退出码 | 含义 |
| --- | ---: | --- |
| `E_MANIFEST_MISSING` | 3 | 清单文件不存在 |
| `E_BAD_MANIFEST` | 2 | 清单 JSON 或字段非法 |
| `E_BAD_RATE` | 2 | 采样率非法 |
| `E_BAD_CHANNELS` | 2 | 声道声明非法 |
| `E_MISSING_TRACK` / `E_UNREADABLE_TRACK` | 3 | 输入缺失或不可解码 |
| `E_METADATA_MISMATCH` | 3 | 声明采样率/声道与实际文件不符 |
| `E_START_OUT_OF_RANGE` / `E_TRACK_OVERRUN` | 4 | 起始或结束越过工程时间轴 |
| `E_DUCK_UNSATISFIABLE` | 5 | 并列最高优先级，无法唯一选择避让胜者 |
| `E_SILENT_MIX` / `E_LOUDNESS_UNREACHABLE` / `E_TRUE_PEAK_UNREACHABLE` | 6 | 响度或真峰值无法达标 |
| `E_STATE_LOCKED` / `E_STATE_CORRUPT` / `E_IO` | 7 | 状态、锁或文件系统错误 |
| `E_INTERNAL` | 8 | 未预期内部错误 |

## 测试

```powershell
python -m unittest discover -v -s tests -t .
```

测试覆盖重采样脉冲对齐、重叠对白避让、非对白不受影响、高波峰真峰值限幅、
版本变化增量与干净全量逐样本/逐字节一致、强杀检查点恢复、缺失文件、非法采样率、
越界起播、并列优先级冲突和元数据不一致。

8 条 48 kHz 立体声 5 分钟轨道的本地性能基准：

```powershell
python scripts\benchmark.py .\perf-sample
```

本仓库的实际验证结果、命令和硬件见 [docs/VALIDATION.md](docs/VALIDATION.md)。
