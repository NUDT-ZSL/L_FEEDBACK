# 离线空间音频渲染器

这是一个可崩溃续渲的确定性离线空间音频渲染器。场景、音频资产、空间化、共享混响、混音、快照和错误码处理全部在本地 CPU 完成，不依赖 GPU、音频设备或外部音频文件。

## 快速开始

环境要求：

- Python 3.9+
- NumPy >= 1.21
- soundfile >= 0.12

```bash
python -m pip install -r requirements.txt

python -m spatial_audio_renderer \
  --scene examples/demo_scene.json \
  --output-dir out \
  --snapshot-dir snapshots \
  --snapshot-interval 30
```

也可以使用：

```bash
python render_audio.py --scene examples/demo_scene.json \
  --output-dir out --snapshot-dir snapshots --snapshot-interval 30
```

若使用项目内虚拟环境，则在当前验证环境中命令为 `.venv/bin/python`。

输出：

- `out/mix.wav`：最终立体声混音。
- `out/dry_<emitter-id>.wav`：每个发射器的单声道干声分轨。
- `out/render_manifest.json`：采样率、位深、混音统计、重采样来源和混响/发射器参数。
- `snapshots/snapshot_<frame>.json`：版本化快照清单。
- `snapshots/snapshot_<frame>.state.npz`：受 SHA-256 校验和保护的渲染状态。

## 场景格式

完整示例见 `examples/demo_scene.json`。核心字段如下：

```json
{
  "sample_rate": 16000,
  "duration_seconds": 1.0,
  "bit_depth": 16,
  "frame_size": 256,
  "listener": {
    "position": [0, 0, 0],
    "forward": [0, 0, -1],
    "up": [0, 1, 0]
  },
  "master_gain_db": 0.0,
  "attenuation": {
    "model": "inverse_distance",
    "reference_distance": 1.0,
    "rolloff": 1.0,
    "min_distance": 0.1,
    "max_distance": 100.0,
    "max_gain": 1.0
  },
  "occlusion": {
    "open_cutoff_hz": 12000.0,
    "closed_cutoff_hz": 900.0
  },
  "reverb": {
    "rt60_seconds": 0.7,
    "pre_delay_seconds": 0.02,
    "wet": 0.35,
    "dry": 1.0,
    "seed": 11,
    "damping_hz": 5000.0
  },
  "assets": [
    {
      "id": "tone_440",
      "type": "sine",
      "frequency_hz": 440.0,
      "amplitude": 0.25,
      "phase": 0.0,
      "duration_seconds": 1.0
    },
    {
      "id": "noise_7",
      "type": "noise",
      "seed": 7,
      "amplitude": 0.08,
      "loop": false
    }
  ],
  "regions": [
    {
      "id": "reverb_zone",
      "shape": "sphere",
      "center": [0, 0, -3],
      "radius": 2.0,
      "reverb": {
        "rt60_seconds": 0.8,
        "pre_delay_seconds": 0.015,
        "wet": 0.4,
        "dry": 0.9,
        "seed": 21
      },
      "lowpass": {"cutoff_hz": 6000.0}
    }
  ],
  "emitters": [
    {
      "id": "tone_source",
      "asset": "tone_440",
      "position": [1.5, 0, -2],
      "gain_db": -3.0,
      "pan": "auto",
      "occlusion": {"amount": 0.25}
    }
  ]
}
```

解析器还接受常用别名，例如 `sampleRate`、`duration`、`audio_assets`、`audio_regions`、`asset_id`、`source_position`、`gainDb`、`min_cutoff_hz`、`closed_cutoff_hz` 等。场景中的必需对象缺失、类型错误、编号重复或数值越界均报告 `INVALID_SCENE`。

### 音频资产

`assets` 可用数组或 `{id: {...}}` 映射定义。只允许确定性本地生成：

- `sine`：`amplitude * sin(2*pi*frequency_hz*t + phase)`。
- `noise`：由 `seed` 驱动的 PCG64 白噪声，同一场景和种子每次一致。

`duration_seconds` 短于场景时默认补零；`loop=true` 时循环。资产内容取当前场景总采样数。发射器引用不存在的资产报告 `AUDIO_ASSET_MISSING`。外部文件资产会被拒绝，不会读取任何外部 WAV。

### 距离衰减

`attenuation.model` 明确支持三种可配置模型：

- `inverse_distance`：`max_gain * min(1, (reference_distance / max(distance, reference_distance))^rolloff)`，并应用 `min_distance`、`max_distance` 和 `max_gain`。
- `linear`：在 `min_distance` 到 `max_distance` 之间线性衰减到零，再乘 `max_gain`。
- `none`：距离增益固定为 `max_gain`。

超过 `max_distance` 时增益为零。发射器增益以 dB 给出，或使用线性 `gain` 字段。

### 声像与遮挡

- 显式 `pan` 取 `[-1, 1]`；`pan: "auto"` 或省略时，用听者 `forward/up` 构成右向量，将声源方向投影到右向量。
- 采用等功率声像：左右增益为 `cos/sin((pan+1)*pi/4)`。
- 遮挡程度 `amount` 取 `[0, 1]`，支持数字、布尔值、`occluded` 字段、`occlusion_amount`、`occlusion_polarity`，或 `{"amount": ..., "closed_cutoff_hz": ...}`。
- 截止频率随遮挡程度线性变化：`fc = open_cutoff_hz + amount * (closed_cutoff_hz - open_cutoff_hz)`，再经过因果一阶低通滤波器。

### 音频区域与混响

区域支持 `sphere`、`box`、`global`。发射器可用 `region_id` 显式路由；否则使用列表顺序中第一个包含该发射器位置的区域。

- 全局 `reverb` 是默认混响；区域 `reverb` 可覆盖 `rt60_seconds`、`pre_delay_seconds`、`wet/dry`、`seed` 和 `damping_hz`。
- 区域低通只作用于送往混响的湿声发送，不改变区域内或区域外的干声样本。
- 每个混响区域对应一条内部卷积通道；所有通道最终汇总到共享立体声湿声总线。
- 混响冲激响应由确定性 PCG64 噪声、RT60 指数衰减包络、预延迟和可选阻尼生成，归一化后使用 FFT 线性卷积。
- 每个区域内发射器按干声能量加权：振幅权重为 `sqrt(E_i / mean(E_lane))`，使湿声功率与干声能量成比例。
- 区域 `dry` 作用于最终混音中的区域干声，`wet` 作用于该区域的湿声发送。干声分轨文件保存声像前、区域混响前的空间化干声，便于独立检查。

最终混音为区域干声总线、共享湿声总线和 `master_gain_db` 的求和，并裁剪到 `[-1, 1]`。

## 帧、快照与恢复

这里的“帧”是长度为 `frame_size` 的音频块，默认 1024 个采样帧，可由场景或 `--frame-size` 设置。`--snapshot-interval N` 表示每处理 N 个音频块写一次快照。场景总帧数为 `ceil(total_samples / frame_size)`。

快照由两部分组成：

1. `snapshot_<render_frame>.json`：`version=1`、场景/渲染指纹、已处理帧与采样范围、每个发射器的距离增益/声像/遮挡截止/区域/资产哈希、混响预计算通道及冲激响应哈希、状态文件大小和 SHA-256、清单自身的 SHA-256。
2. `snapshot_<render_frame>.state.npz`：截至该帧的干声立体声总线和各混响发送通道，均为 float64。

快照以临时文件加原子替换写入，即使进程在写清单时退出，也不会把半写清单当作有效快照。恢复时：

1. 按 `render_frame` 从新到旧扫描。
2. 校验格式、版本、清单校验和、场景/渲染指纹、状态文件大小和 SHA-256。
3. 跳过损坏快照，选择最新有效快照；`--resume` 要求至少存在一个可恢复快照。
4. 恢复后从快照边界继续逐块渲染，最后重新执行确定性的 FFT 混响卷积和 WAV 写出。

默认启动时自动恢复；`--resume` 强制要求恢复。若最新快照版本不受支持且使用 `--resume`，立即拒绝并报告 `SNAP_VERSION_MISMATCH`。若快照目录存在候选但全部损坏或版本不兼容，自动模式也会拒绝续渲并报告 `SNAP_CORRUPTED` 或 `SNAP_VERSION_MISMATCH`，不会静默丢弃进度；只有目录中完全没有快照时才从头渲染。

测试时可使用隐藏测试钩子 `--simulate-crash-after-frames N`：处理到第 N 帧并写完快照后直接 `os._exit(137)`，用于验证真实异常退出后的恢复。正常运行时也可向进程发送 `SIGINT`/`SIGTERM`，处理器会在当前帧结束后写快照并报告 `RENDER_INTERRUPTED`。

## 错误码与退出码

所有业务错误以单行 JSON 写到标准错误，例如：

```json
{"code":"SNAP_VERSION_MISMATCH","details":{"supported_version":1},"exit_code":1,"message":"..."}
```

| 错误码 | 退出码 | 含义 |
|---|---:|---|
| `SNAP_VERSION_MISMATCH` | 1 | 快照版本不匹配，拒绝恢复 |
| `SNAP_CORRUPTED` | 1 | 快照清单/状态校验失败，且无法恢复 |
| `SNAP_NOT_FOUND` | 1 | `--resume` 时没有快照 |
| `RENDER_INTERRUPTED` | 1 | 收到中断并已尽力保存快照 |
| `INVALID_SCENE` | 2 | JSON、字段、类型、范围或引用结构非法 |
| `AUDIO_ASSET_MISSING` | 2 | 发射器引用不存在的资产生成定义 |
| `INVALID_OUTPUT_DIR` | 2 | 输出/快照目录不可写或覆盖目标不可写 |
| `INVALID_ARGUMENT` | 2 | 命令行参数非法 |
| `INTERNAL_ERROR` | 2 | 未预期的内部失败，不输出通用 traceback |

正常完成退出码为 0，并向标准输出写一行 JSON 完成摘要。

## 路径兼容

路径统一经过 `normalize_path`：

- Windows 原生环境使用 `pathlib` 处理盘符和反斜杠。
- POSIX/WSL 环境接受反斜杠作为分隔符；`C:\...` 在存在对应挂载时映射为 `/mnt/c/...`。
- 输出目录在渲染前通过实际创建临时文件校验写权限；已存在的计划覆盖文件也会单独检查可写性。

## 代码结构

- `scene.py`：场景 JSON 解析、校验、指纹。
- `assets.py`：确定性正弦/噪声生成。
- `dsp.py`：距离衰减、声像、低通、FFT 卷积。
- `spatial.py`：发射器空间化、区域路由、能量加权。
- `reverb.py`：确定性立体声混响冲激响应与共享总线。
- `snapshot.py`：版本化快照、校验和、扫描和恢复选择。
- `renderer.py`：逐帧渲染、最终混音、WAV 和清单写出。
- `cli.py`：命令行、JSON 错误、信号和退出码。

## 实际验证记录

验证环境：Ubuntu 26.04 / WSL2，Python 3.14.4，NumPy 2.5.4，soundfile 0.14.0。源码额外通过 Python 3.9 语法解析检查。

```text
$ .venv/bin/python -m unittest discover -s tests -v
Ran 17 tests in 3.473s
OK
```

覆盖内容与实测结果：

- 正常渲染退出码 0；`mix.wav` 为 2 声道，干声分轨为 1 声道，采样率、帧数和 24-bit PCM 头均通过 `soundfile` 与标准库 `wave` 验证。
- 设定 3200 个采样、80 采样块；第 14 帧模拟崩溃，进程退出码 137；`--resume` 恢复后退出码 0，恢复点为 `snapshot_000000000014.json`。
- 崩溃恢复后的 `mix.wav`、`dry_inside.wav`、`dry_outside.wav` 与无中断基线的最大逐样本误差均为 `0.0`，低于要求的 `1e-6`。
- 不加 `--resume` 重新启动也会自动从最新有效快照恢复。
- 修改快照版本得到 `SNAP_VERSION_MISMATCH` / 退出码 1；破坏状态文件得到 `SNAP_CORRUPTED` / 退出码 1；无快照执行 `--resume` 得到 `SNAP_NOT_FOUND` / 退出码 1。
- 缺失资产得到 `AUDIO_ASSET_MISSING` / 退出码 2；非法场景 JSON 得到 `INVALID_SCENE` / 退出码 2；输出路径是文件或无写权限目录得到 `INVALID_OUTPUT_DIR` / 退出码 2。
- 修改全局/区域 RT60 后，未进入区域的发射器 `dry_outside.wav` 最大差异为 0，而最终混音发生变化。
- Windows 风格反斜杠相对路径在当前 POSIX 环境中成功创建并写出 WAV。
