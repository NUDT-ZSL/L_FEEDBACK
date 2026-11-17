# spatialmix：离线 HRTF 空间音频混音工具

`spatialmix` 是一个纯本地、确定性的命令行离线混音器。它读取多声道 WAV
和声源坐标 JSON，按听者位置与朝向计算方位角/仰角，在本地生成的
24 方位角 × 12 仰角 HRTF 数据集中选择最近方向，完成左右耳卷积、
距离衰减、空气吸收、多普勒和分块混音，最后以指定格式原子写出立体声 WAV。

核心特性：

- 支持 8/16/24/32 位 PCM WAV，以及 32/64 位 IEEE float WAV。
- 本地确定性合成 HRTF：24 × 12 方向，每个方向含左右耳 IR。
- overlap-save 分块卷积，块间保留 `IR长度-1` 个重叠样本。
- 每个声源独立设置输入声道、增益、延迟、HRTF 增益及方向偏移。
- 自由场 `1/r` 或 JSON 自定义距离曲线；超过阈值时使用随距离变化的低通系数。
- 支持位置时间序列、相对径向速度和线性插值多普勒重采样。
- 每完成一个声源块即原子写状态；恢复前校验输入/配置哈希、块大小、输出格式等。
- 源贡献、最终混音均流式落盘，工作内存随块大小而非音频总时长增长。
- 先写唯一临时 WAV，成功后 `os.replace`；失败不破坏已有目标文件。

## 环境安装

要求 Python 3.9+。Windows 10/11、Linux/macOS 均可。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

本仓库的开发验证环境使用 Python 3.14.4、NumPy 2.5.5、SciPy 1.18.1、
SoundFile 0.14.0。

## 快速运行

生成确定性的 2 声道 WAV 和 3 声源配置：

```bash
python -m spatialmix.demo --output-dir artifacts/demo --duration 1.0
```

执行空间化：

```bash
python -m spatialmix \
  artifacts/demo/input_2ch.wav \
  artifacts/demo/sources_3.json \
  artifacts/demo/spatialized.wav \
  --output-format float32 \
  --block-size 256 \
  --listener-pos 0 0 0 \
  --listener-forward 0 0 1 \
  --listener-up 0 1 0 \
  --json
```

主命令参数：

| 参数 | 说明 |
|---|---|
| `input_wav` | 输入多声道 WAV |
| `source_config` | 声源坐标 JSON |
| `output_wav` | 输出立体声 WAV |
| `--output-format` | `pcm8`、`pcm16`、`pcm24`、`pcm32`、`float32`、`float64` |
| `--block-size` | 块样本数，默认 256 |
| `--listener-pos X Y Z` | 听者位置，默认 `0 0 0` |
| `--listener-forward X Y Z` | 听者正前方，默认 `0 0 1` |
| `--listener-up X Y Z` | 听者上方向参考，默认 `0 1 0` |
| `--filter-order before\|after` | 空气低通位于 HRTF 卷积前或后，默认 `before` |
| `--speed-of-sound` | 声速，默认 343 m/s |
| `--state PATH` | 状态文件，默认 `<output>.state.json` |
| `--fresh` | 丢弃已有状态和中间文件，重新开始 |
| `--stop-after-blocks N` | 提交 N 个块后优雅停止，用于验证断点续跑 |
| `--stop-after-mix-blocks N` | 提交 N 个最终混音块后停止，用于验证混音阶段恢复 |
| `--json` | 输出机器可读 JSON；业务错误也写入 stderr JSON |
| `--hrtf-info` | 打印本地 HRTF 清单、形状和指纹 |

坐标采用 `[right, up, forward] = [x, y, z]` 右手坐标；默认听者朝 `+Z`，
`+X` 为右，`+Y` 为上。方位角 `atan2(x, z)`，仰角由相对高度的仰角分量计算。

## 声源配置

最小示例：

```json
{
  "hrtf_version": "local-hrtf-v1",
  "distance_model": {
    "type": "free_field",
    "reference_distance": 1.0,
    "minimum_distance": 0.05
  },
  "air_absorption": {
    "enabled": true,
    "threshold": 3.0,
    "coefficient_per_meter": 0.16,
    "min_cutoff_hz": 650.0
  },
  "sources": [
    {
      "id": "left-near",
      "channel": 0,
      "position": [-1.2, 0.15, 2.0],
      "gain": 0.72,
      "delay_samples": 0,
      "spatialize": true,
      "hrtf_gain": 0.95,
      "azimuth_offset_deg": -3.0
    },
    {
      "id": "right-mid",
      "channels": [1],
      "position": [1.5, 0.45, 1.4],
      "gain": 0.55,
      "delay_seconds": 0.002,
      "distance_model": {
        "type": "custom",
        "points": [[0.0, 1.15], [1.0, 1.0], [4.0, 0.42], [8.0, 0.18]]
      }
    },
    {
      "id": "moving",
      "channel": 0,
      "position": [2.6, -0.25, -1.1],
      "positions": [
        [0.0, 2.6, -0.25, -1.1],
        [0.5, 1.2, 0.10, 0.2],
        [1.0, -0.2, 0.20, 1.3]
      ],
      "gain": 0.48,
      "elevation_offset_deg": 4.0
    }
  ]
}
```

字段说明：

- `sources` 必填且非空；每个声源必须有 `position` 或 `positions`。
- `channel` 是 0 基输入声道；缺省为 `source_index % input_channels`。
  `channels: [0,1]` 会先对指定输入声道取平均，再作为一个单声道源空间化。
- `gain`、`hrtf_gain` 非负；`delay_samples` 或 `delay_seconds` 二选一，
  最终按最近整数样本对齐。
- `spatialize: false` 使用左右耳均为单位脉冲的 IR，但距离、延迟和滤波仍生效。
- `azimuth_offset_deg`、`elevation_offset_deg` 可用于独立空间参数调整。
- `positions` 中每项为 `[time_seconds, x, y, z]`，时间必须严格递增。
  `position` 同时存在时用作 HRTF 的参考方向；运动轨迹驱动距离和速度。
- `distance_model.type` 支持 `free_field` 和 `custom`。自定义曲线点必须按距离
  严格递增，增益非负；曲线区间外使用端点增益。
- 全局 `distance_model`/`air_absorption` 可被声源同名字段覆盖。
- `hrtf_version` 必须为本地可用版本 `local-hrtf-v1`，否则返回
  `E_HRTF_VERSION`。

## 处理与恢复语义

1. 对每个声源，先按输入块读取所需样本，并按累计相位做线性插值重采样。
2. 应用声源增益、DistanceModel 增益；可选空气吸收低通。
3. 使用最近邻 HRTF 左右耳 IR 做 overlap-save 卷积。
4. 将非重叠输出段追加到临时浮点源贡献文件；卷积尾部单独追加。
5. 每提交一个源块，原子写入 `processed_samples`、`source_index`、
   `block_index`、累计相位、IIR 状态和重叠历史。
6. 所有声源完成后，按块读取各源贡献、求和并硬限幅到 `[-1, 1]`。
7. 从临时 float64 混音文件分块编码为目标 WAV 临时文件，最后整体替换目标。

状态恢复必须匹配：

- 输入 WAV 文件内容 SHA-256；
- 声源配置原始字节 SHA-256；
- 块大小、输出格式；
- 输出路径、输入帧数/声道数/采样率/子格式；
- 滤波顺序、声速、HRTF 版本和数据集指纹。

任一不一致返回 `E_STATE_MISMATCH`，不会继续写目标文件。用 `--fresh`
可主动丢弃旧状态。处理中断时保留状态和源贡献；业务错误会删除最终 WAV
临时文件，已有目标文件保持不变。

## 空气吸收与顺序等价

空气低通的截止频率由声源代表性距离 `r` 计算：

```text
cutoff = min_cutoff + (base_cutoff - min_cutoff) * exp(-k * (r - threshold))
alpha  = 1 - exp(-2*pi*cutoff/sample_rate)
```

单极点 IIR：

```text
y[n] = y[n-1] + alpha * (x[n] - y[n-1])
```

为使“滤波前/滤波后”严格满足线性时不变交换律，单个声源在其整段处理中使用
一个固定的 `alpha`：静态源取精确距离；运动源取轨迹距离采样均值。这样
`(x * h) * lpf == (x * lpf) * h`，并且分块变化不会改变等价性。若需要逐样本
时变空气吸收，应改为时变滤波模型，此时两个顺序不再数学等价。

## 位深兼容规则

输入浮点格式只能输出浮点格式；输出有效精度不得低于输入有效精度：

| 输入 | 可写输出 |
|---|---|
| PCM8 | `pcm8`、`pcm16`、`pcm24`、`pcm32`、`float32`、`float64` |
| PCM16 | `pcm16`、`pcm24`、`pcm32`、`float32`、`float64` |
| PCM24 | `pcm24`、`pcm32`、`float32`、`float64` |
| PCM32 | `pcm32`、`float64` |
| float32 | `float32`、`float64` |
| float64 | `float64` |

`float32` 的有效尾数精度按 24 位计，因此 PCM32 不能无损写入 `float32`。
不兼容时返回 `E_BIT_DEPTH`，退出码 41。

## 业务错误码

| 错误码 | 退出码 | 含义 |
|---|---:|---|
| `E_INPUT_FORMAT` | 10 | 非 WAV、WAV 子格式不支持或无法识别 |
| `E_INPUT_READ` | 11 | 输入块读取失败 |
| `E_CONFIG_READ` | 20 | 配置读取失败 |
| `E_CONFIG_JSON` | 21 | JSON 语法错误 |
| `E_CONFIG_MISSING_FIELD` | 22 | 缺少 `sources`、位置等必需字段 |
| `E_CONFIG_INVALID` | 23 | 数值、轨迹、声道或参数非法 |
| `E_HRTF_VERSION` | 30 | HRTF 数据集版本不匹配 |
| `E_OUTPUT_FORMAT` | 40 | 未知输出格式 |
| `E_BIT_DEPTH` | 41 | 输出精度与输入不兼容 |
| `E_MEMORY_LIMIT` | 50 | 估算音频工作集超过输入样本字节数的 2 倍 |
| `E_STATE_INVALID` | 60 | 状态文件损坏或中间文件不完整 |
| `E_STATE_MISMATCH` | 61 | 状态与当前参数不一致，拒绝恢复 |
| `E_IO` | 70 | 临时/目标文件 I/O 失败 |
| `E_INTERNAL` | 99 | 未预期内部异常 |
| `E_INTERRUPTED` | 130 | 收到中断或 `--stop-after-blocks`，状态已保留 |

默认向 stderr 输出 `[E_xxx] 类型: 详情`；`--json` 时输出包含
`error_code`、`exit_code`、`error_type`、`message`、`details` 的 JSON。

## 内存控制

输入从不整段解码驻留。每个源块的工作数组约为
`O((block_size + IR长度) * 声道/滤波状态)`；最终混音为
`O(source_count * block_size)`。工具在开始前估算块处理峰值音频工作集，
若超过“输入解码样本总字节数 × 2”则返回 `E_MEMORY_LIMIT`，而不是等到
操作系统 OOM。源贡献与最终混音使用临时文件，因此内存不随总时长增长。

## 验证

```bash
pytest -q
python tools/validate_demo.py
```

实际结果记录在 `artifacts/validation_report.md` 和
`artifacts/validation_results.json`。当前验收结果：

- 22 项自动化测试通过；
- 128/256/512 三种块大小输出逐样本一致；
- HRTF 卷积前/后低通在 128 与 512 混合块下逐样本一致；
- 中断恢复输出与不中断运行逐样本一致；
- 源处理阶段和最终混音阶段中断均可逐样本恢复，输入/配置哈希及块大小不一致会被拒绝；
- 块大小变化被 `E_STATE_MISMATCH` 拒绝；
- 2 秒多普勒测试累计相位漂移远小于 1 个样本；
- 位深、缺失字段、HRTF 版本、内存超限和格式错误均返回稳定错误码。
- 同一测试集在 Python 3.9.25 与 Python 3.14.4 下均通过。

## 设计边界

- 输出固定为左右耳 2 声道；输入可为任意声道数。
- 延迟按最近整数样本实现，不做分数延迟。
- 运动声源的 HRTF 方向使用 `position` 或轨迹首点的参考方向；位置时间序列
  用于距离包络、速度和多普勒。这样避免逐样本切换 IR 引入不连续。
- 多普勒重采样率为 `c / (c + v_radial)`，并限制在 `[1/8, 8]`，防止极端
  配置造成无界 I/O 或数值问题。
- 最终幅度限制为硬限幅；`--json` 结果中的 `peak_abs` 是限幅前峰值。
