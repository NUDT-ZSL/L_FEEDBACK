# 离线空间化音效混音工具

本项目从零实现一个 Windows 可用的离线命令行工具：读取源 WAV 目录和 JSON
事件时间轴，为每个事件计算距离衰减与等功率左右声像，按样本级时间轴拟合、
调度和混音，最终输出 48 kHz、16 位 PCM、立体声 WAV，以及毫秒精度 JSON
报告。整个过程不访问网络、不依赖系统时间、不使用随机数。

## 环境安装

目标环境为 64 位 Windows 和 Python 3.9 或更高版本：

```powershell
py -3.9 -m pip install -r requirements.txt
```

或直接安装任务指定的版本：

```powershell
py -3.9 -m pip install numpy==1.26.4 soundfile==0.12.1
```

`requirements.txt` 已固定为 `numpy==1.26.4` 和 `soundfile==0.12.1`。

## 命令行用法

四个位置参数依次为：源 WAV 目录、事件 JSON、输出混音 WAV、输出报告 JSON。

```powershell
py -3.9 spatial_mixer.py D:\cutscene\sounds D:\cutscene\events.json D:\cutscene\mix.wav D:\cutscene\report.json
```

也可使用命名参数：

```powershell
py -3.9 spatial_mixer.py --source-dir D:\cutscene\sounds --events-file D:\cutscene\events.json --output-mix D:\cutscene\mix.wav --report-file D:\cutscene\report.json
```

也可通过模块入口运行：

```powershell
py -3.9 -m spatialmix D:\cutscene\sounds D:\cutscene\events.json D:\cutscene\mix.wav D:\cutscene\report.json
```

输出文件所在目录必须已经存在。成功时不输出内容并返回退出码 0；输入、处理
或写出失败时返回退出码 1，且 stderr 使用统一格式：

```text
ERROR: <字段名或文件名>: <明确原因>
```

## JSON 事件格式

推荐使用以下规范字段。坐标和距离使用同一任意单位，`+x` 指向听者右侧，
`+y` 指向听者正前方。

```json
{
  "listener": { "x": 0.0, "y": 0.0 },
  "min_gain": 0.05,
  "events": [
    {
      "filename": "wind.wav",
      "position": { "x": -4.0, "y": 2.0 },
      "trigger_time": 1.25,
      "duration": 2.0,
      "priority": 10,
      "conflict_policy": "mix",
      "fit_strategy": "stretch"
    },
    {
      "filename": "rain.wav",
      "position": [3.0, 1.0],
      "trigger_time": 2.0,
      "duration": 4.5,
      "priority": 4,
      "conflict_policy": "delay"
    }
  ]
}
```

字段说明：

- `listener`：必填，二维听者位置，支持 `{"x": ..., "y": ...}` 或 `[x, y]`。
- `min_gain`：可选，距离衰减最小增益，范围 `[0, 1]`，默认 `0.05`。
- `events`：必填，非空事件数组；严格按数组顺序处理。
- `filename`：必填，源目录内的相对 WAV 文件名。
- `position`：必填，二维声源位置，支持对象或 `[x, y]`。
- `trigger_time`：必填，非负触发时间，单位秒。
- `duration`：必填，事件持续时间，单位秒，必须大于 0。
- `priority`：必填，非负优先级，数值越大优先级越高。
- `conflict_policy`：必填，`mix`、`drop` 或 `delay`。
- `fit_strategy`：可选，`stretch`（默认）或 `crop`。

兼容的常见同义字段：

- `listener_position` 等价于 `listener`
- `file`、`audio_file` 等价于 `filename`
- `source`、`source_position` 等价于 `position`
- `start_time` 等价于 `trigger_time`
- `event_duration` 等价于 `duration`
- `conflict_strategy` 等价于 `conflict_policy`
- `audio_fit` 等价于 `fit_strategy`

顶层也可提供 `conflict_policy`/`conflict_strategy` 作为缺省策略，命令行
`--global-conflict-policy` 也可提供兜底策略；事件自身字段优先。

## 处理规则

### 空间化

听者到声源的二维距离为 `d = hypot(dx, dy)`。距离增益采用带上下限的反比模型：

```text
distance_gain = clamp(1.0 / max(d, 1.0), min_gain, 1.0)
```

距离为 0 时增益为 1，不产生除零或无穷大。方位角为
`atan2(dx, dy)`；用 `pan = sin(azimuth)` 映射到 `[-1, 1]`，再使用
`theta = (pan + 1) * pi / 4`、`left = cos(theta)` 和
`right = sin(theta)` 生成等功率声像。正前方和正后方均为左右对称的中心声像，
因此正后方不会出现左右跳变。最终声道增益为距离增益乘以对应声像增益。
多声道源会先取各声道平均值转成单声道，再参与空间化。

### 采样率与事件时长

所有源文件均以 float64 读取，并使用确定性线性插值统一到 48000 Hz。事件目标
样本数由 `round(duration * 48000)` 得到，最少 1 个样本。

- `stretch`：源长度与事件时长不同时，用线性插值伸缩到精确目标长度。
- `crop`：源较长时截取开头；源较短时用静音补齐。

事件不会超出 `[trigger_time, trigger_time + duration)` 半开区间。

### 冲突策略

只有两个非 `mix` 事件的时间区间相交时才形成阻塞冲突。区间按半开区间判断，
因此一个事件结束样本恰好等于另一事件开始样本时不算重叠。

- `mix`：允许与已接受事件重叠，直接进入混音。
- `drop`：当前事件与阻塞事件重叠时，若任一阻塞事件优先级大于等于当前
  事件，则丢弃当前事件；若当前事件优先级高于全部阻塞事件，则丢弃这些阻塞
  事件并保留当前事件。相同优先级时保留先处理的事件。
- `delay`：将当前事件顺延到所有相关阻塞事件结束之后；若顺延后遇到新的阻塞
  事件，则继续顺延，直到无冲突。

所有事件始终按 JSON 顺序调度。被延迟或丢弃的事件都会在报告中说明原因。

### 混音与峰值

事件片段先应用空间增益，再按输入顺序叠加到 float64 立体声缓冲区。工具先
计算未归一化峰值；若超过 0.98，则求统一总增益 `0.98 / peak` 并应用到每个
事件片段后重新混音。最终量化为 16 位 PCM 前还将数值限制在安全整数范围内，
保证 WAV 峰值不超过 0.98 满量程，且不会发生 int16 溢出。

## 报告格式

报告包含输入顺序排列的所有事件。`start_time_ms`、`end_time_ms` 及请求时间
均由最终样本位置换算并四舍五入到毫秒（半向上取整）；被丢弃事件的
`start_time_ms`、`end_time_ms` 为 `null`。

```json
{
  "events": [
    {
      "adjusted": false,
      "audio_fit": "none",
      "conflict_policy": "mix",
      "conflict_with": [],
      "delay_ms": 0,
      "dropped": false,
      "end_sample": 72000,
      "end_time_ms": 1500,
      "filename": "wind.wav",
      "index": 0,
      "priority": 10.0,
      "reason": "no adjustment",
      "requested_start_time_ms": 0,
      "source_sample_rate": 48000,
      "spatial": {
        "azimuth_degrees": 0.0,
        "distance": 2.0,
        "distance_gain": 0.5,
        "left_gain": 0.353553390593,
        "right_gain": 0.353553390593
      },
      "start_sample": 0,
      "start_time_ms": 0,
      "status": "played"
    }
  ],
  "listener": { "x": 0.0, "y": 0.0 },
  "min_gain": 0.05,
  "output": {
    "channels": 2,
    "duration_ms": 1500,
    "filename": "mix.wav",
    "frames": 72000,
    "normalization_gain": 1.0,
    "normalization_required": false,
    "peak_after_normalization": 0.123,
    "peak_before_normalization": 0.123,
    "sample_rate": 48000
  },
  "summary": {
    "dropped_events": 0,
    "played_events": 1,
    "total_events": 1
  }
}
```

`adjusted` 表示触发时间被延迟、事件被丢弃或音频长度被伸缩/裁剪/补齐。
`status` 为 `played` 或 `dropped`。

## 确定性与错误处理

- 事件始终按 JSON 数组顺序读取、调度和叠加。
- 全程使用 float64 和固定运算顺序，不使用随机数。
- 时间先落为整数样本位置，报告再由样本位置换算为毫秒。
- JSON 固定键排序、缩进和换行，不写入时间戳、绝对输入路径或文件元数据。
- 相同输入、相同输出路径重复运行时，混音 WAV 和报告 JSON 字节级一致。
- JSON 字段缺失、字段类型/值域错误、未知策略、文件不存在、WAV 损坏、
  路径不可写等错误均返回退出码 1，并向 stderr 输出统一 `ERROR:` 信息。

## 自动化测试

测试使用固定种子的白噪声、正弦波和常量样本，自动生成临时输入，覆盖：

- 距离衰减、等功率声像、零距离、正后方；
- 非 48 kHz 重采样、线性时间伸缩、裁剪/补齐；
- 样本级时间轴对齐和毫秒报告；
- `mix`、`drop`、`delay` 冲突策略；
- 峰值归一化和 16 位 PCM 输出；
- 负时长、未知策略、缺失字段、坏 WAV、坏 JSON、错误路径和 CLI 参数；
- 重复运行混音与报告字节级一致。

运行：

```powershell
py -3.9 run_tests.py
```

成功时最后一行必须为：

```text
ALL TESTS PASSED
```

## 实际验证记录

当前工作容器为 Linux x86_64，实际解释器和库版本为 Python 3.14.4、
NumPy 2.5.4、soundfile 0.14.0；目标 Windows 环境所需的
NumPy 1.26.4/soundfile 0.12.1 API 兼容性已在实现中约束。全部 13 个 Python
文件还通过 Python 3.9 语法解析检查。

```text
$ python run_tests.py
...
Ran 25 tests in 2.463s

OK
ALL TESTS PASSED
```

测试套件实际运行 25 项，失败 0 项。由于当前容器没有 Windows 运行时，未能在
Windows 实体或 Wine 中执行；代码仅使用跨平台 `pathlib`、`argparse`、
NumPy 和 soundfile API，入口与路径处理均按 Windows 设计。
