# Cinemix 过场动画环境声混音工具

`cinemix` 从本地 WAV 和无网络依赖的 JSON 场景描述生成采样点对齐的立体声混音，并输出 ITU-R BS.1770 响度报告和镜头切换质量检查报告。

## 环境与安装

- Python 3.9+
- NumPy、SoundFile
- FFmpeg，且 `ffmpeg` 在 `PATH` 中

Windows PowerShell:

```powershell
py -3.9 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
where.exe ffmpeg
```

Linux/macOS:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
command -v ffmpeg
```

工具不访问网络，也不依赖任何预置音频数据。以下命令均从项目根目录执行。

## 快速运行

生成确定性测试样本：

```bash
python generate_samples.py samples
```

单场景：

```bash
python main.py samples/forest out_forest
```

批处理（`samples/` 的每个一级子目录是一个场景）：

```bash
python main.py samples out
```

无参数时打印用法并以 `0` 退出。`--help` 也以 `0` 退出。

## 输入结构

单场景目录结构示例：

```text
scene/
  scene.json
  ambience.wav
  hit.wav
  music.wav
```

批处理目录:

```text
batch/
  scene_a/
    scene.json
    *.wav
  scene_b/
    scene.json
    *.wav
```

`scene.json` 示例：

```json
{
  "scene_id": "forest",
  "duration": 8.0,
  "ambience": {
    "file": "ambience.wav",
    "loop_start": 0.35,
    "loop_end": 4.65,
    "gain": 0.75
  },
  "samples": {
    "bird": "bird.wav",
    "branch": "branch.wav"
  },
  "events": [
    {"time": 1.125, "sample": "bird", "gain": 0.65},
    {"time": 3.5, "sample": "branch", "gain": 0.55}
  ],
  "music_layers": [
    {
      "file": "music.wav",
      "start": 0.5,
      "loops": 2,
      "gain": 0.38
    }
  ],
  "dialogue_intervals": [
    {"start": 2.0, "end": 4.4, "duck_depth_db": 10.0}
  ],
  "camera_cuts": [0.75, 3.75, 5.5]
}
```

字段说明：

- 所有时间单位为秒，相对于场景起点；内部全部换算成整数输出采样点，采用四舍五入到最近采样点。
- `duration` 必须大于 0。
- `ambience.file` 为相对 JSON 的 WAV 路径；`loop_start`/`loop_end` 指定源文件中的循环区间，该区间从场景 0 点开始循环填满整个场景。
- `samples` 是离散音效名到相对文件路径的映射。也可在事件对象中直接提供 `file`。
- `events[].time` 是绝对场景时间；音效尾部超过场景时截断。
- `music_layers[].loops` 是整个文件的总播放次数；`start` 后循环放置，超过场景部分截断。
- `gain` 是线性幅度增益；可用 `gain_db` 替代。音效和音乐层必须提供其中一个，环境层缺省为 1.0。
- `dialogue_intervals[].duck_depth_db` 是对话期间对环境底噪和音乐层的衰减量，单位 dB。
- `camera_cuts[]` 是镜头切换时间点，可写数字或 `{"time": 1.25}`。
- 源 WAV 可为单声道或双声道，允许不同采样率和 PCM 位深；输出统一为 48 kHz。
- 路径必须相对，支持正斜杠和 Windows 反斜杠，可使用 `../shared/...`。

解析器也接受若干常见别名：`duration_seconds`、`sound_events`、`music`、`dialogues`、`cuts`、`loop_count`、`start_time`、`end_time`、`gain_db` 等。字段缺失、字段类型错误、非有限或越界时间、未知样本名、奇怪声道数及不可读音频均视为场景错误。

## 处理规则

- 输出固定为 48 kHz、双声道、16-bit PCM WAV。
- 所有输入先线性重采样到 48 kHz；事件、音乐层和循环区间均通过整数采样点定位，循环相位用取模计算，不累加浮点时间，因此长时间轴不会漂移。
- 环境底噪和音乐层先求和，再共同乘对话闪避包络；离散音效不闪避。
- 每个对话区间的进入和退出各使用最多 50 ms 等功率增益过渡。进入从区间起点开始淡入闪避，退出在区间终点前完成并恢复原增益；边界样本增益严格为 1.0，区间外保持原增益。重叠对话区间取更深闪避。
- 每个镜头切换点应用 20 ms 等功率交叉淡化。窗口严格以切点为中心，前后各占 10 ms；窗口首尾样本分别复用原窗口的首尾样本，因此边界采样值连续，总采样数和场景时长不变。混合报告会记录切换窗口、等功率误差、边界误差、峰值和削波数。
- 最终混音超过 16-bit 范围时硬限幅，并在报告中记录削波样本数；应通过输入增益控制留足余量。

## 输出

单场景输出目录:

```text
out_forest/
  mix.wav
  loudness_report.json
  switch_check.json
```

批处理为每个场景创建同名子目录，并额外输出:

```text
out/
  summary.json
  forest/
  city/
  orchestra/
```

`loudness_report.json` 中主要字段：

- `integrated_lufs`：FFmpeg `ebur128` 计算的 K 加权、门限积分响度。
- `true_peak_dbtp`：FFmpeg `ebur128` 真峰值。
- `dialogue_intervals[].max_short_term_lufs`：对话区间内 3 秒短期响度最大值。
- 短场景没有足够的 FFmpeg 3 秒短期帧时，使用 NumPy 实现的 BS.1770 K 加权和 3 秒窗后备计算，`method` 会标明来源。

`switch_check.json` 中主要字段：

- `cuts[].window_start_sample` 到 `window_end_sample_exclusive`：20 ms 对称切换窗口。
- `cuts[].equal_power_weight_max_error`：等功率权重平方和误差。
- `cuts[].boundary_continuity_error`：窗口首尾采样值连续性误差。
- `cuts[].passed`：边界连续、权重有效、无窗口内削波时为 `true`。
- `dialogue_ducking`：每个对话区间的采样点边界、过渡长度、稳态增益和增益连续性。

## 退出码

| 退出码 | 含义 |
|---:|---|
| 0 | 成功；无参数打印用法也返回 0 |
| 1 | 参数错误、输入/输出目录非法、未找到场景 |
| 2 | 单场景 JSON 缺失字段、非法时间或音频资源错误 |
| 3 | 批处理中至少一个场景失败；其他场景继续处理 |
| 4 | 单场景处理阶段失败，例如 FFmpeg 或输出写入失败 |

批处理的成功与失败详情始终写入 `summary.json`。

## 测试与验证

```bash
python -m unittest discover -s tests -v
python generate_samples.py samples
python main.py samples out
python - <<'PY'
import json, soundfile as sf
from pathlib import Path
for path in sorted(Path("out").glob("*/mix.wav")):
    info = sf.info(str(path))
    print(path, info.samplerate, info.channels, info.subtype, info.frames)
print(json.loads(Path("out/summary.json").read_text(encoding="utf-8")))
PY
```

当前工作区的实际验证结果：

- 自动化测试：21 项全部通过，覆盖 CLI 退出码、缺失字段、非法时间、采样点定位、循环不漂移、闪避连续性、切换连续性和生成器确定性。
- 生成样本：`forest`（48 kHz/16-bit）、`city`（44.1/32 kHz/16/24-bit）、`orchestra`（96/48/44.1 kHz、16/24-bit）。
- 批处理：3 个场景全部成功，退出码 `0`。
- 输出格式：3 个 `mix.wav` 均为 `48000 Hz`、`2 channels`、`PCM_16`；帧数分别为 `360000`、`384000`、`432000`，对应 7.5、8.0、9.0 秒。
- 切换检查：全部切点 `boundary_continuity_error <= 6.94e-18`、`equal_power_weight_max_error <= 2.3e-16`、`passed = true`。
- 响度报告：`forest` 整体响度约 `-25.0 LUFS`，真峰值约 `-10.2 dBTP`；每个对话区间均输出 3 秒短期响度最大值。
- 批处理容错测试：有效场景与缺失字段场景混合时，有效场景仍有输出，`summary.json` 记录 1 成功/1 失败，进程退出码为 `3`。
- 实际运行环境：Python 3.14.4、NumPy 2.5.5、SoundFile 0.14.0、FFmpeg 8.0.1（Linux/WSL）。当前容器没有原生 Windows 环境；代码已使用 `pathlib`、无 shell 的参数列表、Windows `CREATE_NO_WINDOW` 和反斜杠路径归一化，但原生 Windows 未在本机执行。
