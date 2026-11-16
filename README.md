# broadcast-master

离线多语种广播母版处理命令行工具。输入多语种 24 位 WAV、帧级语种切换清单和视频容器，输出 EBU R128 `-23 LUFS` 单音轨 24 位 PCM WAV 及 JSON 报告。工具不访问网络，不依赖外部媒体素材。

## 环境

- Windows 10/11 64 位
- Python 3.10 或 3.11
- `numpy==1.24.4`
- `ffmpeg-python==0.2.0`
- FFmpeg 6.0 的 `ffmpeg.exe` 和 `ffprobe.exe` 已在 `PATH` 中

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
ffmpeg -version
ffprobe -version
```

也可以不做 editable 安装，直接使用仓库根目录的 `broadcast-master.cmd`；该脚本会把当前目录加入 `PYTHONPATH` 后调用 `python -m broadcast_master`。

## 快速运行

生成确定性的本地演示素材：

```bat
python tools\make_demo.py samples
```

正式处理：

```bat
broadcast-master.cmd ^
  --manifest samples\manifest.json ^
  --output out\master.wav ^
  --report out\report.json ^
  --temp-root "%TEMP%\broadcast-master"
```

只解析、规划裁剪并估计同步，不写 WAV 和报告：

```bat
broadcast-master --manifest samples\manifest.json --dry-run > dry-run.json
```

也可以不安装入口，直接运行：

```bat
python -m broadcast_master --manifest samples\manifest.json --output out\master.wav --report out\report.json
```

## 清单格式

```json
{
  "schema_version": 1,
  "frame_rate": 25,
  "duration_frames": 200,
  "crossfade_ms": 15,
  "audio": {
    "sample_rate": 48000,
    "channels": 2,
    "bit_depth": 24
  },
  "video": {
    "path": "program.mkv",
    "reference_stream_index": 1
  },
  "languages": [
    {
      "id": "zh-CN",
      "path": "zh-CN.wav",
      "sample_rate": 48000,
      "channels": 2,
      "bit_depth": 24
    },
    {
      "id": "en-US",
      "path": "en-US.wav",
      "sample_rate": 48000,
      "channels": 2,
      "bit_depth": 24
    }
  ],
  "switch_points": [
    {"frame": 0, "language": "zh-CN"},
    {"frame": 50, "language": "en-US"}
  ]
}
```

约束：

- `frame_rate` 支持正数或 `"24000/1001"` 形式的有理数。
- `audio.bit_depth` 必须为 `24`；声道数支持 1 或 2。
- 每个语种 `id` 唯一；WAV 实际采样率、声道数、位深必须与清单一致。
- `duration_frames` 可省略，此时取最短语种轨可覆盖的整帧数。
- `switch_points` 必须非空，帧号唯一。首项若不从 0 开始，工具会用首个语种补一个隐式 0 帧切换点。
- `video.reference_stream_index` 是容器内的 FFmpeg 全局流索引。省略时选第一条音频流；没有音频流时同步状态为 `no_reference`。
- `crossfade_ms` 只接受 `15`。

## 处理语义

- 每个语种轨以声明采样率的 float32 解码，使用 FFmpeg `ebur128=peak=true` 测量积分响度、短期响度、LRA 和真峰值；用 100 ms `astats` 窗口定位超过 `-1 dBTP` 的峰值时间码。积分响度低于 `-40 LUFS` 的语种标记 `silence_risk`，不中断。
- 切换点按 `round(frame * sample_rate / frame_rate)` 映射。若目标语种的 WAV 在对应帧没有有效样本，该切换点被丢弃，上一有效语种继续，并写入 `fallbacks`。
- 切换边界从切换帧开始，入轨淡入、前轨额外保留 15 ms 淡出，使用等功率正弦/余弦曲线；相控或同语种连续边界使用和为 1 的线性互补曲线。
- 合成音频下混为单声道并抽取到 4 kHz，与视频参考轨做归一化 FFT 互相关。正偏移表示“需要延迟合成音频”。补偿上限为 ±500 ms；超限时状态为 `out_of_range`，`applied_shift_samples` 和 `applied_shift_ms` 均为 0。
- 所有中间音频数组为 `float32`。优先静态增益；若峰值余量不足，退化为 FFmpeg 两遍 `loudnorm` 动态限峰，内部真峰值余量设为 `-1.2 dBTP`。最终 24 位文件会再次独立测量验证。
- 输出 WAV 固定为单音频流、24 位 PCM，首样本对应清单第 0 帧，采样数严格等于时间线映射长度。

## 错误与退出码

stderr 始终输出单行 JSON；业务校验失败不创建输出 WAV/报告：

```json
{"code":1003,"message":"audio.bit_depth must be an integer"}
```

- `0`：成功。
- `2`：清单字段缺失/类型错误、字段不一致、CLI 参数错误等业务错误。
- `3`：媒体读取、块对齐、FFmpeg 处理、恢复等相关处理错误。
- `4`：未预期内部错误。

主要错误码：

| code | 含义 |
|---:|---|
| 1001 | 清单不可读 |
| 1002 | JSON 语法错误 |
| 1003 | 必填字段/类型错误 |
| 1004 | 帧率错误 |
| 1005 | 采样率或声道不一致 |
| 1006 | 位深不是 24 或实际位深不一致 |
| 1007 | 语种标识错误 |
| 1008 | 切换点或参考流字段错误 |
| 3001 | WAV 不存在/截断/读取失败 |
| 3002 | WAV 容器结构错误 |
| 3003 | WAV 数据长度不是 block align 的整数倍 |
| 3011/3012 | FFmpeg 解码失败/样本块不完整 |
| 3020-3022 | 响度分析或 loudnorm 失败 |
| 3054 | 最终 24 位文件未通过响度/真峰值复测 |

## 断点恢复

进度标记位于：

```text
%TEMP%\broadcast_master\<manifest-run-key>\progress.json
```

可用 `--temp-root` 固定临时根目录。每次成功完成一个语种分析后，工具原子更新标记并记录源文件大小/mtime 签名。若后续音轨读取失败或 WAV 数据不是 block align 的整数倍：

1. 不生成最终 WAV 或报告。
2. 删除 staging 中所有已完成/半完成临时片段。
3. 保留 `progress.json` 及已完成语种的分析结果。
4. 修复输入后使用同一清单和 `--temp-root` 重跑，已完成分析被跳过。

测试中恢复运行与完整运行生成的 WAV 文件 SHA-256 完全一致。

## 报告字段

报告包括：

- `languages[]`：每语种响度、短期窗口最小/最大/均值、真峰值、超限峰值时间码、静音风险、使用片段。
- `segments[]`：有效切换片段、帧/样本边界、时间码、实际淡入淡出样本数与毫秒数。
- `fallbacks[]`：被拒绝的切换点、原因、回退语种和上一有效切换帧。
- `sync`：参考流、估计偏移、置信度、搜索范围、补偿边界、实际补偿样本数。
- `normalization`：归一化前/后积分响度与真峰值、有效增益、静态或动态方法。
- `verification`：最终 24 位 WAV 的 LUFS 容差和真峰值边界验证。
- `recovery`：是否恢复、跳过语种、标记路径。
- `stage_timings_ms`：各阶段耗时。

## 本地验证

运行自动化测试：

```bat
python -m unittest -v tests.test_broadcast_master
```

1 小时、48 kHz、双声道 dry-run 基准：

```bat
python tools\benchmark_dry_run.py
```

基准脚本生成逻辑时长 1 小时的稀疏 24 位 WAV（前 60 秒为确定性信号，余下为有效零样本），因此不会占用 3 GB 物理磁盘。

本次实际验证结果（当前工作区环境：Linux/WSL，Python 3.14.4，NumPy 2.5.5，FFmpeg 8.0.1；目标 Windows 组合为 Python 3.10/3.11、NumPy 1.24、FFmpeg 6.0）：

| 验证项 | 结果 |
|---|---|
| 自动化测试 | 8/8 通过，约 8.0 s |
| 30 ms 已知偏移 | 估计 `30.144 ms`，应用 `1447 samples / 30.146 ms`，置信度 `0.9253` |
| 750 ms 超限偏移 | `out_of_range`，应用 `0 samples` |
| 非法短轨切换 | 1 条回退事件，回落至上一有效语种 |
| 正常母版 | `-23.0 LUFS`，`-4.8 dBTP`，48 kHz/2ch/24-bit |
| 高波峰比动态路径 | `-23.1 LUFS`，`-1.5 dBTP`，`two_pass_loudnorm` |
| 断点恢复 | 首次失败无最终文件，staging 清理；恢复后 WAV SHA-256 与完整运行相同 |
| 1 小时 dry-run | 墙钟 `0.581 s`；解析 `0.262 ms`、探测 `31.493 ms`、规划 `0.053 ms`、同步 `366.497 ms`、阶段总计 `398.492 ms`，20 ms 偏移估计为 `20.0 ms` |

本工作区没有可用的原生 Windows 执行环境，因此上述命令实际在 Linux/WSL 中执行；代码未使用 POSIX 专用 API，Windows 使用方式已按 `Path`、`subprocess`、`%TEMP%` 和批处理命令给出。
