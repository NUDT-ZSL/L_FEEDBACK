# 游戏过场动画对白优先环绕声母版工具

这是一个完全离线的 NumPy/FFmpeg 处理工具，用于把对白、音乐、音效三类多轨 WAV
整合为广播级多声道母版。它执行抗混叠重采样、WAVE 声道掩码解析、5.1/立体声映射、
对白活跃段检测、软拐点压缩、ITU-R BS.1770 响度归一化、4 倍过采样真峰值限幅，
输出 WAVE_FORMAT_EXTENSIBLE WAV、MP4 和机器可读 JSON 报告。

处理链不使用随机数，不读取系统时间，不依赖外部素材。相同输入目录、配置和参数会
产生比特级一致的 WAV 与报告；MP4 容器中由 FFmpeg 写入的创建时间字段不在确定性
保证范围内。

## 环境与安装

- Python 3.9+
- NumPy 1.23+
- soundfile 0.12+（及其系统 `libsndfile`）
- FFmpeg 5.0+，且 `ffmpeg`、`ffprobe` 位于 `PATH`

```bash
python3 -m pip install -r requirements.txt
```

也可安装本地命令：

```bash
python3 -m pip install .
cinema-master --version
```

## 快速运行

生成自带确定性测试素材：

```bash
python3 -m cinema_master.samplegen demo_input
```

该命令在 `demo_input/` 中生成：

- `dialogue.wav`：44.1 kHz、单声道、普通 PCM，内容位于中置声道
- `music_51.wav`：48 kHz、6 声道、WAVE_FORMAT_EXTENSIBLE，掩码 `0x3f`
- `effects_51.wav`：32 kHz、6 声道、普通 PCM，按标准 5.1 顺序解释
- `config.json`：完整处理配置
- `sample_manifest.json`：生成参数与预期对白片段，不参与母版配置

执行母版处理：

```bash
python3 -m cinema_master demo_input demo_output
```

输出：

- `demo_output/master.wav`：48 kHz、24-bit、6 声道 EXTENSIBLE WAV（掩码 `0x3f`）
- `demo_output/master.mp4`：AAC 或 PCM MP4，含声道布局和响度元数据
- `demo_output/report.json`：处理参数、重采样、映射、活跃段、压缩、响度、真峰值、
  限幅迭代、封装校验和 WAV SHA-256

CLI 成功时向 stdout 输出一行 JSON 摘要并以 0 退出。可显式指定配置：

```bash
python3 -m cinema_master --config alternate.json input_dir output_dir
```

## 配置格式

`config.json` 的完整示例见 `cinema_master/samplegen.py`。关键字段如下：

```json
{
  "schema_version": 1,
  "target_sample_rate": 48000,
  "output_layout": "5.1",
  "target_lufs": -23.0,
  "true_peak_limit_dbtp": -1.0,
  "max_relative_gain_db": 4.0,
  "loudness": {
    "tolerance_lu": 0.5,
    "iteration_limit": 10
  },
  "compression": {
    "enabled": true,
    "threshold_dbfs": -30.0,
    "ratio": 2.5,
    "attack_ms": 15.0,
    "release_ms": 180.0,
    "knee_db": 6.0
  },
  "dialogue_detection": {
    "frame_ms": 400.0,
    "overlap": 0.75,
    "threshold_dbfs": -45.0,
    "min_segment_ms": 180.0
  },
  "limiter": {
    "lookahead_ms": 10.0,
    "release_ms": 120.0,
    "oversample": 4,
    "iteration_limit": 10,
    "safety_margin_db": 0.25
  },
  "output": {
    "basename": "master",
    "audio_codec": "aac",
    "aac_bitrate": "256k",
    "bits_per_sample": 24
  },
  "tracks": [
    {"file": "dialogue.wav", "group": "dialogue", "channel_map": ["FC"]},
    {"file": "music_51.wav", "group": "music"},
    {"file": "effects_51.wav", "group": "effects"}
  ]
}
```

### 轨道与声道

- `group` 支持 `dialogue`、`music`、`effects` 及常见别名（`dx`、`mx`、`sfx` 等）。
- `channel_map` 可为“按源声道顺序”的角色列表，或 `{"FC": 2, "FL": 0}` 形式的
  角色到源索引映射；`null`/空字符串可丢弃不参与映射的源声道。
- `input_layout` 可显式覆盖 WAV 头中的布局。
- WAVE_FORMAT_EXTENSIBLE 从 `fmt` 的 `dwChannelMask` 解析角色。
- 普通 PCM 没有掩码时：1 声道推断为中置，2 声道推断为 L/R，4 声道推断为
  L/R/Ls/Rs，6 声道推断为标准 5.1，8 声道推断为 7.1。
- 5.1 使用 `FL, FR, FC, LFE, BL, BR` 顺序和掩码 `0x3f`。
- 立体声下混采用 ITU-R BS.775：L/R 直通，中置和环绕各乘
  `1/sqrt(2)`；LFE 不进入立体声下混。
- 缺失输出声道填零。所有轨道先重采样，再映射并补齐到统一最大时间轴，不添加
  首尾延迟。

## 处理算法

### 抗混叠重采样

采样率不一致时使用带限 FFT 重采样。输入先做短时反射边界延拓，频域按新采样率的
Nyquist 截断/补零，再做逆变换并按输出帧数精确裁剪。输出帧数为
`round(input_frames * target_rate / source_rate)`。下采样时高于目标 Nyquist 的
频率被截断，报告记录每轨重采样前后采样率、帧数、抗混叠截止频率和方法。

### 对白活跃段

对白分组按配置的帧长和重叠率计算多声道平均能量，默认 400 ms 帧、75% 重叠
（100 ms 步进）。帧能量高于阈值则活跃；连续活跃帧合并，低于最短时长的片段删除。
报告输出每段的 `start_sample`、`end_sample`、起止秒和时长。

### 软拐点压缩

压缩检测器使用对白分组的声道联动峰值包络，增益以 dB 域启动/释放包络平滑，并只
在活跃段内建立压缩目标。压缩增益链路作用于所有分组，因此保持声像与相对关系；
非活跃段目标增益为 0 dB，释放尾音按配置自然收束。软拐点曲线使用标准二次插值区。
报告包含最大衰减、活跃段平均衰减、压缩前后节目响度和活跃样本数。

### 对白优先响度与增益约束

K 加权、400 ms 分块、绝对 `-70 LUFS` 和相对 `-10 LU` 门限均按 ITU-R BS.1770
实现。对白和其他分组分别计算积分响度。全局增益先以“目标响度 - 对白响度”初始化，
再迭代使积分节目响度进入目标 ±`tolerance_lu`。

音乐和音效除满足配置的相对增益上限外，还执行更严格的业务规则：每个非对白分组
的积分响度不得超过对白；若超过，则按测量差值衰减，并对音乐/音效汇总总线再次
校验。所有修正值、最终偏移和是否满足约束均写入报告。若真实峰值或迭代上限使响度
无法进入容差，`loudness.tradeoffs` 会记录取舍码、最终达到的响度和原因。

### 前视真峰值限幅

使用配置倍数（默认 4 倍）FFT 过采样估算样本间真峰值。限幅器以未来前视窗内的
最小允许增益为目标，采用样本级联动增益、快速启动和配置释放时间；所有声道使用
同一增益。限幅后重新计算响度，如果偏离目标超过容差，就调整限幅前增益并再次迭代。
总迭代上限取 `loudness.iteration_limit` 和 `limiter.iteration_limit` 的较小值。
若过采样结果仍超上限，附加安全 trim，并在报告记录。

### 封装与元数据

工具手工写出 WAVE_FORMAT_EXTENSIBLE PCM WAV，保证 5.1 声道掩码正确。随后直接调用
`ffmpeg` 封装 MP4：

- `output.audio_codec = "aac"`：使用配置码率 AAC
- `output.audio_codec = "pcm"`：使用 24-bit 或 16-bit PCM
- 写入标题、声道布局、目标响度、实测响度和真峰值到 MP4 格式注释
- 用 `ffprobe` 校验编码、声道布局、时长和元数据；AAC 默认容差为一个 1024 样本帧
- 所有输出先写入输出目录内的临时暂存目录，成功后原子替换；失败时删除暂存目录

## 错误处理

任何可预期失败都会写 `output_dir/report.json`，内容形如：

```json
{
  "schema_version": 1,
  "status": "error",
  "error": {
    "code": "E_CHANNEL_MISMATCH",
    "message": "channel_map length does not match WAV channels",
    "details": {}
  }
}
```

CLI 同时向 stderr 输出 JSON 并以非零码退出。主要业务码：

| 业务码 | 含义 |
| --- | --- |
| `E_CONFIG_MISSING` | 配置不存在或无法定位 |
| `E_CONFIG_INVALID_JSON` | JSON 非法或不是 UTF-8 |
| `E_CONFIG_SCHEMA` | 配置字段类型、范围或必填项错误 |
| `E_INPUT_MISSING` | 输入目录或轨道文件不存在 |
| `E_AUDIO_READ` | WAV 损坏、格式不支持或数据不完整 |
| `E_CHANNEL_MISMATCH` | 声道数、掩码或映射长度不匹配 |
| `E_NO_DIALOGUE` | 未检测到满足阈值的对白活跃段 |
| `E_FFMPEG_NOT_FOUND` | PATH 中缺少 FFmpeg/FFprobe |
| `E_FFMPEG_FAILED` | FFmpeg 非零退出或未产生 MP4 |
| `E_FFPROBE_FAILED` | 无法探测封装结果 |
| `E_DURATION_MISMATCH` | MP4 与 WAV 时长差超过配置容差 |
| `E_OUTPUT_WRITE` | 输出目录或文件写入失败 |

## 测试与验收

运行全部自动化测试：

```bash
python3 -m unittest discover -s tests -v
```

运行自带 5.1 端到端验收，包括两次 CLI 运行、外部 `ffmpeg ebur128` 测量、
`ffprobe` 时长/元数据校验、强制限幅用例和错误报告用例：

```bash
python3 scripts/run_acceptance.py --root test_tmp/acceptance
```

### 实际验证记录

验证环境：Python 3.14.4、NumPy 2.5.4、soundfile 0.14.0/libsndfile 1.2.2、
FFmpeg/FFprobe 8.0.1。

标准 `-23 LUFS` 验收结果：

- 自动化测试：`26` 项全部通过
- 工具报告节目响度：`-23.0 LUFS`；FFmpeg 独立测量：`-22.9 LUFS`
- 工具报告真峰值：`-7.611813 dBTP`；FFmpeg 独立测量：`-7.6 dBFS`
- WAV 与 MP4 时长：均为 `9.000000 s`
- MP4 编码：AAC、6 声道、`5.1`；响度/真峰值注释已由 FFprobe 验证
- 两次运行 WAV SHA-256：
  `0dcaa536bf1f0569143bd00993776a092c2f2af14145839b171b834d23357e10`
- 两次运行 `report.json` 哈希一致
- 4 个对白活跃段；44.1 kHz 与 32 kHz 轨道均记录重采样到 48 kHz
- 无效 JSON、缺失文件、声道映射不匹配、FFmpeg 非零退出均生成含业务码的 JSON
  报告并以非零退出；临时目录被清理

强制限幅验收（把目标改为 `-16 LUFS`）：

- 最终节目响度：`-15.784693 LUFS`，目标误差 `0.215307 LU`
- 真峰值：`-1.243868 dBTP`，低于 `-1.0 dBTP` 上限
- 限幅触发，最大衰减约 `0.90 dB`，经过 2 次归一化/限幅迭代

## 已知边界

- 当前输出为 RIFF/WAVE，单文件上限约 4 GiB；超长母版应扩展为 RF64。
- 处理全程在内存中完成，适合单部游戏过场；超长素材需要分块架构。
- MP4 AAC 是有损编码，报告真峰值以交付 WAV 为准；AAC 编码后的峰值会受编码器影响。
