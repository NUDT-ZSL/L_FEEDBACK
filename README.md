# EBU R128 对白响度离线处理器

一个零外部素材、纯本地运行的离线命令行工具，对一组 WAV 对白音轨及其同名
SRT 字幕执行：

- 基于 ITU-R BS.1770-4 的 EBU R128 响度测量与 -23 LUFS 标准化（±0.5 LU）
- 44.1/48 kHz、16/24-bit 输入统一重采样为 48 kHz / 24-bit PCM（FFmpeg 抗混叠）
- 4x 过采样真峰值测量与 -1.0 dBTP 前瞻限制器
- 基于静音段（低于 -60 LUFS 且持续 > 200 ms）的 SRT 时间轴校正
- 多声道（单声道 / 立体声 / 5.1）按标准加权，异常检测与可恢复跳过
- 并行处理、独立临时目录、UTF-8 JSON 处理报告

## 环境要求

- Python 3.9+（开发与验证使用 3.14，64 位）
- NumPy 1.23+
- FFmpeg 5.0+，且 `ffmpeg` 在 `PATH` 中（用于重采样与格式转换）

本仓库在无 root 环境下的实际安装方式（Ubuntu，静态 FFmpeg）：

```bash
uv venv .venv
uv pip install --python .venv/bin/python "numpy>=1.23"
mkdir -p ~/.local/ffmpeg-bin
# 下载 https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
# 解压后将 ffmpeg/ffprobe 复制到 ~/.local/ffmpeg-bin
export PATH="$HOME/.local/ffmpeg-bin:$PATH"
```

Windows 下使用官方 64 位 Python 安装包与 `pip install -r requirements.txt`，
并将 FFmpeg 的 `bin` 目录加入系统 `PATH` 即可，代码与平台无关。

## 使用方法

```bash
# 1. 生成确定性样本（9 组 WAV/SRT，含正常与异常样本）
python generate_samples.py --output samples/input

# 2. 运行处理器（默认并行度 2）
python -m audio_norm.cli samples/input samples/output --parallelism 2

# 3. 运行测试
python -m unittest discover -s tests -v
```

命令行参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `input_dir` | 含 `.wav` 与同名 `.srt` 的输入目录 | 必填 |
| `output_dir` | 标准化产物与报告的输出目录 | 必填 |
| `-j, --parallelism` | 并行处理音轨数 | `2` |
| `--report-name` | JSON 报告文件名 | `processing_report.json` |

输出文件命名：`<原名>_normalized.wav`、`<原名>_normalized.srt`。

退出码：`0` 全部成功；`1` 部分成功（个别文件错误被跳过）；`2` 全局致命错误
（输入目录不存在、输出目录不可写、FFmpeg 缺失等）。

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `audio_norm/loudness.py` | K 加权、声道加权、双门限积分响度、瞬时响度、静音检测、4x 过采样真峰值 |
| `audio_norm/limiter.py` | 过采样域前瞻峰值限制、补偿增益与安全护栏 |
| `audio_norm/wavio.py` | 16/24-bit PCM WAV 读写与格式校验 |
| `audio_norm/resample.py` | FFmpeg `aresample` 高质量抗混叠重采样封装 |
| `audio_norm/srt.py` | SRT 解析、格式化与静音对齐 |
| `audio_norm/processor.py` | 单轨流水线、并行编排、临时目录管理 |
| `audio_norm/report.py` | JSON 报告聚合与状态判定 |
| `audio_norm/cli.py` | 命令行入口与退出码 |
| `generate_samples.py` | 确定性样本生成器（固定随机种子） |
| `tests/` | 单元测试与端到端验收测试 |

## 关键实现说明

### BS.1770-4 响度测量

1. K 加权：高频高架滤波器（+3.61 dB @ 1681.97 Hz）后接高通滤波器
   （38.135 Hz），系数按 RBJ Audio EQ Cookbook 计算。
2. 声道加权：L/R/C 权重 1.0，LFE 权重 0，环绕 Ls/Rs 权重 √1.5。
3. 400 ms 块、100 ms 帧移计算加权均方能量，转 LUFS：
   `L = -0.691 + 10 log10(mean-square)`。
4. 双门限：先施加 -70 LUFS 绝对门限，再以均值 -10 LU 作为相对门限。

与 FFmpeg `loudnorm` 参考实现的实测偏差为 0.0–0.36 LU，真峰值读数一致。

### 重采样

仅接受 44100/48000 Hz；其它采样率在报告中标记错误并跳过。重采样经
FFmpeg `aresample=...:filter_size=256:phase_shift=10`（libswresample
长抽头多相位抗混叠核），输出统一为 48000 Hz、24-bit PCM。

### 真峰值与限制器

- 真峰值通过 4x 过采样（129 抽头加窗 sinc 抗成像滤波）取最大绝对样本。
- 归一化后若真峰值 > -1.0 dBTP，则在过采样域检测每采样所需衰减，按
  4:1 分组映射回原始采样率施加前瞻增益，避免二次重采样的群延迟错位。
- 内部以 -1.3 dBTP 为工作天花板，预留补偿与护栏余量；补偿增益在不突破
  天花板的前提下把响度拉回 -23 LUFS，最后由确定性安全护栏兜底。
- 当素材在物理上无法同时满足 -23 LUFS 与 -1.0 dBTP（如持续满幅信号）时，
  优先保证真峰值不超标，并在报告中如实记录响度偏差。

### 字幕时间轴校正

以 200 ms 帧、50 ms 帧移计算瞬时响度，找出低于 -60 LUFS 且持续 ≥ 200 ms
的连续静音段。若字幕起始点落在静音段内，将起点平移到该静音段结束处
（第一个语音帧），持续时间保持不变，并记录原/新时间。

### 异常与临时目录

- WAV 损坏、SRT 格式错误、声道数不在 1/2/6、采样率不在 44100/48000：
  标记该文件错误、跳过并继续处理其余文件（状态 `partial`）。
- 每个音轨使用独立 `tempfile.mkdtemp` 临时目录；成功后清理，失败则保留，
  路径写入报告 `retained_temp_dirs` 与对应条目的 `temp_dir`。

## 验证结果

详见 `VERIFICATION.md`。
