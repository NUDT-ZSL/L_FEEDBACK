# Validation Record

验证日期：2026-11-15

## 环境

- Python 3.14.4
- NumPy 2.5.4
- SoundFile 0.14.0
- FFmpeg / FFprobe 7.0.2 static
- Linux x86_64（WSL2），非容器环境

## 自动化测试

```text
$ python3 -m pytest -q
.......................                                                  [100%]
23 passed in 2.17s
```

其中 `test_loudness_matches_ffmpeg_ebur128` 将本实现的 BS.1770-4 积分
响度与 FFmpeg `ebur128` 对照，确定性信号的误差小于 `0.15 LU`。

## 演示数据端到端结果

使用以下本地确定性生成步骤：

```bash
python3 scripts/generate_demo.py --output-dir demo
python3 -m dialogue_master --config demo/config.json --output-dir out1
python3 -m dialogue_master --config demo/config.json --output-dir out2
```

两次运行退出码均为 `0`。输出文件 SHA-256：

```text
ffc2728d719a3428af9f08b42c79f687a81d314cfa40dfc8d876c82ad1af6ac1  out1/processing_log.json
ffc2728d719a3428af9f08b42c79f687a81d314cfa40dfc8d876c82ad1af6ac1  out2/processing_log.json
3ab963d3858f9596085c92eed99568d6f15427431dd5e82b0ff6a3cd22b1b89d  out1/master.wav
3ab963d3858f9596085c92eed99568d6f15427431dd5e82b0ff6a3cd22b1b89d  out2/master.wav
```

因此相同输入重复运行时，JSON 日志和主轨 WAV 均逐字节一致。

`ffprobe` 对主轨的探测结果：

```text
codec_name=pcm_s24le
sample_rate=48000
channels=2
channel_layout=stereo
bits_per_sample=24
duration_ts=240000
duration=5.000000
```

这确认输出为 5 秒、48 kHz、24-bit 立体声 WAV，且时间轴采样数精确为
`240000`。

FFmpeg `ebur128=peak=true` 对整条主轨的结果：

```text
Integrated loudness: I: -24.0 LUFS
True peak: Peak: -22.0 dBFS
```

两个已放置片段的独立区段测量均约为 `-23.1 LUFS`，目标为
`-23.0 LUFS`。整轨读数约低 1 LU，原因是重叠片段被顺延后首尾相接，
两个片段在连接处的淡出/淡入窗口叠加，使跨越连接点的 400 ms 门限块
能量下降；这不改变各片段的增益与最终采样起点。

日志中的关键放置/响度结果：

```text
line_01         status=success requested=14400 final=14400 shift=0     lufs=-40.970226 gain=17.970226
line_02_overlap status=warning requested=38400 final=52800 shift=14400 lufs=-40.813339 gain=17.813339
bad_format      status=skipped codes=W_FORMAT
broken          status=skipped codes=E_AUDIO
late            status=skipped codes=W_LATE
```

- `line_01` 请求 `300 ms`，最终采样偏移 `14400`，未移动。
- `line_02_overlap` 请求 `800 ms`（采样 `38400`），与第一段冲突后
  顺延到第一段末尾 `1100 ms`（采样 `52800`），`sample_shift=14400`。
- `late` 请求 `4200 ms`（采样 `201600`），片段长 `48000` 采样，
  末端超过 `240000` 采样时间轴，因此 `W_LATE` 跳过。
- 原响度、最终增益、最终起点、采样偏移及状态均写入确定性日志。

## 业务码与退出码

成功混合运行（含可跳过的格式/损坏片段，至少一个片段成功）：

```text
RUN_RC=0
W_SHIFTED, W_FORMAT, E_AUDIO, W_LATE
```

配置缺少 `target_lufs`：

```text
$ python3 -m dialogue_master --config bad_config.json --output-dir never_created
E_CFG: clips[0].target_lufs: missing required field
CFG_RC=1
CFG_OUTPUT_EXISTS=no
```

配置错误不会创建输出目录，也不产生半成品。

仅包含一个格式错误片段和一个损坏片段：

```text
W_FORMAT: clip=bad_format ...
E_AUDIO: clip=broken ...
ALL_SKIP_RC=2
all_skipped_out/processing_log.json
```

全部片段跳过时退出码为 `2`，只保留日志，不生成主轨。

## 其他覆盖

- 8/16/24/32-bit PCM WAVEX 输出的位深和帧数正确。
- 5.1 输出带有正确声道掩码，`ffprobe` 识别为 `channel_layout=5.1`。
- 32-bit float 输入在保有相同位深时被接受，超满幅峰值会被约束并记录
  `W_PEAK`。
- 余弦平方淡变首末采样精确为零，淡变窗口采样数采用 half-up 换算。
- 增益峰值约束、±20 dB 上限、格式/损坏片段跳过、重叠顺延和越界
  跳过均有自动化测试。
- 同一份相对路径配置和音频放到两个不同父目录后，日志与主轨仍逐字节
  一致；解码错误诊断中的宿主绝对目录已替换为配置声明的 `input_path`。
