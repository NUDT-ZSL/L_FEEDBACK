# HLS 多码率音频构建器

这是一个本地命令行工具，用于把同一段音频的多个码率 WAV 版本构建成可自适应切换的 HLS 音频输出。工具以最低码率档位为时间轴基准，在采样点级别统一切片边界，并为每个档位生成独立的 AAC 分片、媒体播放列表和构建报告。

## 功能

- 读取并严格校验 JSON 配置，失败时退出码为 `2`。
- 使用 FFmpeg 将 WAV 解码为 16-bit PCM，使用 ffprobe 读取采样率、声道、时长和 `start_time`。
- 以最低码率档位为参考，按采样点计算相对偏移，支持前导静音和尾部静音填充。
- 所有档位共享同一组样本边界，分片数量、`EXTINF` 累计边界和 AAC 帧数一致。
- 每个分片以目标码率独立编码为 ADTS AAC，移除编码器 priming 帧后时长误差不超过一个 AAC 帧。
- 生成 VOD 媒体播放列表和带 `BANDWIDTH`、`AVERAGE-BANDWIDTH` 的主播放列表。
- 构建完成后执行跨档位一致性校验，失败时退出码为 `3`。
- 输出目录冲突时退出码为 `4`；`overwrite: true` 时清理旧构建产物后重建。
- `build_report.json` 记录分片数、时长、对齐量、实际平均码率、文件大小和每个分片的 SHA-256。
- 相同输入、配置和 FFmpeg 环境下，重复构建的播放列表、分片和报告字节级一致。

## 环境

- Python 3.9 及以上
- NumPy，仅用于生成确定性验证样本
- FFmpeg 6.0 及以上，`ffmpeg` 和 `ffprobe` 位于 `PATH`

当前验证环境：

```text
Python 3.14.4
NumPy 2.5.4
FFmpeg 8.0.1
ffprobe 8.0.1
```

## 快速开始

生成 44.1 kHz 单声道确定性正弦波样本：

```bash
python3 tools/generate_samples.py --output-dir samples
```

生成的文件包括：

```text
samples/audio_64k.wav
samples/audio_128k.wav
samples/audio_192k.wav
samples/config.json
```

128k 样本带 25 ms 前导静音，配置使用 `offset_ms: 25` 裁掉该偏移；192k 样本尾部短 15 ms，构建时由统一时间轴补静音。

执行构建：

```bash
python3 -m hls_audio_builder --config samples/config.json
```

等价入口：

```bash
python3 -m hls_audio_builder samples/config.json
python3 -m hls_audio_builder -c samples/config.json
```

安装为命令后也可使用：

```bash
python3 -m pip install .
hls-audio-builder --config samples/config.json
```

## 配置格式

```json
{
  "output_dir": "../demo_output",
  "segment_duration": 2,
  "overwrite": false,
  "variants": [
    {
      "bitrate_kbps": 64,
      "input": "audio_64k.wav",
      "offset_ms": 0
    },
    {
      "bitrate_kbps": 128,
      "input": "audio_128k.wav",
      "offset_ms": 25
    },
    {
      "bitrate_kbps": 192,
      "input": "audio_192k.wav",
      "offset_ms": 0
    }
  ]
}
```

字段说明：

- `output_dir`：输出目录。相对路径基于配置文件所在目录解析。
- `segment_duration`：分片时长，范围 `1` 到 `30` 秒。
- `overwrite`：可选，默认 `false`。为 `true` 时重建前清理本工具生成的旧文件。
- `variants`：非空档位数组，必须按 `bitrate_kbps` 严格升序排列，且码率不能重复。
- `bitrate_kbps`：AAC 目标码率，单位为 kbps。
- `input`：输入 WAV。相对路径基于配置文件所在目录解析。
- `offset_ms`：可选，默认 `0`，相对最低码率参考档位的对齐修正量。

偏移量符号约定：

- 正值表示该输入内容相对参考档位向后偏移，需要从输入开头裁掉相应采样。例如输入带 25 ms 编码器延迟时设置 `offset_ms: 25`。
- 负值表示需要在该输入开头补静音，使其时间轴向后移动。
- ffprobe 的 `start_time` 差异会叠加到 `offset_ms`，最终以半入方式量化到采样点。

兼容字段别名：`output_directory`、`segment_duration_seconds`、`bitrates`、`bitrate`、`input_wav`。

## 输出结构

```text
demo_output/
├── master.m3u8
├── build_report.json
├── 64k/
│   ├── index.m3u8
│   ├── segment_00000.aac
│   └── ...
├── 128k/
│   ├── index.m3u8
│   └── ...
└── 192k/
    ├── index.m3u8
    └── ...
```

`master.m3u8` 使用相对 URI 指向各档位媒体播放列表；媒体播放列表使用同一目录下的相对分片 URI。

示例媒体播放列表：

```text
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:2
#EXT-X-MEDIA-SEQUENCE:0
#EXT-X-PLAYLIST-TYPE:VOD
#EXT-X-INDEPENDENT-SEGMENTS
#EXTINF:2.000000000,
segment_00000.aac
#EXTINF:2.000000000,
segment_00001.aac
#EXTINF:2.000000000,
segment_00002.aac
#EXTINF:1.300000000,
segment_00003.aac
#EXT-X-ENDLIST
```

## 退出码

- `0`：构建和一致性校验成功。
- `1`：FFmpeg 执行失败、外部程序缺失或其他运行期错误。
- `2`：配置缺失、JSON 非法、码率顺序错误、分片时长越界、输入不存在、采样率不一致、PCM 解码失败等输入错误。
- `3`：生成后跨档位一致性校验失败，错误包含档位和分片索引。
- `4`：输出目标已存在且 `overwrite` 为 `false`。

## 时间轴与编码实现

1. 参考档位为 `variants[0]`，即最低码率。
2. FFmpeg 将各输入解码为交错 16-bit PCM；参考档位 PCM 样本数决定统一时间轴总长度。
3. 分片容量为 `floor(segment_duration * sample_rate)`，边界为 `0, N, 2N, ...`，最后一片截到参考总样本数。
4. 每个档位按相对偏移生成固定长度对齐 PCM：
   - 正偏移从输入开头裁切；
   - 负偏移在开头写静音；
   - 音频提前结束时在尾部补静音。
5. 每个分片调用一次 FFmpeg，传入样本边界换算的精确 `-ss` 和 `-t`，以目标码率编码 AAC。
6. FFmpeg 原生 AAC 编码器会输出一个 priming 帧。工具解析 ADTS 帧并移除首帧，使 `ceil(segment_samples / 1024)` 个 AAC 帧覆盖目标分片，剩余误差小于一个 AAC 帧。
7. 一致性校验重新解析媒体播放列表和 ADTS，交叉检查分片数、累计 `EXTINF` 边界、URI、采样率、ADTS 帧数、ffprobe 包数及跨档位帧数。

ADTS AAC 本身不携带全局 PTS。HLS 的连续时间轴由统一样本边界、`EXTINF` 累计时间和播放器按序拼接共同保证；`-ss` 负责选择精确的绝对时间起点。

## 测试

```bash
python3 -m unittest discover -v
```

测试覆盖：

- 统一样本边界与最后一片长度；
- 正偏移裁切、负偏移补静音；
- 多档位端到端构建；
- 播放列表头、URI、累计边界；
- AAC 帧数与一帧时长容差；
- 报告和全部分片哈希；
- `overwrite` 重建成败及字节稳定性；
- 配置错误退出码 `2`；
- 人为注入帧数不一致时退出码 `3`；
- 输出冲突退出码 `4`。

## 实际验证结果

本工作区执行：

```bash
python3 tools/generate_samples.py --output-dir samples
python3 -m hls_audio_builder --config samples/config.json
```

结果：

```text
built 3 variants x 4 segments at 44100 Hz into .../demo_output
sample_boundaries = [0, 88200, 176400, 264600, 321930]
EXTINF = [2.0, 2.0, 2.0, 1.3] seconds
AAC frame counts = [87, 87, 87, 56] for every bitrate
64k  alignment: trim 0 samples, tail silence 0
128k alignment: trim 1103 samples (25.011337868 ms), tail silence 0
192k alignment: tail silence 662 samples
build_report.json SHA-256 = 5905feb0710d015fbee07e430283c76f309d1152fd071ce0922c44abde3f3df0
master.m3u8 SHA-256 = 608a6ad9c76a1daae06a7c9a0ae8c1a243e17e4b9bcc5230a9670de679a8f666
```

对首分片解码后与 64k 参考档位做互相关：

```text
128k lag = 0 samples
192k lag = 0 samples
```

以 `overwrite: true` 重复构建后，`master.m3u8`、三个媒体播放列表、12 个 AAC 分片和 `build_report.json` 的 SHA-256 均未变化。未设置覆盖时再次运行返回退出码 `4`。

使用 FFmpeg 的 HLS 解复用器读取主播放列表并完整解码：

```bash
ffmpeg -v error -i demo_output/master.m3u8 -map 0:a:0 -f null -
```

命令退出码为 `0`。

自动化测试结果：

```text
Ran 9 tests
OK
```

