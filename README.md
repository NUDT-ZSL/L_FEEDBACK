# lipalign

离线、确定性的多语言配音口型对齐工具。工具从零实现 WAV 读写、静音检测、
WSOLA 与相位声码器时间伸缩，不调用外部音频处理库完成核心伸缩计算。

## 环境

- Windows 10/11 64 位
- Python 3.9 或以上
- NumPy 1.20 或以上
- 默认输出为多声道 WAV，不需要 FFmpeg
- FFmpeg 4.4+ 仅在增加非 WAV 封装时需要；本版本不调用 FFmpeg

安装依赖：

```bat
py -3.9 -m pip install -r requirements.txt
```

不需要安装软件包，在项目根目录直接运行 `python -m lipalign` 即可。

## 快速开始

生成固定种子的样本。样本包含口型时间轴、英语/中文/西班牙语单声道
48 kHz/16-bit PCM 配音，以及低音量粉红噪声背景：

```bat
py -3.9 -m lipalign generate --seed 20251117 --out "samples\demo"
```

执行 WSOLA 对齐和多轨封装：

```bat
py -3.9 -m lipalign align ^
  --timeline "samples\demo\timeline.json" ^
  --audio-dir "samples\demo\audio" ^
  --background "samples\demo\background.wav" ^
  --algorithm wsola ^
  --output "output\multitrack.wav" ^
  --report "output\alignment_report.json" ^
  --cache-dir "cache\fragments"
```

改用相位声码器：

```bat
py -3.9 -m lipalign align ^
  --timeline "samples\demo\timeline.json" ^
  --audio-dir "samples\demo\audio" ^
  --algorithm phase_vocoder ^
  --output "output\multitrack_pv.wav" ^
  --report "output\alignment_pv.json" ^
  --cache-dir "cache\fragments"
```

也可以重复使用 `--audio LANG=PATH`，路径可以含空格：

```bat
--audio en="C:\Voice Assets\English dub.wav" --audio zh="C:\Voice Assets\中文.wav"
```

查看报告和缓存摘要：

```bat
py -3.9 -m lipalign report ^
  --report "output\alignment_report.json" ^
  --cache-dir "cache\fragments"

py -3.9 -m lipalign report ^
  --report "output\alignment_report.json" ^
  --cache-dir "cache\fragments" ^
  --json
```

## 处理流程

1. `generate` 使用 NumPy `PCG64` 种子随机源生成时间轴和所有音频。同一组
   `seed + languages + sample_rate` 会生成逐字节相同的文件。
2. 时间轴以整数采样点为准；每个句子的目标采样数直接是
   `end_sample - start_sample`。
3. 每个语种按 20 ms 帧、5 ms 跳步计算 RMS 能量包络和过零率。仅当能量与
   过零率同时低于阈值时判为静音；静音段可合并、边界细化到采样点。
4. 检测到的活动段按时间中点分配给句子。某句无活动段时，仅该句失败并继续
   处理其它句子。
5. 对每个片段执行 WSOLA 或相位声码器伸缩，然后严格裁剪或补齐到
   `target_samples`。因此最终音轨长度与时间轴长度逐采样一致。
6. 每句的结果先写入缓存，再按时间轴固定位置拼装。片段首尾各使用 10 ms
   等功率交叉淡化，与前句尾部或时间轴静音平滑衔接。
7. 各语种声道与背景声道按列堆叠，写成一个标准 16-bit PCM 多声道 WAV，
   最后重新读取容器校验采样率、声道数和采样点数。

## 伸缩算法

### WSOLA

`lipalign/stretch.py` 内实现。合成帧长默认 40 ms、重叠 20 ms。每帧在
±10 ms 范围内搜索与上一帧重叠区域最相似的波形起点，使用 Hann 窗进行
overlap-add，再按权重归一化。时间尺度由分析跳步
`synthesis_hop / factor` 控制，基频不变。

### 相位声码器

同一模块内实现。默认 2048 点 FFT、4 倍重叠。每帧计算相位与理论相位
推进量之间的主值偏差，修正每个频点的真实相位速度，再按合成跳步累积相位，
通过 ISTFT 重建。最终输出同样严格裁剪/补齐到目标采样数。

两种算法的输出都会记录在报告中。测试使用 440 Hz 纯音做压缩和拉伸，
输出主频仍为 440 Hz（误差小于 2 Hz）。

## 增量缓存

缓存目录下每个片段包含一个 WAV 和一个 JSON sidecar。缓存键由以下字段的
规范化 JSON 计算 SHA-256：

- 片段原始 PCM 内容 SHA-256
- 片段原始采样数
- 目标采样数
- 采样率
- 算法及该算法实际使用的参数

因此：

- 音频内容不变、句子仅移动位置时，可复用缓存。
- 句子目标时长变化时，会重算该片段。
- 采样率或算法变化时，相关缓存全部失效。
- 缓存 WAV 被破坏、长度不一致或输出哈希不符时，视为未命中并自动重算。

每个语种、每个句子都会在报告中记录 `cache=hit` 或
`cache=recomputed`，汇总值位于 `totals.cache_hits` 和
`totals.cache_recomputes`。

## 输入时间轴

核心结构如下；同时提供 `_sample` 和 `_sec` 字段，实际以采样点为准：

```json
{
  "sample_rate": 48000,
  "seed": 20251117,
  "duration_samples": 587305,
  "sentences": [
    {
      "id": "s001",
      "text": "The quick brown fox",
      "start_sample": 14400,
      "end_sample": 93535,
      "phonemes": [
        {
          "phoneme": "AA",
          "mouth": "open",
          "start_sample": 14400,
          "end_sample": 20200
        }
      ]
    }
  ]
}
```

读取器也接受仅含 `start_sec`/`end_sec` 的时间轴，并按“最近整数采样点、
0.5 向上”转换为采样点。口型类别是非空字符串，内置样本使用
`closed/open/wide/round/neutral/labiodental/fricative`。

## 输出与报告

输出 WAV 的声道顺序为：

```text
language:<LANG1>, language:<LANG2>, ..., background:pink_noise
```

所有声道共享同一个帧数。报告为单个 JSON 文件，主要字段包括：

- `timeline`: 时间轴 SHA-256、采样率、总采样点、总时长
- `settings`: 算法、VAD 参数、交叉淡化参数、缓存目录
- `languages[].split`: 静音设置、检测到的活动段起止采样点
- `languages[].sentences[]`: 每句目标/实际起止采样点与秒数
- `source_start_sample/source_end_sample`: 基于 VAD 的切分结果
- `stretch_factor` 与 `stretch_factor_fraction`: 目标采样数/源采样数
- `rounding_policy`: 目标取整、输出裁剪/补齐及 PCM 舍入策略
- `algorithm`: `wsola` 或 `phase_vocoder`
- `output`: 声道名、各声道哈希、总采样点数、时长误差
- `totals`: 处理句数、失败数、缓存命中、重算数和耗时

`output.duration_error_samples` 为 `0`；报告中的时长误差上限为 5 ms。

## 错误处理

预期错误以非零退出码退出，stderr 只输出一行 JSON，例如：

```json
{"code":"E_RATE_MISMATCH","message":"Language sample rate 44100 Hz does not match timeline 48000 Hz","ok":false}
```

| 错误码 | 含义 | 退出码 |
| --- | --- | --- |
| `E_JSON_INVALID` | JSON 语法错误 | 2 |
| `E_TIMELINE_INVALID` | 时间轴结构/语义错误 | 2 |
| `E_WAV_CORRUPT` | RIFF/WAVE 结构损坏、截断或长度不一致 | 2 |
| `E_WAV_FORMAT` | 非 16-bit PCM 或不受支持的格式 | 2 |
| `E_RATE_MISMATCH` | WAV 与时间轴采样率不一致 | 2 |
| `E_CHANNEL_MISMATCH` | 配音/背景不是单声道 | 2 |
| `E_AUDIO_LENGTH` | 音频短于时间轴 | 2 |
| `E_PARTIAL_FAILURE` | 部分句子失败；成功输出与缓存已保留 | 6 |
| `E_OUTPUT` | 输出写入失败 | 2 |
| `E_INTERNAL` | 未预期内部错误 | 1 |

单句失败不会删除其它句子已经写入缓存的片段，也不会阻止多轨文件和报告生成。

## 测试

```bat
py -3.9 -m unittest discover -v
```

测试覆盖：

- 相同种子逐字节确定性
- 48 kHz 单声道 16-bit PCM 样本
- WSOLA/相位声码器输出长度与基频保持
- 多声道 WAV 帧数一致性、背景低音量
- 报告字段完整性与 5 ms 时长限制
- 全量缓存命中
- 单片段内容变化后仅重算 1 段
- 算法切换、采样率切换后的缓存失效
- 坏 JSON、坏 WAV、采样率不匹配的单行 JSON 错误
- 单句缺失时其余句子完成且保留缓存

实际验证结果见 `docs/VERIFICATION.md`。

## 项目结构

```text
lipalign/
  synth.py      确定性样本和粉红噪声合成
  timeline.py   时间轴解析与校验
  vad.py        RMS + ZCR 静音检测和句子切分
  stretch.py    WSOLA 与相位声码器
  cache.py      片段内容哈希缓存
  aligner.py    对齐、拼装、多轨封装、报告
  wavio.py      严格 PCM WAV 读写
  cli.py        generate/align/report 子命令
tests/          标准库 unittest 测试
docs/           实测记录
```
