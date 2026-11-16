# 实际验证记录

验证日期：2026-11-17  
环境：Ubuntu / Python 3.14.4 / NumPy 2.5.5（实现按 Python 3.9+ 兼容编写）  
方式：本地确定性生成样本，无 Docker、无外部音频、无网络访问。

## 1. 样本生成

```bash
python tools/make_samples.py --directory samples
```

生成的 `skill_120_mono.wav`、`percussion_120_stereo.wav`、
`skill_128_stereo.wav`、低置信度噪声及三种坏格式夹具均由代码确定性生成。

## 2. 主验收命令

```bash
python bpm_grain.py \
  --input samples/skill_120_mono.wav \
  --percussion samples/percussion_120_stereo.wav \
  --output verification_run/mix.wav \
  --seed 4242 \
  --log-level INFO
```

退出码 `0`，stdout 为 `0` 字节。stderr：

```text
INFO analyzed BPM 120.000 (confidence 0.986), beat=22050 samples
INFO percussion aligned BPM 120.000 (raw 239.500, confidence 0.773)
INFO percussion aligned at sample 0 on a 352800-sample bar grid
WARNING soft limiter applied: pre-limit peak 1.524 dBFS, gain adjustment -3.229 dB
INFO wrote verification_run/mix.wav (441000 samples), peak -1.706 dBFS, grains=80, limiter=True
```

输出 WAV 实测：

```text
channels=2
sampwidth=2 (16-bit)
framerate=44100
nframes=441000
```

关键报告值：

```text
bpm=120.0
confidence=0.986375
beat_samples=22050
cloud_loop_samples=88200 (=4*22050)
total_samples=441000
grain_count=80
peak_dbfs=-1.705562
peak_level=0.821716309
limiter_gain_adjustment_db=-3.229496
soft_limiting_applied=true
```

`mixing_plan.json` 的 `bpm`、`confidence`、`beat_samples` 与报告一致；所有轨道的
`start_sample` 均为 `0`。`grains.json` 中 80 个粒子的起止差均为 2205 样本
（50 ms），并包含源位置、播放速率、移调半音和跨界环绕标记。

## 3. 确定性复现

使用相同输入与 `--seed 4242` 在另一目录重复运行，结果：

```text
7dca7737828876c6c240f6acc96c2fb9defc7358053201d8e4715efd2cc5d8c9  mix.wav (first)
7dca7737828876c6c240f6acc96c2fb9defc7358053201d8e4715efd2cc5d8c9  mix.wav (repeat)
0b1ef26088d70068fd4df80c462f31f755ff1a121fe0b952a4d4f04f34499cd2  grains.json (first)
0b1ef26088d70068fd4df80c462f31f755ff1a121fe0b952a4d4f04f34499cd2  grains.json (repeat)
wav_identical=yes
grains_identical=yes
```

## 4. 错误路径

低置信度噪声：

```text
exit=1 stdout_bytes=0
ERROR ERR_LOW_CONFIDENCE: BPM confidence 0.083 is below 0.70
output_absent=yes
```

采样率 48 kHz：

```text
exit=2 stdout_bytes=0
ERROR ERR_BAD_FORMAT: samples/bad_sample_rate_48000.wav: sample rate must be 44100 Hz, got 48000 Hz
output_absent=yes
```

8 位输入：

```text
exit=2 stdout_bytes=0
ERROR ERR_BAD_FORMAT: samples/bad_bit_depth_8.wav: bit depth must be 16-bit, got 8-bit
output_absent=yes
```

损坏文件头：

```text
exit=2 stdout_bytes=0
ERROR ERR_BAD_FORMAT: cannot read WAV header/data from samples/bad_header.wav: fmt chunk and/or data chunk missing
output_absent=yes
```

PCM 数据截断：

```text
exit=2 stdout_bytes=0
ERROR ERR_BAD_FORMAT: samples/bad_truncated_data.wav: truncated PCM data (expected 882000 bytes, got 881900)
output_absent=yes
```

自动化测试还验证了坏格式不会覆盖已存在的输出文件。

## 5. 自动化测试

```bash
python -m unittest discover -v
```

实测结果：

```text
Ran 9 tests in 7.178s
OK
```

覆盖项：45/60/90/120/128/130/174/220/240 BPM 检测、低置信度、端到端封装、mixing_plan 字段、
50 ms 粒子起止、种子复现、软限幅、立体声输入、打击乐八度归一、坏格式退出码和原输出保护。
另有一项故障注入测试验证：临时 WAV 写入失败时正式输出保持原字节且临时文件清理。

## 6. BPM 鲁棒性抽查

另用 120 BPM 的 10 秒确定性波形抽查：单样本脉冲、白噪声爆发、低频扫描正弦、
持续音垫叠加及宽立体声节拍。检测结果均为 `120.0 BPM`，置信度分别为
`0.977`、`0.983`、`0.970`、`0.971`、`0.936`，全部高于 `0.7` 阈值。

