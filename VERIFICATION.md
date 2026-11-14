# 实际验证记录

验证日期：2026-11-15（Asia/Shanghai）

环境：

- Python 3.14.4
- NumPy 2.5.4
- SoundFile 0.14.0
- libsndfile 1.x
- Ubuntu 26.04 本地环境
- 未使用 Docker 或其他容器

## 自动化测试

命令：

```bash
.venv/bin/python -m pytest -q
```

结果：

```text
.................                                                        [100%]
23 passed in 2.92s
```

## 样本矩阵

先用以下命令生成本地确定性素材：

```bash
.venv/bin/python scripts/generate_test_samples.py \
  --output-dir testdata --duration 5.2
```

随后对受支持样本执行：

```bash
.venv/bin/python -m loudness_pipeline \
  testdata/<name>.wav /tmp/<name>_norm.wav --target -23
```

| 输入素材 | 采样率 | 位深/子类型 | 声道 | 处理前 LUFS | 处理后 LUFS | 增益 dB |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `stereo_32k_pcm32.wav` | 32000 | PCM_32 | 2 | -24.32 | -23.00 | +1.32 |
| `stereo_44k1_pcm16.wav` | 44100 | PCM_16 | 2 | -24.34 | -23.00 | +1.34 |
| `stereo_48k_pcm24.wav` | 48000 | PCM_24 | 2 | -24.34 | -23.00 | +1.34 |
| `stereo_96k_float.wav` | 96000 | FLOAT | 2 | -24.36 | -23.00 | +1.36 |
| `surround_44k1_double.wav` | 44100 | DOUBLE | 6 | -19.81 | -23.00 | -3.19 |
| `surround_48k_pcm24.wav` | 48000 | PCM_24 | 6 | -19.82 | -23.00 | -3.18 |

输出 WAV 的采样率、子类型、声道数和帧数与输入一致。5.1 报告的权重为
`[1.00, 1.00, 1.00, 0.00, 1.00, 1.00]`。

## 限峰验证

输入 `stereo_48k_limiter_float.wav`，目标 -23 LUFS：

```text
input_lufs: -31.45 LUFS
output_lufs: -23.01 LUFS
gain_db: 8.45 dB
applied_base_gain_db: 10.88 dB
limiter_applied: true
max_limiting_db: 3.18 dB
```

报告中的处理字段：

```text
limited_sample_count: 237
maximum_overshoot: 0.00
limiter_iterations: 4
```

回读输出文件后，最大绝对样本值为 `1.0`，超过 0 dBFS 的样本数为 `0`。

## 错误路径

| 场景 | stderr 摘要 | 退出码 |
| --- | --- | ---: |
| 损坏 WAV | `E_FORMAT: ... Format not recognised` | 3 |
| 零帧 WAV | `E_EMPTY: input WAV contains zero audio frames` | 4 |
| 22050 Hz WAV | `E_SAMPLERATE: unsupported sample rate 22050 Hz` | 5 |
| 目标 -40 LUFS | argparse `usage:` | 2 |
| 输出路径为目录 | `E_WRITE: failed to write output WAV ... Is a directory` | 6 |

写入失败后输入文件 SHA-256 未变化，输出目录中未残留 `.*.tmp` 临时文件。

## 确定性与报告

对 `surround_48k_pcm24.wav`、目标 -23 LUFS，连续运行两次：

```bash
mkdir -p /tmp/loudness_verify_final
python -m loudness_pipeline testdata/surround_48k_pcm24.wav \
  /tmp/loudness_verify_final/repeat.wav --target -23
```

```text
input_unchanged True
wav_sha256_1    07ada57331d9093e95fe8d87138cb26c739b366992e7237b81ccc7d9fc3ffce5
report_sha256_1 f5ca7636e6e5885db3b3b70a7c78ccbf5196ab35c7fe7f88a969dc60d2160aa2
repeat_exact    True
```

报告中的分段为：

```text
(0.00, 3.00, 3.00, before=-19.30, after=-22.48, partial_segment=false)
(3.00, 5.20, 2.20, before=-20.64, after=-23.82, partial_segment=true)
```

整体结果：

```text
before integrated_lufs = -19.82
after  integrated_lufs = -23.00
delta_lufs             =  -3.18
```

JSON 使用自定义序列化器，所有浮点字段在文本中固定两位小数，可被标准 JSON
解析器正常读取。

