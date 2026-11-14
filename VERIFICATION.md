# 验证结果记录

验证环境（本地，无容器、无外部素材）：

- OS：Ubuntu 26.04.1 LTS（x86_64）；代码平台无关，可在 64 位 Windows 运行
- Python：3.14.4（要求 ≥ 3.9）
- NumPy：2.5.4（要求 ≥ 1.23），安装于项目 `.venv`
- FFmpeg：7.0.2 静态构建（要求 ≥ 5.0），位于 `~/.local/ffmpeg-bin`
- 样本：`generate_samples.py` 使用固定随机种子 `20240517` 确定性生成

## 1. 测试套件

命令：`python -m unittest discover -s tests`

结果：`Ran 23 tests ... OK`（15 个单元测试 + 8 个端到端验收测试，全部通过）。

| 测试文件 | 覆盖内容 |
| --- | --- |
| `tests/test_loudness.py` | 增益-响度线性关系、三种声道布局、非法布局拒绝、静音阈值/时长、过采样真峰值、限制器天花板与响度保持 |
| `tests/test_wav_srt.py` | 16/24-bit 读写往返、24-bit 输出、非法采样率/声道/损坏 WAV、SRT 解析与时间轴平移 |
| `tests/test_pipeline.py` | 退出码、总状态、输出排序、48k/24bit 输出、响度达标、真峰值、限制器、字幕调整、错误分类、重采样元数据、致命错误 |

## 2. BS.1770-4 测量准确性（与 FFmpeg loudnorm 参考对比）

| 音轨 | 自研积分响度 (LUFS) | FFmpeg (LUFS) | 自研真峰值 (dBTP) | FFmpeg (dBTP) |
| --- | ---: | ---: | ---: | ---: |
| dialogue_stereo（立体声 48k/24bit） | -10.74 | -10.68 | -1.36 | -1.36 |
| dialogue_mono_quiet（单声道 44.1k/16bit） | -27.16 | -27.16 | -14.83 | -14.80 |
| dialogue_surround（5.1 48k/24bit） | -14.21 | -13.85 | -5.19 | -5.19 |
| dialogue_peaking（立体声 44.1k/24bit） | -9.32 | -9.30 | 0.24 | 0.38 |

积分响度最大偏差 0.36 LU，真峰值读数一致（瞬态信号的过采样峰值估计差异
≤0.14 dB），满足工程精度。

## 3. 响度标准化与真峰值限制（目标 -23 LUFS / -1.0 dBTP）

| 输出音轨 | 最终响度 (LUFS) | 响度误差 (LU) | 最终真峰值 (dBTP) | 限制器 | 字幕调整 |
| --- | ---: | ---: | ---: | --- | ---: |
| dialogue_dense_loud | -23.31 | 0.31 | -1.27 | 触发（124 次，最大衰减 6.35 dB） | 1 |
| dialogue_mono_quiet | -23.00 | 0.00 | -10.68 | 未触发 | 1 |
| dialogue_peaking | -23.00 | 0.00 | -13.44 | 未触发 | 1 |
| dialogue_stereo | -23.00 | 0.00 | -13.62 | 未触发 | 2 |
| dialogue_surround | -23.00 | 0.00 | -13.99 | 未触发 | 1 |

- 所有成功音轨响度误差 ≤ 0.5 LU，真峰值均 ≤ -1.0 dBTP。
- `dialogue_dense_loud` 归一化前真峰值 +5.05 dBTP，限制器将其压到
  -1.27 dBTP，同时响度保持在 -23.31 LUFS（仍在 ±0.5 LU 内）。
- 全部输出经 `ffprobe` 验证为 48000 Hz、24-bit、声道数与输入一致。

## 4. 字幕时间轴校正

以 `dialogue_stereo` 为例（静音段 0.0–1.0 s、3.55–4.5 s 等）：

- 条目 1：`00:00:00,400–00:00:01,400` → `00:00:01,000–00:00:02,000`
  （起点后移到首个语音帧，时长 1.0 s 不变）
- 条目 2：`00:00:04,200–00:00:05,100` → `00:00:04,500–00:00:05,400`
  （时长 0.9 s 不变）
- 条目 3 起点在语音段内，保持不变。

报告 `subtitles.adjustments` 完整记录每条调整的 `original_start/end`、
`new_start/end` 与所在静音段边界。

## 5. 异常处理与退出码

| 场景 | 行为 | 退出码 |
| --- | --- | ---: |
| 9 条中 4 条异常（部分成功） | 跳过异常文件，其余正常输出，状态 `partial` | 1 |
| 全部有效 | 状态 `success` | 0 |
| 输入目录不存在 | 立即终止，状态 `failed` | 2 |
| 输出目录不可写（只读目录） | 立即终止 | 2 |
| FFmpeg 不在 PATH | 立即终止并提示安装 | 2 |

被跳过的异常样本与报告信息：

| 文件 | 报告错误 |
| --- | --- |
| dialogue_corrupt.wav | WAV 损坏，缺少 fmt/data 块 |
| dialogue_quad.wav | 不支持的声道数：4 |
| dialogue_32k.wav | 不支持的采样率：32000 Hz |
| dialogue_bad_srt.wav | SRT 时间轴行格式错误（音频仍完成标准化） |

失败任务的独立临时目录被保留，路径记录在报告 `retained_temp_dirs`；
成功任务临时目录自动清理。

## 6. 并行与顺序

分别以 `-j 1`、`-j 2`、`-j 4` 运行，报告内音轨顺序均与输入文件名排序
完全一致；默认并行度为 2。9 条音轨完整处理耗时约 16 秒（并行度 2）。

## 7. 复现实验

```bash
export PATH="$HOME/.local/ffmpeg-bin:$PATH"
source .venv/bin/activate
python generate_samples.py --output samples/input
python -m audio_norm.cli samples/input samples/output -j 2   # 退出码 1 = partial
python -m unittest discover -s tests                          # OK，23 tests
```
