# 实际验证记录

验证日期：2026-11-15

## 环境

- 执行环境：Ubuntu 26.04 / WSL（代码本身不依赖平台）
- Python：3.14.4
- NumPy：2.3.5
- soundfile：0.13.1，libsndfile 1.2.2
- FFmpeg：7.0.2-static

当前容器无法直接通过 `pip` 访问 Python 包索引，因此验证时把发行版提供的 NumPy、soundfile、cffi 及 BLAS 运行库解包到本地 `.vendor/`，再通过 `PYTHONPATH` / `LD_LIBRARY_PATH` 运行测试。`.vendor/` 不参与交付，正常环境使用 `pip install -r requirements.txt` 即可。

## 自动化测试

执行：

```bash
python -m unittest discover -v
```

结果：`Ran 17 tests in 0.039s`，`OK`。

实际覆盖并验证：

- `0.0005 s @ 1000 Hz` 换算为 1 个采样点，确认半增舍入；采样率为 4/8 Hz 的线性插值结果为 `0, 0.5, 1, 1.5, 2, 2.5, 3, 3`。
- 片段从第 2 个采样点开始，占用采样点 `[2, 5)`，输出值精确落在 `2..4`。
- 单声道复制为双声道；双声道输入到单声道工程时只取第一声道。
- 淡入淡出总长超过片段长度时被等比缩短，包络最小值为 0、最大值不超过 1。
- 两段 `0.9` 满刻度素材完全重叠时，经过阈值 `0.95` 的 tanh 软限幅，最终绝对峰值不超过 `0.95`。
- 单个非重叠片段增益后峰值为 `1.8`，整体归一化增益 `-5.205449 dB`，报告峰值 `-0.1 dBFS`。
- 500 Hz 素材线性重采样到 1000 Hz，输出采样值为 `0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6`，并产生 `RESAMPLED` 警告。
- 时间轴尾部截断、负起点、零时长、素材缺失和损坏 WAV 均记录对应错误码；损坏/缺失片段不阻止成功退出。
- WAV PCM16/PCM24 输出经 soundfile 分别回读为 `PCM_16`/`PCM_24`；FLAC 输出格式与采样率正确。报告中的采样数、三位小数时长和峰值与解码后的实际文件一致。
- 假 FFmpeg 返回码 7 时，CLI 退出码为 6，不留下输出 WAV；`mix.report.json` 保存 `FFMPEG_ENCODE_FAILED`、返回码和 stderr。
- 工程缺失、JSON 损坏、FFmpeg 缺失分别以退出码 2、3、5 结束并向 stderr 输出简短错误。
- 工程中同时存在正常、缺失、负起点、损坏素材时，CLI 退出码仍为 0，片段级状态/错误码均写入报告。

## 编码文件检查

端到端测试实际调用了 FFmpeg，并分别生成：

- `mix.wav`：PCM 24 位，工程采样率 1000 Hz，报告时长与解码采样数一致。
- `mix.flac`：FLAC 无损，源 2000 Hz 线性重采样到 4000 Hz。

## 独立 CLI 演示

另外使用独立临时目录生成 4000 Hz 单声道正弦和 8000 Hz 双声道素材，工程为 8000 Hz、1.4 秒、24 位 WAV。工程中包含重采样片段、重叠片段、缺失素材和越界片段。实际运行：

```bash
python mix_audio.py project.json out --ffmpeg /home/liang/tools/ffmpeg
```

CLI 退出码为 `0`。回读结果：

- 输出 `demo_mix.wav` 为 `PCM_24`、8000 Hz、双声道、11200 个采样点；报告 `duration_sec` 为 `1.400`。
- 回读峰值为 `-5.293 dBFS`，归一化增益为 `0.0 dB`。
- 片段状态依次为：`warning`（`RESAMPLED`）、`success`、`skipped`（`SOURCE_NOT_FOUND`）、`truncated`（`TIMELINE_TRUNCATED`、`RESAMPLED`、`FADE_SCALED`）。
- 输出音频文件名与 `demo_mix.report.json` 的报告主干一致，FFmpeg 返回码为 `0`。

