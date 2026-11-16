# 验证记录

验证日期：2026-11-17（Asia/Shanghai）

运行环境：

- WSL2 Ubuntu 26.04，Linux 6.18
- Python 3.14.4
- NumPy 2.5.4
- SoundFile 0.14.0 / libsndfile 1.2.2
- setuptools 84.2.1（仅用于打包检查）

所有输入 WAV 由 `spatial_audio_renderer.demo` 或测试辅助代码用固定随机种子
生成，未访问网络或外部素材。

## 自动化测试

命令：

```bash
python -m unittest discover -s tests -v
```

结果：

```text
Ran 30 tests in 6.421s

OK
```

## 打包入口

命令：

```bash
python -m pip wheel . --no-deps --no-build-isolation -w /tmp/wheel
```

结果：

```text
Successfully built spatial-audio-renderer
spatial_audio_renderer-1.0.0-py3-none-any.whl
size=27142 bytes
sha256=bc4bca9cbd0a376a836c5d028af10f7cbe1d78e7930429497ce2163d89db9787
```

将 wheel 安装到独立 `--target` 目录后，运行
`python -m spatial_audio_renderer --version` 输出
`spatial-render 1.0.0`，演示模块入口也可正常加载。

## 内置 8 秒场景

命令：

```bash
python -m spatial_audio_renderer.demo /tmp/spatial_verify.mHMa4y \
  --output /tmp/spatial_verify.mHMa4y/mix.wav \
  --normalize-lufs -16 --compress
```

退出码：`0`

stderr 关键输出：

```text
progress: 12.53%
progress: 25.00%
progress: 37.53%
progress: 50.00%
progress: 62.53%
progress: 75.00%
progress: 87.53%
progress: 100.00%
finished: duration=8.000 s, peak=-3.60 dBFS, integrated=-16.00 LUFS
```

输出 WAV 检查：

```text
samplerate=48000
channels=2
frames=384000
duration=8.0
subtype=PCM_16
peak=0.660888671875
```

## 关键 DSP 指标

使用固定 48 kHz 测试信号得到：

| 项目 | 实测结果 |
| --- | --- |
| 右侧声源双耳 RMS（左/右） | `0.077941 / 0.141513` |
| 右/左耳 RMS 比 | `1.815632` |
| 最大 ITD | `0.510204 ms`（约 24.49 采样） |
| 10 m/s 接近声源多普勒速率 | `1.03003003` |
| 10 m/s 远离声源多普勒速率 | `0.97167139` |
| 闭式期望值 | `1.03003003 / 0.97167139` |
| 1 kHz 素材、1.1 倍播放速率 | FFT 峰值 `1100.400 Hz` |
| 双层遮挡截止频率 | `1333.333 Hz` |
| 双层遮挡线性增益 | `0.420000` |
| 整段/逐 256 帧遮挡滤波最大差 | `0` |
| 遮挡切换处最大相邻采样跳变 | `0.039151` |
| 压缩器在 -20 dBFS 阈值处的软拐点减益 | `0.562500 dB` |
| 压缩输出长度与首采样 | 长度不变，首采样立即输出 |

## 错误语义

| 场景 | 实测退出码 | 文件/清理结果 |
| --- | --- | --- |
| 缺失 `missing_narration` 与损坏 `damaged_radio` 轨道 | `0` | 两条命名警告，输出 WAV 正常生成 |
| 声源关键帧时间戳重复 | `1` | 输出文件未创建，错误含 `strictly increasing` |
| 输出父目录不存在 | `2` | 无输出文件，无 `.spatial-render-*` 临时文件 |
| 模拟最终写入失败 | `2` | 无输出文件，临时文件全部清理 |
| 模拟磁盘空间不足 | `2` | 不创建输出文件 |

## Windows 说明

验证环境没有真实 Windows 主机或 Wine，因此 Windows `.cmd` 的实际进程执行
未在此环境运行。代码路径使用 `pathlib.Path`、`os.replace`、Windows 兼容的
`.cmd` 启动脚本和 SoundFile 的 WAV 输出；相关相对路径、反斜杠转换和命令行
退出码在 Linux/WSL 下通过自动化测试。
