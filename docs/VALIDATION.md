# 实际验证记录

验证日期：2026-11-16。

## 环境

- 当前执行环境：WSL2 Linux（Windows 本地宿主），Ubuntu 26.04，24 vCPU，7.6 GiB RAM
- Python 3.14.4（代码要求 Python 3.11+）
- NumPy 2.5.4
- soundfile 0.14.0 / libsndfile 1.2.2
- FFmpeg 8.0.1。核心仅调用 FFmpeg 6.x 已有的 `ebur128=peak=true` 与
  `alimiter`，未使用 FFmpeg 8 专有接口。

Windows 路径、原子替换、进程锁、FFmpeg 可执行文件发现均使用跨平台实现；
Windows 的存活进程判断使用 `OpenProcess/GetExitCodeProcess`，避免
`os.kill(pid, 0)` 在 Windows 上的终止语义。

## 自动化测试

命令：

```bash
PYTHONPATH=src python -m unittest discover -v -s tests -t .
```

结果：10 项测试全部通过，耗时约 7.6 秒。

覆盖内容：

1. 44.1 kHz 脉冲重采样到 48 kHz 后，峰值位置与理论位置误差不超过 1 采样。
2. 高优先级对白使低优先级对白在完整重叠区间降低 12 dB；环境音同样时间窗口
   的幅度比保持 1.0，未被避让。
3. `gain_segments` 从 `first_valid_sample` 连续覆盖到
   `last_valid_sample_exclusive`。
4. 并列最高优先级对白重叠返回 `E_DUCK_UNSATISFIABLE`，无母版输出。
5. 最终响度在目标 ±5 LU 内，最终真峰值不高于上限；高波峰素材触发前瞻限幅。
6. 清单版本变化后复用轨道缓存，但与干净状态全量运行的结果 `array_equal`，
   WAV 文件逐字节一致。
7. `CUTMASTER_FAULT_INJECT=after_track` 强杀后，检查点为 `in_progress`、
   最终母版不存在；再次启动识别恢复状态、调用缓存并输出完整母版。
8. `after_mix` 强杀后无最终 WAV；重跑成功。
9. `E_MISSING_TRACK`、`E_BAD_RATE`、`E_START_OUT_OF_RANGE`、
   `E_METADATA_MISMATCH` 均返回预期 JSON 错误码和非零退出码，无部分输出。

## 性能验收

样本由 `scripts/generate_samples.py --preset performance` 确定性生成：

- 8 条 48 kHz、立体声、300 秒音轨
- 3 条对白存在重叠
- 工程 365 秒
- 源 WAV 合计 691,200,352 字节（约 660 MiB）

运行：

```bash
cd scripts
python benchmark.py /tmp/cutmaster-perf.<unique>
```

实测：

- 全量处理墙钟时间：**11.71 秒 < 120 秒**
- 子进程峰值 RSS：**580.8 MiB < 1.5 GiB**
- 输出：48 kHz 立体声 PCM24，105,120,044 字节
- 最终响度：**-16.0 LUFS**（目标 -16.0 LUFS）
- 最终真峰值：**-4.6 dBTP**（上限 -1.0 dBTP）
- 对白重叠区段：3 个连续区段；避让区间：4 个
- 起始位置最大取整误差：0 采样
- 母版 SHA-256：
  `1a0ecc0e1222f1f28a709077e5d229b96f8a564f936bdf335d93f33ff313a68f`

性能路径没有截短音频、降采样、跳过低响度片段或降低 true-peak 测量精度。
